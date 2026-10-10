"""Regression draft: run only after the deployment-only source is authorized."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from f1ers_v7 import deploy_only as module

from f1ers_v7._core.networks import GaussianPolicy
from f1ers_v7._core.geometry import ClosedRoute
from f1ers_v7._core.integrator import IntegrationState, VehicleParamsV4, WheelCommand, vehicle_step
from f1ers_v7._core.frozen_demands import FrozenDemandDriver
from f1ers_v7._core.learning_base import save_policy as save_legacy_policy
from f1ers_v7._core.base_rules import RuleConfig
from f1ers_v7.rules import EnergyViableRuleShield
from f1ers_v7.learning import _bind, save_policy
from f1ers_v7.simulator import FEATURES as OLD_FEATURES


class ConstantGeometry:
    length_m = 4326.
    def evaluate(self, s):
        return dict(curvature_1pm=.003, drag_area_m2=0., downforce_area_m2=0.)


def shield_fixture(cls=module.DeployOnlyShield):
    geometry = ConstantGeometry()
    grid = {'s_m': np.arange(4326.), 'safe_speed_mps': np.full(4326, 90.),
        'ds_m': 1., 'positive_low_mask': np.zeros(4326, bool),
        'hazard_s_m': np.array([]), 'length_m': 4326.}
    shield = cls({'ers_max_power_kw': 350., 'harvest_max_mj_per_lap': 8.},
        geometry, RuleConfig(tail_guard_m=0.), grid, 1., energy_notice_s=.5)
    state = IntegrationState(100., 50., 0., 3., 0., 0.)
    shield.reset(state)
    return shield, state


def simulator_fixture(radius=20.):
    theta = np.linspace(0., 2*np.pi, 40, endpoint=False)
    route = ClosedRoute(np.column_stack((radius*np.cos(theta), radius*np.sin(theta))), kind='spline')
    rules = {'soc_window_mj': 4., 'ers_max_power_kw': 350., 'harvest_max_mj_per_lap': 8.,
        'alternative_power_zones_route_m': [], 'power_reduction_exceptions_route': [],
        'fia_centreline_length_m': route.source_length,
        'fia_sector_lengths_m': [route.source_length/3]*3}
    return module.DeployOnlySimulator(route.sample(5.), rules,
        rule_config=RuleConfig(tail_guard_m=1000.), margin_s=1., energy_notice_s=.5)


class DeploymentOnlyTests(unittest.TestCase):
    def test_only_genuine_negative_wheel_force_requests_full_regeneration(self):
        shield, state = shield_fixture()
        command = {'wheel_force_n': -12000., 'brake_fraction': 0.}
        original = deepcopy(command)
        decision = shield.decide(state, command, -1.)
        self.assertEqual(command, original)
        self.assertEqual(decision['raw_action'], -1.)
        self.assertEqual(shield.raw_action, -1.)
        self.assertEqual(decision['mapped_action'], 1.)
        self.assertEqual(shield.decisions[-1]['raw_action'], -1.)
        self.assertLess(decision['requested_dc_kw'], 0.)
        # Drag deceleration or a brake flag cannot invent negative wheel force.
        positive = {'wheel_force_n': 1000., 'brake_fraction': 1.,
                    'target_acceleration_mps2': -2.}
        shield.reset(state)
        decision = shield.decide(state, positive, -1.)
        self.assertEqual(decision['mapped_action'], -1.)
        self.assertEqual(decision['requested_dc_kw'], 0.)
        self.assertFalse(shield.state.braking)

    def test_positive_raw_mapping_and_all_prior_energy_fsm_queries_are_identical(self):
        for raw in (-1., -.4, 0., .5, 1.):
            new, state = shield_fixture()
            old, old_state = shield_fixture(EnergyViableRuleShield)
            command = {'wheel_force_n': 12000., 'brake_fraction': 0.}
            a = new.decide(state, command, raw, 'mode_aware')
            b = old.decide(old_state, command, raw, 'mode_aware')
            self.assertEqual(a['requested_dc_kw'], b['requested_dc_kw'])
            self.assertEqual(new.state, old.state)
            self.assertEqual(new.query(state, command, new.geometry, .02),
                             old.query(old_state, command, old.geometry, .02))
            self.assertEqual(a['mapped_action'], raw)

    def test_full_regeneration_still_shares_tire_budget_and_leaves_braking_unchanged(self):
        shield, state = shield_fixture()
        params = VehicleParamsV4(rolling_coefficient=0., drag_area_m2=0.,
                                 downforce_area_m2=0., mu=1.8)
        command = {'wheel_force_n': -20000., 'brake_fraction': 0.}
        shield.decide(state, command, -1.)
        query = shield.query(state, command, shield.geometry, .02)
        # Query contains engineering receipt fields in addition to WheelCommand.
        fields = WheelCommand.__dataclass_fields__
        wheel = WheelCommand(wheel_force_n=command['wheel_force_n'], mode='autonomous',
                             **{key: value for key, value in query.items() if key in fields})
        after, info = vehicle_step(state, wheel, shield.geometry, params, .02)
        friction_after, friction_info = vehicle_step(state,
            replace(wheel, requested_dc_kw=0.), shield.geometry, params, .02)
        self.assertEqual(info['status'], 'accepted')
        self.assertGreater(info['friction_brake_kw'], 0.)
        self.assertLessEqual(abs(info['wheel_force_n']), info['tire_brake_capacity_n']+8.)
        self.assertAlmostEqual(info['mguk_mechanical_kw']-info['friction_brake_kw'],
                               info['wheel_power_kw'], 8)
        self.assertAlmostEqual(info['wheel_force_n'], friction_info['wheel_force_n'], 8)
        self.assertAlmostEqual(after.speed_mps, friction_after.speed_mps, 8)
        self.assertGreater(after.energy_mj, state.energy_mj)
        self.assertLessEqual(abs(info['dc_kw']), 350.)

    def test_full_requested_regeneration_cannot_exceed_energy_or_lap_recharge_room(self):
        params = VehicleParamsV4(rolling_coefficient=0., drag_area_m2=0., downforce_area_m2=0.)
        for energy, recharge in ((4., 0.), (3., 8.)):
            shield, state = shield_fixture()
            state = replace(state, energy_mj=energy, recharge_mj=recharge)
            shield.reset(state)
            command = {'wheel_force_n': -10000., 'brake_fraction': 0.}
            shield.decide(state, command, -1.)
            query = shield.query(state, command, shield.geometry, .02)
            wheel = WheelCommand(wheel_force_n=command['wheel_force_n'], mode='autonomous',
                **{key: value for key, value in query.items() if key in WheelCommand.__dataclass_fields__})
            after, info = vehicle_step(state, wheel, shield.geometry, params, .02)
            self.assertEqual(info['status'], 'accepted')
            self.assertEqual(info['dc_kw'], 0.)
            self.assertEqual(after.energy_mj, energy)
            self.assertEqual(after.recharge_mj, recharge)

    def test_episode_initial_energy_removes_synthetic_terminal_reward_alias(self):
        sim = simulator_fixture(radius=500.)
        a = sim.create_env(initial_speed_mps=50., initial_energy_mj=2.)
        b = sim.create_env(initial_speed_mps=50., initial_energy_mj=3.)
        common = IntegrationState(100., 50., 1., 3.5, 0., 0.)
        for env in (a, b):
            env.state = common
            env.shield.reset(common)
        ao, bo = a.get_obs(), b.get_obs()
        np.testing.assert_array_equal(ao[:33], bo[:33])
        self.assertEqual(ao[33], .5)
        self.assertEqual(bo[33], .75)
        # At a common hypothetical 2.5MJ terminal state the nondeficit outcomes
        # differ; this is an observation-contract scenario, not a reachable lap.
        self.assertTrue(2.5 >= a.initial.energy_mj)
        self.assertFalse(2.5 >= b.initial.energy_mj)

    def test_old_checkpoint_feature_order_and_executor_cannot_be_reinterpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = GaussianPolicy(33, np.zeros(33), np.ones(33), (4,))
            _bind(old, OLD_FEATURES, 'mode_aware', 'v7_reachable_control')
            path = Path(tmp)/'old.npz'
            save_policy(old, path)
            with self.assertRaises(ValueError):
                module.load_deploy_policy(path)
            new = GaussianPolicy(34, np.zeros(34), np.ones(34), (4,))
            _bind(new, module.FEATURES, 'mode_aware', module.PROTOCOL)
            path = Path(tmp)/'new.npz'
            save_policy(new, path)
            restored = module.load_deploy_policy(path)
            self.assertEqual(restored.observation_dim, 34)
            np.testing.assert_array_equal(restored.mean, new.mean)
            np.testing.assert_array_equal(restored.scale, new.scale)
            for original, loaded in zip(new.network.parameters(), restored.network.parameters()):
                np.testing.assert_array_equal(original, loaded)
            # Read-only load cannot silently reorder inputs or change std-head
            # weights/normalization to adapt an incompatible actor.
            reordered = list(module.FEATURES)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            _bind(new, reordered, 'mode_aware', module.PROTOCOL)
            path = Path(tmp)/'reordered.npz'
            save_policy(new, path)
            with self.assertRaises(ValueError):
                module.load_deploy_policy(path)
            _bind(new, module.FEATURES, 'mode_aware', module.PROTOCOL)
            new.v6_execution_protocol = 'v7_reachable_control'
            with self.assertRaises(ValueError):
                module.require_policy_contract(new)
            path = Path(tmp)/'old_protocol.npz'
            save_policy(new, path)
            with self.assertRaises(ValueError):
                module.load_deploy_policy(path)
            path = Path(tmp)/'legacy6.npz'
            save_legacy_policy(old, path)
            with self.assertRaises(ValueError):
                module.load_deploy_policy(path)

    def test_frozen_demand_reset_retains_new_mapping_schema_and_hold_events(self):
        sim = simulator_fixture()
        starts = sim.driver_template.decision_s[:-1]
        profile = {'s_m': starts.copy(), 'wheel_force_n': np.full(len(starts), -100.)}
        fixed = module.freeze_simulator(sim, profile)
        profile['wheel_force_n'][0] = 999.
        env = fixed.create_env(initial_energy_mj=3.)
        self.assertIsInstance(env.shield, module.DeployOnlyShield)
        self.assertIsInstance(env.driver, FrozenDemandDriver)
        self.assertIs(env.driver.shield, env.shield)
        self.assertEqual(env.driver.demand(env.state), -100.)
        self.assertEqual(env.get_obs().shape, (34,))
        self.assertFalse(env.driver.frozen_force.flags.writeable)
        env.shield.state = replace(env.shield.state, phase='HOLD', hold_until_s=.5)
        self.assertEqual(env.driver.next_time_event(.1), .5)
        env.reset()
        self.assertIsInstance(env.shield, module.DeployOnlyShield)
        self.assertEqual(env.observation_dim, 34)
        self.assertEqual(env.driver.demand(env.state), -100.)

    def test_raw_braking_actions_replay_exactly_and_tamper_is_detected_without_future_data(self):
        sim = simulator_fixture()
        starts = sim.driver_template.decision_s[:-1]
        profile = {'s_m': starts.copy(), 'wheel_force_n': np.where(np.arange(len(starts))%2, -500., 500.)}
        fixed = module.freeze_simulator(sim, profile)
        with patch('f1ers_v7._core.data.prepare_new_session', side_effect=AssertionError('FP2')) as acquire, \
                patch('f1ers_v7._core.data.load_new_session', side_effect=AssertionError('FP2')) as load, \
                patch('f1ers_v7._core.protocol.authorize_test_access', side_effect=AssertionError('FP2')) as unseal:
            result = fixed.simulate(initial_speed_mps=10., initial_energy_mj=3.)
        self.assertTrue(result['summary']['completed_lap'])
        self.assertEqual(result['summary']['protocol'], module.PROTOCOL)
        self.assertEqual(result['observations'].shape[1], 34)
        np.testing.assert_array_equal(result['requested_actions'], -np.ones(len(starts)))
        negative = np.flatnonzero(profile['wheel_force_n'] < 0.)
        self.assertGreater(len(negative), 0)
        np.testing.assert_array_equal(result['mapped_actions'][negative], np.ones(len(negative)))
        audit = module.proxy_audit(fixed, result)
        self.assertTrue(audit['passed'])
        altered = deepcopy(result)
        altered['requested_actions'][negative[0]] = 0.
        self.assertFalse(module.proxy_audit(fixed, altered)['passed'])
        acquire.assert_not_called()
        load.assert_not_called()
        unseal.assert_not_called()


if __name__ == '__main__':
    unittest.main()
