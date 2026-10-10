"""Versioned observations, physical carry and independent rejection semantics."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from f1ers_v3.geometry import ClosedRoute
from f1ers_v4.vehicle import IntegrationState
from f1ers_v6.frozen_demands import FrozenDemandDriver
from f1ers_v6.learning import _features, _teacher_arrays
from f1ers_v6.metrics import score
from f1ers_v6.rules import RuleConfig, RuleState
from f1ers_v7.rules import EnergyViableRuleShield
from f1ers_v7.simulator import FEATURES, ReachableSimulator, freeze_simulator


def fixture(radius=20., notice=.25):
    theta = np.linspace(0., 2*np.pi, 40, endpoint=False)
    route = ClosedRoute(np.column_stack((radius*np.cos(theta), radius*np.sin(theta))), kind='spline')
    rules = {'soc_window_mj': 4., 'ers_max_power_kw': 350., 'harvest_max_mj_per_lap': 8.,
        'alternative_power_zones_route_m': [], 'power_reduction_exceptions_route': [],
        'fia_centreline_length_m': route.source_length,
        'fia_sector_lengths_m': [route.source_length/3]*3}
    return ReachableSimulator(route.sample(5.), rules, rule_config=RuleConfig(tail_guard_m=0.),
        margin_s=1., energy_notice_s=notice)


class SimulatorTests(unittest.TestCase):
    def test_environment_and_inherited_learning_agree_on_new_feature_schema(self):
        sim = fixture()
        env = sim.create_env()
        self.assertEqual(len(FEATURES), 33)
        self.assertEqual(env.features, sim.features)
        self.assertEqual(env.observation_dim, len(FEATURES))
        self.assertEqual(env.get_obs().shape, (len(FEATURES),))
        self.assertEqual(_features(sim, env), FEATURES)
        env.reset()
        self.assertEqual(env.observation_dim, len(FEATURES))
        self.assertEqual(env.get_obs().shape, (len(FEATURES),))

    def test_reference_and_target_remove_synthetic_rule_state_alias(self):
        sim = fixture(radius=500.)
        env = sim.create_env(initial_speed_mps=280./3.6, initial_energy_mj=4.)
        env.state = IntegrationState(100., 280./3.6, .25, 4., 0., 200.)
        rows = []
        for reference, target in ((250., 100.), (300., 150.)):
            env.shield.state = replace(env.shield.state, phase='HOLD', elapsed_time_s=.25,
                hold_until_s=1., reduction_mechanical_kw=20.,
                target_reduction_mechanical_kw=target, reference_mechanical_kw=reference,
                previous_dc_kw=200., requested_dc_kw=150., braking=False)
            rows.append(env.get_obs())
        # These allowed synthetic states are not asserted to be reachable laps.
        np.testing.assert_array_equal(rows[0][:31], rows[1][:31])
        self.assertNotEqual(rows[0][31], rows[1][31])
        self.assertNotEqual(rows[0][32], rows[1][32])
        np.testing.assert_allclose(rows[0][-2:], [250./332.5, 100./700.])

    def test_full_rollout_serializes_33_columns_for_inherited_bc_inputs(self):
        sim = fixture()
        result = sim.simulate(action_space='mode_aware')
        self.assertTrue(result['summary']['completed_lap'])
        self.assertTrue(result['summary']['execution_feasible'])
        self.assertEqual(result['summary']['observation_dim'], 33)
        self.assertEqual(result['summary']['protocol'], sim.protocol)
        self.assertEqual(result['observations'].shape[1], 33)
        teachers, observations, actions = _teacher_arrays([result], FEATURES)
        self.assertEqual(len(teachers), 1)
        self.assertEqual(observations[0].shape, (len(actions[0]), 33))

    def test_frozen_demands_preserve_viable_shield_and_immutable_raw_profile(self):
        sim = fixture(notice=.6)
        starts = sim.driver_template.decision_s[:-1]
        profile = {'s_m': starts.copy(), 'wheel_force_n': np.full(len(starts), 7.)}
        fixed = freeze_simulator(sim, profile)
        profile['wheel_force_n'][0] = 999.
        env = fixed.create_env(initial_energy_mj=3.)
        self.assertIsInstance(env.shield, EnergyViableRuleShield)
        self.assertIsInstance(env.driver, FrozenDemandDriver)
        self.assertIs(env.driver.shield, env.shield)
        self.assertEqual(env.shield.energy_notice_s, .6)
        self.assertEqual(env.shield.margin_s, 1.)
        self.assertEqual(env.driver.demand(env.state), 7.)
        self.assertFalse(env.driver.frozen_force.flags.writeable)
        self.assertIs(fixed.geometry_guard_map, sim.geometry_guard_map)
        env.shield.state = replace(env.shield.state, phase='HOLD', hold_until_s=.5)
        self.assertEqual(env.driver.next_time_event(.1), .5)
        env.reset()
        self.assertIsInstance(env.shield, EnergyViableRuleShield)
        self.assertIs(env.driver.shield, env.shield)
        self.assertEqual(env.driver.demand(env.state), 7.)
        self.assertEqual(_features(fixed, env), FEATURES)

    def test_carried_nonzero_dc_reference_and_pending_hold_are_not_reset(self):
        sim = fixture(radius=500.)
        carry = asdict(RuleState(phase='HOLD', elapsed_time_s=10., hold_until_s=10.7,
            previous_dc_kw=100., requested_dc_kw=50., reference_mechanical_kw=95.,
            target_reduction_mechanical_kw=95., reduction_mechanical_kw=0.))
        env = sim.create_env(initial_speed_mps=60., initial_energy_mj=3., carried_rule_state=carry)
        self.assertEqual(env.state.previous_dc_kw, 100.)
        self.assertEqual(env.shield.state.reference_mechanical_kw, 95.)
        self.assertEqual(env.shield.state.target_reduction_mechanical_kw, 95.)
        self.assertEqual(env.shield.state.phase, 'HOLD')
        self.assertAlmostEqual(env.shield.state.hold_until_s, .7)
        self.assertEqual(env.state.energy_mj, 3.)
        self.assertEqual(env.get_obs()[-2], 95./332.5)

    def test_consecutive_laps_carry_actual_speed_energy_and_rule_state(self):
        sim = fixture()
        result = sim.simulate_laps(n_laps=2, initial_energy_mj=3.)
        self.assertTrue(result['physical_control_line_continuity'])
        self.assertEqual(result['within_lap_velocity_resets'], 0)
        self.assertEqual(result['within_lap_energy_resets'], 0)
        self.assertTrue(result['lap_recharge_counter_reset_only'])
        self.assertEqual(result['crossings'][0]['speed_difference_mps'], 0.)
        self.assertEqual(result['crossings'][0]['energy_difference_mj'], 0.)
        self.assertEqual(result['crossings'][0]['dc_difference_kw'], 0.)
        self.assertTrue(result['crossings'][0]['rule_state_carried'])
        self.assertEqual(result['laps'][1]['summary']['initial_rule_state'],
            result['laps'][0]['rule_certificate']['terminal_boundary_state'])
        with self.assertRaises(ValueError):
            sim.simulate_laps(n_laps=2, periodic_shooting=True)

    def test_engineering_rejection_preserves_accepted_physics_and_partial_time(self):
        sim = fixture()
        env = sim.create_env(initial_energy_mj=3.)
        initial_speed = env.state.speed_mps
        env.shield.engineering_violations.append({'reasons': ['synthetic_engineering_failure']})
        _, reward, done, info = env.step(0.)
        self.assertTrue(done)
        self.assertEqual(info['status'], 'physical_fail')
        self.assertEqual(info['rejection_classification'], 'added_engineering_boundary_rejection')
        self.assertEqual(env.failure['frozen_vehicle_kernel_status'], 'accepted')
        self.assertEqual(info['reward_terms']['reward_failure_s'], -100.)
        self.assertAlmostEqual(reward, -info['dt_s']-100.)
        result = env.result()
        self.assertFalse(result['summary']['completed_lap'])
        self.assertFalse(result['summary']['feasible'])
        self.assertFalse(result['summary']['reachable_request_gate_passed'])
        self.assertIsNone(result['summary']['lap_time_s'])
        self.assertIsNone(result['summary']['formal_lap_time_s'])
        self.assertGreater(result['summary']['elapsed_time_s'], 0.)
        self.assertGreater(result['terminal_s_m'], 0.)
        self.assertLess(result['terminal_s_m'], sim.length_m)
        self.assertEqual(result['internal_trace']['speed_mps'][0], initial_speed)
        self.assertEqual(result['observations'].shape[1], 33)
        self.assertFalse(result['rule_certificate']['full_fia_verified'])
        dataset = {'laps': [{'lap_number': 3, 'role': 'calibration'}],
            'audit': {'data_gate_passed': True}}
        accuracy = score(result, dataset)
        self.assertFalse(accuracy['metrics'][0]['valid'])
        self.assertNotIn('speed_rmse_kmh', accuracy['metrics'][0])
        with self.assertRaises(RuntimeError):
            env.step(0.)

    def test_simulation_does_not_access_fp2_acquisition_or_unsealing(self):
        sim = fixture()
        with patch('f1ers_v4.data.prepare_new_session', side_effect=AssertionError('FP2 acquisition')) as acquire, \
             patch('f1ers_v4.data.load_new_session', side_effect=AssertionError('FP2 arrays')) as load, \
             patch('f1ers_v4.protocol.authorize_test_access', side_effect=AssertionError('FP2 unseal')) as unseal:
            result = sim.simulate()
        self.assertTrue(result['summary']['completed_lap'])
        acquire.assert_not_called()
        load.assert_not_called()
        unseal.assert_not_called()


class ExperimentEntryTests(unittest.TestCase):
    def test_existing_experiment_refuses_before_provenance_or_new_data(self):
        from f1ers_v7.experiment import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'results/existing').mkdir(parents=True)
            with patch('f1ers_v7.experiment.provenance') as source, \
                 patch('f1ers_v7.experiment.load_local_dataset') as data:
                with self.assertRaises(RuntimeError):
                    run(root, 'existing')
            source.assert_not_called()
            data.assert_not_called()

    def test_bad_historical_data_gate_stops_before_creating_experiment(self):
        from f1ers_v7.experiment import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('f1ers_v7.experiment.provenance', return_value={}), \
                 patch('f1ers_v7.experiment.read', return_value={}), \
                 patch('f1ers_v7.experiment.load_local_dataset', return_value={'audit': {'data_gate_passed': False}}), \
                 patch('f1ers_v4.data.prepare_new_session', side_effect=AssertionError('FP2 acquisition')) as acquire:
                with self.assertRaisesRegex(ValueError, 'historical data gate failed'):
                    run(root, 'new_run')
            self.assertFalse((root/'results/new_run').exists())
            acquire.assert_not_called()

    def test_dirty_worktree_refuses_before_hashing_or_dependency_capture(self):
        from f1ers_v7.experiment import provenance
        with patch('f1ers_v7.experiment.git', return_value=' M f1ers_v7/simulator.py'), \
             patch('f1ers_v7.experiment.digest') as hashed, \
             patch('f1ers_v7.experiment.importlib.metadata.version') as versions:
            with self.assertRaisesRegex(RuntimeError, 'clean worktree'):
                provenance(Path('.'))
        hashed.assert_not_called()
        versions.assert_not_called()


if __name__ == '__main__':
    unittest.main()
