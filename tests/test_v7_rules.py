import copy
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

from f1ers_v7._core.integrator import IntegrationState
from f1ers_v7._core.base_rules import RuleConfig, RuleState
from f1ers_v7.rules import EnergyViableRuleShield


def fixture(notice=.25):
    geometry = SimpleNamespace(length_m=4326.)
    route = {'s_m': np.arange(4326.), 'safe_speed_mps': np.full(4326, 90.),
             'ds_m': 1., 'positive_low_mask': np.zeros(4326, bool),
             'hazard_s_m': np.array([]), 'length_m': 4326.}
    shield = EnergyViableRuleShield(
        {'ers_max_power_kw': 350., 'harvest_max_mj_per_lap': 8.},
        geometry, RuleConfig(tail_guard_m=0.), route, 1., notice)
    state = IntegrationState(100., 280./3.6, 0., 3.5, 0., 0.)
    shield.reset(state)
    return shield, state, {'wheel_force_n': 12000., 'brake_fraction': 0.}


def record(before, dt, dc):
    return {'status': 'accepted', 's_start_m': before.s_m,
            's_end_m': before.s_m+before.speed_mps*dt,
            'speed_start_mps': before.speed_mps, 'speed_next_mps': before.speed_mps,
            'time_start_s': before.elapsed_time_s, 'time_end_s': before.elapsed_time_s+dt,
            'dt_s': dt, 'dc_kw': dc, 'wheel_force_n': 12000.,
            'energy_next_mj': before.energy_mj-dc*dt/1000.,
            'recharge_next_mj': before.recharge_mj}


class ReachableRulesTests(unittest.TestCase):
    def test_notice_is_explicit_positive_finite_engineering_setting(self):
        for notice in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                fixture(notice)

    def test_floor_does_not_generate_unreachable_positive_reference(self):
        shield, state, command = fixture()
        state = replace(state, energy_mj=2.)
        shield.reset(state)
        shield.decide(state, command, 1.)
        q = shield.query(state, command, shield.geometry, .02)
        self.assertEqual(q['requested_dc_kw'], 0.)
        self.assertEqual(shield.state.reference_mechanical_kw, 0.)
        self.assertTrue(q['constraint_intersection_feasible'])

    def test_new_request_reserves_notice_hold_and_full_ramp(self):
        shield, state, command = fixture()
        state = replace(state, energy_mj=2.5)
        shield.reset(state)
        decision = shield.decide(state, command, 1.)
        p = decision['requested_dc_kw']
        self.assertLess(p, 350.)
        self.assertAlmostEqual(p*(1.+.25)/1000.+p*p/(2000.*100.), .5)
        self.assertTrue(shield.query(state, command, shield.geometry, .02)
                        ['constraint_intersection_feasible'])

    def test_energy_bound_decline_starts_full_causal_reduction(self):
        shield, state, command = fixture()
        # Plenty of the inherited one-second hold+ramp reserve remains; the
        # additional notice is now exhausted, so start the reduction early.
        state = replace(state, energy_mj=2.87, previous_dc_kw=320.)
        shield.reset(state)
        shield.state = replace(shield.state, reference_mechanical_kw=304.)
        decision = shield.decide(state, command, 1.)
        self.assertTrue(decision['energy_viability_audit']['energy_triggered_reduction_to_zero'])
        self.assertEqual(shield.state.phase, 'HOLD')
        self.assertEqual(shield.state.hold_until_s, 1.)
        self.assertEqual(shield.state.reduction_mechanical_kw, 0.)
        self.assertEqual(shield.state.target_reduction_mechanical_kw, 304.)
        q = shield.query(state, command, shield.geometry, .02)
        self.assertAlmostEqual(q['requested_dc_kw'], 320.)
        self.assertTrue(q['constraint_intersection_feasible'])

    def test_reopening_actor_demand_does_not_cancel_reduction_target(self):
        shield, state, command = fixture()
        state = replace(state, previous_dc_kw=300.)
        carry = RuleState(phase='LIMITED', reference_mechanical_kw=285.,
                          target_reduction_mechanical_kw=285., reduction_mechanical_kw=95.,
                          previous_dc_kw=300.)
        shield.reset(state, carry)
        shield.decide(state, command, 1.)
        self.assertEqual(shield.state.target_reduction_mechanical_kw, 285.)
        self.assertAlmostEqual(shield._reduction_at(1.), 190.)

    def test_entire_cold_hold_and_ramp_remain_reachable(self):
        shield, state, command = fixture()
        state = replace(state, energy_mj=2.87, previous_dc_kw=320.)
        shield.reset(state)
        shield.state = replace(shield.state, reference_mechanical_kw=304.)
        phases = set()
        for interval in range(100):
            shield.decide(state, command, 1.)
            records = []
            initial = state
            for step in range(5):
                q = shield.query(state, command, shield.geometry, .01)
                phases.add(q['request_proxy_phase'])
                self.assertTrue(q['constraint_intersection_feasible'], (interval, step, q))
                rec = record(state, .01, q['requested_dc_kw'])
                records.append(rec)
                state = replace(state, s_m=rec['s_end_m'],
                                elapsed_time_s=rec['time_end_s'],
                                energy_mj=rec['energy_next_mj'],
                                previous_dc_kw=rec['dc_kw'])
            shield.accept(state, records)
        self.assertEqual(phases, {'HOLD', 'LIMITED'})
        self.assertGreaterEqual(state.energy_mj, 2.)
        self.assertEqual(state.previous_dc_kw, 0.)
        self.assertTrue(shield.certificate()['reachable_request_gate_passed'])
        self.assertTrue(shield.certificate()['public_rules_passed'])

    def test_pure_queries_emit_intersection_failure_without_mutation(self):
        shield, state, command = fixture()
        state = replace(state, energy_mj=2.1, previous_dc_kw=320.)
        carry = RuleState(phase='HOLD', hold_until_s=1., reference_mechanical_kw=304.,
                          target_reduction_mechanical_kw=304., previous_dc_kw=320.)
        shield.reset(state, carry)
        before = copy.deepcopy(shield.__dict__)
        a = shield.query(state, command, shield.geometry, .02)
        shield.query(state, command, shield.geometry, .01)
        b = shield.query(state, command, shield.geometry, .02)
        self.assertEqual(a, b)
        self.assertFalse(a['constraint_intersection_feasible'])
        self.assertGreater(a['request_proxy_shortfall_dc_kw'], 0.)
        self.assertEqual(shield.state, before['state'])
        self.assertEqual(shield.engineering_violations, [])
        self.assertEqual(shield.energy_audit, [])

    def test_accept_preserves_and_rejects_impossible_cold_hold(self):
        shield, state, command = fixture()
        state = replace(state, energy_mj=2.1, previous_dc_kw=320.)
        carry = RuleState(phase='HOLD', hold_until_s=1., reference_mechanical_kw=304.,
                          target_reduction_mechanical_kw=304., previous_dc_kw=320.)
        shield.reset(state, carry)
        rec = record(state, .02, 319.)
        after = replace(state, s_m=rec['s_end_m'], elapsed_time_s=.02,
                        energy_mj=rec['energy_next_mj'], previous_dc_kw=319.)
        shield.accept(after, [rec])
        self.assertEqual(shield.state.previous_dc_kw, 319.)
        self.assertFalse(shield.certificate()['reachable_request_gate_passed'])
        self.assertIn('reachable_reference_energy_request_intersection_empty',
                      shield.engineering_violations[0]['reasons'])

    def test_notice_exceeded_is_logged_and_not_claimed_universal_safe(self):
        shield, state, command = fixture(.1)
        state = replace(state, previous_dc_kw=200.)
        shield.reset(state)
        shield.decide(state, command, 1.)
        rec = record(state, .12, 200.)
        after = replace(state, s_m=rec['s_end_m'], elapsed_time_s=.12,
                        energy_mj=rec['energy_next_mj'], previous_dc_kw=200.)
        shield.accept(after, [rec])
        cert = shield.certificate()
        self.assertFalse(cert['reachable_request_gate_passed'])
        self.assertIn('energy_decision_notice_exceeded', shield.engineering_violations[0]['reasons'])
        self.assertFalse(cert['energy_viability_guard']['universal_reachability_proof'])

    def test_only_real_negative_wheel_demand_resets_peak_and_target(self):
        shield, state, command = fixture()
        carry = RuleState(phase='LIMITED', reference_mechanical_kw=285.,
                          target_reduction_mechanical_kw=285., reduction_mechanical_kw=95.)
        shield.reset(state, carry)
        shield.decide(state, {**command, 'target_acceleration_mps2': -2.}, 1.)
        self.assertEqual(shield.state.phase, 'LIMITED')
        shield.decide(state, {**command, 'wheel_force_n': -1.}, -1.)
        self.assertEqual(shield.state.phase, 'NORMAL')
        self.assertEqual(shield.state.reference_mechanical_kw, 0.)
        self.assertEqual(shield.state.target_reduction_mechanical_kw, 0.)

    def test_control_line_carries_every_rule_field_without_energy_repair(self):
        shield, state, _ = fixture()
        carry = RuleState(phase='HOLD', elapsed_time_s=70., hold_until_s=70.4,
                          reference_mechanical_kw=285., target_reduction_mechanical_kw=285.,
                          reduction_mechanical_kw=0., previous_dc_kw=300., requested_dc_kw=0.)
        reset = shield.reset(replace(state, previous_dc_kw=300.), asdict(carry))
        self.assertAlmostEqual(reset.hold_until_s, .4)
        self.assertEqual(reset.reference_mechanical_kw, 285.)
        self.assertEqual(reset.target_reduction_mechanical_kw, 285.)
        self.assertFalse(shield.certificate()['full_fia_verified'])

    def test_actual_K_is_separate_from_declared_request(self):
        shield, state, command = fixture()
        state = replace(state, previous_dc_kw=300.)
        carry = RuleState(phase='HOLD', hold_until_s=1., reference_mechanical_kw=285.,
                          target_reduction_mechanical_kw=285., previous_dc_kw=300.)
        shield.reset(state, carry)
        self.assertAlmostEqual(shield.query(state, command, shield.geometry, .02)
                               ['declared_proxy_dc_kw'], 300.)
        cert = shield.certificate()
        self.assertFalse(cert['energy_viability_guard']
                         ['FIA_C5_12_4_actual_K_constant_requirement_claimed'])
        self.assertFalse(cert['full_fia_verified'])


if __name__ == '__main__':
    unittest.main()
