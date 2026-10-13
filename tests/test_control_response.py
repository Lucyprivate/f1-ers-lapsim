"""Synthetic executions check durable budgets and carried response semantics."""
import copy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from f1ers_v7._core.integrator import IntegrationState, IntegratorConfig, VehicleParamsV4
from f1ers_v7.research.control_response import run_control_response, value_hash


class SharedBudget:
    def __init__(self):
        self.entries = {}

    def claim(self, key, details):
        if key in self.entries:
            raise RuntimeError('A branch or clone has already spent this attempt')
        if len(self.entries)>=132:
            raise RuntimeError('Shared maximum reached')
        self.entries[key] = copy.deepcopy(details)
        return {'key': key, 'slot': len(self.entries)}


class FixtureEnv:
    def __init__(self, sim, initial_speed_mps, initial_energy_mj,
                 carried_rule_state, numerics, output_ds_m, **kwargs):
        self.sim = sim
        self.initial = IntegrationState(0., initial_speed_mps, 0., initial_energy_mj, 0., 0.)
        self.state, self.i, self.done = self.initial, 0, False
        self.carried = copy.deepcopy(carried_rule_state)
        self.output_ds_m = output_ds_m
        self.driver = SimpleNamespace(decision_s=np.linspace(0., 160., 17), rows=[])
        self.records, self.raw_actions, self.events, self.history = [], [], [], []
        self.failure = None
        self.speeds, self.energies, self.times = [initial_speed_mps], [initial_energy_mj], [0.]
        self.call_number = len(sim.calls)+1
        sim.calls.append({'boundary': {'speed_mps': initial_speed_mps,
            'energy_mj': initial_energy_mj, 'carried_rule_state': copy.deepcopy(carried_rule_state)},
            'numerics': numerics or sim.config, 'output_ds_m': output_ds_m, 'actions': self.raw_actions,
            'claim_was_present': self.call_number<=len(sim.budget.entries)})

    def get_obs(self):
        return np.array([self.state.s_m/160.])

    def step(self, action):
        if self.call_number==self.sim.interrupt_at_call and self.i==1:
            raise KeyboardInterrupt('Synthetic interruption before a new accepted interval')
        active = self.sim.active_cold if self.carried is None else self.sim.active_warm
        effective = float(action) if active else 0.
        old = self.state
        speed = old.speed_mps+.001/16.+effective*.1
        energy = old.energy_mj+.1/16.+effective*.01
        dt = 20./(old.speed_mps+speed)
        self.state = IntegrationState(old.s_m+10., speed, old.elapsed_time_s+dt, energy, 0., 0.)
        self.records.append({'dc_kw': (old.energy_mj-energy)*1000./dt, 'dt_s': dt,
                             'energy_next_mj': energy})
        self.driver.rows.append({'s_m': old.s_m, 'wheel_force_n': 2000.})
        self.raw_actions.append(float(action))
        self.speeds.append(speed); self.energies.append(energy); self.times.append(self.state.elapsed_time_s)
        self.history.append({'s_m': self.state.s_m})
        self.i += 1
        self.done = self.i==16
        return self.get_obs(), 0., self.done, {}

    def result(self):
        phase_rows = [{'s_m': 10.*i, 'state_after_decide': {
            'phase': 'HOLD' if action!=0. and self.sim.hybrid else 'NORMAL'}}
            for i, action in enumerate(self.raw_actions)]
        carry = {'phase': 'NORMAL', 'braking': False, 'previous_dc_kw': 0.,
                 'elapsed_time_s': self.state.elapsed_time_s, 'hold_until_s': 0.}
        ss = np.arange(len(self.speeds))*10.
        output_s = np.linspace(0., 160., int(np.ceil(160./self.output_ds_m))+1)[:-1]
        summary = {'completed_lap': self.done, 'execution_feasible': True,
            'feasible': True, 'public_rules_passed': True,
            'speed_start_mps': self.initial.speed_mps, 'energy_start_mj': self.initial.energy_mj,
            'energy_end_mj': self.state.energy_mj, 'initial_dc_kw': 0., 'terminal_dc_kw': 0.,
            'initial_rule_state': self.carried, 'energy_balance_residual_mj': 0.,
            'lap_time_s': self.state.elapsed_time_s}
        return {'summary': summary, 'rule_certificate': {'public_rules_passed': True,
            'terminal_boundary_state': carry}, 'terminal_status': 'accepted', 'length_m': 160.,
            'terminal_speed_mps': self.state.speed_mps, 'terminal_time_s': self.state.elapsed_time_s,
            'requested_actions': np.array(self.raw_actions),
            'driver_profile': {'s_m': ss[:-1], 'wheel_force_n': np.full(len(self.raw_actions), 2000.)},
            'energy_viability_audit': phase_rows, 's_m': output_s,
            'speed_mps': np.interp(output_s, ss, self.speeds),
            'internal_trace': {'s_m': ss, 'speed_mps': self.speeds, 'time_s': self.times,
                'energy_mj': self.energies, 'dc_kw': [r['dc_kw'] for r in self.records],
                'mechanical_work_residual_mj': np.zeros(len(self.records))}}


class FixtureSimulator:
    def __init__(self, budget, *, active_cold=True, active_warm=True, interrupt_at_call=None):
        self.budget, self.calls = budget, []
        self.active_cold, self.active_warm = active_cold, active_warm
        self.interrupt_at_call, self.hybrid = interrupt_at_call, True
        self.length_m, self.sector_boundaries_m = 160., [40., 100.]
        self.config = IntegratorConfig()
        self.rule_config = SimpleNamespace(tail_guard_m=1000.)
        self.vehicle = VehicleParamsV4(mu=1.8, drag_area_m2=1.3, downforce_area_m2=3.9,
            equivalent_power_kw=541.25, replay_brake_gain=.55)

    def create_env(self, **kwargs):
        return FixtureEnv(self, **kwargs)


class ControlResponseTests(unittest.TestCase):
    boundary = {'speed_mps': 80., 'energy_mj': 2., 'carried_rule_state': None}
    provenance = {'code_commit': 'a'*40, 'source_frozen': 'b'*40}

    def run_fixture(self, sim, output):
        with patch('f1ers_v7.research.control_response.proxy_audit', return_value={'passed': True}), \
             patch('f1ers_v7.research.control_response.frozen_proxy_audit', return_value={'passed': True}), \
             patch('f1ers_v7.research.control_response.freeze_simulator', side_effect=lambda s, profile: s):
            return run_control_response(sim, output, initial_boundary=self.boundary,
                claim=sim.budget.claim, provenance=self.provenance)

    def test_allocation_never_exceeds_132_and_all_claims_precede_environments(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget)
        with TemporaryDirectory() as tmp:
            report = self.run_fixture(sim, Path(tmp)/'results')
            self.assertEqual(len(budget.entries), 132)
            self.assertEqual(report['attempted_single_lap_calls'], 132)
            self.assertEqual(report['selected_blocks'], {'cold': [1,2,3,4], 'carried_late': [1,2,3,4]})
            self.assertTrue(all(c['claim_was_present'] for c in sim.calls))
            self.assertTrue(report['numerical_checks_passed'])
            self.assertEqual(report['SAC_attempts'], 0)
            self.assertFalse(report['new_teacher_adopted'])

    def test_carried_pool_comes_from_six_actual_runs_without_reset_or_shooting(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget, active_cold=False)
        with TemporaryDirectory() as tmp:
            report = self.run_fixture(sim, Path(tmp)/'results')
            pool = json.loads((Path(tmp)/'results/boundary_pool.json').read_text())
            carried = pool['boundaries']['carried_late']
            self.assertAlmostEqual(carried['speed_mps'], 80.006)
            self.assertAlmostEqual(carried['energy_mj'], 2.6)
            self.assertEqual(pool['sha256']['carried_late'], value_hash(carried))
            self.assertEqual(sim.calls[6]['boundary'], self.boundary)
            warm_calls = [c for c in sim.calls[7:] if c['boundary']['carried_rule_state'] is not None
                          and abs(c['boundary']['energy_mj']-2.6)<1e-9]
            self.assertTrue(warm_calls)
            self.assertTrue(all(c['boundary']==carried for c in warm_calls))
            self.assertTrue(report['baseline_actual_carry']['passed'])
            self.assertEqual(report['within_lap_speed_or_energy_resets'], 0)

    def test_cold_zero_response_does_not_suppress_actual_warm_response(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget, active_cold=False)
        with TemporaryDirectory() as tmp:
            report = self.run_fixture(sim, Path(tmp)/'results')
            self.assertEqual(report['selected_blocks']['cold'], [])
            self.assertEqual(report['selected_blocks']['carried_late'], [1,2,3,4])
            self.assertEqual(report['attempted_single_lap_calls'], 108)
            self.assertFalse(report['unspent_budget_reallocated'])

    def test_hybrid_transition_is_a_directional_response_without_a_jacobian(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget)
        with TemporaryDirectory() as tmp:
            report = self.run_fixture(sim, Path(tmp)/'results')
            row = report['directional_responses']['cold'][0]
            self.assertTrue(row['phase_schedule_changed'])
            self.assertEqual(row['response_interpretation'], 'directional hybrid response')
            self.assertFalse(row['smooth_Jacobian_estimated'])
            self.assertFalse(report['derivative_or_redistribution_dimension_claimed'])

    def test_sustainable_control_failure_does_not_hide_complete_physical_response(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget)
        original = FixtureEnv.result
        def physically_complete_nonperiodic(env):
            result = original(env)
            if any(action!=0. for action in env.raw_actions):
                result['summary']['feasible'] = False
                result['summary']['periodic_speed_closed'] = False
            return result
        with TemporaryDirectory() as tmp, patch.object(FixtureEnv, 'result', physically_complete_nonperiodic):
            report = self.run_fixture(sim, Path(tmp)/'results')
            self.assertEqual(report['selected_blocks']['cold'], [1,2,3,4])
            self.assertTrue(report['directional_responses']['cold'][0]['valid'])
            attempt = next(r for r in report['attempts'] if r['key']=='cold_b01_neg_025')
            self.assertFalse(attempt['control_qualified'])
            self.assertTrue(attempt['physical_public_complete'])
            self.assertTrue(report['numerical_checks_passed'])
            self.assertFalse(report['new_teacher_adopted'])

    def test_shared_clone_claim_rejects_before_starting_a_second_environment(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget, active_cold=False, active_warm=False)
        with TemporaryDirectory() as tmp:
            report = self.run_fixture(sim, Path(tmp)/'first')
            self.assertEqual(report['attempted_single_lap_calls'], 84)
            second = FixtureSimulator(budget)
            with self.assertRaisesRegex(RuntimeError, 'already spent'):
                self.run_fixture(second, Path(tmp)/'second')
            self.assertEqual(len(second.calls), 0)
            self.assertEqual(len(budget.entries), 84)

    def test_interrupted_attempt_is_spent_and_preserves_only_accepted_prefix(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget, interrupt_at_call=1)
        with TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                self.run_fixture(sim, Path(tmp)/'results')
            self.assertEqual(len(budget.entries), 1)
            directory = Path(tmp)/'results/attempts/carry_01'
            receipt = json.loads((directory/'execution.json').read_text())
            self.assertEqual(receipt['status'], 'interrupted')
            self.assertFalse(receipt['formal_accuracy_scored'])
            self.assertFalse((directory/'metrics.json').exists())
            with gzip.open(directory/'accepted_prefix.json.gz', 'rt') as stream:
                prefix = json.load(stream)
            self.assertEqual(len(prefix['accepted_records']), 1)
            self.assertEqual(prefix['last_accepted_state']['s_m'], 10.)
            self.assertFalse(prefix['in_flight_unaccepted_substep_retained'])

    def test_unqualified_baseline_stops_without_using_response_budget(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget)
        with TemporaryDirectory() as tmp, \
             patch('f1ers_v7.research.control_response.proxy_audit', return_value={'passed': False}):
            report = run_control_response(sim, Path(tmp)/'results', initial_boundary=self.boundary,
                claim=budget.claim, provenance=self.provenance)
            self.assertEqual(report['status'], 'baseline_actual_carry_qualification_failed')
            self.assertEqual(report['attempted_single_lap_calls'], 1)
            self.assertTrue((Path(tmp)/'results/attempts/carry_01/trajectory.json.gz').is_file())

    def test_model_or_guard_changes_are_rejected_before_the_first_claim(self):
        budget = SharedBudget(); sim = FixtureSimulator(budget)
        sim.vehicle.mu = 1.75
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'qualified ideal baseline'):
                self.run_fixture(sim, Path(tmp)/'results')
            self.assertEqual(len(budget.entries), 0)
            self.assertFalse((Path(tmp)/'results').exists())


if __name__=='__main__':
    unittest.main()
