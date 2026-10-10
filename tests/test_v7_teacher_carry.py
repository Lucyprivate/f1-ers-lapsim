from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f1ers_v7 import teacher_carry as module


def lap(time=70., *, feasible=True, sustainable=True, completed=True):
    summary = {name: True for name in ('completed_lap', 'execution_feasible', 'public_rules_passed',
        'periodic_speed_closed', 'dc_boundary_closed', 'rule_state_closed', 'reachable_request_gate_passed')}
    summary.update(feasible=feasible, sustainable=sustainable, status='accepted',
                   completed_lap=completed, lap_time_s=time if completed else None,
                   energy_start_mj=3., energy_end_mj=3. if sustainable else 2.99,
                   distance_completed_m=4300. if completed else 2000.)
    return {'summary': summary}


class FakeSim:
    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = []

    def simulate_laps(self, controller, **kwargs):
        self.calls.append(kwargs)
        return deepcopy(self.proposals[len(self.calls)-1])


def sequence(laps, *, crossing=True):
    return {'laps': laps, 'physical_control_line_continuity': crossing, 'crossings': []}


def proxy(*args):
    return {'passed': True, 'reachable_request_gate_passed': True,
            'additional_surrogate_hold_and_limited_request_gate_passed': True,
            'engineering_violations': []}


class CarrySearchTests(unittest.TestCase):
    def test_registry_fixed_eight_baselines_and_finite_bounded_perturbations(self):
        a = module.proposal_registry([.5])
        b = module.proposal_registry([.5])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8)
        self.assertEqual(a[0]['values'], [.5]*6)
        self.assertEqual(a[1]['values'], [0.]*6)
        self.assertEqual(a[2]['values'], [.375]*6)
        self.assertNotEqual(a[3]['values'], a[4]['values'])
        self.assertTrue(all(0. <= v <= 1. for row in a for v in row['values']))
        self.assertEqual(module.CarrySearch, module.carry_search)

    def test_budget_and_initial_family_bounds_cannot_expand(self):
        for evaluations in (2, 9):
            with self.assertRaises(ValueError):
                module.carry_search(FakeSim([]), 'unused', initial_values=[.5],
                                    initial_speed_mps=77., evaluations=evaluations)
        for values in ([], [-.1], [1.1], [float('nan')]):
            with self.assertRaises(ValueError):
                module.proposal_registry(values)
        for blocks in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                module.proposal_registry([.5], blocks=blocks)

    def test_six_lap_nondeficit_beats_faster_last_lap_energy_failure(self):
        good = sequence([lap(72.)]*6)
        faster = sequence([lap(70.)]*5+[lap(70., feasible=False, sustainable=False)])
        moderate = sequence([lap(71.)]*6)
        sim = FakeSim([good, faster, moderate])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(module, 'proxy_audit', side_effect=proxy):
            result = module.carry_search(sim, Path(tmp)/'trial', initial_values=[.5],
                                        initial_speed_mps=77., evaluations=3)
            selected = result['selected_directory']
            self.assertTrue((selected/'lap_6.json.gz').is_file())
            self.assertTrue((selected/'lap_6_proxy.json').is_file())
            failed = json.loads((Path(tmp)/'trial/trial_001/qualification.json').read_text())
            self.assertIsNone(failed['last_two_mean_time_s'])
            self.assertEqual(failed['diagnostic_last_two_complete_lap_mean_time_s'], 70.)
        self.assertEqual(result['report']['best_time_s'], 71.)
        self.assertEqual(result['qualification']['last_two_mean_time_s'], 71.)
        self.assertEqual(result['controller'].values.tolist(), [.375]*6)
        self.assertIsNone(result['report']['history'][1]['objective_s'])
        self.assertEqual(len(sim.calls), 3)
        self.assertTrue(all(k == {'n_laps': 6, 'initial_speed_mps': 77.,
                                 'initial_energy_mj': 2., 'action_space': 'mode_aware'}
                            for k in sim.calls))

    def test_proxy_failure_and_all_failed_traces_are_retained(self):
        sim = FakeSim([sequence([lap()]*6)]*3)
        failed = {**proxy(), 'passed': False, 'reachable_request_gate_passed': False,
                  'engineering_violations': [{'reasons': ['intersection_empty']}]}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(module, 'proxy_audit', return_value=failed):
            target = Path(tmp)/'trial'
            result = module.carry_search(sim, target, initial_values=[.5],
                                        initial_speed_mps=77., evaluations=3)
            self.assertEqual(len(list(target.glob('trial_*/lap_*.json.gz'))), 18)
            self.assertEqual(len(list(target.glob('trial_*/lap_*_proxy.json'))), 18)
            self.assertEqual(len(list(target.glob('trial_*/qualification.json'))), 3)
        self.assertIsNone(result['controller'])
        self.assertIsNone(result['result'])
        self.assertIsNone(result['laps'])
        self.assertIsNone(result['selected_directory'])

    def test_partial_lap_and_bad_physical_crossing_cannot_rank(self):
        partial = sequence([lap()]*3+[lap(feasible=False, completed=False)], crossing=False)
        crossing = sequence([lap()]*6, crossing=False)
        sim = FakeSim([partial, crossing, partial])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(module, 'proxy_audit', side_effect=proxy):
            result = module.carry_search(sim, Path(tmp)/'trial', initial_values=[.5],
                                        initial_speed_mps=77., evaluations=3)
        self.assertIsNone(result['controller'])
        self.assertEqual(result['report']['complete_six_lap_feasible_count'], 0)

    def test_previous_result_cannot_be_rescored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trial'
            path.mkdir()
            (path/'started.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                module.carry_search(FakeSim([]), path, initial_values=[.5],
                                    initial_speed_mps=77., evaluations=3)


if __name__ == '__main__':
    unittest.main()
