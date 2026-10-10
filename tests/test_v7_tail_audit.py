"""No vehicle computation: validate the separate tail experiment contract."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from f1ers_v7 import tail_audit as module


def receipt(name='accepted'):
    saved = module.read(Path(__file__).parent/'fixtures/v7_carry_receipts.json')
    return deepcopy(saved['receipts'][name]['qualification'])


def config():
    value = json.loads((ROOT/'config/v7/tail_guard.json').read_text())
    value.update(prior_source_commit='a'*40, prior_manifest_sha256='b'*64,
                 prior_selection_sha256='c'*64, prior_controller_spec_sha256='d'*64,
                 energy_notice_s=.5, initial_speed_mps=77.)
    return value


def prepare(root, controller=None, include_checkpoint=True):
    previous = root/'results'/'austria_2026_v7B'
    previous.mkdir(parents=True)
    controller = controller or {'kind': 'phase_blocks_with_actual_braking', 'values': [.5],
                                'action_space': 'mode_aware'}
    winner = {'qualification': {'passed': True}, 'controller': controller,
              'energy_notice_s': .5, 'initial_speed_mps': 77.}
    (previous/'selection.json').write_text(json.dumps({'winner': winner}))
    (previous/'experiment.json').write_text(json.dumps({'source_commit': 'a'*40}))
    if controller['kind'] == 'policy':
        checkpoint = previous/controller['checkpoint']
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'immutable_checkpoint_test_bytes')
    manifest = {'files': {p.relative_to(previous).as_posix(): module.digest(p)
                          for p in previous.rglob('*') if p.is_file()
                          and (include_checkpoint or p.name != 'policy.npz')}}
    (previous/'manifest.json').write_text(json.dumps(manifest))
    settings = config()
    settings.update(prior_manifest_sha256=module.digest(previous/'manifest.json'),
                    prior_selection_sha256=module.digest(previous/'selection.json'),
                    prior_controller_spec_sha256=module.canonical_controller_hash(controller))
    (root/'config/v7').mkdir(parents=True)
    (root/'config/v7/tail_guard.json').write_text(json.dumps(settings))
    return settings


class TailContractTests(unittest.TestCase):
    def test_config_rejects_unbound_inputs_and_parameter_expansion(self):
        module.validate_config(config())
        template = config()
        template['prior_manifest_sha256'] = 'FINAL_B_MANIFEST_SHA256_TO_BIND'
        with self.assertRaises(ValueError):
            module.validate_config(template)
        for mutation in ({'tail_guard_choices_m': [0., 350., 700., 1000.]},
                         {'initial_energy_mj': 4.}, {'new_SAC_episodes': 1},
                         {'observed_residual_used_in_selection': True}):
            bad = {**config(), **mutation}
            with self.assertRaises(ValueError):
                module.validate_config(bad)

    def test_controller_hash_is_order_stable_and_values_sensitive(self):
        a = {'values': [.5], 'kind': 'phase', 'action_space': 'mode_aware'}
        b = {'action_space': 'mode_aware', 'kind': 'phase', 'values': [.5]}
        self.assertEqual(module.canonical_controller_hash(a), module.canonical_controller_hash(b))
        self.assertNotEqual(module.canonical_controller_hash(a),
                            module.canonical_controller_hash({**a, 'values': [.6]}))

    def test_inventory_paths_cannot_escape_prior(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            (root/'safe.json').write_text('{}')
            self.assertEqual(module.safe_file(root, 'safe.json'), root/'safe.json')
            for name in ('../safe.json', '/safe.json', 'C:/safe.json', 'a\\safe.json', './safe.json'):
                with self.assertRaises(ValueError):
                    module.safe_file(root, name)

    def test_prior_file_tamper_rejected_before_vehicle_or_scoring(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            prepare(root)
            prior = root/'results/austria_2026_v7B'
            (prior/'experiment.json').write_text(json.dumps({'source_commit': 'f'*40}))
            with patch.object(module, 'provenance', return_value={}), \
                    patch.object(module, 'ReachableSimulator') as sim, \
                    patch.object(module, 'score') as score:
                with self.assertRaisesRegex(ValueError, 'Frozen B artifact changed'):
                    module.tail_guard_audit(root)
            sim.assert_not_called()
            score.assert_not_called()
            self.assertFalse((root/'results/austria_2026_v7C').exists())

    def test_policy_checkpoint_requires_prior_manifest_binding(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            prepare(root, {'kind': 'policy', 'checkpoint': 'weights/policy.npz',
                           'action_space': 'mode_aware'}, include_checkpoint=False)
            with patch.object(module, 'provenance', return_value={'code_commit': 'e'*40}), \
                    patch.object(module, 'ReachableSimulator') as sim, \
                    patch.object(module, 'score') as score:
                with self.assertRaisesRegex(ValueError, 'not bound by final B inventory'):
                    module.tail_guard_audit(root)
            sim.assert_not_called()
            score.assert_not_called()

    def test_existing_tail_experiment_cannot_get_a_second_scoring_opportunity(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            prepare(root)
            output = root/'results/austria_2026_v7C'
            output.mkdir()
            (output/'failed_trace.json').write_text('original trace')
            with patch.object(module, 'provenance') as provenance:
                with self.assertRaisesRegex(RuntimeError, 'cannot overwrite or rescore'):
                    module.tail_guard_audit(root)
            provenance.assert_not_called()
            self.assertEqual((output/'failed_trace.json').read_text(), 'original trace')

    def test_three_primary_guards_keep_initial_boundary_and_rank_without_residual(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            prepare(root)
            calls, scored = [], []

            def simulator(track, rules, **kwargs):
                return SimpleNamespace(guard=kwargs['rule_config'].tail_guard_m)

            def qualify(sim, controller, action_space, directory, initial_speed, data):
                calls.append((sim.guard, initial_speed, action_space))
                directory.mkdir(parents=True)
                # The fastest guard deliberately has worse diagnostic RMSE.
                q = receipt()
                q['last_two_mean_time_s'] = sim.guard/100.+65.
                for summary in q['lap_summaries']:
                    summary['lap_time_s'] = q['last_two_mean_time_s']
                return q, [{'diagnostic_rmse': 50. if sim.guard == 350. else 10.}]*6

            def score(result, data, role):
                scored.append(role)
                return {'aggregate': {'mean_speed_rmse_kmh': result['diagnostic_rmse']},
                        'accuracy_passed': False}

            with patch.object(module, 'provenance', return_value={
                    'code_commit': 'e'*40, 'FP2_ledger_before': {}, 'tracked_input_sha256': {}}), \
                    patch.object(module, 'load_local_dataset', return_value={
                        'audit': {'data_gate_passed': True}, 'track': {}, 'rules': {}}), \
                    patch.object(module, 'ReachableSimulator', side_effect=simulator), \
                    patch.object(module, 'controller_from', return_value=object()), \
                    patch.object(module, 'qualify', side_effect=qualify), \
                    patch.object(module, 'score', side_effect=score), \
                    patch.object(module, 'ledger_paths', return_value={}):
                result = module.tail_guard_audit(root)
            self.assertEqual(calls, [(350., 77., 'mode_aware'), (700., 77., 'mode_aware'),
                                     (1000., 77., 'mode_aware')])
            self.assertEqual(scored, ['calibration']*3)
            self.assertEqual(result['winner']['tail_guard_m'], 350.)
            self.assertEqual(result['report']['actual_cases'], 3)
            self.assertFalse(result['report']['development_scored_inside_audit'])
            self.assertFalse(result['report']['E4_single_lap_sensitivity_in_selection'])

    def test_saved_six_lap_energy_failure_cannot_rank_despite_complete_mean(self):
        good = receipt()
        failed = receipt('complete_energy_failed')
        self.assertTrue(module.qualification_gate_passed(good))
        self.assertIsNotNone(failed['last_two_mean_time_s'])
        self.assertFalse(module.qualification_gate_passed(failed))
        # A mislabeled top-level pass cannot conceal a real carried-state loss.
        tampered = deepcopy(failed)
        tampered['passed'] = True
        self.assertFalse(module.qualification_gate_passed(tampered))

    def test_saved_full_carry_requires_each_rule_and_power_boundary(self):
        original = receipt()
        for gate in ('public_rules_passed', 'dc_boundary_closed', 'rule_state_closed',
                     'reachable_request_gate_passed', 'sustainable'):
            changed = deepcopy(original)
            changed['lap_summaries'][3][gate] = False
            self.assertFalse(module.qualification_gate_passed(changed))
        changed = deepcopy(original)
        changed['crossings'][2]['energy_difference_mj'] = .01
        self.assertFalse(module.qualification_gate_passed(changed))
        changed = deepcopy(original)
        changed['lap_summaries'][4]['energy_end_mj'] -= .1
        self.assertFalse(module.qualification_gate_passed(changed))


if __name__ == '__main__':
    unittest.main()
