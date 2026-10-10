"""Gate regression using temporary receipts; no simulation or training attempts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from f1ers_v7.research.protocol import EpisodeBudget


# The standalone draft can be tested without changing the frozen source. After
# adoption into tests/, import the corresponding installed source normally.
draft = Path(__file__).with_name('runner_learning.py')
if draft.exists():
    spec = importlib.util.spec_from_file_location(
        'f1ers_v7.research._runner_learning_correction_draft', draft)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
else:
    from f1ers_v7.research import runner_learning as runner


class ResearchRunnerGateTests(unittest.TestCase):
    def test_refused_teacher_certifies_none_and_spends_no_episodes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                         prefix='runner_gate_') as temporary:
            root = Path(temporary)
            output = root/'output'
            output.mkdir()
            (output/'teacher.json').write_text(json.dumps({
                'status': 'stopped_teacher_surrogate_validation',
                'learning_allowed': False}), encoding='utf-8')
            ledger = root/'temporary_ledger'
            before = EpisodeBudget(ledger).counts()
            # Any progress beyond the refusal gate is a regression. These mocks
            # forbid actual input processing, qualification, or episode starts.
            with mock.patch.object(runner, 'load_historical', side_effect=AssertionError('Data loaded')), \
                 mock.patch.object(runner, 'fit_bc', side_effect=AssertionError('BC attempted')), \
                 mock.patch.object(runner, 'qualify', side_effect=AssertionError('Qualification attempted')), \
                 mock.patch.object(runner, 'train_case', side_effect=AssertionError('Training attempted')), \
                 mock.patch.object(EpisodeBudget, 'claim', side_effect=AssertionError('Budget claimed')):
                result = runner.run_learning(root/'fake_repo', root/'absent_data',
                    output, ledger, {}, backend='both')
            saved = json.loads((output/'learning.json').read_text(encoding='utf-8'))
            self.assertEqual(result, saved)
            self.assertEqual(saved['status'], 'withheld_teacher_or_physics_gate')
            self.assertEqual(saved['reason'], 'stopped_teacher_surrogate_validation')
            self.assertEqual(saved['candidates'], [])
            self.assertEqual(saved['cases'], [])
            self.assertEqual(saved['learner_certification_counts'], {
                'qualification_attempts': 0, 'passed_candidates': 0})
            self.assertNotIn('best_last_and_BC_six_carry_qualified', saved)
            self.assertEqual(saved['episode_budget_counts'], before)
            self.assertEqual(EpisodeBudget(ledger).counts(), before)
            self.assertEqual(before['attempted'], 0)
            self.assertEqual(before['per_backend'], {'numpy': 0, 'sb3': 0})
            self.assertEqual(list((output/'learning').rglob('qualification.json')), [])

    def test_certificate_counts_include_rejections_without_claiming_success(self):
        candidates = [
            {'kind': 'BC', 'qualification': {'passed': True, 'physical_laps': 6}},
            {'kind': 'best', 'qualification': {'passed': False, 'physical_laps': 2}},
            {'kind': 'last', 'qualification': {'passed': False, 'error': 'Interrupted'}},
            {'kind': 'unqualified_intermediate'}]
        self.assertEqual(runner._qualification_counts(candidates), {
            'qualification_attempts': 3, 'passed_candidates': 1})


if __name__ == '__main__':
    unittest.main()
