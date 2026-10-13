"""Cross-instance attempt accounting and sealed-study path safety."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from f1ers_v7.research.study_protocol import AttemptBudget


class DurableBudgetTests(unittest.TestCase):
    def test_clone_cannot_reuse_expand_or_overspend(self):
        with tempfile.TemporaryDirectory() as name:
            one = AttemptBudget(name, 2)
            self.assertEqual(one.claim('cold_baseline'), 1)
            clone = AttemptBudget(name, 2)
            with self.assertRaises(FileExistsError): clone.claim('cold_baseline')
            self.assertEqual(clone.claim('late_baseline'), 2)
            with self.assertRaises(RuntimeError): one.claim('third')
            self.assertEqual(clone.counts()['attempted'], 2)
            with self.assertRaises(RuntimeError): AttemptBudget(name, 3)

    def test_interruption_key_and_slot_are_retained(self):
        with tempfile.TemporaryDirectory() as name:
            one = AttemptBudget(name, 2)
            one.claim('interrupted', {'condition': 'cold'})
            resumed = AttemptBudget(name, 2)
            self.assertEqual(resumed.counts()['attempted'], 1)
            self.assertTrue((Path(name)/'keys/interrupted.json').is_file())
            for key in ('../escape', 'x:y', '', 'x/y'):
                with self.assertRaises(ValueError): resumed.claim(key)

    def test_crash_between_slot_and_key_spends_capacity(self):
        from f1ers_v7.research import study_protocol as protocol
        with tempfile.TemporaryDirectory() as name:
            one = AttemptBudget(name, 1)
            original = protocol.write_exclusive
            def stop_on_key(path, value):
                if Path(path).parent.name == 'keys':
                    raise KeyboardInterrupt('synthetic interruption after numbered slot')
                return original(path, value)
            with patch.object(protocol, 'write_exclusive', stop_on_key):
                with self.assertRaises(KeyboardInterrupt): one.claim('interrupted')
            clone = AttemptBudget(name, 1)
            self.assertEqual(clone.counts()['attempted'], 1)
            self.assertEqual(clone.counts()['reserved_keys'], 0)
            with self.assertRaises(RuntimeError): clone.claim('replacement')


if __name__ == '__main__': unittest.main()
