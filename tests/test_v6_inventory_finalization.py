"""Standalone inventory finalization safety tests; no repository mutation."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from tools.finalize_v6_inventory import finalize, REASON


def digest(value):
    return hashlib.sha256(value).hexdigest()


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        temp_root=Path(__file__).resolve().parents[1]/'.cache'/'v6-inventory-tests'
        temp_root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=temp_root)
        self.root=Path(self.tmp.name)
        (self.root/'calculation.json').write_bytes(b'{"frozen":true}')
        (self.root/'run.log').write_bytes(b'calculation finished\n')
        self.manifest={'schema_version':6,'code_commit':'frozen-code',
            'files':{'calculation.json':digest((self.root/'calculation.json').read_bytes()),
                'run.log':digest((self.root/'run.log').read_bytes())}}
        self.original=json.dumps(self.manifest,indent=2).encode()
        (self.root/'manifest.json').write_bytes(self.original)
        with (self.root/'run.log').open('ab') as stream:stream.write(b'outer CLI final stdout\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_external_log_digest_changes_and_original_is_preserved(self):
        before=(self.root/'calculation.json').read_bytes()
        audit=finalize(self.root,runner_closed=True)
        updated=json.loads((self.root/'manifest.json').read_text())
        expected=dict(self.manifest)
        expected['files']=dict(self.manifest['files'])
        expected['files']['run.log']=digest((self.root/'run.log').read_bytes())
        self.assertEqual(updated,expected)
        self.assertEqual((self.root/'manifest_at_return.json').read_bytes(),self.original)
        self.assertEqual((self.root/'calculation.json').read_bytes(),before)
        self.assertEqual(audit['reason'],REASON)
        self.assertFalse(audit['calculation_modified'])
        self.assertFalse(audit['frozen_input_modified'])
        self.assertEqual(audit['manifest_sha256_before'],digest(self.original))
        self.assertEqual(audit['manifest_sha256_after'],digest((self.root/'manifest.json').read_bytes()))
        self.assertEqual(audit['log_sha256_before'],self.manifest['files']['run.log'])
        self.assertNotEqual(audit['log_sha256_before'],audit['log_sha256_after'])
        self.assertNotIn('inventory_finalization.json',updated['files'])
        self.assertNotIn('manifest_at_return.json',updated['files'])

    def test_changed_calculation_is_rejected_before_any_write(self):
        (self.root/'calculation.json').write_bytes(b'{"frozen":false}')
        with self.assertRaisesRegex(ValueError,'calculation or frozen artifact changed'):
            finalize(self.root,runner_closed=True)
        self.assertEqual((self.root/'manifest.json').read_bytes(),self.original)
        self.assertFalse((self.root/'manifest_at_return.json').exists())

    def test_relative_traversal_is_rejected(self):
        self.manifest['files']['../outside.json']='0'*64
        changed=json.dumps(self.manifest).encode()
        (self.root/'manifest.json').write_bytes(changed)
        with self.assertRaisesRegex(ValueError,'Unsafe manifest member'):
            finalize(self.root,runner_closed=True)
        self.assertEqual((self.root/'manifest.json').read_bytes(),changed)
        self.assertFalse((self.root/'manifest_at_return.json').exists())

    def test_absolute_and_windows_paths_are_rejected(self):
        for member in ('/outside.json','C:/outside.json',r'..\outside.json'):
            with self.subTest(member=member):
                value=dict(self.manifest);value['files']=dict(self.manifest['files'])
                value['files'][member]='0'*64
                (self.root/'manifest.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError):finalize(self.root,runner_closed=True)
                self.assertFalse((self.root/'manifest_at_return.json').exists())

    def test_duplicate_members_are_rejected(self):
        value='{"files":{"run.log":"'+('0'*64)+'","run.log":"'+('1'*64)+'"}}'
        (self.root/'manifest.json').write_text(value)
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):
            finalize(self.root,runner_closed=True)
        self.assertFalse((self.root/'manifest_at_return.json').exists())

    def test_runner_exit_confirmation_is_required(self):
        with self.assertRaisesRegex(ValueError,'outer runner has exited'):
            finalize(self.root)
        self.assertEqual((self.root/'manifest.json').read_bytes(),self.original)

    def test_second_invocation_cannot_rewrite_the_audit(self):
        finalize(self.root,runner_closed=True)
        original_backup=(self.root/'manifest_at_return.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already exists'):
            finalize(self.root,runner_closed=True)
        self.assertEqual((self.root/'manifest_at_return.json').read_bytes(),original_backup)

    def test_a_calculation_cannot_be_designated_as_the_log_exception(self):
        with self.assertRaisesRegex(ValueError,'Only the external run.log'):
            finalize(self.root,runner_closed=True,log_relative='calculation.json')
        self.assertFalse((self.root/'manifest_at_return.json').exists())

    def test_prior_stage_inventory_cannot_be_modified(self):
        self.manifest['schema_version']=5
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'only finalizes a v6'):
            finalize(self.root,runner_closed=True)
        self.assertFalse((self.root/'manifest_at_return.json').exists())


if __name__=='__main__':unittest.main()
