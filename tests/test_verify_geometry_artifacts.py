"""Independent byte pins detect rewritten evidence without invoking simulation."""
from pathlib import Path
import json
import hashlib
import tempfile
import unittest
from tools.verify_geometry_artifacts import verify, VerificationError, IDENTITY


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeometryArtifactTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'results'; root.mkdir()
        frozen = {'identity': IDENTITY, 'code_commit': 'a'*40,
            'parent_identity': 'parent', 'parent_completion_sha256': 'b'*64}
        files = {'freeze.json': dump(root/'freeze.json', frozen)}
        for stage in ('geometry-audit', 'physics-refit', 'control-response'):
            files[stage+'.json'] = dump(root/(stage+'.json'), {'status': 'saved_diagnostic'})
        manifest_sha = dump(root/'manifest.json', {'identity': IDENTITY, 'files': files})
        pin = {**frozen, 'manifest_sha256': manifest_sha, 'freeze_sha256': files['freeze.json'],
               'stage_sha256': {s: files[s+'.json'] for s in ('geometry-audit', 'physics-refit', 'control-response')}}
        summary = Path(directory)/'summary.json'; dump(summary, pin)
        return root, summary

    def test_valid_read_only_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root, summary = self.fixture(d)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            self.assertTrue(verify(root, summary)['passed'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_stage_rewrite_and_unexpected_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root, summary = self.fixture(d)
            dump(root/'control-response.json', {'status': 'fake_improvement'})
            with self.assertRaises(VerificationError): verify(root, summary)
        with tempfile.TemporaryDirectory() as d:
            root, summary = self.fixture(d)
            (root/'extra.txt').write_text('unexpected')
            with self.assertRaises(VerificationError): verify(root, summary)

    def test_source_commit_tamper_cannot_be_rehashed_under_old_pin(self):
        with tempfile.TemporaryDirectory() as d:
            root, summary = self.fixture(d)
            row = json.loads((root/'freeze.json').read_text()); row['code_commit'] = 'c'*40
            dump(root/'freeze.json', row)
            with self.assertRaises(VerificationError): verify(root, summary)


if __name__ == '__main__': unittest.main()
