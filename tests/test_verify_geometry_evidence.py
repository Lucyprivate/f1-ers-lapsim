"""Additive sealed-evidence regressions; no simulator or telemetry execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.verify_geometry_evidence import verify, VerificationError, IDENTITY


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / 'tools/verify_geometry_evidence.py'
STAGES = ('geometry-audit', 'physics-refit', 'control-response')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return digest(path)


class GeometryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='geometry_evidence_fixture_', dir=REPOSITORY)
        self.base = Path(self.temp.name)
        if not self.base.resolve().is_relative_to(REPOSITORY.resolve()):
            raise RuntimeError('Synthetic fixture escapes its intended checkout')
        self.addCleanup(self.temp.cleanup)
        self.root = self.base / 'artifacts'
        (self.root / 'geometry').mkdir(parents=True)
        self.frozen = {'identity': IDENTITY, 'code_commit': 'a' * 40,
            'parent_identity': 'sealed_parent', 'parent_completion_sha256': 'b' * 64}
        files = {'freeze.json': dump(self.root / 'freeze.json', self.frozen)}
        for stage in STAGES:
            files[stage + '.json'] = dump(self.root / (stage + '.json'), {'status': 'saved'})
        self.leaf = self.root / 'geometry/mass_kinematics_sensitivity.json'
        files['geometry/mass_kinematics_sensitivity.json'] = dump(self.leaf,
            [{'mass_kg': 800, 'demand_force_n': [12.0, -3.0]}, {'mass_kg': 825}])
        self.manifest = {'identity': IDENTITY, 'files': files}
        manifest_sha = dump(self.root / 'manifest.json', self.manifest)
        self.pin = {**self.frozen, 'manifest_sha256': manifest_sha,
            'freeze_sha256': files['freeze.json'], 'manifest_file_count': len(files),
            'stage_sha256': {stage: files[stage + '.json'] for stage in STAGES}}
        self.summary = self.base / 'summary.json'
        dump(self.summary, self.pin)

    def repin_leaf(self, payload):
        self.leaf.write_text(payload, encoding='utf-8')
        self.manifest['files']['geometry/mass_kinematics_sensitivity.json'] = digest(self.leaf)
        self.pin['manifest_sha256'] = dump(self.root / 'manifest.json', self.manifest)
        dump(self.summary, self.pin)

    def test_list_root_artifact_accepted_without_mutation(self):
        before = {path.relative_to(self.root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.root.rglob('*') if path.is_file()}
        result = verify(self.root, self.summary)
        self.assertTrue(result['passed'])
        self.assertEqual(result['declared_files_verified'], 5)
        self.assertFalse(result['simulation_repeated'])
        self.assertFalse(result['telemetry_loaded'])
        after = {path.relative_to(self.root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_other_finite_json_leaf_roots_accepted(self):
        for payload in ('null', 'true', '17', '1.25', '"diagnostic"', '[]'):
            with self.subTest(payload=payload):
                self.repin_leaf(payload)
                self.assertTrue(verify(self.root, self.summary)['passed'])

    def test_list_root_byte_tamper_rejected(self):
        self.leaf.write_text('[{"mass_kg": 700}]', encoding='utf-8')
        with self.assertRaisesRegex(VerificationError, 'Artifact bytes changed'):
            verify(self.root, self.summary)

    def test_duplicate_keys_inside_list_rejected_even_when_hashes_match(self):
        self.repin_leaf('[{"mass_kg": 800, "mass_kg": 825}]')
        with self.assertRaisesRegex(VerificationError, 'Duplicate JSON key'):
            verify(self.root, self.summary)

    def test_nonfinite_numbers_rejected_even_when_hashes_match(self):
        for payload in ('[NaN]', '[Infinity]', '[-Infinity]', '[1e999]', '{"value": -1e999}'):
            with self.subTest(payload=payload):
                self.repin_leaf(payload)
                with self.assertRaisesRegex(VerificationError, 'Non-finite JSON number'):
                    verify(self.root, self.summary)

    def test_manifest_freeze_and_summary_still_require_object_roots(self):
        for name in ('manifest.json', 'freeze.json', 'summary.json'):
            with self.subTest(name=name):
                path = self.summary if name == 'summary.json' else self.root / name
                original = path.read_bytes()
                path.write_text('[]', encoding='utf-8')
                try:
                    with self.assertRaisesRegex(VerificationError, 'Expected a JSON object'):
                        verify(self.root, self.summary)
                finally:
                    path.write_bytes(original)

    def test_unexpected_inventory_entry_rejected(self):
        (self.root / 'extra.json').write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(VerificationError, 'Unexpected or missing artifacts'):
            verify(self.root, self.summary)

    def test_standalone_stdlib_cli_accepts_saved_list(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), str(self.root),
            '--summary', str(self.summary)], cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)['passed'])


if __name__ == '__main__':
    unittest.main()
