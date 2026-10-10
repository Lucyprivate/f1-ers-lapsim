"""Synthetic byte, inventory, and receipt regressions; no project execution."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_research_artifacts.py"
SPEC = importlib.util.spec_from_file_location("research_artifact_verifier", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class ArtifactVerificationTests(unittest.TestCase):
    def setUp(self):
        # Keep all synthetic writes inside the checkout on restricted Windows hosts.
        self.temp = tempfile.TemporaryDirectory(prefix="verifier_fixture_", dir=SCRIPT.parent.parent)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "artifacts"
        self.root.mkdir()
        (self.root / "traces").mkdir()
        self.freeze = {"identity": "synthetic_sealed_research", "code_commit": "1" * 40,
                       "config": {"identity": "synthetic_sealed_research"},
                       "data_sha256": {"unavailable/private/telemetry.json": "7" * 64}}
        write_json(self.root / "freeze.json", self.freeze)
        (self.root / "traces/lap.json.gz").write_bytes(b"opaque bytes; do not deserialize or execute")
        self.manifest = {"schema_version": 1, "identity": self.freeze["identity"], "files": {
            name: digest(self.root / name) for name in ("freeze.json", "traces/lap.json.gz")}}
        write_json(self.root / "manifest.json", self.manifest)
        self.receipt = {"identity": self.freeze["identity"], "source_code_commit": "1" * 40,
                        "source_version": "0.8.0", "result_directory_name": "sealed_case",
                        "manifest_sha256": digest(self.root / "manifest.json"),
                        "freeze_sha256": digest(self.root / "freeze.json"),
                        "episode_budget_counts": {"attempted": 0, "maximum": 120},
                        "FP2_accessed": False}
        self.completion = self.base / "completion.json"
        write_json(self.completion, self.receipt)
        self.evidence = {"identity": self.freeze["identity"], "source_code_commit": "1" * 40,
                         "source_version": "0.8.0", "result_directory_name": "sealed_case",
                         "result_manifest_sha256": digest(self.root / "manifest.json"),
                         "freeze_sha256": digest(self.root / "freeze.json"),
                         "completion_sha256": digest(self.completion),
                         "completion_payload": self.receipt}
        self.summary = self.base / "summary.json"
        write_json(self.summary, self.evidence)

    def verify(self, completion=False):
        return VERIFIER.verify(self.root, self.summary, self.completion if completion else None)

    def repin_manifest(self):
        write_json(self.root / "manifest.json", self.manifest)
        self.evidence["result_manifest_sha256"] = digest(self.root / "manifest.json")
        write_json(self.summary, self.evidence)

    def assert_rejected(self, text, completion=False):
        with self.assertRaisesRegex(VERIFIER.VerificationError, text):
            self.verify(completion)

    def test_valid_byte_verification_does_not_modify_or_load_telemetry(self):
        before = {p.relative_to(self.root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        result = self.verify(True)
        self.assertTrue(result["passed"])
        self.assertTrue(result["completion_checked"])
        self.assertEqual(result["declared_files_checked"], 2)
        self.assertEqual(result["training_attempts_consumed"], 0)
        self.assertFalse(result["telemetry_loaded"])
        self.assertFalse(result["simulation_executed"])
        after = {p.relative_to(self.root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_cli_works_with_only_stdlib_and_does_not_import_project(self):
        result = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--artifacts", str(self.root),
                                 "--summary", str(self.summary), "--completion", str(self.completion)],
                                cwd=self.base, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["passed"])

    def test_missing_declared_artifact_is_rejected(self):
        (self.root / "traces/lap.json.gz").unlink()
        self.assert_rejected("inventory mismatch.*missing")

    def test_content_tampering_is_rejected(self):
        (self.root / "traces/lap.json.gz").write_bytes(b"changed")
        self.assert_rejected("SHA256 mismatch: traces/lap")

    def test_manifest_byte_tampering_is_rejected_even_when_semantically_identical(self):
        path = self.root / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.assert_rejected("Manifest SHA256")

    def test_unexpected_file_is_rejected_even_when_every_declared_hash_matches(self):
        (self.root / "undeclared_optimizer_restart.json").write_text("{}", encoding="utf-8")
        self.assert_rejected("unexpected=.*undeclared_optimizer_restart")

    def test_manifest_identity_tamper_cannot_be_hidden_by_new_manifest_pin(self):
        self.manifest["identity"] = "another_experiment"
        self.repin_manifest()
        self.assert_rejected("Manifest identity")

    def test_freeze_commit_tamper_cannot_be_hidden_by_new_file_hashes(self):
        self.freeze["code_commit"] = "2" * 40
        write_json(self.root / "freeze.json", self.freeze)
        self.manifest["files"]["freeze.json"] = digest(self.root / "freeze.json")
        self.evidence["freeze_sha256"] = digest(self.root / "freeze.json")
        self.repin_manifest()
        self.assert_rejected("Freeze code_commit")

    def test_freeze_pin_and_manifest_must_agree(self):
        self.evidence["freeze_sha256"] = "3" * 64
        write_json(self.summary, self.evidence)
        self.assert_rejected("Manifest freeze SHA256")

    def test_parent_absolute_windows_ads_and_alias_paths_are_rejected(self):
        unsafe = ("../outside.txt", "/outside.txt", "C:/outside.txt", "traces\\lap.json.gz",
                  "traces/lap.json.gz:stream", "traces//lap.json.gz", "./freeze.json",
                  "traces/../freeze.json", "traces/NUL.txt", "traces/trailing. ", "traces/a\x00b")
        original = dict(self.manifest["files"])
        for path in unsafe:
            with self.subTest(path=path):
                self.manifest["files"] = {**original, path: "4" * 64}
                self.repin_manifest()
                self.assert_rejected("artifact path")

    def test_case_aliases_are_rejected_for_cross_platform_inventory(self):
        self.manifest["files"]["FREEZE.json"] = digest(self.root / "freeze.json")
        self.repin_manifest()
        self.assert_rejected("Case-aliased manifest")

    def test_manifest_cannot_declare_its_own_unverifiable_hash(self):
        self.manifest["files"]["manifest.json"] = "5" * 64
        self.repin_manifest()
        self.assert_rejected("must not declare itself")

    def test_bad_sha_pin_is_rejected(self):
        self.manifest["files"]["traces/lap.json.gz"] = "not-a-hash"
        self.repin_manifest()
        self.assert_rejected("Invalid lowercase SHA256")

    def test_duplicate_json_keys_are_not_silently_overwritten(self):
        path = self.root / "manifest.json"
        path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        self.evidence["result_manifest_sha256"] = digest(path)
        write_json(self.summary, self.evidence)
        self.assert_rejected("Duplicate JSON key")

    def test_nonfinite_json_is_rejected(self):
        self.summary.write_text('{"identity":NaN}', encoding="utf-8")
        self.assert_rejected("Non-finite JSON")

    def test_linked_declared_file_is_rejected_even_if_target_bytes_match(self):
        target = self.base / "outside_trace"
        target.write_bytes((self.root / "traces/lap.json.gz").read_bytes())
        path = self.root / "traces/lap.json.gz"
        path.unlink()
        try:
            os.symlink(target, path)
        except OSError as exc:
            self.skipTest(f"OS does not permit synthetic symlink creation: {exc}")
        self.assert_rejected("Linked artifact")

    def test_linked_directory_is_rejected_even_if_target_stays_inside_root(self):
        try:
            os.symlink(self.root / "traces", self.root / "alias", target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"OS does not permit synthetic symlink creation: {exc}")
        self.assert_rejected("Linked artifact")

    def test_completion_byte_tampering_is_rejected(self):
        self.completion.write_bytes(self.completion.read_bytes() + b"\n")
        self.assert_rejected("Completion SHA256", True)

    def test_completion_full_payload_is_required_not_only_its_identity(self):
        self.evidence["completion_payload"] = {**self.receipt, "FP2_accessed": True}
        write_json(self.summary, self.evidence)
        self.assert_rejected("Completion payload", True)

    def test_completion_json_types_must_match_even_when_python_values_compare_equal(self):
        self.evidence["completion_payload"] = {**self.receipt, "FP2_accessed": 0}
        write_json(self.summary, self.evidence)
        self.assert_rejected("Completion payload", True)

    def test_completion_cross_reference_tamper_is_rejected_after_repin(self):
        changed = {**self.receipt, "source_code_commit": "2" * 40}
        write_json(self.completion, changed)
        self.evidence["completion_payload"] = changed
        self.evidence["completion_sha256"] = digest(self.completion)
        write_json(self.summary, self.evidence)
        self.assert_rejected("Completion source_code_commit", True)

    def test_optional_completion_not_requested_does_not_require_receipt(self):
        self.completion.unlink()
        self.evidence.pop("completion_payload")
        self.evidence.pop("completion_sha256")
        write_json(self.summary, self.evidence)
        self.assertFalse(self.verify()["completion_checked"])

    def test_cli_failure_is_nonzero_and_keeps_artifact_bytes(self):
        path = self.root / "traces/lap.json.gz"
        path.write_bytes(b"tampered")
        before = path.read_bytes()
        result = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--artifacts", str(self.root),
                                 "--summary", str(self.summary)], cwd=self.base,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stderr)["passed"])
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
