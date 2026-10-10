"""Synthetic preflight checks only; never package the unsealed real run."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.release_v6 import RESULTS, build, collect, digest, regular_file, validate_r7, validate_seal


class PackageV6PreflightTests(unittest.TestCase):
    def test_historical_access_receipt_requires_its_frozen_bytes(self):
        name = "data/v2/test_access.json"
        payload = (Path(__file__).resolve().parents[1] / name).read_bytes()
        target = self.put(name, payload)
        self.assertEqual(regular_file(self.root, name), target)
        self.put(name, b"not the frozen receipt")
        with self.assertRaisesRegex(ValueError, "historical metadata changed"):
            regular_file(self.root, name)

    def test_actual_r7_freeze_marker_refuses_unfinished_run(self):
        self.put(RESULTS + "/R7/freeze.json", b'{"source":"frozen"}')
        with self.assertRaisesRegex(ValueError, "started but has not completed"):
            validate_r7(self.root, None, "unused")

    def setUp(self):
        base = Path(__file__).resolve().parents[1] / ".cache" / "v6-packaging-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        subprocess.run(["git", "-C", str(self.root), "init", "-q"], capture_output=True, check=True)
        (self.root / ".gitignore").write_text("results/\nvendor/\n", encoding="ascii")
        (self.root / "README.md").write_bytes(b"source\n")
        subprocess.run(["git", "-C", str(self.root), "add", "."], capture_output=True, check=True)
        self.tracked = {"README.md": {}, ".gitignore": {}}

    def put(self, name, payload=b"{}"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def seal_fixture(self):
        code = "a" * 40
        self.put(RESULTS + "/run.log", b"runner closed\n")
        experiment = {"provenance": {"code_commit": code,
            "source_sha256": {"README.md": digest(self.root / "README.md")}, "data_sha256": {}},
            "new_test_opened": False, "gates": {"calibration_fidelity": False}}
        verification = {key: True for key in ("passed", "selected_trajectory_exactly_reproduced",
            "calibration_and_development_reproduced", "all_training_and_failure_logs_hash_verified")}
        verification["new_test_opened"] = False
        manifest = {"schema_version": 6, "code_commit": code,
                    "files": {"run.log": digest(self.root / RESULTS / "run.log")}}
        for name, value in (("experiment.json", experiment), ("verification.json", verification),
                            ("manifest.json", manifest)):
            self.put(RESULTS + "/" + name, json.dumps(value).encode("utf-8"))
        return code, digest(self.root / RESULTS / "manifest.json")

    def r7_fixture(self):
        prior_code, prior_manifest = self.seal_fixture()
        configuration = self.put("config/v6/r7.json", b'{"margins_s":[0.2,0.5,1.0]}')
        config_sha = digest(configuration)
        for args in (("config", "user.name", "Packaging fixture"),
                     ("config", "user.email", "packaging@local.invalid"),
                     ("add", "config/v6/r7.json"), ("commit", "-qm", "fixture source")):
            subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, check=True)
        head = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"],
                              capture_output=True, check=True).stdout.decode().strip()
        checkpoint = self.put(RESULTS + "/R4/2027/raw_bc_policy.npz", b"frozen checkpoint")
        metadata = self.put(RESULTS + "/R4/2027/bc_metadata.json", b'{"seed":2027}')
        inventory = {"checkpoint_relative_to_prior_run": "R4/2027/raw_bc_policy.npz",
            "checkpoint_sha256": digest(checkpoint), "metadata_relative_to_prior_run": "R4/2027/bc_metadata.json",
            "metadata_sha256": digest(metadata), "prior_manifest_sha256": prior_manifest,
            "prior_code_commit": prior_code, "copied_checkpoint_relative": "inputs/raw_bc_policy.npz"}
        experiment = {"schema_version": 7, "kind": "independent_geometry_guard_r7", "source_commit": head,
            "provenance": {"code_commit": head, "source_sha256": {"config/v6/r7.json": config_sha},
                           "data_sha256": {}, "config_sha256": config_sha},
            "checkpoint_inventory": inventory, "gates": {"cold_strong_proxy_six_laps": False},
            "config_path": "config/v6/r7.json", "config_sha256": config_sha,
            "frozen_protocol_path": "frozen_protocol.json", "FP2_accessed": False}
        verification = {"passed": True, "source_commit": head, "manifest_inputs_checked": 1,
                        "checkpoint_hash_passed": True, "source_hashes_passed": True,
                        "dependencies_match": True, "FP2_accessed": False}
        verification.update({key: True for key in ("exact_selected_trace", "exact_selected_terminal",
            "exact_selected_proxy", "exact_selected_calibration", "exact_selected_development", "exact_selected_actions")})
        self.put(RESULTS + "/R7/run.log", b"owned runner closed\n")
        self.put(RESULTS + "/R7/frozen_protocol.json", configuration.read_bytes())
        self.put(RESULTS + "/R7/inputs/raw_bc_policy.npz", checkpoint.read_bytes())
        manifest = {"schema_version": 7, "source_commit": head, "config_sha256": config_sha,
                    "checkpoint_inventory": inventory,
                    "files": {name: digest(self.root / RESULTS / "R7" / name)
                              for name in ("run.log", "frozen_protocol.json", "inputs/raw_bc_policy.npz")}}
        for name, value in (("experiment.json", experiment), ("verification.json", verification),
                            ("manifest.json", manifest)):
            self.put(RESULTS + "/R7/" + name, json.dumps(value).encode())
        return head, digest(self.root / RESULTS / "R7/manifest.json")

    def test_unsealed_call_refuses_before_creating_outputs(self):
        output = Path(self.temp.name) / "delivery"
        with self.assertRaisesRegex(ValueError, "runner exit"):
            build(self.root, output, expected_head="unused", code_commit="unused", tag="unused",
                  manifest_sha256="0" * 64)
        self.assertFalse(output.exists())

    def test_tools_sibling_restore_imports_without_work_layout(self):
        work = Path(__file__).resolve().parents[1] / "tools"
        tools = Path(self.temp.name) / "tools"
        tools.mkdir()
        (tools / "restore_v6.py").write_bytes((work / "restore_v6.py").read_bytes())
        for source, target in (("release_v6.py", "release_v6.py"),
                               ("verify_v6_bundle_recovery.py", "verify_v6_bundle_recovery.py")):
            script = tools / target
            script.write_bytes((work / source).read_bytes())
            process = subprocess.run([sys.executable, str(script), "--help"], capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr.decode(errors="replace"))
        process = subprocess.run([sys.executable, "-c",
            "import tools.release_v6, tools.verify_v6_bundle_recovery"], cwd=tools.parent, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr.decode(errors="replace"))
        self.assertFalse((self.root / RESULTS).exists())

    def test_collect_only_tracked_v6_results_and_direct_wheels(self):
        self.put(RESULTS + "/R1/result.json")
        self.put(RESULTS + "/__pycache__/ignored.pyc")
        self.put("vendor/wheels/runtime.whl")
        self.put("vendor/reporting-wheels/report.whl")
        self.put("results/austria_2026_v5/old.json")
        self.put(".venv/excluded.txt")
        self.assertEqual(collect(self.root, self.tracked), sorted([
            ".gitignore", "README.md", RESULTS + "/R1/result.json",
            "vendor/wheels/runtime.whl", "vendor/reporting-wheels/report.whl"]))

    def test_unknown_wheel_and_fp2_arrays_refused(self):
        for name in ("vendor/wheels/unknown.txt", "vendor/wheels/nested/runtime.whl",
                     RESULTS + "/test_arrays.npz", RESULTS + "/fp2/raw.json"):
            with self.subTest(name=name):
                target = self.put(name)
                with self.assertRaises(ValueError):
                    collect(self.root, self.tracked)
                target.unlink()

    def test_reparse_member_refused(self):
        target = self.put("vendor/wheels/link.whl")
        original = os.lstat
        def fake(path, *args, **kwargs):
            if Path(path) == target:
                return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        with patch(digest.__module__ + ".os.lstat", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "reparse"):
                collect(self.root, self.tracked)

    def test_failed_scientific_gate_can_be_packaged_as_evidence_after_integrity_passes(self):
        code, checksum = self.seal_fixture()
        experiment, _, _ = validate_seal(self.root, checksum, code)
        self.assertIs(experiment["gates"]["calibration_fidelity"], False)

    def test_changed_sealed_calculation_or_source_refused(self):
        code, checksum = self.seal_fixture()
        self.put(RESULTS + "/run.log", b"changed after seal")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            validate_seal(self.root, checksum, code)
        code, checksum = self.seal_fixture()
        self.put("README.md", b"changed source")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            validate_seal(self.root, checksum, code)

    def test_expected_external_seal_and_code_identity_required(self):
        code, checksum = self.seal_fixture()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_seal(self.root, "0" * 64, code)
        with self.assertRaisesRegex(ValueError, "commit differs"):
            validate_seal(self.root, checksum, "b" * 40)

    def test_manifest_traversal_and_unverified_result_refused(self):
        code, _ = self.seal_fixture()
        manifest = self.root / RESULTS / "manifest.json"
        self.put(RESULTS + "/manifest.json", json.dumps({"schema_version": 6, "code_commit": code,
            "files": {"../outside": "0" * 64}}).encode())
        with self.assertRaises(ValueError):
            validate_seal(self.root, digest(manifest), code)
        code, checksum = self.seal_fixture()
        verification = json.loads((self.root / RESULTS / "verification.json").read_text())
        verification["passed"] = False
        self.put(RESULTS + "/verification.json", json.dumps(verification).encode())
        with self.assertRaisesRegex(ValueError, "verification has not passed"):
            validate_seal(self.root, checksum, code)

    def test_unfinished_r7_or_missing_independent_seal_refused(self):
        self.put(RESULTS + "/R7/started.json")
        with self.assertRaisesRegex(ValueError, "not completed"):
            validate_r7(self.root, None, "unused")
        self.put(RESULTS + "/R7/experiment.json")
        with self.assertRaisesRegex(ValueError, "SHA256 is required"):
            validate_r7(self.root, None, "unused")

    def test_independent_r7_verified_and_scientific_failures_preserved(self):
        head, checksum = self.r7_fixture()
        result = validate_r7(self.root, checksum, head)
        self.assertEqual(result[0]["source_commit"], head)
        self.assertEqual(result[0]["manifest_sha256"], checksum)
        self.assertIs(result[0]["recorded_gates"]["cold_strong_proxy_six_laps"], False)

    def test_changed_r7_checkpoint_or_config_refused(self):
        head, checksum = self.r7_fixture()
        self.put(RESULTS + "/R4/2027/raw_bc_policy.npz", b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint input changed"):
            validate_r7(self.root, checksum, head)
        self.put(RESULTS + "/R4/2027/raw_bc_policy.npz", b"frozen checkpoint")
        self.put("config/v6/r7.json", b"changed configuration")
        with self.assertRaisesRegex(ValueError, "frozen source changed"):
            validate_r7(self.root, checksum, head)


if __name__ == "__main__":
    unittest.main()
