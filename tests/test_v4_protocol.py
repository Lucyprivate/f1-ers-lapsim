from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from f1ers_v4.artifacts import (read_json, write_json, sha256, git_state, capture_provenance,
                              freeze_configuration, check_freeze, restore_baseline_artifacts)
from f1ers_v4.protocol import (DataUnavailable, ensure_protocol_lock,
    authorize_test_access, verify_test_authorization, mark_test_arrays_unsealed,
    complete_test_access, read_saved_test_score, require_local_freeze,
    claim_coordinate_registration, complete_coordinate_registration,
    claim_condition_adaptation, complete_condition_adaptation, ledger_paths)
from f1ers_v4.data import load_local_dataset, load_new_session, prepare_new_session

ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


class V4ProtocolTests(unittest.TestCase):
    def repo(self, root):
        root.mkdir(parents=True, exist_ok=True)
        git(root, "init", "-q")
        git(root, "config", "user.name", "F1 ERS Lab")
        git(root, "config", "user.email", "f1ers@local.invalid")
        git(root, "config", "core.autocrlf", "false")
        (root / ".gitignore").write_text("results/\nvendor/wheels/\ndata/v4/fp2/\n", encoding="utf-8")
        (root / ".gitattributes").write_text("* -text\n", encoding="utf-8")
        (root / "source.py").write_bytes(b"# frozen\r\n")
        git(root, "add", ".")
        git(root, "commit", "-qm", "baseline")
        return root

    def protocol_fixture(self, root, ledger):
        self.repo(root)
        p = read_json(ROOT / "config/protocol_austria_2026_v4.json")
        p["test_access"]["ledger_root"] = str(ledger)
        write_json(root / "config/protocol_austria_2026_v4.json", p)
        names = ["config/rules_austria_2026.json", "data/processed/dataset.json", "data/processed/track.npz",
                 "data/raw/tum/Spielberg_minimum_curvature.csv", "data/raw/tum/Spielberg_centerline.csv",
                 "data/v3/local/registration.json"]
        names += [f"data/raw/openf1/{name}.json" for name in
                  ("sessions", "drivers", "laps", "race_control", "weather", "pit", "stints",
                   "car_data_63_laps3to9", "location_63_laps3to9")]
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"[]")
        (root / "requirements-lock.txt").write_text("numpy\n", encoding="utf-8")
        ensure_protocol_lock(root)
        git(root, "add", ".")
        git(root, "commit", "-qm", "protocol")
        gates = {name: True for name in ("data", "convergence", "physical", "grid", "development")}
        local_path, local = freeze_configuration(root, root / "results/test", {"mass_kg": 800},
            local_gates=gates)
        claim_coordinate_registration(root, local_path)
        registration_path = root / "data/v4/fp2/registration.json"
        write_json(registration_path, {"fit_laps": [3, 4, 5], "fit_role": "adaptation"})
        complete_coordinate_registration(root, registration_path)
        claim_condition_adaptation(root, local_path, "early")
        adaptation = {"solver_success": True, "physical_valid": True}
        adaptation_path = root / "results/test/condition_adaptation.json"
        write_json(adaptation_path, adaptation)
        complete_condition_adaptation(root, adaptation_path)
        freeze_path, freeze = freeze_configuration(root, root / "results/test", {"mass_kg": 800},
            stage="cross_session", adaptation_complete=True, local_gates=gates,
            adaptation=adaptation, parent_local_freeze=local)
        return freeze_path, freeze, gates

    def archive(self, path, members):
        with zipfile.ZipFile(path, "w") as z:
            for name, value in members:
                info = zipfile.ZipInfo(name)
                # Windows normally canonicalizes backslashes in ZipInfo;
                # preserve the attacker-supplied raw name for the test.
                info.filename = name
                z.writestr(info, value)
        return sha256(path)

    def test_formal_git_provenance_rejects_dirty_and_preserves_crlf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.repo(Path(temp))
            clean = git_state(root)
            self.assertFalse(clean["dirty"])
            self.assertEqual(len(clean["code_commit"]), 40)
            self.assertEqual((root / "source.py").read_bytes(), b"# frozen\r\n")
            (root / "source.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean Git"):
                git_state(root)
            self.assertTrue(git_state(root, require_clean=False)["dirty"])

    def test_restore_allowlist_idempotence_and_no_source_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.repo(Path(temp) / "repo")
            archive = Path(temp) / "bundle.zip"
            digest = self.archive(archive, [("f1-ers-lab-v3/results/a/a.json", b"{}\n"),
                ("f1-ers-lab-v3/vendor/wheels/a.whl", b"wheel"),
                ("f1-ers-lab-v3/source.py", b"unsafe replacement")])
            result = restore_baseline_artifacts(root, archive, expected_sha256=digest)
            self.assertEqual(len(result["restored"]), 2)
            self.assertEqual((root / "source.py").read_bytes(), b"# frozen\r\n")
            again = restore_baseline_artifacts(root, archive, expected_sha256=digest)
            self.assertEqual(again["restored"], [])
            self.assertEqual(len(again["already_identical"]), 2)
            self.assertFalse(git_state(root)["dirty"])

    def test_restore_bad_sha_traversal_and_duplicates_write_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.repo(Path(temp) / "repo")
            archive = Path(temp) / "bundle.zip"
            good = ("f1-ers-lab-v3/results/a.json", b"good")
            bad_members = ["f1-ers-lab-v3/results/../evil", "/absolute",
                           "f1-ers-lab-v3/results/C:/evil", "f1-ers-lab-v3/results\\evil"]
            for bad in bad_members:
                digest = self.archive(archive, [good, (bad, b"bad")])
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    restore_baseline_artifacts(root, archive, expected_sha256=digest)
                self.assertFalse((root / "results/a.json").exists())
            digest = self.archive(archive, [good, ("f1-ers-lab-v3/results/A.json", b"duplicate")])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                restore_baseline_artifacts(root, archive, expected_sha256=digest)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                restore_baseline_artifacts(root, archive)

    def test_restore_prevalidates_all_existing_destinations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.repo(Path(temp) / "repo")
            (root / "results").mkdir()
            (root / "results/b.json").write_bytes(b"original")
            archive = Path(temp) / "bundle.zip"
            digest = self.archive(archive, [("f1-ers-lab-v3/results/a.json", b"new"),
                                           ("f1-ers-lab-v3/results/b.json", b"different")])
            with self.assertRaisesRegex(ValueError, "existing artifact"):
                restore_baseline_artifacts(root, archive, expected_sha256=digest)
            self.assertFalse((root / "results/a.json").exists())
            self.assertEqual((root / "results/b.json").read_bytes(), b"original")

    def test_durable_claim_survives_independent_clone_and_interruption(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            ledger = parent / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = parent / "repo"
                freeze_path, _, _ = self.protocol_fixture(root, ledger)
                claim = authorize_test_access(root, freeze_path, "first")
                git(root, "checkout", "-qb", "another-branch")
                with self.assertRaisesRegex(DataUnavailable, "already claimed"):
                    authorize_test_access(root, freeze_path, "branch-repeat")
                clone = parent / "clone"
                subprocess.run(["git", "clone", "--local", "--no-hardlinks", "-q", str(root), str(clone)], check=True,
                               capture_output=True)
                clone_freeze = clone / freeze_path.relative_to(root)
                clone_freeze.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(freeze_path, clone_freeze)
                shutil.copyfile(root / "results/test/frozen_configuration.json",
                                clone / "results/test/frozen_configuration.json")
                shutil.copyfile(root / "results/test/condition_adaptation.json",
                                clone / "results/test/condition_adaptation.json")
                (clone / "data/v4/fp2").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / "data/v4/fp2/registration.json",
                                clone / "data/v4/fp2/registration.json")
                with self.assertRaisesRegex(DataUnavailable, "already claimed"):
                    authorize_test_access(clone, clone_freeze, "clone-repeat")
                self.assertEqual(verify_test_authorization(clone)["configuration_sha256"], claim["configuration_sha256"])
                mark_test_arrays_unsealed(root)
                with self.assertRaisesRegex(DataUnavailable, "already unsealed"):
                    mark_test_arrays_unsealed(clone)

    def test_saved_score_read_does_not_reopen_arrays_or_allow_rescore(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = Path(temp) / "repo"
                freeze_path, _, _ = self.protocol_fixture(root, ledger)
                authorize_test_access(root, freeze_path, "once")
                mark_test_arrays_unsealed(root)
                result_path = root / "results/test/test_evaluation.json"
                write_json(result_path, {"evaluation": {"passed": False}, "test_scored": True})
                result_path.with_name("test_trajectories.json.gz").write_bytes(b"retained final trajectories")
                complete_test_access(root, result_path)
                with patch("f1ers_v4.data.raw_arrays", side_effect=AssertionError("must not reopen")):
                    stored = read_saved_test_score(root)
                self.assertFalse(stored["score"]["evaluation"]["passed"])
                with self.assertRaisesRegex(DataUnavailable, "already completed"):
                    verify_test_authorization(root)

    def test_early_pass_claims_cannot_repeat_across_clones_even_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = Path(temp) / "repo"
                self.protocol_fixture(root, ledger)
                local_path = root / "results/test/frozen_configuration.json"
                with self.assertRaisesRegex(DataUnavailable, "already claimed"):
                    claim_coordinate_registration(root, local_path)
                with self.assertRaisesRegex(DataUnavailable, "already claimed"):
                    claim_condition_adaptation(root, local_path, "again")
                clone = Path(temp) / "clone"
                subprocess.run(["git", "clone", "--local", "--no-hardlinks", "-q", str(root), str(clone)],
                               capture_output=True, check=True)
                shutil.copytree(root / "results", clone / "results")
                shutil.copytree(root / "data/v4/fp2", clone / "data/v4/fp2")
                with self.assertRaisesRegex(DataUnavailable, "already claimed"):
                    claim_condition_adaptation(clone, clone / "results/test/frozen_configuration.json", "clone")

    def test_saved_score_receipt_detects_final_trajectory_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = Path(temp) / "repo"
                freeze_path, _, _ = self.protocol_fixture(root, ledger)
                authorize_test_access(root, freeze_path, "once")
                mark_test_arrays_unsealed(root)
                result_path = root / "results/test/test_evaluation.json"
                write_json(result_path, {"test_scored": True})
                trajectory_path = result_path.with_name("test_trajectories.json.gz")
                trajectory_path.write_bytes(b"original trajectories")
                complete_test_access(root, result_path)
                trajectory_path.write_bytes(b"changed trajectories")
                with self.assertRaisesRegex(DataUnavailable, "artifact changed"):
                    read_saved_test_score(root)

    def test_exported_receipt_verifies_offline_without_creating_a_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = Path(temp) / "repo"
                freeze_path, _, _ = self.protocol_fixture(root, ledger)
                authorize_test_access(root, freeze_path, "once")
                mark_test_arrays_unsealed(root)
                result_path = root / "results/test/test_evaluation.json"
                write_json(result_path, {"test_scored": True})
                result_path.with_name("test_trajectories.json.gz").write_bytes(b"original trajectories")
                complete_test_access(root, result_path)
                shutil.rmtree(ledger)
                with patch("f1ers_v4.data.raw_arrays", side_effect=AssertionError("offline arrays prohibited")):
                    value = read_saved_test_score(root, result_path=result_path)
                self.assertTrue(value["score"]["test_scored"])
                self.assertFalse(ledger.exists())
                receipt = result_path.with_name("test_access_receipt.json")
                mutated = read_json(receipt)
                mutated["score"]["test_scored"] = False
                write_json(receipt, mutated)
                with self.assertRaisesRegex(DataUnavailable, "receipt integrity"):
                    read_saved_test_score(root, result_path=result_path)

    def test_external_ledger_is_authoritative_over_exported_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = Path(temp) / "repo"
                freeze_path, _, _ = self.protocol_fixture(root, ledger)
                authorize_test_access(root, freeze_path, "once")
                mark_test_arrays_unsealed(root)
                result_path = root / "results/test/test_evaluation.json"
                write_json(result_path, {"test_scored": True})
                result_path.with_name("test_trajectories.json.gz").write_bytes(b"original trajectories")
                complete_test_access(root, result_path)
                receipt = result_path.with_name("test_access_receipt.json")
                mutated = read_json(receipt)
                mutated["result_sha256"] = "0" * 64
                write_json(receipt, mutated)
                with self.assertRaisesRegex(DataUnavailable, "authoritative external ledger"):
                    read_saved_test_score(root, result_path=result_path)

    def test_freeze_is_detached_and_checks_code_and_config_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = Path(temp) / "repo"
                _, _, gates = self.protocol_fixture(root, ledger)
                provenance = capture_provenance(root)
                path, freeze = freeze_configuration(root, root / "results/local", {},
                    code_provenance=provenance, local_gates=gates)
                self.assertTrue(check_freeze(root, freeze))
                self.assertEqual(require_local_freeze(root, path, gates)["stage"], "local")
                gates["new_test"] = False
                self.assertNotIn("new_test", freeze["local_gates"])
                p = root / "config/rules_austria_2026.json"
                p.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                    check_freeze(root, freeze)

    def test_local_loader_no_mutation_and_only_calibration_development_roles(self):
        inherited = [p for p in (ROOT / "data/v3").rglob("*") if p.is_file()]
        before = {str(p): sha256(p) for p in inherited}
        with patch("f1ers_v3.data.write_json", side_effect=AssertionError("read-only")):
            data = load_local_dataset(ROOT)
        self.assertEqual([l["lap_number"] for l in data["laps"]], [3, 4, 5, 6, 7, 9])
        self.assertFalse(data["audit"]["test_arrays_loaded"])
        self.assertEqual(before, {str(p): sha256(p) for p in inherited})

    def test_early_loader_never_reads_broad_late_performance_files(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger"
            with patch("f1ers_v4.protocol.CANONICAL_LEDGER_ROOT", ledger):
                root = Path(temp) / "repo"
                self.protocol_fixture(root, ledger)
                base = root / "data/v4/fp2"
                early = base / "raw/adaptation"
                lap = {"driver_number": 63, "lap_number": 3, "lap_duration": 70,
                       "date_start": "2026-06-26T10:00:00+00:00", "duration_sector_1": 20,
                       "duration_sector_2": 30, "duration_sector_3": 20}
                for name, value in (("laps.json", [lap]), ("car_data.json", []), ("location.json", [])):
                    write_json(early / name, value)
                write_json(base / "selection.json", {"adaptation_laps": [3], "test_laps": [6, 7, 8]})
                write_json(base / "registration.json", {"fit_role": "adaptation", "fit_laps": [3]})
                write_json(base / "status.json", {"status": "ready", "session": {},
                    "raw_sources": [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)}
                                    for p in early.glob("*.json")]})
                calls = []
                real_read = read_json
                def checked_read(path):
                    calls.append(str(path))
                    if "car_data_63_medium" in str(path) or "location_63_medium" in str(path):
                        raise AssertionError("late raw records opened")
                    return real_read(path)
                processed = {"role": "adaptation", "alignment_audit": {"grid_bracketed": True,
                             "projected_span_fraction": 1}, "scoring_signal_valid_mask": [True]}
                with patch("f1ers_v4.data.read_json", side_effect=checked_read), \
                     patch("f1ers_v4.data.raw_arrays", return_value={}), \
                     patch("f1ers_v4.data.build_track", return_value={}), \
                     patch("f1ers_v4.data.remap_rules", return_value={}), \
                     patch("f1ers_v4.data.inherited.process_lap", return_value=processed):
                    data = load_new_session(root, include_test=False)
                self.assertTrue(data["audit"]["early_loader_materialized_records_only"])
                self.assertFalse(any("car_data_63_medium" in c for c in calls))

    def test_failed_local_gates_prevent_any_remote_attempt(self):
        with patch("f1ers_v4.data.require_local_freeze", side_effect=DataUnavailable("local failure")), \
             patch("f1ers_v4.data._bounded_fetch", side_effect=AssertionError("must not fetch")):
            with self.assertRaisesRegex(DataUnavailable, "local failure"):
                prepare_new_session(ROOT, ROOT / "missing.json", {"development": False})


if __name__ == "__main__":
    unittest.main()
