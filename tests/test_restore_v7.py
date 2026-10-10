"""Meaningful recovery checks; all Git fixtures and temp files stay in work/."""
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
import warnings
import zipfile

try:
    from restore_v7 import ZIP_PREFIX, LEGACY_STATUS, restore
except ImportError:  # Allows later placement in repository tests/.
    from tools.restore_v7 import ZIP_PREFIX, LEGACY_STATUS, restore

RESTORE_MODULE = restore.__module__


class V7RestoreTests(unittest.TestCase):
    def setUp(self):
        # Standalone location now; if copied to tests/, use repository .cache.
        base = Path(__file__).resolve().parent
        if base.name == "tests":
            base = base.parent / ".cache" / "v7-restore-tests"
        else:
            base = base / "test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.command("init", "-q")
        self.command("config", "core.autocrlf", "false")
        self.command("config", "user.name", "F1 restoration fixture")
        self.command("config", "user.email", "restore@local.invalid")
        self.source = b"source remains unchanged\n"
        (self.root / "README.md").write_bytes(self.source)
        (self.root / ".gitignore").write_bytes(b"results/\nvendor/wheels/\nvendor/reporting-wheels/\n")
        self.command("add", ".")
        self.command("commit", "-qm", "fixture")

    def command(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                              check=True).stdout.decode().strip()

    def archive(self, entries, *, linked=None, duplicate=None, nonregular=None, prefix=True):
        path = Path(self.temp.name) / "delivery.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for name, payload in entries.items():
                full_name = ZIP_PREFIX + name if prefix else name
                member = zipfile.ZipInfo(full_name)
                member.filename = full_name
                member.orig_filename = full_name
                if name == linked:
                    member.create_system = 3
                    member.external_attr = (stat.S_IFLNK | 0o777) << 16
                if name == nonregular:
                    member.create_system = 3
                    member.external_attr = (stat.S_IFIFO | 0o600) << 16
                bundle.writestr(member, payload)
            if duplicate is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    bundle.writestr(ZIP_PREFIX + duplicate, b"duplicate")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def assert_no_artifact_writes(self):
        self.assertFalse((self.root / "vendor").exists())
        self.assertFalse((self.root / "results").exists())

    def test_valid_restore_manifest_and_idempotency_preserve_source(self):
        entries = {"README.md": self.source, "vendor/wheels/a.whl": b"wheel",
                   "vendor/reporting-wheels/report.whl": b"reporting wheel",
                   "results/austria_2026_v7/frozen_configuration.json": b"{}",
                   "results/austria_2026_v7/R1/trajectory.json.gz": b"saved model output"}
        path, checksum = self.archive(entries)
        result = restore(self.root, path, checksum)
        self.assertEqual(len(result["restored"]), 4)
        self.assertEqual(result["verified_sources"], ["README.md"])
        self.assertFalse(result["source_overwritten"])
        self.assertFalse(result["late_test_arrays_restored"])
        records = {r["path"]: r for r in result["member_manifest"]}
        for name, payload in entries.items():
            self.assertEqual(records[name]["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(records[name]["bytes"], len(payload))
        self.assertEqual((self.root / "README.md").read_bytes(), self.source)
        second = restore(self.root, path, checksum)
        self.assertEqual(second["restored"], [])
        self.assertEqual(len(second["identical_artifacts"]), 4)
        json.dumps(result)

    def test_v7_both_results_and_exact_r7_dependencies_are_accepted(self):
        entries = {"results/austria_2026_v7/S0/trace.json.gz": b"cold diagnostic",
                   "results/austria_2026_v7B/B5/2026/checkpoint.npz": b"new checkpoint",
                   "results/austria_2026_v7C/C1/trace.json.gz": b"frozen third experiment",
                   "results/austria_2026_v7D/D4/checkpoint.npz": b"frozen fourth experiment",
                   "results/austria_2026_v7_delivery/report.html": b"standalone report",
                   "results/austria_2026_v6/R7/inputs/raw_bc_policy.npz": b"R7 copied input",
                   "results/austria_2026_v6/R4/2027/raw_bc_policy.npz": b"old exact dependency"}
        path, checksum = self.archive(entries)
        result = restore(self.root, path, checksum)
        self.assertEqual(result["schema_version"], 7)
        self.assertEqual(result["zip_prefix"], "f1-ers-lab-v7/")
        self.assertEqual(set(result["restored"]), set(entries))
        self.assertEqual((self.root/"README.md").read_bytes(), self.source)
        for name, payload in entries.items():
            self.assertEqual((self.root/name).read_bytes(), payload)

    def test_broad_v6_results_and_checkpoint_siblings_are_not_allowed(self):
        for bad in ("results/austria_2026_v6/manifest.json", "results/austria_2026_v6/R1/trajectory.json.gz",
                    "results/austria_2026_v6/R4/2027/bc_metadata.json",
                    "results/austria_2026_v6/R4/2028/raw_bc_policy.npz",
                    "results/austria_2026_v6/R4/2027/RAW_BC_POLICY.npz",
                    "results/austria_2026_v6/R4/2027/other/"):
            with self.subTest(bad=bad):
                path, checksum = self.archive({"results/austria_2026_v7/good.json": b"{}", bad: b""})
                with self.assertRaisesRegex(ValueError, "Unlisted"):
                    restore(self.root, path, checksum)
                self.assert_no_artifact_writes()

    def test_result_prefix_suffixes_and_prior_zip_version_are_rejected(self):
        for bad in ("results/austria_2026_v7_extra/a.json", "results/austria_2026_v7B_extra/a.json",
                    "results/austria_2026_v7C_extra/a.json",
                    "results/austria_2026_v7D_extra/a.json",
                    "results/austria_2026_v7_delivery_extra/report.html",
                    "results/austria_2026_v6/R7_extra/a.json"):
            with self.subTest(bad=bad):
                path, checksum = self.archive({"vendor/wheels/good.whl": b"ok", bad: b""})
                with self.assertRaisesRegex(ValueError, "Unlisted"):
                    restore(self.root, path, checksum)
                self.assert_no_artifact_writes()
        path, checksum = self.archive({"f1-ers-lab-v6/vendor/wheels/good.whl": b"ok"}, prefix=False)
        with self.assertRaisesRegex(ValueError, "prefix"):
            restore(self.root, path, checksum)
        self.assert_no_artifact_writes()

    def test_fp2_and_late_arrays_are_rejected_inside_every_allowed_result_prefix(self):
        for prefix in ("results/austria_2026_v7/", "results/austria_2026_v7B/", "results/austria_2026_v7C/", "results/austria_2026_v7D/", "results/austria_2026_v7_delivery/", "results/austria_2026_v6/R7/"):
            for tail in ("late_arrays.npz", "fp2/raw.json", "test_arrays.json.gz"):
                with self.subTest(prefix=prefix, tail=tail):
                    path, checksum = self.archive({"vendor/wheels/good.whl": b"ok", prefix+tail: b"late"})
                    with self.assertRaisesRegex(ValueError, "arrays are forbidden"):
                        restore(self.root, path, checksum)
                    self.assert_no_artifact_writes()

    def test_existing_checkpoint_dependency_never_overwritten(self):
        name = "results/austria_2026_v6/R4/2027/raw_bc_policy.npz"
        target = self.root/name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"original checkpoint")
        path, checksum = self.archive({"results/austria_2026_v7/good.json": b"{}", name: b"different checkpoint"})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            restore(self.root, path, checksum)
        self.assertEqual(target.read_bytes(), b"original checkpoint")
        self.assertFalse((self.root/"results/austria_2026_v7").exists())

    def test_complete_published_sha_required_before_any_write(self):
        path, _ = self.archive({"vendor/wheels/a.whl": b"wheel"})
        for bad in ("0" * 64, "short", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                restore(self.root, path, bad)
            self.assert_no_artifact_writes()

    def test_cli_requires_external_hash(self):
        path, _ = self.archive({"vendor/wheels/a.whl": b"wheel"})
        module = sys.modules[RESTORE_MODULE]
        process = subprocess.run([sys.executable, module.__file__, "--root", str(self.root),
                                  "--archive", str(path)], capture_output=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn(b"--sha256", process.stderr)
        self.assert_no_artifact_writes()

    def test_every_path_is_prevalidated(self):
        for bad in ("../outside", "/absolute", "C:/outside", ".", "a/../b", "a//b", "a\\b",
                    "vendor/wheels/CON.whl", "vendor/wheels/COM1.whl", "vendor/wheels/LPT¹.whl",
                    "vendor/wheels/a.", "vendor/wheels/a ", "vendor/wheels/a:b",
                    "vendor/wheels/a?", "vendor/wheels/REPORT~1.whl"):
            with self.subTest(bad=bad):
                path, checksum = self.archive({"vendor/wheels/good.whl": b"ok", bad: b"bad"})
                with self.assertRaises(ValueError):
                    restore(self.root, path, checksum)
                self.assert_no_artifact_writes()

    def test_required_prefix_is_not_optional(self):
        path, checksum = self.archive({"vendor/wheels/good.whl": b"ok"}, prefix=False)
        with self.assertRaisesRegex(ValueError, "prefix"):
            restore(self.root, path, checksum)
        self.assert_no_artifact_writes()

    def test_unknown_files_old_results_nested_wheels_and_unknown_directories_refused(self):
        for bad in ("new_source.py", "results/austria_2026_v5/a.json", "results/other/a.json",
                    "vendor/wheels/a.txt", "vendor/wheels/nested/a.whl", "unknown/",
                    "vendor/reporting-wheels/nested/", "results/austria_2026_v7"):
            with self.subTest(bad=bad):
                path, checksum = self.archive({"vendor/wheels/good.whl": b"ok", bad: b""})
                with self.assertRaisesRegex(ValueError, "Unlisted"):
                    restore(self.root, path, checksum)
                self.assert_no_artifact_writes()

    def test_modified_missing_or_case_alias_source_never_written(self):
        for name, local in (("README.md", b"working source"), ("README.md", None),
                            ("readme.md", self.source)):
            with self.subTest(name=name, local=local):
                original = self.root / "README.md"
                if local is None:
                    if original.exists():
                        original.unlink()
                else:
                    original.write_bytes(local)
                path, checksum = self.archive({"vendor/wheels/good.whl": b"ok", name: self.source})
                with self.assertRaisesRegex(ValueError, "source|Source"):
                    restore(self.root, path, checksum)
                if local is None:
                    self.assertFalse(original.exists())
                else:
                    self.assertEqual(original.read_bytes(), local)
                self.assert_no_artifact_writes()

    def test_duplicate_case_collision_and_linked_source_prevalidated(self):
        for options in ({"duplicate": "vendor/wheels/a.whl"}, {"duplicate": "vendor/wheels/A.whl"},
                        {"linked": "README.md"}, {"nonregular": "README.md"}):
            with self.subTest(options=options):
                path, checksum = self.archive({"vendor/wheels/a.whl": b"ok", "README.md": self.source},
                                              **options)
                with self.assertRaises(ValueError):
                    restore(self.root, path, checksum)
                self.assert_no_artifact_writes()

    def test_file_descendant_conflict_prevalidated(self):
        path, checksum = self.archive({"vendor/wheels/good.whl": b"ok",
                                       "results/austria_2026_v7/dir": b"file",
                                       "results/austria_2026_v7/dir/sub.json": b"{}"})
        with self.assertRaisesRegex(ValueError, "descendant"):
            restore(self.root, path, checksum)
        self.assert_no_artifact_writes()

    def test_existing_different_artifact_blocks_entire_restore(self):
        target = self.root / "results/austria_2026_v7/a.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing")
        path, checksum = self.archive({"vendor/wheels/good.whl": b"ok",
                                       "results/austria_2026_v7/a.json": b"new"})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            restore(self.root, path, checksum)
        self.assertEqual(target.read_bytes(), b"existing")
        self.assertFalse((self.root / "vendor").exists())

    def test_destination_file_ancestor_blocks_entire_restore(self):
        (self.root / "results").mkdir()
        (self.root / "results/austria_2026_v7").write_bytes(b"file")
        path, checksum = self.archive({"vendor/wheels/good.whl": b"ok",
                                       "results/austria_2026_v7/a.json": b"{}"})
        with self.assertRaisesRegex(ValueError, "not a directory"):
            restore(self.root, path, checksum)
        self.assertFalse((self.root / "vendor").exists())

    def test_reparse_ancestor_prevalidated(self):
        ancestor = self.root / "vendor"
        ancestor.mkdir()
        path, checksum = self.archive({"results/austria_2026_v7/a.json": b"{}",
                                       "vendor/wheels/a.whl": b"wheel"})
        original = os.lstat
        def fake(path, *args, **kwargs):
            if Path(path) == ancestor:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        with patch(RESTORE_MODULE + ".os.lstat", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "reparse"):
                restore(self.root, path, checksum)
        self.assertFalse((self.root / "results").exists())

    def test_artifact_allowlist_does_not_bypass_git_ignore(self):
        (self.root / ".gitignore").write_bytes(b"vendor/wheels/\n")
        path, checksum = self.archive({"vendor/wheels/good.whl": b"ok",
                                       "results/austria_2026_v7/a.json": b"{}"})
        with self.assertRaisesRegex(ValueError, "Git ignored"):
            restore(self.root, path, checksum)
        self.assert_no_artifact_writes()

    def test_tracked_artifact_is_verified_and_never_overwritten(self):
        target = self.root / "results/austria_2026_v7/a.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"{}")
        self.command("add", "-f", "results/austria_2026_v7/a.json")
        path, checksum = self.archive({"vendor/wheels/a.whl": b"ok",
                                       "results/austria_2026_v7/a.json": b"{}"})
        result = restore(self.root, path, checksum)
        self.assertEqual(result["verified_sources"], ["results/austria_2026_v7/a.json"])
        self.assertEqual(target.read_bytes(), b"{}")

    def test_fp2_and_late_arrays_refuse_entire_package(self):
        for bad in ("data/v4/fp2/raw/car_data.json", "data/v5/fp2/early.npz", "data/v4/test.npz",
                    "results/austria_2026_v7/test_arrays.npz", "results/austria_2026_v7/late_arrays.json.gz",
                    "results/austria_2026_v7/fp2_raw.json", "results/austria_2026_v7/fp2/a.json"):
            with self.subTest(bad=bad):
                path, checksum = self.archive({"vendor/wheels/good.whl": b"ok", bad: b"late"})
                with self.assertRaisesRegex(ValueError, "arrays are forbidden"):
                    restore(self.root, path, checksum)
                self.assert_no_artifact_writes()

    def test_legacy_status_requires_exact_tracked_path_fixed_bytes_and_local_match(self):
        from importlib import import_module
        module = import_module(RESTORE_MODULE)
        payload = b'{"historical_status":"pending","late_test_scored":false}\n'
        target = self.root / LEGACY_STATUS
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        self.command("add", LEGACY_STATUS)
        path, checksum = self.archive({"vendor/wheels/a.whl": b"ok", LEGACY_STATUS: payload})
        with patch.dict(module.LEGACY_METADATA_SHA256, {LEGACY_STATUS: hashlib.sha256(payload).hexdigest()}):
            result = restore(self.root, path, checksum)
            self.assertIn(LEGACY_STATUS, result["verified_sources"])
            (self.root / "vendor/wheels/a.whl").unlink()
            target.write_bytes(b"changed local")
            with self.assertRaisesRegex(ValueError, "source differs"):
                restore(self.root, path, checksum)
        with self.assertRaisesRegex(ValueError, "historical metadata changed"):
            restore(self.root, path, checksum)
        self.assertFalse((self.root / "vendor/wheels/a.whl").exists())

    def test_frozen_v2_access_receipt_is_skipped_but_false_receipt_is_rejected(self):
        name = "data/v2/test_access.json"
        location = Path(__file__).resolve()
        repository = (location.parents[1] if location.parent.name == "tests" else
                      location.parents[2] / "projects/f1-ers-lab")
        payload = (repository / name).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         "8bd93ee52906f0d21fb22dab6ffeaff69340dc6aea750d5b39e226e60ce6c23c")
        target = self.root / name
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        self.command("add", name)
        path, checksum = self.archive({name: payload})
        result = restore(self.root, path, checksum)
        self.assertEqual(result["verified_sources"], [name])
        self.assertEqual(result["restored"], [])
        path, checksum = self.archive({"vendor/wheels/a.whl": b"ok", name: b"disguised arrays"})
        with self.assertRaisesRegex(ValueError, "historical metadata changed"):
            restore(self.root, path, checksum)
        self.assertEqual(target.read_bytes(), payload)
        self.assert_no_artifact_writes()

    def test_directory_payload_and_truncated_names_block_all_writes(self):
        path, checksum = self.archive({"vendor/wheels/a.whl": b"ok"})
        original = zipfile.ZipFile.infolist
        def altered(bundle):
            entries = original(bundle)
            entries[0].orig_filename += "\x00hidden"
            return entries
        with patch.object(zipfile.ZipFile, "infolist", altered):
            with self.assertRaisesRegex(ValueError, "truncated"):
                restore(self.root, path, checksum)
        self.assert_no_artifact_writes()
        path, checksum = self.archive({"vendor/wheels/good.whl": b"ok",
                                       "results/austria_2026_v7/dir/": b"not empty"})
        with self.assertRaisesRegex(ValueError, "directory contains"):
            restore(self.root, path, checksum)
        self.assert_no_artifact_writes()

    def test_only_repository_root_is_accepted(self):
        nested = self.root / "nested"
        nested.mkdir()
        path, checksum = self.archive({"vendor/wheels/good.whl": b"ok"})
        with self.assertRaisesRegex(ValueError, "repository root"):
            restore(nested, path, checksum)
        self.assert_no_artifact_writes()


if __name__ == "__main__":
    unittest.main()
