"""Stdlib ZIP recovery prevalidation, source protection and sealed-array cases."""
import hashlib
import json
from pathlib import Path
import os
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

from tools.restore_v5 import restore


class V5RestoreTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/".cache"/"v5-restore-tests"
        cache.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"repo"
        self.root.mkdir()
        self.command("init","-q")
        self.command("config","core.autocrlf","false")
        self.command("config","user.name","F1 restoration fixture")
        self.command("config","user.email","restore@local.invalid")
        (self.root/"README.md").write_bytes(b"source remains unchanged\n")
        (self.root/".gitignore").write_bytes(b"results/\nvendor/wheels/\ndata/v4/fp2/\n")
        self.command("add",".")
        self.command("commit","-qm","fixture")

    def command(self,*args):
        return subprocess.run(["git","-C",str(self.root),*args],capture_output=True,check=True).stdout.decode().strip()

    def archive(self,entries,*,linked=None,duplicate=None,nonregular=None):
        path = Path(self.temp.name)/"delivery.zip"
        with zipfile.ZipFile(path,"w") as bundle:
            for name,payload in entries.items():
                member = zipfile.ZipInfo(name)
                # Keep hostile raw names in the ZIP central directory; the
                # Windows ZipInfo constructor otherwise rewrites backslashes.
                member.filename=name
                member.orig_filename=name
                if name==linked:
                    member.create_system=3;member.external_attr=(stat.S_IFLNK|0o777)<<16
                if name==nonregular:
                    member.create_system=3;member.external_attr=(stat.S_IFIFO|0o600)<<16
                bundle.writestr(member,payload)
            if duplicate is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",UserWarning)
                    bundle.writestr(duplicate,b"duplicate")
        return path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_allowed_artifacts_restore_with_public_manifest_and_idempotency(self):
        entries = {"README.md":b"different source in archive", "DELIVERY-MANIFEST-V5.json":b"{}",
                   "vendor/wheels/a.whl":b"wheel", "results/austria_2026_v5/frozen_configuration.json":b"{}",
                   "results/austria_2026_v4/verification.json":b"{}",
                   "results/austria_2026_v5/test_trajectories.json.gz":b"saved model result"}
        path,checksum=self.archive(entries)
        result=restore(self.root,path,checksum)
        self.assertEqual(len(result["restored"]),4)
        self.assertFalse(result["source_overwritten"])
        self.assertFalse(result["late_test_arrays_restored"])
        records={r["path"]:r for r in result["member_manifest"]}
        for name,payload in entries.items():
            self.assertEqual(records[name]["sha256"],hashlib.sha256(payload).hexdigest())
            self.assertEqual(records[name]["bytes"],len(payload))
        self.assertEqual(records["README.md"]["action"],"skipped_source_or_unlisted_member")
        self.assertEqual((self.root/"README.md").read_bytes(),b"source remains unchanged\n")
        self.assertFalse((self.root/"DELIVERY-MANIFEST-V5.json").exists())
        again=restore(self.root,path,checksum)
        self.assertEqual(again["restored"],[])
        self.assertEqual(len(again["identical_artifacts"]),4)
        json.dumps(result)

    def test_explicit_complete_published_hash_required(self):
        path,_=self.archive({"vendor/wheels/a.whl":b"wheel"})
        for bad in ("0"*64,"not-a-hash",None):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                restore(self.root,path,bad)
        self.assertFalse((self.root/"vendor").exists())

    def test_every_member_path_is_checked_before_any_write(self):
        for bad in ("../outside","/absolute","C:/outside",".","a/../b","a//b", "a\\b",
                    "vendor/wheels/CON","vendor/wheels/COM1.whl","vendor/wheels/LPT¹.txt",
                    "vendor/wheels/CONOUT$","vendor/wheels/a.","vendor/wheels/a ",
                    "vendor/wheels/a:b","vendor/wheels/a?","vendor/wheels/REPORT~1.TXT"):
            with self.subTest(bad=bad):
                path,checksum=self.archive({"vendor/wheels/good.whl":b"ok",bad:b"bad"})
                with self.assertRaises(ValueError):restore(self.root,path,checksum)
                self.assertFalse((self.root/"vendor").exists())

    def test_case_duplicate_and_linked_skipped_source_are_rejected(self):
        for options in ({"duplicate":"vendor/wheels/A.whl"},{"linked":"README.md"},{"nonregular":"README.md"}):
            path,checksum=self.archive({"vendor/wheels/a.whl":b"ok","README.md":b"source"},**options)
            with self.assertRaises(ValueError):restore(self.root,path,checksum)
            self.assertFalse((self.root/"vendor").exists())

    def test_file_and_descendant_conflict_is_prevalidated(self):
        path,checksum=self.archive({"vendor/wheels/good.whl":b"ok","results/austria_2026_v5/dir":b"file",
                                    "results/austria_2026_v5/dir/sub.json":b"{}"})
        with self.assertRaisesRegex(ValueError,"descendant"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"vendor").exists())

    def test_existing_different_artifact_blocks_entire_recovery(self):
        target=self.root/"results/austria_2026_v5/a.json"
        target.parent.mkdir(parents=True);target.write_bytes(b"existing")
        path,checksum=self.archive({"vendor/wheels/good.whl":b"ok","results/austria_2026_v5/a.json":b"new"})
        with self.assertRaisesRegex(ValueError,"overwrite"):restore(self.root,path,checksum)
        self.assertEqual(target.read_bytes(),b"existing")
        self.assertFalse((self.root/"vendor").exists())

    def test_destination_file_ancestor_blocks_all_writes(self):
        parent=self.root/"results";parent.mkdir()
        (parent/"austria_2026_v5").write_bytes(b"file")
        path,checksum=self.archive({"vendor/wheels/good.whl":b"ok","results/austria_2026_v5/a.json":b"{}"})
        with self.assertRaisesRegex(ValueError,"not a directory"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"vendor").exists())

    def test_reparse_ancestor_is_prevalidated(self):
        ancestor=self.root/"vendor";ancestor.mkdir()
        path,checksum=self.archive({"results/austria_2026_v5/a.json":b"{}","vendor/wheels/a.whl":b"wheel"})
        original=os.lstat
        def fake(path):
            if Path(path)==ancestor:return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
            return original(path)
        with patch("tools.restore_v5.os.lstat",side_effect=fake):
            with self.assertRaisesRegex(ValueError,"reparse"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"results").exists())

    def test_tracked_artifact_is_never_overwritten_even_if_identical(self):
        target=self.root/"results/austria_2026_v5/a.json"
        target.parent.mkdir(parents=True);target.write_bytes(b"{}")
        self.command("add","-f","results/austria_2026_v5/a.json")
        self.command("commit","-qm","tracked artifact fixture")
        path,checksum=self.archive({"vendor/wheels/good.whl":b"ok","results/austria_2026_v5/a.json":b"{}"})
        with self.assertRaisesRegex(ValueError,"Git-tracked"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"vendor").exists())

    def test_allowlisted_destination_still_must_be_ignored(self):
        (self.root/".gitignore").write_bytes(b"vendor/wheels/\n")
        path,checksum=self.archive({"vendor/wheels/good.whl":b"ok","results/austria_2026_v5/a.json":b"{}"})
        with self.assertRaisesRegex(ValueError,"Git ignored"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"vendor").exists())

    def test_fp2_windows_and_processed_arrays_refuse_entire_package(self):
        for bad in ("data/v4/fp2/raw/car_data.json","data/v5/fp2/early.npz","data/v4/test_arrays.npz",
                    "results/austria_2026_v5/test_arrays.npz","results/austria_2026_v5/late_arrays.json.gz"):
            with self.subTest(bad=bad):
                path,checksum=self.archive({"vendor/wheels/good.whl":b"ok",bad:b"late"})
                with self.assertRaisesRegex(ValueError,"arrays are forbidden"):restore(self.root,path,checksum)
                self.assertFalse((self.root/"vendor").exists())

    def test_skipped_members_cannot_write_outside_allowlist_or_modified_source(self):
        (self.root/"README.md").write_bytes(b"working source")
        path,checksum=self.archive({"README.md":b"archived source","new_source.py":b"source",
                                    "results/other_experiment/a.json":b"{}","vendor/wheels/a.whl":b"wheel"})
        result=restore(self.root,path,checksum)
        self.assertEqual(result["restored"],["vendor/wheels/a.whl"])
        self.assertEqual((self.root/"README.md").read_bytes(),b"working source")
        self.assertFalse((self.root/"new_source.py").exists())
        self.assertFalse((self.root/"results").exists())

    def test_only_exact_frozen_historical_status_is_skipped(self):
        from tools.restore_v5 import LEGACY_STATUS, LEGACY_STATUS_SHA256
        payload=(Path(__file__).resolve().parents[1]/LEGACY_STATUS).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), LEGACY_STATUS_SHA256)
        path,checksum=self.archive({LEGACY_STATUS:payload,"vendor/wheels/a.whl":b"wheel"})
        result=restore(self.root,path,checksum)
        self.assertEqual(result["restored"],["vendor/wheels/a.whl"])
        self.assertFalse((self.root/"data").exists())
        (self.root/"vendor/wheels/a.whl").unlink()
        path,checksum=self.archive({LEGACY_STATUS:b'{"late_arrays":[42]}',"vendor/wheels/a.whl":b"wheel"})
        with self.assertRaisesRegex(ValueError,"historical metadata changed"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"vendor/wheels/a.whl").exists())

    def test_truncated_names_and_directory_payloads_are_rejected(self):
        path,checksum=self.archive({"vendor/wheels/a.whl":b"ok"})
        original=zipfile.ZipFile.infolist
        def altered(bundle):
            entries=original(bundle);entries[0].orig_filename+="\x00hidden";return entries
        with patch.object(zipfile.ZipFile,"infolist",altered):
            with self.assertRaisesRegex(ValueError,"truncated"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"vendor").exists())
        path,checksum=self.archive({"vendor/wheels/good.whl":b"ok","some_directory/":b"not empty"})
        with self.assertRaisesRegex(ValueError,"directory contains"):restore(self.root,path,checksum)
        self.assertFalse((self.root/"vendor").exists())


if __name__ == "__main__":
    unittest.main()
