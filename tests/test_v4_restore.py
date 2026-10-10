"""Safety checks for dependency-free cold-clone artifact recovery."""
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

from tools.restore_v4 import restore


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.command("init", "-q")
        self.command("config", "core.autocrlf", "false")
        self.command("config", "user.name", "Restore test")
        self.command("config", "user.email", "restore@local.invalid")
        (self.root / "README.md").write_bytes(b"Frozen source\n")
        (self.root / ".gitignore").write_bytes(b"vendor/wheels/\nresults/\ndata/v4/fp2/\n")
        self.command("add", ".")
        self.command("commit", "-qm", "fixture")
        self.tracked = {name: (self.root / name).read_bytes() for name in ("README.md", ".gitignore")}

    def command(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                              capture_output=True).stdout.decode().strip()

    def archive(self, extra, *, duplicate=None, linked=None):
        entries = {**self.tracked, **extra}
        records = {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                   for name, payload in entries.items()}
        manifest = {"schema_version": 4, "result_commit": self.command("rev-parse", "HEAD"), "files": records}
        archive = Path(self.temp.name) / "artifacts.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name, payload in entries.items():
                info = zipfile.ZipInfo("f1-ers-lab-v4/" + name)
                if name == linked:
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, payload)
            if duplicate:
                z.writestr("f1-ers-lab-v4/" + duplicate, b"duplicate")
            z.writestr("f1-ers-lab-v4/DELIVERY-MANIFEST-V4.json", json.dumps(manifest))
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def test_restore_is_idempotent_and_never_writes_source(self):
        archive, checksum = self.archive({"vendor/wheels/a.whl": b"wheel", "results/austria_2026_v4/a.json": b"{}"})
        first = restore(self.root, archive, checksum)
        self.assertEqual(len(first["restored"]), 2)
        self.assertFalse(first["source_overwritten"])
        self.assertEqual(restore(self.root, archive, checksum)["restored"], [])
        self.assertEqual((self.root / "README.md").read_bytes(), self.tracked["README.md"])
        self.assertEqual(self.command("status", "--porcelain"), "")

    def test_wrong_published_hash_writes_nothing(self):
        archive, _ = self.archive({"vendor/wheels/a.whl": b"wheel"})
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            restore(self.root, archive, "0" * 64)
        self.assertFalse((self.root / "vendor").exists())

    def test_path_traversal_and_windows_aliases_writes_nothing(self):
        for bad in ("../outside", "/absolute", "vendor/wheels/CON", "vendor/wheels/a.",
                    "vendor/wheels/a ", "vendor/wheels/a:b", "vendor/wheels/a\\b", "vendor/wheels/a?"):
            with self.subTest(bad=bad):
                archive, checksum = self.archive({"vendor/wheels/good.whl": b"ok", bad: b"bad"})
                with self.assertRaises(ValueError):
                    restore(self.root, archive, checksum)
                self.assertFalse((self.root / "vendor").exists())

    def test_duplicate_and_link_members_are_rejected(self):
        extra = {"vendor/wheels/a.whl": b"wheel"}
        for options in ({"duplicate": "vendor/wheels/A.whl"}, {"linked": "vendor/wheels/a.whl"}):
            archive, checksum = self.archive(extra, **options)
            with self.assertRaises(ValueError):
                restore(self.root, archive, checksum)
            self.assertFalse((self.root / "vendor").exists())

    def test_conflicting_existing_artifact_is_prevalidated(self):
        existing = self.root / "results/austria_2026_v4/a.json"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old")
        archive, checksum = self.archive({"vendor/wheels/good.whl": b"ok", "results/austria_2026_v4/a.json": b"new"})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            restore(self.root, archive, checksum)
        self.assertEqual(existing.read_bytes(), b"old")
        self.assertFalse((self.root / "vendor").exists())

    def test_late_arrays_and_modified_source_are_rejected(self):
        archive, checksum = self.archive({"vendor/wheels/a.whl": b"ok", "data/v4/fp2/test_arrays.npz": b"late"})
        with self.assertRaisesRegex(ValueError, "allowlist"):
            restore(self.root, archive, checksum)
        self.assertFalse((self.root / "vendor").exists())
        archive, checksum = self.archive({"vendor/wheels/a.whl": b"ok"})
        (self.root / "README.md").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "clean Git"):
            restore(self.root, archive, checksum)

    def test_reparse_point_ancestor_is_rejected(self):
        ancestor = self.root / "vendor"
        ancestor.mkdir()
        archive, checksum = self.archive({"vendor/wheels/a.whl": b"wheel"})
        original = __import__("os").lstat
        def fake_stat(path):
            if Path(path) == ancestor:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path)
        with patch("tools.restore_v4.os.lstat", side_effect=fake_stat):
            with self.assertRaisesRegex(ValueError, "Linked"):
                restore(self.root, archive, checksum)
        self.assertFalse((ancestor / "wheels").exists())

    def test_file_ancestor_is_prevalidated_before_any_write(self):
        parent = self.root / "results"
        parent.mkdir()
        (parent / "austria_2026_v4").write_bytes(b"file in place of directory")
        archive, checksum = self.archive({"vendor/wheels/good.whl": b"ok", "results/austria_2026_v4/a.json": b"{}"})
        with self.assertRaisesRegex(ValueError, "not a directory"):
            restore(self.root, archive, checksum)
        self.assertFalse((self.root / "vendor").exists())

    def test_truncated_zip_filename_is_rejected(self):
        archive, checksum = self.archive({"vendor/wheels/a.whl": b"ok"})
        original = zipfile.ZipFile.infolist
        def altered_names(bundle):
            entries = original(bundle)
            entries[0].orig_filename += "\x00hidden"
            return entries
        with patch.object(zipfile.ZipFile, "infolist", altered_names):
            with self.assertRaisesRegex(ValueError, "normalized or truncated"):
                restore(self.root, archive, checksum)
        self.assertFalse((self.root / "vendor").exists())


if __name__ == "__main__":
    unittest.main()
