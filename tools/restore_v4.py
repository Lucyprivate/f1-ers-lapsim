"""Restore a published v4 ZIP into its exact Git checkout using only stdlib.

The fixed ZIP SHA256 comes from the separately published delivery record.
Every member is validated before writes; source files are checked, never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

PREFIX = "f1-ers-lab-v4/"
MANIFEST = "DELIVERY-MANIFEST-V4.json"
PUBLISHED_SHA256 = "e4c3d065593b2727e86d92e8a8106127092a628c5de546fca1b1b64bc5195d3a"
TOOLING_FILES = {"tools/restore_v4.py", "tests/test_v4_restore.py", "RESTORE_V4.md"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if result.returncode:
        raise ValueError("Git check failed: " + result.stderr.decode(errors="replace"))
    return result.stdout.decode("utf-8").strip()


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or any(ord(c) < 32 or c in ':<>"|?*' for c in name)
            or path.is_absolute() or path.as_posix() != name):
        raise ValueError("Unsafe archive path: " + repr(name))
    for part in path.parts:
        device = part.split(".", 1)[0].upper()
        if (part in (".", "..") or part.endswith((".", " "))
                or device in {"CON", "PRN", "AUX", "NUL"}
                or re.fullmatch(r"(?:COM|LPT)[1-9]", device)):
            raise ValueError("Unsafe archive path: " + repr(name))
    return path


def reject_links(root, destination):
    for item in (root, *reversed(destination.relative_to(root).parents),):
        # Relative parents are checked through their absolute paths below.
        candidate = item if item.is_absolute() else root / item
        if candidate.exists() or candidate.is_symlink():
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Linked artifact destination: " + str(candidate))
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Artifact ancestor is not a directory: " + str(candidate))
    if destination.exists() or destination.is_symlink():
        info = os.lstat(destination)
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Linked artifact destination: " + str(destination))


def allowed_artifact(name):
    return (name.startswith("vendor/wheels/") or name.startswith("results/austria_2026_v4/")
            or name == "data/v4/fp2/status.json")


def restore(root, archive, expected_sha256=PUBLISHED_SHA256):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("A published SHA256 is required")
    root, archive = Path(root).absolute(), Path(archive).resolve()
    if digest(archive) != expected_sha256.lower():
        raise ValueError("Published ZIP SHA256 mismatch")
    if root.resolve() != root or Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Destination must be the unlinked Git repository root")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Artifact recovery requires a clean Git checkout")
    tracked = {name for name in git(root, "ls-files", "-z").split("\0") if name}
    planned, identical, checked = [], [], []
    with zipfile.ZipFile(archive) as bundle:
        members, folded = {}, set()
        for member in bundle.infolist():
            if member.orig_filename != member.filename:
                raise ValueError("ZIP filename was normalized or truncated")
            safe_relative(member.orig_filename)
            safe_relative(member.filename)
            if not member.filename.startswith(PREFIX) or member.is_dir():
                raise ValueError("Unexpected ZIP member: " + member.filename)
            name = member.filename[len(PREFIX):]
            safe_relative(name)
            if name.casefold() in folded:
                raise ValueError("Duplicate ZIP destination: " + name)
            folded.add(name.casefold())
            kind = stat.S_IFMT(member.external_attr >> 16)
            if kind not in (0, stat.S_IFREG):
                raise ValueError("Non-regular ZIP member: " + name)
            members[name] = member
        if MANIFEST not in members:
            raise ValueError("Missing delivery manifest")
        manifest = json.loads(bundle.read(members[MANIFEST]))
        records = manifest["files"]
        if manifest.get("schema_version") != 4 or set(members) != set(records) | {MANIFEST}:
            raise ValueError("ZIP members disagree with delivery manifest")
        git(root, "merge-base", "--is-ancestor", manifest["result_commit"], "HEAD")
        if not (tracked - TOOLING_FILES).issubset(records):
            raise ValueError("Manifest omits tracked source files")
        for name, record in records.items():
            relative = safe_relative(name)
            destination = root.joinpath(*relative.parts)
            reject_links(root, destination)
            if not destination.resolve().is_relative_to(root):
                raise ValueError("Artifact destination escapes checkout")
            payload = bundle.read(members[name])
            if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("Invalid member hash: " + name)
            if name in tracked:
                if not destination.is_file() or digest(destination) != record["sha256"]:
                    raise ValueError("Tracked source differs: " + name)
                checked.append(name)
                continue
            if not allowed_artifact(name):
                raise ValueError("Artifact is outside recovery allowlist: " + name)
            ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", name])
            if ignored.returncode != 0:
                raise ValueError("Artifact is not ignored: " + name)
            if destination.exists():
                if not destination.is_file() or digest(destination) != record["sha256"]:
                    raise ValueError("Refusing artifact overwrite: " + name)
                identical.append(name)
            else:
                planned.append((name, destination))
        # No path or hash validation remains after this point.
        for name, destination in planned:
            payload = bundle.read(members[name])
            if (len(payload) != records[name]["bytes"]
                    or hashlib.sha256(payload).hexdigest() != records[name]["sha256"]):
                raise ValueError("Artifact changed after prevalidation: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            reject_links(root, destination)
            with destination.open("xb") as stream:
                stream.write(payload)
    return {"archive_sha256": expected_sha256.lower(), "result_commit": manifest["result_commit"],
            "tracked_files_verified": len(checked), "restored": [name for name, _ in planned],
            "identical_artifacts": identical, "source_overwritten": False,
            "late_test_arrays_restored": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.root, args.archive), ensure_ascii=False, indent=2))
