"""Restore only ignored v6 artifacts after a separately published ZIP SHA256.

Clone the corresponding Git bundle first. ZIP source files must match exact
Git-tracked paths and local bytes, and are never restored. Every member and
destination is prevalidated before the first write. This module uses only the
Python standard library and Git; it never opens sealed FP2 telemetry arrays.
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


ZIP_PREFIX = "f1-ers-lab-v6/"
RESULTS_PREFIX = "results/austria_2026_v6/"
WHEEL_DIRECTORIES = ("vendor/wheels", "vendor/reporting-wheels")
LEGACY_STATUS = "data/v3/fp2/status.json"
LEGACY_STATUS_SHA256 = "d2ff18b420210fd8512cd282838da4e93bac09d2ccf89464a55688670fcc2f16"
LEGACY_METADATA_SHA256 = {
    LEGACY_STATUS: LEGACY_STATUS_SHA256,
    "data/v2/test_access.json": "8bd93ee52906f0d21fb22dab6ffeaff69340dc6aea750d5b39e226e60ce6c23c",
}
REPARSE_POINT = 0x400


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def git_bytes(root, *args):
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if process.returncode:
        raise ValueError("Git check failed: " + process.stderr.decode(errors="replace"))
    return process.stdout


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or not path.parts or "\\" in name
            or any(ord(c) < 32 or c in ':<>"|?*' for c in name)
            or path.is_absolute() or path.as_posix() != name):
        raise ValueError("Unsafe ZIP member: " + repr(name))
    for part in path.parts:
        device = part.split(".", 1)[0].upper()
        if (part in (".", "..") or part.endswith((".", " ")) or "~" in part
                or device in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"}
                or re.fullmatch(r"(?:COM|LPT)(?:[1-9]|[¹²³])", device)):
            raise ValueError("Unsafe Windows alias: " + repr(name))
    return path


def forbidden_arrays(name):
    if name in LEGACY_METADATA_SHA256:
        return False  # Exact tracked historical receipt; its fixed hash is checked.
    parts = PurePosixPath(name.casefold()).parts
    return ("fp2" in parts
            or (len(parts) >= 3 and parts[0] == "data" and parts[2].startswith("test"))
            or any(part.startswith(("test_arrays", "late_arrays", "fp2_raw",
                                    "fp2_processed", "raw_fp2", "processed_fp2"))
                   for part in parts))


def allowed_artifact(name):
    path = PurePosixPath(name)
    return (name.startswith(RESULTS_PREFIX)
            or (path.parent.as_posix() in WHEEL_DIRECTORIES and path.name.endswith(".whl")))


def reject_links(root, destination):
    """Reject links/reparse points even in existing ancestors above the root."""
    ancestors = list(reversed(root.parents)) + [root]
    ancestors += [root / part for part in reversed(destination.relative_to(root).parents)
                  if part != Path(".")]
    for candidate in ancestors:
        try:
            item = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(item.st_mode) or getattr(item, "st_file_attributes", 0) & REPARSE_POINT:
            raise ValueError("Linked or reparse destination ancestor: " + str(candidate))
        if not stat.S_ISDIR(item.st_mode):
            raise ValueError("Destination ancestor is not a directory: " + str(candidate))
    try:
        item = os.lstat(destination)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(item.st_mode) or getattr(item, "st_file_attributes", 0) & REPARSE_POINT:
        raise ValueError("Linked or reparse destination: " + str(destination))


def restore(root, archive, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("An explicit separately published 64-character SHA256 is required")
    root = Path(os.path.abspath(root))
    archive = Path(archive).resolve()
    if digest(archive) != expected_sha256.lower():
        raise ValueError("Published ZIP SHA256 mismatch")
    reject_links(root, root / ".restore-v6-root-check")
    toplevel = Path(git_bytes(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if root.resolve() != root or toplevel != root:
        raise ValueError("Destination must be the unlinked Git repository root")
    tracked = {name.decode("utf-8") for name in git_bytes(root, "ls-files", "-z").split(b"\0") if name}
    tracked_folded = {name.casefold() for name in tracked}
    if len(tracked_folded) != len(tracked):
        raise ValueError("Git index contains case-colliding source paths")
    source_directories = {parent.as_posix() for name in tracked for parent in PurePosixPath(name).parents
                          if parent != PurePosixPath(".")}
    artifact_directories = {"results", "results/austria_2026_v6", "vendor", *WHEEL_DIRECTORIES}
    planned, manifest = [], []
    with zipfile.ZipFile(archive) as bundle:
        members, folded, files = [], set(), set()
        for member in bundle.infolist():
            if member.orig_filename != member.filename:
                raise ValueError("ZIP filename was normalized or truncated")
            full_name = member.filename[:-1] if member.is_dir() else member.filename
            safe_relative(full_name)  # Validate the archive root as well as its descendants.
            if full_name == ZIP_PREFIX[:-1] and member.is_dir():
                name, relative = "", PurePosixPath(".")
            else:
                if not full_name.startswith(ZIP_PREFIX):
                    raise ValueError("ZIP member is outside required prefix: " + full_name)
                name = full_name[len(ZIP_PREFIX):]
                relative = safe_relative(name)
            folded_name = name.casefold()
            if folded_name in folded:
                raise ValueError("Duplicate ZIP destination: " + name)
            folded.add(folded_name)
            kind = stat.S_IFMT(member.external_attr >> 16)
            permitted = (0, stat.S_IFDIR) if member.is_dir() else (0, stat.S_IFREG)
            if kind not in permitted or (member.external_attr & 0xffff) & REPARSE_POINT:
                raise ValueError("Non-regular or linked ZIP member: " + full_name)
            if forbidden_arrays(name):
                raise ValueError("FP2 raw/processed or late arrays are forbidden: " + name)
            members.append((name, relative, member))
            if not member.is_dir():
                files.add(folded_name)
        for name, relative, _ in members:
            if any(parent.as_posix().casefold() in files for parent in relative.parents
                   if parent != PurePosixPath(".")):
                raise ValueError("ZIP file conflicts with a descendant: " + name)
        for name, relative, member in members:
            if member.is_dir():
                if member.file_size or bundle.read(member):
                    raise ValueError("ZIP directory contains file data: " + name)
                if (name and name not in source_directories and name not in artifact_directories
                        and not name.startswith(RESULTS_PREFIX)):
                    raise ValueError("Unlisted ZIP directory: " + name)
                destination = root.joinpath(*relative.parts)
                reject_links(root, destination)
                if destination.exists() and not destination.is_dir():
                    raise ValueError("ZIP directory destination is not a directory: " + name)
                manifest.append({"path": name, "bytes": 0, "sha256": None, "action": "skipped_directory"})
                continue
            payload = bundle.read(member)  # CRC is verified even for source members.
            checksum = hashlib.sha256(payload).hexdigest()
            record = {"path": name, "bytes": len(payload), "sha256": checksum}
            destination = root.joinpath(*relative.parts)
            reject_links(root, destination)
            if not destination.resolve().is_relative_to(root):
                raise ValueError("Artifact destination escapes repository: " + name)
            if name in LEGACY_METADATA_SHA256 and checksum != LEGACY_METADATA_SHA256[name]:
                raise ValueError("Frozen historical metadata changed: " + name)
            if name in tracked:
                if not destination.is_file() or digest(destination) != checksum:
                    raise ValueError("Tracked source differs from corresponding ZIP revision: " + name)
                manifest.append({**record, "action": "verified_tracked_source_skipped"})
                continue
            if name.casefold() in tracked_folded:
                raise ValueError("Source case alias is not an exact Git-tracked path: " + name)
            if name in LEGACY_METADATA_SHA256 or not allowed_artifact(name):
                raise ValueError("Unlisted ZIP file: " + name)
            ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", name],
                                     capture_output=True)
            if ignored.returncode != 0:
                raise ValueError("Artifact destination must be Git ignored: " + name)
            if destination.exists():
                if not destination.is_file() or digest(destination) != checksum:
                    raise ValueError("Refusing to overwrite existing artifact: " + name)
                manifest.append({**record, "action": "identical_artifact"})
            else:
                planned.append((member, destination, record))
                manifest.append({**record, "action": "restored_artifact"})
        # No directory or file is created until all entries/destinations pass.
        for member, destination, record in planned:
            payload = bundle.read(member)
            if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("ZIP member changed after prevalidation: " + record["path"])
            reject_links(root, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            reject_links(root, destination)
            with destination.open("xb") as stream:
                stream.write(payload)
    return {"schema_version": 6, "archive_sha256": expected_sha256.lower(), "zip_prefix": ZIP_PREFIX,
            "restored": [r["path"] for r in manifest if r["action"] == "restored_artifact"],
            "identical_artifacts": [r["path"] for r in manifest if r["action"] == "identical_artifact"],
            "verified_sources": [r["path"] for r in manifest if r["action"] == "verified_tracked_source_skipped"],
            "member_manifest": manifest, "source_overwritten": False, "late_test_arrays_restored": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.root, args.archive, args.sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
