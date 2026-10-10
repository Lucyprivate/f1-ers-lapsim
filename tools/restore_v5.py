"""Recover ignored v5 delivery artifacts with Python's standard library.

The SHA256 must come from the separately published delivery record. Every ZIP
member and destination is checked before the first write. Legal source members
are skipped, never written; FP2 source windows and processed arrays are refused.
The JSON result is a public member manifest including hashes and actions.
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


ALLOWED_PREFIXES = ("results/austria_2026_v5/", "results/austria_2026_v4/", "vendor/wheels/")
LEGACY_STATUS = "data/v3/fp2/status.json"
LEGACY_STATUS_SHA256 = "d2ff18b420210fd8512cd282838da4e93bac09d2ccf89464a55688670fcc2f16"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def git(root, *args):
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if process.returncode:
        raise ValueError("Git check failed: " + process.stderr.decode(errors="replace"))
    return process.stdout.decode("utf-8").strip()


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or not path.parts or "\\" in name or any(ord(c)<32 or c in ':<>"|?*' for c in name)
            or path.is_absolute() or path.as_posix()!=name):
        raise ValueError("Unsafe ZIP member: " + repr(name))
    for part in path.parts:
        device = part.split(".",1)[0].upper()
        if (part in (".","..") or part.endswith(("."," "))
                or device in {"CON","PRN","AUX","NUL","CONIN$","CONOUT$","CLOCK$"}
                or "~" in part
                or re.fullmatch(r"(?:COM|LPT)(?:[1-9]|[¹²³])",device)):
            raise ValueError("Unsafe Windows alias: " + repr(name))
    return path


def forbidden_arrays(name):
    """Scores/receipts are results, while performance source arrays stay sealed."""
    lower = name.casefold()
    parts = PurePosixPath(lower).parts
    if name == LEGACY_STATUS:
        return False  # Fixed historical metadata; checked and skipped below.
    if len(parts)>=3 and parts[0]=="data" and "fp2" in parts:
        return True
    if lower.startswith(("data/v4/test", "data/v3/test", "data/v5/test")):
        return True
    return any(part.startswith(("test_arrays", "late_arrays", "fp2_raw", "fp2_processed")) for part in parts)


def reject_links(root, destination):
    ancestors = list(reversed(root.parents))+[root]
    ancestors += [root / part for part in reversed(destination.relative_to(root).parents) if part!=Path(".")]
    for candidate in ancestors:
        if candidate.exists() or candidate.is_symlink():
            item = os.lstat(candidate)
            if stat.S_ISLNK(item.st_mode) or getattr(item,"st_file_attributes",0)&0x400:
                raise ValueError("Linked or reparse destination ancestor: " + str(candidate))
            if not stat.S_ISDIR(item.st_mode):
                raise ValueError("Destination ancestor is not a directory: " + str(candidate))
    if destination.exists() or destination.is_symlink():
        item = os.lstat(destination)
        if stat.S_ISLNK(item.st_mode) or getattr(item,"st_file_attributes",0)&0x400:
            raise ValueError("Linked or reparse artifact destination: " + str(destination))


def restore(root, archive, expected_sha256):
    if not isinstance(expected_sha256,str) or not re.fullmatch(r"[0-9a-fA-F]{64}",expected_sha256):
        raise ValueError("An explicit published 64-character SHA256 is required")
    root = Path(os.path.abspath(root))
    archive = Path(archive).resolve()
    if digest(archive)!=expected_sha256.lower():
        raise ValueError("Published ZIP SHA256 mismatch")
    if root.resolve()!=root or Path(git(root,"rev-parse","--show-toplevel")).resolve()!=root:
        raise ValueError("Destination must be the unlinked Git repository root")
    reject_links(root,root / ".restore-v5-root-check")
    tracked = {n.casefold() for n in git(root,"ls-files","-z").split("\0") if n}
    planned, manifest = [], []
    with zipfile.ZipFile(archive) as bundle:
        members, folded, files = [], set(), set()
        for member in bundle.infolist():
            if member.orig_filename!=member.filename:
                raise ValueError("ZIP filename was normalized or truncated")
            name = member.filename[:-1] if member.is_dir() else member.filename
            relative = safe_relative(name)
            folded_name = name.casefold()
            if folded_name in folded:
                raise ValueError("Duplicate ZIP destination: " + name)
            folded.add(folded_name)
            kind = stat.S_IFMT(member.external_attr >> 16)
            permitted = (0,stat.S_IFDIR) if member.is_dir() else (0,stat.S_IFREG)
            if kind not in permitted or (member.external_attr & 0xffff)&0x400:
                raise ValueError("Non-regular or linked ZIP member: " + name)
            if forbidden_arrays(name):
                raise ValueError("FP2 raw/processed arrays are forbidden: " + name)
            members.append((name,relative,member))
            if not member.is_dir():
                files.add(folded_name)
        # A regular file cannot also serve as a directory for another member.
        for name, relative, _ in members:
            if any(parent.as_posix().casefold() in files for parent in relative.parents if parent!=PurePosixPath(".")):
                raise ValueError("ZIP file conflicts with a descendant: " + name)
        for name, relative, member in members:
            if member.is_dir():
                if member.file_size or bundle.read(member):
                    raise ValueError("ZIP directory contains file data: " + name)
                manifest.append({"path":name,"bytes":0,"sha256":None,"action":"skipped_directory"})
                continue
            payload = bundle.read(member)  # Validates the CRC of every member.
            checksum = hashlib.sha256(payload).hexdigest()
            if name == LEGACY_STATUS and checksum != LEGACY_STATUS_SHA256:
                raise ValueError("Frozen historical metadata changed: " + name)
            record = {"path":name,"bytes":len(payload),"sha256":checksum}
            if not name.startswith(ALLOWED_PREFIXES):
                manifest.append({**record,"action":"skipped_source_or_unlisted_member"})
                continue
            destination = root.joinpath(*relative.parts)
            reject_links(root,destination)
            if not destination.resolve().is_relative_to(root):
                raise ValueError("Artifact destination escapes repository: " + name)
            if name.casefold() in tracked:
                raise ValueError("Refusing a Git-tracked artifact destination: " + name)
            ignored = subprocess.run(["git","-C",str(root),"check-ignore","-q","--",name],capture_output=True)
            if ignored.returncode!=0:
                raise ValueError("Artifact destination must be Git ignored: " + name)
            if destination.exists():
                if not destination.is_file() or digest(destination)!=checksum:
                    raise ValueError("Refusing to overwrite existing artifact: " + name)
                manifest.append({**record,"action":"identical_artifact"})
            else:
                planned.append((member,destination,record))
                manifest.append({**record,"action":"restored_artifact"})
        # Every member, hash, ignore rule and destination has now been checked.
        for member,destination,record in planned:
            payload = bundle.read(member)
            if len(payload)!=record["bytes"] or hashlib.sha256(payload).hexdigest()!=record["sha256"]:
                raise ValueError("ZIP member changed after prevalidation: " + record["path"])
            destination.parent.mkdir(parents=True,exist_ok=True)
            reject_links(root,destination)
            with destination.open("xb") as stream:
                stream.write(payload)
    return {"schema_version":5,"archive_sha256":expected_sha256.lower(),
            "restored":[r["path"] for r in manifest if r["action"]=="restored_artifact"],
            "identical_artifacts":[r["path"] for r in manifest if r["action"]=="identical_artifact"],
            "member_manifest":manifest,"source_overwritten":False,"late_test_arrays_restored":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip",dest="archive",type=Path,required=True)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.root,args.archive,args.sha256),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
