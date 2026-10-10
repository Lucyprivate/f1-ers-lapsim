"""Verify a sealed research artifact directory using tracked public evidence.

Standard library only. This checks byte identity and inventory, not simulation
correctness or real-world model accuracy. It does not execute project code, load
telemetry, extract archives, modify artifacts, or allocate training attempts.
The tracked summary is the trust anchor; verify its public Git provenance
separately. Supplying a different --summary changes that trust anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any


DEFAULT_SUMMARY = Path(__file__).resolve().parents[1] / "evidence/research/summary.json"
MANIFEST_NAME = "manifest.json"
FREEZE_NAME = "freeze.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class VerificationError(ValueError):
    """The evidence pin, inventory, or artifact content is inconsistent."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise VerificationError(f"Non-finite JSON constant: {value}")


def _is_link(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _file_stat(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if _is_link(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise VerificationError(f"Expected a regular file without links: {path}")
    return metadata


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    # Windows lstat/fstat expose different legacy ctime meanings on some Python
    # versions. Device, file ID, size, and mtime agree for the same open file.
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _read(path: Path, *, limit: int | None = None) -> tuple[str, bytes | None]:
    """Hash an ordinary file and reject changes during the read."""
    before = _file_stat(path)
    if limit is not None and before.st_size > limit:
        raise VerificationError(f"JSON file exceeds {limit} bytes: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks = [] if limit is not None else None
    digest = hashlib.sha256()
    bytes_read = 0
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _identity(opened) != _identity(before):
            raise VerificationError(f"File changed before reading: {path}")
        while chunk := stream.read(1024 * 1024):
            bytes_read += len(chunk)
            if limit is not None and bytes_read > limit:
                raise VerificationError(f"JSON file grew beyond {limit} bytes: {path}")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise VerificationError(f"File changed while reading: {path}")
    if _identity(_file_stat(path)) != _identity(before):
        raise VerificationError(f"File changed after reading: {path}")
    return digest.hexdigest(), b"".join(chunks) if chunks is not None else None


def _json(path: Path, *, limit: int = 16 * 1024 * 1024) -> tuple[str, dict[str, Any]]:
    digest, payload = _read(path, limit=limit)
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_object,
                              parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Invalid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise VerificationError(f"Expected a JSON object: {path}")
    return digest, document


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise VerificationError(f"Invalid lowercase SHA256 pin: {name}")
    return value


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise VerificationError(f"Invalid artifact path: {value!r}")
    parts = value.split("/")
    if (PurePosixPath(value).is_absolute() or PurePosixPath(value).as_posix() != value
            or any(part in ("", ".", "..") for part in parts)):
        raise VerificationError(f"Unsafe artifact path: {value!r}")
    for part in parts:
        if (part[-1] in (" ", ".") or any(ord(char) < 32 or ord(char) == 127 for char in part)
                or any(char in ':*?<>|"' for char in part)
                or part.split(".", 1)[0].upper() in _RESERVED):
            raise VerificationError(f"Nonportable or unsafe artifact path: {value!r}")
    return value


def _root(path: Path) -> Path:
    metadata = path.lstat()
    if _is_link(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise VerificationError(f"Artifact root must be a directory without links: {path}")
    return path.resolve(strict=True)


def _contained(root: Path, relative: str) -> Path:
    path = root
    for part in _relative(relative).split("/"):
        path = path / part
        metadata = path.lstat()
        if _is_link(metadata):
            raise VerificationError(f"Linked artifact entry is forbidden: {relative}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise VerificationError(f"Artifact escapes its root: {relative}")
    _file_stat(path)
    return path


def _inventory(root: Path) -> set[str]:
    files: set[str] = set()
    aliases: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                _relative(relative)
                metadata = entry.stat(follow_symlinks=False)
                if _is_link(metadata):
                    raise VerificationError(f"Linked artifact entry is forbidden: {relative}")
                alias = relative.casefold()
                if alias in aliases:
                    raise VerificationError(f"Case-aliased artifact paths: {relative}")
                aliases.add(alias)
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    files.add(relative)
                else:
                    raise VerificationError(f"Nonregular artifact entry: {relative}")
    return files


def _match(document: dict[str, Any], field: str, expected: Any, description: str) -> None:
    if field not in document or document[field] != expected:
        raise VerificationError(f"{description} {field} does not match tracked evidence")


def verify(artifacts: Path, summary: Path = DEFAULT_SUMMARY,
           completion: Path | None = None) -> dict[str, Any]:
    """Verify bytes against a trusted tracked summary; no simulation is run."""
    _, evidence = _json(Path(summary), limit=4 * 1024 * 1024)
    identity = evidence.get("identity")
    commit = evidence.get("source_code_commit")
    if not isinstance(identity, str) or not identity.strip():
        raise VerificationError("Missing evidence identity")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise VerificationError("Invalid evidence source_code_commit")
    manifest_pin = _sha(evidence.get("result_manifest_sha256"), "result_manifest_sha256")
    freeze_pin = _sha(evidence.get("freeze_sha256"), "freeze_sha256")
    root = _root(Path(artifacts))
    manifest_hash, manifest = _json(_contained(root, MANIFEST_NAME))
    if manifest_hash != manifest_pin:
        raise VerificationError("Manifest SHA256 does not match tracked evidence")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise VerificationError("Unsupported manifest schema_version")
    _match(manifest, "identity", identity, "Manifest")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise VerificationError("Manifest files must be a nonempty path/SHA256 mapping")
    aliases: set[str] = set()
    for relative, expected in declared.items():
        _relative(relative)
        if relative.casefold() == MANIFEST_NAME:
            raise VerificationError("Manifest must not declare itself")
        if relative.casefold() in aliases:
            raise VerificationError(f"Case-aliased manifest paths: {relative}")
        aliases.add(relative.casefold())
        _sha(expected, relative)
    if declared.get(FREEZE_NAME) != freeze_pin:
        raise VerificationError("Manifest freeze SHA256 does not match tracked evidence")
    expected_inventory = set(declared) | {MANIFEST_NAME}
    actual_inventory = _inventory(root)
    if actual_inventory != expected_inventory:
        missing = sorted(expected_inventory - actual_inventory)
        unexpected = sorted(actual_inventory - expected_inventory)
        raise VerificationError(f"Artifact inventory mismatch: missing={missing}, unexpected={unexpected}")
    for relative, expected in declared.items():
        actual, _ = _read(_contained(root, relative))
        if actual != expected:
            raise VerificationError(f"Artifact SHA256 mismatch: {relative}")
    freeze_hash, freeze = _json(_contained(root, FREEZE_NAME))
    if freeze_hash != freeze_pin:
        raise VerificationError("Freeze SHA256 changed during verification")
    _match(freeze, "identity", identity, "Freeze")
    _match(freeze, "code_commit", commit, "Freeze")
    if isinstance(freeze.get("config"), dict) and "identity" in freeze["config"]:
        _match(freeze["config"], "identity", identity, "Frozen config")
    completion_hash = None
    if completion is not None:
        completion_pin = _sha(evidence.get("completion_sha256"), "completion_sha256")
        pinned_payload = evidence.get("completion_payload")
        if not isinstance(pinned_payload, dict):
            raise VerificationError("Tracked evidence must pin the complete completion_payload")
        completion_hash, receipt = _json(Path(completion), limit=4 * 1024 * 1024)
        if completion_hash != completion_pin:
            raise VerificationError("Completion SHA256 does not match tracked evidence")
        # Preserve JSON types: Python would otherwise treat False == 0 and 1 == 1.0.
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=False, allow_nan=False)
        if canonical(receipt) != canonical(pinned_payload):
            raise VerificationError("Completion payload does not match tracked evidence")
        for field, expected in (("identity", identity), ("source_code_commit", commit),
                                ("manifest_sha256", manifest_pin), ("freeze_sha256", freeze_pin)):
            _match(receipt, field, expected, "Completion")
        for field in ("source_version", "result_directory_name"):
            if field in evidence:
                _match(receipt, field, evidence[field], "Completion")
    if _inventory(root) != expected_inventory:
        raise VerificationError("Artifact inventory changed during verification")
    return {"passed": True, "identity": identity, "source_code_commit": commit,
            "result_manifest_sha256": manifest_hash, "freeze_sha256": freeze_pin,
            "declared_files_checked": len(declared), "unexpected_files": [],
            "completion_checked": completion is not None, "completion_sha256": completion_hash,
            "scope": "sealed artifact byte identity and inventory only",
            "simulation_executed": False, "telemetry_loaded": False,
            "artifacts_modified": False, "training_attempts_consumed": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="Sealed artifact directory containing manifest.json and freeze.json")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                        help="Trusted tracked evidence summary (default: evidence/research/summary.json)")
    parser.add_argument("--completion", type=Path,
                        help="Optional canonical completion receipt; requires a pinned hash and payload")
    args = parser.parse_args(argv)
    try:
        result = verify(args.artifacts, args.summary, args.completion)
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc), "simulation_executed": False,
                          "artifacts_modified": False, "training_attempts_consumed": 0}, indent=2),
              file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
