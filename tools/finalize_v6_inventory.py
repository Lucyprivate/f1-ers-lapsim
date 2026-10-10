"""Seal only an external runner log after its stdout stream has closed.

This is an inventory finalization, not a recomputation or experiment amendment.
Every listed calculation must still match the original returned inventory.
The original manifest is preserved byte for byte. Newly created audit sidecars
do not recursively hash themselves into the experiment manifest.

Usage, only AFTER the outer process has exited:
  python finalize_v6_inventory.py --output /absolute/results/name --runner-closed
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


REASON = 'runner stdout closed after experiment returned; no calculation or frozen-input modification'
BACKUP_NAME = 'manifest_at_return.json'
AUDIT_NAME = 'inventory_finalization.json'
TEMP_NAME = '.manifest.inventory-finalization.tmp'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _safe_member(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Manifest members must use nonempty relative POSIX paths')
    path = PurePosixPath(relative)
    parts = relative.split('/')
    if (path.is_absolute() or any(part in ('', '.', '..') for part in parts)
            or any(part.endswith((' ', '.')) for part in parts)
            or any(any(char in part for char in '<>:"|?*') for part in parts)):
        raise ValueError('Unsafe manifest member: ' + relative)
    target = (root / relative).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError('Manifest member escapes output or is not a file: ' + relative)
    return target


def _stable_token(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns


def finalize(output, *, runner_closed=False, log_relative='run.log'):
    """Validate first; change only files[log_relative] and create two sidecars.

    The caller must have observed the outer runner exit. This utility cannot
    infer process completion from a quiet or temporarily unchanged log file.
    A second invocation is rejected rather than rewriting the audit history.
    """
    if runner_closed is not True:
        raise ValueError('Confirm the outer runner has exited before finalizing its log')
    if log_relative != 'run.log':
        raise ValueError('Only the external run.log inventory entry may be finalized')
    root = Path(output).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Output must be an existing directory')
    if (root/'manifest.json').is_symlink():
        raise ValueError('The manifest must not be a symbolic link')
    manifest_path = _safe_member(root, 'manifest.json')
    original_bytes = manifest_path.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    manifest = json.loads(original_bytes.decode('utf-8-sig'), object_pairs_hook=_unique_pairs)
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
        raise ValueError('Expected a manifest object with a files mapping')
    if manifest.get('schema_version') != 6:
        raise ValueError('This utility only finalizes a v6 inventory')
    files = manifest['files']
    if 'manifest.json' in files:
        raise ValueError('The returned manifest must not recursively inventory itself')
    if log_relative not in files:
        raise ValueError('The runner log must already be listed in the returned manifest')
    sidecar_names = {BACKUP_NAME, AUDIT_NAME, TEMP_NAME}
    if sidecar_names.intersection(files):
        raise ValueError('A prior finalization sidecar is already listed')
    for name in sidecar_names:
        if (root/name).exists():
            raise ValueError('Finalization output already exists: ' + name)
    log_path = _safe_member(root, log_relative)
    log_token = _stable_token(log_path)
    seen = set()
    verified = []
    for relative, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('Invalid SHA-256 value: ' + str(relative))
        target = _safe_member(root, relative)
        normalized = str(target).casefold()
        if normalized in seen:
            raise ValueError('Multiple manifest members resolve to the same file')
        seen.add(normalized)
        actual = _sha256(target)
        if relative != log_relative:
            if actual != expected:
                raise ValueError('A calculation or frozen artifact changed: ' + relative)
            verified.append(relative)
    final_log_hash = _sha256(log_path)
    if _stable_token(log_path) != log_token:
        raise ValueError('Runner log changed during verification; await actual process exit')
    if manifest_path.read_bytes() != original_bytes:
        raise ValueError('Manifest changed during verification')
    before_log_hash = files[log_relative]
    files[log_relative] = final_log_hash
    final_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    final_hash = hashlib.sha256(final_bytes).hexdigest()
    audit = {'schema_version': 1, 'operation': 'external_runner_log_inventory_finalization',
        'finalized_at_utc': datetime.now(timezone.utc).isoformat(),
        'reason': REASON, 'runner_closed_confirmed_by_caller': True,
        'manifest_sha256_before': original_hash, 'manifest_sha256_after': final_hash,
        'log_relative_path': log_relative, 'log_sha256_before': before_log_hash,
        'log_sha256_after': final_log_hash, 'only_changed_manifest_field': 'files/' + log_relative,
        'other_listed_files_verified': len(verified), 'other_listed_files': verified,
        'original_manifest_copy': BACKUP_NAME, 'original_manifest_copy_sha256': original_hash,
        'new_sidecars_excluded_from_manifest': [BACKUP_NAME, AUDIT_NAME],
        'calculation_reexecuted': False, 'calculation_modified': False,
        'frozen_input_modified': False}
    # All validation has finished before any write. Exclusive sidecar creation
    # prevents a second run from replacing the original returned inventory.
    with (root/BACKUP_NAME).open('xb') as stream:
        stream.write(original_bytes)
        stream.flush(); os.fsync(stream.fileno())
    temporary = root/TEMP_NAME
    with temporary.open('xb') as stream:
        stream.write(final_bytes)
        stream.flush(); os.fsync(stream.fileno())
    if _stable_token(log_path) != log_token or _sha256(log_path) != final_log_hash:
        temporary.unlink()
        raise ValueError('Runner log changed before commit; original manifest preserved')
    if manifest_path.read_bytes() != original_bytes:
        temporary.unlink()
        raise ValueError('Manifest changed before commit; original manifest preserved')
    temporary.replace(manifest_path)
    with (root/AUDIT_NAME).open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runner-closed', action='store_true',
        help='Confirm the outer process and its stdout redirection have ended')
    args = parser.parse_args()
    audit = finalize(args.output, runner_closed=args.runner_closed)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
