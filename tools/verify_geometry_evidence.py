"""Read-only byte verification of sealed geometry evidence, including JSON lists.

Standard library only. The tracked public summary is the trust anchor; using
--summary changes that anchor. This verifies saved bytes and inventory, never
simulation correctness, real SOC, full FIA compliance, or prediction accuracy.
The original frozen verifier is preserved; this additive entry accepts any JSON
root for leaf artifacts while requiring objects for the manifest, freeze and
public summary. It does not execute the simulator or modify an access ledger.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re

try:
    from .verify_research_artifacts import (
        _read, _root, _relative, _contained, _inventory, _sha, VerificationError)
except ImportError:
    # Explicit sibling loading also works with ``python -I``: only the safe
    # standard-library file primitives are imported, never the simulator.
    _spec = importlib.util.spec_from_file_location(
        'geometry_evidence_safe_files', Path(__file__).with_name('verify_research_artifacts.py'))
    _safe_files = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_safe_files)
    _read, _root, _relative, _contained, _inventory, _sha, VerificationError = (
        getattr(_safe_files, name) for name in
        ('_read', '_root', '_relative', '_contained', '_inventory', '_sha', 'VerificationError'))


IDENTITY = 'spielberg_geometry_registration_2026_02'
STAGES = ('geometry-audit', 'physics-refit', 'control-response')
DEFAULT_SUMMARY = Path(__file__).resolve().parents[1] / 'evidence/geometry/summary.json'
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError('Duplicate JSON key: ' + repr(key))
        result[key] = value
    return result


def _nonfinite(value):
    raise VerificationError('Non-finite JSON number: ' + value)


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _nonfinite(value)
    return number


def _json_value(path, *, require_object=False, limit=16 * 1024 * 1024):
    """Hash and strictly parse the same safe-file read; leaves may be any root."""
    digest, payload = _read(path, limit=limit)
    try:
        document = json.loads(payload.decode('utf-8'), object_pairs_hook=_unique_object,
                              parse_constant=_nonfinite, parse_float=_finite_float)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError('Invalid UTF-8 JSON: ' + str(path) + ': ' + str(exc)) from exc
    if require_object and not isinstance(document, dict):
        raise VerificationError('Expected a JSON object: ' + str(path))
    return digest, document


def verify(artifacts, summary=DEFAULT_SUMMARY):
    """Check the sealed inventory against the public summary without simulation."""
    root = _root(Path(artifacts))
    _, pin = _json_value(Path(summary), require_object=True, limit=4 * 1024 * 1024)
    if pin.get('identity') != IDENTITY:
        raise VerificationError('Wrong public geometry identity')
    commit = pin.get('code_commit')
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise VerificationError('Invalid public source commit')
    parent = pin.get('parent_identity')
    if not isinstance(parent, str) or not parent.strip():
        raise VerificationError('Missing public parent study identity')
    parent_completion = _sha(pin.get('parent_completion_sha256'), 'parent completion')
    manifest_pin = _sha(pin.get('manifest_sha256'), 'manifest')
    freeze_pin = _sha(pin.get('freeze_sha256'), 'freeze')
    stage_pins = pin.get('stage_sha256')
    if not isinstance(stage_pins, dict) or set(stage_pins) != set(STAGES):
        raise VerificationError('Public summary must pin all three terminal stages')
    for stage in STAGES:
        _sha(stage_pins[stage], stage)

    manifest_sha, manifest = _json_value(_contained(root, 'manifest.json'), require_object=True)
    freeze_sha, frozen = _json_value(_contained(root, 'freeze.json'), require_object=True)
    if manifest_sha != manifest_pin or freeze_sha != freeze_pin:
        raise VerificationError('Frozen manifest or source pin changed')
    if any(row.get('identity') != IDENTITY for row in (manifest, frozen)):
        raise VerificationError('Artifact identity changed')
    if frozen.get('code_commit') != commit or frozen.get('parent_identity') != parent:
        raise VerificationError('Source or parent study identity changed')
    if frozen.get('parent_completion_sha256') != parent_completion:
        raise VerificationError('Closed parent completion pin changed')

    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise VerificationError('Invalid artifact inventory')
    aliases = set()
    for name, expected in files.items():
        _relative(name)
        if name.casefold() == 'manifest.json':
            raise VerificationError('Manifest must not declare itself')
        if name.casefold() in aliases:
            raise VerificationError('Case-aliased manifest entries')
        aliases.add(name.casefold())
        _sha(expected, name)
    if files.get('freeze.json') != freeze_pin:
        raise VerificationError('Manifest freeze pin differs from public evidence')
    expected_inventory = set(files) | {'manifest.json'}
    if _inventory(root) != expected_inventory:
        raise VerificationError('Unexpected or missing artifacts')
    json_leaves = 0
    for name, expected in files.items():
        path = _contained(root, name)
        if path.suffix.casefold() == '.json':
            digest, _ = _json_value(path)
            json_leaves += 1
        else:
            digest, _ = _read(path)
        if digest != expected:
            raise VerificationError('Artifact bytes changed: ' + name)
    for stage in STAGES:
        if files.get(stage + '.json') != stage_pins[stage]:
            raise VerificationError('Terminal stage differs from public receipt: ' + stage)
    if _inventory(root) != expected_inventory:
        raise VerificationError('Artifact inventory changed during verification')
    if 'manifest_file_count' in pin and (
            type(pin['manifest_file_count']) is not int or pin['manifest_file_count'] != len(files)):
        raise VerificationError('Public manifest file count differs from the saved inventory')
    return {
        'passed': True, 'identity': IDENTITY, 'declared_files_verified': len(files),
        'json_leaf_artifacts_checked': json_leaves, 'source_commit': commit,
        'manifest_sha256': manifest_sha, 'freeze_sha256': freeze_sha,
        'scope': 'sealed artifact byte identity and strict JSON/inventory only',
        'hash_only': True, 'simulation_repeated': False, 'telemetry_loaded': False,
        'real_world_accuracy_certified': False, 'artifacts_modified': False,
        'shared_budget_restored_or_reset': False, 'FP2_data_loaded': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    try:
        result = verify(args.artifacts, args.summary)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {'passed': False, 'reason': str(exc), 'simulation_repeated': False,
                  'artifacts_modified': False, 'shared_budget_restored_or_reset': False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
