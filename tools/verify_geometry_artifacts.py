"""Read-only, standard-library byte verification of the sealed geometry study."""
import argparse
import json
from pathlib import Path
try:
    from .verify_research_artifacts import (_json, _read, _root, _relative, _contained,
        _inventory, _sha, VerificationError)
except ImportError:
    from verify_research_artifacts import (_json, _read, _root, _relative, _contained,
        _inventory, _sha, VerificationError)

IDENTITY = 'spielberg_geometry_registration_2026_02'
DEFAULT_SUMMARY = Path(__file__).resolve().parents[1]/'evidence/geometry/summary.json'


def verify(artifacts, summary=DEFAULT_SUMMARY):
    """The tracked public summary pins raw bytes, source commit and parent study."""
    root = _root(Path(artifacts))
    _, pin = _json(Path(summary))
    if pin.get('identity') != IDENTITY:
        raise VerificationError('Wrong public geometry identity')
    manifest_sha, manifest = _json(_contained(root, 'manifest.json'))
    freeze_sha, frozen = _json(_contained(root, 'freeze.json'))
    if manifest_sha != _sha(pin['manifest_sha256'], 'manifest') or freeze_sha != _sha(pin['freeze_sha256'], 'freeze'):
        raise VerificationError('Frozen manifest or source pin changed')
    if any(r.get('identity') != IDENTITY for r in (manifest, frozen)):
        raise VerificationError('Artifact identity changed')
    if frozen['code_commit'] != pin['code_commit'] or frozen['parent_identity'] != pin['parent_identity']:
        raise VerificationError('Source or parent study identity changed')
    if frozen['parent_completion_sha256'] != pin['parent_completion_sha256']:
        raise VerificationError('Closed parent completion pin changed')
    files = manifest['files']
    if not isinstance(files, dict) or 'manifest.json' in files:
        raise VerificationError('Invalid artifact inventory')
    aliases = set()
    for name, expected in files.items():
        _relative(name)
        if name.casefold() in aliases:
            raise VerificationError('Case-aliased manifest entries')
        aliases.add(name.casefold())
        path = _contained(root, name)
        digest, _ = _read(path)
        if digest != _sha(expected, name):
            raise VerificationError('Artifact bytes changed: '+name)
        if path.suffix == '.json':
            _json(path)
    if _inventory(root) != set(files)|{'manifest.json'}:
        raise VerificationError('Unexpected or missing artifacts')
    for stage in ('geometry-audit', 'physics-refit', 'control-response'):
        if files.get(stage+'.json') != pin['stage_sha256'].get(stage):
            raise VerificationError('Terminal stage differs from public receipt: '+stage)
    return {'passed': True, 'identity': IDENTITY, 'declared_files_verified': len(files),
            'source_commit': frozen['code_commit'], 'manifest_sha256': manifest_sha,
            'hash_only': True, 'simulation_repeated': False, 'real_world_accuracy_certified': False,
            'shared_budget_restored_or_reset': False, 'FP2_data_loaded': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    try:
        row = verify(args.artifacts, args.summary)
    except (VerificationError, OSError, KeyError, TypeError) as exc:
        row = {'passed': False, 'reason': str(exc)}
    print(json.dumps(row, indent=2))
    return int(not row['passed'])


if __name__ == '__main__': raise SystemExit(main())
