"""Verify the current public file inventory without private data or dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


MANIFEST = 'docs/PUBLIC_SOURCE_INVENTORY.json'


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('package_version') != '0.7.1':
        raise ValueError('Unexpected public inventory schema or package version')
    files = manifest['files']
    problems = []
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append({'path': relative, 'problem': 'missing or outside repository'})
            continue
        payload = path.read_bytes()
        if len(payload) != expected['bytes'] or hashlib.sha256(payload).hexdigest() != expected['sha256']:
            problems.append({'path': relative, 'problem': 'content changed'})
    git_check = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                               capture_output=True, text=True, check=False)
    tracked_inventory_checked = False
    if git_check.returncode == 0 and Path(git_check.stdout.strip()).resolve() == root:
        tracked = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'],
                                 capture_output=True, check=True).stdout.decode('utf-8').split('\0')
        actual = {name for name in tracked if name}
        expected_paths = set(files) | {MANIFEST}
        if actual != expected_paths:
            problems.append({'problem': 'tracked file inventory differs',
                             'unexpected': sorted(actual - expected_paths),
                             'missing': sorted(expected_paths - actual)})
        tracked_inventory_checked = True
    return {'passed': not problems, 'package_version': manifest['package_version'],
            'file_count': len(files) + 1, 'hashes_checked': len(files),
            'tracked_inventory_checked': tracked_inventory_checked,
            'private_data_accessed': False, 'problems': problems}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
