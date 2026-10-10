"""Clean-worktree provenance capture for current experiments."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import gzip, importlib.metadata, json, subprocess
from .protocol import ledger_paths

def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    if Path(path).suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def provenance(root):
    if git(root, 'status', '--porcelain'):
        raise RuntimeError('Formal v7 experiments require a clean worktree')
    tracked = git(root, 'ls-files').splitlines()
    files = {name: digest(root/name) for name in tracked
        if Path(name).suffix in ('.py', '.json', '.toml', '.lock', '.txt', '.csv', '.npz')}
    return {'code_commit': git(root, 'rev-parse', 'HEAD'), 'tracked_input_sha256': files,
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'numba', 'llvmlite')},
        'dependency_wheels_sha256': {p.relative_to(root).as_posix(): digest(p)
            for p in sorted((root/'vendor/wheels').glob('*.whl'))},
        'python_version': __import__('sys').version,
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'historical_data_materialized_including_development': True,
        'FP2_ledger_before': {k: p.exists() for k, p in ledger_paths(root).items()}}
