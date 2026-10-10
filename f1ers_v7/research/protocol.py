"""A durable finite-run identity; experimental starts cannot be reset by a clone."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import sys
import platform
from .._core.provenance import git, digest
from .data import input_files
from .._core.protocol import CANONICAL_LEDGER_ROOT

IDENTITY = 'spielberg_open_source_control_2026_01'
CANONICAL_RESEARCH_LEDGER = CANONICAL_LEDGER_ROOT.parent/'research-control'


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def freeze(root, data_root, output, ledger_root):
    root, output = Path(root).resolve(), Path(output).resolve()
    if Path(ledger_root).resolve()!=CANONICAL_RESEARCH_LEDGER.resolve():
        raise RuntimeError('Use the canonical shared external research-control ledger; clones cannot reset the budget')
    if git(root, 'status', '--porcelain'):
        raise RuntimeError('Formal research requires a clean worktree')
    if output.exists():
        raise RuntimeError('Existing result directory cannot be overwritten')
    config = json.loads((root/'config/research/experiment.json').read_text())
    deps = dependency_identity()
    row = {'identity': IDENTITY, 'code_commit': git(root,'rev-parse','HEAD'),
           'tracked_sha256': {n: digest(root/n) for n in git(root,'ls-files').splitlines()},
           'config': config, 'data_sha256': input_files(data_root), 'dependencies': deps,
           'python':python_identity(),
           'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'FP2_accessed': False,
           'original_data_root': str(Path(data_root).resolve())}
    write_exclusive(Path(ledger_root)/IDENTITY/'run_claim.json', row)
    write_exclusive(output/'freeze.json', row)
    return row


def dependency_identity():
    deps={}
    for name in ('numpy','scipy','numba','llvmlite','torch','stable-baselines3','gymnasium','casadi'):
        try:
            dist = importlib.metadata.distribution(name)
            records = [f for f in (dist.files or []) if str(f).endswith('.dist-info/RECORD')]
            deps[name] = {'version': dist.version, 'installed_RECORD_sha256':
                          digest(dist.locate_file(records[0])) if records else None}
        except importlib.metadata.PackageNotFoundError: deps[name] = None
    return deps


def python_identity():
    return {'version':sys.version,'implementation':platform.python_implementation(),
            'executable_sha256':digest(sys.executable)}


class EpisodeBudget:
    """An attempted episode is spent before reset, even when interrupted."""
    def __init__(self, ledger_root, maximum=120):
        self.root = Path(ledger_root)/IDENTITY/'episodes'
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximum = maximum

    def claim(self, backend, seed, condition, episode):
        if backend not in ('numpy','sb3') or seed not in (2026,2027,2028) or condition not in ('cold','warm') or not 1 <= episode <= 10:
            raise ValueError('Attempt outside preregistered 2x3x2x10 budget')
        if len(list(self.root.glob('*.json'))) >= self.maximum:
            raise RuntimeError('Attempted episode budget exhausted')
        key = f'{backend}_{seed}_{condition}_{episode:02d}'
        write_exclusive(self.root/(key+'.json'), {'backend':backend,'seed':seed,
            'condition':condition,'episode':episode,'started_at_utc':datetime.now(timezone.utc).isoformat()})
        return key

    def counts(self):
        paths = list(self.root.glob('*.json'))
        return {'attempted':len(paths), 'maximum':self.maximum,
                'per_backend': {b:sum(p.name.startswith(b+'_') for p in paths) for b in ('numpy','sb3')}}
