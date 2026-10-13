"""Once-only geometry study and durable cross-clone attempt accounting."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from hashlib import sha256
import json
import re

from .._core.provenance import digest, git, read
from .._core.protocol import CANONICAL_LEDGER_ROOT
from .data import input_files
from .protocol import (CANONICAL_RESEARCH_LEDGER, dependency_identity,
                       python_identity, write_exclusive)

IDENTITY = 'spielberg_geometry_registration_2026_02'
PARENT = 'spielberg_open_source_control_2026_01'
PARENT_COMPLETION_SHA256 = 'e04e38f7891920c27c3dcd087f59c015b041094420efd782375103bb7a47c0c0'


def validate_config(config):
    fixed = {'identity': IDENTITY, 'parent_identity': PARENT,
        'calibration_laps': [3, 4, 5, 6], 'development_laps': [7, 9],
        'vehicle_mass_kg': 800., 'ice_power_kw': 400., 'energy_window_mj': [2., 4.],
        'regularization_lengths_m': [20., 40., 80.], 'fit_node_spacing_max_m': 2.,
        'registration_phase_halfwidth_m': 100., 'registration_clock_halfwidth_s': 1.,
        'registration_units_per_m_bounds': [8., 12.], 'registration_max_evaluations': 120,
        'sobol_seed': 2026, 'sobol_points': 64, 'minimum_viable_fit_starts': 3,
        'fit_max_iterations_per_start': 300, 'fit_max_evaluations_per_start': 600,
        'response_single_lap_calls_maximum': 132, 'response_blocks': 16,
        'response_amplitude': .025, 'response_half_amplitude': .0125,
        'response_selected_blocks_per_boundary_maximum': 4,
        'new_SAC_attempts_maximum': 0, 'full_teacher_search_allowed': False,
        'FP2_sealed': True, 'minimum_position_improvement_fraction': .01,
        'position_selection_tie_fraction': .01, 'parameter_sensitivity_evaluations_maximum': 10,
        'internal_refinements': [[.02, 1.], [.01, .5], [.005, .25]],
        'output_spacings_m': [10., 5., 2.5]}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError('Preregistered study field changed: '+key)
    if config.get('accuracy_limits') != {'mean_speed_rmse_kmh': 15., 'max_abs_lap_error_pct': 2., 'max_abs_sector_error_s': 1.}:
        raise ValueError('Formal accuracy thresholds cannot change')
    return config


def ledger_snapshot(path):
    """Hash existing receipts, without loading any telemetry or creating a claim."""
    path = Path(path)
    return {p.relative_to(path).as_posix(): digest(p)
            for p in sorted(path.rglob('*')) if p.is_file()}


def ensure_study(root, data_root, output, ledger_root):
    root, output, data_root = map(lambda p: Path(p).resolve(), (root, output, data_root))
    ledger = Path(ledger_root).resolve()
    if ledger != CANONICAL_RESEARCH_LEDGER.resolve():
        raise RuntimeError('Use the canonical external ledger; clones cannot reset this study')
    if output.parent != (root/'results').resolve():
        raise ValueError('Output must be an immediate repository/results child')
    if git(root, 'status', '--porcelain'):
        raise RuntimeError('Formal stages require a clean worktree')
    parent = ledger/PARENT/'completion.json'
    if digest(parent) != PARENT_COMPLETION_SHA256:
        raise RuntimeError('Closed parent-study completion identity changed')
    config = validate_config(read(root/'config/research/geometry_registration.json'))
    if config['identity'] != IDENTITY or config['parent_identity'] != PARENT:
        raise ValueError('Frozen study identity changed')
    claim = ledger/IDENTITY/'run_claim.json'
    dependencies, python = dependency_identity(), python_identity()
    if claim.exists():
        frozen = read(claim)
        if not (output/'freeze.json').exists() or read(output/'freeze.json') != frozen:
            raise RuntimeError('Shared identity already claimed by another output or clone')
        if frozen['code_commit'] != git(root, 'rev-parse', 'HEAD'):
            raise RuntimeError('Continuation requires the original frozen code commit')
        if frozen['data_sha256'] != input_files(data_root):
            raise RuntimeError('Frozen historical inputs changed')
        if frozen['dependencies'] != dependencies or frozen['python'] != python:
            raise RuntimeError('Frozen runtime changed')
        if frozen['FP2_ledger_sha256'] != ledger_snapshot(CANONICAL_LEDGER_ROOT):
            raise RuntimeError('FP2 access state changed during the sealed study')
        return frozen
    if output.exists():
        raise RuntimeError('Existing output cannot be overwritten')
    value = {'identity': IDENTITY, 'parent_identity': PARENT,
             'parent_completion_sha256': PARENT_COMPLETION_SHA256,
             'code_commit': git(root, 'rev-parse', 'HEAD'),
             'tracked_sha256': {n: digest(root/n) for n in git(root, 'ls-files').splitlines()},
             'config': config, 'data_sha256': input_files(data_root),
             'dependencies': dependencies, 'python': python,
             'original_data_root': str(data_root),
             'FP2_ledger_sha256': ledger_snapshot(CANONICAL_LEDGER_ROOT),
             'timestamp_utc': datetime.now(timezone.utc).isoformat(),
             'FP2_accessed': False, 'SAC_attempts_allowed': 0}
    write_exclusive(claim, value)
    write_exclusive(output/'freeze.json', value)
    return value


def start_stage(stage, output):
    if stage not in ('geometry-audit', 'physics-refit', 'control-response'):
        raise ValueError('Unknown bounded stage')
    path = CANONICAL_RESEARCH_LEDGER/IDENTITY/(stage+'_started.json')
    write_exclusive(path, {'stage': stage, 'output': str(Path(output).resolve()),
                          'started_at_utc': datetime.now(timezone.utc).isoformat(),
                          'interruption_consumes_stage': True})


class AttemptBudget:
    """Reserve an exclusive key and a numbered slot before invoking the simulator.

    A crash after either reservation consumes that reservation. A concurrent clone
    cannot reuse a key or exceed the fixed number of exclusive slots.
    """
    def __init__(self, directory, maximum):
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError('Positive integer budget required')
        self.root, self.maximum = Path(directory), maximum
        self.root.mkdir(parents=True, exist_ok=True)
        identity = self.root/'budget.json'
        if not identity.exists():
            try:
                write_exclusive(identity, {'maximum': maximum})
            except FileExistsError:
                pass
        if read(identity) != {'maximum': maximum}:
            raise RuntimeError('Existing budget cannot be expanded')

    def claim(self, key, details=None):
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', key):
            raise ValueError('Safe unique attempt key required')
        receipt = {'key': key, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                   'details': details or {}, 'failure_or_interruption_spends_attempt': True}
        key_path = self.root/'keys'/(key+'.json')
        if key_path.exists():
            raise FileExistsError('Attempt key already spent: '+key)
        for ordinal in range(1, self.maximum+1):
            try:
                write_exclusive(self.root/'slots'/f'{ordinal:04d}.json', receipt)
            except FileExistsError:
                continue
            # Allocate capacity first. A crash here still spends a numbered slot.
            # A racing duplicate may spend another slot but cannot run twice.
            write_exclusive(key_path, {**receipt, 'ordinal': ordinal})
            return ordinal
        raise RuntimeError('Fixed attempt budget exhausted')

    def counts(self):
        return {'attempted': len(list((self.root/'slots').glob('*.json'))),
                'reserved_keys': len(list((self.root/'keys').glob('*.json'))),
                'maximum': self.maximum}


def seal_stage(output, stage, result):
    write_exclusive(Path(output)/(stage+'.json'), result)
    write_exclusive(CANONICAL_RESEARCH_LEDGER/IDENTITY/(stage+'_completed.json'),
                    {'stage': stage, 'result_sha256': digest(Path(output)/(stage+'.json'))})


def final_manifest(output):
    output = Path(output)
    value = {'identity': IDENTITY, 'FP2_accessed': False,
             'files': {p.relative_to(output).as_posix(): digest(p)
                       for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    (output/'manifest.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return value


def verify_study(root, data_root, output):
    """Read frozen scores and compare source, input, runtime and shared receipts."""
    root, output = Path(root).resolve(), Path(output).resolve()
    frozen, manifest = read(output/'freeze.json'), read(output/'manifest.json')
    validate_config(frozen['config'])
    if manifest['identity'] != IDENTITY or frozen['identity'] != IDENTITY:
        raise ValueError('Wrong study identity')
    failures = []
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '\\' in name or ':' in name or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        path = output/name
        if path.is_symlink() or not path.resolve().is_relative_to(output) or not path.is_file() or digest(path) != expected:
            failures.append(name)
    inventory = {p.relative_to(output).as_posix() for p in output.rglob('*')
                 if p.is_file() and p != output/'manifest.json'}
    source_equal = all(digest(root/n) == h for n, h in frozen['tracked_sha256'].items()
                       if n.endswith('.py') or n.startswith('config/'))
    import subprocess
    tree_equal = all(sha256(subprocess.check_output(['git', 'show', frozen['code_commit']+':'+n], cwd=root)).hexdigest() == h
                     for n, h in frozen['tracked_sha256'].items())
    claims_equal = read(CANONICAL_RESEARCH_LEDGER/IDENTITY/'run_claim.json') == frozen
    parent_equal = digest(CANONICAL_RESEARCH_LEDGER/PARENT/'completion.json') == PARENT_COMPLETION_SHA256
    stage_receipts = {}
    for stage in ('geometry-audit', 'physics-refit', 'control-response'):
        start = CANONICAL_RESEARCH_LEDGER/IDENTITY/(stage+'_started.json')
        complete = CANONICAL_RESEARCH_LEDGER/IDENTITY/(stage+'_completed.json')
        stage_receipts[stage] = bool(start.is_file() and complete.is_file() and
            read(start).get('stage') == stage and read(complete).get('stage') == stage and
            (output/(stage+'.json')).is_file() and
            read(complete).get('result_sha256') == digest(output/(stage+'.json')))
    inputs_equal = frozen['data_sha256'] == input_files(data_root)
    runtime_equal = frozen['dependencies'] == dependency_identity() and frozen['python'] == python_identity()
    fp2_equal = frozen['FP2_ledger_sha256'] == ledger_snapshot(CANONICAL_LEDGER_ROOT)
    all_stages = all((output/(stage+'.json')).exists() for stage in ('geometry-audit', 'physics-refit', 'control-response'))
    return {'passed': bool(not failures and inventory == set(manifest['files']) and source_equal and tree_equal and
                           claims_equal and parent_equal and all(stage_receipts.values()) and inputs_equal and runtime_equal and fp2_equal and all_stages),
            'artifact_hash_failures': failures, 'inventory_equal': inventory == set(manifest['files']),
            'computational_source_equal': source_equal, 'frozen_git_tree_equal': tree_equal,
            'canonical_claim_equal': claims_equal, 'input_hashes_equal': inputs_equal,
            'closed_parent_completion_equal': parent_equal,
            'shared_stage_receipts_match': stage_receipts,
            'runtime_equal': runtime_equal, 'FP2_ledger_unchanged': fp2_equal,
            'all_stages_terminal': all_stages, 'FP2_accessed': False,
            'optimization_and_training_repeated': False, 'saved_scores_read_only': True,
            'original_code_commit': frozen['code_commit']}
