"""A fixed global control restriction comparison after immutable B selection.

This module scores calibration diagnostics only.  The caller freezes selection
before any historical development score or final numerical audit.  No extra
SAC training, residual-based curve registration or model fitting is performed.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
from hashlib import sha256

import numpy as np

from f1ers_v4.data import load_local_dataset
from f1ers_v4.protocol import ledger_paths
from f1ers_v6.learning import _write
from f1ers_v6.metrics import score
from f1ers_v6.rules import RuleConfig
from f1ers_v6.teacher import BlockController
from f1ers_v7.experiment import controller_from, digest, provenance, qualify, read
from f1ers_v7.simulator import ReachableSimulator


def canonical_controller_hash(spec):
    return sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True).encode('ascii')).hexdigest()


def safe_file(base, name):
    name = str(name)
    path = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or path.is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe frozen inventory path')
    resolved = (base/name).resolve()
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        raise ValueError('Frozen inventory file missing or outside prior result')
    return resolved


def validate_config(config):
    fixed = {'schema_version': 7, 'kind': 'fixed_global_tail_guard_comparison',
        'prior_name': 'austria_2026_v7B', 'geometry_margin_s': 1.,
        'initial_energy_mj': 2., 'tail_guard_choices_m': [350., 700., 1000.],
        'physical_laps_per_case': 6, 'steady_laps': [5, 6],
        'primary_controller': 'immutable_B_selected_controller',
        'fallback_if_all_primary_cases_fail': True, 'fallback_values': [.5],
        'fallback_action_space': 'mode_aware', 'maximum_cases': 6,
        'maximum_physical_lap_integrations': 36, 'new_SAC_episodes': 0,
        'calibration_laps': [3, 4, 5, 6], 'development_score_inside_audit': False,
        'observed_residual_used_in_selection': False,
        'E4_single_lap_sensitivity_in_selection': False,
        'vehicle_mass_kg': 800., 'ice_power_kw': 400.,
        'geometry_or_model_parameter_fit': False, 'FP2_access': 'sealed'}
    if any(config.get(name) != expected for name, expected in fixed.items()):
        raise ValueError('Fixed tail-guard experiment scope changed')
    for field in ('prior_manifest_sha256', 'prior_selection_sha256', 'prior_controller_spec_sha256'):
        value = config.get(field, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Final B input hashes must be bound before starting C')
    value = config.get('prior_source_commit', '')
    if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Full final B source commit must be frozen')
    if (not isinstance(config.get('energy_notice_s'), (int, float))
            or not np.isfinite(config['energy_notice_s']) or config['energy_notice_s'] <= 0):
        raise ValueError('B winner energy notice must be frozen')
    if (not isinstance(config.get('initial_speed_mps'), (int, float))
            or not np.isfinite(config['initial_speed_mps']) or config['initial_speed_mps'] <= 0):
        raise ValueError('B winner cold initial speed must be frozen')


def qualification_gate_passed(qualification):
    """Recompute all original six-lap gates from saved accepted boundaries."""
    summaries = qualification.get('lap_summaries', [])
    proxies = qualification.get('all_cold_and_warm_proxy_gates', [])
    crossings = qualification.get('crossings', [])
    if (qualification.get('physical_laps') != 6 or len(summaries) != 6
            or len(proxies) != 6 or not all(proxies) or len(crossings) != 5
            or not qualification.get('physical_control_line_continuity', False)):
        return False
    required = ('feasible', 'completed_lap', 'execution_feasible', 'public_rules_passed',
                'sustainable', 'periodic_speed_closed', 'dc_boundary_closed',
                'rule_state_closed', 'reachable_request_gate_passed')
    for summary in summaries:
        if not all(summary.get(name, False) for name in required):
            return False
        values = [summary.get(name, np.nan) for name in (
            'energy_start_mj', 'energy_end_mj', 'energy_balance_residual_mj', 'lap_time_s')]
        if not np.isfinite(values).all() or values[1] < values[0]-1e-8 or abs(values[2]) > 1e-7:
            return False
    for crossing in crossings:
        values = [crossing.get(name, np.nan) for name in (
            'speed_difference_mps', 'energy_difference_mj', 'dc_difference_kw')]
        if (not np.isfinite(values).all() or any(abs(value) >= 1e-10 for value in values)
                or not crossing.get('rule_state_carried', False)):
            return False
    return True


def tail_guard_audit(root, *, prior_name='austria_2026_v7B', name='austria_2026_v7C'):
    """Compute exactly three primary rows and at most three fixed fallbacks.

    Returns selected controller/spec/directory/lap6 for the calling frozen final
    scorer.  It never runs development scoring or a new learning episode.
    """
    root = Path(root).resolve()
    if any('/' in n or '\\' in n or n in ('', '.', '..') for n in (prior_name, name)):
        raise ValueError('One safe result directory name required')
    config = read(root/'config/v7/tail_guard.json')
    validate_config(config)
    if prior_name != config['prior_name']:
        raise ValueError('Tail audit is bound to immutable B')
    output, prior = root/'results'/name, root/'results'/prior_name
    if output.exists():
        raise RuntimeError('Tail comparison cannot overwrite or rescore an existing experiment')
    freeze = provenance(root)
    if digest(prior/'manifest.json') != config['prior_manifest_sha256']:
        raise ValueError('Final B manifest changed')
    manifest = read(prior/'manifest.json')
    for relative, expected in manifest['files'].items():
        if digest(safe_file(prior, relative)) != expected:
            raise ValueError('Frozen B artifact changed: '+relative)
    if digest(prior/'selection.json') != config['prior_selection_sha256']:
        raise ValueError('Frozen B selection changed')
    previous = read(prior/'experiment.json')
    if previous['source_commit'] != config['prior_source_commit']:
        raise ValueError('Frozen B source commit changed')
    winner = read(prior/'selection.json')['winner']
    if not winner['qualification']['passed']:
        raise ValueError('Tail comparison requires a six-lap certified prior controller')
    if canonical_controller_hash(winner['controller']) != config['prior_controller_spec_sha256']:
        raise ValueError('Frozen B controller specification changed')
    if (winner['energy_notice_s'] != config['energy_notice_s'] or
            winner['initial_speed_mps'] != config['initial_speed_mps']):
        raise ValueError('Prior notice/initial state differs from frozen tail protocol')
    output.mkdir(parents=True)
    shutil.copyfile(root/'config/v7/tail_guard.json', output/'frozen_protocol.json')
    _write(output/'freeze.json', freeze)
    _write(output/'started.json', {'source_commit': freeze['code_commit'],
        'prior_manifest_sha256': config['prior_manifest_sha256'], 'protocol': config})
    spec = json.loads(json.dumps(winner['controller']))
    input_hashes = {'manifest.json': config['prior_manifest_sha256'],
                    'selection.json': config['prior_selection_sha256']}
    for filename in ('manifest.json', 'selection.json'):
        (output/'inputs').mkdir(exist_ok=True)
        shutil.copyfile(prior/filename, output/'inputs'/filename)
    if spec['kind'] == 'policy':
        source = safe_file(prior, spec['checkpoint'])
        if (spec['checkpoint'] not in manifest['files']
                or digest(source) != manifest['files'][spec['checkpoint']]):
            raise ValueError('Selected policy checkpoint is not bound by final B inventory')
        target = output/'inputs'/'selected_controller_v7.npz'
        shutil.copyfile(source, target)
        input_hashes[source.relative_to(prior).as_posix()] = digest(source)
        spec['checkpoint'] = target.relative_to(output).as_posix()
    _write(output/'inputs/input_bindings.json', {
        'source_controller_spec': winner['controller'], 'portable_controller_spec': spec,
        'prior_input_sha256': input_hashes, 'controller_weights_changed': False})
    data = load_local_dataset(root)
    if not data['audit']['data_gate_passed']:
        raise ValueError('Historical data quality gate failed')
    rows, trajectories, simulators = [], {}, {}

    def run_family(family, controller_spec):
        for guard in config['tail_guard_choices_m']:
            sim = ReachableSimulator(data['track'], data['rules'],
                rule_config=RuleConfig(tail_guard_m=guard),
                margin_s=config['geometry_margin_s'], energy_notice_s=config['energy_notice_s'])
            controller = controller_from(controller_spec, output, sim)
            label = f'{family}_tail_{guard:g}'
            directory = output/'qualification'/label
            qualification, laps = qualify(sim, controller, controller_spec['action_space'],
                directory, config['initial_speed_mps'], data)
            independently_passed = qualification_gate_passed(qualification)
            if bool(qualification['passed']) != independently_passed:
                _write(directory/'qualification_disagreement.json', {
                    'reported_passed': qualification['passed'],
                    'independently_recomputed_passed': independently_passed,
                    'candidate_rejected': True})
                raise RuntimeError('Saved six-lap qualification disagrees with independent gates')
            formal_objective = float(np.mean([row['lap_time_s']
                for row in qualification['lap_summaries'][-2:]])) if independently_passed else None
            diagnostic = score(laps[-1], data, 'calibration')
            _write(directory/'last_lap_calibration_diagnostic.json', diagnostic)
            row = {'name': label, 'family': family, 'tail_guard_m': guard,
                'energy_notice_s': config['energy_notice_s'],
                'initial_speed_mps': config['initial_speed_mps'], 'controller': controller_spec,
                'action_space': controller_spec['action_space'], 'qualification': qualification,
                'result_directory': directory.relative_to(output).as_posix(),
                'independent_qualification_gate_passed': independently_passed,
                'formal_objective_s': formal_objective,
                'complete_failed_last_two_mean_is_diagnostic_only': not independently_passed,
                'calibration_diagnostic': diagnostic,
                'controller_weights_or_block_values_fixed': True,
                'policy_observation_response_can_change': controller_spec['kind'] == 'policy',
                'extra_energy_scenario_in_selection': False}
            rows.append(row)
            trajectories[label], simulators[label] = laps[-1], sim
            _write(output/'tail_guard_rows.json', rows)
            print(json.dumps({'tail_guard_case': label, 'passed': qualification['passed'],
                'diagnostic_last_two_mean_s': qualification['last_two_mean_time_s'],
                'calibration_RMSE': diagnostic['aggregate'].get('mean_speed_rmse_kmh')},
                ensure_ascii=False), flush=True)

    run_family('B_selected', spec)
    if not any(row['qualification']['passed'] for row in rows):
        fallback = BlockController(config['fallback_values'], config['fallback_action_space'])
        run_family('fixed_fallback', fallback.to_dict())
    eligible = [row for row in rows if row['independent_qualification_gate_passed']]
    selected = min(eligible, key=lambda row: row['formal_objective_s']) if eligible else None
    _write(output/'selection.json', {'winner': selected,
        'source_commit': freeze['code_commit'], 'prior_manifest_sha256': config['prior_manifest_sha256'],
        'observed_residual_in_selection': False,
        'ranking': 'all six original/public/engineering/nondeficit/carry gates then last-two mean time',
        'fidelity_result_can_fail_even_when_control_passes': True})
    if selected is not None:
        _write(output/'selected_calibration.json', selected['calibration_diagnostic'])
    unchanged = all(digest(safe_file(prior, relative)) == expected
                    for relative, expected in manifest['files'].items())
    unchanged = unchanged and digest(prior/'manifest.json') == config['prior_manifest_sha256']
    ledger_same = {key: path.exists() for key, path in ledger_paths(root).items()} == freeze['FP2_ledger_before']
    tracked_same = all(digest(root/name) == expected
        for name, expected in freeze['tracked_input_sha256'].items())
    if not (unchanged and ledger_same and tracked_same):
        raise RuntimeError('Frozen source, prior artifact or shared FP2 ledger changed')
    result = {'schema_version': 7, 'kind': 'fixed_global_tail_guard_comparison',
        'source_commit': freeze['code_commit'], 'prior_source_commit': previous['source_commit'],
        'prior_manifest_sha256': config['prior_manifest_sha256'],
        'rows': rows, 'winner': selected, 'new_SAC_episodes': 0,
        'maximum_cases': 6, 'actual_cases': len(rows),
        'actual_physical_laps': sum(row['qualification']['physical_laps'] for row in rows),
        'historical_development_materialized_by_inherited_loader': True,
        'development_scored_inside_audit': False, 'FP2_accessed': False,
        'E4_single_lap_sensitivity_in_selection': False,
        'failed_shorter_guard_does_not_prove_long_guard_necessary_for_other_controllers': True,
        'falsification_scope': 'Same frozen selected controller under global guard changes; no optimized-controller necessity proof',
        'gates': {'cold_and_warm_control': selected is not None,
                  'calibration_accuracy': bool(selected and selected['calibration_diagnostic']['accuracy_passed']),
                  'development_accuracy': 'pending_external_frozen_scorer',
                  'numerical_convergence': 'pending_external_frozen_scorer',
                  'full_FIA_ECU': False, 'true_SOC': False, 'formal_release': False},
        'prior_inputs_unchanged': unchanged, 'source_inputs_unchanged': tracked_same,
        'FP2_ledger_unchanged': ledger_same,
        'global_optimality_proven': False, 'infinite_horizon_proven': False}
    _write(output/'tail_audit.json', result)
    return {'output': output, 'report': result, 'winner': selected,
        'controller_spec': None if selected is None else selected['controller'],
        'selected_directory': None if selected is None else output/selected['result_directory'],
        'selected_lap6': None if selected is None else trajectories[selected['name']],
        'simulator': None if selected is None else simulators[selected['name']], 'data': data}


run = tail_guard_audit
