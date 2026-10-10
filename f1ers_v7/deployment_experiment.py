"""Frozen D deployment-only experiment: analytical clone and bounded SAC.

Selection uses six real carried laps and never observed velocity residuals.
Both best and last retained policies are loaded and qualified. Intermediate
SAC evaluation weights are not saved by this inherited training schedule;
their unavailability is explicitly disclosed rather than called exhaustive.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import importlib.metadata
import json
import shutil
import time
import traceback
import numpy as np

from f1ers_v7._core.data import load_local_dataset
from f1ers_v7._core.protocol import ledger_paths
from f1ers_v7._core.learning_base import _write, _jsonable
from f1ers_v7._core.metrics import score
from f1ers_v7._core.base_rules import RuleConfig
from f1ers_v7._core.provenance import digest, read, provenance, git
from f1ers_v7.deploy_only import (DeployOnlySimulator, FEATURES, PROTOCOL,
    proxy_audit, numerical_audit, load_deploy_policy)
from f1ers_v7.deploy_learning import train_bc, train_sac


class ConstantZeroController:
    """Declared analytical deployment .5 / full requested braking regeneration."""
    def __call__(self, observation):
        return 0.


def validate_config(config):
    required = {'seeds': [2026, 2027, 2028], 'vehicle_mass_kg': 800.,
        'ice_power_kw': 400., 'energy_window_mj': [2., 4.], 'tail_guard_m': 1000.,
        'geometry_margin_s': 1., 'energy_notice_s': .5, 'bc_epochs': 600,
        'dagger_rounds': 3, 'dagger_epochs': 200, 'sac_episodes_per_seed_per_condition': 20,
        'new_SAC_episode_maximum': 120, 'return_scale_s': 73.084831,
        'alpha_seconds': .001, 'critic_warmup_updates': 120,
        'physical_laps_per_candidate': 6, 'parallel_seed_workers': 3,
        'policy_specific_periodic_shooting': False, 'prior_new_SAC_episodes_completed': 0}
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError('D frozen protocol differs: '+name)
    if config.get('features') != list(FEATURES) or config.get('execution_protocol') != PROTOCOL:
        raise ValueError('D state/action contract differs')
    return config


def make_sim(data, config):
    sim = DeployOnlySimulator(data['track'], data['rules'],
        rule_config=RuleConfig(tail_guard_m=config['tail_guard_m']),
        margin_s=config['geometry_margin_s'], energy_notice_s=config['energy_notice_s'])
    if sim.vehicle.mass_kg != config['vehicle_mass_kg'] or sim.vehicle.ice_power_kw != config['ice_power_kw']:
        raise ValueError('Frozen vehicle defaults differ from preregistered assumptions')
    return sim


def safe_file(directory, relative):
    directory = Path(directory).resolve()
    path = (directory/relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('Artifact path outside frozen directory or missing: '+relative)
    return path


def result_directory(root, name):
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or any(character in name for character in ('/', '\\', ':'))):
        raise ValueError('One plain result directory name is required')
    base = (Path(root)/'results').resolve()
    result = (base/name).resolve()
    if result.parent != base:
        raise ValueError('Result path must stay immediately inside results')
    return result


def controller_from(spec, output):
    if spec['kind'] == 'analytical_constant_raw_zero':
        return ConstantZeroController()
    if spec['kind'] == 'policy' and spec['action_space'] == 'mode_aware':
        return load_deploy_policy(safe_file(output, spec['checkpoint']))
    raise ValueError('Unsupported D controller or action contract')


def qualify(sim, controller, output, initial_speed, data):
    # Same physical boundary carry as SimulatorV6.simulate_laps, with each lap
    # flushed before a later lap or proxy calculation could raise an exception.
    laps, summaries, proxies = [], [], []
    speed, energy, carry = initial_speed, 2., None
    for index in range(1, 7):
        result = sim.simulate(controller, initial_speed_mps=speed,
            initial_energy_mj=energy, carried_rule_state=carry, action_space='mode_aware')
        result['summary']['physical_lap_index'] = index-1
        laps.append(result)
        _write(output/f'lap_{index}.json.gz', result)
        proxy = proxy_audit(sim, result)
        _write(output/f'lap_{index}_proxy.json', proxy)
        _write(output/f'lap_{index}_calibration.json', score(result, data, 'calibration'))
        summaries.append(result['summary'])
        proxies.append(proxy)
        if not result['summary']['completed_lap'] or not result['summary']['execution_feasible']:
            break
        speed = result['terminal_speed_mps']
        energy = result['summary']['energy_end_mj']
        carry = result['rule_certificate']['terminal_boundary_state']
    crossings = [{'speed_difference_mps': b['summary']['speed_start_mps']-a['summary']['speed_end_mps'],
        'energy_difference_mj': b['summary']['energy_start_mj']-a['summary']['energy_end_mj'],
        'dc_difference_kw': b['summary']['initial_dc_kw']-a['summary']['terminal_dc_kw'],
        'rule_state_carried': b['summary']['initial_rule_state'] == a['rule_certificate']['terminal_boundary_state']}
        for a, b in zip(laps, laps[1:])]
    continuity = bool(len(laps) == 6 and all(abs(c['speed_difference_mps']) < 1e-10
        and abs(c['energy_difference_mj']) < 1e-10 and abs(c['dc_difference_kw']) < 1e-10
        and c['rule_state_carried'] for c in crossings))
    complete = bool(len(summaries) == 6 and all(s['completed_lap'] and s['execution_feasible']
        and isinstance(s.get('lap_time_s'), (float, int, np.floating, np.integer))
        and np.isfinite(s['lap_time_s']) and s['lap_time_s'] > 0 for s in summaries))
    passed = bool(complete and continuity
        and all(s['feasible'] for s in summaries) and all(p['passed'] for p in proxies))
    row = {'passed': passed, 'physical_laps': len(summaries),
        'all_cold_and_warm_proxy_gates': [p['passed'] for p in proxies],
        'crossings': crossings, 'physical_control_line_continuity': continuity,
        'lap_summaries': summaries,
        'last_two_mean_time_s': float(np.mean([s['lap_time_s'] for s in summaries[-2:]])) if complete else None,
        'formal_objective_s': float(np.mean([s['lap_time_s'] for s in summaries[-2:]])) if passed else None,
        'within_lap_velocity_or_energy_resets': 0, 'policy_specific_periodic_shooting': False,
        'infinite_horizon_proven': False, 'deployment_only_proxy_used': True}
    _write(output/'qualification.json', row)
    return row, laps


def episode_counts(directory):
    """Count saved receipts/terminal trajectories, never max exception episode."""
    directory = Path(directory)
    completed = sorted(directory.glob('training_episode_*.json.gz'))
    started = sorted(directory.glob('episode_*_started.json'))
    physical_complete = sum(bool(read(path)['result']['summary'].get('completed_lap', False)) for path in completed)
    return {'episodes_attempted': len(started), 'episodes_completed': len(completed),
        'physical_full_laps_completed': physical_complete,
        'completed_includes_executor_terminal_failure_episodes': True}


def prior_episode_budget(directory, receipt):
    """Read explicit counts or the one evidenced historical withheld schema.

    Missing counts are never generally interpreted as zero. The old A runner
    predates these fields, and its saved withheld status plus absent training
    records is the only supported compatibility case.
    """
    fields = ('actual_new_SAC_episodes_requested', 'actual_new_SAC_episodes_completed')
    if all(name in receipt for name in fields):
        requested, completed = (receipt[name] for name in fields)
        if (not all(isinstance(value, int) and not isinstance(value, bool) for value in (requested, completed))
                or not 0 <= completed <= requested <= 120):
            raise ValueError('Invalid prior episode budget')
        return {'episodes_requested': requested, 'episodes_completed': completed,
                'count_basis': 'explicit_frozen_experiment_fields'}
    if any(name in receipt for name in fields):
        raise ValueError('Incomplete prior episode budget schema')
    directory = Path(directory)
    if receipt.get('training_status') != 'withheld_no_s1_physical_teacher':
        raise ValueError('Unknown prior experiment without episode counts')
    patterns = ('*sac_metadata.json', '*sac_history.json', 'episode_*_started.json',
                'training_episode_*.json.gz')
    if any(next(directory.rglob(pattern), None) is not None for pattern in patterns):
        raise ValueError('Withheld receipt conflicts with saved training evidence')
    rounds = read(directory/'rounds.json')
    if any(case.get('metadata', {}).get('episodes_requested', 0)
            for row in rounds for case in row.get('cases', []) or []):
        raise ValueError('Withheld receipt conflicts with recorded episode requests')
    return {'episodes_requested': 0, 'episodes_completed': 0,
            'count_basis': 'legacy_withheld_status_and_absent_training_records'}


def qualify_retained(root, output, stage, seed, config, teachers, directory):
    """Load every retained best/last; failure of either is kept independently."""
    data = load_local_dataset(root)
    sim = make_sim(data, config)
    candidates = []
    paths = (('best', 'best_bc_policy.npz'), ('last', 'last_bc_candidate.npz')) if stage == 'D2' else (
        ('best', 'best_feasible_policy.npz'), ('last', 'last_sac_candidate.npz'))
    for retained, filename in paths:
        path = directory/filename
        if not path.exists():
            _write(directory/f'{retained}_qualification_unavailable.json', {'checkpoint': filename,
                'reason': 'Checkpoint was never retained', 'accepted_trajectory_invented': False})
            continue
        name = f'{stage}_seed_{seed}_{retained}'
        qualification_dir = output/'qualification'/name
        try:
            controller = load_deploy_policy(path)
            qualification, laps = qualify(sim, controller, qualification_dir,
                teachers[0]['summary']['speed_start_mps'], data)
        except Exception as error:
            _write(qualification_dir/'qualification_exception.json', {'exception': str(error),
                'traceback': traceback.format_exc(), 'accepted_trajectory_invented': False})
            if hasattr(error, 'trace'):
                _write(qualification_dir/'qualification_exception_trace.json.gz', error.trace)
            qualification = {'passed': False, 'status': 'qualification_exception',
                'last_two_mean_time_s': None, 'exception': str(error)}
        candidates.append({'name': name, 'energy_notice_s': config['energy_notice_s'],
            'tail_guard_m': config['tail_guard_m'], 'action_space': 'mode_aware',
            'controller': {'kind': 'policy', 'checkpoint': path.relative_to(output).as_posix(), 'action_space': 'mode_aware'},
            'checkpoint_sha256': digest(path), 'retained_checkpoint': retained,
            'qualification': qualification, 'result_directory': qualification_dir.relative_to(output).as_posix(),
            'initial_speed_mps': teachers[0]['summary']['speed_start_mps']})
    return candidates


def worker(root, output, stage, seed, config, teacher_paths, checkpoint=None):
    root, output = Path(root), Path(output)
    directory = output/stage/str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    requested = config['sac_episodes_per_seed_per_condition'] if stage in ('D3', 'D4') else 0
    _write(directory/'worker_started.json', {'stage': stage, 'seed': seed, 'episodes_requested': requested,
        'new_SAC_episode_cap_for_case': 20, 'episode_zero_counted_as_new_SAC': False})
    teachers = [read(output/path) for path in teacher_paths]
    failure = None
    try:
        data = load_local_dataset(root)
        sim = make_sim(data, config)
        if stage == 'D2':
            trained = train_bc(teachers, sim, directory, seed=seed,
                epochs=config['bc_epochs'], dagger_rounds=config['dagger_rounds'],
                dagger_epochs=config['dagger_epochs'],
                initial_speed_mps=teachers[0]['summary']['speed_start_mps'], progress=print)
        else:
            policy = load_deploy_policy(safe_file(output, checkpoint))
            pool = [{'speed_mps': r['summary']['speed_start_mps'], 'energy_mj': r['summary']['energy_start_mj'],
                'carried_rule_state': r['summary']['initial_rule_state'], 'source': f'physical_teacher_lap_{i}_start'}
                for i, r in enumerate(teachers[1:], 2)]
            trained = train_sac(sim, policy, directory, seed=seed,
                episodes=requested, return_scale_s=config['return_scale_s'], initial_alpha=config['alpha_seconds'],
                temperature_mode='fixed', mc_warmstart=True,
                critic_warmup_updates=config['critic_warmup_updates'],
                initial_speed_mps=teachers[0]['summary']['speed_start_mps'], initial_energy_mj=2.,
                initial_state_pool=pool if stage == 'D4' else None,
                evaluation_initial_state=pool[-1] if stage == 'D4' else None,
                periodic_evaluation=False, retain_rollouts=False, progress=print)
        metadata = trained['metadata']
    except Exception as error:
        failure = {'stage': stage, 'seed': seed, 'status': 'worker_failed', 'exception': str(error),
            'traceback': traceback.format_exc(), 'accepted_trajectory_invented': False}
        _write(directory/'worker_exception.json', failure)
        if hasattr(error, 'trace'):
            _write(directory/'worker_exception_trace.json.gz', error.trace)
        metadata_path = directory/('bc_metadata.json' if stage == 'D2' else 'sac_metadata.json')
        metadata = read(metadata_path) if metadata_path.exists() else {'status': 'worker_failed_before_metadata'}
    counts = episode_counts(directory)
    if counts['episodes_attempted'] > requested or counts['episodes_completed'] > counts['episodes_attempted']:
        raise RuntimeError('New SAC episode receipt budget violated')
    metadata = {**metadata, 'episodes_requested': requested, **counts,
        'worker_failure': failure, 'periodic_evaluation_policy_weights_retained': False,
        'all_intermediate_checkpoint_qualification_claimed': False}
    _write(directory/'worker_receipt.json', metadata)
    candidates = qualify_retained(root, output, stage, seed, config, teachers, directory)
    _write(directory/'retained_candidates.json', candidates)
    return {'seed': seed, 'metadata': metadata, 'candidates': candidates}


def _rank(candidates):
    return sorted((c for c in candidates if c['qualification']['passed']),
        key=lambda c: (c['qualification']['last_two_mean_time_s'], c['name']))


def run(root, prior_name='austria_2026_v7C', name='austria_2026_v7D'):
    root = Path(root).resolve()
    output, prior = result_directory(root, name), result_directory(root, prior_name)
    if output.exists():
        raise RuntimeError('D cannot overwrite or rescore an experiment')
    frozen = provenance(root)
    config = validate_config(read(root/'config/v7/deployment.json'))
    previous, prior_manifest = read(prior/'experiment.json'), read(prior/'manifest.json')
    if (previous['source_commit'] != config['prior_C_source_commit']
            or digest(prior/'manifest.json') != config['prior_C_manifest_sha256']):
        raise ValueError('D is bound to a different sealed C experiment')
    for relative, expected in prior_manifest['files'].items():
        if digest(safe_file(prior, relative)) != expected:
            raise ValueError('Sealed C artifact changed: '+relative)
    incumbent = previous['winner']
    if not incumbent or not incumbent['qualification']['passed']:
        raise ValueError('A qualified C fallback is required before D learning')
    if (incumbent['controller'].get('kind') != 'phase_blocks_with_actual_braking'
            or incumbent['controller']['values'] != [.5]
            or incumbent['tail_guard_m'] != 1000. or incumbent['energy_notice_s'] != .5):
        raise ValueError('Analytical D teacher requires the fixed .5 positive/full braking baseline')
    prior_budget = {}
    for earlier in ('austria_2026_v7', 'austria_2026_v7B', prior_name):
        receipt = read(root/'results'/earlier/'experiment.json')
        prior_budget[earlier] = prior_episode_budget(root/'results'/earlier, receipt)
    if sum(row['episodes_completed'] for row in prior_budget.values()) != config['prior_new_SAC_episodes_completed']:
        raise ValueError('Prior new SAC budget is inconsistent')
    data = load_local_dataset(root)
    if not data['audit']['data_gate_passed']:
        raise ValueError('Frozen historical data gate failed')
    output.mkdir(parents=True)
    shutil.copyfile(root/'config/v7/deployment.json', output/'frozen_protocol.json')
    inherited = {}
    for filename in ('manifest.json', 'selection.json'):
        source = prior/filename
        expected = digest(source)
        inherited[source.relative_to(root).as_posix()] = expected
        (output/'inputs').mkdir(exist_ok=True)
        shutil.copyfile(source, output/'inputs'/filename)
    _write(output/'inputs/input_bindings.json', {'inherited_ignored_inputs_sha256': inherited,
        'same_unique_basename_copy_rule': True, 'old_checkpoints_or_33_feature_teachers_loaded': False})
    _write(output/'freeze.json', frozen)
    _write(output/'started.json', {'source_commit': frozen['code_commit'], 'protocol': config,
        'prior_source_commit': previous['source_commit'], 'prior_new_SAC_budget': prior_budget,
        'maximum_new_SAC_episodes': 120})
    started = time.perf_counter()
    sim = make_sim(data, config)
    initial = incumbent['initial_speed_mps']
    candidates, rounds = [], []

    def record(identifier, evidence, change, result=None, **extra):
        row = {'round': identifier, 'preceding_evidence': evidence, 'change': change,
            'summary': result['summary'] if result is not None else None,
            'calibration': score(result, data, 'calibration') if result is not None else None, **extra}
        rounds.append(row)
        _write(output/'rounds.json', rounds)
        if result is not None:
            _write(output/f'{identifier}_representative.json.gz', result)
        print(json.dumps({'round': identifier, 'change': change}, ensure_ascii=False), flush=True)

    qualification, teachers = qualify(sim, ConstantZeroController(), output/'qualification/analytical_teacher', initial, data)
    analytic = {'name': 'analytical_teacher', 'energy_notice_s': .5, 'tail_guard_m': 1000.,
        'action_space': 'mode_aware', 'controller': {'kind': 'analytical_constant_raw_zero', 'action_space': 'mode_aware'},
        'qualification': qualification, 'result_directory': 'qualification/analytical_teacher', 'initial_speed_mps': initial}
    candidates.append(analytic)
    _write(output/'candidates.json', candidates)
    record('D0', 'B BC approximated saturated brake labels and changed positive actions; exclusive causality not identified',
        'Declare the deployment-only engineering map with explicit episode initial energy, separate from FIA hard boundaries',
        teachers[-1] if teachers else None, not_a_FIA_rule_addition=True)
    record('D1', 'The incumbent has only a 512 J late-lap margin; old mapped brake labels must not enter new BC',
        'Generate six new physical D laps with exact raw zero and full mapped braking regeneration',
        teachers[-1] if teachers else None, teacher_qualification=qualification,
        old_raw_one_teacher_relabelled=False, policy_specific_periodic_shooting=False)
    teacher_paths = [analytic['result_directory']+f'/lap_{i}.json.gz' for i in range(1, len(teachers)+1)]
    bc_cases = []
    if qualification['passed']:
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, root, output, 'D2', seed, config, teacher_paths) for seed in config['seeds']]
            for future in futures:
                case = future.result()
                bc_cases.append(case)
                candidates.extend(case['candidates'])
    _write(output/'candidates.json', candidates)

    def representative(cases):
        choices = _rank([c for case in cases for c in case['candidates']])
        return read(output/choices[0]['result_directory']/'lap_6.json.gz') if choices else None

    record('D2', 'A constant target can be cloned analytically without a finite tanh saturation error',
        'Run BC 600 + DAgger 3 x 200 for three seeds; verify exact mean zero, qualify retained best and last',
        representative(bc_cases), cases=[{'seed': c['seed'], 'metadata': c['metadata']} for c in bc_cases],
        stage_status='completed' if bc_cases and all(c['metadata'].get('worker_failure') is None for c in bc_cases) else 'withheld_or_partial',
        analytical_clone_is_not_BC_gain=True)
    sac_cases = []
    for stage in ('D3', 'D4'):
        cases = []
        authorized = []
        for bc in bc_cases:
            eligible = _rank(bc['candidates'])
            if eligible:
                # Same deterministic trace; choose declared retained best in ties.
                authorized.append((bc['seed'], eligible[0]['controller']['checkpoint']))
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, root, output, stage, seed, config, teacher_paths, checkpoint)
                for seed, checkpoint in authorized]
            for future in futures:
                case = future.result()
                cases.append(case)
                candidates.extend(case['candidates'])
        sac_cases.extend(cases)
        _write(output/'candidates.json', candidates)
        record(stage, 'Current and episode initial energy are both observed; fixed-boundary objective equals six-carry first-lap boundary',
            'Normalized SAC on '+('cold E2 boundary' if stage == 'D3' else 'round-robin physical teacher lap 2-6 starts'),
            representative(cases), cases=[{'seed': c['seed'], 'metadata': c['metadata']} for c in cases],
            stage_status=('completed' if all(c['metadata'].get('all_training_episodes_completed', False)
                and c['metadata'].get('worker_failure') is None for c in cases) else 'partial_or_worker_failure') if cases else 'withheld_no_qualified_BC',
            BC_episode_zero_is_not_SAC_improvement=True, policy_specific_periodic_shooting=False,
            retained_best_and_last_qualified=True, unsaved_intermediate_weights_exhaustively_qualified=False)
    eligible = _rank(candidates)
    winner = eligible[0] if eligible else None
    _write(output/'selection.json', {'winner': winner, 'source_commit': frozen['code_commit'],
        'ranking': 'all six physical gates then last-two mean time, deterministic lexical tie-break',
        'observed_residual_in_selection': False, 'best_and_last_retained_policies_qualified': True,
        'all_intermediate_checkpoints_qualified': False})
    if winner is not None:
        selected = read(output/winner['result_directory']/'lap_6.json.gz')
        _write(output/'selected_calibration.json', score(selected, data, 'calibration'))
        with (output/'development_claim.json').open('x', encoding='utf-8') as stream:
            json.dump({'selection_sha256': digest(output/'selection.json'), 'laps': [7, 9],
                'historical_reused_development': True, 'new_session_test': False}, stream)
        development = score(selected, data, 'development')
        _write(output/'development_score.json', development)
        numeric = numerical_audit(sim, selected, output/'numerical')
        _write(output/'numerical_audit.json', numeric)
        record('D5', 'The fastest learned checkpoint may still be noncyclic or differ only at an unqualified endpoint',
            'Freeze six-carry selection, score historical development once, verify D mapping under fixed-demand numerical refinement',
            selected, numerical_passed=numeric['passed'])
        sensitivities = []
        for mass, power in ((780., 400.), (820., 400.), (800., 380.), (800., 420.)):
            varied = DeployOnlySimulator(data['track'], data['rules'],
                vehicle=replace(sim.vehicle, mass_kg=mass, ice_power_kw=power),
                rule_config=sim.rule_config, margin_s=sim.margin_s, energy_notice_s=sim.energy_notice_s)
            controller = controller_from(winner['controller'], output)
            physical = varied.simulate_laps(controller, n_laps=6, initial_speed_mps=initial,
                initial_energy_mj=2., action_space='mode_aware')
            for index, result in enumerate(physical['laps'], 1):
                _write(output/f'sensitivity/mass_{mass:g}_power_{power:g}_lap_{index}.json.gz', result)
            sensitivities.append({'mass_kg': mass, 'ice_power_kw': power,
                'lap_summaries': [r['summary'] for r in physical['laps']],
                'physical_control_line_continuity': physical['physical_control_line_continuity'],
                'selection_input': False, 'policy_specific_periodic_shooting': False})
        _write(output/'sensitivity.json', sensitivities)
    else:
        development, numeric = {'accuracy_passed': False}, {'passed': False}
        record('D5', 'No qualified candidate; no prediction can be formally scored',
            'Keep diagnostic failures and withhold selection, development score and training continuation')
    requested = sum(case['metadata']['episodes_requested'] for case in sac_cases)
    completed = sum(case['metadata']['episodes_completed'] for case in sac_cases)
    attempted = sum(case['metadata']['episodes_attempted'] for case in sac_cases)
    if not 0 <= completed <= attempted <= requested <= 120:
        raise RuntimeError('Aggregate D new SAC budget violated')
    inputs_unchanged = all((root/path).is_file() and digest(root/path) == expected
        for group in ('tracked_input_sha256', 'dependency_wheels_sha256') for path, expected in frozen[group].items())
    for relative, expected in prior_manifest['files'].items():
        if digest(safe_file(prior, relative)) != expected:
            raise RuntimeError('Sealed prior experiment changed: '+relative)
    if (not inputs_unchanged or digest(prior/'manifest.json') != config['prior_C_manifest_sha256']
            or {k: p.exists() for k, p in ledger_paths(root).items()} != frozen['FP2_ledger_before']):
        raise RuntimeError('Frozen source, prior manifest or shared FP2 ledger changed')
    for path, expected in inherited.items():
        if digest(output/'inputs'/Path(path).name) != expected:
            raise RuntimeError('Frozen inherited input copy changed')
    gates = {'cold_and_warm_control': winner is not None,
        'calibration_accuracy': read(output/'selected_calibration.json')['accuracy_passed'] if winner else False,
        'development_accuracy': development['accuracy_passed'], 'numerical_convergence': numeric['passed'],
        'new_session_test': False, 'full_FIA_ECU': False, 'true_SOC': False, 'formal_release': False}
    experiment = {'schema_version': 7, 'kind': 'deployment_only_analytical_clone_learning_continuation',
        'source_commit': frozen['code_commit'], 'prior_source_commit': previous['source_commit'],
        'prior_manifest_sha256': config['prior_C_manifest_sha256'], 'rounds': [r['round'] for r in rounds],
        'winner': winner, 'candidates': candidates, 'gates': gates,
        'training_status': ('bounded_cold_and_warm_experiments_completed' if completed == requested
            and all(c['metadata'].get('worker_failure') is None for c in sac_cases) else 'partial_training_or_worker_failure')
            if requested else 'withheld_no_qualified_BC',
        'actual_new_SAC_episodes_requested': requested, 'actual_new_SAC_episodes_attempted': attempted,
        'actual_new_SAC_episodes_completed': completed, 'maximum_new_SAC_episodes': 120,
        'prior_new_SAC_budget': prior_budget, 'inherited_ignored_inputs_sha256': inherited,
        'frozen_inputs_unchanged': inputs_unchanged, 'FP2_ledger_unchanged': True, 'FP2_accessed': False,
        'new_session_status': '新会话验收待完成', 'full_FIA_compliance_proven': False,
        'true_SOC_proven': False, 'global_optimality_proven': False, 'infinite_horizon_proven': False,
        'analytical_clone_is_not_BC_gain': True, 'BC_episode_zero_is_not_SAC_gain': True,
        'periodic_evaluation_policy_weights_retained': False, 'all_intermediate_checkpoints_qualified': False,
        'policy_specific_periodic_shooting': False, 'wall_seconds': time.perf_counter()-started}
    _write(output/'experiment.json', experiment)
    _write(output/'manifest.json', {'source_commit': frozen['code_commit'],
        'files': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p != output/'manifest.json'}})
    print(json.dumps({'winner': None if winner is None else winner['name'], 'gates': gates}, ensure_ascii=False), flush=True)
    return experiment


def verify(root, name='austria_2026_v7D'):
    root = Path(root).resolve()
    output = result_directory(root, name)
    manifest, frozen, config, experiment = (read(output/f) for f in
        ('manifest.json', 'freeze.json', 'frozen_protocol.json', 'experiment.json'))
    validate_config(config)
    files = {p.relative_to(output).as_posix() for p in output.rglob('*')
        if p.is_file() and p != output/'manifest.json'}
    if files != set(manifest['files']):
        raise ValueError('Frozen inventory files were added or removed')
    for relative, expected in manifest['files'].items():
        if digest(safe_file(output, relative)) != expected:
            raise ValueError('Frozen result changed: '+relative)
    input_ok = all((root/path).is_file() and digest(root/path) == expected
        for group in ('tracked_input_sha256', 'dependency_wheels_sha256') for path, expected in frozen[group].items())
    dependencies_ok = {name: importlib.metadata.version(name) for name in frozen['dependencies']} == frozen['dependencies']
    if not input_ok or not dependencies_ok:
        raise ValueError('Frozen source or dependency versions differ')
    if git(root, 'rev-parse', frozen['code_commit']+'^{commit}') != frozen['code_commit']:
        raise ValueError('Frozen source commit is unavailable')
    for relative, expected in experiment['inherited_ignored_inputs_sha256'].items():
        if digest(output/'inputs'/Path(relative).name) != expected:
            raise ValueError('Frozen inherited input copy changed')
    if {k: p.exists() for k, p in ledger_paths(root).items()} != frozen['FP2_ledger_before']:
        raise ValueError('Shared FP2 ledger changed')
    winner = experiment['winner']
    if winner is None:
        return {'passed': False, 'inventory_passed': True, 'reason': 'No qualified six-lap controller', 'FP2_accessed': False}
    data = load_local_dataset(root)
    sim = make_sim(data, config)
    controller = controller_from(winner['controller'], output)
    saved = read(output/winner['result_directory']/'lap_6.json.gz')
    replay = sim.simulate(controller, initial_speed_mps=saved['summary']['speed_start_mps'],
        initial_energy_mj=saved['summary']['energy_start_mj'], carried_rule_state=saved['summary']['initial_rule_state'],
        action_space='mode_aware')
    replay['summary']['physical_lap_index'] = saved['summary']['physical_lap_index']
    exact = {'exact_trace': set(replay['internal_trace']) == set(saved['internal_trace']) and all(
        np.array_equal(np.asarray(replay['internal_trace'][k]), np.asarray(v), equal_nan=True)
        for k, v in saved['internal_trace'].items()),
        'exact_raw_actions': np.array_equal(replay['requested_actions'], saved['requested_actions']),
        'exact_mapped_actions': np.array_equal(replay['mapped_actions'], saved['mapped_actions']),
        'exact_proxy': _jsonable(proxy_audit(sim, replay)) == read(output/winner['result_directory']/'lap_6_proxy.json'),
        'exact_saved_calibration': _jsonable(score(replay, data, 'calibration')) == read(output/'selected_calibration.json'),
        'exact_saved_development': _jsonable(score(replay, data, 'development')) == read(output/'development_score.json')}
    return {'passed': all(exact.values()), **exact, 'inventory_passed': True,
        'source_hashes_passed': input_ok, 'dependency_versions_passed': dependencies_ok,
        'new_selection_or_training': False, 'FP2_accessed': False, 'accuracy_gates': experiment['gates'],
        'policy_specific_periodic_shooting': False, 'formal_release': False}
