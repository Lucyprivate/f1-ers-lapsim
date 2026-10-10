"""A second frozen experiment after the first run identified terminal drift."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import replace
import json
import shutil
import time
import numpy as np

from f1ers_v4.data import load_local_dataset
from f1ers_v4.protocol import ledger_paths
from f1ers_v6.learning import train_bc, _write, _jsonable
from f1ers_v6.metrics import score
from f1ers_v6.teacher import BlockController
from f1ers_v7.experiment import digest, read, provenance, make_sim, qualify, controller_from
from f1ers_v7.learning import train_sac, load_policy, save_policy
from f1ers_v7.simulator import proxy_audit, numerical_audit, ReachableSimulator
from f1ers_v7.teacher_carry import carry_search


def _worker(root, output, stage, seed, notice, config, teacher_paths, checkpoint=None):
    """Independent deterministic seed; writes one disjoint ignored directory."""
    root, output = Path(root), Path(output)
    data = load_local_dataset(root)
    sim = make_sim(data, notice, config)
    teachers = [read(output/path) for path in teacher_paths]
    directory = output/stage/str(seed)
    if stage == 'B3':
        trained = train_bc(teachers, sim, directory, seed=seed,
            epochs=config['bc_epochs'], dagger_rounds=config['dagger_rounds'],
            dagger_epochs=config['dagger_epochs'], action_space='mode_aware',
            initial_speed_mps=teachers[0]['summary']['speed_start_mps'], progress=print)
    else:
        policy = load_policy(output/checkpoint, expected_features=sim.features,
            expected_action_space='mode_aware', expected_execution_protocol=sim.protocol)
        pool = [{'speed_mps': r['summary']['speed_start_mps'], 'energy_mj': r['summary']['energy_start_mj'],
            'carried_rule_state': r['summary']['initial_rule_state']} for r in teachers[1:]]
        trained = train_sac(sim, policy, directory, seed=seed,
            episodes=config['sac_episodes_per_seed_per_condition'],
            return_scale_s=config['return_scale_s'], initial_alpha=config['alpha_seconds'],
            temperature_mode='fixed', mc_warmstart=True,
            critic_warmup_updates=config['critic_warmup_updates'],
            initial_speed_mps=teachers[0]['summary']['speed_start_mps'],
            initial_state_pool=pool if stage == 'B5' else None,
            evaluation_initial_state=pool[-1] if stage == 'B5' else None,
            periodic_evaluation=stage == 'B4', retain_rollouts=True, progress=print)
        for index, trajectory in enumerate(trained['rollouts'], 1):
            stem = directory/f'training_episode_{index:03d}'
            _write(stem.with_suffix('.json.gz'), trajectory)
            _write(stem.with_suffix('.calibration.json'), score(trajectory['result'], data, 'calibration'))
    metadata = trained['metadata']
    policy = trained['policy']
    if policy is None or trained['result'] is None:
        return {'seed': seed, 'metadata': metadata, 'candidate': None}
    portable = directory/'portable_selected_policy_v7.npz'
    save_policy(policy, portable)
    if stage == 'B3':
        _write(directory/'checkpoint_conversion.json', {
            'source_training_routine': 'frozen v6 BC with explicit 33-feature v7 executor',
            'source_checkpoint_schema': 6, 'portable_checkpoint_schema': 7,
            'weights_changed_by_conversion': False, 'portable_sha256': digest(portable)})
    controller_spec = {'kind': 'policy', 'checkpoint': portable.relative_to(output).as_posix(),
        'action_space': 'mode_aware'}
    # A deterministic single-boundary improvement still needs cold six-lap
    # qualification. All seeds undergo the same independent acceptance.
    qualification_dir = output/'qualification'/f'{stage}_seed_{seed}'
    qualification, laps = qualify(sim, policy, 'mode_aware', qualification_dir,
        teachers[0]['summary']['speed_start_mps'], data)
    candidate = {'name': f'{stage}_seed_{seed}', 'energy_notice_s': notice,
        'action_space': 'mode_aware', 'controller': controller_spec,
        'qualification': qualification,
        'result_directory': qualification_dir.relative_to(output).as_posix(),
        'initial_speed_mps': teachers[0]['summary']['speed_start_mps']}
    return {'seed': seed, 'metadata': metadata, 'candidate': candidate}


def worker(root, output, stage, seed, notice, config, teacher_paths, checkpoint=None):
    try:
        return _worker(root, output, stage, seed, notice, config, teacher_paths, checkpoint)
    except Exception as error:
        import traceback
        directory = Path(output)/stage/str(seed)
        history = read(directory/'sac_history.json') if (directory/'sac_history.json').exists() else []
        completed = max((int(row.get('episode', 0)) for row in history), default=0)
        failure = {'seed': seed, 'status': 'worker_failed', 'exception': str(error),
            'traceback': traceback.format_exc(), 'accepted_trajectory_invented': False,
            'episodes_requested': config['sac_episodes_per_seed_per_condition'] if stage != 'B3' else 0,
            'episodes_completed': completed}
        _write(directory/'worker_exception.json', failure)
        if hasattr(error, 'trace'):
            _write(directory/'worker_exception_trace.json.gz', error.trace)
        return {'seed': seed, 'metadata': failure, 'candidate': None}


def run(root, prior_name='austria_2026_v7', name='austria_2026_v7B'):
    root = Path(root).resolve()
    if any('/' in n or '\\' in n or n in ('', '.', '..') for n in (name, prior_name)):
        raise ValueError('One result directory name required')
    output, prior = root/'results'/name, root/'results'/prior_name
    if output.exists():
        raise RuntimeError('Continuation cannot overwrite an experiment')
    frozen = provenance(root)
    config = read(root/'config/v7/continuation.json')
    previous = read(prior/'experiment.json')
    manifest = read(prior/'manifest.json')
    if previous['source_commit'] != config['prior_v7_code_commit'] or digest(prior/'manifest.json') != config['prior_v7_manifest_sha256']:
        raise ValueError('Continuation is bound to a different prior experiment')
    for relative, expected in manifest['files'].items():
        if digest(prior/relative) != expected:
            raise ValueError('Prior experiment changed: '+relative)
    incumbent = previous['winner']
    if incumbent is None or not incumbent['qualification']['passed']:
        raise ValueError('No previously certified six-lap fallback; learning withheld')
    if incumbent['controller']['kind'] != 'phase_blocks_with_actual_braking':
        raise ValueError('Continuation requires the declared numerical fallback')
    data = load_local_dataset(root)
    if not data['audit']['data_gate_passed']:
        raise ValueError('Historical data gate failed')
    output.mkdir(parents=True)
    shutil.copyfile(root/'config/v7/continuation.json', output/'frozen_protocol.json')
    shutil.copytree(prior/'inputs', output/'inputs')
    _write(output/'freeze.json', frozen)
    _write(output/'started.json', {'source_commit': frozen['code_commit'], 'protocol': config,
        'prior_source_commit': previous['source_commit'], 'prior_manifest_sha256': digest(prior/'manifest.json')})
    notice = incumbent['energy_notice_s']
    initial = incumbent['initial_speed_mps']
    sim = make_sim(data, notice, config)
    candidates, rounds = [], []
    started = time.perf_counter()

    def record(identifier, evidence, change, result=None, **extra):
        row = {'round': identifier, 'preceding_evidence': evidence, 'change': change,
            'summary': result['summary'] if result is not None else None,
            'calibration': score(result, data, 'calibration') if result is not None else None, **extra}
        rounds.append(row)
        _write(output/'rounds.json', rounds)
        if result is not None:
            _write(output/f'{identifier}_representative.json.gz', result)
        print(json.dumps({'round': identifier, 'change': change}, ensure_ascii=False), flush=True)

    def add_search(name, found):
        if found['controller'] is None:
            return None
        spec = found['controller'].to_dict()
        directory = Path(found['selected_directory'])
        candidate = {'name': name, 'energy_notice_s': notice, 'action_space': spec['action_space'],
            'controller': spec, 'qualification': found['qualification'],
            'result_directory': directory.relative_to(output).as_posix(), 'initial_speed_mps': initial}
        candidates.append(candidate)
        _write(output/'candidates.json', candidates)
        return candidate

    controller = BlockController(incumbent['controller']['values'], 'mode_aware')
    first_qualification, first_laps = qualify(sim, controller, 'mode_aware',
        output/'qualification/prior_fallback', initial, data)
    if not first_qualification['passed']:
        raise RuntimeError('Frozen fallback failed independent replay')
    candidates.append({'name': 'prior_fallback', 'energy_notice_s': notice,
        'action_space': 'mode_aware', 'controller': controller.to_dict(),
        'qualification': first_qualification, 'result_directory': 'qualification/prior_fallback',
        'initial_speed_mps': initial})
    record('B0', 'A repaired the request proxy but its strong deployment lost energy; fallback passed',
        'Freeze a separate continuation, independently reproduce all six fallback laps', first_laps[-1])
    signed = carry_search(sim, output/'B1', initial_values=controller.values, blocks=6,
        action_space='signed', initial_speed_mps=initial,
        evaluations=config['carry_teacher_evaluations'], seed=2026)
    add_search('carry_signed_teacher', signed)
    record('B1', 'Cold-only time search cannot detect carried energy drift',
        'Rank every signed numerical proposal only after six actual carried laps', signed['result'], report=signed['report'])
    aware = carry_search(sim, output/'B2', initial_values=signed['controller'].values if signed['controller'] is not None else controller.values,
        blocks=8, action_space='mode_aware', initial_speed_mps=initial,
        evaluations=config['carry_teacher_evaluations'], seed=2026)
    teacher_candidate = add_search('carry_mode_aware_teacher', aware)
    record('B2', 'Action mapping and physical carry must share the same objective',
        'Eight-block mode-aware search with per-lap nondeficit and full proxy gates', aware['result'], report=aware['report'])
    if teacher_candidate is None:
        raise RuntimeError('Mode-aware carry teacher absent; diagnostics retained, no ML authorized')
    teacher_paths = [teacher_candidate['result_directory']+f'/lap_{i}.json.gz' for i in range(1, 7)]
    teachers = [read(output/path) for path in teacher_paths]
    bc_cases = []
    # Processes have disjoint result directories, identical frozen inputs and
    # independent RNGs. No training output from one seed enters another.
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, root, output, 'B3', seed, notice, config, teacher_paths) for seed in config['seeds']]
        for future in futures:
            case = future.result()
            bc_cases.append(case)
            if case['candidate'] is not None:
                candidates.append(case['candidate'])
    _write(output/'candidates.json', candidates)
    def representative(cases):
        eligible = [c['candidate'] for c in cases if c['candidate'] is not None and c['candidate']['qualification']['passed']]
        if not eligible:
            return None
        chosen = min(eligible, key=lambda c: c['qualification']['last_two_mean_time_s'])
        return read(output/chosen['result_directory']/'lap_6.json.gz')
    record('B3', '33-feature policies need independent cloning and carried-state validation',
        'Three-seed BC/DAgger trained on certified cold and warm teacher observations', representative(bc_cases),
        cases=[{'seed': c['seed'], 'metadata': c['metadata']} for c in bc_cases])
    for stage in ('B4', 'B5'):
        cases = []
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, root, output, stage, case['seed'], notice, config, teacher_paths,
                case['candidate']['controller']['checkpoint']) for case in bc_cases if case['candidate'] is not None and case['candidate']['qualification']['passed']]
            for future in futures:
                case = future.result()
                cases.append(case)
                if case['candidate'] is not None:
                    candidates.append(case['candidate'])
        _write(output/'candidates.json', candidates)
        record(stage, 'Unscaled critics jointly underfit; warm energy differs from cold training',
            'Return-scaled SAC with '+('frozen cold boundary' if stage == 'B4' else 'frozen physics-carried boundary pool'),
            representative(cases), cases=[{'seed': c['seed'], 'metadata': c['metadata']} for c in cases],
            stage_status=('completed' if all(c['metadata'].get('all_training_episodes_completed', False) for c in cases)
                else 'partial_or_worker_failure') if cases else 'withheld_no_qualified_BC',
            BC_episode_zero_is_not_SAC_improvement=True)
    eligible = [c for c in candidates if c['qualification']['passed']]
    winner = min(eligible, key=lambda c: c['qualification']['last_two_mean_time_s'])
    _write(output/'selection.json', {'winner': winner, 'source_commit': frozen['code_commit'],
        'ranking': 'all six cold/warm gates then last-two mean time', 'observed_residual_in_selection': False})
    selected = read(output/winner['result_directory']/'lap_6.json.gz')
    _write(output/'selected_calibration.json', score(selected, data, 'calibration'))
    with (output/'development_claim.json').open('x', encoding='utf-8') as stream:
        json.dump({'selection_sha256': digest(output/'selection.json'), 'laps': [7, 9],
            'historical_reused_development': True, 'new_session_test': False}, stream)
    development = score(selected, data, 'development')
    _write(output/'development_score.json', development)
    selected_sim = make_sim(data, winner['energy_notice_s'], config)
    numeric = numerical_audit(selected_sim, selected, output/'numerical')
    _write(output/'numerical_audit.json', numeric)
    record('B6', 'Learned or faster policies must survive fixed-demand numerical validation',
        'Freeze selected controller before historical development scoring; separate output and internal refinement',
        selected, numerical_passed=numeric['passed'])
    sensitivities = []
    for mass, power in ((780., 400.), (820., 400.), (800., 380.), (800., 420.)):
        varied = ReachableSimulator(data['track'], data['rules'],
            vehicle=replace(selected_sim.vehicle, mass_kg=mass, ice_power_kw=power),
            rule_config=selected_sim.rule_config, margin_s=selected_sim.margin_s,
            energy_notice_s=selected_sim.energy_notice_s)
        varied_controller = controller_from(winner['controller'], output, varied)
        result = varied.periodic_simulate(varied_controller, action_space=winner['action_space'], max_cycles=4)
        _write(output/f'sensitivity/mass_{mass:g}_power_{power:g}.json.gz', result)
        sensitivities.append({'mass_kg': mass, 'ice_power_kw': power, 'summary': result['summary'], 'selection_input': False})
    _write(output/'sensitivity.json', sensitivities)
    inputs_unchanged = all((root/name).is_file() and digest(root/name) == expected
        for group in ('tracked_input_sha256', 'dependency_wheels_sha256') for name, expected in frozen[group].items())
    ledger_after = {k: p.exists() for k, p in ledger_paths(root).items()}
    if not inputs_unchanged or ledger_after != frozen['FP2_ledger_before']:
        raise RuntimeError('Frozen input or shared FP2 ledger changed')
    if digest(prior/'manifest.json') != read(output/'started.json')['prior_manifest_sha256']:
        raise RuntimeError('Prior experiment inventory changed')
    if any(digest(prior/name) != expected for name, expected in manifest['files'].items()):
        raise RuntimeError('Prior experiment artifact changed')
    if any(digest(output/'inputs'/Path(name).name) != expected for name, expected in previous['inherited_ignored_inputs_sha256'].items()):
        raise RuntimeError('Copied inherited inputs changed')
    learning_cases = [case for row in rounds if row['round'] in ('B4', 'B5') for case in row['cases']]
    requested = sum(c['metadata']['episodes_requested'] for c in learning_cases)
    completed = sum(c['metadata']['episodes_completed'] for c in learning_cases)
    gates = {'cold_and_warm_control': True,
        'calibration_accuracy': read(output/'selected_calibration.json')['accuracy_passed'],
        'development_accuracy': development['accuracy_passed'], 'numerical_convergence': numeric['passed'],
        'new_session_test': False, 'full_FIA_ECU': False, 'true_SOC': False, 'formal_release': False}
    experiment = {'schema_version': 7, 'kind': 'carry_objective_learning_continuation',
        'source_commit': frozen['code_commit'], 'prior_source_commit': previous['source_commit'],
        'rounds': [r['round'] for r in rounds], 'winner': winner, 'candidates': candidates, 'gates': gates,
        'training_status': ('bounded_cold_and_warm_experiments_completed' if completed == requested
            else 'partial_training_or_worker_failure') if requested else 'withheld_no_qualified_BC',
        'actual_new_SAC_episodes_requested': requested, 'actual_new_SAC_episodes_completed': completed,
        'maximum_new_SAC_episodes': 120, 'inherited_ignored_inputs_sha256': previous['inherited_ignored_inputs_sha256'],
        'frozen_inputs_unchanged': inputs_unchanged, 'FP2_ledger_unchanged': True,
        'FP2_accessed': False, 'new_session_status': '新会话验收待完成',
        'full_FIA_compliance_proven': False, 'true_SOC_proven': False, 'global_optimality_proven': False,
        'wall_seconds': time.perf_counter()-started}
    _write(output/'experiment.json', experiment)
    _write(output/'manifest.json', {'source_commit': frozen['code_commit'],
        'files': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}})
    print(json.dumps({'winner': winner['name'], 'gates': gates}, ensure_ascii=False), flush=True)
    return experiment
