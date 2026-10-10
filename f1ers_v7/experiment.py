"""Bounded continuation with frozen provenance and physical carried selection."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import gzip
import importlib.metadata
import json
import shutil
import subprocess
import time
import numpy as np

from f1ers_v4.data import load_local_dataset
from f1ers_v4.protocol import ledger_paths
from f1ers_v6.learning import load_policy as load_old_policy, train_bc, _write, _jsonable
from f1ers_v6.metrics import score
from f1ers_v6.rules import RuleConfig
from f1ers_v6.teacher import BlockController, search
from f1ers_v6.geometry_guard import GeometryGuardSimulator
from f1ers_v6.r7 import proxy_for_r7
from .simulator import ReachableSimulator, proxy_audit, numerical_audit
from .learning import train_sac, load_policy, save_policy


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


def make_sim(data, notice, config):
    return ReachableSimulator(data['track'], data['rules'],
        rule_config=RuleConfig(tail_guard_m=config['tail_guard_m']),
        margin_s=config['geometry_margin_s'], energy_notice_s=notice)


def controller_from(spec, output, sim):
    if spec['kind'] == 'neutral':
        return None
    if spec['kind'] == 'phase_blocks_with_actual_braking':
        return BlockController(spec['values'], spec['action_space'])
    if spec['kind'] == 'policy':
        return load_policy(output/spec['checkpoint'], expected_features=sim.features,
            expected_action_space=spec['action_space'], expected_execution_protocol=sim.protocol)
    raise ValueError('Unsupported frozen controller')


def qualify(sim, controller, space, output, initial_speed, data):
    physical = sim.simulate_laps(controller, n_laps=6, initial_speed_mps=initial_speed,
        initial_energy_mj=2., action_space=space)
    summaries, proxies = [], []
    for i, result in enumerate(physical['laps']):
        _write(output/f'lap_{i+1}.json.gz', result)
        proxy = proxy_audit(sim, result)
        _write(output/f'lap_{i+1}_proxy.json', proxy)
        _write(output/f'lap_{i+1}_calibration.json', score(result, data, 'calibration'))
        summaries.append(result['summary'])
        proxies.append(proxy)
    passed = bool(len(summaries) == 6 and physical['physical_control_line_continuity'] and
        all(s['feasible'] for s in summaries) and all(p['passed'] for p in proxies))
    mean = float(np.mean([s['lap_time_s'] for s in summaries[-2:]])) if len(summaries) == 6 and all(s['completed_lap'] for s in summaries) else None
    row = {'passed': passed, 'physical_laps': len(summaries),
        'all_cold_and_warm_proxy_gates': [p['passed'] for p in proxies],
        'crossings': physical['crossings'], 'physical_control_line_continuity': physical['physical_control_line_continuity'],
        'lap_summaries': summaries, 'last_two_mean_time_s': mean,
        'within_lap_velocity_or_energy_resets': 0, 'infinite_horizon_proven': False}
    _write(output/'qualification.json', row)
    return row, physical['laps']


def run(root, name='austria_2026_v7'):
    root = Path(root).resolve()
    if '/' in name or '\\' in name or name in ('', '.', '..'):
        raise ValueError('One new result directory name required')
    output = root/'results'/name
    if output.exists():
        raise RuntimeError('Existing experiment cannot be overwritten or rescored')
    freeze = provenance(root)
    config = read(root/'config/v7/protocol.json')
    data = load_local_dataset(root)
    if not data['audit']['data_gate_passed']:
        raise ValueError('Frozen historical data gate failed')
    output.mkdir(parents=True)
    shutil.copyfile(root/'config/v7/protocol.json', output/'frozen_protocol.json')
    _write(output/'freeze.json', freeze)
    _write(output/'started.json', {'source_commit': freeze['code_commit'], 'protocol': config})
    rounds, candidates = [], []
    start_wall = time.perf_counter()

    def record(identifier, hypothesis, evidence, change, result=None, **extra):
        row = {'round': identifier, 'hypothesis': hypothesis, 'preceding_evidence': evidence,
            'change': change, 'summary': result['summary'] if result is not None else None,
            'calibration': score(result, data, 'calibration') if result is not None else None, **extra}
        rounds.append(row)
        _write(output/'rounds.json', rounds)
        if result is not None:
            _write(output/f'{identifier}_representative.json.gz', result)
        print(json.dumps({'round': identifier, 'change': change,
            'time_s': None if result is None else result['summary']['lap_time_s'],
            'feasible': None if result is None else result['summary']['feasible']}, ensure_ascii=False), flush=True)

    def add(name, sim, controller, space, spec, result):
        initial = result['summary']['speed_start_mps']
        directory = output/'qualification'/name
        qualification, laps = qualify(sim, controller, space, directory, initial, data)
        candidate = {'name': name, 'energy_notice_s': sim.energy_notice_s,
            'action_space': space, 'controller': spec,
            'qualification': qualification, 'result_directory': directory.relative_to(output).as_posix(),
            'initial_speed_mps': initial}
        candidates.append(candidate)
        _write(output/'candidates.json', candidates)
        print(json.dumps({'candidate': name, 'six_cold_warm_passed': qualification['passed'],
            'steady_mean_time_s': qualification['last_two_mean_time_s']}, ensure_ascii=False), flush=True)
        return candidate, laps

    # S0 is a new reconstruction of the frozen R7 cold condition, not a new
    # reinterpretation of its old results or an old checkpoint on new features.
    prior = root/'results/austria_2026_v6'
    checkpoint = prior/'R4/2027/raw_bc_policy.npz'
    saved_r7 = read(prior/'R7/margin_1/lap_1.json.gz')
    if digest(checkpoint) != 'c9fd321a130dffdb370b0aaf87adb7e54d51c60d97c2d25da5f3dc464f4bb611':
        raise ValueError('Inherited checkpoint changed')
    if digest(prior/'R7/manifest.json') != '9b3e1c3d30b96f5ae49cd3fcc847bc84ca760671dc6bb40f0628d8fdadf0c5a1':
        raise ValueError('Inherited R7 inventory changed')
    prior_manifest = read(prior/'R7/manifest.json')
    for relative, expected in prior_manifest['files'].items():
        if digest(prior/'R7'/relative) != expected:
            raise ValueError('Inherited R7 artifact changed: '+relative)
    inherited_paths = [checkpoint, prior/'R7/manifest.json', prior/'R7/margin_1/lap_1.json.gz']
    inherited_hashes = {path.relative_to(root).as_posix(): digest(path) for path in inherited_paths}
    for path in inherited_paths:
        target = output/'inputs'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    old = GeometryGuardSimulator(data['track'], data['rules'],
        rule_config=RuleConfig(tail_guard_m=1000.), margin_s=1.)
    old_policy = load_old_policy(checkpoint, expected_features=old.features, expected_action_space='mode_aware')
    _write(output/'prior_checkpoint.json', {'path': checkpoint.relative_to(root).as_posix(),
        'sha256': digest(checkpoint), 'frozen_r7_manifest_sha256': digest(prior/'R7/manifest.json')})
    baseline = old.simulate(old_policy, initial_speed_mps=saved_r7['summary']['speed_start_mps'],
        initial_energy_mj=2., action_space='mode_aware')
    baseline_proxy = proxy_for_r7(old, baseline)
    _write(output/'S0_proxy.json', baseline_proxy)
    record('S0', 'Availability clipped the historical request during HOLD/LIMITED',
        'Frozen R7 cold proxy failed; demand and actual supply were distinct',
        'Reconstruct cold baseline using original 31-feature executor', baseline,
        stronger_proxy=baseline_proxy)

    viable = []
    for notice in config['energy_notice_choices_s']:
        sim = make_sim(data, notice, config)
        controller = BlockController([.7], 'mode_aware')
        trial = sim.periodic_simulate(controller, action_space='mode_aware', max_cycles=4)
        candidate, laps = add(f'notice_{notice:g}', sim, controller, 'mode_aware', controller.to_dict(), trial)
        if candidate['qualification']['passed']:
            viable.append((candidate, sim, controller, laps))
    best_viable = min(viable, key=lambda v: v[0]['qualification']['last_two_mean_time_s']) if viable else None
    record('S1', 'Proactive energy reserve can make the declared reduction reachable',
        {'cold_baseline_proxy': baseline_proxy['additional_surrogate_hold_and_limited_request_gate_passed']},
        'Separate engineering viability from public rules; expose two hidden rule-state features',
        best_viable[3][-1] if best_viable else None, tested_notice_seconds=config['energy_notice_choices_s'],
        state_schema_dimension=33, all_candidates_saved=True)
    if best_viable is None:
        for i, amount in enumerate((0., .25, .5, .9, 1.), 2):
            sim = make_sim(data, .5, config)
            ctrl = BlockController([amount], 'mode_aware')
            result = sim.periodic_simulate(ctrl, action_space='mode_aware', max_cycles=4)
            candidate, laps = add(f'fallback_{i}', sim, ctrl, 'mode_aware', ctrl.to_dict(), result)
            record(f'S{i}', 'Reduce deployment to resolve an infeasible request intersection',
                'No S1 six-lap candidate passed; ML would learn a failed executor',
                f'Bounded deployment magnitude {amount:g}', laps[-1], qualification=candidate['qualification'])
        training_status = 'withheld_no_s1_physical_teacher'
    else:
        _, sim, _, _ = best_viable
        signed = search(sim, output/'S2', blocks=6, evaluations=config['signed_teacher_evaluations'], action_space='signed')
        signed_laps = None
        if signed['result'] is not None:
            _, signed_laps = add('signed_teacher', sim, signed['controller'], 'signed', signed['controller'].to_dict(), signed['result'])
        record('S2', 'Hard projection plateaus favor a finite numerical baseline',
            {'S1_time': best_viable[0]['qualification']['last_two_mean_time_s']},
            'Six-block signed coarse and coordinate search', signed['result'], search=signed['report'])
        aware = search(sim, output/'S3', blocks=8, evaluations=config['mode_aware_teacher_evaluations'],
            action_space='mode_aware', initial=None if signed['controller'] is None else signed['controller'].values)
        teacher_laps, teacher_qualification = None, None
        if aware['result'] is not None:
            teacher_candidate, teacher_laps = add('mode_aware_teacher', sim, aware['controller'], 'mode_aware', aware['controller'].to_dict(), aware['result'])
            teacher_qualification = teacher_candidate['qualification']
        record('S3', 'Braking-aware action mapping reduces unreachable exploration',
            {'S2_complete_count': signed['report']['complete_feasible_count']},
            'Eight-block mode-aware teacher; same full-period constraints', aware['result'], search=aware['report'])
        if teacher_laps is None or not teacher_qualification['passed']:
            record('S4', 'A failed teacher must not authorize policy training', 'S3 teacher missing or infeasible', 'Withhold BC', None)
            record('S5', 'Failed physics cannot be fixed by reward scaling', 'No certified teacher', 'Withhold cold SAC', None)
            record('S6', 'Warm state training also needs certified carried states', 'No certified teacher', 'Withhold warm SAC', None)
            training_status = 'withheld_no_s3_teacher'
        else:
            bc_runs = []
            for seed in config['seeds']:
                trained = train_bc(teacher_laps, sim, output/'S4'/str(seed), seed=seed,
                    epochs=config['bc_epochs'], dagger_rounds=config['dagger_rounds'],
                    dagger_epochs=config['dagger_epochs'], action_space='mode_aware',
                    initial_speed_mps=teacher_laps[0]['summary']['speed_start_mps'], progress=print)
                bc_runs.append(trained)
                if trained['policy'] is not None:
                    cp = f'S4/{seed}/best_bc_policy.npz'
                    save_policy(trained['policy'], output/cp)
                    add(f'BC_{seed}', sim, trained['policy'], 'mode_aware',
                        {'kind': 'policy', 'checkpoint': cp, 'action_space': 'mode_aware'}, trained['result'])
            bc_best = min((r['result'] for r in bc_runs if r['result'] is not None),
                key=lambda r: r['summary']['lap_time_s'], default=None)
            record('S4', 'A 33-feature actor can distinguish held reference and target',
                {'teacher_boundary_states': [r['summary']['energy_start_mj'] for r in teacher_laps]},
                'Retrain three-seed BC/DAgger using cold and carried teacher states', bc_best,
                metadata=[r['metadata'] for r in bc_runs])
            pool = [{'speed_mps': r['summary']['speed_start_mps'], 'energy_mj': r['summary']['energy_start_mj'],
                'carried_rule_state': r['summary']['initial_rule_state']} for r in teacher_laps[1:]]
            for condition, identifier in (('cold', 'S5'), ('warm', 'S6')):
                cases, representatives = [], []
                for seed, bc in zip(config['seeds'], bc_runs):
                    if bc['policy'] is None:
                        cases.append({'seed': seed, 'status': 'withheld_no_BC_incumbent'})
                        continue
                    trained = train_sac(sim, bc['policy'], output/identifier/str(seed), seed=seed,
                        episodes=config['sac_episodes_per_seed_per_condition'],
                        return_scale_s=config['return_scale_s'], initial_alpha=config['alpha_seconds'],
                        temperature_mode='fixed', mc_warmstart=True,
                        critic_warmup_updates=config['critic_warmup_updates'],
                        initial_speed_mps=teacher_laps[0]['summary']['speed_start_mps'],
                        initial_state_pool=pool if condition == 'warm' else None,
                        evaluation_initial_state=pool[-1] if condition == 'warm' else None,
                        periodic_evaluation=condition == 'cold', retain_rollouts=True, progress=print)
                    for episode_index, trajectory in enumerate(trained['rollouts'], 1):
                        stem = output/identifier/str(seed)/f'training_episode_{episode_index:03d}'
                        _write(stem.with_suffix('.json.gz'), trajectory)
                        _write(stem.with_suffix('.calibration.json'), score(trajectory['result'], data, 'calibration'))
                    cases.append(trained['metadata'])
                    if trained['policy'] is not None and trained['result'] is not None:
                        cp = f'{identifier}/{seed}/best_sac_policy.npz'
                        save_policy(trained['policy'], output/cp)
                        add(f'{identifier}_seed_{seed}', sim, trained['policy'], 'mode_aware',
                            {'kind': 'policy', 'checkpoint': cp, 'action_space': 'mode_aware'},
                            trained['result'])
                        representatives.append(trained['result'])
                representative = min(representatives, key=lambda r: r['summary']['lap_time_s'], default=None)
                record(identifier, 'Critic return scale and carried-state distribution need separate checks',
                    {'v6_MC_Q_error_seconds': 'about 38; shared underfit', 'condition': condition},
                    'Normalize internal Q and entropy consistently; '+condition+' initial-state training',
                    representative, cases=cases, new_episode_budget_maximum=60,
                    retained_BC_fallback_is_not_SAC_improvement=True)
            training_status = 'bounded_cold_and_warm_experiments_completed'

    eligible = [c for c in candidates if c['qualification']['passed']]
    winner = min(eligible, key=lambda c: c['qualification']['last_two_mean_time_s']) if eligible else None
    selection = {'winner': winner, 'ranking': config['selection'], 'observed_speed_residual_in_selection': False,
        'source_commit': freeze['code_commit'], 'new_global_optimality_proof': False}
    _write(output/'selection.json', selection)
    gates = {'cold_and_warm_control': bool(winner), 'calibration_accuracy': False,
        'development_accuracy': False, 'numerical_convergence': False,
        'new_session_test': False, 'full_FIA_ECU': False, 'true_SOC': False, 'formal_release': False}
    if winner:
        directory = winner['result_directory']
        selected = read(output/directory/'lap_6.json.gz')
        selected_sim = make_sim(data, winner['energy_notice_s'], config)
        _write(output/'selected_calibration.json', score(selected, data, 'calibration'))
        with (output/'development_claim.json').open('x', encoding='utf-8') as stream:
            json.dump({'selection_sha256': digest(output/'selection.json'), 'laps': [7, 9],
                'historical_reused_development': True, 'new_session_test': False}, stream)
        dev = score(selected, data, 'development')
        _write(output/'development_score.json', dev)
        numeric = numerical_audit(selected_sim, selected, output/'numerical')
        _write(output/'numerical_audit.json', numeric)
        gates.update(calibration_accuracy=read(output/'selected_calibration.json')['accuracy_passed'],
            development_accuracy=dev['accuracy_passed'], numerical_convergence=numeric['passed'])
        sensitivities = []
        for mass, power in ((780., 400.), (820., 400.), (800., 380.), (800., 420.)):
            from dataclasses import replace
            varied = ReachableSimulator(data['track'], data['rules'],
                vehicle=replace(selected_sim.vehicle, mass_kg=mass, ice_power_kw=power),
                rule_config=selected_sim.rule_config, margin_s=selected_sim.margin_s,
                energy_notice_s=selected_sim.energy_notice_s)
            controller = controller_from(winner['controller'], output, varied)
            varied_result = varied.periodic_simulate(controller, action_space=winner['action_space'], max_cycles=4)
            _write(output/f'sensitivity/mass_{mass:g}_power_{power:g}.json.gz', varied_result)
            sensitivities.append({'mass_kg': mass, 'ice_power_kw': power, 'summary': varied_result['summary'],
                'selection_input': False})
        _write(output/'sensitivity.json', sensitivities)
    inputs_unchanged = all((root/name).is_file() and digest(root/name) == value
        for group in ('tracked_input_sha256', 'dependency_wheels_sha256')
        for name, value in freeze[group].items())
    inherited_unchanged = all(digest(root/name) == value for name, value in inherited_hashes.items())
    ledger_after = {k: p.exists() for k, p in ledger_paths(root).items()}
    if not inputs_unchanged or not inherited_unchanged or ledger_after != freeze['FP2_ledger_before']:
        raise RuntimeError('Frozen source/data or external FP2 ledger changed during experiment')
    experiment = {'schema_version': 7, 'source_commit': freeze['code_commit'],
        'rounds': [r['round'] for r in rounds], 'candidates': candidates, 'winner': winner, 'gates': gates,
        'training_status': training_status, 'maximum_new_SAC_episodes': config['new_SAC_episode_maximum'],
        'FP2_accessed': False, 'new_session_status': '新会话验收待完成',
        'frozen_inputs_unchanged': inputs_unchanged, 'FP2_ledger_unchanged': ledger_after == freeze['FP2_ledger_before'],
        'inherited_ignored_inputs_sha256': inherited_hashes, 'inherited_inputs_unchanged': inherited_unchanged,
        'full_FIA_compliance_proven': False, 'true_SOC_proven': False, 'global_optimality_proven': False,
        'wall_seconds': time.perf_counter()-start_wall}
    _write(output/'experiment.json', experiment)
    _write(output/'manifest.json', {'source_commit': freeze['code_commit'],
        'files': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}})
    print(json.dumps({'output': str(output), 'winner': None if winner is None else winner['name'], 'gates': gates}, ensure_ascii=False), flush=True)
    return experiment


def verify(root, name='austria_2026_v7'):
    root = Path(root).resolve()
    if '/' in name or '\\' in name or name in ('', '.', '..'):
        raise ValueError('One result directory name required')
    output = root/'results'/name
    manifest, freeze, config, experiment = (read(output/f) for f in ('manifest.json', 'freeze.json', 'frozen_protocol.json', 'experiment.json'))
    for relative, expected in manifest['files'].items():
        path = (output/relative).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != expected:
            raise ValueError('Frozen result changed: '+relative)
    input_ok = all((root/name).is_file() and digest(root/name) == expected
        for group in ('tracked_input_sha256', 'dependency_wheels_sha256')
        for name, expected in freeze[group].items())
    dependencies_ok = {name: importlib.metadata.version(name) for name in freeze['dependencies']} == freeze['dependencies']
    if not input_ok or not dependencies_ok:
        raise ValueError('Frozen inputs or runtime dependencies differ')
    if git(root, 'rev-parse', freeze['code_commit']+'^{commit}') != freeze['code_commit']:
        raise ValueError('Frozen source commit unavailable')
    for name, expected in experiment['inherited_ignored_inputs_sha256'].items():
        if digest(output/'inputs'/Path(name).name) != expected:
            raise ValueError('Frozen inherited input copy changed')
    winner = experiment['winner']
    if winner is None:
        return {'passed': False, 'inventory_passed': True, 'reason': 'No six-lap candidate; diagnostics preserved'}
    data = load_local_dataset(root)
    sim = make_sim(data, winner['energy_notice_s'], config)
    controller = controller_from(winner['controller'], output, sim)
    saved = read(output/winner['result_directory']/'lap_6.json.gz')
    replay = sim.simulate(controller, initial_speed_mps=saved['summary']['speed_start_mps'],
        initial_energy_mj=saved['summary']['energy_start_mj'], carried_rule_state=saved['summary']['initial_rule_state'],
        action_space=winner['action_space'])
    replay['summary']['physical_lap_index'] = saved['summary']['physical_lap_index']
    trace_exact = all(np.array_equal(np.asarray(replay['internal_trace'][k]), np.asarray(v), equal_nan=True)
        for k, v in saved['internal_trace'].items())
    proxy_exact = _jsonable(proxy_audit(sim, replay)) == read(output/winner['result_directory']/'lap_6_proxy.json')
    development_exact = _jsonable(score(replay, data, 'development')) == read(output/'development_score.json')
    calibration_exact = _jsonable(score(replay, data, 'calibration')) == read(output/'selected_calibration.json')
    return {'passed': bool(trace_exact and proxy_exact and development_exact and calibration_exact),
        'source_hashes_passed': input_ok, 'dependency_versions_passed': dependencies_ok,
        'inventory_passed': True, 'exact_trace': trace_exact, 'exact_proxy': proxy_exact,
        'exact_saved_calibration': calibration_exact, 'exact_saved_development': development_exact,
        'new_selection_or_training': False, 'FP2_accessed': False,
        'accuracy_gates': experiment['gates'], 'formal_release': False}
