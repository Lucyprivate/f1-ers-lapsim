"""Frozen final scorer for the separate global tail-guard experiment."""
from pathlib import Path
import json
import time
import numpy as np

from f1ers_v4.data import load_local_dataset
from f1ers_v4.protocol import ledger_paths
from f1ers_v6.learning import _write, _jsonable
from f1ers_v6.metrics import score
from f1ers_v6.rules import RuleConfig
from f1ers_v7.experiment import digest, read, controller_from, git
from f1ers_v7.simulator import ReachableSimulator, proxy_audit, numerical_audit
from f1ers_v7.tail_audit import tail_guard_audit, safe_file


def run(root, name='austria_2026_v7C'):
    root = Path(root).resolve()
    started = time.perf_counter()
    audit = tail_guard_audit(root, name=name)
    output, report = audit['output'], audit['report']
    selected = audit['selected_lap6']
    gates = dict(report['gates'])
    gates.update(new_session_test=False, development_accuracy=False, numerical_convergence=False)
    if selected is not None:
        # The selection was written by the calibration-only audit before this
        # historical development read. This is not another unseen test.
        with (output/'development_claim.json').open('x', encoding='utf-8') as stream:
            json.dump({'selection_sha256': digest(output/'selection.json'), 'laps': [7, 9],
                'historical_reused_development': True, 'new_session_test': False}, stream)
        development = score(selected, audit['data'], 'development')
        _write(output/'development_score.json', development)
        numeric = numerical_audit(audit['simulator'], selected, output/'numerical')
        _write(output/'numerical_audit.json', numeric)
        gates.update(development_accuracy=development['accuracy_passed'], numerical_convergence=numeric['passed'])
    freeze = read(output/'freeze.json')
    unchanged = all((root/name).is_file() and digest(root/name) == expected
        for group in ('tracked_input_sha256', 'dependency_wheels_sha256')
        for name, expected in freeze[group].items())
    ledger_unchanged = {k: p.exists() for k, p in ledger_paths(root).items()} == freeze['FP2_ledger_before']
    prior = root/'results'/read(output/'frozen_protocol.json')['prior_name']
    bindings = read(output/'inputs/input_bindings.json')
    prior_unchanged = all(digest(safe_file(prior, name)) == expected
        for name, expected in bindings['prior_input_sha256'].items())
    if not (unchanged and ledger_unchanged and prior_unchanged):
        raise RuntimeError('Frozen inputs or shared FP2 ledger changed during final scoring')
    experiment = {**report, 'gates': gates, 'rounds': ['C0_fixed_tail_comparison', 'C1_final_numerical_scoring'],
        'frozen_inputs_unchanged': unchanged, 'FP2_ledger_unchanged': ledger_unchanged,
        'prior_inputs_unchanged': prior_unchanged, 'new_session_status': '新会话验收待完成',
        'full_FIA_compliance_proven': False, 'true_SOC_proven': False,
        'actual_new_SAC_episodes_requested': 0, 'actual_new_SAC_episodes_completed': 0,
        'wall_seconds': time.perf_counter()-started}
    _write(output/'experiment.json', experiment)
    _write(output/'manifest.json', {'source_commit': freeze['code_commit'],
        'files': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}})
    print(json.dumps({'winner': None if report['winner'] is None else report['winner']['name'],
        'gates': gates}, ensure_ascii=False), flush=True)
    return experiment


def verify(root, name='austria_2026_v7C'):
    root = Path(root).resolve()
    if '/' in name or '\\' in name or name in ('', '.', '..'):
        raise ValueError('One safe result directory name required')
    output = root/'results'/name
    manifest, freeze, config, experiment = (read(output/f) for f in
        ('manifest.json', 'freeze.json', 'frozen_protocol.json', 'experiment.json'))
    for relative, expected in manifest['files'].items():
        if digest(safe_file(output, relative)) != expected:
            raise ValueError('Frozen result changed: '+relative)
    input_ok = all((root/name).is_file() and digest(root/name) == expected
        for group in ('tracked_input_sha256', 'dependency_wheels_sha256')
        for name, expected in freeze[group].items())
    import importlib.metadata
    dependencies_ok = {name: importlib.metadata.version(name) for name in freeze['dependencies']} == freeze['dependencies']
    if not input_ok or not dependencies_ok:
        raise ValueError('Frozen inputs or runtime dependencies differ')
    if git(root, 'rev-parse', freeze['code_commit']+'^{commit}') != freeze['code_commit']:
        raise ValueError('Frozen source commit unavailable')
    bindings = read(output/'inputs/input_bindings.json')
    for name, expected in bindings['prior_input_sha256'].items():
        filename = Path(name).name if name in ('manifest.json', 'selection.json') else 'selected_controller_v7.npz'
        if digest(output/'inputs'/filename) != expected:
            raise ValueError('Frozen inherited input copy changed')
    winner = experiment['winner']
    if winner is None:
        return {'passed': False, 'inventory_passed': True, 'reason': 'No six-lap candidate; failures retained'}
    data = load_local_dataset(root)
    sim = ReachableSimulator(data['track'], data['rules'],
        rule_config=RuleConfig(tail_guard_m=winner['tail_guard_m']),
        margin_s=config['geometry_margin_s'], energy_notice_s=winner['energy_notice_s'])
    controller = controller_from(winner['controller'], output, sim)
    saved = read(output/winner['result_directory']/'lap_6.json.gz')
    replay = sim.simulate(controller, initial_speed_mps=saved['summary']['speed_start_mps'],
        initial_energy_mj=saved['summary']['energy_start_mj'], carried_rule_state=saved['summary']['initial_rule_state'],
        action_space=winner['action_space'])
    replay['summary']['physical_lap_index'] = saved['summary']['physical_lap_index']
    exact = {
        'exact_trace': all(np.array_equal(np.asarray(replay['internal_trace'][k]), np.asarray(v), equal_nan=True)
            for k, v in saved['internal_trace'].items()),
        'exact_proxy': _jsonable(proxy_audit(sim, replay)) == read(output/winner['result_directory']/'lap_6_proxy.json'),
        'exact_saved_development': _jsonable(score(replay, data, 'development')) == read(output/'development_score.json'),
        'exact_saved_calibration': _jsonable(score(replay, data, 'calibration')) == read(output/'selected_calibration.json')}
    return {'passed': all(exact.values()), **exact, 'source_hashes_passed': input_ok,
        'dependency_versions_passed': dependencies_ok, 'inventory_passed': True,
        'new_selection_or_training': False, 'FP2_accessed': False,
        'accuracy_gates': experiment['gates'], 'formal_release': False}
