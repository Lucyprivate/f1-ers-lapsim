"""Calibration-only route diagnosis before one bounded two-path physical refit."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import numpy as np

from .._core import input_channels as channels
from .._core.input_io import read_json
from .._core.learning_base import _write
from .._core.provenance import read, digest
from .study_protocol import (IDENTITY, CANONICAL_RESEARCH_LEDGER, AttemptBudget,
                             write_exclusive)
from .data import ROLES, load_historical, load_role_source


def load_frozen_view(data_root, public_root, spec, role='calibration'):
    """Consume a frozen positional route; fitting never loads a development view."""
    if role not in ROLES:
        raise ValueError('Only fixed historical calibration/development views supported')
    from .geometry_registration import build_track_from_spec, remap_rules
    document, rows, arrays = load_role_source(data_root, role)
    rows = [r for r in rows if r['role'] == role]
    if tuple(int(r['lap_number']) for r in rows) != ROLES[role]:
        raise ValueError('Historical lap identity changed')
    registration = spec['registration']
    if registration['fit_laps'] != [3, 4, 5, 6]:
        raise ValueError('Final common route requires calibration-only registration')
    track = build_track_from_spec(data_root, spec, ds_m=5.)
    laps = [channels.process_lap(r, arrays, track, registration, strict_signal_window=True) for r in rows]
    audit = {'role': role, 'test_arrays_loaded': False, 'geometry': track['geometry_audit'],
             'registration': registration, 'read_only_external_inputs': True,
             'data_gate_passed': bool(all(l['alignment_audit']['grid_bracketed'] and
                 .96 <= l['alignment_audit']['projected_span_fraction'] <= 1.04 and
                 np.isfinite(l['speed_mps']).all() for l in laps)),
             'alignment_per_lap': [{'lap_number': l['lap_number'], **l['alignment_audit']} for l in laps]}
    return {'track': track, 'rules': remap_rules(read_json(Path(public_root)/'config/rules.json'), track),
            'laps': laps, 'audit': audit, 'session': document['session'], 'driver': document['driver'],
            'registration': registration,
            'boundary_reference': np.loadtxt(Path(data_root)/'data/raw/tum/Spielberg_centerline.csv', delimiter=',')}


def run_geometry(data_root, root, output, config):
    from .geometry_registration import audit_geometry
    from .experiment import make_sim
    from .audit import audit_inputs
    folder = Path(output)/'geometry'
    spec, report = audit_geometry(data_root, root, output_dir=folder)
    # The inherited route remains a separately named diagnostic baseline.
    cal = load_historical(data_root, root, role='calibration')
    hypotheses = audit_inputs(cal, make_sim(cal, config))
    _write(folder/'kinematics_baseline.json', hypotheses)
    from .physics import _sim
    baseline_params = make_sim(cal, config).vehicle
    mass_sensitivity = [
        {'mass_kg': mass, 'engineering_assumption': True,
         'diagnostic': audit_inputs(cal, _sim(cal, replace(baseline_params, mass_kg=mass), config))}
        for mass in config['diagnostic_mass_sensitivity_kg']]
    _write(folder/'mass_kinematics_sensitivity.json', mass_sensitivity)
    write_exclusive(Path(output)/'route_freeze.json', spec)
    return {'geometry': report, 'geometry_gate_passed': bool(report['geometry_gate_passed']),
            'route_freeze_sha256': digest(Path(output)/'route_freeze.json'),
            'kinematics_baseline': hypotheses, 'formal_prediction': False,
            'mass_sensitivity': mass_sensitivity,
            'development_loaded': False, 'FP2_accessed': False}


def run_refit(data_root, root, output, config):
    from .physics import (sobol_viability, capped_fit, evaluate_model,
                          selected_numerical_audit, PARAMETERS, BOUNDS, _json_finite)
    geometry = read(Path(output)/'geometry-audit.json')
    if not geometry['geometry_gate_passed']:
        return {'status': 'withheld_geometry_gate', 'screen_points_attempted': 0,
                'fit_evaluations_attempted': 0, 'development_loaded': False,
                'reason': 'No positional route passed every fold and continuous geometry gate',
                'accuracy_improvement_demonstrated': False, 'FP2_accessed': False}
    spec = read(Path(output)/'route_freeze.json')
    if digest(Path(output)/'route_freeze.json') != geometry['route_freeze_sha256']:
        raise RuntimeError('Frozen route changed before the physical screen')
    data = load_frozen_view(data_root, root, spec)
    if not data['audit']['data_gate_passed']:
        return {'status': 'withheld_calibration_input_quality', 'screen_points_attempted': 0,
                'fit_evaluations_attempted': 0, 'development_loaded': False,
                'calibration_audit': data['audit'], 'FP2_accessed': False}
    candidate = {'name': 'registered_boundary_route_independent_braking',
                 'independent_replay_braking': True, 'axle_budget_enabled': False}
    folder = Path(output)/'physics'
    folder.mkdir(exist_ok=False)
    ledger = CANONICAL_RESEARCH_LEDGER/IDENTITY
    screen_budget = AttemptBudget(ledger/'physics_screen', 64)
    fit_budgets = [AttemptBudget(ledger/f'physics_fit_{i}', 600) for i in (1, 2, 3)]
    def evaluate(values, key):
        details = {'parameters': np.asarray(values).tolist(), 'candidate': candidate}
        if key.startswith('screen_'):
            screen_budget.claim(key, details)
        else:
            index = int(key.split('_')[1])-1
            fit_budgets[index].claim(key, details)
        return evaluate_model(data, values, candidate, folder/key, config,
                              initial_speed_mps=config['cold_initial_speed_mps'])
    screening = sobol_viability(evaluate)
    _write(folder/'screening.json', _json_finite(screening))
    fitting = capped_fit(evaluate, screening,
                         max_iterations=config['fit_max_iterations_per_start'],
                         max_evaluations=config['fit_max_evaluations_per_start'])
    _write(folder/'fit.json', _json_finite(fitting))
    result = {'status': fitting['status'], 'screen_points_attempted': screen_budget.counts()['attempted'],
              'screen_viable_points': screening['viable_points'], 'fit': fitting,
              'fit_evaluations_attempted': sum(b.counts()['attempted'] for b in fit_budgets),
              'selected_parameters': fitting['selected_parameters'], 'development_loaded': False,
              'candidate': candidate, 'accuracy_improvement_demonstrated': False,
              'FP2_accessed': False, 'control_qualification_used_for_model_fit': False}
    if not fitting['converged']:
        return _json_finite(result)
    binding = {'route_freeze_sha256': geometry['route_freeze_sha256'],
               'parameters': fitting['selected_parameters'], 'candidate': candidate,
               'estimated_laps': [3, 4, 5, 6], 'development_not_yet_loaded': True}
    write_exclusive(folder/'parameter_freeze.json', binding)
    # Late historical laps become visible only after route and all fit parameters freeze.
    dev = load_frozen_view(data_root, root, spec, 'development')
    result['development_loaded'] = True
    development = evaluate_model(dev, fitting['selected_parameters'], candidate,
                                 folder/'development_once', config, role='development',
                                 initial_speed_mps=config['cold_initial_speed_mps'])
    result['development'] = development
    selected = {'selected_parameters': fitting['selected_parameters'],
                'candidate': candidate, 'fit': fitting}
    numeric = selected_numerical_audit(data, selected, folder/'numerics', config)
    result['numerical'] = numeric
    result['two_path_development_accuracy_passed'] = bool(
        development['replay_accuracy_gates']['accuracy_passed'] and
        development['autonomous_accuracy_gates']['accuracy_passed'])
    # Local sensitivities describe uncertainty; no extra model or policy is selected.
    values = np.asarray(fitting['selected_parameters'])
    sensitivity = []
    sensitivity_budget = AttemptBudget(ledger/'parameter_sensitivity', 10)
    for i, name in enumerate(PARAMETERS):
        for sign in (-1, 1):
            trial = values.copy()
            trial[i] = np.clip(trial[i]+sign*.01*(BOUNDS[i, 1]-BOUNDS[i, 0]), *BOUNDS[i])
            sensitivity_budget.claim(f'{name}_{"lower" if sign<0 else "upper"}', {'parameters': trial.tolist()})
            row = evaluate_model(data, trial, candidate, folder/f'sensitivity_{name}_{sign:+d}', config,
                                 initial_speed_mps=config['cold_initial_speed_mps'])
            sensitivity.append({'parameter': name, 'direction': sign, 'evaluation': row})
    result['parameter_sensitivity'] = sensitivity
    result['bound_hits'] = [name for name, x, bounds in zip(PARAMETERS, values, BOUNDS)
                            if min(abs(x-bounds[0]), abs(x-bounds[1])) < 1e-5]
    result['accuracy_improvement_demonstrated'] = bool(numeric['passed'] and result['two_path_development_accuracy_passed'])
    result['new_session_acceptance'] = 'pending; FP2 remains sealed'
    return _json_finite(result)


def run_response(data_root, root, output, config, frozen):
    from .control_response import run_control_response
    from .experiment import make_sim
    cal = load_historical(data_root, root, role='calibration')
    budget = AttemptBudget(CANONICAL_RESEARCH_LEDGER/IDENTITY/'control_response', 132)
    return run_control_response(make_sim(cal, config), Path(output)/'control_response',
        initial_boundary={'speed_mps': config['cold_initial_speed_mps'],
                          'energy_mj': 2., 'carried_rule_state': None},
        claim=budget.claim, provenance={'identity': IDENTITY, 'code_commit': frozen['code_commit'],
            'data_sha256': frozen['data_sha256'], 'config': config}, calibration_data=cal)
