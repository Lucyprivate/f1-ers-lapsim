"""Bounded directional control diagnostics on actual carried boundaries.

This experiment is not a teacher search. It preserves the existing ideal
energy executor, mode-aware scalar action mapping and live public-rule FSM.
Every attempted single-lap execution is charged before creating its environment.
No actor is selected, no action Jacobian is advertised and no SAC is started.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np

from .._core.learning_base import _jsonable, _write
from .._core.metrics import score, sector_energy
from ..deploy_only import freeze_simulator, proxy_audit as frozen_proxy_audit
from .environment import proxy_audit, validate_boundary
from .teacher import BlockController

BLOCKS = 16
AMPLITUDE = .025
HALF_AMPLITUDE = .0125
MAX_SELECTED_BLOCKS = 4
MAX_CALLS = 132
RESPONSE_LIMITS = {'speed_rmse_kmh': 1e-6, 'lap_time_difference_s': 1e-6,
                   'energy_difference_mj': 1e-8}


def value_hash(value):
    """Hash the explicit JSON value, including the complete carried FSM."""
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('utf-8')
    return sha256(raw).hexdigest()


def _terminal_boundary(result):
    return {'speed_mps': result['terminal_speed_mps'],
        'energy_mj': result['summary']['energy_end_mj'],
        'carried_rule_state': result['rule_certificate']['terminal_boundary_state']}


def _boundary_after(result):
    return validate_boundary(_terminal_boundary(result))


def _phase_schedule(result):
    return [{'s_m': float(row['s_m']), 'phase': row['state_after_decide']['phase']}
            for row in result.get('energy_viability_audit', [])]


def _phase_changed(first, second):
    # Decision locations are positioned with the frozen 1e-5 m event tolerance.
    # Their final floating-point bit is not evidence of an FSM transition.
    return (len(first) != len(second) or any(a['phase'] != b['phase'] or
        abs(a['s_m']-b['s_m'])>1e-5 for a, b in zip(first, second)))


def _complete_public(result, proxy):
    summary = result['summary']
    return bool(summary.get('completed_lap') and summary.get('execution_feasible')
        and result.get('rule_certificate', {}).get('public_rules_passed',
                                                   summary.get('public_rules_passed', False))
        and proxy.get('passed'))


def _trace_checkpoint(env):
    """Retain the last accepted prefix, without synthesizing an end-of-lap score."""
    state = getattr(env, 'state', None)
    driver = getattr(env, 'driver', None)
    return {'diagnostic_only': True, 'formal_accuracy_scored': False,
        'last_accepted_state': asdict(state) if is_dataclass(state) else state,
        'accepted_records': getattr(env, 'records', []),
        'events': getattr(env, 'events', []), 'history': getattr(env, 'history', []),
        'requested_actions': getattr(env, 'raw_actions', []),
        'driver_rows': getattr(driver, 'rows', []),
        'failure': getattr(env, 'failure', None),
        'in_flight_unaccepted_substep_retained': False}


class _Execution:
    def __init__(self, sim, output, claim, provenance, calibration_data):
        self.sim, self.output, self.claim = sim, Path(output), claim
        self.provenance, self.calibration_data = deepcopy(provenance), calibration_data
        self.attempts = []

    def run(self, key, boundary, actions, *, simulator=None, numerics=None,
            output_ds_m=5., frozen=False, details=None):
        if not re.fullmatch(r'[a-z0-9_]+', key):
            raise ValueError('A stable single-lap attempt slug is required')
        if len(self.attempts) >= MAX_CALLS:
            raise RuntimeError('The fixed 132 single-lap diagnostic budget is exhausted')
        boundary = validate_boundary(boundary)
        # The shared durable claim is deliberately before environment creation.
        metadata = {'boundary_sha256': value_hash(boundary),
            'original_action_spec_sha256': value_hash(actions.actions if
                isinstance(actions, BlockController) else actions),
            'numerics': (numerics or self.sim.config).to_dict(),
            'output_ds_m': float(output_ds_m), 'frozen_raw_demands': bool(frozen),
            'provenance': self.provenance, **(details or {})}
        receipt = self.claim(key, metadata)
        directory = self.output/'attempts'/key
        directory.mkdir(parents=True, exist_ok=False)
        _write(directory/'claim.json', {'shared_receipt': receipt, **metadata,
            'initial_boundary': boundary})
        row = {'key': key, 'attempt_spent': True, **metadata,
               'status': 'claimed', 'formal_accuracy_scored': False}
        self.attempts.append(row)
        env = None
        actual_sim = simulator or self.sim
        try:
            # This is the same create_env/step/result path used by simulate,
            # retaining an environment handle for interrupted accepted prefixes.
            env = actual_sim.create_env(action_space='mode_aware',
                initial_speed_mps=boundary['speed_mps'],
                initial_energy_mj=boundary['energy_mj'],
                carried_rule_state=deepcopy(boundary['carried_rule_state']),
                output_ds_m=output_ds_m, numerics=numerics)
            if not callable(actions):
                actions = np.asarray(actions, float).copy()
                if (actions.shape != (len(env.driver.decision_s)-1,)
                        or not np.isfinite(actions).all() or np.any(abs(actions)>1.)):
                    raise ValueError('One finite original raw action per driver decision required')
            while not env.done:
                action = actions(env.get_obs()) if callable(actions) else float(actions[env.i])
                env.step(action)
            result = env.result()
            _write(directory/'trajectory.json.gz', result)
            auditor = frozen_proxy_audit if frozen else proxy_audit
            proxy = auditor(actual_sim, result)
            _write(directory/'proxy.json', proxy)
            full = _complete_public(result, proxy)
            row.update(status='completed_execution', physical_public_complete=full,
                control_qualified=bool(full and result['summary'].get('feasible')),
                summary=result['summary'], proxy=proxy,
                terminal_status=result.get('terminal_status'),
                requested_actions_sha256=value_hash(result['requested_actions']),
                driver_profile_sha256=value_hash(result['driver_profile']),
                phase_schedule=_phase_schedule(result),
                terminal_boundary_sha256=value_hash(_terminal_boundary(result)) if full else None)
            metrics = {'sector_energy': sector_energy(result, self.sim.sector_boundaries_m),
                'energy_swing_mj': float(np.ptp(result['internal_trace']['energy_mj'])),
                'balance_residual_mj': result['summary'].get('energy_balance_residual_mj'),
                'complete_physical_public': full,
                'control_qualified': row['control_qualified'],
                'real_SOC_or_full_FIA_validation_claimed': False}
            if self.calibration_data is not None:
                metrics['calibration'] = score(result, self.calibration_data, 'calibration')
                row['formal_accuracy_scored'] = bool(metrics['calibration']['aggregate'])
            _write(directory/'metrics.json', metrics)
            _write(directory/'execution.json', row)
            return {'result': result, 'row': row, 'boundary': boundary}
        except BaseException as exc:
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            row.update(status='interrupted' if interrupted else 'execution_or_audit_exception',
                error_type=type(exc).__name__, error=str(exc),
                physical_public_complete=False, control_qualified=False,
                formal_accuracy_scored=False)
            if env is not None:
                _write(directory/'accepted_prefix.json.gz', _trace_checkpoint(env))
            _write(directory/'execution.json', row)
            _write(self.output/'interrupted_or_failed_attempts.json', self.attempts)
            if interrupted:
                raise
            return {'result': None, 'row': row, 'boundary': boundary}


def _comparison(reference, changed, length_m, *, output=False):
    valid = bool(reference['row'].get('physical_public_complete')
                 and changed['row'].get('physical_public_complete'))
    row = {'valid': valid, 'partial_laps_scored': False}
    if not valid:
        return {**row, 'reason': 'Both laps must be complete, physical and public-rule feasible'}
    a, b = reference['result'], changed['result']
    common = np.linspace(0., length_m, int(np.ceil(length_m/.5))+1)
    def velocity(result):
        if output:
            return np.interp(common, np.r_[result['s_m'], length_m],
                np.r_[result['speed_mps'], result['terminal_speed_mps']])
        return np.interp(common, result['internal_trace']['s_m'],
                         result['internal_trace']['speed_mps'])
    speed = float(np.sqrt(np.mean((velocity(b)-velocity(a))**2))*3.6)
    delta = float(b['terminal_time_s']-a['terminal_time_s'])
    energy = float(b['summary']['energy_end_mj']-a['summary']['energy_end_mj'])
    phases_a, phases_b = _phase_schedule(a), _phase_schedule(b)
    phase_changed = _phase_changed(phases_a, phases_b)
    row.update(speed_rmse_kmh=speed, signed_lap_time_difference_s=delta,
        lap_time_difference_s=abs(delta), signed_energy_difference_mj=energy,
        energy_difference_mj=abs(energy),
        terminal_speed_difference_mps=float(b['terminal_speed_mps']-a['terminal_speed_mps']),
        phase_schedule_changed=phase_changed,
        response_interpretation='directional hybrid response' if phase_changed
            else 'directional response under unchanged recorded phase schedule',
        smooth_Jacobian_estimated=False)
    row['measurable'] = any(row[key]>limit for key, limit in RESPONSE_LIMITS.items())
    row['numerical_agreement_passed'] = bool(abs(delta)<=.1 and speed<=.5)
    return row


def _numeric_baseline(execution, reference, name):
    fixed = freeze_simulator(execution.sim, reference['result']['driver_profile'])
    actions = np.asarray(reference['result']['requested_actions'], float).copy()
    runs, comparisons = {}, {}
    for group, settings in (('output', ((10., .02, 1.), (5., .02, 1.), (2.5, .02, 1.))),
                            ('integration', ((5., .02, 1.), (5., .01, .5), (5., .005, .25)))):
        values = []
        for index, (ds, dt, max_ds) in enumerate(settings, 1):
            values.append(execution.run(f'{name}_numeric_{group}_{index:02d}',
                reference['boundary'], actions, simulator=fixed, frozen=True,
                output_ds_m=ds, numerics=replace(execution.sim.config, dt_max=dt, max_ds=max_ds),
                details={'category': 'baseline_numerics', 'source_attempt': reference['row']['key']}))
        rows = [_comparison(a, b, execution.sim.length_m, output=group=='output')
                for a, b in zip(values[:-1], values[1:])]
        comparisons[group] = rows
        runs[group] = [r['row']['key'] for r in values]
    return {'comparisons': comparisons, 'attempt_keys': runs,
        'passed': all(r.get('numerical_agreement_passed', False)
                      for group in comparisons.values() for r in group),
        'fixed_original_raw_actions_and_wheel_demands': True,
        'output_resampling_and_internal_convergence_separate': True}


def run_control_response(sim, output, *, initial_boundary, claim, provenance,
                         calibration_data=None):
    """Run the preregistered maximum 132 single-lap diagnostic executions.

    ``claim(key, details)`` must atomically charge a canonical external shared
    ledger, so branches and clones cannot reset this budget. ``provenance``
    contains at least ``code_commit``; callers additionally freeze source,
    configuration, data and dependency inventories before invocation. Results
    require a fresh destination. Fewer responsive blocks leave budget unused.
    """
    if not callable(claim):
        raise TypeError('A durable shared attempt-claim callback is required')
    if not re.fullmatch(r'[0-9a-f]{40}', str(provenance.get('code_commit', ''))):
        raise ValueError('The clean frozen code commit must be explicitly recorded')
    if abs(float(sim.rule_config.tail_guard_m)-1000.)>1e-9:
        raise ValueError('This diagnostic preserves the frozen 1000 m tail guard')
    if (abs(float(sim.vehicle.mass_kg)-800.)>1e-9 or
            abs(float(sim.vehicle.ice_power_kw)-400.)>1e-9):
        raise ValueError('Control response uses the unchanged qualified 800 kg / 400 kW baseline')
    baseline_parameters = {'mu': 1.8, 'drag_area_m2': 1.3, 'downforce_area_m2': 3.9,
        'equivalent_power_kw': 541.25, 'replay_brake_gain': .55,
        'storage_charge_efficiency': 1., 'storage_discharge_efficiency': 1.,
        'auxiliary_power_w': 0.}
    if (any(abs(float(getattr(sim.vehicle, key))-value)>1e-9
            for key, value in baseline_parameters.items()) or
            sim.vehicle.independent_replay_braking or sim.vehicle.axle_budget_enabled):
        raise ValueError('Control diagnostics cannot replace the qualified ideal baseline parameters')
    if (abs(float(sim.config.dt_max)-.02)>1e-12 or
            abs(float(sim.config.max_ds)-1.)>1e-12):
        raise ValueError('The original integration limits must be 0.02 seconds / 1 metre')
    if calibration_data is not None:
        from .physics import require_calibration
        require_calibration(calibration_data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    initial = validate_boundary(initial_boundary)
    execution = _Execution(sim, output, claim, provenance, calibration_data)
    report = {'status': 'running', 'maximum_single_lap_calls': MAX_CALLS,
        'planned_allocation': {'actual_carry_baseline': 6, 'original_responses': 66,
            'half_amplitude_responses': 16, 'baseline_numerics': 12,
            'selected_original_internal_refinements': 32},
        'blocks': BLOCKS, 'amplitude': AMPLITUDE, 'half_amplitude': HALF_AMPLITUDE,
        'measurable_response_limits': RESPONSE_LIMITS, 'provenance': deepcopy(provenance),
        'SAC_attempts': 0, 'teacher_optimization_calls': 0, 'new_teacher_adopted': False,
        'global_or_six_carry_optimality_proven': False, 'FP2_accessed': False,
        'smooth_Jacobian_estimated': False,
        'boundary_pool_source': 'cold and actual terminal state after six carried baseline laps',
        'geometry_source': 'unchanged qualified baseline, independent of new positional route'}
    _write(output/'protocol.json', report)
    carried = initial
    baseline_runs = []
    for index in range(1, 7):
        run = execution.run(f'carry_{index:02d}', carried, BlockController(np.zeros(BLOCKS)),
                            details={'category': 'actual_carry_baseline', 'lap_index': index})
        baseline_runs.append(run)
        if not run['row'].get('control_qualified'):
            report.update(status='baseline_actual_carry_qualification_failed',
                reason='The original baseline must qualify before directional diagnostics')
            break
        carried = _boundary_after(run['result'])
    crossings = [{'from': a['row']['key'], 'to': b['row']['key'],
        'speed_error_mps': b['boundary']['speed_mps']-a['result']['terminal_speed_mps'],
        'energy_error_mj': b['boundary']['energy_mj']-a['result']['summary']['energy_end_mj'],
        'dc_error_kw': float((b['boundary']['carried_rule_state'] or {}).get('previous_dc_kw', 0.))
            -a['result']['summary']['terminal_dc_kw'],
        'FSM_equal': b['boundary']['carried_rule_state']==a['result']['rule_certificate']['terminal_boundary_state']}
        for a, b in zip(baseline_runs[:-1], baseline_runs[1:]) if a['result'] is not None]
    qualification = bool(len(baseline_runs)==6 and all(r['row'].get('control_qualified')
        for r in baseline_runs) and all(abs(c['speed_error_mps'])<1e-10 and
        abs(c['energy_error_mj'])<1e-10 and abs(c['dc_error_kw'])<1e-10
        and c['FSM_equal'] for c in crossings))
    report['baseline_actual_carry'] = {'passed': qualification, 'crossings': crossings,
        'attempt_keys': [r['row']['key'] for r in baseline_runs],
        'within_lap_speed_or_energy_resets': 0, 'policy_specific_boundary_shooting': False}
    if not qualification:
        report.update(attempted_single_lap_calls=len(execution.attempts), attempts=execution.attempts)
        _write(output/'control_response.json', report)
        return report
    pool = {'cold': initial, 'carried_late': carried}
    _write(output/'boundary_pool.json', {'boundaries': pool,
        'sha256': {name: value_hash(value) for name, value in pool.items()}})
    references, original, selections, directional = {}, {}, {}, {}
    # Original amplitude screening is exhaustive over the fixed 16 blocks.
    for name, boundary in pool.items():
        reference = execution.run(f'{name}_base', boundary, BlockController(np.zeros(BLOCKS)),
                                 details={'category': 'original_responses', 'baseline': True})
        references[name] = reference
        original[name], directional[name] = {}, []
        for block in range(BLOCKS):
            signs = {}
            for label, sign in (('neg', -1.), ('pos', 1.)):
                actions = np.zeros(BLOCKS); actions[block] = sign*AMPLITUDE
                run = execution.run(f'{name}_b{block+1:02d}_{label}_025', boundary,
                    BlockController(actions), details={'category': 'original_responses',
                    'block': block+1, 'delta_action': sign*AMPLITUDE})
                comparison = _comparison(reference, run, sim.length_m)
                run['comparison'] = comparison; signs[label] = run
                directional[name].append({'block': block+1, 'sign': label,
                    'attempt_key': run['row']['key'], **comparison})
            original[name][block] = signs
        selections[name] = [block for block, signs in original[name].items()
            if all(r['comparison']['valid'] for r in signs.values())
            and any(r['comparison'].get('measurable') for r in signs.values())][:MAX_SELECTED_BLOCKS]
    _write(output/'directional_responses.json', directional)
    half_results = {}
    for name, blocks in selections.items():
        half_results[name] = []
        for block in blocks:
            for label, sign in (('neg', -1.), ('pos', 1.)):
                actions = np.zeros(BLOCKS); actions[block] = sign*HALF_AMPLITUDE
                run = execution.run(f'{name}_b{block+1:02d}_{label}_0125', pool[name],
                    BlockController(actions), details={'category': 'half_amplitude_responses',
                    'block': block+1, 'delta_action': sign*HALF_AMPLITUDE})
                comparison = _comparison(references[name], run, sim.length_m)
                full = original[name][block][label]['comparison']
                consistency = {}
                if comparison['valid'] and full['valid']:
                    consistency = {'twice_half_minus_full_time_difference_s':
                        2*comparison['signed_lap_time_difference_s']-full['signed_lap_time_difference_s'],
                        'twice_half_minus_full_energy_difference_mj':
                        2*comparison['signed_energy_difference_mj']-full['signed_energy_difference_mj']}
                half_results[name].append({'block': block+1, 'sign': label,
                    'attempt_key': run['row']['key'], **comparison,
                    'amplitude_consistency_diagnostic': consistency})
    numerical = {'baseline': {}, 'selected_original': {}}
    for name, reference in references.items():
        if reference['row'].get('physical_public_complete'):
            numerical['baseline'][name] = _numeric_baseline(execution, reference, name)
        else:
            numerical['baseline'][name] = {'passed': False, 'reason': 'Baseline response is incomplete'}
    for name, blocks in selections.items():
        numerical['selected_original'][name] = []
        for block in blocks:
            for label, sign in (('neg', -1.), ('pos', 1.)):
                reference = original[name][block][label]
                fixed = freeze_simulator(sim, reference['result']['driver_profile'])
                raw_actions = np.asarray(reference['result']['requested_actions'], float).copy()
                values = [reference]
                for index, (dt, ds) in enumerate(((.01, .5), (.005, .25)), 1):
                    values.append(execution.run(f'{name}_b{block+1:02d}_{label}_025_refine_{index:02d}',
                        reference['boundary'], raw_actions, simulator=fixed, frozen=True,
                        numerics=replace(sim.config, dt_max=dt, max_ds=ds),
                        details={'category': 'selected_original_internal_refinements',
                            'source_attempt': reference['row']['key'], 'block': block+1,
                            'delta_action': sign*AMPLITUDE}))
                rows = [_comparison(a, b, sim.length_m) for a, b in zip(values[:-1], values[1:])]
                numerical['selected_original'][name].append({'block': block+1, 'sign': label,
                    'attempt_keys': [v['row']['key'] for v in values], 'comparisons': rows,
                    'passed': all(r.get('numerical_agreement_passed', False) for r in rows),
                    'fixed_original_raw_actions_and_wheel_demands': True})
    numerical_passed = bool(all(r['passed'] for r in numerical['baseline'].values()) and
        all(r['passed'] for rows in numerical['selected_original'].values() for r in rows))
    any_response = any(selections.values())
    report.update(status='directional_diagnostics_completed' if numerical_passed and any_response
            else 'response_unidentifiable_or_numerically_unstable',
        attempted_single_lap_calls=len(execution.attempts), attempts=execution.attempts,
        selected_blocks={name: [block+1 for block in blocks] for name, blocks in selections.items()},
        directional_responses=directional, half_amplitude_responses=half_results,
        numerics=numerical, numerical_checks_passed=numerical_passed,
        measurable_response_on_either_boundary=any_response,
        derivative_or_redistribution_dimension_claimed=False,
        unspent_budget_reallocated=False, within_lap_speed_or_energy_resets=0)
    _write(output/'control_response.json', report)
    return report
