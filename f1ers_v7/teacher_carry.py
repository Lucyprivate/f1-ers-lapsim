"""Finite control search ranked only by physically carried sustainable laps.

The family and random seed are fixed before running.  Real observed speed or
its residual never enters selection.  Every proposal carries speed, energy,
actual DC and the declared reduction state through six consecutive laps.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from f1ers_v6.learning import _write
from f1ers_v6.teacher import BlockController
from .simulator import proxy_audit


def proposal_registry(initial_values, *, blocks=6, seed=2026):
    initial = np.asarray(initial_values, float).reshape(-1)
    if (initial.size == 0 or not np.isfinite(initial).all()
            or np.any(initial < 0.) or np.any(initial > 1.)):
        raise ValueError('Initial block magnitudes must be finite in [0,1]')
    if not isinstance(blocks, int) or blocks <= 0:
        raise ValueError('A positive integer block count is required')
    if len(initial) != blocks:
        initial = np.interp((np.arange(blocks)+.5)/blocks,
                            (np.arange(len(initial))+.5)/len(initial), initial)
    rng = np.random.default_rng(seed)
    values = [initial.copy(), np.zeros(blocks), .75*initial]
    labels = ['inherited_eligible_baseline', 'zero_deployment_control', 'reduced_baseline_75_percent']
    for index in range(5):
        values.append(np.clip(initial+rng.normal(0., .15, blocks), 0., 1.))
        labels.append(f'fixed_seeded_perturbation_{index+1}')
    return [{'index': index, 'kind': kind, 'values': value.tolist()}
            for index, (kind, value) in enumerate(zip(labels, values))]


def carry_search(sim, output, *, initial_values, action_space='mode_aware',
                 initial_speed_mps, evaluations=8, blocks=6, seed=2026,
                 progress=print):
    """Run a bounded eight-proposal search without independently closing laps.

    Return the best controller, its six actual carried laps, the last lap and
    all proposal qualifications.  ``result`` is a physical warm lap, never a
    separately solved cold proxy.  If no proposal is fully feasible, these
    selected fields are ``None`` and every failure remains on disk.
    """
    if action_space not in ('signed', 'mode_aware'):
        raise ValueError('Only the declared signed or braking-aware mapping is supported')
    if not isinstance(evaluations, int) or not 3 <= evaluations <= 8:
        raise ValueError('Carry search requires three baseline controls and at most eight proposals')
    if not np.isfinite(initial_speed_mps) or float(initial_speed_mps) <= 0.:
        raise ValueError('One fixed finite positive initial speed is required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('An existing carry search cannot be overwritten or rescored')
    output.mkdir(parents=True, exist_ok=True)
    registry = proposal_registry(initial_values, blocks=blocks, seed=seed)[:evaluations]
    settings = {'algorithm': 'frozen finite six-lap carried control search',
                'seed': seed, 'blocks': blocks, 'evaluations': evaluations,
                'initial_speed_mps': float(initial_speed_mps), 'initial_energy_mj': 2.,
                'physical_laps_per_proposal': 6, 'objective_laps': [5, 6],
                'action_space': action_space, 'registry': registry,
                'independent_per_lap_periodic_shooting': False,
                'observed_speed_or_residual_used': False,
                'original_physics_or_thresholds_modified': False,
                'global_optimality_claimed': False}
    _write(output/'frozen_registry.json', settings)
    history = []
    best, best_laps, best_qualification, best_directory = None, None, None, None
    best_time = float('inf')
    for proposal in registry:
        values = np.asarray(proposal['values'], float)
        controller = BlockController(values, action_space)
        trial = output/f"trial_{proposal['index']:03d}"
        physical = sim.simulate_laps(controller, n_laps=6,
            initial_speed_mps=float(initial_speed_mps), initial_energy_mj=2.,
            action_space=action_space)
        summaries, proxy_summaries, reasons = [], [], []
        laps = physical['laps']
        for index, result in enumerate(laps, 1):
            _write(trial/f'lap_{index}.json.gz', result)
            proxy = proxy_audit(sim, result)
            _write(trial/f'lap_{index}_proxy.json', proxy)
            summary = result['summary']
            summaries.append(summary)
            proxy_summaries.append({
                'lap': index, 'passed': bool(proxy['passed']),
                'reachable_request_gate_passed': proxy['reachable_request_gate_passed'],
                'all_HOLD_LIMITED_proxy_queries_consistent':
                    proxy['additional_surrogate_hold_and_limited_request_gate_passed'],
                'engineering_violations': proxy['engineering_violations']})
            if not summary.get('feasible', False):
                failed = [name for name in ('completed_lap', 'execution_feasible', 'public_rules_passed',
                    'sustainable', 'periodic_speed_closed', 'dc_boundary_closed', 'rule_state_closed',
                    'reachable_request_gate_passed') if not summary.get(name, False)]
                reasons.append({'lap': index, 'failed_gates': failed,
                    'status': summary.get('status'),
                    'distance_completed_m': summary.get('distance_completed_m'),
                    'net_energy_mj': summary.get('energy_end_mj', 0.)-summary.get('energy_start_mj', 0.)})
            if not proxy['passed']:
                reasons.append({'lap': index, 'failed_gates': ['all_phase_request_proxy_consistency'],
                                'engineering_violations': proxy['engineering_violations']})
        if len(laps) != 6:
            reasons.append({'failed_gates': ['six_physical_laps'], 'physical_laps': len(laps)})
        if not physical['physical_control_line_continuity']:
            reasons.append({'failed_gates': ['physical_control_line_continuity'],
                            'crossings': physical['crossings']})
        passed = bool(len(laps) == 6 and physical['physical_control_line_continuity']
                      and all(s.get('feasible', False) for s in summaries)
                      and all(p['passed'] for p in proxy_summaries))
        complete = bool(len(laps) == 6 and all(s.get('completed_lap', False) for s in summaries))
        diagnostic_time = float(np.mean([s['lap_time_s'] for s in summaries[-2:]])) if complete else None
        objective = diagnostic_time if passed else None
        improved = bool(passed and objective < best_time-1e-8)
        qualification = {'passed': passed, 'physical_laps': len(laps),
            'initial_speed_mps': float(initial_speed_mps), 'initial_energy_mj': 2.,
            'physical_control_line_continuity': physical['physical_control_line_continuity'],
            'crossings': physical['crossings'], 'lap_summaries': summaries,
            'proxy_summaries': proxy_summaries, 'rejection_reasons': reasons,
            'formal_objective_s': objective,
            'last_two_mean_time_s': objective,
            'diagnostic_last_two_complete_lap_mean_time_s': diagnostic_time,
            'diagnostic_mean_is_formal_when_rejected': False,
            'within_lap_velocity_or_energy_resets': 0,
            'infinite_horizon_proven': False}
        _write(trial/'qualification.json', qualification)
        row = {**proposal, 'passed': passed, 'selected_incumbent': improved,
               'objective_s': objective, 'qualification_path':
                   (trial/'qualification.json').relative_to(output).as_posix(),
               'rejection_reasons': reasons}
        history.append(row)
        _write(output/'search_history.json', history)
        if improved:
            best, best_laps, best_qualification = controller, laps, qualification
            best_directory = trial
            best_time = objective
        progress(f"carry teacher {proposal['index']+1}/{evaluations}: "
                 +(f'{objective:.6f}s' if passed else 'rejected'), flush=True)
    report = {**settings, 'history': history,
              'complete_six_lap_feasible_count': sum(h['passed'] for h in history),
              'best': None if best is None else best.to_dict(),
              'best_time_s': best_time if np.isfinite(best_time) else None,
              'all_failed_proposal_traces_retained': True}
    _write(output/'teacher.json', report)
    return {'controller': best, 'result': None if best_laps is None else best_laps[-1],
            'laps': best_laps, 'qualification': best_qualification, 'report': report,
            'selected_directory': best_directory}


CarrySearch = carry_search
