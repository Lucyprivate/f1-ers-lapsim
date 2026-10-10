"""Sector errors and accepted-state energy accounting, outside policy inputs."""
from __future__ import annotations
import numpy as np


def sector_energy(result, boundaries):
    trace = result['internal_trace']
    if 'dc_kw' not in trace or 'energy_mj' not in trace:
        return [{'sector':i+1,'available':False,'reason':'No electrical ledger in historical execution'} for i in range(3)]
    s, t = np.asarray(trace['s_m']), np.asarray(trace['time_s'])
    dc = np.asarray(trace['dc_kw'])
    dt = np.diff(t)
    energy = np.asarray(trace['energy_mj'])
    deploy = np.r_[0., np.cumsum(np.maximum(dc, 0.)*dt/1000.)]
    harvest = np.r_[0., np.cumsum(np.maximum(-dc, 0.)*dt/1000.)]
    bounds = np.r_[0., boundaries, result['length_m']]
    rows = []
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        if end > s[-1]+1e-5:
            rows.append({'sector': i+1, 'complete': False, 'energy_not_extrapolated': True})
            continue
        es, ee = np.interp([start, end], s, energy)
        dep = np.diff(np.interp([start, end], s, deploy))[0]
        har = np.diff(np.interp([start, end], s, harvest))[0]
        rows.append({'sector': i+1, 'complete': True, 'deploy_mj': float(dep),
            'harvest_mj': float(har), 'energy_start_mj': float(es), 'energy_end_mj': float(ee),
            'net_energy_mj': float(ee-es), 'balance_residual_mj': float(ee-es+dep-har),
            'time_s': float(np.diff(np.interp([start,end],s,t))[0])})
    return rows


def score(result, dataset, role='calibration'):
    """Partial trajectories get no formal accuracy or fabricated sectors."""
    selected = [l for l in dataset['laps'] if l['role'] == role]
    summary = result['summary']
    full = bool(summary.get('completed_lap') and summary.get('execution_feasible'))
    trace = result['internal_trace']
    s, v, t = (np.asarray(trace[k], float) for k in ('s_m','speed_mps','time_s'))
    metrics = []
    for lap in selected:
        row = {'lap_number': int(lap['lap_number']), 'role': role, 'valid': full,
            'control_legal': bool(summary.get('feasible', False))}
        if not full:
            row['reason'] = 'Partial/physically failed lap not scored'
            metrics.append(row); continue
        ss = np.asarray(lap['s_m']); errors = (np.interp(ss,s,v)-lap['speed_mps'])*3.6
        bounds = np.asarray(lap.get('sector_boundaries_m',[]),float)
        if bounds.size!=2:
            bounds=np.interp(np.cumsum(lap['sector_times_s'])[:2],lap['t_s'],ss)
        pred_sectors = np.diff(np.r_[0.,np.interp(bounds,s,t),t[-1]])
        obs_sectors = np.asarray(lap['sector_times_s'])
        masks = [(ss>=a)&(ss<b) for a,b in zip(np.r_[0.,bounds],np.r_[bounds,result['length_m']+1.])]
        row.update(speed_rmse_kmh=float(np.sqrt(np.mean(errors**2))),
            signed_bias_kmh=float(np.mean(errors)), speed_p95_absolute_error_kmh=float(np.quantile(abs(errors),.95)),
            lap_time_s=float(t[-1]), observed_lap_time_s=float(lap['lap_time_s']),
            lap_error_pct=float(100*(t[-1]-lap['lap_time_s'])/lap['lap_time_s']),
            sector_times_s=pred_sectors.tolist(), sector_error_s=(pred_sectors-obs_sectors).tolist(),
            sector_speed_rmse_kmh=[float(np.sqrt(np.mean(errors[m]**2))) if m.any() else None for m in masks],
            sector_signed_bias_kmh=[float(np.mean(errors[m])) if m.any() else None for m in masks],
            sector_energy=sector_energy(result,bounds))
        metrics.append(row)
    valid = bool(metrics and all(m['valid'] for m in metrics))
    aggregate = {} if not valid else {
        'mean_speed_rmse_kmh': float(np.mean([m['speed_rmse_kmh'] for m in metrics])),
        'sector_speed_rmse_kmh': np.mean([m['sector_speed_rmse_kmh'] for m in metrics],axis=0).tolist(),
        'max_abs_lap_error_pct': max(abs(m['lap_error_pct']) for m in metrics),
        'max_abs_sector_error_s': max(abs(e) for m in metrics for e in m['sector_error_s'])}
    thresholds = {'mean_speed_rmse_kmh':15., 'max_abs_lap_error_pct':2., 'max_abs_sector_error_s':1.}
    data_valid=bool(dataset.get('audit',{}).get('data_gate_passed',False))
    accuracy_passed = valid and data_valid and all(aggregate[k]<=limit for k,limit in thresholds.items())
    energy = np.asarray(trace.get('energy_mj', []))
    accounting = {'initial_energy_mj': float(energy[0]) if energy.size else None,
        'terminal_energy_mj': float(energy[-1]) if energy.size else None,
        'energy_swing_mj':float(np.ptp(energy)) if energy.size else None,
        'dc_balance_residual_mj': summary.get('energy_balance_residual_mj'),
        'mechanical_residual_max_abs_mj':float(np.max(abs(np.asarray(trace['mechanical_work_residual_mj'])),initial=0.)),
        'soc_interpretation':'ideal CU-K DC-equivalent energy; true ES SOC/losses remain unverified'}
    return {'role':role,'metrics':metrics,'aggregate':aggregate,'thresholds':thresholds,
        'accuracy_passed':bool(accuracy_passed),'data_gate_passed':data_valid,'control_legal':bool(summary.get('feasible',False)),
        'energy_accounting':accounting, 'future_observation_in_policy_or_reward':False,
        'formal_release_passed':bool(accuracy_passed and summary.get('feasible',False))}


def legal_time(result):
    from f1ers_v7._core.learning_base import is_legal
    return float(result['summary']['lap_time_s']) if is_legal(result) else float('inf')
