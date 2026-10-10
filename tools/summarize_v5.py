"""Read-only calibration-screen failure summary; writes JSON only to stdout.

Usage: python tools/summarize_v5.py --root .
No simulation, fitting, development view, network request or result mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


def summarize(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    from f1ers_v5.data import load_dataset
    from f1ers_v5.adapters import VehicleParamsV5
    from f1ers_v4.simulator import ContinuousGeometry

    hashes = {}; input_errors = []
    def read(path, compressed=False):
        payload = path.read_bytes()
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(payload).hexdigest()
        return json.loads(gzip.decompress(payload) if compressed else payload)

    geometry_path = root/'results/austria_2026_v5/geometry_spec.json'
    selected = read(geometry_path) if geometry_path.exists() else {"kind":"baseline"}
    datasets = {}
    def dataset(spec):
        key = json.dumps(spec,sort_keys=True)
        if key not in datasets:
            value = load_dataset(root,geometry_spec=spec,role='calibration')
            if [x['lap_number'] for x in value['laps']] != [3,4,5,6] or any(x['role']!='calibration' for x in value['laps']):
                raise ValueError('Only calibration laps 3–6 may be analyzed')
            datasets[key] = value
        return datasets[key]

    def stats(rows,key):
        a = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return {"min":float(np.min(a)),"median":float(np.median(a)),"max":float(np.max(a))} if a else None

    def stage_budget(s0,s1,v0,v1,geometry,params,wheel=0.):
        # This reproduces the stated THREE STAGE geometry budget directly,
        # without calling the kernel or relying on its reported tyre excess.
        ss = np.array([s0,.5*(s0+s1),s1],float)
        vv = np.array([v0,.5*(v0+v1),v1],float)
        g = geometry.evaluate(ss)
        total = params.mu*params.session_grip_scale*(params.mass_kg*9.81 +
                 .5*params.air_density_kgpm3*np.asarray(g['downforce_area_m2'])*vv**2)
        lateral = params.mass_kg*vv**2*np.abs(np.asarray(g['curvature_1pm']))
        remaining = np.sqrt(np.maximum(0.,total**2-lateral**2))
        return {"pure_lateral_excess_n":float(np.max(lateral-total)),
                "joint_excess_n":float(np.max(np.hypot(wheel,lateral)-total)),
                "maximum_lateral_utilization":float(np.max(lateral/total)),
                "minimum_remaining_longitudinal_capacity_n":float(np.min(remaining))}

    candidates = []; saved_rows = {}; candidate_root = root/'results/austria_2026_v5/candidates'
    for directory in sorted(p for p in candidate_root.iterdir() if p.is_dir()):
        report_path = directory/'calibration.json'; report = read(report_path) if report_path.exists() else None
        if report is not None and report.get('calibration_summary',{}).get('role')!='calibration':
            raise ValueError('Screening report must retain calibration-only provenance')
        spec = ((report or {}).get('candidate') or {}).get('geometry')
        if spec is None:spec = {"kind":"baseline"} if directory.name.startswith('baseline_') else selected
        cal = dataset(spec); lap_by_number = {x['lap_number']:x for x in cal['laps']}
        rows = []; autos = []; observed_indices = set(); skipped_non_screening=0
        failures = sorted((directory/'failures').glob('evaluation_*.json.gz'))
        for path in failures:
            try:
                item = read(path,compressed=True)
            except (OSError,EOFError,json.JSONDecodeError) as exc:
                input_errors.append({"path":path.relative_to(root).as_posix(),"reason":str(exc),"potential_in_progress_write":True})
                continue
            if item.get('stage')!='screening':
                skipped_non_screening += 1; continue
            if item.get('candidate') != directory.name:
                raise ValueError('Failure candidate identity does not match its directory')
            comparison = item['comparison']
            if comparison.get('role')!='calibration' or set(map(int,comparison['replay']['results']))!={3,4,5,6}:
                raise ValueError('Held-out or incomplete role membership in failure archive')
            for mode in ('autonomous','replay'):
                if any(m.get('role')!='calibration' or int(m.get('lap_number',-1)) not in (3,4,5,6) for m in comparison[mode].get('metrics',[])):
                    raise ValueError('Held-out metrics in failure archive')
            observed_indices.add(int(path.name.split('_')[1].split('.')[0]))
            params = VehicleParamsV5.from_dict(item['parameters'])
            geometry = ContinuousGeometry(cal['track'],cal['rules'],params)
            auto = comparison['autonomous']['result']['summary']
            autos.append(bool(auto.get('completed_lap') and auto.get('feasible') and auto.get('execution_feasible')))
            for number,result in comparison['replay']['results'].items():
                n=int(number); proposal=result.get('failure_proposal') or {}; trace=result.get('internal_trace',{})
                row={"lap_number":n,"evaluation":path.name,"status":result.get('terminal_status',result['summary'].get('status')),
                     "parameter_values":item['values'],
                     "completed_lap":bool(result['summary'].get('completed_lap')),
                     "distance_m":result.get('terminal_s_m',result.get('distance_completed_m')),
                     "time_s":result.get('terminal_time_s'),"reasons":proposal.get('guard_reasons',[]),
                     "rejected_wheel_force_n":proposal.get('wheel_force_n'),
                     "rejected_reported_remaining_capacity_n":proposal.get('tire_brake_capacity_n'),
                     "terminal_brake_fraction":proposal.get('driver_brake_fraction')}
                forces=np.asarray(trace.get('wheel_force_n',[]),float)
                raw_request=np.asarray(trace.get('raw_requested_force_n',[]),float)
                ix=np.flatnonzero(np.isfinite(raw_request)&(raw_request<0)) if raw_request.shape==forces.shape else np.flatnonzero(forces<0)
                if len(ix):
                    i=int(ix[0]); ts=np.asarray(trace['time_s'],float); ss=np.asarray(trace['s_m'],float); vv=np.asarray(trace['speed_mps'],float)
                    driver=lap_by_number[n]['driver_trace']; dt=np.asarray(driver['time_s']); brake=np.asarray(driver['brake'])
                    di=max(0,int(np.searchsorted(dt,ts[i],side='right')-1))
                    row.update(first_brake_time_s=float(ts[i]),first_brake_switch=float(brake[di]),first_executed_brake_force_n=float(forces[i]))
                    if raw_request.shape==forces.shape and np.isfinite(raw_request[i]):
                        row.update(first_requested_brake_force_n=float(raw_request[i]),request_source='saved_raw_requested_force_n')
                    else:
                        budget=stage_budget(ss[i],ss[i+1],vv[i],vv[i+1],geometry,params)
                        row.update(first_requested_brake_force_n=-budget['minimum_remaining_longitudinal_capacity_n']*params.replay_brake_gain*float(brake[di]),
                                   request_source='recomputed remaining-capacity mapping, not measured pressure')
                if row['status']=='physical_stop':
                    driver=lap_by_number[n]['driver_trace'];dt=np.asarray(driver['time_s']);brake=np.asarray(driver['brake'])
                    releases=dt[(dt>row['time_s']+1e-8)&(brake<.5)]
                    current=max(0,int(np.searchsorted(dt,row['time_s'],side='right')-1))
                    row['next_release_minus_stop_s']=float(releases[0]-row['time_s']) if len(releases) and brake[current]>=.5 and row['terminal_brake_fraction'] is not None and row['terminal_brake_fraction']>0 else None
                if row['status']=='physical_fail' and 'combined_tire_force' in row['reasons'] and all(k in proposal for k in ('s_start_m','s_end_m','speed_start_mps','speed_next_mps','wheel_force_n')):
                    budget=stage_budget(proposal['s_start_m'],proposal['s_end_m'],proposal['speed_start_mps'],proposal['speed_next_mps'],geometry,params,proposal['wheel_force_n'])
                    row.update(budget)
                    row['reported_minus_recomputed_joint_excess_n']=float(proposal['combined_tire_force_excess_n']-budget['joint_excess_n'])
                rows.append(row)
        screening=(report or {}).get('screening',[])
        # Complete-physical screening rows imply both paths were complete even
        # when no failure archive was written. No forces are invented for them.
        feasible={int(p['index']) for p in screening if p['physical_valid']}
        inferred_feasible_missing=feasible-observed_indices
        invalid_missing={int(p['index']) for p in screening if not p['physical_valid']}-observed_indices
        points=len(screening) if report is not None else None
        preregistered=(report or {}).get('budget',{}).get('screening_points',64)
        combined=[r for r in rows if 'combined_tire_force' in r['reasons']]
        lateral=[r for r in combined if r.get('pure_lateral_excess_n',-np.inf)>8.]
        stops=[r for r in rows if r['status']=='physical_stop']
        state_counts=Counter(r['status'] for r in rows);state_counts['accepted']+=4*len(inferred_feasible_missing)
        if not state_counts['accepted']:del state_counts['accepted']
        by_lap={}
        for n in (3,4,5,6):
            subset=[r for r in rows if r['lap_number']==n]
            per_states=Counter(r['status'] for r in subset);per_states['accepted']+=len(inferred_feasible_missing)
            if not per_states['accepted']:del per_states['accepted']
            by_lap[n]={"states":dict(per_states),"saved_trace_distance_m":stats(subset,'distance_m'),
                       "first_brake_time_s":stats(subset,'first_brake_time_s'),
                       "first_requested_brake_force_n":stats(subset,'first_requested_brake_force_n'),
                       "first_executed_brake_force_n":stats(subset,'first_executed_brake_force_n'),
                       "firstcorner_tire_failures":sum('combined_tire_force'in r['reasons'] and 180<=r['distance_m']<=320 for r in subset),
                       "stops_before_release":sum(r.get('next_release_minus_stop_s',0)>0 for r in subset)}
        candidates.append({"candidate":directory.name,"screening_report_available":report is not None,
            "preregistered_screening_points":preregistered,"screening_points_completed":points,
            "screening_feasible_count":len(feasible) if report is not None else None,
            "observed_screen_failure_evaluations":len(observed_indices),"skipped_non_screening_failure_files":skipped_non_screening,
            "status":"complete" if points==preregistered and not invalid_missing else 'in_progress_or_missing_records',
            "missing_invalid_screening_archive_indices":sorted(invalid_missing),
            "feasible_points_without_failure_archives":len(inferred_feasible_missing),
            "force_analysis_scope":"Saved screening-failure trajectories only; no force evidence inferred for feasible rows",
            "replay_trajectories_analyzed":len(rows),"replay_terminal_state_counts":dict(state_counts),
            "replay_completed_count":sum(r['completed_lap'] for r in rows)+4*len(inferred_feasible_missing),
            "replay_reason_counts":dict(Counter(q for r in rows for q in r['reasons'])),
            "autonomous_complete_physical_count":sum(autos)+len(inferred_feasible_missing),
            "autonomous_count_source":"Saved failure-file autonomous trace plus complete-physical screening rows without failure files",
            "firstcorner_180_320m_combined_failures":sum('combined_tire_force'in r['reasons'] and 180<=r['distance_m']<=320 for r in rows),
            "firstcorner_180_320m_all_terminal_count":sum(r.get('distance_m')is not None and 180<=r['distance_m']<=320 for r in rows),
            "physical_stops_before_release":sum(r.get('next_release_minus_stop_s',0)>0 for r in stops),
            "stop_distance_m":stats(stops,'distance_m'),"next_release_minus_stop_s":stats(stops,'next_release_minus_stop_s'),
            "combined_failures_with_independent_force_reconstruction":sum('pure_lateral_excess_n'in r for r in combined),
            "pure_lateral_excess_above_8n_count":len(lateral),
            "joint_excess_above_8n_without_pure_lateral_excess_above_8n_count":sum(r.get('joint_excess_n',0)>8 and r.get('pure_lateral_excess_n',np.inf)<=8 for r in combined),
            "zero_executed_wheel_force_at_combined_failure_count":sum(r.get('rejected_wheel_force_n')is not None and abs(r['rejected_wheel_force_n'])<=1e-9 for r in combined),
            "zero_recomputed_remaining_capacity_count":sum(r.get('minimum_remaining_longitudinal_capacity_n',np.inf)<=1e-9 for r in combined),
            "combined_failures_with_brake_released":sum(r['terminal_brake_fraction']==0 for r in combined),
            "pure_lateral_excess_n":stats(combined,'pure_lateral_excess_n'),
            "maximum_lateral_utilization":stats(combined,'maximum_lateral_utilization'),
            "reported_minus_recomputed_joint_excess_n":stats(combined,'reported_minus_recomputed_joint_excess_n'),
            "first_brake_switch_counts":dict(Counter(r.get('first_brake_switch') for r in rows)),
            "first_requested_brake_force_n":stats(rows,'first_requested_brake_force_n'),
            "first_executed_brake_force_n":stats(rows,'first_executed_brake_force_n'),
            "by_lap":by_lap})
        saved_rows[directory.name] = rows

    paired_comparison = None
    if all(name in saved_rows for name in ('baseline_remaining','baseline_independent')):
        left={(r['evaluation'],r['lap_number']):r for r in saved_rows['baseline_remaining']}
        right={(r['evaluation'],r['lap_number']):r for r in saved_rows['baseline_independent']}
        pairs=[]
        for key in sorted(left.keys() & right.keys()):
            a,b=left[key],right[key]
            if a['parameter_values']!=b['parameter_values']:
                raise ValueError('Paired baseline screening values differ; cannot present a controlled contrast')
            pairs.append({"state_transition":a['status']+' -> '+b['status'],
                          "distance_change_m":b['distance_m']-a['distance_m'],
                          "time_change_s":b['time_s']-a['time_s']})
        paired_comparison={"left":"baseline_remaining","right":"baseline_independent",
            "paired_saved_failure_trajectories":len(pairs),"identical_screening_parameter_vectors_verified":True,
            "state_transitions":dict(Counter(p['state_transition']for p in pairs)),
            "right_minus_left_distance_m":stats(pairs,'distance_change_m'),
            "right_minus_left_time_s":stats(pairs,'time_change_s'),
            "farther_by_more_than_0_1m":sum(p['distance_change_m']>.1 for p in pairs),
            "earlier_by_more_than_0_1m":sum(p['distance_change_m']<-.1 for p in pairs),
            "distance_comparison_threshold_m":.1,
            "interpretation":"Partial-trajectory termination contrast only, never formal accuracy or a candidate-selection criterion"}

    for relative in ('f1ers_v5/data.py','f1ers_v5/geometry.py','f1ers_v5/adapters.py','f1ers_v3/data.py','f1ers_v3/geometry.py','f1ers_v4/simulator.py','f1ers_v4/vehicle.py','f1ers/model.py',
        'data/raw/openf1/laps.json','data/raw/openf1/car_data_63_laps3to9.json','data/raw/openf1/location_63_laps3to9.json',
        'data/raw/tum/Spielberg_minimum_curvature.csv','data/raw/tum/Spielberg_centerline.csv','data/processed/dataset.json','data/processed/track.npz','data/v3/local/registration.json','config/rules_austria_2026.json'):
        path=root/relative
        if path.exists():hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema_version":1,"diagnostic_only":True,"candidate_selection_influenced":False,
        "tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration_laps":[3,4,5,6],"development_views_opened":False,"new_telemetry_requested":False,
        "method":{"stage_s":"[s_start, (s_start+s_end)/2, s_end]",
                  "stage_speed":"[v_start, (v_start+v_end)/2, v_end]",
                  "total_tire_force_n":"mu * session_grip_scale * (mass*9.81 + 0.5*rho*downforce_area_at_stage*v_stage**2)",
                  "lateral_force_n":"mass*v_stage**2*abs(continuous_route_curvature_at_stage)",
                  "remaining_longitudinal_capacity_n":"sqrt(max(0,total_tire_force**2-lateral_force**2)); minimum over all three stages",
                  "pure_lateral_excess_n":"max_over_stages(lateral-total)",
                  "joint_excess_n":"max_over_stages(hypot(rejected_executed_wheel_force,lateral)-total)",
                  "kernel_force_function_called":False,"kernel_guard_threshold_n":8.,"zero_force_tolerance_n":1e-9,
                  "firstcorner_window_m":[180.,320.],"request_interpretation":"Model demand, never real brake pressure",
                  "source_caveat":"Inherited raw caches may contain multiple historical roles; only the loader's explicit calibration view is processed"},
        "candidates":candidates,"comparison":[{k:c[k] for k in ('candidate','status','screening_points_completed','screening_feasible_count','replay_completed_count','autonomous_complete_physical_count','pure_lateral_excess_above_8n_count','physical_stops_before_release')}for c in candidates],
        "paired_baseline_comparison":paired_comparison,
        "input_errors":input_errors,"input_sha256":dict(sorted(hashes.items()))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root),ensure_ascii=False,indent=2,allow_nan=False))
