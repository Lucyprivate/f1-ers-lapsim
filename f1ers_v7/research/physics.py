"""Finite optional physics experiments; no telemetry acquisition or training.

All model estimation takes an explicit calibration-only view. Development
scores occur only after fit parameters are frozen. Axle parameters and battery
losses are declared engineering assumptions, not identified FIA car data.
The mathematical models are independently implemented, without imported
GPL/LGPL source. Legacy executor defaults and rule classes remain available.
"""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .._core.integrator import (VehicleParamsV4, storage_reserve_mj,
    storage_deployment_bound_kw, IntegratorConfig)
from .._core.base_rules import RuleConfig, _get
from .._core.base_simulator import SimulatorV4
from .._core.learning_base import _write
from .._core.metrics import score
from .._core.proxy_audit import audit_proxy_consistency
from .._core.provenance import read
from ..deploy_only import DeployOnlyShield
from .environment import ResearchEnv, ResearchSimulator
from .data import require_role

PARAMETERS = ('mu','drag_area_m2','downforce_area_m2','equivalent_power_kw','replay_brake_gain')
BOUNDS = np.asarray([[1.2,2.4],[.6,2.],[1.8,6.],[350.,732.5],[.1,1.]])
LOSS_PROTOCOL = 'v7_opensource_storage_loss_sensitivity'
CANDIDATES = (
    {'name':'baseline','independent_replay_braking':False,'axle_budget_enabled':False},
    {'name':'independent_braking','independent_replay_braking':True,'axle_budget_enabled':False},
    {'name':'independent_braking_axles','independent_replay_braking':True,'axle_budget_enabled':True})


def require_calibration(data):
    require_role(data,'calibration')
    if tuple(int(l['lap_number']) for l in data['laps']) != (3,4,5,6):
        raise ValueError('Only the fixed calibration laps 3–6 may enter estimation')
    if not data['audit'].get('data_gate_passed',False):
        raise ValueError('Calibration alignment/boundary quality gate must pass before estimation')
    return data


def axle_sensitivity_registry():
    """Nominal plus the eight preregistered engineering corners, never fitted."""
    rows = [{'name':'nominal','rear_static_fraction':.5,'rear_aero_fraction':.5,
             'height_over_wheelbase':.1}]
    for i,(static,aero,height) in enumerate(product((.45,.55),(.45,.55),(.05,.15))):
        rows.append({'name':f'corner_{i+1}','rear_static_fraction':static,
                     'rear_aero_fraction':aero,'height_over_wheelbase':height})
    return rows


def loss_sensitivity_registry():
    return [{'name':f'eta_{eta:.2f}_aux_{aux:.0f}',
        'storage_charge_efficiency':eta,'storage_discharge_efficiency':eta,
        'auxiliary_power_w':aux} for eta,aux in product((1.,.98,.95),(0.,50.,100.))]


def segment_boundary_audit(route_xy,centerline,direction=1):
    """Independent nearest-segment projection and signed inherited widths.

    TUM columns 2/3 are right/left in centerline point order. Reversing travel
    swaps the reported sides after the geometric projection; it cannot alter
    the physical clearance. No line is fitted or shifted by this diagnostic.
    """
    xy,centre = np.asarray(route_xy,float),np.asarray(centerline,float)
    if (xy.ndim!=2 or xy.shape[1]!=2 or centre.ndim!=2 or centre.shape[1]<4 or len(centre)<3
        or not np.isfinite(xy).all() or not np.isfinite(centre).all() or direction not in (-1,1)):
        raise ValueError('Finite route coordinates, closed centerline/widths and direction required')
    delta = np.roll(centre[:,:2],-1,axis=0)-centre[:,:2]
    norm2 = np.sum(delta*delta,axis=1)
    if np.any(norm2<=1e-12) or np.any(centre[:,2:4]<0):
        raise ValueError('Centerline has duplicate neighbours or negative widths')
    ids,fractions,offsets,distances = [],[],[],[]
    # Chunked projection avoids a full dense route×centerline tensor.
    for chunk in np.array_split(xy,max(1,int(np.ceil(len(xy)/256)))):
        displacement = chunk[:,None,:]-centre[None,:,:2]
        fraction = np.clip(np.einsum('nkj,kj->nk',displacement,delta)/norm2,0.,1.)
        residual = displacement-fraction[:,:,None]*delta[None,:,:]
        distance2 = np.sum(residual*residual,axis=2)
        nearest = np.argmin(distance2,axis=1)
        at = np.arange(len(chunk));t = fraction[at,nearest]
        tangent = delta[nearest]/np.sqrt(norm2[nearest])[:,None]
        normal = np.column_stack([-tangent[:,1],tangent[:,0]])
        offset = np.sum(residual[at,nearest]*normal,axis=1)
        ids.extend(nearest.tolist());fractions.extend(t.tolist());offsets.extend(offset.tolist())
        distances.extend(np.sqrt(distance2[at,nearest]).tolist())
    ids,t,offset = np.asarray(ids,int),np.asarray(fractions),np.asarray(offsets)
    next_ids = (ids+1)%len(centre)
    width = (1.-t[:,None])*centre[ids,2:4]+t[:,None]*centre[next_ids,2:4]
    right,left = width[:,0]+offset,width[:,1]-offset
    if direction<0: left,right = right,left
    return {'method':'nearest closed centerline segment, linearly interpolated inherited widths',
        'direction':direction,'minimum_left_clearance_m':float(left.min()) if len(left) else None,
        'minimum_right_clearance_m':float(right.min()) if len(right) else None,
        'minimum_boundary_clearance_m':float(min(left.min(),right.min())) if len(left) else None,
        'outside_samples':int(np.sum((left<0)|(right<0))),
        'mean_projection_distance_m':float(np.mean(distances)) if distances else None,
        'signed_offset_m':offset.tolist(),'left_clearance_m':left.tolist(),'right_clearance_m':right.tolist(),
        'geometry_changed':False,'actual_surveyed_boundary_verified':False}


class LossAwareShield(DeployOnlyShield):
    """The same public-subset FSM with a finite loss-aware energy reserve.

    Effective requests may be reduced by this engineering guard. An empty
    intersection with a committed reference is recorded and rejected; actual
    supplied DC is never rewritten. The kernel remains authoritative at the
    true storage floor, including a cold start with auxiliary consumption.
    """
    def __init__(self,*args,storage_charge_efficiency=1.,storage_discharge_efficiency=1.,
                 auxiliary_power_w=0.,**kwargs):
        self.storage_charge_efficiency = float(storage_charge_efficiency)
        self.storage_discharge_efficiency = float(storage_discharge_efficiency)
        self.auxiliary_power_w = float(auxiliary_power_w)
        storage_reserve_mj(0.,0.,100.,self.storage_discharge_efficiency,self.auxiliary_power_w)
        if not np.isfinite(self.storage_charge_efficiency) or not 0 < self.storage_charge_efficiency <= 1:
            raise ValueError('Battery charging efficiency must lie in (0,1]')
        super().__init__(*args,**kwargs)

    def _reserve(self,power_kw,hold_s):
        return storage_reserve_mj(power_kw,max(0.,hold_s),self.rate_dc,
            self.storage_discharge_efficiency,self.auxiliary_power_w)

    def _energy_bound(self,state,duration_s):
        credit = max(0.,float(_get(state,'energy_mj',2.))-self.config.energy_floor_mj)
        return storage_deployment_bound_kw(credit,max(0.,float(duration_s)),self.rate_dc,
            self.storage_discharge_efficiency,self.auxiliary_power_w)

    def query(self,state,command,geometry,dt_s):
        result = super().query(state,command,geometry,dt_s)
        t = float(_get(state,'elapsed_time_s',0.))
        braking = self._braking(command)
        bound = self._energy_bound(state,float(dt_s)+self._hold_reserve_s(t))
        if not braking:
            result['requested_dc_kw'] = min(max(0.,result['requested_dc_kw']),bound)
        declared = result['declared_proxy_dc_kw']
        effective = max(0.,result['requested_dc_kw'])
        gap = max(0.,min(declared,result['deployment_cap_kw'])-effective) if not braking else 0.
        result.update(energy_bound_dc_kw=bound,effective_positive_request_dc_kw=effective,
            request_proxy_shortfall_dc_kw=gap,
            constraint_intersection_feasible=bool(gap<=self.config.tolerance),
            storage_loss_model=True,auxiliary_power_w=self.auxiliary_power_w,
            no_feasible_positive_DC_root=bool(not braking and gap>self.config.tolerance))
        return result

    def certificate(self):
        result = super().certificate()
        result['assumptions'] = dict(result['assumptions'],
            energy='Declared battery efficiency and auxiliary-load engineering sensitivity; true SOC unverified',
            auxiliary='Explicit causal storage-side auxiliary consumption, not a post-hoc time correction')
        result['storage_model'] = {'charge_efficiency':self.storage_charge_efficiency,
            'discharge_efficiency':self.storage_discharge_efficiency,
            'auxiliary_power_w':self.auxiliary_power_w,'lap_recharge_counter':'DC-side',
            'finite_hold_ramp_notice_reserve':True,'infinite_horizon_viability_proven':False}
        return result


class LossAwareEnv(ResearchEnv):
    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs)
        self.shield = self.sim.make_shield()
        self.shield.reset(self.state,self.carried_rule_state)
        self.driver.shield = self.shield
        return self.get_obs()

    def result(self):
        result = super().result()
        fields = ('requested_wheel_force_n','front_longitudinal_capacity_n','rear_longitudinal_capacity_n',
            'front_normal_load_n','rear_normal_load_n','front_lateral_force_n','rear_lateral_force_n',
            'front_wheel_force_n','rear_wheel_force_n','front_friction_brake_force_n',
            'rear_friction_brake_force_n','rear_regeneration_force_n','storage_loss_mj',
            'auxiliary_work_mj','storage_ledger_residual_mj','dc_deployment_mj','dc_recharge_mj')
        for key in fields:
            result['internal_trace'][key] = np.asarray([r[key] for r in self.records],float)
        summary = result['summary']
        loss = sum(r['storage_loss_mj'] for r in self.records)
        auxiliary = sum(r['auxiliary_work_mj'] for r in self.records)
        dc_residual = summary['energy_balance_residual_mj']
        storage_residual = dc_residual+loss+auxiliary
        summary.update(protocol=LOSS_PROTOCOL,
            research_energy_model='Explicit storage efficiency and auxiliary-load sensitivity',
            frozen_vehicle_kernel=False,optional_physics_engineering_model=True,
            storage_loss_mj=loss,auxiliary_work_mj=auxiliary,
            dc_only_balance_residual_mj=dc_residual,
            energy_balance_residual_mj=storage_residual,
            storage_balance_residual_mj=storage_residual,
            storage_charge_efficiency=self.sim.vehicle.storage_charge_efficiency,
            storage_discharge_efficiency=self.sim.vehicle.storage_discharge_efficiency,
            auxiliary_power_w=self.sim.vehicle.auxiliary_power_w,
            axle_budget_enabled=self.sim.vehicle.axle_budget_enabled,
            rear_regeneration_shares_friction_budget=True,
            actual_SOC_verified=False,training_uses_this_loss_model=False)
        summary['feasible'] = bool(summary['completed_lap'] and summary['execution_feasible']
            and summary['sustainable'] and summary['periodic_speed_closed'] and summary['dc_boundary_closed']
            and summary['rule_state_closed'] and summary['public_rules_passed']
            and summary['reachable_request_gate_passed'] and abs(storage_residual)<=1e-7)
        return result


class LossAwareSimulator(ResearchSimulator):
    env_class = LossAwareEnv
    protocol = LOSS_PROTOCOL
    def create_env(self,action_space='mode_aware',reward_config=None,initial_speed_mps=None,
                   carried_rule_state=None,output_ds_m=5.,numerics=None,initial_energy_mj=2.):
        if action_space != 'mode_aware':
            raise ValueError('Loss sensitivity retains deployment-only mode-aware actions')
        if isinstance(numerics,dict): numerics = IntegratorConfig(**numerics)
        # Sensitivity has no learned actor and carries the true terminal FSM.
        # The restricted learned-policy boundary pool must not reset that FSM.
        return self.env_class(self,action_space,reward_config,initial_speed_mps,
            deepcopy(carried_rule_state),output_ds_m,numerics,initial_energy_mj)

    def make_shield(self):
        return LossAwareShield(self.rules,self.geometry,self.rule_config,
            self.geometry_guard_map,self.margin_s,energy_notice_s=self.energy_notice_s,
            storage_charge_efficiency=self.vehicle.storage_charge_efficiency,
            storage_discharge_efficiency=self.vehicle.storage_discharge_efficiency,
            auxiliary_power_w=self.vehicle.auxiliary_power_w)


def loss_proxy_audit(sim,result):
    """Reconstruct this same loss-aware shield; do not claim ideal-model audit."""
    if result['summary'].get('protocol') != LOSS_PROTOCOL:
        raise ValueError('Explicit loss-sensitivity execution protocol required')
    shield = sim.make_shield()
    report = audit_proxy_consistency(result,sim,shield)
    exact = (shield.deployment_only_audit == result.get('deployment_only_mapping_audit',[]))
    report.update(same_loss_aware_shield_replayed=True,
        engineering_violations=list(shield.engineering_violations),
        deployment_only_mapping_reconstruction_exact=exact,
        passed=bool(exact and not shield.engineering_violations and
                    report['additional_surrogate_hold_and_limited_request_gate_passed']))
    return report


def sobol_viability(evaluate,*,seed=2026,points=64):
    """Keep all 64 records, including failures; only complete physics ranks."""
    if seed != 2026 or points != 64:
        raise ValueError('This protocol fixes the 64-point Sobol seed to 2026')
    values = qmc.scale(qmc.Sobol(len(PARAMETERS),scramble=True,seed=seed).random_base2(6),
                       BOUNDS[:,0],BOUNDS[:,1])
    rows = []
    for index,x in enumerate(values):
        record = evaluate(x.copy(),f'screen_{index+1:02d}')
        finite = _finite_objective(record)
        viable = bool(record.get('complete') and record.get('physical') and finite)
        rows.append({**record,'index':index+1,'parameters':x.tolist(),'viable':viable})
    starts = sorted((r for r in rows if r['viable']),key=lambda r:(r['objective'],r['index']))[:3]
    return {'seed':seed,'points':points,'records':rows,'viable_points':sum(r['viable'] for r in rows),
            'starts':starts,'fit_allowed':len(starts)==3}


def capped_fit(evaluate,screening,*,max_iterations=300,max_evaluations=600):
    """Three fixed starts. Each objective call is counted before execution."""
    if not 1 <= max_iterations <= 300 or not 1 <= max_evaluations <= 600:
        raise ValueError('A fit cannot expand the frozen per-start budget')
    starts = screening.get('starts',[])
    if len(starts) != 3 or not screening.get('fit_allowed'):
        return {'status':'withheld_fewer_than_three_viable_starts','starts':[],
                'selected_parameters':None,'converged':False,'evaluations':0}
    if len({tuple(r['parameters']) for r in starts}) != 3 or not all(r.get('viable') for r in starts):
        raise ValueError('Three distinct complete physical starts are required')
    reports = []
    for start_index,start in enumerate(starts):
        calls = []
        def objective(x):
            if len(calls) >= max_evaluations:
                raise RuntimeError('explicit_fit_evaluation_budget')
            x = np.asarray(x,float)
            if not np.isfinite(x).all() or np.any(x<BOUNDS[:,0]) or np.any(x>BOUNDS[:,1]):
                raise ValueError('Fit proposal outside the frozen parameter bounds')
            receipt = {'parameters':x.tolist(),'started':True,'viable':False}
            calls.append(receipt)
            record = evaluate(x.copy(),f'fit_{start_index+1}_{len(calls):03d}')
            viable = bool(record.get('complete') and record.get('physical') and
                          _finite_objective(record))
            receipt.update(record,parameters=x.tolist(),viable=viable)
            return float(record['objective']) if viable else 1e6
        try:
            result = minimize(objective,np.asarray(start['parameters']),method='Nelder-Mead',bounds=BOUNDS,
                options={'maxiter':max_iterations,'maxfev':max_evaluations,'xatol':1e-5,'fatol':1e-7})
            converged = bool(result.success)
            iterations,message = int(result.nit),str(result.message)
        except RuntimeError as exc:
            if str(exc) != 'explicit_fit_evaluation_budget': raise
            converged,iterations,message = False,None,str(exc)
        viable_calls = [r for r in calls if r['viable']]
        best = min(viable_calls,key=lambda r:r['objective']) if viable_calls else start
        reports.append({'start_parameters':start['parameters'],'converged':converged,
            'iterations':iterations,'evaluations':len(calls),'message':message,
            'best':best,'records':calls})
    eligible = [r for r in reports if r['converged'] and r['best'].get('viable')]
    best = min(eligible,key=lambda r:r['best']['objective']) if eligible else None
    finite = [r['best'] for r in reports if r['best'].get('viable')]
    reference = min((r['objective'] for r in finite),default=math.inf)
    near = [r for r in finite if r['objective'] <= reference*1.01+1e-12]
    return {'status':'converged' if best else 'nonconverged','starts':reports,
        'converged':best is not None,'selected_parameters':best['best']['parameters'] if best else None,
        'best_diagnostic_parameters':min(finite,key=lambda r:r['objective'])['parameters'] if finite else None,
        'near_optimal_parameter_family':near,'near_optimal_fraction':.01,
        'evaluations':sum(r['evaluations'] for r in reports),'bounds_expanded':False}


def _model(values,spec):
    options = {key:value for key,value in spec.items() if key!='name'}
    return VehicleParamsV4(mass_kg=800.,ice_power_kw=400.,**dict(zip(PARAMETERS,map(float,values))),**options)


def _finite_objective(record):
    value = record.get('objective')
    return isinstance(value,(int,float,np.number)) and bool(np.isfinite(value))


def _rule_config(config):
    return RuleConfig(tail_guard_m=float(config.get('tail_guard_m',1000.)))


def _sim(data,params,config,*,loss=False):
    cls = LossAwareSimulator if loss else ResearchSimulator
    return cls(data['track'],data['rules'],params,rule_config=_rule_config(config),
               margin_s=float(config.get('geometry_margin_s',1.)),
               energy_notice_s=float(config.get('energy_notice_s',.5)))


def _normalized_objective(replay_scores,auto_score):
    paths = []
    for rows in (replay_scores,[auto_score]):
        metrics = [m for report in rows for m in report['metrics']]
        if not metrics or not all(m['valid'] for m in metrics): return math.inf
        paths.append(float(np.mean([np.mean([m['speed_rmse_kmh']/15.,abs(m['lap_error_pct'])/2.,
                    np.mean(np.abs(m['sector_error_s']))]) for m in metrics])))
    return float(np.mean(paths))


def _path_gates(reports):
    metrics = [m for report in reports for m in report['metrics']]
    full = bool(metrics and all(m['valid'] for m in metrics))
    aggregate = {} if not full else {
        'mean_speed_rmse_kmh':float(np.mean([m['speed_rmse_kmh'] for m in metrics])),
        'max_abs_lap_error_pct':max(abs(m['lap_error_pct']) for m in metrics),
        'max_abs_sector_error_s':max(abs(e) for m in metrics for e in m['sector_error_s'])}
    passed = full and all(report['data_gate_passed'] for report in reports) and all(
        aggregate[key]<=limit for key,limit in (('mean_speed_rmse_kmh',15.),
            ('max_abs_lap_error_pct',2.),('max_abs_sector_error_s',1.)))
    return {'complete_physical_scoring':full,'aggregate':aggregate,'accuracy_passed':bool(passed)}


def evaluate_model(data,values,spec,output,config,*,role='calibration',initial_speed_mps=None):
    """Standalone two-path prediction. Replay retains only observed v(0)."""
    if role == 'calibration': require_calibration(data)
    elif role == 'development':
        require_role(data,'development')
        if tuple(int(l['lap_number']) for l in data['laps']) != (7,9):
            raise ValueError('Only fixed development laps 7/9 may be scored')
    else: raise ValueError('No final-test input is supported')
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    params = _model(values,spec)
    replay_sim = SimulatorV4(data['track'],data['rules'],params)
    replay_results,replay_scores = [],[]
    for lap in data['laps']:
        result = replay_sim.simulate(mode='replay',driver_trace=lap['driver_trace'],output_ds_m=5.)
        _write(output/f'replay_lap_{lap["lap_number"]}.json.gz',result)
        view = {**data,'laps':[lap]}
        report = score(result,view,role)
        _write(output/f'replay_lap_{lap["lap_number"]}_score.json',report)
        replay_results.append(result);replay_scores.append(report)
    sim = _sim(data,params,config)
    action = float(config.get('physics_baseline_action',0.))
    if not -1 <= action <= 1: raise ValueError('Fixed baseline raw action outside [-1,1]')
    if initial_speed_mps is None:
        initial_speed_mps = float(np.median([l['initial_speed_mps'] for l in data['laps']]))
    auto = sim.simulate(lambda obs:action,initial_speed_mps=initial_speed_mps,output_ds_m=5.)
    _write(output/'autonomous.json.gz',auto)
    auto_score = score(auto,data,role)
    _write(output/'autonomous_score.json',auto_score)
    all_results = replay_results+[auto]
    complete = all(r['summary']['completed_lap'] for r in all_results)
    physical = all(r['summary']['execution_feasible'] for r in all_results)
    objective = _normalized_objective(replay_scores,auto_score)
    report = {'role':role,'complete':bool(complete),'physical':bool(physical),
        'objective':objective if np.isfinite(objective) else None,
        'trajectory_directory':str(output),'parameters':dict(zip(PARAMETERS,map(float,values))),
        'candidate':spec,'replay_scores':replay_scores,'autonomous_score':auto_score,
        'replay_accuracy_gates':_path_gates(replay_scores),
        'autonomous_accuracy_gates':_path_gates([auto_score]),
        'autonomous_control_qualified':bool(auto['summary']['feasible']),
        'autonomous_control_gates':{key:auto['summary'].get(key) for key in
            ('completed_lap','execution_feasible','sustainable','periodic_speed_closed',
             'dc_boundary_closed','rule_state_closed','public_rules_passed','reachable_request_gate_passed')},
        'physical_flag_meaning':'complete signed-force integration only; control and accuracy gates are separate',
        'initial_speed_mps':initial_speed_mps,'autonomous_baseline_raw_action':action,
        'evaluation_wall_s':perf_counter()-started,
        'unobserved_ERS_fit':False,'partial_laps_formally_scored':False,'FP2_accessed':False,
        'failures':[{'mode':r['mode'],'terminal_s_m':r['terminal_s_m'],
            'status':r['terminal_status'],'failure':r['failure_proposal']}
            for r in all_results if not r['summary']['completed_lap'] or not r['summary']['execution_feasible']]}
    _write(output/'evaluation.json',report)
    # The finite optimizer API uses infinity only in memory, never in JSON.
    return {**report,'objective':objective}


def _physics_evaluation_worker(data,values,spec,output,config,initial_speed):
    # This top-level target is spawn-pickleable on Windows. Each independent
    # trial owns one exclusive directory and receives only calibration data.
    return evaluate_model(data,values,spec,output,config,initial_speed_mps=initial_speed)


def replay_numerical_audit(sim,driver_trace,output):
    """Refine replay only; never substitute the v4 autonomous driver."""
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    trace = {k:np.asarray(driver_trace[k],float).copy() for k in ('time_s','speed_mps','throttle','brake')}
    for value in trace.values(): value.setflags(write=False)
    binding = sha256(b''.join(trace[k].tobytes() for k in sorted(trace))).hexdigest()
    initial_speed = float(np.interp(0.,trace['time_s'],trace['speed_mps']))
    common = np.linspace(0.,sim.length_m,int(np.ceil(sim.length_m/.5))+1)
    comparisons,groups = {},{}
    for group,settings in (
        ('output',((10.,.02,1.),(5.,.02,1.),(2.5,.02,1.))),
        ('integration',((5.,.02,1.),(5.,.01,.5),(5.,.005,.25)))):
        runs=[]
        for index,(ds,dt,max_ds) in enumerate(settings):
            result = sim.simulate(mode='replay',driver_trace=trace,initial_speed_mps=initial_speed,
                output_ds_m=ds,numerics=replace(sim.config,dt_max=dt,max_ds=max_ds))
            _write(output/f'{group}_{index}.json.gz',result);runs.append(result)
        rows=[]
        for a,b in zip(runs[:-1],runs[1:]):
            valid = all(r['summary']['completed_lap'] and r['summary']['execution_feasible'] for r in (a,b))
            def velocity(result):
                if group=='integration':
                    return np.interp(common,result['internal_trace']['s_m'],result['internal_trace']['speed_mps'])
                return np.interp(common,np.r_[result['s_m'],result['terminal_s_m']],
                    np.r_[result['speed_mps'],result['terminal_speed_mps']])
            delta = abs(a['terminal_time_s']-b['terminal_time_s']) if valid else None
            rmse = float(np.sqrt(np.mean((velocity(a)-velocity(b))**2))*3.6) if valid else None
            rows.append({'valid':bool(valid),'lap_time_difference_s':delta,'speed_rmse_kmh':rmse,
                         'passed':bool(valid and delta<=.1 and rmse<=.5)})
        comparisons[group] = rows
        groups[group] = [r['summary'] for r in runs]
    report = {'passed':all(r['passed'] for rows in comparisons.values() for r in rows),
        'comparisons':comparisons,'summaries':groups,'mode':'replay',
        'raw_time_channels_sha256':binding,'initial_speed_mps':initial_speed,
        'same_raw_driver_channels_and_initial_state':True,'autonomous_driver_invoked':False,
        'output_resampling_and_internal_convergence_separate':True,
        'thresholds':{'lap_time_difference_s':.1,'speed_rmse_kmh':.5},'failed_trajectories_preserved':True}
    _write(output/'numerical.json',report)
    return report


def selected_numerical_audit(data,selected,output,config):
    require_calibration(data)
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    values,spec = selected['selected_parameters'],selected['candidate']
    params = _model(values,spec)
    from .environment import numerical_audit
    sim = _sim(data,params,config)
    fitted = next(row['best'] for row in selected['fit']['starts']
        if row['converged'] and row['best']['parameters']==values)
    baseline = read(Path(fitted['trajectory_directory'])/'autonomous.json.gz')
    auto = numerical_audit(sim,baseline,output/'autonomous')
    replay_sim = SimulatorV4(data['track'],data['rules'],params)
    replay = [{'lap_number':int(lap['lap_number']),**replay_numerical_audit(
        replay_sim,lap['driver_trace'],output/f'replay_lap_{lap["lap_number"]}')}
        for lap in data['laps']]
    report = {'passed':bool(auto['passed'] and all(r['passed'] for r in replay)),
        'autonomous':auto,'replay':replay,'candidate':spec,
        'candidate_vehicle_params_retained':True,'fixed_autonomous_raw_5m_demands_and_actions':True,
        'fitted_model_is_diagnostic_until_this_gate_passes':True}
    _write(output/'selected_numerical.json',report)
    return report


def _screen_model(data,spec,folder,config,initial_speed):
    workers = int(config.get('physics_screen_workers',1))
    if not 1 <= workers <= 4:
        raise ValueError('Fixed screening workers must lie in [1,4]')
    def evaluate(values,key):
        return evaluate_model(data,values,spec,folder/key,config,initial_speed_mps=initial_speed)
    if workers == 1: return sobol_viability(evaluate)
    points = qmc.scale(qmc.Sobol(len(PARAMETERS),scramble=True,seed=2026).random_base2(6),
                       BOUNDS[:,0],BOUNDS[:,1])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {f'screen_{i+1:02d}':pool.submit(_physics_evaluation_worker,
            data,x,spec,folder/f'screen_{i+1:02d}',config,initial_speed) for i,x in enumerate(points)}
        # Results and fit starts are consumed in the original Sobol order;
        # process completion order has no selection influence.
        report = sobol_viability(lambda values,key:futures[key].result())
    return {**report,'independent_trial_workers':workers,'completion_order_used_for_selection':False}


def qualify_sensitivity(sim,controller,output,initial_speed,*,initial_energy=2.):
    """Six actual carried laps; rejected complete times remain diagnostic."""
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    rows,proxies,crossings = [],[],[]
    speed,energy,carry = initial_speed,initial_energy,None
    for index in range(6):
        result = sim.simulate(controller,initial_speed_mps=speed,initial_energy_mj=energy,
                              carried_rule_state=carry)
        _write(output/f'lap_{index+1}.json.gz',result)
        summary = result['summary'];rows.append(summary)
        complete = bool(summary['completed_lap'] and summary['execution_feasible'])
        if complete:
            proxy = loss_proxy_audit(sim,result)
        else:
            proxy = {'passed':False,'reason':'Incomplete physical trajectory; saved prefix retained',
                     'actual_vehicle_reintegrated':False}
        proxies.append(proxy);_write(output/f'lap_{index+1}_proxy.json',proxy)
        if not complete: break
        next_carry = result['rule_certificate']['terminal_boundary_state']
        if index:
            previous = rows[index-1]
            crossings.append({'speed_error_mps':summary['speed_start_mps']-previous['speed_end_mps'],
                'energy_error_mj':summary['energy_start_mj']-previous['energy_end_mj'],
                'dc_error_kw':summary['initial_dc_kw']-previous['terminal_dc_kw'],
                'FSM_equal':summary['initial_rule_state']==carry})
        speed,energy,carry = result['terminal_speed_mps'],summary['energy_end_mj'],next_carry
    full = len(rows)==6 and all(r['completed_lap'] and r['execution_feasible'] for r in rows)
    continuity = full and all(abs(c['speed_error_mps'])<1e-10 and abs(c['energy_error_mj'])<1e-10
        and abs(c['dc_error_kw'])<1e-10 and c['FSM_equal'] for c in crossings)
    passed = bool(full and continuity and all(r['feasible'] for r in rows) and all(p['passed'] for p in proxies))
    report = {'passed':passed,'complete_six_actual_carried_laps':full,
        'physical_control_line_continuity':bool(continuity),'lap_summaries':rows,'crossings':crossings,
        'formal_objective_s':float(np.mean([r['lap_time_s'] for r in rows[-2:]])) if passed else None,
        'diagnostic_last_two_mean_s':float(np.mean([r['lap_time_s'] for r in rows[-2:]])) if full else None,
        'storage_energy_slope_mj_per_lap':float(np.mean([r['energy_end_mj']-r['energy_start_mj'] for r in rows[-2:]])) if full else None,
        'proxy_gates_passed':all(p['passed'] for p in proxies),'future_observed_velocity_used':False,
        'all_failures_saved':True,'infinite_horizon_proven':False,'global_optimality_proven':False}
    report['numerical_convergence_verified'] = False
    _write(output/'qualification.json',report)
    return report


def run_physics(data_cal,data_dev,output,config):
    """Frozen finite orchestration entry; root owns formal-run authorization."""
    require_calibration(data_cal);require_role(data_dev,'development')
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    config = dict(config)
    if (config.get('sobol_points',64)!=64 or config.get('sobol_seed',2026)!=2026
        or config.get('minimum_viable_fit_starts',3)!=3):
        raise ValueError('Fixed 64/2026/three-start protocol required')
    initial_speed = float(config.get('cold_initial_speed_mps',77.79423955757463))
    summaries=[]
    boundary = {'status':'not_available','reason':'No centerline/width array in the explicit calibration view',
                'geometry_changed':False}
    if 'boundary_reference' in data_cal:
        route = data_cal['track']['continuous_route']
        dense = route.evaluate(np.arange(0.,route.length_m,1.))
        boundary = segment_boundary_audit(np.c_[dense['x_m'],dense['y_m']],
                                          data_cal['boundary_reference'],route.direction)
        old = data_cal['track']['geometry_audit'].get('minimum_boundary_clearance_m')
        boundary.update(inherited_nearest_node_clearance_m=old,
            segment_vs_node_minimum_difference_m=None if old is None else
                boundary['minimum_boundary_clearance_m']-old)
    _write(output/'independent_boundary_audit.json',boundary)
    for spec in CANDIDATES:
        folder = output/spec['name'];folder.mkdir()
        def evaluate(values,key):
            return evaluate_model(data_cal,values,spec,folder/key,config,
                                   initial_speed_mps=initial_speed)
        screening = _screen_model(data_cal,spec,folder,config,initial_speed)
        fitting = capped_fit(evaluate,screening,
            max_iterations=int(config.get('fit_max_iterations_per_start',300)),
            max_evaluations=int(config.get('fit_max_evaluations_per_start',600)))
        row = {'candidate':spec,'screening':screening,'fit':fitting,
               'development':None,'selected_parameters':fitting['selected_parameters']}
        # Freeze fitted parameters before the separately tagged development
        # view is consumed. No residual-based re-registration is performed.
        if fitting['converged']:
            binding = {'candidate':spec,'parameters':fitting['selected_parameters'],
                       'estimated_laps':[3,4,5,6],'initial_speed_mps':initial_speed}
            encoded = json.dumps(binding,sort_keys=True,allow_nan=False).encode()
            _write(folder/'parameter_freeze.json',{**binding,'sha256':sha256(encoded).hexdigest()})
            row['development'] = evaluate_model(data_dev,fitting['selected_parameters'],spec,
                folder/'development',config,role='development',initial_speed_mps=initial_speed)
        _write(folder/'candidate.json',_json_finite(row));summaries.append(row)
    eligible = [r for r in summaries if r['development'] and r['development']['complete']
                and r['development']['physical'] and r['development']['autonomous_control_qualified']
                and np.isfinite(r['development']['objective'])]
    selected = None
    if eligible:
        minimum = min(r['development']['objective'] for r in eligible)
        # Equal parameter counts: prefer fewer extra engineering assumptions.
        tied = [r for r in eligible if r['development']['objective']<=minimum*1.01+1e-12]
        selected = min(tied,key=lambda r:next(i for i,c in enumerate(CANDIDATES) if c['name']==r['candidate']['name']))
    numerical = selected_numerical_audit(data_cal,selected,output/'selected_numerical',config) if selected else {
        'passed':False,'status':'withheld_no_eligible_physical_model',
        'reason':'No converged complete physical/control-qualified development candidate'}
    sensitivity=[]
    frozen = VehicleParamsV4()
    base_values = selected['selected_parameters'] if selected else [getattr(frozen,key) for key in PARAMETERS]
    base_spec = selected['candidate'] if selected else CANDIDATES[0]
    # Sensitivities never rescue a failed candidate or select a faster policy.
    for row in loss_sensitivity_registry():
        params = replace(_model(base_values,base_spec),**{k:v for k,v in row.items() if k!='name'})
        sim = _sim(data_cal,params,config,loss=True)
        report = qualify_sensitivity(sim,lambda obs:float(config.get('physics_baseline_action',0.)),
            output/'loss_sensitivity'/row['name'],initial_speed)
        sensitivity.append({'parameters':row,'qualification':report,'used_for_candidate_selection':False})
    axle_rows=[]
    for row in axle_sensitivity_registry():
        params = replace(_model(base_values,base_spec),axle_budget_enabled=True,
                         **{k:v for k,v in row.items() if k!='name'})
        sim = _sim(data_cal,params,config,loss=True)
        report = qualify_sensitivity(sim,lambda obs:float(config.get('physics_baseline_action',0.)),
            output/'axle_sensitivity'/row['name'],initial_speed)
        axle_rows.append({'parameters':row,'qualification':report,'used_for_candidate_selection':False})
    report = {'candidates':summaries,'selected_candidate':selected['candidate']['name'] if selected else None,
        'selected_parameters':dict(zip(PARAMETERS,selected['selected_parameters'])) if selected else None,
        'selected_two_path_accuracy_passed':bool(selected and
            selected['development']['replay_accuracy_gates']['accuracy_passed'] and
            selected['development']['autonomous_accuracy_gates']['accuracy_passed']),
        'selected_numerical_gates':numerical,
        'selected_model_formal_release_passed':bool(selected and numerical['passed'] and
            selected['development']['replay_accuracy_gates']['accuracy_passed'] and
            selected['development']['autonomous_accuracy_gates']['accuracy_passed']),
        'selection_uses_development_laps':[7,9],'fit_uses_only_calibration_laps':[3,4,5,6],
        'loss_sensitivities':sensitivity,'axle_sensitivities':axle_rows,
        'independent_boundary_audit':boundary,
        'sensitivity_fallback_is_diagnostic_only':selected is None,
        'sensitivity_fallback_parameters':dict(zip(PARAMETERS,base_values)) if selected is None else None,
        'cold_initial_speed_mps':initial_speed,
        'learning_keeps_ideal_baseline':True,'FP2_accessed':False,'bounds_expanded':False,
        'all_failure_trajectories_saved':True,'true_SOC_or_axle_parameters_identified':False}
    report = _json_finite(report)
    _write(output/'physics.json',report)
    return report


def _json_finite(value):
    """JSON has no Infinity; failures retain null objectives with explicit flags."""
    if isinstance(value,dict): return {k:_json_finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_finite(v) for v in value]
    if isinstance(value,np.ndarray): return _json_finite(value.tolist())
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.generic): return value.item()
    return value
