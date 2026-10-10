"""R7 convergence helpers preserving geometric protection and causal rules."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np

from f1ers_v4.vehicle import IntegratorConfig
from .frozen_demands import FrozenDemandDriver
from .geometry_guard import GeometryGuardSimulator, GeometryGuardEnv, GeometryAdvanceRuleShield
from .proxy_audit import audit_proxy_consistency
from .learning import _write


def label_elapsed_time(result):
    """Do not call a physically stopped partial trajectory's elapsed time a lap."""
    complete=bool(result['summary']['completed_lap'])
    result['summary']['elapsed_time_s']=float(result['terminal_state']['elapsed_time_s'])
    result['summary']['formal_lap_time_s']=float(result['terminal_state']['elapsed_time_s']) if complete else None
    result['summary']['time_interpretation']='complete_lap' if complete else 'partial_elapsed_to_stop'
    if not complete:
        result['summary']['lap_time_s']=None
    return result


def proxy_for_r7(sim,result):
    shield=GeometryAdvanceRuleShield(sim.rules,sim.geometry,sim.rule_config,
        sim.geometry_guard_map,sim.margin_s)
    return audit_proxy_consistency(result,sim,shield)


def freeze_r7_simulator(sim,profile):
    """Freeze original demands; retain the R7 shield and fixed 5 m decisions."""
    immutable={key:np.asarray(profile[key],float).copy() for key in ('s_m','wheel_force_n')}
    for value in immutable.values():value.setflags(write=False)

    class FrozenR7Env(GeometryGuardEnv):
        def reset(self,*args,**kwargs):
            super().reset(*args,**kwargs)
            self.driver=FrozenDemandDriver(self.sim.geometry,self.sim.vehicle,immutable)
            self.driver.shield=self.shield
            return self.get_obs()

    class FrozenR7Simulator(GeometryGuardSimulator):
        def create_env(self,action_space='signed',reward_config=None,initial_speed_mps=None,
                       carried_rule_state=None,output_ds_m=5.,numerics=None,initial_energy_mj=2.):
            if isinstance(numerics,dict):numerics=IntegratorConfig(**numerics)
            return FrozenR7Env(self,action_space,reward_config,initial_speed_mps,
                carried_rule_state,output_ds_m,numerics,initial_energy_mj)

    return FrozenR7Simulator(sim.track,sim.rules,sim.vehicle,sim.config,sim.rule_config,
        margin_s=sim.margin_s)


def numerical_audit_r7(sim,result,output):
    """Six fixed-demand runs: output resampling separately from integration."""
    output=Path(output)
    frozen=freeze_r7_simulator(sim,result['driver_profile'])
    actions=np.asarray(result['requested_actions'],float).copy()
    initial=float(result['summary']['speed_start_mps'])
    energy=float(result['summary']['energy_start_mj'])
    carry=result['summary']['initial_rule_state']
    space=result['summary']['raw_action_space']
    groups={};comparisons={}
    for group,settings in (
        ('output',[(10.,.02,1.),(5.,.02,1.),(2.5,.02,1.)]),
        ('integration',[(5.,.02,1.),(5.,.01,.5),(5.,.005,.25)])):
        runs=[];proxies=[]
        for i,(ds,dt,max_ds) in enumerate(settings):
            cfg=replace(sim.config,dt_max=dt,max_ds=max_ds)
            run=frozen.simulate(actions,initial_speed_mps=initial,initial_energy_mj=energy,
                carried_rule_state=carry,action_space=space,output_ds_m=ds,numerics=cfg)
            label_elapsed_time(run)
            proxy=proxy_for_r7(frozen,run)
            _write(output/f'{group}_{i}.json.gz',run)
            _write(output/f'{group}_{i}_proxy.json',proxy)
            runs.append(run);proxies.append(proxy)
        rows=[]
        for i,(a,b) in enumerate(zip(runs[:-1],runs[1:])):
            valid=all(r['summary']['completed_lap'] and r['summary']['execution_feasible'] and
                r['summary']['public_rules_passed'] and r['summary']['sustainable'] and
                r['summary']['periodic_speed_closed'] and r['summary']['dc_boundary_closed'] and
                r['summary']['rule_state_closed'] for r in (a,b))
            valid=valid and all(p['additional_surrogate_hold_and_limited_request_gate_passed'] for p in proxies[i:i+2])
            common=np.linspace(0.,sim.length_m,int(np.ceil(sim.length_m/.5))+1)
            def velocities(r):
                if group=='integration':
                    return np.interp(common,r['internal_trace']['s_m'],r['internal_trace']['speed_mps'])
                return np.interp(common,np.r_[r['s_m'],sim.length_m],np.r_[r['speed_mps'],r['terminal_speed_mps']])
            delta=abs(a['terminal_time_s']-b['terminal_time_s']) if valid else None
            rmse=float(np.sqrt(np.mean((velocities(a)-velocities(b))**2))*3.6) if valid else None
            rows.append({'valid':valid,'lap_time_difference_s':delta,'speed_rmse_kmh':rmse,
                'passed':bool(valid and delta<=.1 and rmse<=.5)})
        groups[group]={'summaries':[r['summary'] for r in runs],
                       'proxy_gates':[p['additional_surrogate_hold_and_limited_request_gate_passed'] for p in proxies]}
        comparisons[group]=rows
    return {'passed':all(row['passed'] for rows in comparisons.values() for row in rows),
        'comparisons':comparisons,'groups':groups,'fixed_raw_actor_actions':True,
        'fixed_original_5m_wheel_demands':True,'driver_and_actor_decision_spacing_m':5.,
        'geometry_guard_preserved':True,'causal_guard_and_FSM_queries_retained':True,
        'output_sampling_and_internal_convergence_separate':True,
        'same_initial_v_energy_and_carried_rule_state':True,
        'observed_future_velocity_used':False,'whole_lap_power_scaling':False}
