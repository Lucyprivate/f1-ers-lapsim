"""Replay accepted records to audit the research surrogate's own requests.

The FIA maximum driver power-demand map is unavailable. Comparing this
research surrogate with an availability-limited effective request is an added
engineering gate; it does not prove that a real car violated C5.12.4, nor does
that article require constant actual K power. Actual power remains separate.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from f1ers_v7._core.integrator import IntegrationState
from f1ers_v7._core.base_rules import RuleShield


def audit_proxy_consistency(result, sim, shield=None):
    """Pure-record reconstruction: never integrate or read observed velocity."""
    tr={k:np.asarray(v) for k,v in result['internal_trace'].items()}
    shield=shield or RuleShield(sim.rules,sim.geometry,sim.rule_config)
    initial=result['rule_certificate']['initial_boundary_state']
    current=IntegrationState(float(tr['s_m'][0]),float(tr['speed_mps'][0]),float(tr['time_s'][0]),
        float(tr['energy_mj'][0]),float(tr['recharge_mj'][0]),float(initial['previous_dc_kw']))
    shield.reset(current,result['summary']['initial_rule_state'])
    stats={phase:{'samples':0,'effective_query_shortfall_samples':0,'max_shortfall_mechanical_kw':0.,
        'public_cap_samples':0,'actual_allocation_shortfall_samples':0,'first_shortfalls':[],
        'largest_shortfall':None} for phase in ('NORMAL','HOLD','LIMITED')}
    normal_history=[];cursor=0;eta=shield.config.deploy_efficiency
    tolerance=shield.config.tolerance
    action_space=result['summary']['raw_action_space']
    for i,decision in enumerate(result['decision_info']):
        command={'wheel_force_n':result['driver_profile']['wheel_force_n'][i],'brake_fraction':0.}
        shield.decide(current,command,result['requested_actions'][i],action_space)
        end=int(np.searchsorted(tr['s_m'],decision['s_m']-1e-7,side='left'))
        if end<cursor or end>=len(tr['s_m']) or abs(float(tr['s_m'][end])-decision['s_m'])>1e-5:
            raise ValueError('Saved decision boundary does not match internal trace')
        records=[]
        for j in range(cursor,end):
            state=IntegrationState(float(tr['s_m'][j]),float(tr['speed_mps'][j]),float(tr['time_s'][j]),
                float(tr['energy_mj'][j]),float(tr['recharge_mj'][j]),
                float(tr['dc_kw'][j-1]) if j>0 else float(initial['previous_dc_kw']))
            dt=float(tr['time_s'][j+1]-tr['time_s'][j])
            phase=shield.state.phase
            if phase=='HOLD' and state.elapsed_time_s>=shield.state.hold_until_s-tolerance:
                phase='LIMITED'
            query=shield.query(state,command,sim.geometry,dt)
            declared=max(0.,shield.state.reference_mechanical_kw-shield._reduction_at(state.elapsed_time_s+dt))
            public=float(query['deployment_cap_kw'])*eta
            effective=float(query['requested_dc_kw'])*eta
            actual=float(tr['mguk_mechanical_kw'][j])
            row={'s_m':state.s_m,'time_s':state.elapsed_time_s,'dt_s':dt,'energy_mj':state.energy_mj,
                'wheel_force_n':float(tr['wheel_force_n'][j]),'wheel_power_kw':float(tr['wheel_power_kw'][j]),
                'phase':phase,'declared_surrogate_reference_minus_reduction_kw':declared,
                'effective_query_dc_kw':float(query['requested_dc_kw']),
                'effective_query_mechanical_kw':effective,'actual_dc_kw':float(tr['dc_kw'][j]),
                'actual_K_mechanical_kw':actual,'public_dc_cap_kw':float(query['deployment_cap_kw']),
                'committed_rule_state':asdict(shield.state)}
            st=stats[phase];st['samples']+=1
            if command['wheel_force_n']>=0. and declared>tolerance:
                gap=min(declared,public)-effective
                if public<declared-tolerance:st['public_cap_samples']+=1
                if phase in ('HOLD','LIMITED') and gap>tolerance:
                    row['surrogate_shortfall_mechanical_kw']=gap
                    row['classification']='effective_query_energy_reserve_clip_below_declared_proxy'
                    st['effective_query_shortfall_samples']+=1
                    if len(st['first_shortfalls'])<3:st['first_shortfalls'].append(row)
                    if gap>st['max_shortfall_mechanical_kw']:
                        st['max_shortfall_mechanical_kw']=gap;st['largest_shortfall']=row
                if actual<effective-tolerance:st['actual_allocation_shortfall_samples']+=1
                if phase=='NORMAL' and effective<declared-tolerance and public>=declared-tolerance:
                    if len(normal_history)<3:
                        row['classification']='NORMAL_effective_energy_bound_below_historical_peak_without_HOLD'
                        normal_history.append(row)
            records.append({'s_start_m':float(tr['s_m'][j]),'s_end_m':float(tr['s_m'][j+1]),
                'time_start_s':float(tr['time_s'][j]),'time_end_s':float(tr['time_s'][j+1]),
                'speed_start_mps':float(tr['speed_mps'][j]),'speed_next_mps':float(tr['speed_mps'][j+1]),
                'dc_kw':float(tr['dc_kw'][j]),'dt_s':dt,'wheel_force_n':float(tr['wheel_force_n'][j]),
                'energy_next_mj':float(tr['energy_mj'][j+1]),'recharge_next_mj':float(tr['recharge_mj'][j+1])})
        if end==cursor:
            continue
        current=IntegrationState(float(tr['s_m'][end]),float(tr['speed_mps'][end]),float(tr['time_s'][end]),
            float(tr['energy_mj'][end]),float(tr['recharge_mj'][end]),float(tr['dc_kw'][end-1]))
        shield.accept(current,records);cursor=end
    terminal=asdict(shield.state)
    original=result['rule_certificate']['terminal_boundary_state']
    replay_certificate=shield.certificate()
    exact=terminal==original
    count_same=replay_certificate['accepted_records_checked']==result['rule_certificate']['accepted_records_checked']
    hold=stats['HOLD']['effective_query_shortfall_samples']==0
    limited=stats['LIMITED']['effective_query_shortfall_samples']==0
    return {'additional_surrogate_hold_request_gate_passed':bool(hold and exact and count_same),
        'additional_surrogate_hold_and_limited_request_gate_passed':bool(hold and limited and exact and count_same),
        'FIA_C5_12_4_actual_K_constant_requirement_claimed':False,
        'FIA_violation_proven':False,'full_FIA_ECU_legality_proven':False,
        'gate_interpretation':'Added reachable-surrogate reference/reduction vs effective query consistency; actual K and public cap are separate; original frozen run gates are unchanged',
        'state_reconstruction':{'terminal_all_fields_match':exact,'accepted_records_count_match':count_same,
            'replayed_accepted_records':replay_certificate['accepted_records_checked'],
            'saved_accepted_records':result['rule_certificate']['accepted_records_checked'],
            'terminal_rule_state':terminal,'saved_terminal_rule_state':original},
        'phase_statistics':stats,'NORMAL_historical_peak_unreachable_first_samples':normal_history,
        'actual_vehicle_reintegrated':False}
