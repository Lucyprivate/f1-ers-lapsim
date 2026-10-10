"""Predeclared, resumable local control experiments; no late-data access."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import gzip
import json
import time
import numpy as np
from f1ers_v4.artifacts import read_json, write_json, sha256, capture_provenance
from f1ers_v4.calibration import write_compressed
from f1ers_v4.data import load_local_dataset
from f1ers_v4.vehicle import IntegratorConfig
from f1ers_v4.vehicle import VehicleParamsV4
from .simulator import SimulatorV6
from .rules import RuleConfig
from .teacher import BlockController, search
from .metrics import score, legal_time


def provenance(root, require_clean=True):
    value=capture_provenance(root,require_clean=require_clean)
    for directory in ('f1ers_v5','f1ers_v6','config/v6'):
        value['source_sha256'].update({p.relative_to(root).as_posix():sha256(p)
            for p in sorted((root/directory).rglob('*')) if p.is_file() and p.suffix in ('.py','.json')})
    value['schema_version']=6
    return value


def numerical_audit(sim, controller, result, output, action_space):
    initial=float(result['summary']['speed_start_mps'])
    initial_energy=float(result['internal_trace']['energy_mj'][0])
    carried=result['summary']['initial_rule_state']
    from .frozen_demands import freeze_simulator
    frozen=freeze_simulator(sim,result['driver_profile'])
    actions=np.asarray(result['requested_actions'],float)
    groups={}; comparisons={}
    for group, settings in (
        ('output',[(10.,.02,1.),(5.,.02,1.),(2.5,.02,1.)]),
        ('integration',[(5.,.02,1.),(5.,.01,.5),(5.,.005,.25)])):
        runs=[]
        for i,(ds,dt,max_ds) in enumerate(settings):
            cfg=replace(sim.config,dt_max=dt,max_ds=max_ds)
            run=frozen.simulate(actions,initial_speed_mps=initial,initial_energy_mj=initial_energy,
                carried_rule_state=carried,action_space=action_space,output_ds_m=ds,numerics=cfg)
            write_compressed(output/f'{group}_{i}.json.gz',run)
            runs.append(run)
        rows=[]
        for a,b in zip(runs[:-1],runs[1:]):
            valid=all(r['summary']['completed_lap'] and r['summary']['execution_feasible'] and
                r['summary']['public_rules_passed'] for r in (a,b))
            # Output sampling is measured from returned samples, integration
            # from accepted internal states. These are independent checks.
            common=np.linspace(0.,sim.length_m,int(np.ceil(sim.length_m/.5))+1)
            def velocities(r):
                if group=='integration': return np.interp(common,r['internal_trace']['s_m'],r['internal_trace']['speed_mps'])
                return np.interp(common,np.r_[r['s_m'],sim.length_m],np.r_[r['speed_mps'],r['terminal_speed_mps']])
            delta=abs(a['terminal_time_s']-b['terminal_time_s']) if valid else None
            rmse=float(np.sqrt(np.mean((velocities(a)-velocities(b))**2))*3.6) if valid else None
            rows.append({'valid':valid,'lap_time_difference_s':delta,'speed_rmse_kmh':rmse,
                'passed':bool(valid and delta<=.1 and rmse<=.5)})
        groups[group]=[r['summary'] for r in runs]; comparisons[group]=rows
    return {'passed':all(x['passed'] for rows in comparisons.values() for x in rows),
        'comparisons':comparisons,'summaries':groups,'original_driver_and_ers_decision_spacing_m':5.,
        'observed_speed_used':False,'output_sampling_and_internal_convergence_separate':True,
        'raw_wheel_demand_and_ers_actions_frozen':True,
        'control_note':'Freeze original signed wheel demand and raw ERS decisions at the same 5m boundaries; retain causal rule execution'}


def contiguous_laps(sim, controller, selected, output, action_space, cycles=6):
    start=float(selected['summary']['speed_start_mps']); energy=2.; carry=None; rows=[]
    for i in range(cycles):
        result=sim.simulate(controller,initial_speed_mps=start,initial_energy_mj=energy,
            carried_rule_state=carry,action_space=action_space)
        write_compressed(output/f'lap_{i+1}.json.gz',result)
        s=result['summary']; terminal=result['terminal_state']
        rows.append({'lap':i+1,'time_s':s['lap_time_s'],'initial_speed_mps':start,
            'terminal_speed_mps':terminal['speed_mps'],'initial_energy_mj':energy,
            'terminal_energy_mj':terminal['energy_mj'],'energy_change_mj':terminal['energy_mj']-energy,
            'physical':bool(s['completed_lap'] and s['execution_feasible']),
            'per_lap_certified':bool(s['feasible']),
            'public_subset_passed':s['public_rules_passed'],
            'dc_balance_residual_mj':s['energy_balance_residual_mj'],
            'initial_rule_state':carry,'terminal_rule_state':result['rule_certificate']['terminal_state'],
            'auxiliary_adjusted_change_mj':terminal['energy_mj']-energy-50*s['lap_time_s']/1e6})
        if not rows[-1]['physical']: break
        start=terminal['speed_mps']; energy=terminal['energy_mj']; carry=result['rule_certificate']['terminal_state']
    passed=len(rows)==cycles and all(r['per_lap_certified'] and r['energy_change_mj']>=-1e-8 for r in rows)
    return {'passed':passed,'cycles':cycles,'rows':rows,'energy_or_rule_reset_between_laps':False,
        'harvest_counter_reset_only':True,'true_ES_SOC_proven':False,
        'speed_boundary_jump_mps':0.0,'auxiliary_50w_all_laps_nonnegative':all(r['auxiliary_adjusted_change_mj']>=-1e-8 for r in rows),
        'stable_window_mean_time_s':float(np.mean([r['time_s'] for r in rows[-2:]])),
        'stable_window_energy_drift_mj':[r['energy_change_mj'] for r in rows[-2:]],
        'representative_result_path':str(output/f'lap_{len(rows)}.json.gz'),
        'infinite_horizon_periodicity_proven':False}


def run(root, name='austria_2026_v6'):
    from .learning import train_bc, train_sac
    from .reward import RewardConfig
    root=Path(root).resolve(); output=root/'results'/name
    if (output/'experiment.json').exists() or (output/'started.json').exists():
        raise RuntimeError('Existing formal run cannot be overwritten; verify it or select a new name')
    source=provenance(root,True); protocol=read_json(root/'config/v6/protocol.json')
    dataset=load_local_dataset(root); output.mkdir(parents=True,exist_ok=True)
    write_json(output/'started.json',{'provenance':source,'protocol':protocol,'new_test_opened':False})
    started=time.perf_counter(); rounds=[]; candidates=[]
    def record(identifier,hypothesis,change,result,**extra):
        comparison=None if result is None else score(result,dataset,'calibration')
        row={'round':identifier,'hypothesis':hypothesis,'change':change,
            'calibration':comparison,'summary':None if result is None else result['summary'],**extra}
        rounds.append(row); write_json(output/'rounds.json',rounds)
        if result is not None: write_compressed(output/f'{identifier}_representative.json.gz',result)
        print(f'{identifier}: '+('no feasible candidate' if result is None else
            f"{result['summary']['lap_time_s']:.6f}s; legal={result['summary'].get('feasible')}; sectors="+
            str(comparison['aggregate'].get('sector_speed_rmse_kmh'))),flush=True)
        return row
    # R0 reproduces the frozen mechanism. The audited saved preflight is kept
    # outside the new formal source freeze and labeled accordingly.
    from f1ers_v4.simulator import SimulatorV4
    base_vehicle=VehicleParamsV4(mu=1.8,drag_area_m2=1.3,downforce_area_m2=3.9,
        equivalent_power_kw=541.25,replay_brake_gain=.55)
    old=SimulatorV4(dataset['track'],dataset['rules'],base_vehicle)
    old_result=old.simulate(mode='autonomous')
    record('R0','Fixed ICE-only target acceleration suppresses K speed response',
        'Audit frozen neutral dynamics; compare measured preflight positive-K experiment',old_result,
        historic_ml_training_run=False,preflight_directory='preflight',
        frozen_model_error_is_not_a_SAC_error=True)
    neutral=None; heuristic_cases=[]
    for guard in protocol['tail_guard_choices_m']:
        sim=SimulatorV6(dataset['track'],dataset['rules'],rule_config=RuleConfig(tail_guard_m=guard))
        if neutral is None:
            neutral=sim.periodic_simulate(None,max_cycles=4)
            candidates.append({'name':'neutral','controller':None,'result':neutral,'sim':sim,'space':'signed'})
        heuristic=sim.periodic_simulate(BlockController([.7]),max_cycles=4)
        write_compressed(output/'R1'/f'guard_{int(guard)}.json.gz',heuristic)
        carry_check=contiguous_laps(sim,BlockController([.7]),heuristic,output/'R1'/f'carry_{int(guard)}','signed',cycles=3)
        heuristic_cases.append({'guard_m':guard,'summary':heuristic['summary'],'contiguous':carry_check})
        if carry_check['passed']:
            candidates.append({'name':f'heuristic_{int(guard)}','controller':BlockController([.7]),
                'result':heuristic,'sim':sim,'space':'signed'})
    best=min(candidates,key=lambda c:legal_time(c['result']))
    record('R1','Power response improves pace but free control-line reset can create false gains',
        'Shared execution; strict SOC/speed/DC/FSM gates; conservative tail-mask comparison',best['result'],cases=heuristic_cases)
    sim=best['sim']
    if any(abs(getattr(sim.vehicle,key)-value)>1e-12 for key,value in protocol['vehicle'].items()
           if key not in ('energy_start_mj','energy_window_mj')):
        raise ValueError('Actual vehicle differs from the declared protocol')
    signed=search(sim,output/'R2',blocks=6,evaluations=protocol['rounds'][2]['evaluations'],action_space='signed')
    record('R2','Gradient search is unreliable on hard-shield action plateaus',
        'Bounded seeded six-block signed numerical search',signed['result'],search=signed['report'])
    if signed['result'] is not None:
        candidates.append({'name':'signed_teacher','controller':signed['controller'],'result':signed['result'],'sim':sim,'space':'signed'})
    initial=None if signed['controller'] is None else signed['controller'].values
    aware=search(sim,output/'R3',blocks=8,evaluations=protocol['rounds'][3]['evaluations'],action_space='mode_aware',initial=initial)
    record('R3','Unreachable negative drive commands and coarse blocks waste exploration',
        'Mode-aware braking/drive action and eight-block refinement',aware['result'],search=aware['report'])
    if aware['result'] is not None:
        candidates.append({'name':'mode_aware_teacher','controller':aware['controller'],'result':aware['result'],'sim':sim,'space':'mode_aware'})
    if aware['result'] is None:
        raise RuntimeError('No complete sustainable numerical teacher; ML withheld and all trials retained')
    teacher=aware['result']; bc_cases=[]; bc_runs=[]
    for seed in protocol['seeds']:
        trained=train_bc(teacher,sim,output/'R4'/str(seed),seed=seed,
            epochs=protocol['rounds'][4]['bc_epochs'],dagger_rounds=protocol['rounds'][4]['dagger_rounds'],
            dagger_epochs=protocol['rounds'][4]['dagger_epochs'],action_space='mode_aware',progress=print)
        bc_runs.append(trained); bc_cases.append(trained['metadata'])
        if trained['result'] is not None:
            candidates.append({'name':f'BC_{seed}','controller':trained['policy'],'result':trained['result'],'sim':sim,'space':'mode_aware',
                'checkpoint':f'R4/{seed}/best_bc_policy.npz'})
    bc_best=min([r['result'] for r in bc_runs if r['result'] is not None],key=legal_time,default=None)
    record('R4','An action table and an incompatible legacy checkpoint are not deployed state feedback',
        'Versioned current-state MLP with three seeds and disclosed fixed-label DAgger',bc_best,cases=bc_cases)
    sac_cases=[]; sac_runs=[]
    for seed,bc in zip(protocol['seeds'],bc_runs):
        warm=bc['policy'] or bc['last_policy']
        for temperature in ('fixed','auto'):
            trained=train_sac(sim,warm,output/'R5'/f'{seed}_{temperature}',
                episodes=protocol['rounds'][5]['episodes_per_case'],seed=seed,
                initial_alpha=protocol['sac_budget']['initial_alpha'],
                temperature_mode=temperature,action_space='mode_aware',progress=print)
            sac_runs.append(trained); sac_cases.append(trained['metadata'])
            if trained['result'] is not None:
                candidates.append({'name':f'SAC_{seed}_{temperature}','controller':trained['policy'],'result':trained['result'],
                    'sim':sim,'space':'mode_aware','checkpoint':f'R5/{seed}_{temperature}/best_feasible_policy.npz'})
    sac_best=min([r['result'] for r in sac_runs if r['result'] is not None],key=legal_time,default=None)
    record('R5','Entropy and inaccurate long-horizon critics can destroy a good warm start',
        'Gamma 1, seconds reward, fixed/automatic entropy with strict incumbent selection',sac_best,cases=sac_cases)
    correction_cases=[]; correction_runs=[]
    for seed,bc in zip(protocol['seeds'],bc_runs):
        warm=bc['policy'] or bc['last_policy']
        trained=train_sac(sim,warm,output/'R6'/str(seed),episodes=protocol['rounds'][6]['episodes_per_seed'],seed=seed,
            initial_alpha=protocol['sac_budget']['initial_alpha'],
            temperature_mode='fixed',action_space='mode_aware',mc_warmstart=True,
            reward_config=RewardConfig(potential_seconds_per_mj=protocol['rounds'][6]['potential_seconds_per_mj']),progress=print)
        correction_runs.append(trained); correction_cases.append(trained['metadata'])
        if trained['result'] is not None:
            candidates.append({'name':f'SAC_MC_{seed}','controller':trained['policy'],'result':trained['result'],
                'sim':sim,'space':'mode_aware','checkpoint':f'R6/{seed}/best_feasible_policy.npz'})
    corrected=min([r['result'] for r in correction_runs if r['result'] is not None],key=legal_time,default=None)
    record('R6','Bootstrap critic errors and delayed energy credit need causal training correction',
        'Realised MC critic initialization; terminal-zero telescoping energy potential; retain incumbent',corrected,cases=correction_cases,
        correction_uses_R5_logs=True,reward_shaping_changes_complete_episode_time_objective=False)
    # Single-lap success is insufficient: certify each faster incumbent on
    # real consecutive laps before accepting it. All rejected attempts remain.
    multi_rejections=[]; certified=[]
    for index,candidate in enumerate(sorted(candidates,key=lambda c:legal_time(c['result']))):
        if not np.isfinite(legal_time(candidate['result'])): continue
        attempt=contiguous_laps(candidate['sim'],candidate['controller'],candidate['result'],
            output/'contiguous'/f'candidate_{index:02d}',candidate['space'])
        if attempt['passed']:
            certified.append((attempt['stable_window_mean_time_s'],candidate,attempt))
        else: multi_rejections.append({'candidate':candidate['name'],'continuity':attempt})
    if not certified:
        write_json(output/'multi_cycle_rejections.json',multi_rejections)
        raise RuntimeError('No actual multi-cycle sustainable candidate; results retained, no optimum invented')
    _,best,continuity=min(certified,key=lambda row:row[0])
    with gzip.open(continuity['representative_result_path'],'rt',encoding='utf-8') as stream:
        selected=json.load(stream)
    sim=best['sim']; controller=best['controller']; space=best['space']
    write_compressed(output/'selected_trajectory.json.gz',selected)
    final_cal=score(selected,dataset,'calibration'); development=score(selected,dataset,'development')
    numerical=numerical_audit(sim,controller,selected,output/'numerical',space)
    sensitivity=[]
    for field,values in [('mass_kg',[780.,820.]),('ice_power_kw',[380.,420.])]:
        for value in values:
            perturbed=SimulatorV6(dataset['track'],dataset['rules'],replace(sim.vehicle,**{field:value}),rule_config=sim.rule_config)
            run=perturbed.periodic_simulate(controller,action_space=space,max_cycles=4)
            write_compressed(output/'sensitivity'/f'{field}_{int(value)}.json.gz',run)
            sensitivity.append({'field':field,'value':value,'summary':run['summary'],'calibration':score(run,dataset,'calibration'),
                'used_for_controller_selection':False})
    out={'schema_version':6,'name':name,'provenance':source,'protocol':protocol,'rounds':rounds,
        'selected':{k:v for k,v in best.items() if k not in ('controller','result','sim')},
        'controller':controller.to_dict() if isinstance(controller,BlockController) else
            ({'kind':'MLP','checkpoint':best['checkpoint']} if controller is not None else {'kind':'neutral'}),
        'vehicle':sim.vehicle.to_dict(),'rule_config':selected['rule_certificate']['config'],
        'selected_summary':selected['summary'],'calibration':final_cal,'development':development,
        'numerical':numerical,'contiguous_laps':continuity,'multi_cycle_rejections':multi_rejections,'sensitivity':sensitivity,
        'candidates':[{'name':c['name'],'summary':c['result']['summary']} for c in candidates],
        'multi_cycle_certified_candidates':[{'name':c['name'],'stable_window_mean_time_s':cost,'continuity':carry}
            for cost,c,carry in certified],
        'baseline_time_s':neutral['summary']['lap_time_s'],'tested_model_gain_s':neutral['summary']['lap_time_s']-selected['summary']['lap_time_s'],
        'gates':{'public_subset_control':selected['summary']['feasible'],'numerical':numerical['passed'],
            'contiguous_sustainable':continuity['passed'],'calibration_fidelity':final_cal['accuracy_passed'],
            'development_fidelity':development['accuracy_passed'],'new_session':False,'full_fia_verified':False},
        'new_test_opened':False,'policy_training_authorized_for_local_research':True,
        'real_vehicle_benefit_claim':False,'global_optimality_proven':False,
        'fastest_scope':'retained round incumbents that pass six physically contiguous laps; all discarded teacher proposals and unselected ML checkpoints are preserved but not exhaustively multi-cycle ranked',
        'browser_verification':{'status':'not_measured'},'elapsed_seconds':time.perf_counter()-started}
    write_json(output/'experiment.json',out)
    inventory={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write_json(output/'manifest.json',{'schema_version':6,'code_commit':source['code_commit'],'files':inventory})
    return out
