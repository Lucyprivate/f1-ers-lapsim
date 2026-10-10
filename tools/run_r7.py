"""Independent R7 experiment and exact saved-result verification.

Copy this entry point into the repository and commit it with its modules and
config before running. Outputs must use a new ignored directory. External
stdout logs are not part of the owned calculation inventory.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import time


def digest(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_member(root,name):
    p=PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or any(x in ('','.','..') for x in name.split('/')) or ':' in name:
        raise ValueError('Unsafe inventory path: '+str(name))
    result=(root/name).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError('Inventory file is missing or escapes its root: '+str(name))
    return result


def validate_inventory(root,manifest):
    for name,expected in manifest['files'].items():
        if digest(safe_member(root,name))!=expected:raise ValueError('Inventory digest changed: '+name)


def imports(repo):
    sys.path.insert(0,str(repo))
    import numpy as np
    from f1ers_v4.data import load_local_dataset
    from f1ers_v6.experiment import provenance
    from f1ers_v6.geometry_guard import GeometryGuardSimulator
    from f1ers_v6.rules import RuleConfig
    from f1ers_v6.learning import load_policy,_write,_jsonable
    from f1ers_v6.metrics import score
    from f1ers_v6.r7 import proxy_for_r7,numerical_audit_r7,label_elapsed_time
    return locals()


def dependencies():
    return {name:importlib.metadata.version(name) for name in ('numpy','scipy','numba')}


def frozen_inputs_unchanged(repo,freeze):
    return all(digest(safe_member(repo,name))==expected
        for group in ('source_sha256','data_sha256')
        for name,expected in freeze.get(group,{}).items())


def validate_config(config):
    required={'schema_version':7,'prior_code_commit':'82d28f8a50c58bd8b77e47a44723076c56be4748',
        'margins_s':[.2,.5,1.],'physical_laps_per_margin':6,'steady_lap_numbers':[5,6],
        'policy_seed':2027,'policy_variant':'raw_bc','action_space':'mode_aware','initial_energy_mj':2.,
        'tail_guard_m':1000.,'vehicle_mass_kg':800.,'geometry_power_threshold_kw':332.5,
        'calibration_laps':[3,4,5,6],'development_laps':[7,9],
        'calibration_residual_used_for_selection':False,'new_SAC_episodes':0,
        'policy_relative_to_prior_run':'R4/2027/raw_bc_policy.npz',
        'policy_metadata_relative_to_prior_run':'R4/2027/bc_metadata.json',
        'numerical_output_spacing_m':[10.,5.,2.5],
        'numerical_internal_steps':[[.02,1.],[.01,.5],[.005,.25]],
        'numerical_time_tolerance_s':.1,'numerical_speed_tolerance_kmh':.5}
    if any(config.get(k)!=v for k,v in required.items()):
        raise ValueError('Fixed R7 policy, splits, margins or numerical budget changed')


def load_data(repo,api):
    data=api['load_local_dataset'](repo)
    cal=sorted(int(l['lap_number']) for l in data['laps'] if l['role']=='calibration')
    dev=sorted(int(l['lap_number']) for l in data['laps'] if l['role']=='development')
    if cal!=[3,4,5,6] or dev!=[7,9] or not data.get('audit',{}).get('data_gate_passed'):
        raise ValueError('Frozen local data/split gate failed; no alternate test source allowed')
    return data


def make_sim(api,data,margin):
    return api['GeometryGuardSimulator'](data['track'],data['rules'],
        rule_config=api['RuleConfig'](tail_guard_m=1000.),margin_s=margin)


def verify(repo,output,api,write_result=True):
    """Replay only the frozen winner; saved development scores cannot reselect."""
    np=api['np'];experiment=read_json(output/'experiment.json')
    freeze=read_json(output/'freeze.json')
    source_ok=frozen_inputs_unchanged(repo,freeze)
    manifest=read_json(output/'manifest.json') if (output/'manifest.json').exists() else None
    if manifest is not None:validate_inventory(output,manifest)
    checkpoint=output/experiment['checkpoint_inventory']['copied_checkpoint_relative']
    checkpoint_ok=digest(checkpoint)==experiment['checkpoint_inventory']['checkpoint_sha256']
    basic={'schema_version':7,'source_commit':experiment['source_commit'],
        'source_hashes_passed':source_ok,'checkpoint_hash_passed':checkpoint_ok,
        'manifest_inputs_checked':manifest is not None,'dependencies_match':dependencies()==freeze['dependencies'],
        'FP2_accessed':False,'new_policy_selection_or_training':False}
    if not source_ok or not checkpoint_ok or not basic['dependencies_match']:
        raise ValueError('Frozen R7 source, checkpoint or dependency input changed')
    if digest(output/'frozen_protocol.json')!=experiment['config_sha256']:
        raise ValueError('Frozen R7 configuration bytes changed')
    if experiment['selected_margin_s'] is None:
        basic.update(passed=False,reason='No eligible steady research candidate to replay')
    else:
        import gzip
        with gzip.open(output/experiment['selected_result_path'],'rt',encoding='utf-8') as stream:saved=json.load(stream)
        data=load_data(repo,api);sim=make_sim(api,data,experiment['selected_margin_s'])
        policy=api['load_policy'](checkpoint,expected_features=sim.features,expected_action_space='mode_aware')
        replay=sim.simulate(policy,initial_speed_mps=saved['summary']['speed_start_mps'],
            initial_energy_mj=saved['summary']['energy_start_mj'],
            carried_rule_state=saved['summary']['initial_rule_state'],action_space='mode_aware')
        api['label_elapsed_time'](replay)
        replay['summary']['physical_lap_index']=saved['summary']['physical_lap_index']
        proxy=api['proxy_for_r7'](sim,replay)
        saved_proxy=read_json(output/experiment['selected_proxy_path'])
        calibration=api['score'](replay,data,'calibration')
        development=api['score'](replay,data,'development')
        saved_cal=read_json(output/experiment['selected_calibration_path'])
        saved_dev=read_json(output/'development_score.json')
        trace_exact=all(np.array_equal(np.asarray(replay['internal_trace'][k]),np.asarray(v),equal_nan=True)
            for k,v in saved['internal_trace'].items())
        basic.update(exact_selected_trace=trace_exact,
            exact_selected_terminal=api['_jsonable'](replay['terminal_state'])==saved['terminal_state'],
            exact_selected_proxy=api['_jsonable'](proxy)==saved_proxy,
            exact_selected_calibration=api['_jsonable'](calibration)==saved_cal,
            exact_selected_development=api['_jsonable'](development)==saved_dev,
            exact_selected_actions=np.array_equal(replay['requested_actions'],np.asarray(saved['requested_actions'])),
            saved_development_reproduction_only=True,selected_margin_unchanged=True,
            original_subset_passed=replay['summary']['feasible'],
            stronger_steady_proxy_passed=proxy['additional_surrogate_hold_and_limited_request_gate_passed'])
        basic['passed']=all(basic[k] for k in ('exact_selected_trace','exact_selected_terminal',
            'exact_selected_proxy','exact_selected_calibration','exact_selected_development','exact_selected_actions',
            'original_subset_passed','stronger_steady_proxy_passed'))
    if write_result:api['_write'](output/'verification.json',basic)
    return basic


def run(repo,prior,output,config_path,api):
    if output.exists():raise ValueError('R7 output already exists; no overwrite or new scoring opportunity')
    if output==prior or prior.is_relative_to(output) or not output.is_relative_to(repo/'results'):
        raise ValueError('Use a new ignored results directory, never overwrite the prior experiment')
    config=read_json(config_path);validate_config(config)
    prior_manifest=read_json(prior/'manifest.json');validate_inventory(prior,prior_manifest)
    prior_started=read_json(prior/'started.json')
    if prior_started['provenance']['code_commit']!=config['prior_code_commit']:
        raise ValueError('Wrong frozen prior code commit')
    freeze=api['provenance'](repo,require_clean=True)
    script=Path(__file__).resolve()
    if not script.is_relative_to(repo):raise ValueError('Commit the R7 runner inside the repository first')
    for path in (script,config_path):
        relative=path.relative_to(repo).as_posix()
        subprocess.run(['git','ls-files','--error-unmatch',relative],cwd=repo,check=True,capture_output=True)
        freeze['source_sha256'][relative]=digest(path)
    checkpoint=safe_member(prior,config['policy_relative_to_prior_run'])
    metadata_path=safe_member(prior,config['policy_metadata_relative_to_prior_run'])
    metadata=read_json(metadata_path)
    initial_speed=float(metadata['raw_summary']['speed_start_mps'])
    freeze.update(schema_version=7,code_commit=freeze['code_commit'],
        config_path=config_path.relative_to(repo).as_posix(),config_sha256=digest(config_path),
        dependencies=dependencies(),initial_speed_mps=initial_speed,
        captured_at_utc=datetime.now(timezone.utc).isoformat())
    checkpoint_inventory={'checkpoint_relative_to_prior_run':config['policy_relative_to_prior_run'],
        'checkpoint_sha256':digest(checkpoint),'copied_checkpoint_relative':'inputs/raw_bc_policy.npz',
        'metadata_relative_to_prior_run':config['policy_metadata_relative_to_prior_run'],
        'metadata_sha256':digest(metadata_path),'prior_manifest_sha256':digest(prior/'manifest.json'),
        'prior_code_commit':config['prior_code_commit']}
    data=load_data(repo,api)
    output.mkdir(parents=True);(output/'inputs').mkdir()
    shutil.copyfile(checkpoint,output/'inputs/raw_bc_policy.npz')
    shutil.copyfile(config_path,output/'frozen_protocol.json')
    api['_write'](output/'freeze.json',freeze)
    candidates=[]
    for margin in config['margins_s']:
        started=time.perf_counter();sim=make_sim(api,data,margin)
        policy=api['load_policy'](checkpoint,expected_features=sim.features,expected_action_space='mode_aware')
        physical=sim.simulate_laps(policy,n_laps=6,initial_speed_mps=initial_speed,
            initial_energy_mj=2.,action_space='mode_aware')
        directory=f'margin_{margin:g}';summaries=[];proxies=[]
        for index,result in enumerate(physical['laps']):
            api['label_elapsed_time'](result)
            stem=f'{directory}/lap_{index+1}'
            api['_write'](output/(stem+'.json.gz'),result)
            proxy=api['proxy_for_r7'](sim,result)
            api['_write'](output/(stem+'_proxy.json'),proxy);proxies.append(proxy)
            calibration=api['score'](result,data,'calibration')
            api['_write'](output/(stem+'_calibration.json'),calibration)
            summaries.append(result['summary'])
        all_subset=len(summaries)==6 and physical['physical_control_line_continuity'] and all(s['feasible'] for s in summaries)
        cold_strong=all_subset and all(p['additional_surrogate_hold_and_limited_request_gate_passed'] for p in proxies)
        steady=all_subset and all(p['additional_surrogate_hold_and_limited_request_gate_passed'] for p in proxies[-2:])
        row={'margin_s':margin,'six_original_subset_passed':all_subset,
            'cold_strong_proxy_six_laps':cold_strong,'steady_strong_proxy_laps5_6':steady,
            'physical_control_line_continuity':physical['physical_control_line_continuity'],
            'crossings':physical['crossings'],'lap_summaries':summaries,
            'fullperiod_proxy_gates':[p['additional_surrogate_hold_and_limited_request_gate_passed'] for p in proxies],
            'cold_result_path':directory+'/lap_1.json.gz',
            'steady_result_paths':[directory+'/lap_5.json.gz',directory+'/lap_6.json.gz'] if len(summaries)==6 else [],
            'last_two_mean_time_s':sum(s['lap_time_s'] for s in summaries[-2:])/2. if len(summaries)==6 and all(s['completed_lap'] for s in summaries) else None,
            'result_directory':directory,'wall_seconds':time.perf_counter()-started}
        candidates.append(row)
        api['_write'](output/'candidate_summaries.json',candidates)
        print(json.dumps({'margin_s':margin,'six_subset':all_subset,'cold_strong':cold_strong,'steady_strong':steady,
            'steady_mean_time_s':row['last_two_mean_time_s']}),flush=True)
    eligible=[c for c in candidates if c['steady_strong_proxy_laps5_6']]
    winner=min(eligible,key=lambda c:c['last_two_mean_time_s']) if eligible else None
    selection={'schema_version':7,'selected_margin_s':winner['margin_s'] if winner else None,
        'selection_uses_observed_speed_residual':False,'development_seen_in_selection':False,
        'scope':'Steady research after six physical carried laps; cold strong qualification is separate',
        'ranking':'eligible last-two mean physical lap time','source_commit':freeze['code_commit']}
    api['_write'](output/'selection.json',selection)
    gates={'original_subset_six_laps':False,'cold_strong_proxy_six_laps':False,
        'steady_strong_proxy_laps5_6':False,'calibration_accuracy':False,'development_accuracy':False,
        'numerical_convergence':False,'new_session_test':False,'formal_release':False}
    experiment={'schema_version':7,'kind':'independent_geometry_guard_r7','source_commit':freeze['code_commit'],
        'provenance':freeze,'config_path':freeze['config_path'],'config_sha256':freeze['config_sha256'],
        'frozen_protocol_path':'frozen_protocol.json','checkpoint_inventory':checkpoint_inventory,
        'selected_margin_s':selection['selected_margin_s'],'candidate_summaries':candidates,
        'selected_result_path':None,'selected_proxy_path':None,'selected_calibration_path':None,
        'gates':gates,'new_SAC_episodes':0,'FP2_accessed':False,'new_session_status':'验收待完成',
        'FIA_violation_proven':False,'full_FIA_ECU_legality_proven':False,'global_optimality_proven':False,
        'selection_qualification':'controlled_warmup_steady_research_candidate_only',
        'actual_K_constant_FIA_requirement_claimed':False,
        'actual_supply_and_FIA_maximum_driver_demand_are_distinct':True,
        'old_frozen_experiment_files_modified':False}
    if winner:
        import gzip
        stem=winner['result_directory']+'/lap_6'
        with gzip.open(output/(stem+'.json.gz'),'rt',encoding='utf-8') as stream:selected=json.load(stream)
        experiment.update(selected_result_path=stem+'.json.gz',selected_proxy_path=stem+'_proxy.json',
            selected_calibration_path=stem+'_calibration.json')
        # Selection is immutable before the single final development score.
        with (output/'development_claim.json').open('x',encoding='utf-8') as stream:
            json.dump({'selection_sha256':digest(output/'selection.json'),'laps':[7,9],'once_only':True},stream)
        development=api['score'](selected,data,'development')
        api['_write'](output/'development_score.json',development)
        numerical=api['numerical_audit_r7'](make_sim(api,data,winner['margin_s']),selected,output/'numerical')
        api['_write'](output/'numerical_audit.json',numerical)
        gates.update(original_subset_six_laps=winner['six_original_subset_passed'],
            cold_strong_proxy_six_laps=winner['cold_strong_proxy_six_laps'],
            steady_strong_proxy_laps5_6=winner['steady_strong_proxy_laps5_6'],
            calibration_accuracy=read_json(output/(stem+'_calibration.json'))['accuracy_passed'],
            development_accuracy=development['accuracy_passed'],numerical_convergence=numerical['passed'])
    experiment['frozen_inputs_unchanged']=frozen_inputs_unchanged(repo,freeze)
    if not experiment['frozen_inputs_unchanged']:raise ValueError('R7 frozen input changed during execution')
    api['_write'](output/'experiment.json',experiment)
    def inventory():
        return {'schema_version':7,'source_commit':freeze['code_commit'],'config_path':freeze['config_path'],
            'config_sha256':freeze['config_sha256'],'checkpoint_inventory':checkpoint_inventory,
            'files':{p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*'))
                if p.is_file() and p!=output/'manifest.json'}}
    # The provisional inventory is verified before replay; the final inventory
    # additionally covers the saved verification receipt, never itself.
    api['_write'](output/'manifest.json',inventory())
    verification=verify(repo,output,api)
    manifest=inventory()
    api['_write'](output/'manifest.json',manifest)
    print(json.dumps({'output':str(output),'selected_margin_s':selection['selected_margin_s'],
        'verification_passed':verification['passed'],'gates':gates}),flush=True)
    return experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--prior-run',type=Path,default=Path('results/austria_2026_v6'))
    parser.add_argument('--output',type=Path,default=Path('results/austria_2026_v6/R7'))
    parser.add_argument('--config',type=Path,default=Path('config/v6_r7/protocol.json'))
    parser.add_argument('--verify',type=Path)
    args=parser.parse_args();repo=args.repo.resolve()
    api=imports(repo)
    def rooted(path):return (repo/path).resolve() if not path.is_absolute() else path.resolve()
    if args.verify:
        result=verify(repo,rooted(args.verify),api,write_result=False)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['passed'] else 1
    run(repo,rooted(args.prior_run),rooted(args.output),rooted(args.config),api)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
