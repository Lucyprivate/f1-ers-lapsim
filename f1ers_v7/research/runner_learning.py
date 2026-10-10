"""Finite learning orchestration over the frozen qualified ideal-energy model."""
from pathlib import Path
import numpy as np

from ..learning import save_policy
from .._core.learning_base import _write
from .._core.provenance import read,digest
from .._core.metrics import score
from .data import load_historical
from .protocol import EpisodeBudget
from .teacher import qualify
from .backends import differential_audit,make_backend
from .learning import fit_bc,train_case,rollout,_warmstart,frozen_policy_audit,physical_noop_audit


def _baseline_identity(laps,output):
    checks=[]
    for i,lap in enumerate(laps):
        path=Path(output)/f'baseline/lap_{i+1}.json.gz'
        reference=read(path)
        action_equal=np.array_equal(lap['requested_actions'],reference['requested_actions'])
        profile_equal=all(np.array_equal(lap['driver_profile'][k],reference['driver_profile'][k])
                          for k in ('s_m','wheel_force_n'))
        boundary_equal=all(lap['summary'].get(k)==reference['summary'].get(k) for k in
            ('speed_start_mps','energy_start_mj','initial_dc_kw','initial_rule_state'))
        checks.append({'lap':i+1,'raw_actions_equal':action_equal,'raw_driver_profile_equal':profile_equal,
            'entire_initial_boundary_equal':boundary_equal,'baseline_lap_sha256':digest(path)})
    return {'passed':len(checks)==6 and all(r['raw_actions_equal'] and r['raw_driver_profile_equal']
            and r['entire_initial_boundary_equal'] for r in checks),'lap_identity_checks':checks}


def run_learning(root,data_root,output,ledger_root,config,backend='both'):
    """No source edits, hyperparameter selection, FP2 reads or policy speed shooting."""
    from .experiment import make_sim,numerical_coverage
    root,output=Path(root),Path(output)
    if backend not in ('both','numpy','sb3'): raise ValueError('Unknown research backend')
    folder=output/'learning';folder.mkdir(parents=True,exist_ok=False)
    budget=EpisodeBudget(ledger_root)
    summary={'status':'started','backend_request':backend,'maximum_new_training_episodes':120,
        'maximum_per_backend':60,'case_budget':10,'seeds':[2026,2027,2028],
        'conditions':['cold','warm'],'selected_energy_model':'unchanged ideal DC-equivalent model',
        'real_car_gain_claimed':False,'FP2_accessed':False,'dagger_enabled':False,
        'same_state_expert_available':False,'candidates':[],'cases':[],
        'all_intermediate_checkpoints_retained':True,'all_intermediate_six_carry_qualified':False,
        'best_last_and_BC_six_carry_qualified':True,'policy_specific_periodic_shooting':False}
    def persist():
        summary['episode_budget_counts']=budget.counts()
        _write(output/'learning.json',summary)
    persist()
    teacher=read(output/'teacher.json')
    if not teacher.get('learning_allowed'):
        summary.update(status='withheld_teacher_or_physics_gate',reason=teacher.get('status'))
        persist();return summary
    if not read(output/'baseline/numerics.json').get('passed'):
        summary.update(status='withheld_numerical_gate');persist();return summary
    cal=load_historical(data_root,root,role='calibration')
    dev=load_historical(data_root,root,role='development')
    sim=make_sim(cal,config)
    pool=read(output/'boundary_pool.json')
    teachers=[read(output/f'bc_teacher/lap_{i}.json.gz') for i in range(1,7)]
    try:
        differential=differential_audit(folder/'differential')
    except ImportError as exc:
        summary.update(status='withheld_optional_dependency_missing',error=str(exc))
        persist();return summary
    if not differential['passed']:
        summary.update(status='withheld_differential_gate',differential=differential)
        persist();return summary
    summary['differential_passed']=True
    policies={}
    for seed in (2026,2027,2028):
        bc=fit_bc(teachers,seed=seed)
        policies[seed]=bc['policy'];bc_out=folder/f'bc_seed_{seed}'
        save_policy(bc['policy'],bc_out/'policy.npz')
        _write(bc_out/'metadata.json',bc['metadata']);_write(bc_out/'training_curve.json',bc['curve'])
        certificate,laps=qualify(sim,bc['policy'],bc_out/'qualification',pool['cold']['speed_mps'],
            initial_energy=pool['cold']['energy_mj'],initial_rule=pool['cold']['carried_rule_state'])
        if laps:
            _write(bc_out/'score_calibration.json',score(laps[-1],cal,'calibration'))
            _write(bc_out/'score_development.json',score(laps[-1],dev,'development'))
        identity=_baseline_identity(laps,output) if certificate['passed'] and bc['metadata']['analytical_zero_mean_clone'] else {'passed':False}
        if certificate['passed'] and identity['passed']:
            numerical={**read(output/'baseline/numerics.json'),
                'reused_qualified_baseline_with_exact_execution_identity':True,'execution_identity':identity}
        elif certificate['passed']:
            numerical=numerical_coverage(sim,laps,bc_out/'numerics')
        else: numerical={'passed':False,'reason':'BC physical qualification failed'}
        _write(bc_out/'numerics.json',numerical)
        summary['candidates'].append({'kind':'BC','seed':seed,'path':str(bc_out/'policy.npz'),
            'qualification_path':str(bc_out/'qualification'),
            'qualification':certificate,'analytical_zero_mean_clone':bc['metadata']['analytical_zero_mean_clone'],
            'numerical_passed':bool(numerical['passed']),'numerical_reports':numerical,
            'learned_gain_claimed':False})
        persist()
        if not certificate['passed'] or not numerical['passed']:
            summary.update(status='withheld_BC_physical_or_numerical_gate',failed_seed=seed)
            persist();return summary
    # This physical proof precedes any masked actor training; synthetic Q-flatness
    # alone cannot establish that a stateful executor ignores raw brake actions.
    noop=physical_noop_audit(sim,policies[2026],pool['cold'],folder/'physical_noop')
    if not noop['passed']:
        summary.update(status='withheld_brake_noop_gate');persist();return summary
    audit_backend=make_backend('numpy',policies[2026],2026)
    reference=rollout(sim,policies[2026],pool['cold'])
    _write(folder/'frozen_Q/reference.json.gz',reference)
    _write(folder/'frozen_Q/warmup.json',_warmstart(audit_backend,reference,2026))
    audit=frozen_policy_audit(sim,audit_backend,pool['cold'],folder/'frozen_Q')
    summary['frozen_policy_audit']=audit
    summary['frozen_policy_audit_scope']='One preregistered BC policy, warmstarted critics; 60 diagnostic suffixes'
    if not audit['passed']:
        summary.update(status='withheld_frozen_physical_or_numerical_gate');persist();return summary
    stop=False
    for name in (('numpy','sb3') if backend=='both' else (backend,)):
        for seed in (2026,2027,2028):
            for condition in ('cold','warm'):
                case_out=folder/f'{name}_seed_{seed}_{condition}'
                result=train_case(sim,policies[seed],case_out,backend=name,seed=seed,
                    condition=condition,boundary_pool=[pool[condition]],budget=budget,
                    evaluation_boundary=pool[condition],episodes=10)
                summary['cases'].append({'backend':name,'seed':seed,'condition':condition,
                    'metadata':result['metadata']})
                persist()
                for kind in ('best','last'):
                    policy=result[kind+'_policy'];qual_out=case_out/f'qualification_{kind}'
                    try:
                        certificate,laps=qualify(sim,policy,qual_out,pool['cold']['speed_mps'],
                            initial_energy=pool['cold']['energy_mj'],initial_rule=pool['cold']['carried_rule_state'])
                        if laps:
                            _write(qual_out/'score_calibration.json',score(laps[-1],cal,'calibration'))
                            _write(qual_out/'score_development.json',score(laps[-1],dev,'development'))
                    except Exception as exc:
                        certificate={'passed':False,'formal_objective_s':None,
                                     'error':f'{type(exc).__name__}: {exc}'}
                        _write(qual_out/'qualification.json',certificate)
                    summary['candidates'].append({'kind':kind,'backend':name,'seed':seed,'condition':condition,
                        'path':str(case_out/f'{kind}_policy.npz'),'qualification':certificate,
                        'qualification_path':str(qual_out),
                        'best_episode':result['metadata']['best_episode'],
                        'single_boundary_gain':result['metadata']['sac_gain_demonstrated'],
                        'learned_gain_claimed':False})
                    persist()
                if result['metadata']['stop_all_training']:
                    stop=True;summary.update(status='stopped_physical_numerical_or_learning_exception',
                        stopping_case={'backend':name,'seed':seed,'condition':condition})
                    break
            if stop: break
        if stop: break
    legal=[c for c in summary['candidates'] if c['qualification'].get('passed')]
    incumbent=min((c['qualification']['formal_objective_s'] for c in legal
                   if c['kind']=='BC' and c.get('numerical_passed')),default=np.inf)
    eligible=[c for c in legal if c['kind']=='BC' or
              c['qualification']['formal_objective_s']<incumbent-.10]
    winner=None
    # Numerical convergence is action- and boundary-dependent. Teacher tests
    # cannot certify a learned actor: audit each finalist and re-rank on failure.
    for index,candidate in enumerate(sorted(eligible,key=lambda c:c['qualification']['formal_objective_s'])):
        if candidate.get('numerical_passed'):
            winner=candidate;break
        base=Path(candidate['qualification_path'])
        numerical={}
        try:
            laps=[read(base/f'lap_{lap_index}.json.gz') for lap_index in (1,6)]
            numerical=numerical_coverage(sim,laps,folder/f'finalist_{index}_numerics')
        except Exception as exc:
            numerical={'passed':False,'error':f'{type(exc).__name__}: {exc}'}
        candidate['numerical_passed']=bool(numerical.get('passed'))
        candidate['numerical_reports']=numerical
        persist()
        if candidate['numerical_passed']:
            winner=candidate;break
    gain=incumbent-winner['qualification']['formal_objective_s'] if winner else None
    summary.update(winner=winner,model_six_carry_gain_s=gain,
        gain_threshold_s=.10,
        sac_six_carry_gain_demonstrated=bool(winner and winner['kind']!='BC' and gain>.10),
        global_optimality_proven=False,real_car_gain_claimed=False,FP2_accessed=False)
    if not stop: summary['status']='completed_in_model_research' if winner else 'no_numerically_qualified_finalist'
    persist()
    return summary
