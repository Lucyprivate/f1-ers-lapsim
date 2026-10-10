"""Durable bounded learning and frozen-policy conditional-return diagnostics."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
import csv
import hashlib
import numpy as np

from ..learning import _bind, save_policy, monte_carlo_returns
from .._core.networks import GaussianPolicy, ReplayBuffer, clone_behaviour
from .._core.reward import RewardConfig
from .._core.learning_base import _write, is_legal
from .environment import FEATURES, PROTOCOL, validate_boundary, require_policy_contract
from .backends import BackendConfig, actor_active, make_backend


def fit_bc(teacher_traces, *, seed, epochs=600):
    """True supervised cloning; exact constant zero heads are an analytical prior."""
    if seed not in (2026,2027,2028) or not 1<=epochs<=600:
        raise ValueError('Fixed three seeds and at most 600 BC epochs required')
    obs,labels=[],[]
    for trace in teacher_traces:
        result=trace.get('result',trace)
        summary=result['summary']
        if summary.get('protocol')!=PROTOCOL or tuple(summary.get('feature_schema',()))!=FEATURES or not is_legal(result):
            raise ValueError('BC requires complete legal teacher executions bound to 36 features')
        x=np.asarray(result['observations'],float)
        y=np.asarray(result['requested_actions'],float).reshape(-1)
        if x.shape!=(len(y),len(FEATURES)) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid teacher observations and raw action labels')
        obs.append(x);labels.append(y)
    if not obs: raise ValueError('No teacher data supplied')
    x,y=np.vstack(obs),np.concatenate(labels)
    analytical=bool(np.all(y==0))
    initial=None
    if analytical:
        initial=GaussianPolicy(len(FEATURES),x.mean(0),np.maximum(x.std(0),.05),(32,32),seed)
        initial.network.weights[-1][:,0]=0.;initial.network.biases[-1][0]=0.
    policy,curve=clone_behaviour(x,y,seed=seed,epochs=epochs,initial_policy=initial)
    _bind(policy,FEATURES,'mode_aware',PROTOCOL)
    if analytical and (np.any(policy.network.weights[-1][:,0]!=0) or policy.network.biases[-1][0]!=0):
        raise AssertionError('The analytical zero mean head moved')
    return {'policy':policy,'curve':curve,'metadata':{'seed':seed,'epochs':epochs,
        'analytical_zero_mean_clone':analytical,'analytical_clone_counted_as_learning_gain':False,
        'teacher_labels_are_recorded_raw_actions':True,'brake_mapping_is_fixed_by_executor':True,
        'std_regularizer':'logstd toward -2.5, not empirical variance identification',
        'dagger_enabled':False,'same_state_expert_available':False}}


def _new_env(sim,boundary):
    b=validate_boundary(boundary)
    return sim.create_env(initial_speed_mps=b['speed_mps'],initial_energy_mj=b['energy_mj'],
        carried_rule_state=b['carried_rule_state'],reward_config=RewardConfig(energy_start_mj=b['energy_mj']))


def _empty_trace():
    return {key:[] for key in ('observations','next_observations','actions','rewards','dones',
                              'log_probabilities','actor_active','terminated','truncated','infos','losses')}


def _finish_trace(rows,env,error=None):
    value={k:np.asarray(v) if k not in ('infos','losses') else v for k,v in rows.items()}
    try: value['result']=env.result()
    except Exception as exc: value['result_error']=f'{type(exc).__name__}: {exc}'
    value.update(features=FEATURES,protocol=PROTOCOL,error=error,
                 accepted_trajectory_invented=False,within_lap_state_resets=0)
    return value


def rollout(sim,policy,boundary,*,rng=None,backend=None,replay=None,config=None):
    """Retain failed prefixes; numerical truncations are excluded from replay."""
    require_policy_contract(policy)
    cfg=config or BackendConfig();env=_new_env(sim,boundary);rows=_empty_trace()
    max_steps=int(getattr(sim,'n_decisions',np.ceil(sim.length_m/5.)))+2
    obs=np.asarray(env.get_obs(),float)
    error=None
    for index in range(max_steps):
        try:
            if rng is None:
                action=float(policy(obs));_,lp,_=policy.sample(obs,np.random.default_rng(0),epsilon=np.zeros((1,1)))
            else:
                a,lp,_=backend.sample(obs,rng) if backend else policy.sample(obs,rng)
                action=float(a[0,0])
            nxt,reward,done,info=env.step(action)
            nxt=np.asarray(nxt,float)
            numeric=bool(done and info.get('status')=='numerical_fail')
            if obs.shape!=(len(FEATURES),) or nxt.shape!=obs.shape or not np.isfinite(nxt).all() or not np.isfinite(reward):
                raise FloatingPointError('Nonfinite or incompatible accepted transition')
            active=bool(actor_active(obs)[0,0])
            info={**info,'actor_active_pre_transition':active,'raw_requested_action':action,
                  'trainable_transition':not numeric,'bootstrap_allowed':not done}
            values=(obs.copy(),nxt.copy(),action,float(reward),bool(done),float(lp[0,0]),
                    active,bool(done and not numeric),numeric,info)
            for key,v in zip(('observations','next_observations','actions','rewards','dones','log_probabilities',
                              'actor_active','terminated','truncated','infos'),values): rows[key].append(v)
            if replay is not None and not numeric: replay.add(obs,action,float(reward),nxt,done)
            if (backend is not None and replay is not None and not numeric
                    and not (done and info.get('status')!='accepted')
                    and replay.size>=cfg.batch_size and (index+1)%cfg.update_every==0):
                rows['losses'].append(backend.update(replay.sample(cfg.batch_size,rng)))
            obs=nxt
            if done: break
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}';break
    if not rows['dones'] or not rows['dones'][-1]:
        error=error or 'Administrative incomplete prefix: executor did not reach an absorbing boundary'
    return _finish_trace(rows,env,error)


def _hard_stop(trace):
    if trace.get('error'): return True
    summary=trace.get('result',{}).get('summary',{})
    return (not bool(summary.get('completed_lap')) or not bool(summary.get('execution_feasible'))
            or not bool(summary.get('public_rules_passed')))


def _objective(trace):
    result=trace.get('result',{})
    return float(result['summary']['lap_time_s']) if is_legal(result) else np.inf


def _warmstart(backend,trace,seed):
    if _hard_stop(trace): raise ValueError('Critic initialization requires a complete physical reference')
    obs=np.asarray(trace['observations']);actions=np.asarray(trace['actions']).reshape(-1,1)
    masked_lp=np.asarray(trace['log_probabilities'])*np.asarray(trace['actor_active'])
    labels=monte_carlo_returns(trace['rewards'],trace['dones'],log_probabilities=masked_lp,
                             alpha=backend.alpha)/backend.return_scale_s
    rng=np.random.default_rng(seed+32452843);history=[]
    for _ in range(backend.config.critic_warmup):
        ix=rng.integers(0,len(obs),size=backend.config.batch_size)
        history.append(backend.supervised_critic_update(obs[ix],actions[ix],labels[ix,None]))
    backend.synchronize_targets()
    return {'iterations':backend.config.critic_warmup,'actor_moved':False,
        'masked_future_entropy':True,'history':history,
        'method':'Realized deterministic-mean reference soft-return regression',
        'Gaussian_policy_value_identified':False,
        'coverage_limitation':'This narrow initialization is not a frozen stochastic conditional-value test'}


def train_case(simulator,bc_policy,output_dir,*,backend,seed,condition,boundary_pool,budget,
               evaluation_boundary=None,config=None,episodes=10):
    """Spend exclusive episode claims before reset; never resume an interrupted case."""
    cfg=(config or BackendConfig()).validate()
    if seed not in (2026,2027,2028) or condition not in ('cold','warm') or not 1<=episodes<=10:
        raise ValueError('Preregistered three-seed cold/warm budget: at most ten episodes per case')
    require_policy_contract(bc_policy)
    pool=[validate_boundary(v) for v in boundary_pool]
    if not pool: raise ValueError('Approved initial boundary pool is empty')
    eval_b=validate_boundary(evaluation_boundary or pool[-1])
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    model=make_backend(backend,bc_policy,seed,cfg)
    rows=[];retained=[];best=last=bc_policy;best_trace=last_trace=None;best_episode=0
    counters={'requested':episodes,'attempted':0,'completed':0,'physical_full_laps':0,'feasible_full_laps':0}
    metadata={'backend':backend,'seed':seed,'condition':condition,'configuration':asdict(cfg),
        'dependency_evidence':model.dependency_evidence(),'training_boundaries':pool,'evaluation_boundary':eval_b,
        'policy_specific_periodic_shooting':False,'best_is_single_declared_boundary_only':True,
        'every_episode_checkpoint_saved':True,'raw_actions_retained':True,
        'physical_or_numerical_failure_stops_all_cases':True,'counters':counters,'history':rows,
        'sac_gain_demonstrated':False,'best_episode':0,'stop_all_training':False}
    metadata['sac_gain_scope']='Only the fixed declared single-boundary evaluation; six-carry gain is external'
    def persist():
        _write(output/'metadata.json',metadata)
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['episode','attempt_identity','full_lap','feasible','lap_time_s','error'])
            writer.writeheader();writer.writerows(rows)
    persist()
    try:
        initial=rollout(simulator,bc_policy,eval_b)
        _write(output/'reference_episode0.json.gz',initial)
        best_trace=last_trace=initial
        model.save(output/'episode_000_policy.npz')
        retained.append({'episode':0,'path':str(output/'episode_000_policy.npz'),'analytical_or_BC_incumbent':True})
        if _hard_stop(initial):
            metadata.update(stop_all_training=True,reason='Reference physical/numerical failure before new training')
            persist()
            return {'metadata':metadata,'retained_checkpoints':retained,'best_policy':best,'last_policy':last,
                    'best_result':initial.get('result'),'last_result':initial.get('result'),'backend_object':model}
        metadata['critic_warmstart']=_warmstart(model,initial,seed)
        model.save(output/'warmstarted_policy.npz')
        replay=ReplayBuffer(100000,len(FEATURES));rng=np.random.default_rng(seed)
        best_score=_objective(initial)
        for episode in range(1,episodes+1):
            identity=budget.claim(backend,seed,condition,episode)
            counters['attempted']+=1
            _write(output/f'episode_{episode:03d}_started.json',{'identity':identity,'claimed_before_reset':True,
                'episode':episode,'backend':backend,'seed':seed,'condition':condition})
            persist()
            # The claim remains consumed even if reset, a native operation or serialization fails.
            try:
                trace=rollout(simulator,model.export_numpy(),pool[(episode-1)%len(pool)],
                              rng=rng,backend=model,replay=replay,config=cfg)
            except Exception as exc:
                trace={**_empty_trace(),'error':f'{type(exc).__name__}: {exc}',
                    'failure_before_first_accepted_transition':True,'accepted_trajectory_invented':False}
            _write(output/f'episode_{episode:03d}_training.json.gz',trace)
            counters['completed']+=1
            summary=trace.get('result',{}).get('summary',{})
            full=bool(summary.get('completed_lap') and summary.get('execution_feasible'))
            counters['physical_full_laps']+=int(full)
            counters['feasible_full_laps']+=int(is_legal(trace.get('result',{})))
            row={'episode':episode,'attempt_identity':identity,'full_lap':full,
                'feasible':bool(is_legal(trace.get('result',{}))),
                'lap_time_s':summary.get('lap_time_s'),'error':trace.get('error')}
            rows.append(row)
            path=output/f'episode_{episode:03d}_policy.npz';model.save(path)
            retained.append({'episode':episode,'path':str(path),'analytical_or_BC_incumbent':False})
            last=model.export_numpy();last_trace=trace;persist()
            if _hard_stop(trace):
                metadata.update(stop_all_training=True,reason='Physical/numerical/public-rule failure',
                                failure_episode=episode)
                break
            # Evaluate every retained actor at the preregistered fixed full boundary.
            evaluated=rollout(simulator,last,eval_b)
            _write(output/f'episode_{episode:03d}_evaluation.json.gz',evaluated)
            if _hard_stop(evaluated):
                metadata.update(stop_all_training=True,reason='Fixed-policy evaluation physical/numerical failure',
                                failure_episode=episode)
                break
            score=_objective(evaluated)
            if score<best_score-1e-8:
                best,best_trace,best_score,best_episode=last,evaluated,score,episode
            metadata.update(best_episode=best_episode,sac_gain_demonstrated=best_episode>0)
            persist()
    except Exception as exc:
        metadata.update(stop_all_training=True,reason='Case exception',error=f'{type(exc).__name__}: {exc}')
    finally:
        metadata.update(best_episode=best_episode,sac_gain_demonstrated=best_episode>0,
            retained_checkpoints=retained,completed_means_retained_attempt_not_successful_lap=True)
        save_policy(best,output/'best_policy.npz');save_policy(last,output/'last_policy.npz');persist()
    return {'metadata':metadata,'retained_checkpoints':retained,'best_policy':best,'last_policy':last,
            'best_result':best_trace.get('result') if best_trace else None,
            'last_result':last_trace.get('result') if last_trace else None,'backend_object':model}


def _parameter_hash(backend):
    h=hashlib.sha256()
    for name,value in sorted(backend.parameter_arrays().items()):
        h.update(name.encode());h.update(np.asarray(value,dtype=np.float64).tobytes())
    return h.hexdigest()


def physical_noop_audit(simulator,policy,boundary,output_dir=None):
    """Check actual accepted state, FSM and electrical ledger, not neural algebra."""
    require_policy_contract(policy)
    env=_new_env(simulator,boundary);snapshot=None
    n=int(getattr(simulator,'n_decisions',np.ceil(simulator.length_m/5.)))
    for index in range(n+2):
        if not bool(actor_active(env.get_obs())[0,0]):
            snapshot=deepcopy(env);break
        _,_,done,_=env.step(policy(env.get_obs()))
        if done: break
    if snapshot is None:
        report={'passed':False,'reason':'No genuine negative-wheel demand in frozen reference',
                'engineering_noop_verified':False,'new_training_episodes':0}
    else:
        variants=[]
        for raw in (-1.,0.,1.):
            fork=deepcopy(snapshot)
            obs,reward,done,info=fork.step(raw)
            result=fork.result()
            variants.append({'raw_action':raw,'next_observation':np.asarray(obs).tolist(),
                'integration_state':asdict(fork.state) if is_dataclass(fork.state) else deepcopy(fork.state),
                'terminal_rule_state':result['rule_certificate']['terminal_boundary_state'],
                'internal_trace':result['internal_trace'],'reward':reward,'done':done,
                'physical_status':info.get('status')})
        first=variants[0]
        fields=('s_m','time_s','speed_mps','energy_mj','dc_kw','wheel_force_n')
        equal=[]
        for v in variants[1:]:
            ledger=all(np.array_equal(np.asarray(v['internal_trace'].get(k,[])),
                                     np.asarray(first['internal_trace'].get(k,[]))) for k in fields)
            equal.append(bool(v['integration_state']==first['integration_state'] and
                v['terminal_rule_state']==first['terminal_rule_state'] and ledger and
                np.array_equal(v['next_observation'],first['next_observation']) and
                v['reward']==first['reward'] and v['done']==first['done']))
        report={'passed':all(equal) and all(v['physical_status']=='accepted' for v in variants),
            'engineering_noop_verified':all(equal),'decision':index,
            'signed_wheel_demand_feature':float(snapshot.get_obs()[10]),'variants':variants,
            'complete_state_and_FSM_compared':True,'actual_electrical_ledger_compared':True,
            'raw_action_is_still_recorded':True,'new_training_episodes':0}
    if output_dir is not None: _write(Path(output_dir)/'physical_noop_audit.json',report)
    return report


def frozen_policy_audit(simulator,backend,boundary,output_dir=None,*,snapshot_indices=None,
                        first_actions=(-.5,0.,.5),continuations=4,maximum_suffixes=60,seed=2026):
    """Paired frozen stochastic suffix returns at exact carried physical/FSM states.

    Diagnostic forks do not train or select a model. The sampled first action
    has no entropy term in Q(s,a); only subsequent active actions contribute.
    Brake no-ops are reported as unidentifiable, not ranked as genuine controls.
    """
    if continuations!=4 or tuple(first_actions)!=(-.5,0.,.5) or maximum_suffixes!=60:
        raise ValueError('Five snapshots, three first actions and four continuations: cap60')
    policy=backend.export_numpy();before=_parameter_hash(backend)
    env=_new_env(simulator,boundary);snapshots=[]
    n=int(getattr(simulator,'n_decisions',np.ceil(simulator.length_m/5.)))
    indices=list(snapshot_indices) if snapshot_indices is not None else np.linspace(0,max(0,n-2),5,dtype=int).tolist()
    if (len(indices)!=5 or len(set(indices))!=5 or
            any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<n for i in indices)):
        raise ValueError('Five distinct preregistered valid decision snapshots required')
    for i in range(n+2):
        if i in indices: snapshots.append((i,deepcopy(env)))
        _,_,done,info=env.step(policy(env.get_obs()))
        if done: break
    reference=env.result()
    rows=[];hard_failure=False;spent=0
    for index,snapshot in snapshots:
        obs=snapshot.get_obs().copy();active=bool(actor_active(obs)[0,0])
        q=backend.q_values(np.tile(obs,(3,1)),np.asarray(first_actions)[:,None])
        samples=[[] for _ in first_actions]
        executions=[[] for _ in first_actions]
        for replicate in range(continuations):
            # Reuse a continuation epsilon stream for all three first actions.
            for ai,first in enumerate(first_actions):
                if spent>=maximum_suffixes: break
                spent+=1;fork=deepcopy(snapshot);rng=np.random.default_rng(seed+index*101+replicate)
                trace=_empty_trace();total=0.;error=None
                for j in range(n+2):
                    try:
                        o=fork.get_obs().copy()
                        if j==0: a,lp=float(first),0.
                        else:
                            act,lprob,_=policy.sample(o,rng);a,lp=float(act[0,0]),float(lprob[0,0])
                        next_o,reward,done,info=fork.step(a)
                        if j==0:
                            executions[ai].append({'raw_first_action':a,'mapped_first_action':info.get('mapped_action'),
                                'requested_dc_kw':info.get('requested_dc_kw'),
                                'executed_dc_kw':info.get('dc_kw'),
                                'projection_fraction':info.get('projection_fraction'),
                                'energy_delta_mj':float(fork.state.energy_mj-snapshot.state.energy_mj),
                                'distance_delta_m':float(fork.state.s_m-snapshot.state.s_m)})
                        m=bool(actor_active(o)[0,0]);total+=float(reward)-(backend.alpha*lp*m if j else 0.)
                        for key,v in zip(('observations','next_observations','actions','rewards','dones','log_probabilities',
                            'actor_active','terminated','truncated','infos'),
                            (o,next_o,a,reward,done,lp,m,done and info.get('status')!='numerical_fail',
                             done and info.get('status')=='numerical_fail',info)): trace[key].append(v)
                        if done: break
                    except Exception as exc: error=f'{type(exc).__name__}: {exc}';break
                value=_finish_trace(trace,fork,error)
                if not trace['dones'] or not trace['dones'][-1]: value['error']=error or 'Incomplete frozen suffix'
                failure=_hard_stop(value);hard_failure|=failure
                if output_dir is not None: _write(Path(output_dir)/f'snapshot_{index}_action_{ai}_repeat_{replicate}.json.gz',value)
                # Incomplete physical/numerical prefixes never become MC value targets.
                samples[ai].append(None if failure else total)
                if failure: break
            if hard_failure: break
        counts=np.asarray([sum(v is not None for v in values) for values in samples])
        means=np.asarray([np.mean([v for v in values if v is not None]) if count else np.nan
                          for values,count in zip(samples,counts)])
        stderr=np.asarray([np.std([v for v in values if v is not None],ddof=1)/np.sqrt(count)
                           if count>1 else np.nan for values,count in zip(samples,counts)])
        paired=[]
        for ai in (0,1):
            d=np.asarray([y-x for x,y in zip(samples[ai],samples[ai+1]) if x is not None and y is not None])
            se=float(np.std(d,ddof=1)/np.sqrt(len(d))) if len(d)>1 else np.nan
            paired.append({'actions':[first_actions[ai],first_actions[ai+1]],
                'valid_complete_pairs':len(d),'return_difference_s':float(d.mean()) if len(d) else None,
                'approximate95_interval_s':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)] if len(d)>1 else None})
        rows.append({'decision':index,'actor_active':active,'brake_action_unidentifiable':not active,
            'raw_and_executed_first_actions':executions,
            'q1_s':q[:,0].tolist(),'q2_s':q[:,1].tolist(),'conditional_soft_return_mean_s':means.tolist(),
            'approximate95_interval_s':np.c_[means-1.96*stderr,means+1.96*stderr].tolist(),
            'q_min_minus_conditional_mean_s':(q.min(1)-means).tolist(),'paired_common_random_numbers':paired,
            'complete_value_targets_per_action':counts.tolist(),'incomplete_returns_averaged':False,
            'q_rank':np.argsort(q.min(1)).tolist() if active and np.all(counts==continuations) else None,
            'conditional_rank':np.argsort(means).tolist() if active and np.all(counts==continuations) else None})
        if hard_failure: break
    after=_parameter_hash(backend)
    noop=physical_noop_audit(simulator,policy,boundary,output_dir)
    report={'passed':bool(before==after and not hard_failure and len(snapshots)==5 and spent==60 and noop['passed']),
        'frozen_parameters_before':before,'frozen_parameters_after':after,'policy_or_critic_updated':False,
        'conditional_suffixes':spent,'new_training_episodes':0,'snapshots':rows,
        'reference_physical_complete':bool(reference['summary'].get('completed_lap')),
        'physical_or_numerical_failure':hard_failure,'same_full_state_FSM_forks':True,
        'first_action_entropy_excluded':True,'future_brake_entropy_masked':True,
        'physical_brake_noop':{k:v for k,v in noop.items() if k!='variants'},
        'approximate_intervals_are_not_convergence_proof':True,'FP2_accessed':False}
    report['ranking_order']='Ascending soft return; larger return is preferred under seconds objective'
    if output_dir is not None: _write(Path(output_dir)/'frozen_policy_audit.json',report)
    return report
