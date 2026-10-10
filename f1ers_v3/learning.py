"""Gate-protected schema-3 BC and DAgger; no SAC experiment is run in v3."""
from pathlib import Path
import numpy as np
from f1ers.learning import GaussianPolicy, clone_behaviour
from .simulator import PROTOCOL

SCHEMA_VERSION=3
OBSERVATION_DIM=13
REQUIRED_GATES=('convergence','physical','grid','development','new_test')


def save_policy_v3(policy,path):
    if policy.observation_dim!=OBSERVATION_DIM:raise ValueError('v3 requires 13 observations')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(schema_version=np.asarray(SCHEMA_VERSION),protocol=np.asarray(PROTOCOL),
        observation_dim=np.asarray(OBSERVATION_DIM),sizes=np.asarray(policy.network.sizes),
        observation_mean=policy.mean,observation_scale=policy.scale)
    for i,(w,b) in enumerate(zip(policy.network.weights,policy.network.biases)):
        arrays[f'w{i}']=w;arrays[f'b{i}']=b
    np.savez_compressed(path,**arrays)


def load_policy_v3(path):
    with np.load(path,allow_pickle=False) as data:
        required={'schema_version','protocol','observation_dim','sizes','observation_mean','observation_scale'}
        if not required.issubset(data.files) or int(data['schema_version'])!=3 or str(data['protocol'])!=PROTOCOL or int(data['observation_dim'])!=13:
            raise ValueError('Checkpoint is not the schema-3 signed causal execution protocol')
        sizes=tuple(int(x) for x in data['sizes'])
        if len(sizes)<2 or sizes[0]!=13 or sizes[-1]!=2:raise ValueError('Invalid v3 policy shape')
        if not np.isfinite(data['observation_mean']).all() or not np.isfinite(data['observation_scale']).all() or np.any(data['observation_scale']<=0):
            raise ValueError('Invalid checkpoint normalisation')
        policy=GaussianPolicy(13,data['observation_mean'],data['observation_scale'],sizes[1:-1])
        for i,(w,b) in enumerate(zip(policy.network.weights,policy.network.biases)):
            if data[f'w{i}'].shape!=w.shape or data[f'b{i}'].shape!=b.shape:raise ValueError('Checkpoint parameter shape mismatch')
            w[:]=data[f'w{i}'];b[:]=data[f'b{i}']
        if not all(np.isfinite(x).all() for x in policy.network.parameters()):raise ValueError('Nonfinite checkpoint parameters')
        return policy


def train_bc_dagger_v3(sim,teacher,models_dir,gates,seeds=(0,1,2),bc_epochs=600,dagger_rounds=3,dagger_epochs=600):
    """Retained v2 imitation budget, released only when every v3 gate passes.

    DAgger labels the visited current route node from the frozen numerical
    teacher. The shield and Env state are identical to deployed evaluation.
    """
    if not all(gates.get(key) is True for key in REQUIRED_GATES):
        return {'skipped':True,'reason':'v3 release gates are incomplete or failed',
            'gates':dict(gates),'controllers':[],'sac_expanded':False}
    if not teacher['summary'].get('feasible') or len(teacher['observations'])!=sim.n_nodes:
        raise ValueError('Training requires a complete feasible numerical teacher')
    models_dir=Path(models_dir);models_dir.mkdir(parents=True,exist_ok=True)
    base_x=np.asarray(teacher['observations']);labels=np.asarray(teacher['requested_actions'])
    records=[];controllers=[];trajectories={}
    for seed in seeds:
        x=base_x.copy();y=labels.copy()
        policy,_=clone_behaviour(x,y,seed=int(seed),epochs=bc_epochs)
        best=None;candidate_records=[];trajectories[str(seed)]={}
        for stage in range(dagger_rounds+1):
            result=sim.simulate(mode='autonomous',controller=policy)
            path=models_dir/f'bc_dagger_seed_{seed}_round_{stage}.npz'
            save_policy_v3(policy,path)
            legal=bool(result['summary'].get('feasible'))
            trajectories[str(seed)][str(stage)]=result
            candidate_records.append({'round':stage,'feasible':legal,'lap_time_s':result['summary']['lap_time_s'],'checkpoint':path.name})
            if legal and (best is None or result['summary']['lap_time_s']<best[0]):best=(result['summary']['lap_time_s'],path,result)
            if stage<dagger_rounds:
                env=sim.create_env();obs=env.reset();xs=[];ys=[]
                while not env.done:
                    xs.append(obs.copy());ys.append(labels[env.i])
                    obs,_,_,_=env.step(policy(obs))
                x=np.concatenate([x,np.asarray(xs)]);y=np.concatenate([y,np.asarray(ys)])
                policy,_=clone_behaviour(x,y,seed=int(seed),epochs=dagger_epochs,initial_policy=policy)
        records.append({'seed':int(seed),'candidates':candidate_records,'selected_checkpoint':best[1].name if best else None})
        if best:
            reloaded=load_policy_v3(best[1]);replay=sim.simulate(controller=reloaded)
            for key in ('speed_mps','time_s','energy_mj'):
                if not np.allclose(replay[key],best[2][key],rtol=0,atol=1e-12):raise RuntimeError('Checkpoint '+key+' replay mismatch')
            controllers.append({'id':f'bc_dagger_seed_{seed}','checkpoint':best[1].name,'result':replay})
    return {'skipped':False,'seeds':list(seeds),'records':records,'controllers':controllers,'trajectories':trajectories,
        'bc_epochs':bc_epochs,'dagger_rounds':dagger_rounds,'dagger_epochs':dagger_epochs,'sac_expanded':False,
        'training_passed':len(controllers)==len(seeds),
        'failed_seeds':[r['seed'] for r in records if r['selected_checkpoint'] is None]}
