"""Matched masked SAC: manual NumPy and actual SB3 networks/PyTorch autograd.

The second implementation is an SB3-aligned custom float64 learner, not a
claim that stock ``SAC.learn`` executed these experiments. Its actual SB3
actor/critic modules are used, with explicitly aligned normalization, clipping,
separate critic optimizers, fixed entropy, and braking-no-op canonicalization.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import numpy as np

from ..learning import ReturnScaledSAC, _bind, save_policy
from .._core.networks import GaussianPolicy
from .environment import FEATURES, PROTOCOL, require_policy_contract


@dataclass(frozen=True)
class BackendConfig:
    return_scale_s: float = 73.084831
    alpha_seconds: float = .001
    gamma: float = 1.
    learning_rate: float = 3e-4
    tau: float = .005
    batch_size: int = 64
    critic_warmup: int = 120
    update_every: int = 2
    hidden: tuple = (32, 32)
    log_std_min: float = -5.
    log_std_max: float = 1.
    gradient_clip: float = 10.
    braking_noop_mask: bool = True

    def validate(self):
        if asdict(self) != asdict(BackendConfig()):
            raise ValueError('Research backend configuration is preregistered and fixed')
        return self


def actor_active(observations):
    """Pre-transition signed wheel demand: only strict-negative demand is no-op."""
    obs = np.atleast_2d(np.asarray(observations, float))
    if obs.shape[1] != len(FEATURES) or not np.isfinite(obs).all():
        raise ValueError('Exact finite 36-feature observations required')
    return (obs[:, 10:11] >= 0.).astype(float)


def _arrays(backend):
    networks = (backend.policy.network, backend.q1, backend.q2,
                backend.target_q1, backend.target_q2)
    arrays = {f'{name}_{i}': p.copy() for name, n in zip(
        ('actor', 'q1', 'q2', 'target_q1', 'target_q2'), networks)
        for i, p in enumerate(n.parameters())}
    arrays.update(observation_mean=backend.policy.mean.copy(),observation_scale=backend.policy.scale.copy())
    return arrays


class MaskedNumpySAC(ReturnScaledSAC):
    backend_name = 'numpy'
    implementation = 'manual NumPy SAC, canonical braking action and masked entropy'

    def __init__(self, policy, seed=0, config=None):
        self.config = (config or BackendConfig()).validate()
        require_policy_contract(policy)
        super().__init__(_bind(policy.copy(), FEATURES, 'mode_aware', PROTOCOL), seed,
                         return_scale_s=self.config.return_scale_s,
                         initial_alpha=self.config.alpha_seconds, temperature_mode='fixed')

    def export_numpy(self):
        return _bind(self.policy.copy(), FEATURES, 'mode_aware', PROTOCOL)

    def sample(self, observations, rng, epsilon=None):
        return self.policy.sample(observations, rng, epsilon)

    def actor_loss_and_gradients(self, observations, epsilon=None):
        actions, logp, cache = self.policy.sample(observations, self.rng, epsilon)
        mask = actor_active(observations)
        inputs = np.column_stack((self.policy.normalise(observations), actions*mask))
        q1, c1 = self.q1.forward(inputs, cache=True)
        q2, c2 = self.q2.forward(inputs, cache=True)
        choose = q1 <= q2
        g1, _ = self.q1.backward(c1, choose.astype(float))
        g2, _ = self.q2.backward(c2, (~choose).astype(float))
        dq = (g1+g2)[:, -1:]*mask
        da = 1-actions**2
        alpha = self.normalized_alpha
        mu_grad = 2*alpha*actions*mask-dq*da
        std_grad = alpha*(-1+2*actions*cache['std_epsilon'])*mask-dq*da*cache['std_epsilon']
        std_grad *= cache['log_std_mask']
        _, grads = self.policy.network.backward(cache['network'],
            np.column_stack((mu_grad, std_grad))/len(inputs))
        return float(np.mean(alpha*mask*logp-np.minimum(q1,q2))), grads, logp

    def bellman_target(self, batch, *, next_actions=None, next_log_probabilities=None):
        if next_actions is None:
            next_actions, next_log_probabilities, _ = self.sample(batch['next_obs'], self.rng)
        mask = actor_active(batch['next_obs'])
        inputs = np.column_stack((self.policy.normalise(batch['next_obs']), next_actions*mask))
        q = np.minimum(self.target_q1.forward(inputs), self.target_q2.forward(inputs))
        return batch['rewards']/self.return_scale_s+(1-batch['dones'])*(
            q-self.normalized_alpha*mask*next_log_probabilities)

    def gradients(self, batch, next_epsilon, actor_epsilon):
        a, lp, _ = self.sample(batch['next_obs'], self.rng, next_epsilon)
        target = self.bellman_target(batch, next_actions=a, next_log_probabilities=lp)
        x = np.column_stack((self.policy.normalise(batch['obs']),
                             batch['actions']*actor_active(batch['obs'])))
        qgrads, qloss = [], []
        for q in (self.q1,self.q2):
            y, cache = q.forward(x, cache=True)
            _, g = q.backward(cache,2*(y-target)/len(x))
            qgrads.append(g); qloss.append(float(np.mean((y-target)**2)))
        aloss, ag, lp = self.actor_loss_and_gradients(batch['obs'], actor_epsilon)
        return {'target':target, 'q_gradients':qgrads, 'q_losses':qloss,
                'actor_loss':aloss, 'actor_gradients':ag, 'logp':lp}

    def update(self, batch, *, next_epsilon=None, actor_epsilon=None):
        n = len(batch['obs'])
        next_epsilon = self.rng.normal(size=(n,1)) if next_epsilon is None else next_epsilon
        actor_epsilon = self.rng.normal(size=(n,1)) if actor_epsilon is None else actor_epsilon
        a,lp,_ = self.sample(batch['next_obs'],self.rng,next_epsilon)
        target = self.bellman_target(batch,next_actions=a,next_log_probabilities=lp)
        qloss = self.supervised_critic_update(batch['obs'],batch['actions'],target)
        loss,gradients,lp = self.actor_loss_and_gradients(batch['obs'],actor_epsilon)
        self.actor_optimiser.step(gradients)
        self.target_q1.polyak_update(self.q1,self.tau)
        self.target_q2.polyak_update(self.q2,self.tau)
        self.updates += 1
        return {'q1_loss_normalized':qloss[0], 'q2_loss_normalized':qloss[1],
                'actor_loss_normalized':loss, 'alpha_seconds':self.alpha,
                'alpha_normalized':self.normalized_alpha,'gamma':1.,
                'masked_brake_fraction':float(1-actor_active(batch['obs']).mean()),
                'raw_replay_retained':True,'updates':self.updates}

    def supervised_critic_update(self, obs, raw_actions, target):
        x = np.column_stack((self.policy.normalise(obs),
                             np.asarray(raw_actions).reshape(-1,1)*actor_active(obs)))
        losses=[]
        for q,opt in ((self.q1,self.q1_optimiser),(self.q2,self.q2_optimiser)):
            y,cache=q.forward(x,cache=True)
            _,g=q.backward(cache,2*(y-target)/len(x))
            opt.step(g);losses.append(float(np.mean((y-target)**2)))
        return losses

    def synchronize_targets(self):
        self.target_q1,self.target_q2=self.q1.copy(),self.q2.copy()

    def q_values(self, obs, raw_actions):
        x=np.column_stack((self.policy.normalise(obs),np.asarray(raw_actions).reshape(-1,1)*actor_active(obs)))
        return np.c_[self.q1.forward(x),self.q2.forward(x)]*self.return_scale_s

    def parameter_arrays(self):
        return _arrays(self)

    def save(self, path):
        path=Path(path);save_policy(self.export_numpy(),path)
        arrays=self.parameter_arrays()
        for name,opt in (('actor',self.actor_optimiser),('q1',self.q1_optimiser),('q2',self.q2_optimiser)):
            arrays[name+'_adam_steps']=np.asarray(opt.steps)
            for i,(m,v) in enumerate(zip(opt.first,opt.second)):
                arrays[f'{name}_adam_m_{i}']=m;arrays[f'{name}_adam_v_{i}']=v
        np.savez_compressed(path.with_name(path.stem+'_training_state.npz'),**arrays)

    def dependency_evidence(self):
        return {'implementation':self.implementation,'stock_sb3_learn_executed':False,
                'numpy_version':np.__version__,'dtype':'float64'}


class SB3AlignedSAC:
    """Actual SB3 modules, matched float64 custom update instead of stock learn."""
    backend_name='sb3'
    implementation='SB3 network modules with explicitly aligned custom float64 SAC updates'

    def __init__(self,policy,seed=0,config=None):
        import torch
        import gymnasium as gym
        import stable_baselines3 as sb3
        self.torch,self.sb3=torch,sb3
        self.config=(config or BackendConfig()).validate()
        reference=MaskedNumpySAC(policy,seed,self.config)
        self.mean,self.scale=policy.mean.copy(),policy.scale.copy()
        self.rng=np.random.default_rng(seed+7919)
        self.rng.bit_generator.state=deepcopy(reference.rng.bit_generator.state)
        class SpaceOnly(gym.Env):
            observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(len(FEATURES),),dtype=np.float64)
            action_space=gym.spaces.Box(-1.,1.,shape=(1,),dtype=np.float64)
            def reset(self,**kw): return np.zeros(len(FEATURES),np.float64),{}
            def step(self,action): raise RuntimeError('Space-only SB3 adapter cannot simulate a lap')
        self.model=sb3.SAC('MlpPolicy',SpaceOnly(),learning_rate=self.config.learning_rate,
            buffer_size=128,learning_starts=0,batch_size=64,tau=.005,gamma=1.,
            train_freq=2,gradient_steps=1,ent_coef=self.config.alpha_seconds/self.config.return_scale_s,
            policy_kwargs={'net_arch':{'pi':[32,32],'qf':[32,32]},'activation_fn':torch.nn.Tanh},
            device='cpu',seed=seed,verbose=0)
        self.model.policy.to(torch.float64)
        self.actor=self.model.actor
        self.critics=list(self.model.critic.q_networks)
        self.targets=list(self.model.critic_target.q_networks)
        self._load(reference)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=self.config.learning_rate)
        self.critic_opts=[torch.optim.Adam(q.parameters(),lr=self.config.learning_rate) for q in self.critics]
        self.return_scale_s=self.config.return_scale_s
        self.alpha=self.config.alpha_seconds
        self.normalized_alpha=self.alpha/self.return_scale_s
        self.updates=0

    def _linears(self,module):
        return [m for m in module.modules() if isinstance(m,self.torch.nn.Linear)]

    def _load(self,reference):
        torch=self.torch
        with torch.no_grad():
            for layer,w,b in zip(self._linears(self.actor.latent_pi),reference.policy.network.weights[:-1],reference.policy.network.biases[:-1]):
                layer.weight.copy_(torch.as_tensor(w.T));layer.bias.copy_(torch.as_tensor(b))
            for i,head in enumerate((self.actor.mu,self.actor.log_std)):
                head.weight.copy_(torch.as_tensor(reference.policy.network.weights[-1][:,i:i+1].T))
                head.bias.copy_(torch.as_tensor(reference.policy.network.biases[-1][i:i+1]))
            for module,network in zip(self.critics+self.targets,(reference.q1,reference.q2,reference.target_q1,reference.target_q2)):
                for layer,w,b in zip(self._linears(module),network.weights,network.biases):
                    layer.weight.copy_(torch.as_tensor(w.T));layer.bias.copy_(torch.as_tensor(b))

    def _tensor(self,array): return self.torch.as_tensor(np.asarray(array),dtype=self.torch.float64)
    def _normalise(self,obs):
        return self._tensor(np.clip((np.atleast_2d(obs)-self.mean)/self.scale,-8,8))

    def _sample_tensor(self,obs,epsilon):
        torch=self.torch
        latent=self.actor.latent_pi(self._normalise(obs))
        mu,raw=self.actor.mu(latent),self.actor.log_std(latent)
        # Exact strict clipping derivative matches the audited NumPy primitive.
        ls=torch.where((raw>-5)&(raw<1),raw,torch.clamp(raw,-5,1).detach())
        eps=self._tensor(epsilon)
        z=mu+torch.exp(ls)*eps
        a=torch.tanh(z)
        ld=2*(np.log(2)-z-torch.nn.functional.softplus(-2*z))
        lp=-.5*eps**2-ls-.5*np.log(2*np.pi)-ld
        return a,lp

    def sample(self,observations,rng,epsilon=None):
        obs=np.atleast_2d(observations)
        eps=rng.normal(size=(len(obs),1)) if epsilon is None else epsilon
        with self.torch.no_grad(): a,lp=self._sample_tensor(obs,eps)
        return a.numpy(),lp.numpy(),{}

    def export_numpy(self):
        p=GaussianPolicy(len(FEATURES),self.mean,self.scale,(32,32))
        for i,layer in enumerate(self._linears(self.actor.latent_pi)):
            p.network.weights[i][...]=layer.weight.detach().numpy().T
            p.network.biases[i][...]=layer.bias.detach().numpy()
        p.network.weights[-1][...]=np.c_[self.actor.mu.weight.detach().numpy().T,self.actor.log_std.weight.detach().numpy().T]
        p.network.biases[-1][...]=np.r_[self.actor.mu.bias.detach().numpy(),self.actor.log_std.bias.detach().numpy()]
        return _bind(p,FEATURES,'mode_aware',PROTOCOL)

    @property
    def policy(self): return self.export_numpy()

    def bellman_target(self,batch,*,next_actions=None,next_log_probabilities=None):
        if next_actions is None:
            next_actions,next_log_probabilities,_=self.sample(batch['next_obs'],self.rng)
        with self.torch.no_grad():
            mask=self._tensor(actor_active(batch['next_obs']))
            x=self.torch.cat((self._normalise(batch['next_obs']),self._tensor(next_actions)*mask),dim=1)
            q=self.torch.minimum(self.targets[0](x),self.targets[1](x))
            t=self._tensor(batch['rewards'])/self.return_scale_s+(1-self._tensor(batch['dones']))*(q-self.normalized_alpha*mask*self._tensor(next_log_probabilities))
        return t.numpy()

    def _actor_loss(self,obs,epsilon):
        a,lp=self._sample_tensor(obs,epsilon)
        mask=self._tensor(actor_active(obs))
        x=self.torch.cat((self._normalise(obs),a*mask),dim=1)
        q1,q2=self.critics[0](x),self.critics[1](x)
        # Select q1 at a tie, consistently with the manual implementation.
        q=self.torch.where(q1<=q2,q1,q2)
        return (self.normalized_alpha*mask*lp-q).mean(),lp

    def _standard_actor_grads(self,grads):
        mapping={id(p):g for p,g in zip(self.actor.parameters(),grads)}
        values=[]
        for l in self._linears(self.actor.latent_pi):
            values += [mapping[id(l.weight)].detach().numpy().T,mapping[id(l.bias)].detach().numpy()]
        values += [np.c_[mapping[id(self.actor.mu.weight)].detach().numpy().T,mapping[id(self.actor.log_std.weight)].detach().numpy().T],
                   np.r_[mapping[id(self.actor.mu.bias)].detach().numpy(),mapping[id(self.actor.log_std.bias)].detach().numpy()]]
        return values

    def gradients(self,batch,next_epsilon,actor_epsilon):
        a,lp,_=self.sample(batch['next_obs'],self.rng,next_epsilon)
        target=self._tensor(self.bellman_target(batch,next_actions=a,next_log_probabilities=lp))
        x=self.torch.cat((self._normalise(batch['obs']),self._tensor(batch['actions']*actor_active(batch['obs']))),dim=1)
        qgrads,qloss=[],[]
        for q in self.critics:
            loss=((q(x)-target)**2).mean()
            g=self.torch.autograd.grad(loss,tuple(q.parameters()))
            qgrads.append([v.detach().numpy().T if v.ndim==2 else v.detach().numpy() for v in g]);qloss.append(float(loss.detach()))
        loss,lp=self._actor_loss(batch['obs'],actor_epsilon)
        g=self.torch.autograd.grad(loss,tuple(self.actor.parameters()))
        return {'target':target.numpy(),'q_gradients':qgrads,'q_losses':qloss,
                'actor_loss':float(loss.detach()),'actor_gradients':self._standard_actor_grads(g),'logp':lp.detach().numpy()}

    def _step(self,opt,parameters,loss):
        parameters=tuple(parameters)
        grads=self.torch.autograd.grad(loss,parameters)
        if not all(self.torch.isfinite(g).all() for g in grads): raise FloatingPointError('Nonfinite autograd gradient')
        norm=self.torch.sqrt(sum((g*g).sum() for g in grads))
        scale=min(1.,10./max(float(norm),1e-12))
        opt.zero_grad(set_to_none=True)
        for p,g in zip(parameters,grads): p.grad=g*scale
        opt.step()

    def supervised_critic_update(self,obs,raw_actions,target):
        x=self.torch.cat((self._normalise(obs),self._tensor(np.asarray(raw_actions).reshape(-1,1)*actor_active(obs))),dim=1)
        labels=self._tensor(target);losses=[]
        for q,opt in zip(self.critics,self.critic_opts):
            loss=((q(x)-labels)**2).mean();losses.append(float(loss.detach()));self._step(opt,q.parameters(),loss)
        return losses

    def synchronize_targets(self):
        for t,q in zip(self.targets,self.critics): t.load_state_dict(q.state_dict())

    def update(self,batch,*,next_epsilon=None,actor_epsilon=None):
        n=len(batch['obs'])
        ne=self.rng.normal(size=(n,1)) if next_epsilon is None else next_epsilon
        ae=self.rng.normal(size=(n,1)) if actor_epsilon is None else actor_epsilon
        a,lp,_=self.sample(batch['next_obs'],self.rng,ne)
        target=self.bellman_target(batch,next_actions=a,next_log_probabilities=lp)
        qloss=self.supervised_critic_update(batch['obs'],batch['actions'],target)
        loss,lp=self._actor_loss(batch['obs'],ae)
        self._step(self.actor_opt,self.actor.parameters(),loss)
        with self.torch.no_grad():
            for t,q in zip(self.targets,self.critics):
                for tp,p in zip(t.parameters(),q.parameters()): tp.mul_(1-self.config.tau).add_(p,alpha=self.config.tau)
        self.updates+=1
        return {'q1_loss_normalized':qloss[0],'q2_loss_normalized':qloss[1],
                'actor_loss_normalized':float(loss.detach()),'alpha_seconds':self.alpha,
                'alpha_normalized':self.normalized_alpha,'gamma':1.,
                'masked_brake_fraction':float(1-actor_active(batch['obs']).mean()),
                'raw_replay_retained':True,'updates':self.updates}

    def q_values(self,obs,raw_actions):
        x=self.torch.cat((self._normalise(obs),self._tensor(np.asarray(raw_actions).reshape(-1,1)*actor_active(obs))),dim=1)
        with self.torch.no_grad(): q=self.torch.cat([n(x) for n in self.critics],dim=1)
        return q.numpy()*self.return_scale_s

    def parameter_arrays(self):
        values={f'actor_{i}':p.copy() for i,p in enumerate(self.export_numpy().network.parameters())}
        values.update(observation_mean=self.mean.copy(),observation_scale=self.scale.copy())
        for name,q in zip(('q1','q2','target_q1','target_q2'),self.critics+self.targets):
            for i,l in enumerate(self._linears(q)):
                values[f'{name}_{2*i}']=l.weight.detach().numpy().T.copy()
                values[f'{name}_{2*i+1}']=l.bias.detach().numpy().copy()
        return values

    def save(self,path):
        path=Path(path);save_policy(self.export_numpy(),path)
        arrays=self.parameter_arrays()
        # Portable non-pickle training state including all actual Adam moments.
        for name,opt in [('actor',self.actor_opt),('q1',self.critic_opts[0]),('q2',self.critic_opts[1])]:
            for i,group in enumerate(opt.param_groups):
                for j,p in enumerate(group['params']):
                    for key,v in opt.state.get(p,{}).items():
                        arrays[f'{name}_adam_{i}_{j}_{key}']=v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
        np.savez_compressed(path.with_name(path.stem+'_training_state.npz'),**arrays)

    def dependency_evidence(self):
        p=Path(self.sb3.__file__)
        return {'implementation':self.implementation,'stock_sb3_learn_executed':False,
            'actual_sb3_modules_used':['SACPolicy.Actor.latent_pi','Actor.mu','Actor.log_std','ContinuousCritic.q_networks'],
            'sb3_version':self.sb3.__version__,'sb3_module_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'torch_version':self.torch.__version__,'dtype':'float64',
            'aligned_changes':['float64','fixed normalizer','strict logstd clipping derivative',
              'per-critic Adam','gradient norm clipping10','pre-demand braking no-op mask']}


def make_backend(name,policy,seed,config=None):
    if name=='numpy': return MaskedNumpySAC(policy,seed,config)
    if name=='sb3': return SB3AlignedSAC(policy,seed,config)
    raise ValueError('Only preregistered numpy and sb3 backends are allowed')


def differential_audit(output_dir=None, *, seed=2026, tolerance=1e-9):
    """Synthetic common-batch float64 audit; never runs a vehicle or trains episodes."""
    rng=np.random.default_rng(seed)
    p=_bind(GaussianPolicy(len(FEATURES),seed=seed),FEATURES,'mode_aware',PROTOCOL)
    a,b=MaskedNumpySAC(p,seed),SB3AlignedSAC(p,seed)
    obs=rng.normal(size=(64,len(FEATURES))); nxt=rng.normal(size=obs.shape)
    obs[:21,10]=-abs(obs[:21,10]);obs[21:,10]=abs(obs[21:,10])
    batch={'obs':obs,'next_obs':nxt,'actions':rng.uniform(-1,1,(64,1)),
           'rewards':-rng.uniform(.01,1,(64,1)),'dones':(rng.random((64,1))<.2).astype(float)}
    cases=[]
    # Default update streams must also match after critic initialization consumed RNG.
    a.update(batch);b.update(batch)
    common_rng_error=max(float(np.max(abs(a.parameter_arrays()[k]-b.parameter_arrays()[k])))
                         for k in a.parameter_arrays())
    for step in range(10):
        fixture='ordinary' if step<3 else ('saturated_mean' if step==3 else
                'clipped_logstd' if step==4 else 'equal_twin_Q' if step==5 else
                'terminal_zero_bootstrap' if step==6 else 'negative_saturation_lower_clip_edge' if step==7 else
                'upper_clip_edge' if step==8 else 'global_gradient_clipping')
        if step in (3,4,7,8):
            # Force explicit extreme fixtures, with exactly shared parameters.
            a.policy.network.weights[-1][...]=0.
            a.policy.network.biases[-1][...]=({3:[30.,-8.],4:[0.,3.],7:[-30.,-5.],8:[0.,1.]})[step]
            b._load(a)
        if step==5:
            for dst,src in zip(a.q2.parameters(),a.q1.parameters()): dst[...]=src
            for dst,src in zip(a.target_q2.parameters(),a.target_q1.parameters()): dst[...]=src
            b._load(a)
        if step==6: batch['dones'][...]=1.
        if step==9: batch['rewards'][...]=-100000.
        ne=rng.normal(size=(64,1));ae=rng.normal(size=(64,1))
        ga,gb=a.gradients(batch,ne,ae),b.gradients(batch,ne,ae)
        diffs={'target':float(np.max(abs(ga['target']-gb['target']))),
               'actor_loss':abs(ga['actor_loss']-gb['actor_loss']),
               'logp':float(np.max(abs(ga['logp']-gb['logp'])))}
        for key in ('actor_gradients','q_gradients'):
            aa=ga[key] if key=='actor_gradients' else sum(ga[key],[])
            bb=gb[key] if key=='actor_gradients' else sum(gb[key],[])
            diffs[key]=max(float(np.max(abs(x-y))) for x,y in zip(aa,bb))
        a.update(batch,next_epsilon=ne,actor_epsilon=ae);b.update(batch,next_epsilon=ne,actor_epsilon=ae)
        va,vb=a.parameter_arrays(),b.parameter_arrays()
        diffs['Adam_and_Polyak']=max(float(np.max(abs(va[k]-vb[k]))) for k in va)
        if step==6:
            diffs['terminal_bootstrap']=float(np.max(abs(ga['target']-batch['rewards']/a.return_scale_s)))
        cases.append({'update':step+1,'fixture':fixture,'errors':diffs,'passed':all(v<=tolerance for v in diffs.values())})
    brake=obs.copy();brake[:,10]=-1
    brake_gradient=a.actor_loss_and_gradients(brake,epsilon=np.ones((64,1)))[1]
    flatq=a.q_values(brake,np.ones((64,1)))-a.q_values(brake,-np.ones((64,1)))
    no_op=bool(all(np.all(g==0) for g in brake_gradient) and np.all(flatq==0))
    report={'passed':all(c['passed'] for c in cases) and no_op and common_rng_error<=tolerance,'tolerance':tolerance,
        'cases':cases,'braking_actor_gradient_exact_zero':no_op,
        'default_common_update_RNG_error':common_rng_error,
        'braking_Q_action_invariant':bool(np.all(flatq==0)),'episodes_spent':0,
        'common_batch_common_epsilon':True,'dependency_evidence':b.dependency_evidence()}
    if output_dir is not None:
        from .._core.learning_base import _write
        _write(Path(output_dir)/'differential_audit.json',report)
    return report
