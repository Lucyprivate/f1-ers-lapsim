"""Synthetic regressions: no private data, vehicle training or FP2 access."""
import importlib.util
import tempfile
import unittest
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from f1ers_v7.learning import _bind, load_policy
from f1ers_v7._core.networks import GaussianPolicy, ReplayBuffer
from f1ers_v7.research.environment import (FEATURES,PROTOCOL,ResearchSimulator,
    GymResearchEnv,validate_boundary,require_policy_contract)
from f1ers_v7.research.backends import MaskedNumpySAC,differential_audit,actor_active
from f1ers_v7.research.learning import (fit_bc,train_case,rollout,physical_noop_audit,
    frozen_policy_audit)
from f1ers_v7.research.protocol import EpisodeBudget

BOUNDARY={'speed_mps':77.,'energy_mj':2.,'carried_rule_state':None}


@dataclass
class ToyState:
    s_m:float=0.
    speed_mps:float=77.
    energy_mj:float=2.
    previous_dc_kw:float=0.
    elapsed_time_s:float=0.


class ToyEnv:
    def __init__(self,sim,boundary):
        self.sim,self.initial=sim,ToyState(speed_mps=boundary['speed_mps'],energy_mj=boundary['energy_mj'])
        self.state=ToyState(**asdict(self.initial));self.i=0;self.obs=[];self.actions=[]
    def get_obs(self):
        x=np.zeros(36);x[0]=self.i/8;x[10]=-1. if self.i%3==1 else 1.
        x[33]=self.initial.energy_mj/4;x[34]=self.initial.speed_mps/100
        return x
    def step(self,a):
        self.obs.append(self.get_obs());self.actions.append(a)
        self.i+=1;self.state.s_m=5.*self.i;self.state.elapsed_time_s=.1*self.i
        fail=self.sim.numeric and self.i==3
        done=self.i==8 or fail
        return self.get_obs(),-.1,done,{'status':'numerical_fail' if fail else 'accepted',
                                      'step_certified':not fail}
    def result(self):
        full=self.i==8 and not self.sim.numeric
        summary={k:full for k in ('completed_lap','execution_feasible','feasible','sustainable',
          'public_rules_passed','periodic_speed_closed','dc_boundary_closed','rule_state_closed')}
        summary.update(protocol=PROTOCOL,feature_schema=FEATURES,lap_time_s=.8 if full else None,
          energy_start_mj=self.initial.energy_mj,energy_end_mj=self.state.energy_mj,
          periodic_speed_residual_mps=0.)
        return {'summary':summary,'observations':np.asarray(self.obs).reshape(-1,36),
          'requested_actions':np.asarray(self.actions),'terminal_speed_mps':self.state.speed_mps,
          'rule_certificate':{'terminal_boundary_state':{'phase':'NORMAL','requested_dc_kw':0.}},
          'internal_trace':{'s_m':[0.,self.state.s_m],'time_s':[0.,self.state.elapsed_time_s],
            'speed_mps':[77.,77.],'energy_mj':[self.initial.energy_mj,self.state.energy_mj],
            'dc_kw':[0.],'wheel_force_n':[1.]}}


class ToySimulator:
    n_decisions=8
    length_m=40.
    def __init__(self,numeric=False,reset_error=False):
        self.numeric=numeric;self.reset_error=reset_error;self.created=0
    def create_env(self,initial_speed_mps,initial_energy_mj,carried_rule_state,**kw):
        self.created+=1
        if self.reset_error and self.created>1: raise RuntimeError('Synthetic reset interruption')
        return ToyEnv(self,dict(speed_mps=initial_speed_mps,energy_mj=initial_energy_mj,
                               carried_rule_state=carried_rule_state))


def policy(seed=2026):
    return _bind(GaussianPolicy(36,seed=seed),FEATURES,'mode_aware',PROTOCOL)


class ResearchLearningTests(unittest.TestCase):
    def test_exact_schema_and_boundary_guards(self):
        p=policy();require_policy_contract(p)
        p.v6_features=FEATURES[:-1]
        with self.assertRaises(ValueError): require_policy_contract(p)
        with self.assertRaises(ValueError): validate_boundary({**BOUNDARY,'carried_rule_state':{'phase':'HOLD'}})
        with self.assertRaises(ValueError): validate_boundary({**BOUNDARY,'carried_rule_state':{'phase':'NORMAL','reference_mechanical_kw':1.}})
        with self.assertRaises(ValueError): ResearchSimulator.periodic_simulate(object())

    def test_pretransition_mask_and_raw_replay(self):
        o=np.zeros((3,36));o[:,10]=[-1.,0.,1.]
        np.testing.assert_array_equal(actor_active(o).ravel(),[0,1,1])
        replay=ReplayBuffer(3,36);replay.add(o[0],.7,-1.,o[1],True)
        self.assertEqual(replay.actions[0,0],.7)
        b=MaskedNumpySAC(policy());brake=o[:1]
        np.testing.assert_array_equal(b.q_values(brake,[-1.]),b.q_values(brake,[1.]))
        _,g,_=b.actor_loss_and_gradients(brake,np.ones((1,1)))
        self.assertTrue(all(np.all(v==0) for v in g))

    @unittest.skipUnless(importlib.util.find_spec('gymnasium'),'optional gymnasium unavailable')
    def test_gym_termination_truncation_and_frozen_boundary(self):
        g=GymResearchEnv(ToySimulator(numeric=True),[BOUNDARY])
        obs,info=g.reset(options={'boundary_id':0})
        self.assertEqual(obs.shape,(36,))
        with self.assertRaises(ValueError): g.reset(options={'energy_mj':3.})
        with self.assertRaises(ValueError): g.reset(options={'boundary_id':1})
        for i in range(3): obs,r,term,trunc,info=g.step([0.])
        self.assertFalse(term);self.assertTrue(trunc)
        self.assertFalse(info['trainable_transition']);self.assertFalse(info['bootstrap_allowed'])
        g=GymResearchEnv(ToySimulator(),[BOUNDARY]);g.reset()
        for i in range(8): obs,r,term,trunc,info=g.step([0.])
        self.assertTrue(term);self.assertFalse(trunc);self.assertFalse(info['bootstrap_allowed'])

    def test_numerical_prefix_excluded_from_replay(self):
        replay=ReplayBuffer(20,36)
        tr=rollout(ToySimulator(numeric=True),policy(),BOUNDARY,replay=replay)
        self.assertEqual(len(tr['actions']),3);self.assertEqual(replay.size,2)
        self.assertTrue(tr['truncated'][-1]);self.assertFalse(tr['terminated'][-1])

    def test_constant_clone_disclosed_and_exact(self):
        env=ToyEnv(ToySimulator(),BOUNDARY)
        for i in range(8): env.step(0.)
        bc=fit_bc([env.result()],seed=2026,epochs=3)
        self.assertTrue(bc['metadata']['analytical_zero_mean_clone'])
        self.assertTrue(np.all(bc['policy'].network.weights[-1][:,0]==0))
        self.assertEqual(bc['policy'].network.biases[-1][0],0.)
        self.assertFalse(bc['metadata']['analytical_clone_counted_as_learning_gain'])

    def test_durable_attempt_before_reset_failure(self):
        temporary_root=Path(__file__).resolve().parents[1]/'work'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as tmp:
            root=Path(tmp);budget=EpisodeBudget(root/'ledger')
            result=train_case(ToySimulator(reset_error=True),policy(),root/'case',backend='numpy',
                seed=2026,condition='cold',boundary_pool=[BOUNDARY],budget=budget,episodes=2)
            self.assertEqual(budget.counts()['attempted'],1)
            self.assertTrue(result['metadata']['stop_all_training'])
            self.assertTrue((root/'case/episode_001_training.json.gz').exists())
            self.assertTrue((root/'case/episode_001_policy.npz').exists())
            with self.assertRaises(FileExistsError):
                train_case(ToySimulator(),policy(),root/'case',backend='numpy',seed=2026,
                    condition='cold',boundary_pool=[BOUNDARY],budget=budget)
            with self.assertRaises(FileExistsError): budget.claim('numpy',2026,'cold',1)
            restored=load_policy(root/'case/episode_001_policy.npz',expected_features=FEATURES,
              expected_action_space='mode_aware',expected_execution_protocol=PROTOCOL)
            require_policy_contract(restored)

    def test_frozen_counterfactual_noop_and_no_optimizer_updates(self):
        sim=ToySimulator();b=MaskedNumpySAC(policy())
        self.assertTrue(physical_noop_audit(sim,b.policy,BOUNDARY)['passed'])
        report=frozen_policy_audit(sim,b,BOUNDARY)
        self.assertTrue(report['passed']);self.assertEqual(report['conditional_suffixes'],60)
        self.assertEqual(report['frozen_parameters_before'],report['frozen_parameters_after'])
        self.assertEqual(b.updates,0)

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'),'optional SB3 unavailable')
    def test_float64_differential_forward_gradient_adam_polyak(self):
        report=differential_audit()
        self.assertTrue(report['passed'],report)
        self.assertFalse(report['dependency_evidence']['stock_sb3_learn_executed'])
        self.assertEqual(report['episodes_spent'],0)
        self.assertIn('terminal_zero_bootstrap',[c['fixture'] for c in report['cases']])


if __name__=='__main__': unittest.main()
