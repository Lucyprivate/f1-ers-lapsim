"""Regression checks for geometric notice, actual hold and causal boundaries."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from f1ers_v6.geometry_guard import GeometryAdvanceRuleShield
from f1ers_v4.vehicle import IntegrationState
from f1ers_v6.rules import RuleConfig
from f1ers_v6.r7 import freeze_r7_simulator


class Route:
    length_m = 1000.


def fixture(margin=.2, initial_dc=350., position=100.):
    s = np.arange(1000.)
    low = (s >= 500.) & (s < 510.)
    mapping = {'s_m': s, 'safe_speed_mps': np.ones(1000)*100., 'ds_m':1.,
        'positive_low_mask':low,'hazard_s_m':s[low],'length_m':1000.}
    shield = GeometryAdvanceRuleShield({}, Route(), RuleConfig(tail_guard_m=0.), mapping, margin)
    state = IntegrationState(position, 80., 0., 3.8, 0., initial_dc)
    shield.reset(state)
    return shield, state


class GuardTests(unittest.TestCase):
    def test_fast_arrival_and_margin_cap_are_geometric(self):
        shield, state = fixture()
        hazard = shield.hazard(state)
        self.assertEqual(hazard['hazard_distance_m'], 400.)
        self.assertEqual(hazard['arrival_lower_bound_s'],4.)
        self.assertAlmostEqual(hazard['guard_dc_request_upper_kw'],280.)

    def test_request_taper_keeps_real_hold_and_pure_queries(self):
        shield, state = fixture()
        command={'wheel_force_n':12000.,'brake_fraction':0.}
        info=shield.decide(state,command,1.,'mode_aware')
        self.assertAlmostEqual(info['requested_dc_kw'],280.)
        self.assertEqual(shield.state.phase,'HOLD')
        self.assertEqual(shield.state.hold_until_s,1.)
        before=asdict(shield.state)
        audit_before=deepcopy(shield.guard_audit)
        one=shield.query(state,command,None,.5)
        two=shield.query(state,command,None,.5)
        self.assertEqual(one,two)
        self.assertEqual(one['requested_dc_kw'],350.)
        self.assertEqual(before,asdict(shield.state))
        self.assertEqual(audit_before,shield.guard_audit)
        future=IntegrationState(250.,80.,1.25,2.9,0.,350.)
        self.assertAlmostEqual(shield.query(future,command,None,.25)['requested_dc_kw'],300.)

    def test_late_guard_does_not_fake_a_reachable_existing_dc(self):
        shield,state=fixture(position=490.)
        info=shield.decide(state,{'wheel_force_n':97.5,'brake_fraction':0.},1.,'mode_aware')
        self.assertEqual(info['requested_dc_kw'],0.)
        self.assertEqual(shield.query(state,{'wheel_force_n':97.5},None,.01)['requested_dc_kw'],350.)
        self.assertEqual(shield.state.previous_dc_kw,350.)

    def test_braking_is_actual_signed_demand_and_retains_actor_harvest(self):
        shield,state=fixture(position=490.)
        info=shield.decide(state,{'wheel_force_n':-1000.,'brake_fraction':0.},1.,'mode_aware')
        self.assertLess(info['requested_dc_kw'],0.)
        self.assertEqual(shield.state.phase,'NORMAL')
        self.assertTrue(shield.state.braking)
        other,st=fixture(position=500.)
        other.decide(st,{'wheel_force_n':1.,'brake_fraction':0.},1.,'mode_aware')
        self.assertEqual(other.state.phase,'HOLD')
        self.assertFalse(other.state.braking)

    def test_hazard_wraps_without_resetting_carried_dc(self):
        shield,state=fixture(position=950.)
        self.assertEqual(shield.hazard(state)['hazard_distance_m'],550.)
        carry=asdict(shield.state)
        new,_=fixture(initial_dc=350.,position=0.)
        crossing=IntegrationState(0.,80.,0.,3.,0.,350.)
        new.reset(crossing,carry)
        self.assertEqual(new.state.previous_dc_kw,350.)
        self.assertEqual(new.state.reference_mechanical_kw,332.5)

    def test_no_extra_policy_observation_keys(self):
        shield,state=fixture()
        baseline=shield.features(state,{'wheel_force_n':12000.})
        self.assertNotIn('arrival_lower_bound_s',baseline)
        self.assertNotIn('guard_dc_request_upper_kw',baseline)
        self.assertFalse(shield.certificate()['experimental_geometry_advance_guard']['new_fia_rule_claimed'])

    def test_freezing_retains_geometry_shield_for_fixed_demands_and_reset(self):
        from f1ers_v6.geometry_guard import GeometryGuardSimulator, GeometryGuardEnv
        reference,state=fixture()
        attributes={'track':{},'rules':{},'geometry':Route(),'vehicle':object(),
            'config':object(),'rule_config':RuleConfig(tail_guard_m=0.),
            'margin_s':.2,'geometry_guard_map':reference.geometry_map,
            'driver_template':object()}
        sim=SimpleNamespace(**attributes)
        profile={'s_m':np.array([0.,5.]),'wheel_force_n':np.array([7.,-9.])}
        class FixedDriver:
            def __init__(self,geometry,vehicle,frozen):
                self.force=frozen['wheel_force_n'];self.decision_s=np.array([0.,5.,10.])
            def demand(self,state):return float(self.force[0 if state.s_m<5. else 1])
            def next_time_event(self,t):return self.shield.next_time_event(t)
        def fake_init(instance,*args,**kwargs):instance.__dict__.update(attributes)
        def fake_reset(env,*args,**kwargs):
            env.state=state
            env.shield=GeometryAdvanceRuleShield({},Route(),env.sim.rule_config,
                env.sim.geometry_guard_map,env.sim.margin_s)
            env.shield.reset(state)
            return np.zeros(31)
        with patch.object(GeometryGuardSimulator,'__init__',fake_init), \
             patch.object(GeometryGuardEnv,'reset',fake_reset), \
             patch.object(GeometryGuardEnv,'get_obs',return_value=np.zeros(31)), \
             patch('f1ers_v6.r7.FrozenDemandDriver',FixedDriver):
            frozen=freeze_r7_simulator(sim,profile)
            profile['wheel_force_n'][0]=999.
            env=frozen.create_env(action_space='mode_aware')
            self.assertIsInstance(env.shield,GeometryAdvanceRuleShield)
            self.assertIs(env.driver.shield,env.shield)
            self.assertEqual(env.driver.demand(IntegrationState(0.,80.,0.,3.,0.,0.)),7.)
            np.testing.assert_array_equal(env.driver.decision_s,[0.,5.,10.])
            env.shield.decide(state,{'wheel_force_n':7.,'brake_fraction':0.},1.,'mode_aware')
            self.assertEqual(env.driver.next_time_event(0.),1.)
            env.reset()
            self.assertIsInstance(env.shield,GeometryAdvanceRuleShield)
            self.assertIs(env.driver.shield,env.shield)
            self.assertEqual(env.driver.demand(IntegrationState(0.,80.,0.,3.,0.,0.)),7.)


if __name__=='__main__':
    unittest.main()
