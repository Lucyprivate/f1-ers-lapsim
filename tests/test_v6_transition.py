"""Meaningful regressions for physical coupling and one execution path."""
import unittest
from dataclasses import replace
import numpy as np

from f1ers_v3.geometry import ClosedRoute
from f1ers_v4.vehicle import IntegrationState, VehicleParamsV4, IntegratorConfig
from f1ers_v4.simulator import ContinuousGeometry
from f1ers_v6.driver import PowerResponsiveDriver
from f1ers_v6.simulator import SimulatorV6
from f1ers_v6.environment import FEATURES


def fixture(radius=20.):
    theta=np.linspace(0.,2*np.pi,40,endpoint=False)
    route=ClosedRoute(np.column_stack((radius*np.cos(theta),radius*np.sin(theta))),kind='spline')
    rules={'soc_window_mj':4.,'ers_max_power_kw':350.,'harvest_max_mj_per_lap':8.,
        'alternative_power_zones_route_m':[],'power_reduction_exceptions_route':[],
        'fia_centreline_length_m':route.source_length,
        'fia_sector_lengths_m':[route.source_length/3]*3}
    return route.sample(5.),rules


class V6TransitionTests(unittest.TestCase):
    def test_geometry_ceiling_does_not_include_ice_only_target(self):
        track,rules=fixture(500.)
        p=VehicleParamsV4()
        g=ContinuousGeometry(track,rules,p)
        driver=PowerResponsiveDriver(g,p)
        other=PowerResponsiveDriver(g,replace(p,ice_power_kw=200.))
        np.testing.assert_array_equal(driver.safe_speed,other.safe_speed)
        command=driver.decide(IntegrationState(0.,60.,energy_mj=3.,recharge_mj=0.))
        self.assertIsNone(command['target_acceleration_mps2'])
        self.assertEqual(command['wheel_force_n'],12000.)

    def test_deployment_changes_speed_when_not_geometry_limited(self):
        track,rules=fixture(500.)
        sim=SimulatorV6(track,rules)
        a=sim.create_env(initial_speed_mps=60.)
        b=sim.create_env(initial_speed_mps=60.)
        a.state=IntegrationState(0.,60.,0.,3.,0.,0.)
        b.state=IntegrationState(0.,60.,0.,3.,0.,0.)
        _,_,_,ia=a.step(0.)
        _,_,_,ib=b.step(1.)
        self.assertEqual(ia['status'],'accepted')
        self.assertEqual(ib['status'],'accepted')
        self.assertGreater(b.state.speed_mps,a.state.speed_mps+.01)
        self.assertLess(b.state.energy_mj,a.state.energy_mj)

    def test_environment_and_simulation_have_exact_parity(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules)
        env=sim.create_env()
        initial=env.reset()
        self.assertEqual(initial.shape,(len(FEATURES),))
        while not env.done:env.step(0.)
        a,b=env.result(),sim.simulate()
        for key in ('speed_mps','time_s','energy_mj','mguk_dc_kw','observations'):
            np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue(a['summary']['completed_lap'])
        self.assertTrue(a['summary']['execution_feasible'])
        self.assertTrue(a['summary']['sustainable'])
        self.assertLess(abs(a['summary']['energy_balance_residual_mj']),1e-8)
        self.assertEqual(a['summary']['intermediate_speed_resets'],0)

    def test_failed_proposal_and_partial_trajectory_retained(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules)
        result=sim.simulate(initial_speed_mps=80.)
        self.assertFalse(result['summary']['completed_lap'])
        self.assertFalse(result['summary']['feasible'])
        self.assertIsNotNone(result['failure_proposal'])
        self.assertLess(result['terminal_s_m'],sim.length_m)
        self.assertEqual(result['terminal_speed_mps'],80.)

    def test_bad_action_and_terminal_reentry_rejected(self):
        track,rules=fixture()
        env=SimulatorV6(track,rules).create_env()
        with self.assertRaises(ValueError):env.step(float('nan'))
        while not env.done:env.step(0.)
        with self.assertRaises(RuntimeError):env.step(0.)

    def test_policy_specific_shooting_without_velocity_resets(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules)
        result=sim.periodic_simulate(max_cycles=4)
        self.assertTrue(result['summary']['shooting']['converged'])
        self.assertLessEqual(result['summary']['periodic_speed_residual_mps'],1e-4)
        self.assertEqual(result['summary']['shooting']['within_lap_velocity_resets'],0)

    def test_positive_target_at_drag_deceleration_is_not_braking(self):
        track,rules=fixture(500.)
        p=VehicleParamsV4(drag_area_m2=2.)
        driver=PowerResponsiveDriver(ContinuousGeometry(track,rules,p),p)
        s=IntegrationState(0.,60.,energy_mj=3.,recharge_mj=0.)
        cmd=driver.decide(s)
        self.assertGreater(cmd['wheel_force_n'],0.)
        self.assertEqual(cmd['brake_fraction'],0.)
        self.assertIsNone(cmd['target_acceleration_mps2'])

    def test_numerical_root_failure_never_invents_complete_interval(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules,config=IntegratorConfig(root_iterations=1))
        result=sim.simulate()
        self.assertFalse(result['summary']['completed_lap'])
        self.assertFalse(result['summary']['feasible'])
        self.assertEqual(result['terminal_s_m'],0.)

    def test_energy_and_rule_state_cross_the_physical_control_line(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules)
        pair=sim.simulate_laps(n_laps=2,initial_energy_mj=3.)
        self.assertTrue(pair['physical_control_line_continuity'])
        self.assertEqual(pair['laps'][0]['summary']['energy_end_mj'],3.)
        self.assertEqual(pair['laps'][1]['summary']['energy_start_mj'],3.)
        self.assertTrue(pair['crossings'][0]['rule_state_carried'])
        self.assertEqual(pair['crossings'][0]['speed_difference_mps'],0.)
        self.assertEqual(pair['crossings'][0]['energy_difference_mj'],0.)

    def test_carried_previous_dc_is_not_silently_zeroed(self):
        from f1ers_v6.rules import RuleState
        from dataclasses import asdict
        track,rules=fixture(500.)
        carry=asdict(RuleState(previous_dc_kw=100.,reference_mechanical_kw=95.))
        env=SimulatorV6(track,rules).create_env(initial_speed_mps=60.,
            initial_energy_mj=3.,carried_rule_state=carry)
        self.assertEqual(env.state.previous_dc_kw,100.)
        self.assertEqual(env.shield.state.previous_dc_kw,100.)
        self.assertEqual(env.state.energy_mj,3.)

    def test_rule_hold_event_is_visible_to_frozen_integrator(self):
        track,rules=fixture()
        driver=PowerResponsiveDriver(ContinuousGeometry(track,rules,VehicleParamsV4()),VehicleParamsV4())
        class Hold:
            def next_time_event(self,t):return .5 if t<.5 else float('inf')
        driver.shield=Hold()
        self.assertEqual(driver.next_time_event(.1),.5)
        self.assertEqual(driver.next_time_event(.5),float('inf'))

    def test_initial_energy_outside_frozen_window_is_rejected(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules)
        for value in (1.9,4.1,float('nan')):
            with self.assertRaises(ValueError):sim.create_env(initial_energy_mj=value)

    def test_missing_rule_certificate_cannot_create_a_pass(self):
        track,rules=fixture()
        env=SimulatorV6(track,rules).create_env()
        env.shield.certificate=lambda:{}
        result=env.result()
        self.assertFalse(result['summary']['public_rules_passed'])
        self.assertFalse(result['summary']['rule_state_closed'])
        self.assertFalse(result['summary']['feasible'])

    def test_consecutive_physical_laps_reject_per_lap_shooting(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules)
        with self.assertRaises(ValueError):sim.simulate_laps(n_laps=2,periodic_shooting=True)

    def test_controller_none_is_zero_dc_in_both_action_spaces(self):
        track,rules=fixture()
        sim=SimulatorV6(track,rules)
        for space in ('signed','mode_aware'):
            result=sim.simulate(action_space=space)
            self.assertEqual(result['summary']['deploy_mj'],0.)
            self.assertEqual(result['summary']['harvest_mj'],0.)
            np.testing.assert_array_equal(result['requested_actions'],
                np.full(len(result['requested_actions']),-1. if space=='mode_aware' else 0.))


if __name__=='__main__':unittest.main()
