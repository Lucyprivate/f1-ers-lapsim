"""Independent mechanics and storage oracles for optional research models."""
import math
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from f1ers_v7._core.integrator import (IntegrationState, IntegratorConfig,
    VehicleParamsV4, WheelCommand, vehicle_step, advance_interval,
    storage_reserve_mj, storage_deployment_bound_kw)
from f1ers_v7.research.physics import (LossAwareShield, sobol_viability, capped_fit,
    segment_boundary_audit, require_calibration, axle_sensitivity_registry, loss_sensitivity_registry,
    BOUNDS)
from f1ers_v7.research.physics import LossAwareSimulator, LOSS_PROTOCOL
from f1ers_v7.research.physics import replay_numerical_audit
from f1ers_v7._core.base_simulator import SimulatorV4
from f1ers_v7._core.geometry import ClosedRoute
from f1ers_v7._core.base_rules import RuleConfig, RuleState


class Geometry:
    def __init__(self, curvature=0., drag=0., downforce=0.):
        self.curvature, self.drag, self.downforce = curvature, drag, downforce
    def evaluate(self, s):
        return dict(curvature_1pm=self.curvature, drag_area_m2=self.drag,
                    downforce_area_m2=self.downforce)


class CommandDriver:
    def __init__(self, command): self.command = command
    def query(self, t, s): return self.command


class OptionalPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.params = VehicleParamsV4(rolling_coefficient=0., drag_area_m2=0.,
            downforce_area_m2=0., replay_brake_gain=.5)
        self.geometry = Geometry()

    def test_legacy_default_golden_transitions(self):
        p, g = VehicleParamsV4(), Geometry(.005,1.15,3.6)
        cases = [
            (IntegrationState(0.,40.), WheelCommand(power_kw=600.),
             40.27032187640988, .8027032187640988,12000.,None),
            (IntegrationState(0.,40.), WheelCommand(brake_fraction=1.),
             39.73870501483465,.7973870501483465,-9279.103363060793,None),
            (IntegrationState(0.,70.,energy_mj=3.,recharge_mj=0.),
             WheelCommand(power_kw=400.,mode='autonomous',requested_dc_kw=200.),
             70.12490742662982,1.4012490742662982,8421.05819493836,2.996)]
        for state,cmd,v,s,force,e in cases:
            out,row = vehicle_step(state,cmd,g,p,.02)
            self.assertEqual(row['status'],'accepted')
            self.assertEqual(out.speed_mps,v)
            self.assertEqual(out.s_m,s)
            self.assertEqual(out.energy_mj,e)
            self.assertEqual(row['wheel_force_n'],force)

    def test_independent_brake_request_is_not_remaining_capability(self):
        state = IntegrationState(0.,30.)
        # Start at 95% lateral use; both models have the same physical budget.
        g = Geometry(.95*self.params.mu*9.81/30.**2)
        rows = [vehicle_step(state,WheelCommand(brake_fraction=1.),g,
            replace(self.params,independent_replay_braking=b),.02)[1] for b in (False,True)]
        self.assertTrue(all(r['status']=='accepted' for r in rows))
        self.assertAlmostEqual(rows[1]['requested_wheel_force_n'],
            -.5*self.params.mu*self.params.mass_kg*9.81,8)
        self.assertGreater(abs(rows[1]['requested_wheel_force_n']),abs(rows[0]['requested_wheel_force_n']))
        self.assertGreater(abs(rows[1]['wheel_force_n']),abs(rows[0]['wheel_force_n']))
        self.assertLessEqual(abs(rows[1]['wheel_force_n']),rows[1]['tire_brake_capacity_n']+1e-8)
        self.assertFalse(rows[1]['ers_validated'])

    def test_rear_drive_uses_rear_capacity_and_transfer_sign(self):
        p = replace(self.params,axle_budget_enabled=True)
        out,r = vehicle_step(IntegrationState(0.,20.),WheelCommand(power_kw=700.),self.geometry,p,.02)
        self.assertEqual(r['status'],'accepted')
        # Independent closed-form rear-drive traction at static lateral zero:
        # F = mu*(rear_static*mg + (F/m)*m*h/L).
        expected = p.mu*p.rear_static_fraction*p.mass_kg*9.81/(1-p.mu*p.height_over_wheelbase)
        self.assertAlmostEqual(r['wheel_force_n'],expected,5)
        self.assertEqual(r['front_wheel_force_n'],0.)
        self.assertAlmostEqual(r['rear_wheel_force_n'],r['wheel_force_n'],9)
        self.assertAlmostEqual(r['rear_normal_load_n']+r['front_normal_load_n'],p.mass_kg*9.81,8)
        self.assertGreater(r['rear_normal_load_n'],r['front_normal_load_n'])
        self.assertGreater(out.speed_mps,20.)

    def test_rear_regeneration_and_friction_share_one_axle_budget(self):
        p = replace(self.params,axle_budget_enabled=True,mu=1.2)
        cmd = WheelCommand(mode='autonomous',wheel_force_n=-10000.,requested_dc_kw=-350.)
        state = IntegrationState(0.,40.,energy_mj=2.,recharge_mj=0.)
        out,r = vehicle_step(state,cmd,self.geometry,p,.02)
        self.assertEqual(r['status'],'accepted')
        self.assertLess(r['rear_normal_load_n'],r['front_normal_load_n'])
        self.assertAlmostEqual(r['front_wheel_force_n']+r['rear_wheel_force_n'],r['wheel_force_n'],8)
        rear_sum = r['rear_regeneration_force_n']+r['rear_friction_brake_force_n']
        self.assertLessEqual(rear_sum,r['rear_longitudinal_capacity_n']+1e-8)
        self.assertLess(abs(r['dc_kw']),350.)
        self.assertAlmostEqual(r['mguk_mechanical_kw']-r['friction_brake_kw'],r['wheel_power_kw'],8)
        self.assertGreater(out.energy_mj,2.)

    def test_invalid_negative_load_is_not_certified_by_clamping(self):
        p = replace(self.params,axle_budget_enabled=True,height_over_wheelbase=1.,mu=2.4)
        state = IntegrationState(0.,40.)
        out,r = vehicle_step(state,WheelCommand(wheel_force_n=-10000.),self.geometry,p,.02)
        self.assertEqual(out,state)
        self.assertEqual(r['status'],'physical_fail')
        self.assertIn('negative_axle_normal_load',r['guard_reasons'])

    def test_dc_counter_and_storage_charge_loss_are_distinct(self):
        p = replace(self.params,storage_charge_efficiency=.95,auxiliary_power_w=100.)
        state = IntegrationState(0.,40.,energy_mj=2.,recharge_mj=0.)
        out,r = vehicle_step(state,WheelCommand(mode='autonomous',wheel_force_n=-10000.,
            requested_dc_kw=-100.),self.geometry,p,.02)
        self.assertEqual(r['status'],'accepted')
        self.assertAlmostEqual(out.recharge_mj,.002,12)
        self.assertAlmostEqual(out.energy_mj-2.,.002*.95-.000002,12)
        self.assertAlmostEqual(r['storage_loss_mj'],.0001,12)
        self.assertAlmostEqual(r['storage_ledger_residual_mj'],0.,12)
        self.assertAlmostEqual(r['auxiliary_work_mj'],.000002,12)

    def test_storage_discharge_and_auxiliary_loss_oracle(self):
        p = replace(self.params,storage_discharge_efficiency=.98,auxiliary_power_w=50.)
        state = IntegrationState(0.,70.,energy_mj=3.,recharge_mj=0.)
        out,r = vehicle_step(state,WheelCommand(mode='autonomous',power_kw=400.,
            requested_dc_kw=100.),self.geometry,p,.02)
        self.assertEqual(r['status'],'accepted')
        self.assertAlmostEqual(3.-out.energy_mj,.002/.98+.000001,12)
        self.assertEqual(out.recharge_mj,0.)
        self.assertLess(abs(r['storage_ledger_residual_mj']),1e-12)

    def test_loss_cannot_expand_dc_lap_recharge_allowance(self):
        p = replace(self.params,storage_charge_efficiency=.95)
        state = IntegrationState(0.,40.,energy_mj=3.,recharge_mj=7.999)
        out,r = vehicle_step(state,WheelCommand(mode='autonomous',wheel_force_n=-10000.,
            requested_dc_kw=-350.),self.geometry,p,.02)
        self.assertEqual(r['status'],'accepted')
        self.assertAlmostEqual(out.recharge_mj,8.,12)
        self.assertAlmostEqual(out.energy_mj-3.,.001*.95,12)

    def test_reserve_matches_hold_and_ramp_integration(self):
        p,hold,ramp,eta,aux = 130.,.75,100.,.95,100.
        expected = (p*hold+.5*p*p/ramp)/eta/1000.+aux*(hold+p/ramp)/1e6
        reserve = storage_reserve_mj(p,hold,ramp,eta,aux)
        self.assertAlmostEqual(reserve,expected,13)
        self.assertAlmostEqual(storage_deployment_bound_kw(reserve,hold,ramp,eta,aux),p,10)
        self.assertLess(storage_deployment_bound_kw(reserve,hold,ramp,eta,aux),
                        storage_deployment_bound_kw(reserve,hold,ramp))

    def test_cold_aux_floor_event_precedes_future_geometry_failure(self):
        class BadFuture(Geometry):
            def evaluate(self,s):
                return dict(curvature_1pm=0. if s<.1 else .1,drag_area_m2=0.,downforce_area_m2=0.)
        p = replace(self.params,auxiliary_power_w=50.)
        state = IntegrationState(0.,40.,energy_mj=2.,recharge_mj=0.)
        cmd = WheelCommand(mode='autonomous',power_kw=400.)
        out,r = vehicle_step(state,cmd,BadFuture(),p,.02)
        self.assertEqual(out,state)
        self.assertEqual(r['status'],'physical_fail')
        self.assertEqual(r['guard_reasons'],['energy_floor_event'])
        self.assertEqual(r['dt_s'],0.)
        self.assertEqual(r['ds_m'],0.)
        self.assertTrue(r['step_certified'])
        final,report = advance_interval(state,CommandDriver(cmd),None,1.,self.geometry,p)
        self.assertEqual(final,state)
        self.assertEqual(report['accepted_steps'],0)
        self.assertEqual(report['events'][0]['kind'],'energy_floor')

    def test_floor_event_is_located_before_stop_and_prefix_preserved(self):
        p = replace(self.params,auxiliary_power_w=100.,replay_brake_gain=1.)
        state = IntegrationState(0.,1.,energy_mj=2.000001,recharge_mj=0.)
        cmd = WheelCommand(mode='autonomous',wheel_force_n=-p.mass_kg*p.mu*9.81)
        out,r = vehicle_step(state,cmd,self.geometry,p,.2)
        self.assertEqual(r['guard_reasons'],['energy_floor_event'])
        self.assertTrue(r['accepted_prefix'])
        self.assertAlmostEqual(out.elapsed_time_s,.01,7)
        self.assertGreater(out.speed_mps,0.)
        self.assertAlmostEqual(out.energy_mj,2.,11)
        final,report = advance_interval(state,CommandDriver(cmd),None,1.,self.geometry,p)
        self.assertEqual(report['status'],'physical_fail')
        self.assertAlmostEqual(final.elapsed_time_s,.01,7)
        self.assertGreater(report['accepted_steps'],0)
        self.assertAlmostEqual(report['step_records'][-1]['s_end_m'],final.s_m,10)

    def test_physical_stop_precedes_later_floor(self):
        p = replace(self.params,auxiliary_power_w=100.)
        state = IntegrationState(0.,1.,energy_mj=2.0001,recharge_mj=0.)
        out,r = vehicle_step(state,WheelCommand(mode='autonomous',wheel_force_n=-20000.),self.geometry,p,.2)
        self.assertEqual(r['status'],'physical_stop')
        self.assertFalse(r['energy_floor_event'])
        self.assertGreater(out.energy_mj,2.)


class ResearchPhysicsProtocolTests(unittest.TestCase):
    def setUp(self):
        self.params = VehicleParamsV4(rolling_coefficient=0.,drag_area_m2=0.,downforce_area_m2=0.)
        self.geometry = Geometry()
    def test_segment_projection_and_reversed_width_labels(self):
        centre = np.array([[0.,0.,4.,6.],[100.,0.,4.,6.],
                           [100.,100.,4.,6.],[0.,100.,4.,6.]])
        xy = np.array([[50.,1.],[50.,-2.]])
        forward = segment_boundary_audit(xy,centre,1)
        backward = segment_boundary_audit(xy,centre,-1)
        self.assertEqual(forward['signed_offset_m'],[1.,-2.])
        self.assertEqual(forward['left_clearance_m'],[5.,8.])
        self.assertEqual(forward['right_clearance_m'],[5.,2.])
        self.assertEqual(backward['left_clearance_m'],forward['right_clearance_m'])
        self.assertEqual(backward['right_clearance_m'],forward['left_clearance_m'])
        self.assertEqual(forward['minimum_boundary_clearance_m'],backward['minimum_boundary_clearance_m'])
        outside = segment_boundary_audit([[50.,8.]],centre)
        self.assertEqual(outside['outside_samples'],1)
        self.assertEqual(outside['minimum_left_clearance_m'],-2.)

    def test_fit_views_reject_development_ids_and_failed_quality(self):
        good = {'audit':{'role':'calibration','data_gate_passed':True},
            'laps':[{'role':'calibration','lap_number':n} for n in (3,4,5,6)]}
        self.assertIs(require_calibration(good),good)
        for view in ({**good,'laps':[{'role':'calibration','lap_number':n} for n in (3,4,7,9)]},
                     {**good,'audit':{'role':'calibration','data_gate_passed':False}},
                     {**good,'laps':[{'role':'development','lap_number':n} for n in (3,4,5,6)]}):
            with self.assertRaises(ValueError): require_calibration(view)

    def test_fixed_sobol_screen_keeps_partial_failures_and_stops_fit(self):
        calls=[]
        def evaluator(x,key):
            calls.append((key,x.copy()))
            # Excellent residual cannot promote an incomplete/failed rollout.
            return {'complete':len(calls)<=2,'physical':len(calls)<=2,'objective':0.,
                    'failure_position_m':12. if len(calls)>2 else None}
        report = sobol_viability(evaluator)
        self.assertEqual(len(calls),64)
        self.assertEqual(report['viable_points'],2)
        self.assertFalse(report['fit_allowed'])
        self.assertEqual(len(report['records']),64)
        self.assertEqual(report['records'][63]['failure_position_m'],12.)
        fitted = capped_fit(lambda *args:self.fail('Fit ran despite fewer than three seeds'),report)
        self.assertEqual(fitted['evaluations'],0)
        self.assertIsNone(fitted['selected_parameters'])
        again=[]
        sobol_viability(lambda x,key:(again.append(x.copy()) or
            {'complete':False,'physical':False,'objective':None}))
        np.testing.assert_array_equal(np.stack(again),np.stack([x for _,x in calls]))
        with self.assertRaises(ValueError): sobol_viability(evaluator,seed=2027)

    def test_real_optimizer_obeys_three_independent_hard_call_budgets(self):
        center = BOUNDS.mean(axis=1)
        scale = np.diff(BOUNDS,axis=1).ravel()
        def oracle(x,key):
            return {'complete':True,'physical':True,'objective':float(np.sum(((x-center)/scale)**2))}
        screening = sobol_viability(oracle)
        calls=[]
        def recorded(x,key):
            calls.append(key)
            self.assertTrue(np.all(x>=BOUNDS[:,0]) and np.all(x<=BOUNDS[:,1]))
            return oracle(x,key)
        fitted = capped_fit(recorded,screening,max_iterations=300,max_evaluations=8)
        self.assertEqual(fitted['evaluations'],24)
        self.assertEqual([r['evaluations'] for r in fitted['starts']],[8,8,8])
        self.assertEqual(len(set(calls)),24)
        self.assertFalse(fitted['converged'])
        self.assertIsNone(fitted['selected_parameters'])
        self.assertIsNotNone(fitted['best_diagnostic_parameters'])
        with self.assertRaises(ValueError): capped_fit(recorded,screening,max_evaluations=601)

    def test_registries_do_not_fit_axles_or_losses(self):
        axles,losses = axle_sensitivity_registry(),loss_sensitivity_registry()
        self.assertEqual(len(axles),9)
        self.assertEqual(len(losses),9)
        self.assertEqual(axles[0]['rear_static_fraction'],.5)
        self.assertEqual(len({tuple(r[k] for k in ('rear_static_fraction','rear_aero_fraction','height_over_wheelbase')) for r in axles}),9)
        self.assertTrue(all(r['storage_charge_efficiency']==r['storage_discharge_efficiency'] for r in losses))

    def test_loss_aware_committed_reference_intersection_is_explicit(self):
        geometry = SimpleNamespace(length_m=4326.)
        grid = {'s_m':np.arange(4326.),'safe_speed_mps':np.full(4326,90.),'ds_m':1.,
                'positive_low_mask':np.zeros(4326,bool),'hazard_s_m':np.array([]),'length_m':4326.}
        shield = LossAwareShield({'ers_max_power_kw':350.,'harvest_max_mj_per_lap':8.},
            geometry,RuleConfig(tail_guard_m=0.),grid,1.,energy_notice_s=.5,
            storage_charge_efficiency=.95,storage_discharge_efficiency=.95,auxiliary_power_w=50.)
        state = IntegrationState(100.,70.,0.,2.01,0.,320.)
        carry = RuleState(phase='HOLD',hold_until_s=1.,reference_mechanical_kw=304.,
            previous_dc_kw=320.,requested_dc_kw=320.,target_reduction_mechanical_kw=304.)
        shield.reset(state,carry)
        q = shield.query(state,{'wheel_force_n':10000.},geometry,.02)
        self.assertFalse(q['constraint_intersection_feasible'])
        self.assertTrue(q['no_feasible_positive_DC_root'])
        self.assertGreater(q['request_proxy_shortfall_dc_kw'],0.)
        self.assertEqual(shield.state.previous_dc_kw,320.)
        self.assertEqual(shield.state.reference_mechanical_kw,304.)
        reserve = shield._reserve(100.,.5)
        self.assertGreater(reserve,.1*.5+.05)
        self.assertEqual(shield.certificate()['storage_model']['lap_recharge_counter'],'DC-side')

    def test_real_loss_environment_keeps_cold_event_and_decomposed_ledger(self):
        theta = np.linspace(0.,2*np.pi,40,endpoint=False)
        route = ClosedRoute(np.c_[20*np.cos(theta),20*np.sin(theta)])
        rules = {'ers_max_power_kw':350.,'harvest_max_mj_per_lap':8.,'soc_window_mj':4.,
            'alternative_power_zones_route_m':[],'power_reduction_exceptions_route':[],
            'fia_centreline_length_m':route.source_length,'fia_sector_lengths_m':[route.source_length/3]*3}
        params = replace(self.params,storage_discharge_efficiency=.95,
            storage_charge_efficiency=.95,auxiliary_power_w=50.)
        sim = LossAwareSimulator(route.sample(5.),rules,params,
            rule_config=RuleConfig(tail_guard_m=1000.),margin_s=1.,energy_notice_s=.5)
        cold = sim.simulate(lambda observation:0.,initial_speed_mps=10.)
        self.assertEqual(cold['summary']['protocol'],LOSS_PROTOCOL)
        self.assertFalse(cold['summary']['completed_lap'])
        self.assertEqual(cold['terminal_s_m'],0.)
        self.assertEqual(cold['terminal_time_s'],0.)
        self.assertIn('energy_floor_event',cold['failure_proposal']['guard_reasons'])
        env = sim.create_env(initial_speed_mps=10.,initial_energy_mj=3.)
        env.step(0.)
        result = env.result()
        summary = result['summary']
        self.assertGreater(summary['auxiliary_work_mj'],0.)
        self.assertGreater(abs(summary['dc_only_balance_residual_mj']),0.)
        self.assertLess(abs(summary['storage_balance_residual_mj']),1e-10)
        self.assertFalse(summary['actual_SOC_verified'])
        self.assertFalse(summary['training_uses_this_loss_model'])

    def test_replay_numerical_audit_keeps_raw_inputs_and_never_calls_autonomy(self):
        theta = np.linspace(0.,2*np.pi,40,endpoint=False)
        route = ClosedRoute(np.c_[10*np.cos(theta),10*np.sin(theta)])
        rules = {'soc_window_mj':4.}
        class ReplayOnly(SimulatorV4):
            calls=[]
            def simulate(self,mode='autonomous',**kwargs):
                if mode!='replay': raise AssertionError('Autonomy cannot certify replay convergence')
                self.calls.append(mode)
                return super().simulate(mode=mode,**kwargs)
        sim = ReplayOnly(route.sample(5.),rules,self.params)
        trace = {'time_s':np.array([0.,20.]),'speed_mps':np.array([5.,5.]),
                 'throttle':np.zeros(2),'brake':np.zeros(2)}
        original = {key:value.copy() for key,value in trace.items()}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            output = Path(directory)/'audit'
            report = replay_numerical_audit(sim,trace,output)
            self.assertTrue(report['passed'])
            self.assertEqual(sim.calls,['replay']*6)
            self.assertEqual(len(list(output.glob('*.json.gz'))),6)
            self.assertFalse(report['autonomous_driver_invoked'])
            self.assertEqual(report['initial_speed_mps'],5.)
            with self.assertRaises(FileExistsError): replay_numerical_audit(sim,trace,output)
        for key in trace: np.testing.assert_array_equal(trace[key],original[key])

    def test_cold_regeneration_can_pay_auxiliary_load(self):
        p = replace(self.params,storage_charge_efficiency=.95,auxiliary_power_w=100.)
        out,r = vehicle_step(IntegrationState(0.,40.,energy_mj=2.,recharge_mj=0.),
            WheelCommand(mode='autonomous',wheel_force_n=-10000.,requested_dc_kw=-100.),self.geometry,p,.02)
        self.assertEqual(r['status'],'accepted')
        self.assertGreater(out.energy_mj,2.)
        self.assertFalse(r['energy_floor_event'])


if __name__ == '__main__': unittest.main()
