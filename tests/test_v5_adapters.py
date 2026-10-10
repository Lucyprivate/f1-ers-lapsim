"""Causal limited adapters and exact frozen-mechanics reuse regressions."""
import math
import unittest
from dataclasses import replace
import numpy as np

from f1ers_v3.geometry import ClosedRoute
from f1ers_v4.simulator import SimulatorV4
from f1ers_v4.vehicle import IntegrationState, IntegratorConfig, VehicleParamsV4, WheelCommand
from f1ers_v4.vehicle import vehicle_step as frozen_vehicle_step
from f1ers_v5.adapters import VehicleParamsV5, ReplayDriverV5, GeometryDriverV5
from f1ers_v5.integration import advance_interval, vehicle_step, _query_command
from f1ers_v5.simulator import SimulatorV5


class ConstantGeometry:
    def __init__(self, curvature=0., drag=0., downforce=0.):
        self.curvature, self.drag, self.downforce = curvature, drag, downforce
    def evaluate(self, s):
        return dict(curvature_1pm=self.curvature, drag_area_m2=self.drag, downforce_area_m2=self.downforce)
    def next_space_event(self, s):
        return math.inf


def fixture():
    angle = np.linspace(0, 2*np.pi, 40, endpoint=False)
    route = ClosedRoute(np.column_stack([20*np.cos(angle),20*np.sin(angle)]), kind="spline")
    return route.sample(5), {"soc_window_mj":4., "ers_max_power_kw":350., "harvest_max_mj_per_lap":8.}


def trace(release=.2, final=1., speed=1., power=100.):
    return {"time_s":[0.,release,final], "speed_mps":[speed,speed,speed],
            "throttle":[0.,power,power], "brake":[1.,0.,0.]}


class V5AdaptersTests(unittest.TestCase):
    def setUp(self):
        self.params = VehicleParamsV5(brake_mapping="independent", replay_brake_gain=1.,
            drag_area_m2=0., downforce_area_m2=0., rolling_coefficient=0.)
        self.straight = ConstantGeometry()

    def test_parameter_validation(self):
        for value in (0., .09, 5.01, float("nan")):
            with self.assertRaises(ValueError):
                VehicleParamsV5(brake_mapping="independent", brake_decay_tau_s=value)
        for value in (0., .049, 1.001, float("nan")):
            with self.assertRaises(ValueError):
                VehicleParamsV5(traction_tau_s=value)
        with self.assertRaises(ValueError):
            VehicleParamsV5(brake_mapping="other")
        with self.assertRaises(ValueError):
            VehicleParamsV5(brake_decay_tau_s=1.)

    def test_independent_request_does_not_shrink_with_lateral_usage(self):
        state = IntegrationState(0.,30.)
        demands = []
        for curvature in (0., .012):
            geometry = ConstantGeometry(curvature)
            driver = ReplayDriverV5(trace(speed=30.), self.params, geometry)
            command = _query_command(driver, state, geometry, .01, None)
            _, info = vehicle_step(state, command, geometry, self.params, .01)
            demands.append(info["requested_force_n"])
            self.assertLessEqual(abs(info["executed_force_n"]), info["tire_brake_capacity_n"]+8.)
            if curvature:
                self.assertIn("remaining_joint_tire_budget", info["demand_clip_reasons"])
                self.assertGreater(info["lateral_utilization"],0)
        self.assertEqual(demands[0],demands[1])

    def test_request_uses_current_integrated_speed_not_observation(self):
        params = replace(self.params, downforce_area_m2=3.6)
        geometry = ConstantGeometry(downforce=3.6)
        raw = trace(speed=1.)
        driver = ReplayDriverV5(raw, params, geometry)
        command = driver.query_state(IntegrationState(0.,30.), .005, .15, .01)
        raw["speed_mps"] = [1000.,1000.,1000.]
        other = ReplayDriverV5(raw, params, geometry).query_state(IntegrationState(0.,30.), .005, .15, .01)
        self.assertEqual(command["wheel_force_n"], other["wheel_force_n"])
        expected = -params.mu*(params.mass_kg*9.81+.5*params.air_density_kgpm3*3.6*30**2)
        self.assertAlmostEqual(command["wheel_force_n"],expected,8)

    def test_decay_uses_past_onset_and_not_future_release(self):
        params = replace(self.params, brake_decay_tau_s=.5)
        a = ReplayDriverV5(trace(release=.8,final=2.,speed=30.),params,self.straight)
        b = ReplayDriverV5(trace(release=1.5,final=2.,speed=30.),params,self.straight)
        state = IntegrationState(0.,30.,.2)
        for driver in (a,b):
            command = driver.query_state(state,.21,0.,.02)
            self.assertAlmostEqual(command["wheel_force_n"],-params.mu*params.mass_kg*9.81*math.exp(-.21/.5),8)
        for key in ("wheel_force_n", "requested_force_n", "brake_elapsed_s"):
            self.assertEqual(a.query_state(state,.21,0.,.02)[key],b.query_state(state,.21,0.,.02)[key])

    def test_brake_release_before_stop_is_event(self):
        driver = ReplayDriverV5(trace(release=.025),self.params,self.straight)
        final, info = advance_interval(IntegrationState(0.,1.),driver,None,.5,self.straight,self.params)
        self.assertEqual(info["status"],"accepted")
        self.assertGreater(final.speed_mps,1.)
        for row in info["step_records"]:
            if row["driver_brake_fraction"]:
                self.assertLessEqual(row["time_end_s"],.025+1e-8)
        self.assertTrue(any(row["kind"]=="time" for row in info["events"]))

    def test_actual_stop_before_release_preserves_partial(self):
        driver = ReplayDriverV5(trace(release=.2),self.params,self.straight)
        final, info = advance_interval(IntegrationState(0.,1.),driver,None,5.,self.straight,self.params)
        self.assertEqual(info["status"],"physical_stop")
        self.assertAlmostEqual(final.elapsed_time_s,1/(self.params.mu*9.81),8)
        self.assertLess(final.s_m,5.)

    def test_trace_exhaustion_never_invents_completion(self):
        raw = {"time_s":[0.,.01],"speed_mps":[10.,20.],"throttle":[0.,0.],"brake":[0.,0.]}
        driver = ReplayDriverV5(raw,self.params,self.straight)
        final, info = advance_interval(IntegrationState(0.,10.),driver,None,5.,self.straight,self.params)
        self.assertEqual(info["status"],"physical_fail")
        self.assertIn("driver_trace_exhausted", info["guard_reasons"])
        self.assertLess(final.s_m,5.)

    def test_no_implicit_brake_or_intermediate_speed_reset(self):
        a = {"time_s":[0.,1.],"speed_mps":[25.,10000.],"throttle":[0.,0.],"brake":[0.,0.]}
        driver = ReplayDriverV5(a,self.params,self.straight)
        final, info = advance_interval(IntegrationState(0.,25.),driver,None,1.,self.straight,self.params)
        self.assertEqual(info["status"],"accepted")
        self.assertAlmostEqual(final.speed_mps,25.,7)
        self.assertTrue(all(abs(r["wheel_force_n"])<1e-10 for r in info["step_records"]))

    def test_frozen_root_failure_retained(self):
        driver = ReplayDriverV5(trace(speed=30.),self.params,self.straight)
        state = IntegrationState(0.,30.)
        command = _query_command(driver,state,self.straight,.01,None)
        final, info = vehicle_step(state,command,self.straight,self.params,.01,IntegratorConfig(root_iterations=1))
        self.assertEqual(final,state)
        self.assertEqual(info["status"],"numerical_fail")
        self.assertIn("velocity_root_not_converged",info["guard_reasons"])

    def test_shared_regen_budget_and_mechanical_work(self):
        state = IntegrationState(0.,40.,energy_mj=2.,recharge_mj=0.)
        command = WheelCommand(mode="autonomous",wheel_force_n=-10000.,requested_dc_kw=-350.)
        final, info = vehicle_step(state,command,self.straight,self.params,.02)
        self.assertEqual(info["status"],"accepted")
        self.assertGreater(final.energy_mj,state.energy_mj)
        self.assertAlmostEqual(info["mguk_mechanical_kw"]-info["friction_brake_kw"],info["wheel_power_kw"],9)
        self.assertLess(abs(info["wheel_work_mj"]-info["resistance_work_mj"]-info["kinetic_delta_mj"]),1e-8)

    def test_shared_energy_floor_and_ramp_reserve(self):
        state = IntegrationState(0.,70.,energy_mj=2.01,recharge_mj=0.)
        final, info = vehicle_step(state,WheelCommand(mode="autonomous",power_kw=400.,requested_dc_kw=350.),self.straight,self.params,.02)
        self.assertGreaterEqual(final.energy_mj,2.)
        self.assertGreaterEqual(final.energy_mj-2.+1e-8,info["ramp_reserve_mj"])

    def _traction_driver(self):
        track, rules = fixture()
        params = replace(self.params,traction_tau_s=.5)
        sim = SimulatorV5(track,rules,params)
        n = len(np.linspace(0,sim.length_m,int(np.ceil(sim.length_m/5))+1))-1
        positions = np.linspace(0,sim.length_m,n+1)[:-1]
        profile = dict(s_m=positions,target_acceleration_mps2=np.r_[-5.,np.full(n-1,3.)],
                       power_kw=np.full(n,400.),brake_fraction=np.zeros(n),requested_dc_kw=np.zeros(n))
        return GeometryDriverV5(sim.geometry,params,profile),sim

    def test_traction_is_causal_and_queries_are_pure(self):
        driver, _ = self._traction_driver()
        driver.decide(IntegrationState(0.,10.,0.))
        driver.decide(IntegrationState(driver.decision_s[1],10.,.5))
        state = IntegrationState(driver.decision_s[1],10.,.5)
        a = driver.query_state(state,.51,state.s_m,.02)
        driver.query_state(state,.52,state.s_m,.04)
        b = driver.query_state(state,.51,state.s_m,.02)
        self.assertEqual(a,b)
        self.assertLess(a["traction_response_fraction"],1.)
        self.assertGreater(driver.query_state(state,.8,state.s_m,.6)["traction_response_fraction"],a["traction_response_fraction"])

    def test_traction_cannot_be_bypassed_by_target_acceleration(self):
        driver, sim = self._traction_driver()
        driver.decide(IntegrationState(0.,10.,0.))
        state = IntegrationState(driver.decision_s[1],10.,.5,2.,0.)
        driver.decide(state)
        command = _query_command(driver,state,sim.geometry,.02,None)
        self.assertIsNotNone(command.target_acceleration_mps2)
        self.assertLess(command.wheel_force_n,command.raw_requested_force_n)
        _, info = vehicle_step(state,command,sim.geometry,sim.vehicle,.02)
        self.assertLessEqual(info["executed_force_n"],command.wheel_force_n+1e-5)
        self.assertIn("causal_traction_establishment",info["demand_clip_reasons"])

    def test_negative_autonomous_demand_is_not_filtered(self):
        driver, sim = self._traction_driver()
        state = IntegrationState(0.,10.)
        driver.decide(state)
        command = driver.query_state(state,.01,0.,.02)
        self.assertNotIn("wheel_force_n",command)
        self.assertEqual(command["requested_force_n"],command["raw_requested_force_n"])

    def test_explicit_autonomous_brake_cannot_turn_into_positive_force(self):
        driver, _ = self._traction_driver()
        state = IntegrationState(0.,10.)
        driver.decide(state)
        driver.command.update(target_acceleration_mps2=3., brake_fraction=.5)
        command = driver.query_state(state,.01,0.,.02)
        self.assertLess(command["requested_force_n"],0.)
        self.assertNotIn("wheel_force_n",command)
        self.assertEqual(command["brake_fraction"],.5)

    def test_no_tau_autonomous_uses_kernel_midpoint_demand_without_false_clip(self):
        track,rules = fixture()
        params = replace(self.params,drag_area_m2=1.3,rolling_coefficient=.012)
        sim = SimulatorV5(track,rules,params)
        n = len(np.linspace(0,sim.length_m,int(np.ceil(sim.length_m/5))+1))-1
        profile = dict(s_m=np.linspace(0,sim.length_m,n+1)[:-1],
                       target_acceleration_mps2=np.full(n,3.),power_kw=np.full(n,400.),brake_fraction=np.zeros(n))
        driver = GeometryDriverV5(sim.geometry,params,profile)
        state = IntegrationState(0.,10.,0.,2.,0.)
        driver.decide(state)
        command = _query_command(driver,state,sim.geometry,.02,None)
        self.assertIsNone(command.wheel_force_n)
        actual, info = vehicle_step(state,command,sim.geometry,params,.02)
        frozen, old = frozen_vehicle_step(state,command,sim.geometry,params,.02)
        self.assertEqual(actual,frozen)
        for key,value in old.items():
            self.assertEqual(info[key],value)
        self.assertNotAlmostEqual(command.requested_force_n,params.mass_kg*3.+old["drag_force_n"],5)
        self.assertAlmostEqual(info["requested_force_n"],params.mass_kg*3.+old["drag_force_n"],10)
        self.assertEqual(info["raw_requested_force_n"],info["requested_force_n"])
        self.assertEqual(info["demand_clip_reasons"],[])

    def test_no_tau_autonomous_independent_mapping_preserves_full_trajectory(self):
        track,rules = fixture()
        params = VehicleParamsV5(brake_mapping="independent")
        actual = SimulatorV5(track,rules,params).simulate()
        old = SimulatorV4(track,rules,params).simulate()
        for key in ("speed_mps","speed_next_mps","time_s","wheel_force_n","energy_mj","mguk_dc_kw"):
            np.testing.assert_array_equal(actual[key],old[key])
        self.assertEqual(actual["terminal_time_s"],old["terminal_time_s"])
        self.assertTrue(all(reason in ("remaining_joint_tire_budget","power_or_tractive_force")
                            for row in actual["internal_trace"]["demand_clip_reasons"] for reason in row))

    def test_no_option_baseline_is_exactly_v4(self):
        track, rules = fixture()
        a = SimulatorV4(track,rules,VehicleParamsV4())
        b = SimulatorV5(track,rules,VehicleParamsV5())
        def equal(x,y):
            if isinstance(x,np.ndarray):
                np.testing.assert_array_equal(x,y)
            elif isinstance(x,dict):
                self.assertEqual(x.keys(),y.keys())
                for key in x:equal(x[key],y[key])
            elif isinstance(x,list):
                self.assertEqual(len(x),len(y))
                for i,j in zip(x,y):equal(i,j)
            else:self.assertEqual(x,y)
        equal(a.simulate(),b.simulate())
        raw = trace(release=.025,speed=10.)
        equal(a.simulate("replay",driver_trace=raw),b.simulate("replay",driver_trace=raw))

    def test_output_spacing_and_raw_requests_separate_from_execution(self):
        track,rules = fixture()
        sim = SimulatorV5(track,rules,replace(self.params, traction_tau_s=.5))
        a = sim.simulate(output_ds_m=10.)
        b = sim.simulate(output_ds_m=2.5)
        self.assertEqual(a["terminal_time_s"],b["terminal_time_s"])
        self.assertEqual(a["terminal_speed_mps"],b["terminal_speed_mps"])
        self.assertNotIn("wheel_force_n",a["driver_profile"])
        self.assertIn("target_acceleration_mps2",a["driver_profile"])
        frozen = sim.simulate(controller={"driver_profile":a["driver_profile"],
                                          "raw_requested_dc_kw":a["driver_profile"]["requested_dc_kw"]},
                              initial_speed_mps=a["summary"]["speed_start_mps"])
        np.testing.assert_array_equal(a["driver_profile"]["target_acceleration_mps2"],frozen["driver_profile"]["target_acceleration_mps2"])
        self.assertTrue(frozen["summary"]["raw_driver_requests_preserved"])

    def test_numerical_audit_keeps_raw_profile_and_original_replay(self):
        track,rules = fixture()
        sim = SimulatorV5(track,rules,self.params)
        raw = {"time_s":[0.,20.],"speed_mps":[10.,9999.],"throttle":[0.,0.],"brake":[0.,0.]}
        saved = {k:list(v) for k,v in raw.items()}
        audit = sim.numerical_audit(raw)
        self.assertTrue(audit["passed"],audit)
        self.assertTrue(audit["physical_passed"])
        self.assertEqual(raw,saved)
        self.assertTrue(audit["modes"]["autonomous"]["raw_requests_frozen"])
        self.assertEqual(len(audit["modes"]["autonomous"]["raw_request_sha256"]),64)

    def test_numerical_audit_does_not_drop_partial_replay(self):
        track,rules = fixture()
        sim = SimulatorV5(track,rules,self.params)
        raw = {"time_s":[0.,.01],"speed_mps":[10.,9999.],"throttle":[0.,0.],"brake":[0.,0.]}
        audit = sim.numerical_audit(raw)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["modes"]["replay"]["passed"])
        failed = [item for item in audit["failed_trajectories"] if item["mode"]=="replay"]
        self.assertEqual(len(failed),6)
        self.assertTrue(all(not item["result"]["summary"]["completed_lap"] for item in failed))
        self.assertTrue(all(item["result"]["terminal_s_m"]>0 for item in failed))


if __name__ == "__main__":
    unittest.main()
