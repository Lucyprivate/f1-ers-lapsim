"""Driving-path and output/integration separation regressions."""
import unittest
import numpy as np
from f1ers_v3.geometry import ClosedRoute
from f1ers_v4.simulator import (SimulatorV4, ReplayDriver, ContinuousGeometry,
                               GeometryDriver, ERSRequests, _sample_states)
from f1ers_v4.vehicle import IntegrationState, VehicleParamsV4


def fixture():
    angle=np.linspace(0,2*np.pi,40,endpoint=False)
    route=ClosedRoute(np.column_stack([20*np.cos(angle),20*np.sin(angle)]),kind="spline")
    return route.sample(5), {"soc_window_mj":4.,"ers_max_power_kw":350.,"harvest_max_mj_per_lap":8.}


class TestV4Simulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track,cls.rules=fixture()
        cls.params=VehicleParamsV4()
        cls.sim=SimulatorV4(cls.track,cls.rules,cls.params)
        cls.auto=cls.sim.simulate()

    def test_autonomous_complete_joint_budget_and_periodic(self):
        result=self.auto
        self.assertTrue(result["summary"]["feasible"],result["summary"])
        self.assertTrue(result["summary"]["execution_feasible"])
        self.assertLessEqual(result["summary"]["max_joint_tire_force_excess_n"],8)
        self.assertLess(result["summary"]["mechanical_work_max_segment_residual_mj"],1e-8)
        self.assertLessEqual(result["summary"]["periodic_speed_residual_mps"],.02)

    def test_replay_uses_initial_speed_only_and_preserves_failure(self):
        trace={"time_s":np.array([0.,.02,.04]),"speed_mps":np.array([10.,12.,11.]),
               "throttle":np.array([0.,.3,.5]),"brake":np.array([0.,0.,0.])}
        a=self.sim.simulate("replay",driver_trace=trace)
        trace["speed_mps"][1:]=10000
        b=self.sim.simulate("replay",driver_trace=trace)
        self.assertEqual(a["terminal_time_s"],b["terminal_time_s"])
        self.assertEqual(a["terminal_speed_mps"],b["terminal_speed_mps"])
        self.assertFalse(a["summary"]["completed_lap"])
        self.assertEqual(a["summary"]["intermediate_speed_resets"],0)
        self.assertFalse(a["summary"]["future_speed_caps_used"])
        self.assertGreater(a["terminal_s_m"],0)
        self.assertGreater(len(a["internal_trace"]["s_m"]),1)
        self.assertFalse(a["ers_validated"])
        for key in ("energy_mj","soc_fraction","mguk_dc_kw","mguk_mechanical_kw"):
            self.assertIsNone(a[key])
        self.assertIn("driver_trace_exhausted",a["summary"]["guard_failures"][0]["guard_reasons"])

    def test_raw_throttle_linear_brake_previous_hold(self):
        driver=ReplayDriver({"time_s":[0,.1,.2],"speed_mps":[10,10,10],
                             "throttle":[0,100,50],"brake":[1,0,1]},self.params,self.sim.geometry)
        self.assertEqual(driver.query(.05,0)["power_kw"],300)
        self.assertEqual(driver.query(.09999,0)["brake_fraction"],1)
        self.assertEqual(driver.query(.1,0)["brake_fraction"],0)
        self.assertEqual(driver.next_time_event(0),.1)

    def test_invalid_replay_channels_rejected(self):
        for brake in ([0,.5],[0,float("nan")]):
            with self.assertRaises(ValueError):
                ReplayDriver({"time_s":[0,1],"speed_mps":[10,10],"throttle":[0,0],"brake":brake},self.params,self.sim.geometry)
        with self.assertRaises(ValueError):
            self.sim.simulate("replay",driver_trace={},controller=[0])
        with self.assertRaises(ValueError):
            self.sim.simulate("autonomous",driver_trace={})

    def test_output_spacing_does_not_change_execution(self):
        a=self.sim.simulate(output_ds_m=10)
        b=self.sim.simulate(output_ds_m=2.5)
        self.assertEqual(a["terminal_time_s"],b["terminal_time_s"])
        self.assertEqual(a["terminal_speed_mps"],b["terminal_speed_mps"])
        self.assertEqual(a["summary"]["internal_steps"],b["summary"]["internal_steps"])
        self.assertNotEqual(len(a["s_m"]),len(b["s_m"]))
        self.assertEqual(len(a["time_s"]),len(a["s_m"])+1)

    def test_audit_freezes_original_requests(self):
        profile=self.auto["driver_profile"]
        self.assertIn("target_acceleration_mps2",profile)
        self.assertNotIn("wheel_force_n",profile)
        frozen=self.sim.simulate(controller={"driver_profile":profile,"raw_requested_dc_kw":profile["requested_dc_kw"]},
                                 initial_speed_mps=self.auto["summary"]["speed_start_mps"])
        self.assertTrue(frozen["summary"]["raw_requests_fixed"])
        np.testing.assert_array_equal(profile["target_acceleration_mps2"],frozen["driver_profile"]["target_acceleration_mps2"])
        audit=self.sim.numerical_audit()
        self.assertTrue(audit["modes"]["autonomous"]["raw_requests_frozen"])
        self.assertTrue(audit["passed"],audit)
        self.assertTrue(audit["physical_passed"])

    def test_geometry_closed_and_rule_coordinates(self):
        geometry=self.sim.geometry
        a,b=geometry.evaluate(0),geometry.evaluate(geometry.length_m)
        for key in ("x_m","y_m","curvature_1pm"):
            self.assertAlmostEqual(float(a[key]),float(b[key]),places=10)
        rules={**self.rules,"alternative_power_zones_route_m":[[10,20]]}
        requests=ERSRequests(rules,geometry)
        self.assertEqual(requests.cap(15,50),250)
        self.assertEqual(requests.cap(25,50),350)
        rule=requests.query(IntegrationState(15,90,0,2.5,0,0),{},geometry,.01)
        self.assertEqual(rule["alternative_threshold_kph"],310)
        self.assertEqual(rule["alternative_cap_kw"],250)

    def test_braking_output_uses_quadratic_position_and_time_energy(self):
        # v=20-5t, s=20t-2.5t². At s=17.5 m the exact state is t=1,v=15;
        # linear interpolation in distance would incorrectly give t=7/6,v=85/6.
        times,speeds,ids,fractions=_sample_states([0,17.5,30],[0,30],[0,2],[20,10])
        np.testing.assert_allclose(times,[0,1,2],atol=1e-12)
        np.testing.assert_allclose(speeds,[20,15,10],atol=1e-12)
        energy=np.array([2.5,2.3])
        sampled=energy[ids]+fractions*(energy[ids+1]-energy[ids])
        np.testing.assert_allclose(sampled,[2.5,2.4,2.3],atol=1e-12)


if __name__=="__main__":unittest.main()
