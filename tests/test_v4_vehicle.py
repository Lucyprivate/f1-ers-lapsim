"""Regression cases for defects found in the frozen v3 mechanics."""
import math
import unittest
import numpy as np
from f1ers_v4.vehicle import (IntegrationState, IntegratorConfig, VehicleParamsV4,
                             WheelCommand, vehicle_step, advance_interval)


class ConstantGeometry:
    def __init__(self,curvature=0.,drag=0.,downforce=0.):
        self.curvature,self.drag,self.downforce=curvature,drag,downforce

    def evaluate(self,s):
        return dict(curvature_1pm=self.curvature,drag_area_m2=self.drag,downforce_area_m2=self.downforce)


class SwitchDriver:
    def __init__(self,release=math.inf,power=0.,brake=1.):
        self.release,self.power,self.brake=release,power,brake

    def query(self,t,s):
        return dict(power_kw=self.power if t>=self.release else 0.,
                    brake_fraction=self.brake if t<self.release else 0.,mode="replay")

    def next_time_event(self,t):
        return self.release if t<self.release-1e-10 else math.inf


class VehicleV4Tests(unittest.TestCase):
    def setUp(self):
        self.params=VehicleParamsV4(rolling_coefficient=0.,drag_area_m2=0.,downforce_area_m2=0.,replay_brake_gain=1.)
        self.straight=ConstantGeometry()

    def test_midpoint_conserves_mechanical_work(self):
        state=IntegrationState(0.,30.)
        result,info=vehicle_step(state,WheelCommand(power_kw=732.5),self.straight,self.params,.02)
        self.assertEqual(info["status"],"accepted")
        self.assertGreater(result.speed_mps,state.speed_mps)
        self.assertLess(abs(info["wheel_work_mj"]-info["resistance_work_mj"]-info["kinetic_delta_mj"]),1e-8)

    def test_endpoint_joint_budget_clips_drive_force(self):
        # An accelerating start lies within the lateral limit, while the
        # endpoint has less longitudinal room. v3 certified this incorrectly.
        geometry=ConstantGeometry(curvature=.012)
        state=IntegrationState(0.,35.)
        result,info=vehicle_step(state,WheelCommand(power_kw=732.5),geometry,self.params,.2)
        self.assertEqual(info["status"],"accepted")
        for speed in (state.speed_mps,.5*(state.speed_mps+result.speed_mps),result.speed_mps):
            fy=self.params.mass_kg*speed**2*.012
            total=self.params.mass_kg*self.params.mu*9.81
            self.assertLessEqual(math.hypot(info["wheel_force_n"],fy),total+8.)

    def test_midpoint_joint_budget_with_curvature_peak(self):
        class Peak:
            def evaluate(self,s):
                return dict(curvature_1pm=.015*math.exp(-((s-.5)/.15)**2),drag_area_m2=0.,downforce_area_m2=0.)
        result,info=vehicle_step(IntegrationState(0.,30.),WheelCommand(power_kw=732.5),Peak(),self.params,1/30)
        self.assertEqual(info["status"],"accepted")
        self.assertLessEqual(info["combined_tire_force_excess_n"],8.)

    def test_infeasible_start_retains_failure_proposal(self):
        state=IntegrationState(0.,50.)
        result,info=vehicle_step(state,WheelCommand(power_kw=0.),ConstantGeometry(.02),self.params,.02)
        self.assertEqual(result,state)
        self.assertEqual(info["status"],"physical_fail")
        self.assertIn("combined_tire_force",info["guard_reasons"])
        self.assertIn("proposed_state",info)

    def test_stop_is_actual_partial_time_and_distance(self):
        state=IntegrationState(0.,1.)
        result,info=vehicle_step(state,WheelCommand(brake_fraction=1.),self.straight,self.params,.2)
        expected_time=1./(self.params.mu*9.81)
        self.assertEqual(info["status"],"physical_stop")
        self.assertAlmostEqual(result.elapsed_time_s,expected_time,8)
        self.assertAlmostEqual(result.s_m,.5*expected_time,8)
        self.assertEqual(result.speed_mps,0.)
        self.assertLess(abs(info["mechanical_work_residual_mj"]),1e-8)

    def test_release_before_stop_never_holds_brake_across_event(self):
        state=IntegrationState(0.,1.)
        final,info=advance_interval(state,SwitchDriver(.025,power=350.),None,.5,self.straight,self.params)
        self.assertEqual(info["status"],"accepted")
        self.assertGreater(final.speed_mps,1.)
        for row in info["step_records"]:
            if row["driver_brake_fraction"]:
                self.assertLessEqual(row["time_end_s"],.025+1e-8)
        self.assertTrue(any(event["kind"]=="time" for event in info["events"]))

    def test_stop_before_release_preserves_shortened_path(self):
        final,info=advance_interval(IntegrationState(0.,1.),SwitchDriver(.2,power=350.),None,5.,self.straight,self.params)
        self.assertEqual(info["status"],"physical_stop")
        self.assertLess(final.elapsed_time_s,.2)
        self.assertLess(final.s_m,5.)

    def test_no_hidden_braking_no_speed_reset(self):
        state=IntegrationState(0.,25.)
        final,info=advance_interval(state,SwitchDriver(0.,power=0.),None,1.,self.straight,self.params)
        self.assertEqual(info["status"],"accepted")
        self.assertAlmostEqual(final.speed_mps,25.,7)
        self.assertTrue(all(abs(row["wheel_force_n"])<1e-10 for row in info["step_records"]))

    def test_brake_gain_changes_demand_not_capacity(self):
        state=IntegrationState(0.,30.)
        p=VehicleParamsV4.from_dict({**self.params.to_dict(),"replay_brake_gain":.3})
        _,info=vehicle_step(state,WheelCommand(brake_fraction=1.),self.straight,p,.02)
        self.assertAlmostEqual(-info["wheel_force_n"]/info["tire_brake_capacity_n"],.3,10)

    def test_regen_and_friction_share_signed_wheel_force(self):
        state=IntegrationState(0.,40.,energy_mj=2.,recharge_mj=0.)
        command=WheelCommand(brake_fraction=.8,mode="autonomous",requested_dc_kw=-350.)
        result,info=vehicle_step(state,command,self.straight,self.params,.02)
        self.assertEqual(info["status"],"accepted")
        self.assertGreater(result.energy_mj,state.energy_mj)
        self.assertAlmostEqual(info["mguk_mechanical_kw"]-info["friction_brake_kw"],info["wheel_power_kw"],9)
        self.assertLessEqual(abs(info["wheel_force_n"]),info["tire_brake_capacity_n"]+8.)

    def test_floor_and_ramp_reserve_restrict_deployment(self):
        command=WheelCommand(power_kw=400.,mode="autonomous",requested_dc_kw=350.)
        final,info=vehicle_step(IntegrationState(0.,70.,energy_mj=2.01,recharge_mj=0.),command,self.straight,self.params,.02)
        self.assertEqual(info["status"],"accepted")
        self.assertGreaterEqual(final.energy_mj,2.)
        self.assertGreaterEqual(final.energy_mj-2.+1e-8,info["ramp_reserve_mj"])
        final_floor,floor_info=vehicle_step(IntegrationState(0.,70.,energy_mj=2.,recharge_mj=0.),command,self.straight,self.params,.02)
        self.assertEqual(floor_info["dc_kw"],0.)
        self.assertEqual(final_floor.energy_mj,2.)

    def test_regen_capacity_and_lap_budget(self):
        command=WheelCommand(brake_fraction=1.,mode="autonomous",requested_dc_kw=-350.,harvest_limit_mj=8.5)
        _,info=vehicle_step(IntegrationState(0.,40.,energy_mj=4.,recharge_mj=0.),command,self.straight,self.params,.02)
        self.assertEqual(info["dc_kw"],0.)
        _,info=vehicle_step(IntegrationState(0.,40.,energy_mj=3.,recharge_mj=8.5),command,self.straight,self.params,.02)
        self.assertEqual(info["dc_kw"],0.)

    def test_zero_or_harvest_request_continues_required_down_ramp(self):
        state=IntegrationState(0.,70.,energy_mj=3.,recharge_mj=0.,previous_dc_kw=200.)
        for requested in (0.,-350.):
            command=WheelCommand(power_kw=400.,mode="autonomous",requested_dc_kw=requested)
            final,info=vehicle_step(state,command,self.straight,self.params,.02)
            self.assertEqual(info["status"],"accepted")
            self.assertAlmostEqual(info["dc_kw"],198.,8)
            self.assertEqual(final.recharge_mj,0.)

    def test_low_speed_superclip_respects_ice_force_limit(self):
        command=WheelCommand(power_kw=400.,mode="autonomous",requested_dc_kw=-350.)
        _,info=vehicle_step(IntegrationState(0.,10.,energy_mj=2.,recharge_mj=0.),command,
            self.straight,self.params,.02)
        self.assertEqual(info["status"],"accepted")
        engine_force_power=self.params.ice_max_tractive_force_n*.5*(10.+info["speed_next_mps"])/1000.
        self.assertLessEqual(info["ice_mechanical_kw"],engine_force_power+1e-3)

    def test_bad_root_budget_returns_numerical_failure(self):
        state=IntegrationState(0.,30.)
        final,info=vehicle_step(state,WheelCommand(power_kw=400.),self.straight,self.params,.02,
            IntegratorConfig(root_iterations=1))
        self.assertEqual(final,state)
        self.assertEqual(info["status"],"numerical_fail")
        self.assertIn("velocity_root_not_converged",info["guard_reasons"])

    def test_alternative_zone_cap_checks_endpoint_speed(self):
        p=VehicleParamsV4.from_dict({**self.params.to_dict(),"drag_area_m2":2.})
        geometry=ConstantGeometry(drag=2.)
        state=IntegrationState(0.,86.15,energy_mj=3.,recharge_mj=0.)
        command=WheelCommand(power_kw=400.,mode="autonomous",target_acceleration_mps2=-1.,
            requested_dc_kw=350.,deployment_cap_kw=350.,alternative_threshold_kph=310.,alternative_cap_kw=250.)
        final,info=vehicle_step(state,command,geometry,p,.2)
        self.assertEqual(info["status"],"accepted")
        self.assertGreater(state.speed_mps*3.6,310.)
        self.assertLess(final.speed_mps*3.6,310.)
        self.assertLessEqual(info["dc_kw"],250.)

    def test_autonomous_input_cannot_increase_ice_source(self):
        command=WheelCommand(power_kw=10000.,mode="autonomous")
        _,info=vehicle_step(IntegrationState(0.,40.,energy_mj=2.,recharge_mj=0.),command,
            self.straight,self.params,.02)
        self.assertEqual(info["status"],"accepted")
        self.assertLessEqual(info["ice_mechanical_kw"],self.params.ice_power_kw+1e-3)

    def test_trace_exhaustion_does_not_extrapolate_input(self):
        driver=SwitchDriver(0.,power=350.)
        driver.final_time_s=.02
        final,info=advance_interval(IntegrationState(0.,30.),driver,None,5.,self.straight,self.params)
        self.assertEqual(info["status"],"physical_fail")
        self.assertEqual(info["guard_reasons"],["driver_trace_exhausted"])
        self.assertLessEqual(final.elapsed_time_s,.02+1e-8)

    def test_replay_rejects_autonomous_demands(self):
        with self.assertRaises(ValueError):
            vehicle_step(IntegrationState(0.,30.),WheelCommand(target_acceleration_mps2=-5.),self.straight,self.params,.02)

    def test_numerical_refinement_consistency(self):
        class LinearThrottle:
            def query(self,t,s):
                return dict(power_kw=350.+50.*t,brake_fraction=0.,mode="replay")
        states=[]
        for dt,ds in ((.02,1.),(.01,.5),(.005,.25)):
            state,info=advance_interval(IntegrationState(0.,30.),LinearThrottle(),None,20.,self.straight,self.params,
                IntegratorConfig(dt_max=dt,max_ds=ds))
            self.assertEqual(info["status"],"accepted")
            states.append(state)
        self.assertLess(abs(states[-1].elapsed_time_s-states[-2].elapsed_time_s),.1)
        self.assertLess(abs(states[-1].speed_mps-states[-2].speed_mps)*3.6,.5)


if __name__ == "__main__":
    unittest.main()
