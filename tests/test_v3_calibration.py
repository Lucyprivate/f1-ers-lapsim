import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np

from f1ers_v3.calibration import (adapt_session, fit_candidate, parameter_sensitivity,
                                  select_candidate, compare_paths, mesh_audit, _constraint_penalty)
from f1ers_v3.diagnostics import accuracy_gate, compare_lap, residual_vector
from f1ers_v3.geometry import ClosedRoute
from f1ers_v3.vehicle import VehicleParamsV3


def lap_fixture(number=3, role="calibration"):
    s=np.arange(20)*50.;time=s/50
    return {"lap_number":number,"role":role,"s_m":s,"speed_mps":np.full(20,50.),
            "t_s":time,"lap_time_s":20.,"sector_times_s":[6.,7.,7.],
            "brake":np.zeros(20),"throttle":np.ones(20),
            "driver_trace":{"time_s":np.array([0.,50.]),"speed_mps":np.full(2,50.),
                            "throttle":np.ones(2),"brake":np.zeros(2)}}


def data_fixture(roles=None):
    roles=roles or [(3,"calibration"),(4,"calibration"),(5,"calibration"),(6,"calibration"),(7,"development"),(9,"development")]
    return {"track":{"s_m":np.arange(20)*50.,"length_m":1000.,"curvature_1pm":np.zeros(20),
                     "x_m":np.arange(20)*50.,"y_m":np.zeros(20),"corner_s_m":{"1":500.}},
            "rules":{},"laps":[lap_fixture(n,r) for n,r in roles]}


def fake_compare(dataset, vehicle, role="development", diagnostics=True):
    laps=[l for l in dataset["laps"] if l["role"]==role]
    metrics=[]
    for lap in laps:
        err=vehicle.mu-1.75
        metrics.append({"lap_number":lap["lap_number"],"role":role,"valid":True,
                        "speed_rmse_kmh":abs(err)+1,"lap_error_pct":err,"lap_error_s":err,
                        "real_lap_time_s":20.,"sector_error_s":[err,err,err]})
    result={"s_m":np.arange(20)*50.,"speed_mps":np.full(20,50.),"dt_s":np.ones(20),
            "summary":{"completed_lap":True,"feasible":True,"execution_feasible":True,"lap_time_s":20.}}
    output={"physical_valid":True,"passed":True,"role":role,"diagnostics":{},
            "replay":{"metrics":metrics,"results":{str(l["lap_number"]):result for l in laps},"gate":{"passed":True}},
            "autonomous":{"metrics":metrics,"result":result,"gate":{"passed":True}}}
    residual=residual_vector(output);output["normalized_objective"]=float(residual@residual)
    return output


class V3CalibrationTests(unittest.TestCase):
    def test_sector_and_single_lap_failure_cannot_cancel(self):
        metrics=[{"valid":True,"speed_rmse_kmh":14,"lap_error_pct":0,"sector_error_s":[1.1,-1.1,0]},
                 {"valid":True,"speed_rmse_kmh":16,"lap_error_pct":2.1,"sector_error_s":[0,0,0]}]
        self.assertFalse(accuracy_gate({"metrics":metrics})["passed"])
        self.assertEqual(accuracy_gate({"metrics":metrics})["observed"]["mean_speed_rmse_kmh"],15.)

    def test_empty_sectors_missing_laps_nonfinite_fail(self):
        metric={"valid":True,"speed_rmse_kmh":1,"lap_error_pct":0,"sector_error_s":[]}
        self.assertFalse(accuracy_gate({"metrics":[metric]})["passed"])
        metric["sector_error_s"]=[0,0,0]
        self.assertFalse(accuracy_gate({"metrics":[metric],"expected_laps":2})["passed"])
        metric["lap_error_pct"]=np.nan
        self.assertFalse(accuracy_gate({"metrics":[metric]})["passed"])

    def test_partial_prediction_never_scored_or_filled(self):
        data=data_fixture();result={"s_m":np.arange(5.),"speed_mps":np.ones(5),"dt_s":np.ones(5),
                                  "summary":{"completed_lap":False,"feasible":False,"execution_feasible":False}}
        metric=compare_lap(data["track"],data["laps"][0],result,"replay")
        self.assertFalse(metric["valid"]);self.assertNotIn("speed_rmse_kmh",metric)
        self.assertEqual(metric["completed_segments"],5)

    def test_fitting_rejects_heldout_and_historical_arrays(self):
        for role in ("test","historical","adaptation"):
            data=data_fixture();data["laps"].append(lap_fixture(10,role))
            with self.assertRaises(ValueError):fit_candidate(data,maxiter=1,maxfev=1)

    def test_candidate_tie_prefers_fewer_parameters_then_earlier(self):
        base={"solver_success":True,"physical_valid":True,"development_objective":1.,"parameter_count":5,"candidate_order":2}
        reports=[base,{**base,"development_objective":1.009,"parameter_count":4,"candidate_order":1},
                 {**base,"development_objective":1.008,"parameter_count":4,"candidate_order":0}]
        self.assertEqual(select_candidate(reports),2)
        reports[2]["solver_success"]=False
        self.assertEqual(select_candidate(reports),1)

    def test_adaptation_changes_only_two_scales(self):
        data=data_fixture([(13,"adaptation"),(14,"adaptation"),(15,"adaptation")])
        vehicle=VehicleParamsV3(mu=1.4,brake_ratio=.8,equivalent_power_kw=720.)
        with patch("f1ers_v3.calibration.compare_paths",side_effect=fake_compare):
            fitted,report,_=adapt_session(data,vehicle,maxiter=30,maxfev=60)
        for key,value in vehicle.to_dict().items():
            if key not in ("session_grip_scale","replay_power_scale"):
                self.assertEqual(fitted.to_dict()[key],value)
        self.assertLessEqual(fitted.equivalent_power_kw*fitted.replay_power_scale,732.5)
        self.assertEqual(report["adaptation_laps"],[13,14,15])

    def test_adaptation_rejects_late_test_even_if_three_laps(self):
        with self.assertRaises(ValueError):
            adapt_session(data_fixture([(13,"adaptation"),(14,"adaptation"),(15,"test")]),VehicleParamsV3())

    def test_svd_is_finite_and_calibration_only(self):
        data=data_fixture([(3,"calibration"),(4,"calibration")])
        with patch("f1ers_v3.calibration.compare_paths",side_effect=fake_compare):
            sensitivity=parameter_sensitivity(data,VehicleParamsV3(mu=1.8))
        self.assertTrue(sensitivity["available"])
        self.assertTrue(np.isfinite(sensitivity["singular_values"]).all())
        self.assertEqual(sensitivity["rank_at_relative_1e_6"],1)
        data["laps"].append(lap_fixture(10,"test"))
        with self.assertRaises(ValueError):parameter_sensitivity(data,VehicleParamsV3())

    def test_fit_runtime_failure_retained_not_aborted(self):
        with patch("f1ers_v3.calibration.compare_paths",side_effect=RuntimeError("fixture failure")):
            _,report,results=fit_candidate(data_fixture(),maxiter=2,maxfev=3)
        self.assertFalse(report["solver_success"])
        self.assertGreater(report["invalid_evaluation_count"],0)
        self.assertIn("fixture failure",results["calibration"]["failure_exception"])

    def test_mixed_legacy_guard_shapes_remain_finite_during_fit(self):
        def failed(dataset,vehicle,role="development",diagnostics=True):
            comparison=fake_compare(dataset,vehicle,role,diagnostics)
            auto=comparison["autonomous"]["result"]
            auto["summary"].update(feasible=False,execution_feasible=False,
                guard_failures=[["power_curve","tire_force"],("reserve",{"tire_force_excess_n":800.}),
                                {"proposal":{"lateral_tire_force_excess_n":np.inf},"reasons":["invalid"]},None])
            comparison["physical_valid"]=False;comparison["normalized_objective"]=None
            return comparison
        comparison=failed(data_fixture(),VehicleParamsV3(),"calibration")
        self.assertTrue(np.isfinite(_constraint_penalty(comparison,20)))
        with patch("f1ers_v3.calibration.compare_paths",side_effect=failed):
            _,report,_=fit_candidate(data_fixture(),maxiter=3,maxfev=8)
        self.assertFalse(report["physical_valid"])
        self.assertEqual(len(report["multistart"]),3)
        self.assertTrue(all(np.isfinite(run["objective"]) for run in report["multistart"]))

    def test_real_executor_replay_and_autonomous_are_separate(self):
        data=data_fixture([(7,"development")]);comparison=compare_paths(data,VehicleParamsV3())
        self.assertFalse(comparison["replay"]["results"]["7"]["ers_validated"])
        self.assertTrue(comparison["autonomous"]["result"]["ers_validated"])
        self.assertIn("normalized_objective",comparison)

    def test_mesh_without_continuous_route_cannot_pass(self):
        self.assertFalse(mesh_audit(data_fixture(),VehicleParamsV3())["passed"])

    def test_mesh_uses_one_curve_control_and_initial_condition(self):
        route=ClosedRoute(np.array([[0.,0.],[100.,0.],[100.,100.],[0.,100.]]),kind="spline")
        data=data_fixture([(3,"calibration")]);data["track"]={**data["track"],**route.sample(5.)}
        calls=[]
        class FakeSimulator:
            def __init__(self,track,rules,vehicle):
                self.track=track;self.vehicle=vehicle
            def simulate(self,mode="autonomous",controller=None,initial_speed_mps=None,driver_trace=None):
                n=len(self.track["s_m"]);speed=80.
                calls.append({"mode":mode,"curve_id":self.track["continuous_route"].curve_id,
                              "initial":initial_speed_mps,"controller":controller,"vehicle":self.vehicle})
                return {"s_m":self.track["s_m"],"speed_mps":np.full(n,speed),
                        "dt_s":np.full(n,self.track["length_m"]/n/speed),"requested_actions":np.full(n,.5),
                        "driver_profile":{"s_m":self.track["s_m"],"drive_fraction":np.ones(n),"brake_fraction":np.zeros(n)},
                        "summary":{"completed_lap":True,"execution_feasible":True,"feasible":True,
                                   "speed_start_mps":speed,"speed_end_mps":speed,"lap_time_s":self.track["length_m"]/speed}}
        with patch("f1ers_v3.simulator.SimulatorV3",FakeSimulator),patch("f1ers_v3.calibration._autonomous",lambda sim:sim.simulate()):
            audit=mesh_audit(data,VehicleParamsV3())
        self.assertTrue(audit["passed"]);self.assertFalse(audit["per_mesh_refit"])
        auto=[call for call in calls if call["mode"]=="autonomous" and call["initial"] is not None]
        self.assertEqual(len(auto),3);self.assertEqual({call["initial"] for call in auto},{80.})
        self.assertEqual({call["curve_id"] for call in calls},{route.curve_id})
        self.assertTrue(all(call["controller"]["driver_profile"] is auto[0]["controller"]["driver_profile"] for call in auto))

    def test_calibration_calls_never_receive_development_labels(self):
        def checked(dataset,vehicle,role="development",diagnostics=True):
            if role=="calibration":
                self.assertTrue(all(lap["role"]=="calibration" for lap in dataset["laps"]))
            return fake_compare(dataset,vehicle,role,diagnostics)
        with patch("f1ers_v3.calibration.compare_paths",side_effect=checked):
            fit_candidate(data_fixture(),maxiter=2,maxfev=6)


if __name__=="__main__":unittest.main()
