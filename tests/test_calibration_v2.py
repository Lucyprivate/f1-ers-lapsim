import unittest
import numpy as np
from f1ers.calibration_v2 import accuracy_gate,compare_observables,ACCURACY_TARGET


class AccuracyTests(unittest.TestCase):
    def test_sector_cancellation_does_not_pass_gate(self):
        report={"metrics":[{"speed_rmse_kmh":10,"lap_error_pct":.1,"sector_error_s":[1.2,-1.2,0]}]}
        self.assertFalse(accuracy_gate(report)["passed"])

    def test_missing_nonfinite_or_physically_invalid_test_fails(self):
        self.assertFalse(accuracy_gate({"metrics":[]})["passed"])
        report={"metrics":[{"speed_rmse_kmh":np.nan,"lap_error_pct":0,"sector_error_s":[0,0,0]}]}
        self.assertFalse(accuracy_gate(report)["passed"])
        report["metrics"][0]["speed_rmse_kmh"]=1
        self.assertFalse(accuracy_gate(report,physical=False)["passed"])
        report["metrics"].append({"speed_rmse_kmh":1,"lap_error_pct":np.nan,"sector_error_s":[0,0,np.nan]})
        self.assertFalse(accuracy_gate(report)["passed"])

    def test_precommitted_targets_and_mean_lap_weighting(self):
        report={"metrics":[{"speed_rmse_kmh":14,"lap_error_pct":-2,"sector_error_s":[1,0,0]},
            {"speed_rmse_kmh":16,"lap_error_pct":2,"sector_error_s":[0,0,-1]}]}
        gate=accuracy_gate(report)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["observed"]["mean_speed_rmse_kmh"],15)
        self.assertEqual(gate["thresholds"],ACCURACY_TARGET)

    def test_comparison_uses_terminal_speed_and_actual_segment_time(self):
        s=np.arange(30.)
        lap={"lap_number":10,"role":"test","lap_time_s":6,"s_m":s,"speed_mps":np.full(30,10),
            "sector_times_s":[2,2,2],"t_s":s*.2,"brake":np.zeros(30),"throttle":np.ones(30)}
        dataset={"track":{"length_m":30,"curvature_1pm":np.zeros(30)},"laps":[lap]}
        result={"s_m":s,"speed_mps":np.full(30,10),"dt_s":np.r_[np.full(10,.1),np.full(10,.2),np.full(10,.3)],
            "summary":{"lap_time_s":6,"speed_end_mps":10}}
        metrics=compare_observables(dataset,result,"test")["metrics"][0]
        np.testing.assert_allclose(metrics["sim_sector_times_s"],[1,2,3])
        self.assertEqual(metrics["speed_rmse_kmh"],0)
        self.assertEqual(metrics["grouped_speed_errors"]["full_throttle_straight"]["samples"],30)
        self.assertEqual(compare_observables(dataset,result,"calibration")["metrics"],[])
        result["summary"]["completed_lap"]=False
        self.assertEqual(compare_observables(dataset,result,"test")["metrics"],[])
        result["summary"]["completed_lap"]=True
        result["speed_mps"]=result["speed_mps"][:-1]
        self.assertEqual(compare_observables(dataset,result,"test")["metrics"],[])


if __name__=="__main__":unittest.main()
