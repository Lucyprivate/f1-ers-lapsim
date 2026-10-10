import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from f1ers_v3.artifacts import jsonable, write_json, read_json
from f1ers_v3.cli import release_gates, train_if_released, score_cross_session
from f1ers_v3.reporting import export_trace, render_report
from f1ers_v3.verification import assert_trace_equal


class V3PipelineTests(unittest.TestCase):
    def test_pending_new_session_keeps_training_closed(self):
        report = {"solver_success": True, "physical_valid": True, "development_gate": {"passed": True}}
        gates = release_gates(report, {"passed": True, "physical_passed": True}, {"status": "pending"})
        self.assertFalse(gates["new_test"])
        with patch("f1ers_v3.learning.train_bc_dagger_v3", side_effect=AssertionError("Must not train")):
            value = train_if_released({}, None, gates, Path("unused"))
        self.assertTrue(value["skipped"])

    def test_no_test_unseal_when_source_is_pending(self):
        with patch("f1ers_v3.data.load_new_session", side_effect=AssertionError("Must not load")):
            value = score_cross_session(Path("unused"), Path("unused"), None, "alignment", "name",
                                       {"status": "pending", "reason": "HTTP 401"})
        self.assertFalse(value["test_scored"])
        self.assertEqual(value["reason"], "HTTP 401")

    def test_failed_early_adaptation_does_not_burn_final_holdout(self):
        from f1ers_v3.vehicle import VehicleParamsV3
        with TemporaryDirectory() as folder:
            with patch("f1ers_v3.data.load_new_session", return_value={}) as loader, \
                 patch("f1ers_v3.calibration.adapt_session", return_value=(VehicleParamsV3(),
                       {"solver_success":True,"physical_valid":False},{})), \
                 patch("f1ers_v3.cli._export_paths"), \
                 patch("f1ers_v3.protocol.authorize_test_access", side_effect=AssertionError("Must remain sealed")):
                value=score_cross_session(Path(folder),Path(folder),None,"alignment","name",{"status":"ready"})
            self.assertEqual(value["status"],"deferred")
            self.assertFalse(value["test_scored"])
            self.assertFalse(loader.call_args.kwargs["include_test"])

    def test_failed_development_cannot_release(self):
        report = {"solver_success": True, "physical_valid": True, "development_gate": {"passed": False}}
        cross = {"status": "evaluated", "adaptation": {"solver_success": True,"physical_valid": True},
                 "evaluation": {"passed": True}}
        self.assertFalse(release_gates(report,{"passed":True,"physical_passed":True},cross)["development"])

    def test_csv_terminal_has_state_and_no_invented_power(self):
        r = {"s_m": [0,5],"speed_mps":[10,11],"time_s":[0,.5,1],"dt_s":[.5,.5],
             "wheel_power_kw":[2,3],"energy_mj":None,"summary":{"speed_end_mps":12,"distance_completed_m":10}}
        with TemporaryDirectory() as folder:
            path=Path(folder)/"trace.csv";export_trace(path,r)
            with path.open(encoding="utf-8-sig",newline="") as handle: rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),3)
            self.assertEqual(float(rows[-1]["s_m"]),10)
            self.assertEqual(rows[-1]["wheel_power_kw"],"")
            self.assertEqual(rows[-1]["energy_mj"],"")

    def test_portable_json_preserves_applicability_and_drops_only_route_object(self):
        value=jsonable({"continuous_route":object(),"energy":None,"a":np.array([1,np.nan]),"label":"observed"})
        self.assertEqual(value,{"energy":None,"a":[1.0,None],"label":"observed"})
        with TemporaryDirectory() as folder:
            path=Path(folder)/"a.json";write_json(path,value);self.assertEqual(read_json(path),value)

    def test_exact_replay_retains_failure_and_detects_changes(self):
        r={"s_m":[0],"speed_mps":[10],"dt_s":[.5],"time_s":[0,.5],"energy_mj":None,
           "summary":{"lap_time_s":.5,"completed_lap":False,"execution_feasible":False,"feasible":False,"ers_validated":False}}
        assert_trace_equal(r,r,"partial")
        with self.assertRaises(AssertionError):
            assert_trace_equal({**r,"speed_mps":[11]},r,"partial")

    def test_exact_replay_detects_changed_rejected_proposal(self):
        r={"summary":{"lap_time_s":0,"guard_failures":[{"guard_reasons":["lateral"],"force_n":10}]},
           "guard_diagnostics":[{"step_certified":False,"force_n":10}]}
        assert_trace_equal(r,r,"rejected")
        changed={**r,"guard_diagnostics":[{"step_certified":False,"force_n":11}]}
        with self.assertRaises(AssertionError):
            assert_trace_equal(changed,r,"rejected")

    def test_csv_identifies_replay_and_retains_terminal_failure_reason(self):
        r={"mode":"replay","ers_validated":False,"s_m":[0],"speed_mps":[10],"time_s":[0,.5],
           "guard_diagnostics":[{"step_certified":True},{"step_certified":False,"guard_reasons":["lateral"]}],
           "summary":{"speed_end_mps":11,"distance_completed_m":5}}
        with TemporaryDirectory() as folder:
            path=Path(folder)/"trace.csv";export_trace(path,r)
            with path.open(encoding="utf-8-sig",newline="") as handle: rows=list(csv.DictReader(handle))
            self.assertEqual(rows[0]["mode"],"replay")
            self.assertEqual(rows[0]["ers_validated"],"False")
            self.assertIn("lateral",rows[-1]["guard_reason"])

    def test_csv_retains_initial_state_when_first_step_is_rejected(self):
        r={"mode":"replay","s_m":[],"speed_mps":[],"time_s":[0],
           "guard_diagnostics":[{"step_certified":False,"guard_reasons":["lateral"]}],
           "summary":{"speed_end_mps":40,"distance_completed_m":0}}
        with TemporaryDirectory() as folder:
            path=Path(folder)/"trace.csv";export_trace(path,r)
            with path.open(encoding="utf-8-sig",newline="") as handle: rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),1)
            self.assertEqual(float(rows[0]["speed_mps"]),40)
            self.assertEqual(rows[0]["wheel_power_kw"],"")
            self.assertIn("lateral",rows[0]["guard_reason"])

    def test_report_handles_missing_metrics_and_partial_candidates(self):
        comp={"metrics":[{"lap_number":7,"valid":False}],"gate":{"passed":False,"observed":{}}}
        bundle={"status":"未完成圈","candidates":[{"candidate":"alignment","report":{"candidate":"alignment"},
                 "development":{"replay":comp,"autonomous":comp}}],"cross_session":{"status":"pending"}}
        with TemporaryDirectory() as folder:
            path=render_report(Path(folder),bundle);page=path.read_text(encoding="utf-8")
            self.assertIn("不完整 / 无可评分圈",page)
            self.assertIn("x.valid!==false",page)
            self.assertNotIn("__PAYLOAD__",page)


if __name__ == "__main__":
    unittest.main()
