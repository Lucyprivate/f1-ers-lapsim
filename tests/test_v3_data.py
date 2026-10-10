from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from f1ers_v3.data import (raw_arrays,previous_value,process_lap,select_stint_roles,
                          load_local_dataset,prepare_new_session,metadata_quality)
from f1ers_v3.geometry import ClosedRoute
from f1ers_v3.protocol import (DataUnavailable,ensure_protocol_lock,authorize_test_access,
                              verify_test_authorization,read_json,write_json,sha256)

ROOT=Path(__file__).resolve().parents[1]
BASE=datetime(2026,6,26,10,tzinfo=timezone.utc)
def date(seconds):return (BASE+timedelta(seconds=seconds)).isoformat()


class DataTests(unittest.TestCase):
    def signals(self):
        car=[{"date":date(i),"speed":100,"throttle":x,"brake":b}
             for i,(x,b) in enumerate([(0,0),(104,104),(100,100),(50,0)])]
        loc=[{"date":date(i),"x":i,"y":0} for i in range(4)]
        return raw_arrays(car,loc)

    def test_brake_switch_zero_order_hold_no_future_switch(self):
        arrays=self.signals()
        np.testing.assert_array_equal(arrays["brake"],[0,0,1,0])
        query=np.array([.5,1.5,1.999,2.,2.5])+BASE.timestamp()
        np.testing.assert_array_equal(previous_value(arrays["car_t"],arrays["brake"],query),[0,0,0,1,1])
        self.assertAlmostEqual(arrays["throttle"][1],.5)
        self.assertEqual(arrays["signal_audit"]["brake_invalid_samples"],1)

    def test_original_validity_and_raw_values_survive_repair(self):
        arrays=self.signals()
        self.assertEqual(arrays["brake_raw_value"][1],104)
        self.assertFalse(arrays["brake_original_valid"][1])
        self.assertFalse(arrays["throttle_original_valid"][1])

    def test_absolute_projection_not_first_sample_reset_or_span_stretch(self):
        radius=50.;angle=np.linspace(0,2*np.pi,200,endpoint=False)
        points=np.column_stack([radius*np.cos(angle),radius*np.sin(angle)])
        route=ClosedRoute(points,kind="spline")
        track=route.sample(5.);track["sector_end_s_m"]=[route.length_m/3,2*route.length_m/3,route.length_m]
        speed=route.length_m/10
        times=np.linspace(-1,11,301)
        car=[{"date":date(t),"speed":speed*3.6,"throttle":100,"brake":0} for t in times]
        location=[]
        for t in times:
            a=(12+speed*t)/radius
            location.append({"date":date(t),"x":radius*np.cos(a),"y":radius*np.sin(a)})
        arrays=raw_arrays(car,location)
        lap={"lap_number":3,"date_start":date(0),"lap_duration":10.,"role":"calibration"}
        reg={"location_clock_offset_s":0.,"rotation":[[1,0],[0,1]],"translation_raw_units":[0,0],"raw_location_units_per_m":1.}
        result=process_lap(lap,arrays,track,reg)
        self.assertGreater(result["alignment_audit"]["absolute_projected_start_m"],11.9)
        self.assertLess(result["alignment_audit"]["absolute_projected_start_m"],12.1)
        self.assertFalse(result["alignment_audit"]["per_lap_first_sample_reset"])
        self.assertFalse(result["alignment_audit"]["per_lap_span_stretch"])
        self.assertLess(result["t_s"][0],0)
        self.assertFalse(result["driver_trace"]["source_sample_original"][0])
        self.assertEqual(len(result["driver_trace"]["throttle_raw_value"]),len(result["driver_trace"]["time_s"]))
        strict=process_lap(lap,arrays,track,reg,strict_signal_window=True)
        self.assertTrue(np.isnan(strict["speed_mps"][0]))
        self.assertGreater(strict["alignment_audit"]["outside_metadata_lap_scoring_nodes"],0)

    def test_chronological_role_selection_ignores_pace(self):
        stints=[{"driver_number":63,"compound":"MEDIUM","stint_number":1,"lap_start":3,"lap_end":10},
                {"driver_number":63,"compound":"MEDIUM","stint_number":2,"lap_start":11,"lap_end":17}]
        laps=[{"lap_number":i,"date_start":date(i*100),"lap_duration":1000-i} for i in range(3,18)]
        statuses={i:{"lap_number":i,"passed":i!=4,"reasons":[] if i!=4 else ["gap"]} for i in range(3,18)}
        selection=select_stint_roles(stints,laps,statuses)
        self.assertEqual(selection["stint_number"],1)
        self.assertEqual(selection["adaptation_laps"],[3,5,6])
        self.assertEqual(selection["test_laps"],[7,8,9])
        self.assertEqual(selection["rejected_laps"][0]["lap_number"],4)

    def test_duration_is_not_quality_pace_selection(self):
        lap={"lap_number":4,"driver_number":63,"date_start":date(0),"lap_duration":300.,
             "duration_sector_1":100.,"duration_sector_2":100.,"duration_sector_3":100.}
        metadata={"pit":[],"race_control":[],"weather":[{"date":date(0),"rainfall":0}]}
        self.assertTrue(metadata_quality(lap,metadata,{})["passed"])

    def test_local_loader_fixed_roles_no_historical_arrays(self):
        with patch("numpy.load",wraps=np.load) as opened:
            data=load_local_dataset(ROOT,"spline",5.)
        self.assertEqual([l["lap_number"] for l in data["laps"]],[3,4,5,6,7,9])
        self.assertEqual([l["role"] for l in data["laps"]],["calibration"]*4+["development"]*2)
        self.assertFalse(data["audit"]["test_arrays_loaded"])
        self.assertFalse(any("lap_10" in str(call.args[0]) or "data/v2" in str(call.args[0]) for call in opened.call_args_list))

    def test_mesh_loader_does_not_mutate_frozen_local_audit(self):
        load_local_dataset(ROOT,"spline",5.)
        path=ROOT/"data/v3/local/audit_spline_5m.json";before=sha256(path)
        load_local_dataset(ROOT,"spline",10.)
        load_local_dataset(ROOT,"spline",2.5)
        self.assertEqual(sha256(path),before)

    def protocol_fixture(self,root):
        write_json(root/"config/protocol_austria_2026_v3.json",read_json(ROOT/"config/protocol_austria_2026_v3.json"))
        names=["config/rules_austria_2026.json","data/processed/dataset.json","data/raw/tum/Spielberg_minimum_curvature.csv","data/raw/tum/Spielberg_centerline.csv"]
        names += [f"data/raw/openf1/{name}.json" for name in ("sessions","drivers","laps","race_control","weather","pit","stints","car_data_63_laps3to9","location_63_laps3to9")]
        for name in names:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("[]",encoding="utf-8")

    def test_preregister_before_new_telemetry_and_lock_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.protocol_fixture(root)
            lock=ensure_protocol_lock(root)
            self.assertTrue(lock["frozen_before_fp2_download"])
            (root/"data/raw/openf1/laps.json").write_text("[1]",encoding="utf-8")
            with self.assertRaises(DataUnavailable):ensure_protocol_lock(root)

    def test_test_claim_exclusive_and_bound_to_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.protocol_fixture(root);ensure_protocol_lock(root)
            freeze=root/"results/freeze.json";write_json(freeze,{"model":"frozen"})
            authorize_test_access(root,freeze,"official")
            self.assertEqual(verify_test_authorization(root)["experiment_name"],"official")
            with self.assertRaises(DataUnavailable):authorize_test_access(root,freeze,"alias")
            write_json(freeze,{"model":"retuned"})
            with self.assertRaises(DataUnavailable):verify_test_authorization(root)

    def test_denied_new_session_is_pending_and_not_substituted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.protocol_fixture(root)
            with patch("f1ers_v3.data._bounded_fetch",side_effect=DataUnavailable("HTTP 401 live-session global restriction")) as request:
                status=prepare_new_session(root)
                repeat=prepare_new_session(root,allow_network=False)
            self.assertEqual(request.call_count,1)
            self.assertEqual(status,repeat)
            self.assertEqual(status["status"],"pending")
            self.assertFalse(status["session_substitution"])
            self.assertFalse(status["late_test_scored"])

    def test_unqueried_offline_placeholder_allows_one_later_public_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.protocol_fixture(root)
            offline=prepare_new_session(root,allow_network=False)
            self.assertFalse(offline["query_attempted"])
            with patch("f1ers_v3.data._bounded_fetch",side_effect=DataUnavailable("HTTP 401 global access restricted")) as request:
                online=prepare_new_session(root,allow_network=True)
                digest=sha256(root/"data/v3/fp2/status.json")
                repeat=prepare_new_session(root,allow_network=True)
            self.assertEqual(request.call_count,1)
            self.assertTrue(online["query_attempted"])
            self.assertEqual(online,repeat)
            self.assertEqual(sha256(root/"data/v3/fp2/status.json"),digest)

    def test_unqueried_offline_cache_is_immutable_after_v3_freeze_or_test_claim(self):
        for marker in ("results/custom/frozen_configuration.json","data/v3/test_access.json",
                       "data/v3/test_arrays_unsealed.json","data/v3/experiment_lock.json"):
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.protocol_fixture(root)
                offline=prepare_new_session(root,allow_network=False)
                digest=sha256(root/"data/v3/fp2/status.json")
                write_json(root/marker,{"schema_version":3,"model":"frozen"})
                with patch("f1ers_v3.data._bounded_fetch") as request:
                    self.assertEqual(prepare_new_session(root,allow_network=True),offline)
                request.assert_not_called()
                self.assertEqual(sha256(root/"data/v3/fp2/status.json"),digest)

    def test_copied_v2_freeze_does_not_block_first_v3_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.protocol_fixture(root)
            prepare_new_session(root,allow_network=False)
            write_json(root/"results/austria_2026_v2/frozen_configuration.json",{"schema_version":2})
            with patch("f1ers_v3.data._bounded_fetch",side_effect=DataUnavailable("401")) as request:
                online=prepare_new_session(root,allow_network=True)
            self.assertEqual(request.call_count,1)
            self.assertTrue(online["query_attempted"])

    def test_ready_or_previously_queried_pending_cache_never_requeries(self):
        cached_cases=[{"status":"ready"},
                      {"status":"pending","reason":"HTTP 401","queried_at_utc":"2026-10-09T00:00:00Z"},
                      {"status":"pending","reason":"Offline mode: exact FP2 public sources are not cached","query_attempted":True}]
        for cached in cached_cases:
            with self.subTest(cached=cached),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.protocol_fixture(root);ensure_protocol_lock(root)
                path=root/"data/v3/fp2/status.json";write_json(path,cached);digest=sha256(path)
                with patch("f1ers_v3.data._bounded_fetch") as request:
                    self.assertEqual(prepare_new_session(root,allow_network=True),cached)
                request.assert_not_called()
                self.assertEqual(sha256(path),digest)


if __name__=="__main__":unittest.main()
