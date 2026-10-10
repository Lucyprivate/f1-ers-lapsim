"""Evidence gates cannot convert noisy or held-out channels into fitted truth."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from f1ers_v5.diagnostics import diagnose, _kinematics, _forces, _phase_clock_sensitivity, _proper_similarity_cost


def dataset(quantified=False, braking=False):
    t = np.arange(0.,6.,.1)
    v = 100.-40.*np.clip(t-1.,0,1.)-np.clip(t-2.,0,2.) if braking else np.full(len(t),30.)
    p = np.r_[0.,np.cumsum(.5*(v[:-1]+v[1:])*np.diff(t))]
    nodes = np.arange(0.,1000.,5.)
    track = {"length_m":1000.,"s_m":nodes,"curvature_1pm":np.zeros(len(nodes)),"corner_s_m":{"T1":50.}}
    if quantified:
        track.update(curvature_lower_1pm=np.zeros(len(nodes)),curvature_upper_1pm=np.zeros(len(nodes)))
    raws = []
    for lap in (3,4,5,6):
        raws.append({"lap_number":lap,"role":"calibration","time_s":t.copy(),"speed_mps":v.copy(),
                     "throttle":np.ones(len(t)),"brake":((t>=1.)&(t<4.)).astype(float) if braking else np.zeros(len(t)),
                     "location_time_s":t.copy(),"progress_m":p.copy(),"speed_integral_progress_m":p.copy(),
                     "source_sample_original":np.ones(len(t),bool),"speed_original_valid":np.ones(len(t),bool),
                     "throttle_original_valid":np.ones(len(t),bool),"brake_original_valid":np.ones(len(t),bool),
                     "location_valid_mask":np.ones(len(t),bool),"location_source_original":np.ones(len(t),bool)})
    return {"track":track,"raw_laps":raws,"laps":[{"lap_number":n,"role":"calibration"} for n in (3,4,5,6)],
            "registration":{"rmse_m":0.,"location_clock_offset_s":.02}}


def comparison(data, complete=True):
    nodes = data["track"]["s_m"]
    v = np.full(len(nodes),30.)
    # Correct approach/anchor, systematic excess only after the frozen apex.
    v[(nodes>=75.)&(nodes<=170.)] += 10./3.6
    result = {"s_m":nodes.copy(),"speed_mps":v,"dt_s":np.full(len(nodes),.2),
              "summary":{"completed_lap":complete,"feasible":complete,"execution_feasible":complete,
                         "lap_time_s":40.,"speed_end_mps":30.}}
    return {"role":"calibration","autonomous":{"result":result},"replay":{"results":{}}}


class PositionRoute:
    """Asymmetric closed test curve makes phase and clock separately observable."""
    def evaluate(self,s):
        angle = np.asarray(s,float)*2*np.pi/1000.
        return {"x_m":200*np.cos(angle)+30*np.cos(2*angle),
                "y_m":180*np.sin(angle)+20*np.sin(3*angle),
                "curvature_1pm":np.zeros_like(angle)}


def profile_dataset():
    d = dataset(); route=PositionRoute(); d["track"]["continuous_route"]=route
    for raw in d["raw_laps"]:
        t = raw["time_s"]
        raw["speed_mps"] = 30.+20*np.sin(t)
        q = np.r_[0.,np.cumsum(.5*(raw["speed_mps"][:-1]+raw["speed_mps"][1:])*np.diff(t))]
        raw["speed_integral_progress_m"]=q
        raw["location_speed_integral_progress_m"]=q.copy()
        raw["progress_m"]=q.copy()
        actual = route.evaluate(np.interp(t-.1,t,q)+5.)
        raw["location_xy_m"]=np.column_stack([actual["x_m"],actual["y_m"]])
    return d


class DiagnosticsV5Tests(unittest.TestCase):
    def test_rejects_development_test_and_changed_lap_identity(self):
        for kind in ("development","test","historical_diagnostic"):
            d = dataset(); d["raw_laps"][0]["role"] = kind
            with self.assertRaises(ValueError):diagnose(d)
        d = dataset(); d["raw_laps"][0]["lap_number"] = 7
        with self.assertRaises(ValueError):diagnose(d)

    def test_rejects_held_out_baseline_residuals(self):
        d = dataset(); b = comparison(d); b["role"] = "development"
        with self.assertRaises(ValueError):diagnose(d,b)
        b = comparison(d); b["autonomous"]["metrics"]=[{"role":"development","lap_number":7}]
        with self.assertRaises(ValueError):diagnose(d,b)
        b = comparison(d); b["replay"]["results"]={"7":{}}
        with self.assertRaises(ValueError):diagnose(d,b)

    def test_linear_acceleration_inside_quantization_interval(self):
        t = np.arange(0.,3.,.1); v = 30.+2.*t
        centre,low,high = _kinematics(t,v,np.ones(len(t),bool))
        valid = np.isfinite(low)
        self.assertTrue(valid.any()); np.testing.assert_allclose(centre[valid],2.,atol=1e-10)
        self.assertTrue(np.all(low[valid]<=2.)); self.assertTrue(np.all(high[valid]>=2.))

    def test_sparse_samples_do_not_create_derivatives(self):
        t = np.arange(0.,4.,.5)
        _,low,high = _kinematics(t,30.+t,np.ones(len(t),bool))
        self.assertFalse(np.isfinite(low).any()); self.assertFalse(np.isfinite(high).any())

    def test_force_and_capacity_intervals_bound_joint_demand(self):
        f = _forces(np.array([30.]),np.array([2.]),np.array([3.]),np.array([.001]),np.array([.002]))
        self.assertLess(f["wheel_force_lower_n"][0],f["wheel_force_upper_n"][0])
        self.assertLess(f["joint_utilization_lower"][0],f["joint_utilization_upper"][0])
        impossible = _forces(np.array([100.]),np.array([0.]),np.array([0.]),np.array([.1]),np.array([.1]))
        self.assertGreater(impossible["pure_lateral_utilization_lower"][0],1.)

    def test_json_safe_and_raw_inputs_not_modified(self):
        d = dataset(); before = copy.deepcopy(d); report = diagnose(d)
        json.dumps(report,allow_nan=False)
        for a,b in zip(d["raw_laps"],before["raw_laps"]):
            for key in ("speed_mps","time_s","throttle","brake","progress_m"):
                np.testing.assert_array_equal(a[key],b[key])
        self.assertFalse(report["future_test_arrays_read"])
        self.assertFalse(report["acceptance_thresholds_changed"])

    def test_unquantified_geometry_blocks_decay_candidate(self):
        report = diagnose(dataset(braking=True))
        self.assertFalse(report["brake_decay_eligible"])
        self.assertEqual(report["candidate_evidence_gates"]["brake_decay"]["status"],"not_identifiable")

    def test_robust_brake_decline_is_detected_after_geometry_sensitivity(self):
        report = diagnose(dataset(quantified=True,braking=True))
        self.assertTrue(report["brake_decay_eligible"])
        self.assertEqual(report["candidate_evidence_gates"]["brake_decay"]["supporting_laps"],[3,4,5,6])
        self.assertIn("real brake pressure",report["unknowns"])

    def test_gate_requires_three_distinct_laps(self):
        d = dataset()
        def events(a,*args):
            return [{"supports_decay":bool(a["brake"][0])}]
        d["raw_laps"][0]["brake"][0]=1.; d["raw_laps"][1]["brake"][0]=1.
        with patch("f1ers_v5.diagnostics._brake_events",side_effect=events):
            r = diagnose(d)
        self.assertFalse(r["brake_decay_eligible"])
        self.assertEqual(r["candidate_evidence_gates"]["brake_decay"]["supporting_laps"],[3,4])
        d["raw_laps"][2]["brake"][0]=1.
        with patch("f1ers_v5.diagnostics._brake_events",side_effect=events):
            self.assertTrue(diagnose(d)["brake_decay_eligible"])

    def test_partial_replay_request_execution_is_retained_without_accuracy(self):
        d = dataset(); b = comparison(d)
        b["replay"]["results"]={"3":{"summary":{"completed_lap":False},"terminal_status":"physical_stop",
                                    "terminal_s_m":10.,"terminal_time_s":1.,"terminal_speed_mps":0.,
                                    "wheel_force_n":[-5000.],"failure_proposal":{"command":{"brake_fraction":1.}}}}
        r = diagnose(d,b); fail = r["partial_replay_failure_evidence"][0]
        self.assertFalse(fail["scored"]); self.assertEqual(fail["last_executed_wheel_force_n"],[-5000.])
        self.assertEqual(fail["failure_proposal"]["command"]["brake_fraction"],1.)
        self.assertNotIn("speed_rmse_kmh",fail)

    def test_partial_autonomous_cannot_support_traction(self):
        d = dataset(quantified=True); r = diagnose(d,comparison(d,complete=False))
        self.assertFalse(r["traction_eligible"])
        self.assertTrue(all(not x["autonomous_residual_structure"]["complete_autonomous"] for x in r["by_lap"]))

    def test_exit_error_without_raw_throttle_build_is_not_identifiable(self):
        d = dataset(quantified=True)
        self.assertFalse(diagnose(d,comparison(d))["traction_eligible"])

    def test_exit_growth_plus_control_build_supports_traction(self):
        d = dataset(quantified=True)
        for raw in d["raw_laps"]:
            raw["throttle"] = np.clip((raw["time_s"]-1.5)/1.5,.2,1.)
        r = diagnose(d,comparison(d))
        self.assertTrue(r["traction_eligible"])

    def test_straight_power_is_conditional_not_ice_soc_measurement(self):
        r = diagnose(dataset()); p = r["by_lap"][0]["full_throttle_straight_power"]
        self.assertEqual(p["status"],"interval_available")
        self.assertFalse(p["ice_ers_separation_identifiable"])
        self.assertNotIn("measured_ice_power_kw",p); self.assertNotIn("soc",p)

    def test_ledger_has_expected_observation_and_falsification_condition(self):
        d=dataset()
        for raw in d["raw_laps"]:
            raw["speed_mps"][:]=80.
            raw["progress_m"]=80.*raw["time_s"]
            raw["speed_integral_progress_m"]=raw["progress_m"].copy()
        r=diagnose(d)
        self.assertTrue(all(all(k in h for k in ("expected_observation","falsification_condition","conclusion")) for h in r["hypotheses_ledger"]))
        power=next(h for h in r["hypotheses_ledger"] if h["hypothesis"]=="zero_MGUK_power_assumption")
        self.assertEqual(power["status"],"not_identifiable")
        self.assertIn("not falsified",power["conclusion"])
        self.assertTrue(all(x["relation"]=="contains_assumed_power" for x in power["evidence_by_lap"]))

    def test_position_integral_profile_recovers_synthetic_shift_without_extrapolation(self):
        r=_phase_clock_sensitivity(profile_dataset())
        self.assertEqual(r["status"],"profile_available")
        self.assertAlmostEqual(r["minimum_phase_shift_m"],5.)
        self.assertAlmostEqual(r["minimum_added_clock_s"],.1)
        self.assertLess(r["minimum_cost_m2"],1e-20)
        self.assertFalse(r["extrapolation_used"])
        self.assertTrue(all(c["sample_time_bounds_s"][0]>=c["integral_time_bounds_s"][0]+.3-1e-10 for c in r["coverage"]))
        self.assertFalse(r["formal_model_re_registration_permitted"])

    def test_profile_ignores_predicted_speed_and_keeps_registration_bytes(self):
        d=profile_dataset(); frozen=json.dumps(d["registration"],sort_keys=True).encode()
        b=comparison(d); first=diagnose(d,b)["global_phase_clock_sensitivity"]
        b["autonomous"]["result"]["speed_mps"] += 20.
        second=diagnose(d,b)["global_phase_clock_sensitivity"]
        self.assertEqual(first,second)
        self.assertEqual(frozen,json.dumps(d["registration"],sort_keys=True).encode())
        self.assertFalse(first["uses_predicted_speed"])

    def test_profile_respects_original_absolute_clock_bounds(self):
        d=profile_dataset(); d["registration"]["location_clock_offset_s"]=.95
        r=_phase_clock_sensitivity(d)
        self.assertTrue(np.all(np.asarray(r["absolute_clock_grid_s"])<=1.+1e-12))
        self.assertTrue(np.all(np.asarray(r["absolute_clock_grid_s"])>=-1.-1e-12))

    def test_profile_insufficient_common_coverage_is_not_extrapolated(self):
        d=profile_dataset()
        for raw in d["raw_laps"]:
            raw["location_valid_mask"][:]=False; raw["location_valid_mask"][:5]=True
        r=_phase_clock_sensitivity(d)
        self.assertEqual(r["status"],"not_identifiable")
        self.assertFalse(r["extrapolation_used"])

    def test_similarity_nuisance_cannot_reflect_route(self):
        x=np.array([[0.,0.],[2.,0.],[1.,3.],[4.,2.]])
        y=x*np.array([-1.,1.])
        cost,nuisance=_proper_similarity_cost(x,y)
        self.assertGreater(cost,.01)
        self.assertAlmostEqual(nuisance["rotation_determinant"],1.)


if __name__=="__main__":unittest.main()
