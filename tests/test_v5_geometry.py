"""Position-only smoothing, control-line preservation and frozen-role regressions."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from f1ers_v3.geometry import ClosedRoute
from f1ers_v3.protocol import DataUnavailable
from f1ers_v4.data import load_local_dataset as load_v4
from f1ers_v5.data import load_dataset
from f1ers_v5.geometry import (SharedPositionRoute, _fit_spec, _position_samples,
                               baseline_spec, prepare_geometry, remap_rules)


def fixture(offset=1., noise=.5, phase=31.):
    angle = np.linspace(0., 2*np.pi, 256, endpoint=False)
    points = np.column_stack([300*np.cos(angle), 300*np.sin(angle)])
    route = ClosedRoute(points, kind="spline", phase=phase, legacy_phase=17., legacy_direction=1)
    centre = np.column_stack([points, np.full(256, 6.), np.full(256, 6.)])
    samples = []
    a = np.linspace(0., 2*np.pi, 500, endpoint=False)
    for lap in (3, 4, 5, 6):
        radius = 300+offset+noise*np.sin(65*a+lap)
        xy = np.column_stack([radius*np.cos(a), radius*np.sin(a)])
        samples.append({"lap_number": lap, "role": "calibration", "lap_time_s": 100.,
                        "location_time_s": np.arange(len(a))*100/len(a),
                        "location_xy_m": xy, "location_valid_mask": np.ones(len(a), bool),
                        "speed_mps": np.full(len(a), 20.)})
    return {"role": "calibration", "track": route.sample(5.), "raw_laps": samples,
            "boundary_reference": centre}


class TestV5Geometry(unittest.TestCase):
    def test_periodic_noise_is_smoothed_and_spec_roundtrips(self):
        data = fixture()
        spec, report = prepare_geometry(data)
        self.assertEqual(spec["kind"], "shared_position_spline", report)
        route = SharedPositionRoute(json.loads(json.dumps(spec)))
        xy = route.evaluate([0., route.length_m])
        np.testing.assert_allclose(xy["x_m"][:1], xy["x_m"][1:], atol=1e-10, rtol=0)
        np.testing.assert_allclose(xy["tangent"][:1], xy["tangent"][1:], atol=1e-10, rtol=0)
        self.assertTrue(all(x["fit_laps"] == [3, 4, 5, 6] for x in [spec, report]))
        dense = route.evaluate(np.arange(0, route.length_m, 1.))["curvature_1pm"]
        raw_curve = ClosedRoute(data["raw_laps"][0]["location_xy_m"], kind="spline")
        raw = raw_curve.evaluate(np.arange(0, raw_curve.length_m, 1.))["curvature_1pm"]
        self.assertLess(float(np.std(dense)), .1*float(np.std(raw)))
        self.assertLess(abs(route.length_m-2*np.pi*301), .5)

    def test_cv_is_only_calibration_positions_and_independent_of_speed(self):
        data = fixture()
        first = prepare_geometry(data)
        changed = copy.deepcopy(data)
        for lap in changed["raw_laps"]: lap["speed_mps"][:] = 99999
        self.assertEqual(first, prepare_geometry(changed))
        for candidate in first[1]["candidates"]:
            for fold in candidate["folds"]:
                self.assertEqual(set(fold["fit_laps"]), {3, 4, 5, 6}-{fold["held_out_lap"]})
        data["raw_laps"][0]["role"] = "development"
        with self.assertRaises(DataUnavailable): prepare_geometry(data)

    def test_control_origin_is_registered_not_first_observation(self):
        data = fixture(noise=0, phase=123.)
        base = data["track"]["continuous_route"]
        samples = [_position_samples(x, base) for x in data["raw_laps"]]
        spec = _fit_spec(samples, base, 40.)
        route = SharedPositionRoute(spec)
        old_control = (base.phase-base.legacy_phase)*base.legacy_direction
        self.assertLess(min(float(route.legacy_to_s(old_control)),
                            route.length_m-float(route.legacy_to_s(old_control))), 1e-8)
        query = np.linspace(0., base.source_length-1e-3, 81)
        np.testing.assert_allclose(route.s_to_legacy(route.legacy_to_s(query)), query, atol=2e-7)
        origin = route.evaluate(0.)
        first = data["raw_laps"][0]["location_xy_m"][0]
        self.assertGreater(np.hypot(origin["x_m"]-first[0], origin["y_m"]-first[1]), 100.)

    def test_boundaries_fail_to_unchanged_baseline(self):
        spec, report = prepare_geometry(fixture(offset=20., noise=0))
        self.assertEqual(spec, baseline_spec())
        self.assertEqual(report["selected_kind"], "baseline")
        self.assertTrue(any("leaves_inherited_TUM_boundaries" in x.get("final_geometry", {}).get("reasons", [])
                            for x in report["candidates"]))

    def test_rules_use_nonlinear_reference_map(self):
        data = fixture(noise=0)
        base = data["track"]["continuous_route"]
        spec = _fit_spec([_position_samples(x, base) for x in data["raw_laps"]], base, 40.)
        route = SharedPositionRoute(spec)
        rules = {"fia_centreline_length_m": 2000., "alternative_power_zones_m": [[100., 300.]],
                 "power_reduction_exceptions": [{"start_m": 700., "end_m": 900.}]}
        mapped = remap_rules(rules, route.sample())
        expected = float(route.legacy_to_s(100.*base.source_length/2000.))
        self.assertEqual(mapped["alternative_power_zones_route_m"][0][0], expected)
        self.assertEqual(rules["alternative_power_zones_m"], [[100., 300.]])

    def test_local_baseline_matches_v4_and_returns_only_requested_role(self):
        root = Path(__file__).resolve().parents[1]
        inherited = load_v4(root)
        for role, expected in (("calibration", [3, 4, 5, 6]), ("development", [7, 9])):
            data = load_dataset(root, role=role)
            self.assertEqual([x["lap_number"] for x in data["laps"]], expected)
            self.assertEqual([x["lap_number"] for x in data["raw_laps"]], expected)
            self.assertEqual(data["track"]["curve_id"], inherited["track"]["curve_id"])
            self.assertEqual(data["rules"], inherited["rules"])
            for lap in data["laps"]:
                old = next(x for x in inherited["laps"] if x["lap_number"] == lap["lap_number"])
                for key in ("s_m", "speed_mps", "t_s", "progress_m", "initial_speed_mps"):
                    np.testing.assert_array_equal(lap[key], old[key])
                for key in ("time_s", "speed_mps", "throttle", "brake"):
                    np.testing.assert_array_equal(lap["driver_trace"][key], old["driver_trace"][key])
            for raw in data["raw_laps"]:
                self.assertTrue(np.all(raw["source_sample_original"]))
                self.assertEqual(len(raw["location_time_s"]), len(raw["progress_m"]))
                self.assertEqual(len(raw["time_s"]), len(raw["speed_integral_progress_m"]))
                self.assertFalse(raw["provenance"]["per_lap_first_sample_reset"])
                self.assertFalse(raw["provenance"]["per_lap_span_stretch"])
        with self.assertRaises(DataUnavailable): load_dataset(root, role="test")


if __name__ == "__main__": unittest.main()
