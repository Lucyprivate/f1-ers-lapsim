"""Calibration-position-only shared geometry with a frozen coordinate transform.

Gaussian scales are spatial standard deviations in metres, not knot spacing or
speed-error tuning parameters.  A candidate changes the common shape; it never
subtracts a lap's first position or rescales a lap's progress.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from f1ers_v3.geometry import ClosedRoute, build_track as inherited_build_track
from f1ers_v3.data import _project
from f1ers_v3.protocol import DataUnavailable, read_json

SCALES_M = (20.0, 40.0, 80.0)
FIT_DS_M = 2.0
CALIBRATION_LAPS = (3, 4, 5, 6)


def baseline_spec():
    return {"schema_version": 5, "kind": "baseline", "method": "unchanged_frozen_v4_spline",
            "fit_role": "calibration", "fit_laps": list(CALIBRATION_LAPS),
            "uses_speed_residuals": False}


class SharedPositionRoute(ClosedRoute):
    """A true-arclength curve with a non-affine map from the legacy source.

    ``source_to_s`` still receives this curve's polyline projection parameter.
    Legacy sector/rule coordinates instead pass through the stored reference
    knots, so a new circumference does not stretch old coordinates linearly.
    """
    def __init__(self, spec, registration=None):
        points = np.asarray(spec["points_m"], float)
        self.reference_u_m = np.asarray(spec["source_reference_u_m"], float)
        self.reference_length_m = float(spec["source_reference_length_m"])
        if (self.reference_u_m.shape != (len(points)+1,) or
                not np.all(np.diff(self.reference_u_m) > 0) or
                self.reference_u_m[0] != 0 or
                self.reference_u_m[-1] != self.reference_length_m):
            raise DataUnavailable("Invalid frozen reference-to-shared-curve map")
        from f1ers_v3.geometry import polyline_length
        candidate_u = polyline_length(points)
        phase = float((registration or {}).get("original_route_control_line_s_m",
                                              spec["control_reference_u_m"]))
        direction = int((registration or {}).get("direction", spec["travel_direction"]))
        mapped_phase = float(np.interp(phase % self.reference_length_m,
                                      self.reference_u_m, candidate_u))
        super().__init__(points, kind="spline", phase=mapped_phase, direction=direction)
        self.legacy_phase = float(spec["legacy_phase_m"])
        self.legacy_direction = int(spec["legacy_direction"])

    def legacy_to_s(self, old_s):
        original = np.mod(self.legacy_phase+self.legacy_direction*np.asarray(old_s, float),
                          self.reference_length_m)
        current = np.interp(original, self.reference_u_m, self.source_s)
        return self.source_to_s(current)

    def s_to_legacy(self, s):
        current = self.s_to_source(s)
        original = np.interp(current, self.source_s, self.reference_u_m)
        legacy = np.mod(self.legacy_direction*(original-self.legacy_phase),
                        self.reference_length_m)
        # Monotone cubic inverse roundoff may put an exact zero just below 0.
        return np.where(np.abs(legacy-self.reference_length_m) <= 1e-7, 0., legacy)


def _boundary_clearance(xy, centre, direction=1):
    centre = np.asarray(centre, float)
    if (centre.ndim != 2 or centre.shape[1] < 4 or len(centre) < 4 or
            not np.isfinite(centre).all() or np.any(centre[:, 2:4] <= 0)):
        raise DataUnavailable("Inherited boundary reference is incomplete")
    xy = np.asarray(xy, float)
    ids = np.argmin(np.sum((xy[:, None, :]-centre[None, :, :2])**2, axis=2), axis=1)
    tangent = np.roll(centre[:, :2], -1, axis=0)-np.roll(centre[:, :2], 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offset = np.sum((xy-centre[ids, :2])*normal[ids], axis=1)
    left, right = centre[ids, 3]-offset, centre[ids, 2]+offset
    return (right, left) if direction < 0 else (left, right)


def _validate_route(route, centre, baseline_route):
    s = np.arange(0., route.length_m, 1.)
    values = route.evaluate(s)
    xy = np.column_stack([values["x_m"], values["y_m"]])
    left, right = _boundary_clearance(xy, centre, route.direction)
    k = values["curvature_1pm"]
    b = baseline_route.evaluate(np.arange(0., baseline_route.length_m, 1.))["curvature_1pm"]
    noise = float(np.sqrt(np.mean((k-gaussian_filter1d(k, 5., mode="wrap"))**2)))
    baseline_noise = float(np.sqrt(np.mean((b-gaussian_filter1d(b, 5., mode="wrap"))**2)))
    integral = float(np.mean(np.abs(k))*route.length_m)
    fine = route.evaluate(np.arange(0., route.length_m, .5))["curvature_1pm"]
    fine_integral = float(np.mean(np.abs(fine))*route.length_m)
    rel = abs(integral-fine_integral)/max(fine_integral, 1e-12)
    closure = route.evaluate([0., route.length_m])
    closure_error = float(np.hypot(np.diff(closure["x_m"])[0], np.diff(closure["y_m"])[0]))
    tangent_error = float(np.linalg.norm(np.diff(closure["tangent"], axis=0)))
    reasons = []
    clearance = min(float(left.min()), float(right.min()))
    if clearance <= 0: reasons.append("leaves_inherited_TUM_boundaries")
    if closure_error > 1e-8 or tangent_error > 1e-8: reasons.append("closure_failed")
    if not np.isfinite(k).all() or np.max(np.abs(k)) > max(.08, 2*np.max(np.abs(b))):
        reasons.append("curvature_peak_unstable")
    if noise > max(.0005, 1.5*baseline_noise): reasons.append("curvature_high_frequency_unstable")
    if rel > .02: reasons.append("curvature_sampling_unstable")
    return {"passed": not reasons, "reasons": reasons, "minimum_boundary_clearance_m": clearance,
            "closure_position_error_m": closure_error, "closure_tangent_error": tangent_error,
            "maximum_abs_curvature_1pm": float(np.max(np.abs(k))),
            "curvature_high_frequency_rms_1pm": noise,
            "curvature_absolute_integral_1m_vs_half_m_relative_difference": rel,
            "analytic_curvature": True, "true_arc_length": True, "arc_quadrature_spacing_m": .1}


def _position_samples(raw, route):
    xy = np.asarray(raw["location_xy_m"], float)
    valid = np.asarray(raw.get("location_valid_mask", np.ones(len(xy), bool)), bool)
    valid = valid.copy() & np.asarray(raw.get("location_source_original", np.ones(len(xy), bool)), bool)
    time = np.asarray(raw["location_time_s"], float)
    duration = float(raw["lap_time_s"])
    valid &= np.isfinite(xy).all(axis=1) & (time >= 0) & (time <= duration)
    xy = xy[valid]
    if len(xy) < 100: raise DataUnavailable("Too few original in-lap positions for shared geometry")
    source, _ = _project(xy, route.points, route.source_s)
    s = route.source_to_s(source)
    gaps = np.diff(np.r_[np.sort(s), np.min(s)+route.length_m])
    if np.max(gaps) > max(100., .03*route.length_m):
        raise DataUnavailable("Position coverage does not support a common closed shape")
    values = route.evaluate(s)
    normal = np.column_stack([-values["tangent"][:, 1], values["tangent"][:, 0]])
    base_xy = np.column_stack([values["x_m"], values["y_m"]])
    offset = np.sum((xy-base_xy)*normal, axis=1)
    return {"s_m": s, "offset_m": offset, "xy_m": xy,
            "lap_number": int(raw["lap_number"])}


def _fit_spec(samples, route, scale):
    n = int(np.ceil(route.length_m/FIT_DS_M))
    ds = route.length_m/n
    numerator, denominator = np.zeros(n), np.zeros(n)
    # Each lap contributes equally even if its original sampling is denser.
    for sample in samples:
        ids = np.floor(np.mod(sample["s_m"], route.length_m)/ds).astype(int)
        sums = np.bincount(ids, weights=sample["offset_m"], minlength=n)
        count = np.bincount(ids, minlength=n)
        observed = count > 0
        numerator[observed] += sums[observed]/count[observed]
        denominator[observed] += 1.
    num = gaussian_filter1d(numerator, scale/ds, mode="wrap", truncate=4.)
    den = gaussian_filter1d(denominator, scale/ds, mode="wrap", truncate=4.)
    if np.min(den) <= 1e-12: raise DataUnavailable("Uncovered periodic smoothing window")
    offset = num/den
    reference_u = np.linspace(0., route.source_length, n, endpoint=False)
    s = route.source_to_s(reference_u)
    lateral = np.interp(s, np.r_[np.arange(n)*ds, route.length_m], np.r_[offset, offset[0]])
    values = route.evaluate(s)
    normal = np.column_stack([-values["tangent"][:, 1], values["tangent"][:, 0]])
    xy = np.column_stack([values["x_m"], values["y_m"]])+lateral[:, None]*normal
    return {"schema_version": 5, "kind": "shared_position_spline", "fit_role": "calibration",
            "fit_laps": sorted(int(x["lap_number"]) for x in samples), "method": "periodic_Gaussian_shared_lateral_offset",
            "smoothing_scale_m": float(scale), "smoothing_scale_definition": "Gaussian spatial standard deviation",
            "fit_grid_max_spacing_m": FIT_DS_M, "uses_speed_residuals": False,
            "points_m": xy.tolist(), "source_reference_u_m": np.r_[reference_u, route.source_length].tolist(),
            "source_reference_length_m": route.source_length, "control_reference_u_m": route.phase,
            "travel_direction": route.direction, "legacy_phase_m": route.legacy_phase,
            "legacy_direction": route.legacy_direction}


def _position_rmse(xy, route):
    # Distance to the curve, rather than distance to a potentially noisy timestamp.
    _, error = _project(xy, route.points, route.source_s)
    return float(np.sqrt(np.mean(error**2)))


def prepare_geometry(calibration_dataset):
    """Select using only leave-one-calibration-lap-out position residuals.

    Every fold and the final fit must be valid.  A candidate must improve the
    baseline CV position RMSE by at least 1%; within 1% of the best candidate the
    largest Gaussian sigma wins.  Failures preserve the unchanged baseline.
    """
    laps = calibration_dataset.get("raw_laps", [])
    if (calibration_dataset.get("role", "calibration") != "calibration" or
            [int(x["lap_number"]) for x in laps] != list(CALIBRATION_LAPS) or
            any(x.get("role") != "calibration" for x in laps)):
        raise DataUnavailable("Shared geometry may use only calibration laps 3-6")
    route = calibration_dataset["track"]["continuous_route"]
    report = {"schema_version": 5, "fit_laps": list(CALIBRATION_LAPS), "fit_role": "calibration",
              "selection_metric": "leave_one_lap_out_original_position_RMSE_m",
              "smoothing_scales_m": list(SCALES_M), "smoothing_scale_definition": "Gaussian spatial standard deviation",
              "within_best_relative_tolerance": .01, "baseline_improvement_required": .01,
              "speed_channels_used": False, "speed_residuals_used": False,
              "per_lap_first_sample_reset": False, "per_lap_span_stretch": False,
              "candidates": [], "selected_kind": "baseline"}
    try:
        centre = np.asarray(calibration_dataset["boundary_reference"], float)
        samples = [_position_samples(x, route) for x in laps]
        baseline_scores = [_position_rmse(x["xy_m"], route) for x in samples]
        baseline_score = float(np.sqrt(np.mean(np.square(baseline_scores))))
        report["baseline_position_cv_rmse_m"] = baseline_score
        eligible = []
        for scale in SCALES_M:
            record = {"smoothing_scale_m": scale, "folds": [], "passed": False}
            report["candidates"].append(record)
            try:
                for i, held in enumerate(samples):
                    spec = _fit_spec([x for j, x in enumerate(samples) if j != i], route, scale)
                    candidate = SharedPositionRoute(spec)
                    checks = _validate_route(candidate, centre, route)
                    record["folds"].append({"held_out_lap": held["lap_number"],
                        "fit_laps": [x["lap_number"] for j, x in enumerate(samples) if j != i],
                        "position_rmse_m": _position_rmse(held["xy_m"], candidate), "physical_geometry": checks})
                full_spec = _fit_spec(samples, route, scale)
                final_checks = _validate_route(SharedPositionRoute(full_spec), centre, route)
                record["final_geometry"] = final_checks
                record["position_cv_rmse_m"] = float(np.sqrt(np.mean([
                    x["position_rmse_m"]**2 for x in record["folds"]])))
                record["passed"] = final_checks["passed"] and all(x["physical_geometry"]["passed"] for x in record["folds"])
                if record["passed"] and record["position_cv_rmse_m"] < baseline_score*.99:
                    eligible.append((record, full_spec))
            except (DataUnavailable, ValueError, FloatingPointError) as exc:
                record["reason"] = str(exc)
        if not eligible:
            report["reason"] = "No boundary-stable shared candidate improves position CV by at least 1%"
            return baseline_spec(), report
        best = min(x[0]["position_cv_rmse_m"] for x in eligible)
        options = [x for x in eligible if x[0]["position_cv_rmse_m"] <= best*1.01]
        chosen, spec = max(options, key=lambda x: x[0]["smoothing_scale_m"])
        report.update(selected_kind=spec["kind"], selected_smoothing_scale_m=chosen["smoothing_scale_m"],
                      selected_position_cv_rmse_m=chosen["position_cv_rmse_m"], reason="Position-only CV and geometry checks passed")
        return spec, report
    except (DataUnavailable, ValueError, KeyError, FloatingPointError) as exc:
        report["reason"] = f"Shared position geometry unsupported: {exc}"
        return baseline_spec(), report


def remap_rules(rules, track):
    route = track["continuous_route"]
    if not isinstance(route, SharedPositionRoute):
        from f1ers_v3.geometry import remap_rules as inherited_remap_rules
        return inherited_remap_rules(rules, track)
    result = copy.deepcopy(rules)
    to_route = lambda value: float(route.legacy_to_s(float(value)*route.reference_length_m/rules["fia_centreline_length_m"]))
    result["alternative_power_zones_route_m"] = [[to_route(a), to_route(b)] for a, b in rules["alternative_power_zones_m"]]
    result["power_reduction_exceptions_route"] = [{**z, "start_m": to_route(z["start_m"]), "end_m": to_route(z["end_m"])}
                                                    for z in rules["power_reduction_exceptions"]]
    # The v4 geometry adapter reads this optional field as a fresh FIA mapping.
    # Materialize its coordinates once and remove that second mapping request.
    if "aero_straight_zones_m" in result:
        zones = result.pop("aero_straight_zones_m")
        track["approximate_aero_zones_m"] = [{"start_m": to_route(z["start_m"] if isinstance(z, dict) else z[0]),
                                              "end_m": to_route(z["end_m"] if isinstance(z, dict) else z[1])} for z in zones]
    result["coordinate_transform"] = {"method": "FIA fraction -> frozen legacy reference -> nonlinear shared-curve arc length",
        "route_length_m": route.length_m, "source_length_m": route.reference_length_m,
        "curve_id": route.curve_id, "surveyed": False}
    return result


def build_track_from_spec(root, spec=None, ds_m=5., registration=None):
    root = Path(root)
    spec = spec or baseline_spec()
    registration = registration or read_json(root/"data/v3/local/registration.json")
    if spec.get("kind") == "baseline":
        return inherited_build_track(root, "spline", ds_m, registration)
    if spec.get("kind") != "shared_position_spline" or spec.get("fit_role") != "calibration" or spec.get("fit_laps") != list(CALIBRATION_LAPS):
        raise DataUnavailable("Unknown or non-calibration frozen geometry specification")
    baseline = inherited_build_track(root, "spline", ds_m, registration)
    route = SharedPositionRoute(spec, registration)
    centre = np.loadtxt(root/"data/raw/tum/Spielberg_centerline.csv", delimiter=",")
    check = _validate_route(route, centre, baseline["continuous_route"])
    if not check["passed"]: raise DataUnavailable("Frozen shared geometry failed validation: "+", ".join(check["reasons"]))
    track = route.sample(ds_m)
    xy = np.column_stack([track["x_m"], track["y_m"]])
    left, right = _boundary_clearance(xy, centre, route.direction)
    old = read_json(root/"data/processed/dataset.json")["track"]
    track.update(name=old["name"], width_left_m=left, width_right_m=right,
        candidate="shared_position_spline", line_type="Calibration-only shared position spline",
        corner_s_m={k: float(route.legacy_to_s(v)) for k, v in old["corner_s_m"].items()},
        sector_end_s_m=[float(route.legacy_to_s(v)) for v in old["sector_end_s_m"][:-1]]+[route.length_m],
        approximate_aero_zones_m=[{**z, "start_m": float(route.legacy_to_s(z["start_m"])),
                                  "end_m": float(route.legacy_to_s(z["end_m"]))} for z in old["approximate_aero_zones_m"]],
        fia_s_m=route.s_to_legacy(track["s_m"])*4326./route.reference_length_m,
        geometry_audit={**check, "curve_id": route.curve_id, "length_m": route.length_m,
            "geometry_source": "Shared calibration positions; not a surveyed or per-lap racing line",
            "smoothing_scale_m": spec["smoothing_scale_m"], "mesh_refit": False,
            "fit_laps": list(CALIBRATION_LAPS), "speed_residuals_used": False})
    return track
