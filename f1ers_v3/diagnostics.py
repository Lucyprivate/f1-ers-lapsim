"""Observable comparisons without filling missing portions of a failed lap.

The braking/apex windows are determined by the frozen route, not speed residuals.
They are diagnostics of equivalent dynamics, not measurements of real brake force.
"""
from __future__ import annotations

import numpy as np

ACCURACY_TARGET = {"mean_speed_rmse_kmh": 15.0,
                   "max_abs_lap_error_pct": 2.0,
                   "max_abs_sector_error_s": 1.0}


def valid_result(result, expected_nodes=None):
    summary = result.get("summary", {})
    required = ("completed_lap", "feasible", "execution_feasible")
    if not all(bool(summary.get(k, False)) for k in required):
        return False
    arrays = [np.asarray(result.get(k, []), float)
              for k in ("s_m", "speed_mps", "dt_s")]
    if not arrays[0].size or any(not np.isfinite(a).all() for a in arrays):
        return False
    n = expected_nodes or arrays[0].size
    if any(a.size != n for a in arrays):
        return False
    if not np.isfinite([summary.get("speed_end_mps", arrays[1][-1]),
                        summary.get("lap_time_s", np.sum(arrays[2]))]).all():
        return False
    return bool(np.all(arrays[1] > 0) and np.all(arrays[2] > 0)
                and np.all(np.diff(arrays[0]) > 0))


def _nodes(result, length):
    s = np.asarray(result["s_m"], float)
    v = np.asarray(result["speed_mps"], float)
    terminal = float(result["summary"].get("speed_end_mps", v[-1]))
    return np.r_[s, length], np.r_[v, terminal], np.r_[0., np.cumsum(result["dt_s"])]


def compare_lap(track, lap, result, mode):
    """Score a complete prediction; a partial result returns no artificial metric."""
    identity = {"lap_number": int(lap["lap_number"]), "role": lap["role"], "mode": mode}
    if not valid_result(result, len(track["s_m"])):
        return {**identity, "valid": False,
                "invalid_reason": "Incomplete, nonfinite or physically invalid trajectory; not scored",
                "completed_segments": len(result.get("s_m", [])),
                "summary": result.get("summary", {})}
    length = float(track["length_m"])
    ss, vv, tt = _nodes(result, length)
    s = np.asarray(lap["s_m"], float)
    real_v = np.asarray(lap["speed_mps"], float)
    real_t = float(lap["lap_time_s"])
    sectors = np.asarray(lap["sector_times_s"], float)
    if (len(s) != len(real_v) or not np.isfinite(np.r_[s, real_v, real_t, sectors]).all()
            or len(sectors) != 3 or real_t <= 0):
        return {**identity, "valid": False, "invalid_reason": "Nonfinite/incomplete observation"}
    predicted = np.interp(s, ss, vv)
    error = predicted - real_v
    observed_time = np.asarray(lap["t_s"], float)
    brake = np.asarray(lap["brake"], float)
    throttle = np.asarray(lap["throttle"], float)
    if (any(len(a) != len(s) or not np.isfinite(a).all() for a in (observed_time,brake,throttle))
            or not np.all(np.diff(s)>0) or not np.all(np.diff(observed_time)>0)
            or not np.all(sectors>0)):
        return {**identity,"valid":False,"invalid_reason":"Unusable time/input observation mapping"}
    boundaries = np.asarray(lap.get("sector_boundaries_m", []), float)
    if boundaries.size != 2:
        boundaries = np.interp(np.cumsum(sectors)[:2], observed_time, s)
    lap_t = float(tt[-1])
    sim_sectors = np.diff(np.r_[0., np.interp(boundaries, ss, tt), lap_t])
    curvature = np.interp(s, track["s_m"], np.abs(track["curvature_1pm"]))
    brake = brake >= .5
    masks = {"braking": brake, "partial_throttle": (throttle < .95) & ~brake,
             "full_throttle_straight": (throttle >= .95) & ~brake & (curvature < .002),
             "corner": curvature >= .002}
    groups = {key: {"samples": int(mask.sum()),
                    "speed_rmse_kmh": float(np.sqrt(np.mean(error[mask] ** 2)) * 3.6) if mask.any() else None,
                    "signed_speed_bias_kmh": float(np.mean(error[mask]) * 3.6) if mask.any() else None}
              for key, mask in masks.items()}
    return {**identity, "valid": True, "real_lap_time_s": real_t, "sim_lap_time_s": lap_t,
            "lap_error_s": lap_t-real_t, "lap_error_pct": 100*(lap_t-real_t)/real_t,
            "speed_rmse_kmh": float(np.sqrt(np.mean(error**2))*3.6),
            "speed_mae_kmh": float(np.mean(np.abs(error))*3.6),
            "signed_speed_bias_kmh": float(np.mean(error)*3.6), "sample_count": len(s),
            "real_sector_times_s": sectors.tolist(), "sim_sector_times_s": sim_sectors.tolist(),
            "sector_error_s": (sim_sectors-sectors).tolist(), "grouped_speed_errors": groups,
            "sector_boundary_method": "frozen observed telemetry time/distance mapping",
            "ers_validated": bool(result.get("summary", {}).get("ers_validated", False))}


def accuracy_gate(comparison, physical=True):
    metrics = comparison.get("metrics", [])
    reasons = []
    expected = comparison.get("expected_laps", len(metrics))
    if not metrics or len(metrics) != expected or any(not m.get("valid", True) for m in metrics):
        reasons.append("All precommitted laps must have complete valid predictions")
    finite = all(len(m.get("sector_error_s", [])) == 3 and
                 np.isfinite([m.get("speed_rmse_kmh", np.nan), m.get("lap_error_pct", np.nan),
                              *m.get("sector_error_s", [np.nan])]).all() for m in metrics)
    if not finite:
        reasons.append("Nonfinite or missing observable error")
    observed = {}
    if metrics and finite:
        observed = {"mean_speed_rmse_kmh": float(np.mean([m["speed_rmse_kmh"] for m in metrics])),
                    "max_abs_lap_error_pct": max(abs(m["lap_error_pct"]) for m in metrics),
                    "max_abs_sector_error_s": max(abs(e) for m in metrics for e in m["sector_error_s"])}
        reasons.extend(f"{k}: {v:.6g} exceeds {ACCURACY_TARGET[k]}"
                       for k, v in observed.items() if v > ACCURACY_TARGET[k])
    if not physical:
        reasons.append("Physical checks failed")
    return {"passed": not reasons, "observed": observed, "thresholds": dict(ACCURACY_TARGET),
            "reasons": reasons, "precommitted": True}


def braking_diagnostics(track, lap, result):
    if not valid_result(result, len(track["s_m"])):
        return {"valid": False, "corners": [], "reason": "Partial prediction retained without invented diagnostics"}
    length = float(track["length_m"])
    s = np.asarray(lap["s_m"], float)
    obs_v = np.asarray(lap["speed_mps"], float)
    ss, vv, _ = _nodes(result, length)
    pred_v = np.interp(s, ss, vv)
    wheel = result.get("wheel_force_n", result.get("wheel_power_kw", np.zeros(len(ss)-1)))
    sim_brake = np.interp(s, result["s_m"], np.asarray(wheel, float)) < -1e-4
    real_brake = np.asarray(lap["brake"], float) >= .5
    corners = []
    for corner, apex in track.get("corner_s_m", {}).items():
        offset = (s-float(apex)+length/2) % length-length/2
        approach = (offset >= -350) & (offset <= 20)
        apex_mask = np.abs(offset) <= 80
        if not approach.any() or not apex_mask.any():
            continue
        def onset(mask):
            active = offset[approach & mask]
            return float(active.min()+float(apex)) % length if active.size else None
        obs_start, pred_start = onset(real_brake), onset(sim_brake)
        # v dv/ds is the spatially differentiated acceleration; no hidden brake-pressure claim.
        order = np.argsort(offset[approach]); local_s = offset[approach][order]
        def decel(v):
            a = np.gradient(v[approach][order]**2 / 2, local_s)
            negative = a[a < 0]
            return float(np.mean(negative)) if negative.size else 0.
        signed = None if obs_start is None or pred_start is None else (pred_start-obs_start+length/2) % length-length/2
        corners.append({"corner": str(corner), "frozen_apex_s_m": float(apex),
                        "observed_brake_onset_s_m": obs_start, "predicted_brake_onset_s_m": pred_start,
                        "brake_onset_error_m": signed, "observed_mean_deceleration_mps2": decel(obs_v),
                        "predicted_mean_deceleration_mps2": decel(pred_v),
                        "observed_apex_speed_kmh": float(obs_v[apex_mask].min()*3.6),
                        "predicted_apex_speed_kmh": float(pred_v[apex_mask].min()*3.6),
                        "apex_speed_error_kmh": float((pred_v[apex_mask].min()-obs_v[apex_mask].min())*3.6)})
    return {"valid": True, "corners": corners, "windows_m": {"approach": [-350, 20], "apex": [-80, 80]},
            "interpretation": "Frozen-route equivalent dynamics diagnostics; actual pressure and braking force unobserved"}


def residual_vector(comparison):
    """Its squared norm equals equal-mode/lap/group normalized squared error."""
    modes = [comparison["replay"], comparison["autonomous"]]
    residual = []
    for mode in modes:
        metrics = mode["metrics"]
        if not metrics or any(not m.get("valid", True) for m in metrics):
            return None
        weight = np.sqrt(2*len(metrics)*3)
        for m in metrics:
            residual.extend([m["speed_rmse_kmh"]/15/weight,
                             m["lap_error_s"]/(.02*m["real_lap_time_s"])/weight])
            residual.extend(np.asarray(m["sector_error_s"])/np.sqrt(3)/weight)
    vector = np.asarray(residual, float)
    return vector if np.isfinite(vector).all() else None
