"""Reuse the canonical one-shot transfer protocol without dropping v5 structure."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from f1ers_v4 import data as inherited
from f1ers_v4.artifacts import read_json, write_json
from f1ers_v4.protocol import (claim_condition_adaptation, complete_condition_adaptation,
                              authorize_test_access, complete_test_access)
from f1ers_v4.calibration import write_compressed, metrics_only
from .artifacts import check_freeze, freeze
from .calibration import compare_paths, require_role


def load_view(root, geometry_spec, include_test=False):
    """The inherited loader claims late access before any late record is opened."""
    dataset = inherited.load_new_session(root, include_test=include_test)
    if geometry_spec.get("kind") == "baseline":
        return dataset
    from .geometry import build_track_from_spec, remap_rules
    raw = Path(root) / "data/v4/fp2/raw"
    paths = ([raw / n for n in ("laps.json", "car_data_63_medium.json", "location_63_medium.json")]
             if include_test else [raw / "adaptation" / n for n in ("laps.json", "car_data.json", "location.json")])
    # These same source files were hash-checked and authorized by the inherited loader.
    _, car, location = [read_json(p) for p in paths]
    first = min(inherited.epoch(l["date_start"]) for l in dataset["laps"]) - 3
    last = max(inherited.epoch(l["date_start"]) + l["lap_duration"] for l in dataset["laps"]) + 3
    car = [r for r in car if first <= inherited.epoch(r["date"]) <= last]
    location = [r for r in location if first <= inherited.epoch(r["date"]) <= last]
    arrays = inherited.raw_arrays(car, location)
    track = build_track_from_spec(root, geometry_spec, 5., dataset["registration"])
    laps = [inherited.inherited.process_lap(l, arrays, track, dataset["registration"], strict_signal_window=True)
            for l in dataset["laps"]]
    dataset.update(track=track, rules=remap_rules(read_json(Path(root) / "config/rules_austria_2026.json"), track), laps=laps)
    dataset["audit"]["data_gate_passed"] = all(l["alignment_audit"]["grid_bracketed"] and
        .96 <= l["alignment_audit"]["projected_span_fraction"] <= 1.04 and
        np.all(l["scoring_signal_valid_mask"]) for l in laps)
    dataset["audit"]["v5_shape_refitted"] = False
    return dataset


def score_cross_session(root, output, vehicle, geometry_spec, freeze_path, gates, name, offline):
    if not all(gates.values()):
        return {"status": "pending", "test_scored": False, "query_attempted": False,
                "reason": "Local fidelity gates failed; FP2 remains unopened", "interpretation": "新会话验收待完成"}
    check_freeze(root, freeze_path)
    status = inherited.prepare_new_session(root, freeze_path, gates, allow_network=not offline)
    if status.get("status") != "ready":
        return {"status": "pending", "test_scored": False, "quality": status, "interpretation": "新会话验收待完成"}
    early = load_view(root, geometry_spec)
    if not early["audit"]["data_gate_passed"]:
        return {"status": "pending", "test_scored": False, "reason": "Early quality failed; late laps sealed"}
    require_role(early, "adaptation")
    claim_condition_adaptation(root, freeze_path, name)
    # The fitted base power stays frozen within its original bounds. Session
    # scaling is the separate preregistered 0.85–1.15 condition adjustment.
    bounds = np.array([[.85, 1.15], [.85, 1.15]])
    cache = {}
    def objective(x):
        key = tuple(np.round(x, 9))
        if key in cache:
            return cache[key]
        params = replace(vehicle, session_grip_scale=float(x[0]), replay_power_scale=float(x[1]))
        comp = compare_paths(early, params, "adaptation")
        value = comp["normalized_objective"] if comp["physical_valid"] else 1e12
        if not comp["physical_valid"]:
            write_compressed(output / "failures" / f"adaptation_{len(cache):05d}.json.gz",
                             {"parameters": params.to_dict(), "comparison": comp})
        cache[key] = value
        return value
    optimum = minimize(objective, [1., min(1., bounds[1, 1])], method="Nelder-Mead", bounds=bounds,
                       options={"maxiter": 200, "maxfev": 400, "xatol": 1e-5, "fatol": 1e-7})
    adapted = replace(vehicle, session_grip_scale=float(optimum.x[0]), replay_power_scale=float(optimum.x[1]))
    comparison = compare_paths(early, adapted, "adaptation")
    adaptation = {"solver_success": bool(optimum.success), "physical_valid": comparison["physical_valid"],
                  "parameters": {"session_grip_scale": optimum.x[0], "replay_power_scale": optimum.x[1]},
                  "structure_and_geometry_refitted": False, "bounds": bounds.tolist(),
                  "iterations": int(optimum.nit), "evaluations": int(optimum.nfev), "message": str(optimum.message)}
    write_json(output / "condition_adaptation.json", adaptation)
    write_compressed(output / "adaptation_trajectories.json.gz", comparison)
    complete_condition_adaptation(root, output / "condition_adaptation.json")
    if not (optimum.success and comparison["physical_valid"]):
        return {"status": "pending", "test_scored": False, "adaptation": adaptation,
                "reason": "Early adaptation failed; late laps sealed"}
    path, certificate = freeze(root, output, adapted, stage="cross_session", local_gates=gates,
                              geometry_spec=geometry_spec, adaptation_complete=True,
                              parent_local_freeze=read_json(freeze_path), adaptation=adaptation, experiment_name=name)
    check_freeze(root, path)
    authorize_test_access(root, path, name)
    final_view = load_view(root, geometry_spec, include_test=True)
    final = compare_paths(final_view, adapted, "test")
    final["passed"] = bool(final["passed"] and final_view["audit"]["data_gate_passed"])
    result = {"status": "evaluated", "test_scored": True, "adaptation": adaptation,
              "freeze": certificate, "adapted_vehicle": adapted.to_dict(), "evaluation": metrics_only(final),
              "test_quality": final_view["audit"],
              "interpretation": "条件适配后的跨会话测试"}
    write_compressed(output / "test_trajectories.json.gz", final)
    write_json(output / "test_evaluation.json", result)
    complete_test_access(root, output / "test_evaluation.json")
    return result
