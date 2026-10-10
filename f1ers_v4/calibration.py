"""Calibration-only feasible screening followed by bounded three-start fitting."""
from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from f1ers_v3.artifacts import jsonable, write_json
from f1ers_v3.diagnostics import (compare_lap, accuracy_gate, braking_diagnostics,
                                  residual_vector, valid_result)

PARAMETERS = ("mu", "drag_area_m2", "downforce_area_m2", "equivalent_power_kw", "replay_brake_gain")
BOUNDS = np.array([[1.2, 2.4], [.6, 2.0], [1.8, 6.0], [350., 732.5], [.1, 1.]])
BUDGET = {"screening_points": 64, "screening_seed": 2026,
          "starts": 3, "maxiter_per_start": 300, "maxfev_per_start": 600}


def fitting_view(dataset, role="calibration"):
    if role not in ("calibration", "adaptation"):
        raise ValueError("Only calibration or early adaptation may be fitted")
    return {**dataset, "laps": [lap for lap in dataset["laps"] if lap["role"] == role]}


def _failure(mode, exc):
    return {"mode": mode, "s_m": [], "speed_mps": [], "dt_s": [], "time_s": [0.],
            "guard_diagnostics": [], "summary": {"completed_lap": False, "feasible": False,
            "execution_feasible": False, "ers_validated": False,
            "failure_exception": f"{type(exc).__name__}: {exc}"}}


def compare_paths(dataset, vehicle, role="development", diagnostics=True):
    from .simulator import SimulatorV4
    laps = [lap for lap in dataset["laps"] if lap["role"] == role]
    if not laps:
        raise ValueError("Missing precommitted role")
    sim = SimulatorV4(dataset["track"], dataset["rules"], vehicle)
    try:
        auto = sim.simulate(mode="autonomous")
    except (RuntimeError, ArithmeticError, ValueError) as exc:
        auto = _failure("autonomous", exc)
    replay = {}; am = []; rm = []; local = {"autonomous": [], "replay": []}
    for lap in laps:
        try:
            result = sim.simulate(mode="replay", driver_trace=lap["driver_trace"])
        except (RuntimeError, ArithmeticError, ValueError) as exc:
            result = _failure("replay", exc)
        replay[str(lap["lap_number"])] = result
        am.append(compare_lap(dataset["track"], lap, auto, "autonomous"))
        rm.append(compare_lap(dataset["track"], lap, result, "replay"))
        if diagnostics:
            for mode, trace in (("autonomous", auto), ("replay", result)):
                local[mode].append({"lap_number": int(lap["lap_number"]),
                    **braking_diagnostics(dataset["track"], lap, trace)})
    rc = {"metrics": rm, "expected_laps": len(laps), "results": replay}
    ac = {"metrics": am, "expected_laps": len(laps), "result": auto}
    n = len(dataset["track"]["s_m"])
    rc["gate"] = accuracy_gate(rc, all(valid_result(r, n) for r in replay.values()))
    ac["gate"] = accuracy_gate(ac, valid_result(auto, n))
    vector = residual_vector({"replay": rc, "autonomous": ac})
    return {"role": role, "replay": rc, "autonomous": ac,
            "physical_valid": all(m.get("valid", False) for m in am + rm),
            "normalized_objective": None if vector is None else float(vector @ vector),
            "passed": rc["gate"]["passed"] and ac["gate"]["passed"], "diagnostics": local}


def metrics_only(comparison):
    return {key: value for key, value in comparison.items() if key not in ("replay", "autonomous")} | {
        mode: {k: v for k, v in comparison[mode].items() if k not in ("result", "results")}
        for mode in ("replay", "autonomous")}


def write_compressed(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def fit_model(dataset, output, progress=print):
    from .vehicle import VehicleParamsV4
    cal = fitting_view(dataset)
    if len(cal["laps"]) != 4 or {int(l["lap_number"]) for l in cal["laps"]} != {3, 4, 5, 6}:
        raise ValueError("Local fitting must use exactly race laps 3–6")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    base = VehicleParamsV4()
    cache = {}; failures = []; screen = []

    def evaluate(x, stage):
        x = np.clip(np.asarray(x, float), BOUNDS[:, 0], BOUNDS[:, 1])
        key = tuple(np.round(x, 9))
        if key in cache:
            return cache[key]
        vehicle = replace(base, **dict(zip(PARAMETERS, map(float, x))))
        comparison = compare_paths(cal, vehicle, "calibration", diagnostics=False)
        objective = comparison["normalized_objective"]
        complete = bool(comparison["physical_valid"] and objective is not None)
        if complete:
            value = float(objective)
        else:
            # Used only to reject a proposal. Accuracy is never computed for partial laps.
            value = 1e12
            traces = list(comparison["replay"]["results"].values()) + [comparison["autonomous"]["result"]]
            value += sum(1 - min(1., float(r.get("distance_completed_m", 0)) /
                                 float(dataset["track"]["length_m"])) for r in traces) * 1e6
            failure_path = output / "failures" / f"evaluation_{len(cache):05d}.json.gz"
            write_compressed(failure_path, {"parameters": vehicle.to_dict(), "stage": stage,
                                            "comparison": comparison})
            failures.append({"evaluation": len(cache), "stage": stage,
                             "parameters": dict(zip(PARAMETERS, map(float, x))),
                             "trace_path": failure_path.relative_to(output).as_posix()})
        cache[key] = (value, complete)
        return value, complete

    points = qmc.scale(qmc.Sobol(5, scramble=True, seed=2026).random_base2(6), BOUNDS[:, 0], BOUNDS[:, 1])
    for index, point in enumerate(points):
        value, complete = evaluate(point, "screening")
        screen.append({"index": index, "parameters": point.tolist(),
                       "physical_valid": complete, "objective": value if complete else None})
        progress(f"Screening {index+1}/64: {'complete physical' if complete else 'incomplete/invalid'}", flush=True)
    write_json(output / "screening.json", {"budget": BUDGET, "parameter_order": PARAMETERS, "points": screen})
    feasible = sorted([s for s in screen if s["physical_valid"]], key=lambda s: (s["objective"], s["index"]))
    starts = []; solutions = []
    if len(feasible) >= 3:
        for index, entry in enumerate(feasible[:3]):
            optimum = minimize(lambda x: evaluate(x, f"fit_start_{index}")[0], entry["parameters"],
                method="Nelder-Mead", bounds=BOUNDS,
                options={"maxiter": 300, "maxfev": 600, "xatol": 1e-5, "fatol": 1e-7})
            value, physical = evaluate(optimum.x, f"fit_start_{index}")
            row = {"start_screening_index": entry["index"], "initial_parameters": entry["parameters"],
                   "parameters": optimum.x.tolist(), "numeric_success": bool(optimum.success),
                   "physical_valid": physical, "objective": value if physical else None,
                   "iterations": int(optimum.nit), "function_evaluations": int(optimum.nfev),
                   "message": str(optimum.message)}
            starts.append(row)
            if physical:
                solutions.append(row)
            progress(f"Fit start {index+1}/3: {row['message']}", flush=True)
    if solutions:
        eligible = [s for s in solutions if s["numeric_success"]]
        best = min(eligible or solutions, key=lambda s: s["objective"])
        parameters = best["parameters"]
        success = bool(eligible)
    elif feasible:
        best = feasible[0]; parameters = best["parameters"]; success = False
    else:
        # Diagnostic fallback is predeclared, not a selected or calibrated model.
        parameters = BOUNDS.mean(axis=1); success = False
    vehicle = replace(base, **dict(zip(PARAMETERS, map(float, parameters))))
    calibration = compare_paths(cal, vehicle, "calibration")
    development = compare_paths(dataset, vehicle, "development")
    objective = calibration["normalized_objective"]
    family = [s for s in solutions if objective is not None and s["objective"] <= objective * 1.01 + 1e-12]
    sensitivity = parameter_sensitivity(cal, vehicle, output) if calibration["physical_valid"] else {
        "available": False, "reason": "No complete physical calibration reference"}
    report = {"schema_version": 4, "candidate": "fixed_periodic_spline", "budget": BUDGET,
        "parameter_order": PARAMETERS, "bounds": dict(zip(PARAMETERS, BOUNDS.tolist())),
        "fitted_vehicle": vehicle.to_dict(), "screening_feasible_count": len(feasible),
        "multistart": starts, "solver_success": success,
        "diagnostic_selection": not success, "physical_valid": calibration["physical_valid"],
        "reason": "Fewer than three feasible screening points; fitting stopped" if len(feasible) < 3 else None,
        "calibration_objective": objective, "development_objective": development["normalized_objective"],
        "development_gate": {mode: development[mode]["gate"] for mode in ("replay", "autonomous")},
        "near_optimal_parameter_family": family, "sensitivity": sensitivity,
        "evaluations": len(cache), "failed_evaluations": failures,
        "calibration_summary": metrics_only(calibration), "development_summary": metrics_only(development)}
    write_json(output / "calibration.json", report)
    write_compressed(output / "trajectories.json.gz", {"calibration": calibration, "development": development})
    return vehicle, report, {"calibration": calibration, "development": development}


def parameter_sensitivity(cal, vehicle, output=None):
    if any(l["role"] not in ("calibration", "adaptation") for l in cal["laps"]):
        raise ValueError("Sensitivity cannot receive held-out observations")
    columns = []
    for name, (lo, hi) in zip(PARAMETERS, BOUNDS):
        center = getattr(vehicle, name); delta = .002 * (hi-lo)
        plus, minus = min(hi, center+delta), max(lo, center-delta)
        cp = compare_paths(cal, replace(vehicle, **{name: plus}), cal["laps"][0]["role"], False)
        cm = compare_paths(cal, replace(vehicle, **{name: minus}), cal["laps"][0]["role"], False)
        failed_paths = []
        if output is not None:
            for label, comparison in (("plus", cp), ("minus", cm)):
                if not comparison["physical_valid"]:
                    path = Path(output)/"failures"/f"sensitivity_{name}_{label}.json.gz"
                    write_compressed(path, comparison)
                    failed_paths.append(path.relative_to(output).as_posix())
        rp, rm = residual_vector(cp), residual_vector(cm)
        if rp is None or rm is None or rp.shape != rm.shape:
            return {"available": False, "reason": f"Nonphysical perturbed trajectory: {name}",
                    "failure_trace_paths": failed_paths}
        columns.append((rp-rm) / ((plus-minus)/(hi-lo)))
    jac = np.column_stack(columns); singular = np.linalg.svd(jac, compute_uv=False)
    return {"available": True, "parameter_order": PARAMETERS, "step_fraction": .002,
            "jacobian": jac, "singular_values": singular,
            "rank_at_relative_1e_6": int(np.sum(singular > singular[0]*1e-6)),
            "interpretation": "Local equivalent-parameter sensitivity, not true pressure or unique identification"}


def mass_sensitivity(dataset, vehicle, output=None):
    cases = []
    for mass in (800., 825., 850.):
        comparison = compare_paths(dataset, replace(vehicle, mass_kg=mass), "development")
        row = {"mass_kg": mass, "comparison": metrics_only(comparison)}
        if output is not None and not comparison["physical_valid"]:
            path = Path(output)/"failures"/f"mass_{int(mass)}.json.gz"
            write_compressed(path, comparison)
            row["failure_trace_path"] = path.relative_to(output).as_posix()
        cases.append(row)
    return {"reference_mass_kg": 800, "fitted": False, "cases": cases}


def adapt_session(early, base, progress=print):
    if len(early["laps"]) != 3 or any(l["role"] != "adaptation" for l in early["laps"]):
        raise ValueError("Only three precommitted early laps may adapt")
    bounds = np.array([[.85, 1.15], [.85, min(1.15, 732.5/base.equivalent_power_kw)]])
    cache = {}
    def evaluate(x):
        key = tuple(map(float, x))
        if key not in cache:
            vehicle = replace(base, session_grip_scale=float(x[0]), replay_power_scale=float(x[1]))
            comp = compare_paths(early, vehicle, "adaptation", False)
            cache[key] = comp["normalized_objective"] if comp["physical_valid"] else None
        return 1e12 if cache[key] is None else cache[key]
    optimum = minimize(evaluate, np.array([1., min(1., bounds[1, 1])]), method="Nelder-Mead",
                       bounds=bounds, options={"maxiter": 200, "maxfev": 400, "xatol": 1e-5, "fatol": 1e-7})
    adapted = replace(base, session_grip_scale=float(optimum.x[0]), replay_power_scale=float(optimum.x[1]))
    comparison = compare_paths(early, adapted, "adaptation")
    return adapted, {"solver_success": bool(optimum.success), "physical_valid": comparison["physical_valid"],
            "bounds": bounds, "parameters": {"session_grip_scale": optimum.x[0], "replay_power_scale": optimum.x[1]},
            "iterations": int(optimum.nit), "evaluations": int(optimum.nfev), "message": str(optimum.message)}, comparison
