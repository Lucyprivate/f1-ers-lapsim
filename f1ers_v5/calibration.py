"""Precommitted limited candidates, complete-lap fitting and Huber sensitivity."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from f1ers_v3.diagnostics import compare_lap, accuracy_gate, residual_vector, valid_result
from f1ers_v4.calibration import (PARAMETERS as BASE_PARAMETERS, BOUNDS as BASE_BOUNDS,
                                 metrics_only, write_compressed, _failure)
from f1ers_v4.artifacts import write_json

BUDGET = {"screening_points": 64, "screening_seed": 2026, "starts": 3,
          "maxiter_per_start": 300, "maxfev_per_start": 600}


def require_role(dataset, role):
    expected = {"calibration": [3, 4, 5, 6], "development": [7, 9]}
    laps = dataset["laps"]
    if role in ("adaptation", "test"):
        expected[role] = dataset.get("audit", {}).get("selection", {}).get(
            "adaptation_laps" if role == "adaptation" else "test_laps", [])
        if len(expected[role]) != 3:
            raise ValueError("Transfer view must retain the original three selected chronological laps")
    if role not in expected or [int(l["lap_number"]) for l in laps] != expected[role] or any(l["role"] != role for l in laps):
        raise ValueError("A fitting/scoring view must contain exactly its preregistered role")


def compare_paths(dataset, vehicle, role="calibration"):
    from .simulator import SimulatorV5
    require_role(dataset, role)
    sim = SimulatorV5(dataset["track"], dataset["rules"], vehicle)
    try:
        auto = sim.simulate(mode="autonomous")
    except (RuntimeError, ArithmeticError, ValueError) as exc:
        auto = _failure("autonomous", exc)
    replays, am, rm = {}, [], []
    for lap in dataset["laps"]:
        try:
            replay = sim.simulate(mode="replay", driver_trace=lap["driver_trace"])
        except (RuntimeError, ArithmeticError, ValueError) as exc:
            replay = _failure("replay", exc)
        replays[str(lap["lap_number"])] = replay
        am.append(compare_lap(dataset["track"], lap, auto, "autonomous"))
        rm.append(compare_lap(dataset["track"], lap, replay, "replay"))
    count = len(dataset["laps"])
    ac = {"metrics": am, "expected_laps": count, "result": auto}
    rc = {"metrics": rm, "expected_laps": count, "results": replays}
    n = len(dataset["track"]["s_m"])
    ac["gate"] = accuracy_gate(ac, valid_result(auto, n))
    rc["gate"] = accuracy_gate(rc, all(valid_result(r, n) for r in replays.values()))
    comparison = {"role": role, "autonomous": ac, "replay": rc,
                  "physical_valid": all(m["valid"] for m in am + rm),
                  "passed": ac["gate"]["passed"] and rc["gate"]["passed"]}
    residual = residual_vector(comparison)
    comparison["normalized_objective"] = None if residual is None else float(residual @ residual)
    comparison["huber_objective"] = huber_objective(dataset, comparison)
    return comparison


def huber_objective(dataset, comparison):
    """Equal mode/lap/group objective; only speed loss changes, delta=15 km/h."""
    total = 0.
    length = dataset["track"]["length_m"]
    for mode in ("replay", "autonomous"):
        for lap, metric in zip(dataset["laps"], comparison[mode]["metrics"]):
            if not metric.get("valid"):
                continue
            trace = (comparison[mode]["result"] if mode == "autonomous" else
                     comparison[mode]["results"][str(lap["lap_number"])])
            s = np.r_[trace["s_m"], length]
            v = np.r_[trace["speed_mps"], trace["summary"]["speed_end_mps"]]
            errors = (np.interp(lap["s_m"], s, v) - lap["speed_mps"]) * 3.6 / 15.
            z = np.abs(errors)
            speed = float(np.mean(np.where(z <= 1., z*z, 2*z-1.)))
            time = (metric["lap_error_s"] / (.02*metric["real_lap_time_s"]))**2
            sectors = float(np.mean(np.asarray(metric["sector_error_s"])**2))
            total += (speed + time + sectors) / (6*len(dataset["laps"]))
            metric["speed_median_absolute_error_kmh"] = float(np.median(z)*15.)
            metric["speed_p95_absolute_error_kmh"] = float(np.quantile(z, .95)*15.)
    return total if comparison["physical_valid"] else None


def candidates(geometry_spec, diagnosis):
    rows = [{"name": "baseline_remaining", "geometry": {"kind": "baseline"}, "brake_mapping": "remaining"},
            {"name": "baseline_independent", "geometry": {"kind": "baseline"}, "brake_mapping": "independent"}]
    selected = geometry_spec if geometry_spec.get("kind") != "baseline" else {"kind": "baseline"}
    if selected.get("kind") != "baseline":
        rows.extend({"name": "shared_line_" + mapping, "geometry": selected, "brake_mapping": mapping}
                    for mapping in ("remaining", "independent"))
    # Eligibility is derived only from calibration observations, before development.
    gates = diagnosis.get("candidate_gates", diagnosis)
    decay = bool(gates.get("brake_decay_eligible", False))
    traction = bool(gates.get("traction_eligible", False))
    if decay:
        rows.append({"name": "independent_decay", "geometry": selected,
                     "brake_mapping": "independent", "fit_brake_decay": True})
    if traction:
        rows.append({"name": "remaining_traction", "geometry": selected,
                     "brake_mapping": "remaining", "fit_traction": True})
    if decay and traction:
        rows.append({"name": "independent_decay_traction", "geometry": selected,
                     "brake_mapping": "independent", "fit_brake_decay": True, "fit_traction": True})
    return rows


def candidate_bounds(spec):
    names, bounds = list(BASE_PARAMETERS), BASE_BOUNDS.copy().tolist()
    if spec.get("fit_brake_decay"):
        names.append("brake_decay_tau_s"); bounds.append([.1, 5.])
    if spec.get("fit_traction"):
        names.append("traction_tau_s"); bounds.append([.05, 1.])
    return names, np.asarray(bounds, float)


def params_from(spec, values):
    from .adapters import VehicleParamsV5
    names, _ = candidate_bounds(spec)
    return VehicleParamsV5(brake_mapping=spec["brake_mapping"], **dict(zip(names, map(float, values))))


def fit_candidate(dataset, spec, output, progress=print):
    require_role(dataset, "calibration")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    names, bounds = candidate_bounds(spec)
    cache, failures = {}, []

    def evaluate(values, stage):
        values = np.clip(np.asarray(values, float), bounds[:, 0], bounds[:, 1])
        key = tuple(np.round(values, 9))
        if key in cache:
            return cache[key]
        comparison = compare_paths(dataset, params_from(spec, values))
        objective = comparison["normalized_objective"]
        physical = comparison["physical_valid"] and objective is not None
        if not physical:
            path = output / "failures" / f"evaluation_{len(cache):05d}.json.gz"
            write_compressed(path, {"stage": stage, "candidate": spec["name"], "values": values,
                                    "parameters": params_from(spec, values).to_dict(), "comparison": comparison})
            failures.append({"evaluation": len(cache), "stage": stage,
                             "trace_path": path.relative_to(output).as_posix()})
        record = (1e12 if objective is None else float(objective), bool(physical),
                  comparison["huber_objective"])
        cache[key] = record
        return record

    points = qmc.scale(qmc.Sobol(len(names), scramble=True, seed=2026).random_base2(6), bounds[:, 0], bounds[:, 1])
    screening = []
    for i, point in enumerate(points):
        value, physical, robust = evaluate(point, "screening")
        screening.append({"index": i, "parameters": point.tolist(), "physical_valid": physical,
                          "objective": value if physical else None, "huber_objective": robust})
        progress(f"{spec['name']} screening {i+1}/64: {'complete physical' if physical else 'incomplete/invalid'}", flush=True)
    feasible = sorted((s for s in screening if s["physical_valid"]), key=lambda s: (s["objective"], s["index"]))
    fits, robust_fits = [], []
    if len(feasible) >= 3:
        for loss, collection in (("squared", fits), ("huber", robust_fits)):
            for i, initial in enumerate(feasible[:3]):
                def loss_value(x):
                    squared, complete, robust = evaluate(x, f"{loss}_start_{i}")
                    return squared if loss == "squared" or not complete else robust
                result = minimize(loss_value, initial["parameters"], method="Nelder-Mead", bounds=bounds,
                                  options={"maxiter": 300, "maxfev": 600, "xatol": 1e-5, "fatol": 1e-7})
                squared, physical, robust = evaluate(result.x, f"{loss}_start_{i}")
                collection.append({"initial_screening_index": initial["index"], "parameters": result.x.tolist(),
                                   "solver_success": bool(result.success), "physical_valid": physical,
                                   "objective": squared if physical else None, "huber_objective": robust,
                                   "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                                   "message": str(result.message), "loss": loss})
                progress(f"{spec['name']} {loss} start {i+1}/3: {result.message}", flush=True)
    eligible = [r for r in fits if r["solver_success"] and r["physical_valid"]]
    # Only original objective can select the formal fitted parameters.
    best = min(eligible, key=lambda r: r["objective"]) if eligible else None
    values = best["parameters"] if best else bounds.mean(axis=1)
    vehicle = params_from(spec, values)
    comparison = compare_paths(dataset, vehicle)
    family = [r for r in fits if best and r["physical_valid"] and r["objective"] <= best["objective"]*1.01+1e-12]
    report = {"schema_version": 5, "candidate": spec, "budget": BUDGET, "parameter_order": names,
              "bounds": dict(zip(names, bounds.tolist())), "screening": screening,
              "screening_feasible_count": len(feasible), "multistart": fits,
              "huber_sensitivity": {"delta_kmh": 15., "multistart": robust_fits,
                                    "used_for_formal_selection": False,
                                    "status": "evaluated" if robust_fits else "unavailable_no_three_feasible_starts"},
              "solver_success": best is not None, "diagnostic_selection": best is None,
              "fitted_vehicle": vehicle.to_dict(), "near_optimal_parameter_family": family,
              "evaluations": len(cache), "failed_evaluations": failures,
              "reason": "Fewer than three complete physically feasible starts" if len(feasible) < 3 else
                        ("No converged physical squared-loss solution" if best is None else None),
              "calibration_summary": metrics_only(comparison),
              "sensitivity": parameter_sensitivity(dataset, spec, vehicle, output) if best else
                             {"available": False, "reason": "No converged physical reference"}}
    write_json(output / "calibration.json", report)
    write_compressed(output / "calibration_trajectories.json.gz", comparison)
    return vehicle, report, comparison


def parameter_sensitivity(dataset, spec, vehicle, output):
    names, bounds = candidate_bounds(spec)
    columns = []
    for name, (lo, hi) in zip(names, bounds):
        center = getattr(vehicle, name); delta = .002*(hi-lo)
        plus, minus = min(hi, center+delta), max(lo, center-delta)
        comparisons = [compare_paths(dataset, replace(vehicle, **{name: value})) for value in (plus, minus)]
        vectors = [residual_vector(c) for c in comparisons]
        if any(v is None for v in vectors):
            for label, comp in zip(("plus", "minus"), comparisons):
                if not comp["physical_valid"]:
                    write_compressed(Path(output) / "failures" / f"sensitivity_{name}_{label}.json.gz", comp)
            return {"available": False, "reason": "Nonphysical perturbation: " + name}
        columns.append((vectors[0]-vectors[1])/((plus-minus)/(hi-lo)))
    matrix = np.column_stack(columns); singular = np.linalg.svd(matrix, compute_uv=False)
    return {"available": True, "parameter_order": names, "step_fraction": .002,
            "jacobian": matrix, "singular_values": singular,
            "interpretation": "Equivalent parameter sensitivity, not unique vehicle identification"}


def select_candidate(rows):
    eligible = [r for r in rows if r["report"]["solver_success"] and
                r["development"]["physical_valid"] and r["development"]["normalized_objective"] is not None]
    if not eligible:
        return None
    best = min(r["development"]["normalized_objective"] for r in eligible)
    tied = [r for r in eligible if r["development"]["normalized_objective"] <= best*1.01+1e-12]
    return min(tied, key=lambda r: (len(r["report"]["parameter_order"]), r["index"]))
