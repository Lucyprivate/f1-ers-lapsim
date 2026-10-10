"""Calibration-only equivalent dynamics fitting and precommitted dual-path gates."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import minimize

from .vehicle import VehicleParamsV3
from .diagnostics import (accuracy_gate, braking_diagnostics, compare_lap,
                          residual_vector, valid_result)

PARAMETERS = ("mu", "drag_area_m2", "downforce_area_m2", "equivalent_power_kw")
BASE_BOUNDS = np.array([[1.2, 2.4], [.6, 2.0], [1.8, 6.0], [350., 732.5]])
CANDIDATE_ORDER = ("alignment", "spline", "anisotropic")


def _simulator(dataset, vehicle):
    from .simulator import SimulatorV3
    return SimulatorV3(dataset["track"], dataset["rules"], vehicle)


def _autonomous(sim):
    from .solvers import solve_heuristic_v3
    return solve_heuristic_v3(sim)


def compare_paths(dataset, vehicle, role="development", diagnostics=True):
    """One geometry-driven circle and separate open-loop actual-input replays."""
    laps = [lap for lap in dataset["laps"] if lap["role"] == role]
    if not laps:
        raise ValueError(f"No precommitted {role} laps supplied")
    try:
        sim = _simulator(dataset, vehicle)
    except (RuntimeError,FloatingPointError,ArithmeticError,ValueError) as exc:
        return _failure_comparison(dataset,role,exc)
    try:
        auto = _autonomous(sim)
    except (RuntimeError,FloatingPointError,ArithmeticError,ValueError) as exc:
        auto = _failure_result("autonomous",exc)
    replay_results = {}; replay_metrics = []; auto_metrics = []
    local_diagnostics = {"replay": [], "autonomous": []}
    for lap in laps:
        trace = lap["driver_trace"]
        try:
            replay = sim.simulate(mode="replay", driver_trace=trace)
        except (RuntimeError,FloatingPointError,ArithmeticError,ValueError) as exc:
            replay = _failure_result("replay",exc)
        replay_results[str(lap["lap_number"])] = replay
        replay_metrics.append(compare_lap(dataset["track"], lap, replay, "replay"))
        auto_metrics.append(compare_lap(dataset["track"], lap, auto, "autonomous"))
        if diagnostics:
            for mode, result in (("replay", replay), ("autonomous", auto)):
                local_diagnostics[mode].append({"lap_number": int(lap["lap_number"]),
                    **braking_diagnostics(dataset["track"], lap, result)})
    replay_comp = {"metrics": replay_metrics, "expected_laps": len(laps), "results": replay_results}
    auto_comp = {"metrics": auto_metrics, "expected_laps": len(laps), "result": auto}
    replay_comp["gate"] = accuracy_gate(replay_comp,
        all(valid_result(r, len(dataset["track"]["s_m"])) for r in replay_results.values()))
    auto_comp["gate"] = accuracy_gate(auto_comp, valid_result(auto, len(dataset["track"]["s_m"])))
    vector = residual_vector({"replay": replay_comp, "autonomous": auto_comp})
    return {"replay": replay_comp, "autonomous": auto_comp, "role": role,
            "passed": replay_comp["gate"]["passed"] and auto_comp["gate"]["passed"],
            "physical_valid": all(m.get("valid", False) for m in replay_metrics+auto_metrics),
            "normalized_objective": float(vector @ vector) if vector is not None else None,
            "diagnostics": local_diagnostics,
            "interpretation": "Replay verifies equivalent dynamics, autonomous verifies prediction; neither substitutes for the other"}


def metrics_only(comparison):
    """JSON-compatible summary; trajectories are retained by the caller separately."""
    return {k: v for k, v in comparison.items() if k not in ("replay", "autonomous")} | {
        mode: {key: value for key, value in comparison[mode].items() if key not in ("result", "results")}
        for mode in ("replay", "autonomous")}


def _reject_fitting_roles(dataset, allowed):
    forbidden = [lap.get("role") for lap in dataset["laps"] if lap.get("role") not in allowed]
    if forbidden:
        raise ValueError("Held-out, historical, or unknown roles must not be supplied to fitting")


def _failure_comparison(dataset, role, exc):
    laps = [lap for lap in dataset["laps"] if lap["role"] == role]
    result = _failure_result("unavailable",exc)
    metrics = [{"lap_number":l["lap_number"],"role":role,"valid":False,
                "invalid_reason":result["summary"]["failure_exception"]} for l in laps]
    modes = {"replay":{"metrics":metrics,"results":{str(l["lap_number"]):result for l in laps},"gate":accuracy_gate({"metrics":metrics},False)},
             "autonomous":{"metrics":metrics,"result":result,"gate":accuracy_gate({"metrics":metrics},False)}}
    return {**modes,"role":role,"passed":False,"physical_valid":False,"normalized_objective":None,
            "diagnostics":{},"failure_exception":result["summary"]["failure_exception"]}


def _failure_result(mode,exc):
    return {"s_m":np.array([]),"speed_mps":np.array([]),"dt_s":np.array([]),"time_s":np.array([0.]),
            "x_m":np.array([]),"y_m":np.array([]),"wheel_power_kw":np.array([]),"mode":mode,
            "ers_validated":False,"summary":{"completed_lap":False,"feasible":False,
                "execution_feasible":False,"mode":mode,"ers_validated":False,
                "failure_exception":f"{type(exc).__name__}: {exc}"}}


def _safe_compare(dataset, vehicle, role, diagnostics=True):
    try:
        return compare_paths(dataset,vehicle,role,diagnostics)
    except (RuntimeError,FloatingPointError,ArithmeticError,ValueError) as exc:
        return _failure_comparison(dataset,role,exc)


def _constraint_penalty(comparison, node_count):
    """An explicit feasibility surrogate, never a partial accuracy score."""
    results = list(comparison["replay"]["results"].values()) + [comparison["autonomous"]["result"]]
    incomplete = sum(1-min(1,len(result.get("s_m",[]))/node_count) for result in results)/len(results)
    periodic = sum(min(100.,_finite_magnitude(result.get("summary",{}).get("periodic_speed_error_mps",0.),100.))
                   for result in results)
    invalid = sum(not valid_result(result,node_count) for result in results)
    # Guard magnitudes distinguish otherwise flat same-segment stop plateaus.
    # These are constraint violations, never residuals from an uncompleted lap.
    violation = 0.
    for result in results:
        for failure in _guard_records(result.get("summary",{}).get("guard_failures",[])):
            force = _finite_magnitude(failure.get("lateral_tire_force_excess_n",failure.get("tire_force_excess_n",0.)),800.*9.81*10.)
            violation += min(10.,force/(800.*9.81))
    return 1e6 + 1e4*incomplete + 100*invalid + 100*violation + periodic


def _guard_records(value):
    """Legacy guards may be reason strings or nested lists, not force records."""
    if isinstance(value,dict):
        yield value
        for nested in value.values():
            if isinstance(nested,(dict,list,tuple)):
                yield from _guard_records(nested)
    elif isinstance(value,(list,tuple)):
        for nested in value:
            yield from _guard_records(nested)


def _finite_magnitude(value,nonfinite):
    try:
        value=abs(float(value))
    except (TypeError,ValueError):
        return 0.
    return value if np.isfinite(value) else float(nonfinite)


def fit_candidate(dataset, previous=None, kind="alignment", progress=None, maxiter=300, maxfev=600):
    """Three fixed bounded starts; development never influences parameter fitting."""
    if kind not in CANDIDATE_ORDER:
        raise ValueError("Unknown precommitted candidate")
    _reject_fitting_roles(dataset, {"calibration", "development"})
    cal_laps = [lap for lap in dataset["laps"] if lap["role"] == "calibration"]
    if [lap["lap_number"] for lap in cal_laps] != [3, 4, 5, 6]:
        raise ValueError("Exactly fixed calibration laps 3,4,5,6 in chronological order are required")
    cal_data = {**dataset, "laps": cal_laps}
    names = PARAMETERS + (("brake_ratio",) if kind == "anisotropic" else ())
    bounds = np.vstack([BASE_BOUNDS, [.5, 1.2]]) if kind == "anisotropic" else BASE_BOUNDS.copy()
    base = VehicleParamsV3()
    previous = VehicleParamsV3.from_dict(previous) if isinstance(previous, dict) else previous or base
    # Previous v2 supplied no equivalent power or brake-ratio estimate: these start at 600 / 1.
    starts = [np.array([getattr(previous, n, getattr(base, n)) for n in names], float),
              np.array([getattr(base, n) for n in names], float), bounds.mean(axis=1)]
    starts[0][names.index("equivalent_power_kw")] = 600.
    if "brake_ratio" in names:
        starts[0][-1] = 1.
    cache = {}; evaluations = 0; failure_evaluations = []
    def evaluate(x):
        nonlocal evaluations
        key = tuple(np.round(x, 9))
        if key not in cache:
            vehicle = replace(base, **dict(zip(names, map(float, x))))
            comp = _safe_compare(cal_data, vehicle, role="calibration", diagnostics=False)
            objective = comp["normalized_objective"]
            if objective is None or not comp["physical_valid"]:
                # Missing portions are not scored. Failure count gives no apparent partial-lap benefit.
                objective = _constraint_penalty(comp,len(dataset["track"]["s_m"]))
                failure_evaluations.append({"parameters":dict(zip(names,map(float,x))),
                    "reason":comp.get("failure_exception","Incomplete or physically invalid prediction"),
                    "completed_segments":{**{f"replay_{n}":len(r.get("s_m",[])) for n,r in comp["replay"]["results"].items()},
                                          "autonomous":len(comp["autonomous"]["result"].get("s_m",[]))}})
            cache[key] = (float(objective), vehicle, metrics_only(comp))
            evaluations += 1
            if progress and (evaluations == 1 or evaluations % 50 == 0):
                progress(f"{kind}: calibration evaluation {evaluations}, dual-path objective {objective:.5g}")
        return cache[key]
    runs = []; choices = []
    for index, start in enumerate(starts):
        if progress:
            progress(f"{kind}: fixed start {index+1}/3")
        optimum = minimize(lambda x: evaluate(x)[0], np.clip(start, bounds[:, 0], bounds[:, 1]),
                           method="Nelder-Mead", bounds=bounds,
                           options={"maxiter": int(maxiter), "maxfev": int(maxfev),
                                    "xatol": 1e-3, "fatol": 1e-3})
        objective, vehicle, comp = evaluate(optimum.x)
        record = {"start": start.tolist(), "parameters": dict(zip(names, map(float, optimum.x))),
                  "objective": objective, "solver_success": bool(optimum.success),
                  "iterations": int(optimum.nit), "function_evaluations": int(optimum.nfev),
                  "message": str(optimum.message), "physical_valid": bool(comp["physical_valid"])}
        runs.append(record)
        if optimum.success and comp["physical_valid"]:
            choices.append((objective, index, vehicle, comp))
    if choices:
        objective, chosen, vehicle, _ = min(choices, key=lambda item: item[0])
    else:
        objective, vehicle, _ = min(cache.values(), key=lambda item: item[0]); chosen = None
    cal_result = _safe_compare(cal_data, vehicle, "calibration")
    development = _safe_compare(dataset, vehicle, "development")
    vector = residual_vector(cal_result)
    sensitivity = parameter_sensitivity(cal_data, vehicle, names, bounds, vector, progress)
    report = {"candidate": kind, "parameter_count": len(names), "candidate_order": CANDIDATE_ORDER.index(kind),
              "method": "calibration-only, fixed three-start bounded Nelder-Mead",
              "fitted_parameters": list(names), "bounds": dict(zip(names, bounds.tolist())),
              "fitted_vehicle": vehicle.to_dict(), "multistart": runs,
              "selected_converged_start": chosen, "solver_success": chosen is not None,
              "calibration_evaluations": evaluations, "calibration_objective": objective,
              "invalid_evaluation_count":len(failure_evaluations),"invalid_evaluation_samples":failure_evaluations[:20],
              "invalid_objective":"1e6 feasibility base + missing-segment fraction + failure count + normalized failed-step tire excess + periodic mismatch; no partial accuracy score",
              "calibration": metrics_only(cal_result), "development": metrics_only(development),
              "development_objective": development["normalized_objective"],
              "physical_valid": cal_result["physical_valid"] and development["physical_valid"],
              "development_gate": {"passed": development["passed"],
                                   **{m: development[m]["gate"] for m in ("replay", "autonomous")}},
              "near_optimal_parameter_families": [r for r in runs if r["physical_valid"] and r["objective"] <= objective*1.01+1e-12],
              "parameter_sensitivity": sensitivity, "validation_used_for_fitting": False,
              "test_loaded_for_fitting": False, "calibration_laps": [3,4,5,6], "development_laps": [7,9],
              "budget": {"starts": 3, "maxiter_per_start": int(maxiter), "maxfev_per_start": int(maxfev)},
              "identification_warning": "Equivalent parameters; finite-difference SVD does not establish true vehicle identification",
              "objective": "equal paths, equal laps, equal groups: normalized speed RMSE/15, lap error/(2% real time), sector errors/1s"}
    return vehicle, report, {"calibration": cal_result, "development": development}


def parameter_sensitivity(dataset, vehicle, names=PARAMETERS, bounds=BASE_BOUNDS, reference=None, progress=None):
    """Finite-difference Jacobian in normalized parameter coordinates, calibration only."""
    _reject_fitting_roles(dataset, {"calibration", "adaptation"})
    role = "calibration" if any(l["role"] == "calibration" for l in dataset["laps"]) else "adaptation"
    if reference is None:
        reference = residual_vector(_safe_compare(dataset, vehicle, role, diagnostics=False))
    if reference is None:
        return {"available": False, "reason": "No complete calibration reference"}
    columns = []; step = .002
    for name, (lo, hi) in zip(names, bounds):
        value = getattr(vehicle, name); delta = step*(hi-lo)
        plus = min(hi, value+delta); minus = max(lo, value-delta)
        fp = residual_vector(_safe_compare(dataset, replace(vehicle, **{name: plus}), role, diagnostics=False))
        fm = residual_vector(_safe_compare(dataset, replace(vehicle, **{name: minus}), role, diagnostics=False))
        if fp is None or fm is None or fp.shape != reference.shape or fm.shape != reference.shape:
            return {"available": False, "reason": f"Nonfinite/invalid perturbed trajectory for {name}"}
        columns.append((fp-fm)/((plus-minus)/(hi-lo)))
    jacobian = np.column_stack(columns)
    if not np.isfinite(jacobian).all():
        return {"available": False, "reason": "Nonfinite finite-difference Jacobian"}
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = singular[0]*1e-6 if singular.size else 0.
    return {"available": True, "parameters": list(names), "normalized_parameter_step": step,
            "jacobian": jacobian.tolist(), "singular_values": singular.tolist(),
            "rank_at_relative_1e_6": int(np.sum(singular > threshold)),
            "condition_number": float(singular[0]/singular[-1]) if singular[-1] > 1e-12 else None,
            "interpretation": "Local aggregate-observable sensitivity; not proof of uniqueness or true parameter identification"}


def select_candidate(reports):
    """Return the index of the best converged complete candidate, using 1% complexity tie-break."""
    eligible = [(index, report) for index, report in enumerate(reports)
                if report.get("solver_success") and report.get("physical_valid")
                and report.get("development_objective") is not None
                and np.isfinite(report["development_objective"])]
    if not eligible:
        return None
    best = min(report["development_objective"] for _, report in eligible)
    close = [(index, report) for index, report in eligible
             if report["development_objective"] <= best*1.01+1e-12]
    return min(close, key=lambda entry: (entry[1]["parameter_count"], entry[1]["candidate_order"]))[0]


def adapt_session(dataset, selected_vehicle, progress=None, maxiter=150, maxfev=300):
    """Fit only two condition scales from exactly the three early adaptation laps."""
    _reject_fitting_roles(dataset, {"adaptation"})
    if len(dataset["laps"]) != 3:
        raise ValueError("Exactly three precommitted adaptation laps required")
    base = VehicleParamsV3.from_dict(selected_vehicle) if isinstance(selected_vehicle, dict) else selected_vehicle
    names = ("session_grip_scale", "replay_power_scale")
    bounds = np.array([[.85, 1.15], [.85, min(1.15, 732.5/base.equivalent_power_kw)]])
    cache = {}
    def evaluate(x):
        key = tuple(np.round(x, 9))
        if key not in cache:
            candidate = replace(base, **dict(zip(names, map(float, x))))
            comp = _safe_compare(dataset, candidate, "adaptation", diagnostics=False)
            score = comp["normalized_objective"]
            cache[key] = (score if score is not None and comp["physical_valid"] else
                          _constraint_penalty(comp,len(dataset["track"]["s_m"])), candidate, metrics_only(comp))
            if progress and len(cache) % 25 == 0:
                progress(f"FP2 condition adaptation {len(cache)} evaluations")
        return cache[key]
    optimum = minimize(lambda x: evaluate(x)[0], np.array([1., min(1., bounds[1,1])]),
                       method="Nelder-Mead", bounds=bounds,
                       options={"maxiter": int(maxiter), "maxfev": int(maxfev), "xatol":1e-3,"fatol":1e-3})
    objective, vehicle, _ = evaluate(optimum.x)
    results = _safe_compare(dataset, vehicle, "adaptation")
    frozen_fields = {k: v for k, v in base.to_dict().items() if k not in names}
    if any(vehicle.to_dict()[k] != v for k, v in frozen_fields.items()):
        raise AssertionError("Adaptation changed a frozen model parameter")
    report = {"method": "single early-lap conditional adaptation, bounded Nelder-Mead",
              "fitted_parameters": list(names), "bounds": dict(zip(names,bounds.tolist())),
              "fitted_scales": dict(zip(names,map(float,optimum.x))), "frozen_fields": frozen_fields,
              "solver_success": bool(optimum.success), "physical_valid": results["physical_valid"],
              "objective": float(objective), "iterations": int(optimum.nit), "function_evaluations": len(cache),
              "message":str(optimum.message), "adaptation_laps":[l["lap_number"] for l in dataset["laps"]],
              "test_loaded_for_fitting":False, "comparison":metrics_only(results),
              "interpretation":"Condition-adapted cross-session model; not zero-shot transfer"}
    return vehicle, report, results


def fuel_mass_sensitivity(dataset, vehicle, role="development"):
    """Frozen 800/825/850 kg scenarios; never choose a mass from these diagnostics."""
    return {"selection_used": False, "fixed_mass_scenarios_kg": [800.,825.,850.],
            "scenarios": {str(int(mass)): metrics_only(_safe_compare(dataset, replace(vehicle, mass_kg=mass), role))
                          for mass in (800.,825.,850.)},
            "interpretation":"Fuel/mass sensitivity; reference mass remains 800 kg"}


def mesh_audit(dataset, vehicle, laps=None):
    """Resample one frozen curve; freeze driving commands and initial conditions.

    This audit is deliberately distinct from a separately re-optimized mesh.
    Different endpoint speeds are reported, rather than requiring a separate
    periodic initial-speed solution on every mesh and hiding that change.
    """
    from .simulator import SimulatorV3
    track = dataset["track"]
    route = track.get("continuous_route")
    if route is None:
        return {"passed":False,"physical_passed":False,"reasons":["No frozen continuous route supplied"],"scenarios":{}}
    tracks = {ds: {**track, **route.sample(ds)} for ds in (10.,5.,2.5)}
    try:
        baseline_sim = SimulatorV3(tracks[5.], dataset["rules"], vehicle)
        baseline = _autonomous(baseline_sim)
    except (RuntimeError,FloatingPointError,ArithmeticError,ValueError) as exc:
        baseline = _failure_result("autonomous",exc)
    if not valid_result(baseline, len(tracks[5.]["s_m"])):
        return {"passed":False,"physical_passed":False,"reasons":["No complete physical 5m autonomous reference"],
                "scenarios":{},"failed_reference_summary":baseline.get("summary",{}),
                "failed_trajectories":{"autonomous_reference_5m":_retained_trace(baseline)}}
    initial = baseline["summary"]["speed_start_mps"]
    profile = baseline["driver_profile"]
    action_s = np.asarray(baseline["s_m"],float)
    action_v = np.asarray(baseline["requested_actions"],float)
    def actions_for(s):
        ids = np.clip(np.searchsorted(action_s,np.asarray(s),side="right")-1,0,len(action_s)-1)
        return action_v[ids]
    all_results = {"autonomous":{}}
    for ds in (10.,5.,2.5):
        try:
            sim = SimulatorV3(tracks[ds],dataset["rules"],vehicle)
            all_results["autonomous"][ds] = sim.simulate(mode="autonomous",
                controller={"actions":actions_for(tracks[ds]["s_m"]),"driver_profile":profile},
                initial_speed_mps=initial)
        except (RuntimeError,FloatingPointError,ArithmeticError,ValueError) as exc:
            all_results["autonomous"][ds] = _failure_result("autonomous",exc)
    frozen_laps = dataset["laps"] if laps is None else laps
    for lap in frozen_laps:
        key = f"replay_lap_{lap['lap_number']}"; all_results[key] = {}
        for ds in (10.,5.,2.5):
            try:
                sim = SimulatorV3(tracks[ds],dataset["rules"],vehicle)
                all_results[key][ds] = sim.simulate(mode="replay",driver_trace=lap["driver_trace"])
            except (RuntimeError,FloatingPointError,ArithmeticError,ValueError) as exc:
                all_results[key][ds] = _failure_result("replay",exc)
    common_s = np.asarray(tracks[2.5]["s_m"],float)
    summary = {}; reasons=[]; failed_traces={}
    for scenario, results in all_results.items():
        physical={}
        for ds,result in results.items():
            s = result.get("summary",{})
            # Execution certification covers wheel/energy/slew constraints. A
            # mesh-specific periodic solve would violate fixed initial conditions.
            physical[ds] = bool(s.get("execution_feasible") and s.get("completed_lap")
                and len(result.get("s_m",[]))==len(tracks[ds]["s_m"])
                and all(np.isfinite(result.get(k,[])).all() for k in ("speed_mps","dt_s")))
        differences = {}
        if all(physical.values()):
            for coarse, fine in ((10.,5.),(5.,2.5)):
                cs,cv,_ = _mesh_nodes(results[coarse],track["length_m"])
                fs,fv,_ = _mesh_nodes(results[fine],track["length_m"])
                rmse=float(np.sqrt(np.mean((np.interp(common_s,cs,cv)-np.interp(common_s,fs,fv))**2))*3.6)
                time_error=abs(results[coarse]["summary"]["lap_time_s"]-results[fine]["summary"]["lap_time_s"])
                differences[f"{coarse:g}_to_{fine:g}"]={"lap_time_difference_s":float(time_error),"speed_curve_rmse_kmh":rmse}
        acceptance = differences.get("5_to_2.5",{})
        passed = bool(all(physical.values()) and acceptance.get("lap_time_difference_s",np.inf)<=.1
                      and acceptance.get("speed_curve_rmse_kmh",np.inf)<=.5)
        if not passed:reasons.append(f"{scenario}: physical or 5→2.5m convergence failed")
        for ds,okay in physical.items():
            if not okay:failed_traces[f"{scenario}_{ds:g}m"]=_retained_trace(results[ds])
        summary[scenario]={"passed":passed,"physical_valid":{str(ds):value for ds,value in physical.items()},
            "meshes":{str(ds):result["summary"] for ds,result in results.items()},"differences":differences}
    return {"passed":not reasons,"physical_passed":all(all(s["physical_valid"].values()) for s in summary.values()),
            "reasons":reasons,"scenarios":summary,"failed_trajectories":failed_traces,
            "thresholds":{"lap_time_difference_s":.1,"speed_curve_rmse_kmh":.5},
            "curve_id":route.curve_id,"requested_mesh_m":[10.,5.,2.5],"formal_mesh_m":5.,
            "fixed_curve_parameters_control_initial":True,"driver_profile_source":"frozen 5m autonomous reference",
            "autonomous_initial_speed_mps":float(initial),"per_mesh_refit":False,
            "endpoint_speed_check":"recorded; no separate periodic initial condition fitted on each mesh"}


def _mesh_nodes(result,length):
    return (np.r_[result["s_m"],length],
            np.r_[result["speed_mps"],result["summary"]["speed_end_mps"]],
            np.r_[0.,np.cumsum(result["dt_s"])])


def _retained_trace(result):
    return {key:np.asarray(result[key]).tolist() for key in
            ("s_m","speed_mps","dt_s","time_s","wheel_force_n","wheel_power_kw","requested_actions")
            if result.get(key) is not None} | {"summary":result.get("summary",{}),"partial_lap_not_interpolated":True}
