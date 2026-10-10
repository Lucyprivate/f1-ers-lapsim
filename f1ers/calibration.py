"""Separate calibration-only fitting and held-out telemetry diagnostics."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from .model import Simulator, VehicleParams
from .solvers import solve_heuristic


def _lap_arrays(lap):
    return np.asarray(lap["s_m"], dtype=float), np.asarray(lap["speed_mps"], dtype=float)


def compare_lap(result: dict, lap: dict, length_m: float) -> dict:
    s, observed = _lap_arrays(lap)
    predicted = np.interp(np.mod(s, length_m), np.r_[result["s_m"], length_m],
                           np.r_[result["speed_mps"], result["speed_mps"][0]])
    error = predicted - observed
    observed_time = float(lap["lap_time_s"])
    return dict(lap_number=int(lap["lap_number"]), driver_number=int(lap.get("driver_number", 0)),
                role=lap.get("role", "unknown"), speed_rmse_mps=float(np.sqrt(np.mean(error**2))),
                speed_mae_mps=float(np.mean(np.abs(error))),
                speed_rmse_kph=float(np.sqrt(np.mean(error**2)) * 3.6),
                observed_lap_time_s=observed_time, simulated_lap_time_s=result["summary"]["lap_time_s"],
                lap_time_error_s=result["summary"]["lap_time_s"] - observed_time,
                comparison_caveat="Actual ERS, fuel load, tyre state and race traffic are unobserved")


def fit_vehicle(dataset: dict, rules: dict) -> tuple[VehicleParams, dict]:
    track = dataset["track"]
    laps = dataset.get("laps", [])
    calibration = [lap for lap in laps if lap.get("role") == "calibration"]
    validation = [lap for lap in laps if lap.get("role") == "validation"]
    if not calibration:
        raise ValueError("At least one calibration-only lap is required")
    base = VehicleParams()
    # ICE is a fixed 400kW mechanical assumption from a 3000MJ/h thermal flow
    # with 48% efficiency. It is not fitted as observed total power and then
    # supplemented by K, which would count electrical propulsion twice.
    cached = {}
    calibration_evaluations = 0

    def evaluate(x):
        nonlocal calibration_evaluations
        key = tuple(np.round(x, 7))
        if key not in cached:
            params = replace(base, mu=float(x[0]), drag_area_m2=float(x[1]), downforce_area_m2=float(x[2]))
            result = solve_heuristic(Simulator(track, rules, params))
            errors = []
            for lap in calibration:
                s, observed = _lap_arrays(lap)
                predicted = np.interp(np.mod(s, track["length_m"]), np.r_[result["s_m"], track["length_m"]],
                                      np.r_[result["speed_mps"], result["speed_mps"][0]])
                errors.append(np.mean((predicted - observed)**2))
            # Small regularizer communicates/prioritizes the assumed envelope;
            # available telemetry cannot identify a unique aerodynamic/grip set.
            regularization = 0.5 * np.sum(((np.asarray(x) - np.array([1.75, 1.15, 3.6])) /
                                           np.array([0.6, 0.7, 2.0]))**2)
            cached[key] = (float(np.mean(errors)) + regularization, result, params)
            calibration_evaluations += 1
        return cached[key]

    optimization = minimize(lambda x: evaluate(x)[0], [base.mu, base.drag_area_m2, base.downforce_area_m2],
                            method="Nelder-Mead", bounds=[(1.2, 2.4), (0.6, 2.0), (1.8, 6.0)],
                            options={"maxiter": 20, "xatol": 0.04, "fatol": 0.2})
    _, fitted_result, params = evaluate(optimization.x)
    calibration_metrics = [compare_lap(fitted_result, lap, track["length_m"]) for lap in calibration]
    validation_metrics = [compare_lap(fitted_result, lap, track["length_m"]) for lap in validation]
    failures = []
    if not fitted_result["summary"]["feasible"]:
        failures.append("Calibrated vehicle fails physical/rule-subset checks")
    if not validation:
        failures.append("No independent validation lap is available")
    if not fitted_result["summary"]["coupling_converged"]:
        failures.append("Speed-energy coupling did not converge")
    report = dict(method="calibration_only_Nelder_Mead_GGV_aerodynamic_fit",
                  fitted_parameters=["mu", "drag_area_m2", "downforce_area_m2"],
                  calibration_laps=[int(lap["lap_number"]) for lap in calibration],
                  validation_laps=[int(lap["lap_number"]) for lap in validation],
                  calibration_metrics=calibration_metrics, validation_metrics=validation_metrics,
                  ready_for_training=not failures, failure_reasons=failures,
                  solver_success=bool(optimization.success), solver_message=str(optimization.message),
                  calibration_evaluations=calibration_evaluations,
                  physical_check=fitted_result["summary"],
                  fixed_assumptions=base.to_dict(), fitted_vehicle=params.to_dict(),
                  identification_warning="Public speed data do not uniquely identify grip, downforce, drag or ERS strategy",
                  ers_calibration_policy="Disclosed sustainable heuristic; actual team K/SOC unavailable",
                  validation_used_for_fitting=False,
                  no_precommitted_accuracy_gate=True,
                  fit_status="observational calibration, not independent power-unit identification")
    return params, report
