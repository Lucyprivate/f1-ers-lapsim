"""Ordered v4 experiment: clean source, local gates, freeze, then conditional transfer."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

from f1ers_v3.artifacts import write_json, read_json
from .calibration import fit_model, compare_paths, metrics_only, mass_sensitivity, adapt_session, write_compressed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "austria_2026_v4"


def release_gates(report, numerical, development, data_valid=False):
    return {"data": bool(data_valid), "convergence": bool(report["solver_success"]),
            "physical": bool(report["physical_valid"] and development["physical_valid"]
                             and numerical.get("physical_passed", False)),
            "grid": bool(numerical.get("passed", False)),
            "development": bool(development["passed"])}


def audit(root):
    from .data import load_local_dataset
    from .protocol import ensure_protocol_lock
    from .artifacts import capture_provenance
    data = load_local_dataset(root)
    lock = ensure_protocol_lock(root)
    result = {"schema_version": 4, "candidate": "fixed_periodic_spline",
        "lap_roles": [{"lap_number": l["lap_number"], "role": l["role"]} for l in data["laps"]],
        "geometry": data["track"].get("geometry_audit"), "protocol_lock": lock,
        "provenance": capture_provenance(root, require_clean=False),
        "new_test_opened": False, "training_enabled": False}
    write_json(Path(root)/"results"/EXPERIMENT/"audit.json", result)
    return result


def score_cross_session(root, output, vehicle, freeze_path, gates, name, offline):
    from .data import prepare_new_session, load_new_session
    from .artifacts import freeze_configuration
    from .protocol import (authorize_test_access, complete_test_access,
                           claim_condition_adaptation, complete_condition_adaptation)
    if not all(gates.values()):
        return {"status": "pending", "test_scored": False,
                "reason": "Local fidelity gates failed; FP2 telemetry remains unopened",
                "interpretation": "新会话验收待完成"}
    status = prepare_new_session(root, freeze_path, gates, allow_network=not offline)
    if status.get("status") != "ready":
        return {"status": "pending", "quality": status, "test_scored": False,
                "reason": status.get("reason"), "interpretation": "新会话验收待完成"}
    early = load_new_session(root, include_test=False)
    if not early["audit"]["data_gate_passed"]:
        return {"status": "pending", "test_scored": False,
                "reason": "Early projection quality failed; condition adaptation remains unopened",
                "interpretation": "新会话验收待完成"}
    claim_condition_adaptation(root, freeze_path, name)
    adapted, adaptation, traces = adapt_session(early, vehicle)
    write_json(output/"condition_adaptation.json", adaptation)
    complete_condition_adaptation(root, output/"condition_adaptation.json")
    write_compressed(output/"adaptation_trajectories.json.gz", traces)
    if not (adaptation["solver_success"] and adaptation["physical_valid"]):
        return {"status": "pending", "test_scored": False, "adaptation": adaptation,
                "reason": "Early adaptation did not converge to complete physical laps; late laps sealed",
                "interpretation": "新会话验收待完成"}
    cross_path, cross_freeze = freeze_configuration(root, output, adapted, stage="cross_session",
        adaptation_complete=True, adaptation=adaptation, local_gates=gates,
        parent_local_freeze=read_json(freeze_path), experiment_name=name)
    authorize_test_access(root, cross_path, name)
    test = load_new_session(root, include_test=True)
    test = {**test, "laps": [l for l in test["laps"] if l["role"] == "test"]}
    final = compare_paths(test, adapted, "test")
    final["passed"] = bool(final["passed"] and test["audit"]["data_gate_passed"])
    result = {"status": "evaluated", "test_scored": True, "quality": status,
        "adaptation": adaptation, "adapted_vehicle": adapted.to_dict(),
        "freeze": cross_freeze, "evaluation": metrics_only(final),
        "interpretation": "条件适配后的跨会话测试"}
    write_compressed(output/"test_trajectories.json.gz", final)
    write_json(output/"test_evaluation.json", result)
    complete_test_access(root, output/"test_evaluation.json")
    return result


def run(args):
    from .data import load_local_dataset
    from .artifacts import capture_provenance, freeze_configuration
    from .simulator import SimulatorV4
    from .vehicle import IntegratorConfig
    from .reporting import render_report, export_comparison
    root = Path(args.root).resolve(); output = root/"results"/args.name
    if (output/"experiment.json").exists() or (output/"calibration.json").exists():
        raise RuntimeError("An existing formal run cannot be overwritten; use verify-v4")
    provenance = capture_provenance(root, require_clean=True)
    dataset = load_local_dataset(root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    vehicle, report, comparisons = fit_model(dataset, output)
    print("Fixed original demands: output consistency and integration convergence", flush=True)
    sim = SimulatorV4(dataset["track"], dataset["rules"], vehicle)
    numerical = sim.numerical_audit(driver_trace=next(l["driver_trace"] for l in dataset["laps"]
                                                  if l["role"] == "calibration"))
    if "failed_trajectories" in numerical:
        write_compressed(output/"numerical_failures.json.gz", numerical.pop("failed_trajectories"))
    write_json(output/"numerical_audit.json", numerical)
    write_json(output/"mass_sensitivity.json", mass_sensitivity(dataset, vehicle, output))
    gates = release_gates(report, numerical, comparisons["development"], dataset["audit"]["data_gate_passed"])
    freeze_path, freeze = freeze_configuration(root, output, vehicle, stage="local",
        code_provenance=provenance, local_gates=gates,
        numerics=IntegratorConfig().to_dict(), experiment_name=args.name)
    cross = score_cross_session(root, output, vehicle, freeze_path, gates, args.name, args.offline)
    gates = {**gates, "new_test": bool(cross.get("status") == "evaluated" and cross["evaluation"]["passed"])}
    experiment = {"schema_version": 4, "name": args.name, "vehicle": vehicle.to_dict(),
        "report": report, "local": {r: metrics_only(c) for r, c in comparisons.items()},
        "numerical": numerical, "gates": gates, "cross_session": cross,
        "code_provenance": provenance, "freeze": freeze,
        "freeze_path": freeze_path.relative_to(root).as_posix(),
        "training": {"skipped": True, "reason": "Stage 4 is fidelity-first; no policy training", "sac_expanded": False},
        "benefit_claim_allowed": False, "elapsed_seconds": time.perf_counter()-started,
        "browser_verification": {"status": "not_measured"},
        "historical_v3_correction": "V3 reproduction passed, but endpoint joint tire force and true-time input events were not fully enforced; legacy autonomous feasibility is not a v4 certificate"}
    write_json(output/"experiment.json", experiment)
    write_json(output/"gates.json", gates)
    export_comparison(output, comparisons)
    render_report(output/"report.html", experiment, comparisons, dataset)
    print(json.dumps({"gates": gates, "screening_feasible": report["screening_feasible_count"],
                      "training_run": False, "new_session": cross["status"]}, indent=2), flush=True)
    return experiment


def main():
    parser = argparse.ArgumentParser(description="Independent v4 fidelity and Git audit")
    parser.add_argument("command", choices=("audit-v4", "run-v4", "verify-v4", "restore-artifacts"))
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--name", default=EXPERIMENT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.command == "run-v4":
        run(args); return
    if args.command == "audit-v4":
        result = audit(args.root)
    elif args.command == "verify-v4":
        from .verification import verify
        result = verify(args.root, args.name)
    else:
        if args.archive is None:
            parser.error("restore-artifacts requires --archive")
        from .artifacts import restore_baseline_artifacts
        result = restore_baseline_artifacts(args.root, args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
