"""Offline-first diagnostics followed by bounded, individually gated candidates."""
from __future__ import annotations
import argparse
from dataclasses import replace
import html
import json
from pathlib import Path
import time

from f1ers_v4.artifacts import read_json, write_json, sha256
from f1ers_v4.calibration import metrics_only, write_compressed
from f1ers_v4.vehicle import IntegratorConfig
from .artifacts import provenance, freeze
from .calibration import compare_paths, candidates, fit_candidate, select_candidate, BASE_BOUNDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "austria_2026_v5"


def diagnostic_stage(root, output):
    from .data import load_dataset
    from .geometry import prepare_geometry
    from .diagnostics import diagnose
    from .adapters import VehicleParamsV5
    calibration = load_dataset(root, role="calibration")
    vehicle = VehicleParamsV5(**dict(zip(("mu", "drag_area_m2", "downforce_area_m2", "equivalent_power_kw",
                                        "replay_brake_gain"), BASE_BOUNDS.mean(axis=1))))
    baseline = compare_paths(calibration, vehicle)
    diagnosis = diagnose(calibration, baseline_comparison=baseline)
    spec, geometry = prepare_geometry(calibration)
    write_json(output / "diagnosis_baseline.json", diagnosis)
    write_json(output / "geometry_selection.json", geometry)
    write_json(output / "geometry_spec.json", spec)
    write_compressed(output / "diagnostic_trajectories.json.gz", baseline)
    if spec.get("kind") != "baseline":
        selected = load_dataset(root, geometry_spec=spec, role="calibration")
        selected_comparison = compare_paths(selected, vehicle)
        diagnosis = diagnose(selected, baseline_comparison=selected_comparison)
        write_compressed(output / "diagnostic_shared_line_trajectories.json.gz", selected_comparison)
    write_json(output / "diagnosis_selected_geometry.json", diagnosis)
    return spec, geometry, diagnosis


def diagnose_command(root, name=EXPERIMENT):
    output = Path(root) / "results" / name
    if any((output / item).exists() for item in ("experiment.json", "frozen_configuration.json", "candidates")):
        raise RuntimeError("Do not overwrite a frozen experiment; use verify-v5")
    output.mkdir(parents=True, exist_ok=True)
    spec, geometry, diagnosis = diagnostic_stage(root, output)
    return {"schema_version": 5, "calibration_laps": [3, 4, 5, 6], "geometry_kind": spec["kind"],
            "brake_decay_eligible": diagnosis.get("brake_decay_eligible", False),
            "traction_eligible": diagnosis.get("traction_eligible", False),
            "new_test_opened": False, "training_run": False,
            "geometry": geometry, "hypotheses": diagnosis.get("hypotheses_ledger", [])}


def render(output, experiment, comparisons, dataset):
    # Reuse the checked plotting implementation without editing its frozen source.
    from f1ers_v4.reporting import render_report, export_comparison
    path = output / "report.html"
    render_report(path, experiment, comparisons, dataset)
    export_comparison(output, comparisons)
    text = path.read_text(encoding="utf-8")
    text = text.replace("<title>F1 ERS v4 模型验收</title>", "<title>F1 ERS v5 结构误差与模型验收</title>")
    text = text.replace("<h1>F1 ERS · v4 模型验收</h1>", "<h1>F1 ERS · v5 结构误差与模型验收</h1>")
    summary = {"geometry": experiment["geometry_selection"], "candidate_gates": experiment["candidate_gates"],
               "hypotheses": experiment["diagnosis"].get("hypotheses_ledger", []),
               "candidate_results": experiment["candidate_summaries"]}
    section = "<section><h2>校准圈归因与有限候选</h2><p>诊断不参与途中状态更新；中位误差不替代 RMSE 验收。</p><pre>" + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + "</pre></section>"
    text = text.replace("<section><h2>放行状态</h2>", section + "<section><h2>放行状态</h2>")
    path.write_text(text, encoding="utf-8")


def run(args):
    from .data import load_dataset
    from .simulator import SimulatorV5
    from .transfer import score_cross_session
    root = Path(args.root).resolve(); output = root / "results" / args.name
    if (output / "experiment.json").exists() or (output / "frozen_configuration.json").exists() or (output / "candidates").exists():
        raise RuntimeError("A formal v5 run cannot be overwritten; use verify-v5")
    captured = provenance(root, require_clean=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spec, geometry, diagnosis = diagnostic_stage(root, output)
    planned = candidates(spec, diagnosis)
    write_json(output / "candidate_plan.json", {"candidates": planned, "frozen_before_development": True,
                                               "eligibility_fit_laps": [3, 4, 5, 6]})
    rows = []
    for index, candidate in enumerate(planned):
        calibration = load_dataset(root, candidate["geometry"], role="calibration")
        vehicle, report, comparison = fit_candidate(calibration, candidate, output / "candidates" / candidate["name"])
        development_data = load_dataset(root, candidate["geometry"], role="development")
        development = compare_paths(development_data, vehicle, "development")
        write_json(output / "candidates" / candidate["name"] / "development.json", metrics_only(development))
        write_compressed(output / "candidates" / candidate["name"] / "development_trajectories.json.gz", development)
        rows.append({"index": index, "spec": candidate, "vehicle": vehicle, "report": report,
                     "calibration": comparison, "development": development})
        print(f"Candidate {candidate['name']}: {report['screening_feasible_count']}/64 complete feasible starts", flush=True)
    selected = select_candidate(rows)
    # Diagnostic fallback is fixed candidate zero, never a failed fit promoted to a model.
    displayed = selected or rows[0]
    vehicle = displayed["vehicle"]; selected_spec = displayed["spec"]["geometry"]
    data = load_dataset(root, selected_spec, role="calibration")
    development_data = load_dataset(root, selected_spec, role="development")
    simulator = SimulatorV5(data["track"], data["rules"], vehicle)
    numerical = simulator.numerical_audit(driver_trace=data["laps"][0]["driver_trace"])
    if "failed_trajectories" in numerical:
        write_compressed(output / "numerical_failures.json.gz", numerical.pop("failed_trajectories"))
    write_json(output / "numerical_audit.json", numerical)
    mass = []
    for kg in (800., 825., 850.):
        case = compare_paths(data, replace(vehicle, mass_kg=kg))
        mass.append({"mass_kg": kg, "calibration": metrics_only(case), "used_for_selection": False})
        write_compressed(output / "mass" / f"{int(kg)}kg.json.gz", case)
    write_json(output / "mass_sensitivity.json", {"fitted": False, "reference_mass_kg": 800, "cases": mass})
    local = {"calibration": displayed["calibration"], "development": displayed["development"]}
    gates = {"data": bool(data["audit"]["data_gate_passed"] and development_data["audit"]["data_gate_passed"]),
             "convergence": selected is not None,
             "physical": bool(displayed["calibration"]["physical_valid"] and displayed["development"]["physical_valid"] and numerical.get("physical_passed")),
             "grid": bool(numerical.get("passed")), "development": bool(displayed["development"]["passed"])}
    path, certificate = freeze(root, output, vehicle, code_provenance=captured, local_gates=gates,
                              geometry_spec=selected_spec, selected_candidate=displayed["spec"],
                              numerics=IntegratorConfig().to_dict(), experiment_name=args.name)
    cross = score_cross_session(root, output, vehicle, selected_spec, path, gates, args.name, args.offline)
    all_gates = {**gates, "new_test": bool(cross.get("status") == "evaluated" and cross["evaluation"]["passed"])}
    summaries = [{"name": r["spec"]["name"], "parameter_count": len(r["report"]["parameter_order"]),
                  "screening_feasible_count": r["report"]["screening_feasible_count"],
                  "solver_success": r["report"]["solver_success"], "diagnostic_selection": r["report"]["diagnostic_selection"],
                  "calibration": r["report"]["calibration_summary"], "development": metrics_only(r["development"])} for r in rows]
    experiment = {"schema_version": 5, "name": args.name, "code_provenance": captured,
                  "vehicle": vehicle.to_dict(), "report": displayed["report"], "local": {k: metrics_only(v) for k, v in local.items()},
                  "geometry_selection": geometry, "geometry_spec": selected_spec, "diagnosis": diagnosis,
                  "candidate_gates": {"brake_decay_eligible": diagnosis.get("brake_decay_eligible", False),
                                      "traction_eligible": diagnosis.get("traction_eligible", False)},
                  "candidate_plan": planned, "candidate_summaries": summaries,
                  "selected_candidate": selected["spec"]["name"] if selected else None,
                  "displayed_diagnostic_candidate": displayed["spec"]["name"], "numerical": numerical,
                  "gates": all_gates, "freeze_path": path.relative_to(root).as_posix(), "freeze": certificate,
                  "cross_session": cross, "training": {"skipped": True, "sac_expanded": False},
                  "benefit_claim_allowed": False, "elapsed_seconds": time.perf_counter()-started,
                  "browser_verification": {"status": "not_measured", "actual_browser_interaction": False}}
    write_json(output / "experiment.json", experiment)
    write_json(output / "gates.json", all_gates)
    write_compressed(output / "trajectories.json.gz", local)
    # Both display roles share the selected frozen geometry.
    plotting_data = {**data, "laps": data["laps"]+development_data["laps"]}
    render(output, experiment, local, plotting_data)
    print(json.dumps({"gates": all_gates, "selected_candidate": experiment["selected_candidate"],
                      "candidates": [{"name": s["name"], "feasible": s["screening_feasible_count"]} for s in summaries],
                      "new_session": cross["status"], "training_run": False}, ensure_ascii=False, indent=2), flush=True)
    return experiment


def main():
    parser = argparse.ArgumentParser(description="v5 structure diagnostics and limited model experiments")
    parser.add_argument("command", choices=("diagnose-v5", "run-v5", "verify-v5"))
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--name", default=EXPERIMENT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    # Prevent names from escaping the ignored result tree.
    if not args.name or Path(args.name).name != args.name or any(c in args.name for c in ("/", "\\", ":")) or args.name in (".", ".."):
        parser.error("Experiment name must be one local directory name")
    if args.command == "diagnose-v5":
        result = diagnose_command(args.root, args.name)
    elif args.command == "run-v5":
        run(args); return
    else:
        from .verification import verify
        result = verify(args.root, args.name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
