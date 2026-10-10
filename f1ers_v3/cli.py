"""Ordered dual-path experiment; test access is the last irreversible boundary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from .artifacts import freeze_configuration, read_json, sha256, write_json
from .reporting import export_trace, render_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "austria_2026_v3"
CANDIDATES = ("alignment", "spline", "anisotropic")


def release_gates(report, mesh, cross):
    """Missing new-session evidence is a closed training gate, never a pass."""
    return {"convergence": bool(report.get("solver_success")),
            "physical": bool(report.get("physical_valid")) and bool(mesh.get("physical_passed")),
            "grid": bool(mesh.get("passed")),
            "development": bool(report.get("development_gate", {}).get("passed")),
            "new_test": bool(cross.get("status") == "evaluated"
                             and cross.get("adaptation", {}).get("solver_success")
                             and cross.get("adaptation", {}).get("physical_valid")
                             and cross.get("evaluation", {}).get("passed"))}


def train_if_released(dataset, vehicle, gates, output):
    if not all(value is True for value in gates.values()):
        return {"skipped": True, "reason": "Required fidelity gates failed or remain pending",
                "gates": gates, "controllers": [], "sac_expanded": False,
                "budget": {"seeds": [0,1,2], "bc_epochs": 600,
                           "dagger_rounds": 3, "dagger_epochs": 600}}
    from .simulator import SimulatorV3
    from .solvers import solve_heuristic_v3, solve_optimal_v3
    from .learning import train_bc_dagger_v3
    sim = SimulatorV3(dataset["track"], dataset["rules"], vehicle)
    teacher = solve_optimal_v3(sim, solve_heuristic_v3(sim))
    result = train_bc_dagger_v3(sim, teacher, output / "models", gates)
    result["teacher"] = teacher
    return result


def _export_paths(output, prefix, comparison):
    export_trace(output / f"trace_{prefix}_autonomous.csv", comparison["autonomous"]["result"])
    for number, result in comparison["replay"]["results"].items():
        export_trace(output / f"trace_{prefix}_replay_lap_{number}.csv", result)


def score_cross_session(root, output, vehicle, candidate, experiment_name, status, progress=print):
    """Only early laps may adapt; late laps open after a second concrete freeze."""
    from .data import load_new_session
    from .protocol import authorize_test_access
    from .calibration import adapt_session, compare_paths
    if status.get("status") != "ready":
        return {"status": "pending", "quality": status,
                "reason": status.get("reason", "FP2 quality-qualified source unavailable"),
                "test_scored": False, "interpretation": "新会话验收待完成"}
    early = load_new_session(root, include_test=False, candidate=candidate, ds_m=5)
    progress("FP2: one-shot adaptation from the three precommitted early laps")
    adapted, adaptation, traces = adapt_session(early, vehicle, progress=progress)
    write_json(output / "condition_adaptation.json", adaptation)
    _export_paths(output, "fp2_adaptation", traces)
    if not (adaptation.get("solver_success") and adaptation.get("physical_valid")):
        return {"status":"deferred", "quality":status, "adaptation":adaptation,
                "adapted_vehicle":adapted.to_dict(), "test_scored":False,
                "reason":"Early condition adaptation did not yield a converged complete physical model; late laps remain sealed",
                "interpretation":"Condition adaptation failed before final test access"}
    freeze_path, freeze = freeze_configuration(root, output, adapted, candidate, "cross_session",
        adaptation=adaptation, experiment_name=experiment_name,
        test_scope="condition-adapted later-lap transfer, not whole-session blind")
    authorize_test_access(root, freeze_path, experiment_name)
    progress("FP2: frozen model, one final late-lap scoring pass")
    test = load_new_session(root, include_test=True, candidate=candidate, ds_m=5)
    test_only = {**test, "laps": [lap for lap in test["laps"] if lap["role"] == "test"]}
    final = compare_paths(test_only, adapted, role="test")
    result = {"status": "evaluated", "quality": status, "adaptation": adaptation,
              "adapted_vehicle": adapted.to_dict(), "evaluation": final,
              "freeze": freeze, "test_scored": True,
              "interpretation": "Condition-adapted cross-session later-lap test; not zero-shot"}
    write_json(output / "test_evaluation.json", result)
    _export_paths(output, "fp2_test", final)
    access_path = root / "data/v3/test_access.json"
    access = read_json(access_path)
    access.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  result_sha256=sha256(output / "test_evaluation.json"))
    write_json(access_path, access)
    return result


def run_v3(args):
    from .data import load_local_dataset, prepare_new_session
    from .calibration import (fit_candidate, select_candidate, mesh_audit,
                              fuel_mass_sensitivity)
    from .vehicle import VehicleParamsV3
    root = Path(args.root).resolve()
    output = root / "results" / args.name
    if (output / "experiment.json").exists():
        raise RuntimeError("Saved v3 experiment exists. Use verify-v3; do not silently refit or rescore it.")
    if (root / "data/v3/test_access.json").exists():
        raise RuntimeError("The FP2 late-lap dataset is already claimed. Aliases cannot rescore it.")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    def progress(message):
        print(message, flush=True)
    progress("[1/7] Audit local sources, freeze FP2 identity and quality rules")
    datasets = [load_local_dataset(root, candidate=kind, ds_m=5) for kind in CANDIDATES]
    previous = VehicleParamsV3.from_dict(read_json(root / "results/austria_2026_v2/vehicle.json"))
    candidates = []; vehicles = []
    for index, (kind, dataset) in enumerate(zip(CANDIDATES, datasets)):
        progress(f"[2/7] Candidate {index+1}/3: {kind}, fixed three-start calibration")
        vehicle, report, comparisons = fit_candidate(dataset, previous, kind=kind, progress=progress,
            maxiter=300, maxfev=600)
        vehicles.append(vehicle)
        entry = {"candidate": kind, "report": report, "dataset": dataset,
                 "calibration": comparisons["calibration"], "development": comparisons["development"]}
        candidates.append(entry)
        candidate_dir = output / kind
        write_json(candidate_dir / "calibration.json", report)
        write_json(candidate_dir / "trajectories.json", comparisons)
        _export_paths(candidate_dir, "development", comparisons["development"])
    return finalize_v3(root, output, args.name, datasets, vehicles, candidates, start, progress,
                       allow_network=not args.offline)


def finalize_v3(root, output, experiment_name, datasets, vehicles, candidates, start, progress=print,
                execution_metadata=None, allow_network=False):
    """Finish the same gates from complete independent candidate calculations."""
    from .calibration import select_candidate, mesh_audit, fuel_mass_sensitivity
    if tuple(entry["candidate"] for entry in candidates) != CANDIDATES:
        raise ValueError("All three precommitted candidates must finish before finalization")
    for entry in candidates:
        report = entry["report"]
        if (report.get("budget") != {"starts":3,"maxiter_per_start":300,"maxfev_per_start":600}
                or len(report.get("multistart", [])) != 3):
            raise ValueError("Only the complete precommitted three-start budget can be finalized")
    selected = select_candidate([entry["report"] for entry in candidates])
    diagnostic_selection = selected is None
    if selected is None:
        selected = min(range(len(candidates)), key=lambda i: (
            candidates[i]["report"]["calibration_objective"],
            candidates[i]["report"]["parameter_count"], i))
    vehicle, dataset, entry = vehicles[selected], datasets[selected], candidates[selected]
    progress(f"[3/7] Fixed-parameter mesh audit for {entry['candidate']}")
    mesh = mesh_audit(dataset, vehicle)
    write_json(output / "mesh.json", mesh)
    progress("[4/7] Fixed 800/825/850 kg sensitivity; no mass selection")
    mass = fuel_mass_sensitivity(dataset, vehicle)
    write_json(output / "mass_sensitivity.json", mass)
    freeze_path, freeze = freeze_configuration(root, output, vehicle, entry["candidate"], "local",
        diagnostic_selection=diagnostic_selection, selected_candidate_index=selected,
        mesh=mesh, model_selection="local calibration/development only")
    progress("[5/7] Cross-session quality, permitted early adaptation and late-lap scoring")
    from .data import prepare_new_session
    new_status = prepare_new_session(root, allow_network=allow_network)
    write_json(output / "new_session_quality.json", new_status)
    if diagnostic_selection and new_status.get("status") == "ready":
        cross = {"status":"deferred", "quality":new_status, "test_scored":False,
                 "reason":"No complete, converged physical candidate; final late laps remain sealed",
                 "interpretation":"Cross-session test requires an eligible frozen model"}
    else:
        cross = score_cross_session(root, output, vehicle, entry["candidate"], experiment_name, new_status, progress)
    gates = release_gates(entry["report"], mesh, cross)
    gates["data"] = bool(dataset["audit"].get("data_gate_passed"))
    progress("[6/7] Check all release gates before numerical teacher and BC/DAgger")
    training = train_if_released(dataset, vehicle, gates, output)
    status = ("双路径与跨会话精度通过" if all(gates.values()) else
              "本地验收未全部通过；新会话验收待完成；策略训练暂停" if cross["status"] == "pending" else
              "精度或物理门槛未通过；策略训练暂停")
    bundle = {"schema_version": 3, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": status, "candidates": candidates, "selected_candidate_index": selected,
              "diagnostic_selection": diagnostic_selection, "selected_vehicle": vehicle.to_dict(),
              "local_dataset": dataset, "audit": dataset["audit"], "mesh": mesh,
              "mass_sensitivity": mass, "cross_session": cross, "training": training,
              "gates": gates, "benefit_claim_allowed": all(gates.values()),
              "runtime": {"elapsed_s": time.perf_counter()-start, "freeze": freeze,
                          "execution": execution_metadata or {"mode":"sequential", "elapsed_scope":"full_experiment"},
                          "freeze_path": freeze_path.relative_to(root).as_posix()},
              "verification": {"browser_visual_verification": "页面交互尚未实测；计算验证独立记录。"}}
    write_json(output / "experiment.json", bundle)
    write_json(output / "summary.json", {key: bundle[key] for key in (
        "schema_version", "status", "selected_candidate_index", "diagnostic_selection", "selected_vehicle",
        "gates", "benefit_claim_allowed", "runtime")})
    progress("[7/7] Save complete/failed trajectories and dual-path report")
    render_report(output, bundle)
    return {"status": status, "gates": gates, "report": str(output / "report.html")}


def audit_v3(args):
    from .data import load_local_dataset
    root = Path(args.root).resolve()
    data = load_local_dataset(root, candidate="alignment", ds_m=5)
    cache = root / "data/v3/fp2/status.json"
    status = read_json(cache) if cache.exists() else {
        "status":"pending", "reason":"FP2 acquisition and coordinate adaptation run after the local model freeze",
        "late_test_scored":False, "session_substitution":False}
    return {"local": data["audit"],
            "new_session": status,
            "test_arrays_loaded": False}


def main():
    parser = argparse.ArgumentParser(description="Stage-three equivalent vehicle and causal ERS fidelity")
    parser.add_argument("command", choices=("audit-v3", "run-v3", "verify-v3"))
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--name", default=EXPERIMENT)
    parser.add_argument("--offline", action="store_true", help="Use cached data; do not request the public API")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Experiment name must be a single directory name")
    if args.command == "run-v3":
        value = run_v3(args)
    elif args.command == "audit-v3":
        value = audit_v3(args)
    else:
        from .verification import verify_v3
        value = verify_v3(args.root, args.name)
    import json
    from .artifacts import jsonable
    print(json.dumps(jsonable(value), ensure_ascii=False, indent=2), flush=True)

