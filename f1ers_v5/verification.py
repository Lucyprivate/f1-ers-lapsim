"""Exact offline local reproduction and saved-score-only transfer verification."""
import gzip
import json
from pathlib import Path
from f1ers_v3.verification import assert_diagnostic_equal
from f1ers_v4.artifacts import read_json, write_json, sha256
from .artifacts import check_freeze
from .calibration import compare_paths, metrics_only


def validate_experiment(experiment, certificate, candidate_plan):
    if experiment.get("schema_version") != 5 or certificate != experiment.get("freeze"):
        raise ValueError("Saved experiment differs from its freeze")
    recorded = experiment.get("code_provenance", {})
    if not recorded or any(certificate.get(key) != value for key, value in recorded.items()):
        raise ValueError("Reported code provenance differs from frozen provenance")
    if certificate["vehicle"] != experiment["vehicle"] or experiment["report"]["fitted_vehicle"] != experiment["vehicle"]:
        raise ValueError("Frozen and reported parameters differ")
    if experiment.get("geometry_spec") != certificate.get("geometry_spec"):
        raise ValueError("Reported geometry differs from the frozen shape")
    local_keys = ("data", "convergence", "physical", "grid", "development")
    if {k: experiment["gates"].get(k) for k in local_keys} != certificate["local_gates"]:
        raise ValueError("Reported local gates differ from frozen gates")
    frozen = certificate["selected_candidate"]
    if experiment.get("displayed_diagnostic_candidate") != frozen["name"]:
        raise ValueError("Displayed candidate differs from frozen candidate")
    expected_selection = frozen["name"] if certificate["local_gates"]["convergence"] else None
    if experiment.get("selected_candidate") != expected_selection:
        raise ValueError("Failed diagnostic candidate was promoted to a fitted model")
    if experiment.get("candidate_plan") != candidate_plan["candidates"]:
        raise ValueError("Reported candidate plan differs from preregistered candidates")
    cross = experiment["cross_session"]
    expected_test = cross.get("status") == "evaluated" and bool(cross["evaluation"]["passed"])
    if experiment["gates"].get("new_test") is not bool(expected_test):
        raise ValueError("Reported final gate differs from saved test status")
    if experiment.get("training") != {"skipped": True, "sac_expanded": False} or experiment.get("benefit_claim_allowed") is not False:
        raise ValueError("Stage 5 cannot claim training or strategy benefits")


def verify(root, name="austria_2026_v5", save=True):
    from .data import load_dataset
    from .adapters import VehicleParamsV5
    root = Path(root).resolve(); output = root / "results" / name
    experiment = read_json(output / "experiment.json")
    certificate = read_json(root / experiment["freeze_path"])
    validate_experiment(experiment, certificate, read_json(output / "candidate_plan.json"))
    check_freeze(root, certificate)
    for key, filename in (("numerical", "numerical_audit.json"), ("diagnosis", "diagnosis_selected_geometry.json"),
                          ("geometry_selection", "geometry_selection.json")):
        if experiment[key] != read_json(output / filename):
            raise ValueError("Reported " + key + " differs from frozen calculations")
    expected_summaries = []
    for spec in experiment["candidate_plan"]:
        folder = output / "candidates" / spec["name"]
        report = read_json(folder / "calibration.json")
        if spec["name"] == certificate["selected_candidate"]["name"] and report != experiment["report"]:
            raise ValueError("Displayed report differs from frozen candidate calculation")
        params = VehicleParamsV5.from_dict(report["fitted_vehicle"])
        for role in ("calibration", "development"):
            actual = compare_paths(load_dataset(root, spec["geometry"], role=role), params, role)
            with gzip.open(folder / f"{role}_trajectories.json.gz", "rt", encoding="utf-8") as stream:
                stored = json.load(stream)
            assert_diagnostic_equal(actual, stored, spec["name"] + "/" + role)
            expected = report["calibration_summary"] if role == "calibration" else read_json(folder / "development.json")
            assert_diagnostic_equal(metrics_only(actual), expected, spec["name"] + "/" + role + "/metrics")
            if spec["name"] == certificate["selected_candidate"]["name"]:
                assert_diagnostic_equal(metrics_only(actual), experiment["local"][role], "displayed/" + role)
        expected_summaries.append({"name": spec["name"], "parameter_count": len(report["parameter_order"]),
                                   "screening_feasible_count": report["screening_feasible_count"],
                                   "solver_success": report["solver_success"], "diagnostic_selection": report["diagnostic_selection"],
                                   "calibration": report["calibration_summary"],
                                   "development": read_json(folder / "development.json")})
    if expected_summaries != experiment["candidate_summaries"]:
        raise ValueError("Candidate summaries differ from frozen calculations")
    if experiment["cross_session"].get("status") == "evaluated":
        from f1ers_v4.protocol import read_saved_test_score
        receipt = read_saved_test_score(root, output / "test_evaluation.json")
        if receipt is None:
            raise ValueError("Saved final score receipt missing")
        check_freeze(root, experiment["cross_session"]["freeze"])
        assert_diagnostic_equal(receipt["score"], experiment["cross_session"], "saved_final_score")
        if receipt["trajectory_sha256"] != sha256(output / "test_trajectories.json.gz"):
            raise ValueError("Saved final trace changed")
    check_freeze(root, certificate)
    result = {"schema_version": 5, "artifact_integrity_passed": True, "exact_local_replay_passed": True,
              "candidate_count": len(experiment["candidate_plan"]), "code_commit": experiment["code_provenance"]["code_commit"],
              "fidelity_gates": experiment["gates"], "new_session_status": experiment["cross_session"]["status"],
              "late_arrays_reloaded": False, "training_run": False,
              "interpretation": "Reproduced every fixed-parameter calibration/development candidate including failures; final score read from saved receipt only"}
    if save: write_json(output / "verification.json", result)
    return result
