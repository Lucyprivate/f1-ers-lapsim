"""Offline exact replay, separating implementation integrity from fidelity."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .artifacts import check_freeze, jsonable, read_json, sha256, write_json


def assert_diagnostic_equal(actual, stored, label):
    """Verify rejected proposals as well as accepted trajectories."""
    actual, stored = jsonable(actual), jsonable(stored)
    if isinstance(actual, dict) and isinstance(stored, dict):
        if actual.keys() != stored.keys():
            raise ValueError(f"Diagnostic keys changed: {label}")
        for key in actual:
            assert_diagnostic_equal(actual[key], stored[key], label+"/"+key)
    elif isinstance(actual, list) and isinstance(stored, list):
        if len(actual) != len(stored):
            raise ValueError(f"Diagnostic length changed: {label}")
        for i, (left, right) in enumerate(zip(actual, stored)):
            assert_diagnostic_equal(left, right, f"{label}/{i}")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(stored, (int, float)):
        np.testing.assert_allclose(actual, stored, atol=1e-8, rtol=0, err_msg=label)
    elif actual != stored:
        raise ValueError(f"Diagnostic changed: {label}")


def assert_trace_equal(actual, stored, label):
    for key in ("s_m", "speed_mps", "dt_s", "time_s", "energy_mj", "mguk_dc_kw",
                "equivalent_drive_kw", "equivalent_brake_force_n"):
        if actual.get(key) is None or stored.get(key) is None:
            if actual.get(key) is not None or stored.get(key) is not None:
                raise ValueError(f"Replay field applicability changed: {label}/{key}")
            continue
        np.testing.assert_allclose(actual[key], stored[key], atol=1e-8, rtol=0,
                                   err_msg=f"{label}/{key}")
    for key in ("completed_lap", "execution_feasible", "feasible", "ers_validated"):
        if actual["summary"].get(key) != stored["summary"].get(key):
            raise ValueError(f"Replay status changed: {label}/{key}")
    for key in ("lap_time_s", "failure_exception", "guard_failures"):
        assert_diagnostic_equal(actual["summary"].get(key), stored["summary"].get(key), label+"/"+key)
    assert_diagnostic_equal(actual.get("guard_diagnostics"), stored.get("guard_diagnostics"),
                            label+"/guard_diagnostics")


def assert_model_freeze(root, bundle):
    freeze = bundle["runtime"]["freeze"]
    path = Path(root) / bundle["runtime"]["freeze_path"]
    if read_json(path) != freeze:
        raise ValueError("Saved and embedded local configuration differ")
    selected = bundle["candidates"][bundle["selected_candidate_index"]]
    if (freeze["candidate"] != selected["candidate"] or
            freeze["vehicle"] != bundle["selected_vehicle"] or
            selected["report"]["fitted_vehicle"] != bundle["selected_vehicle"] or
            freeze["selected_candidate_index"] != bundle["selected_candidate_index"] or
            freeze["diagnostic_selection"] != bundle["diagnostic_selection"]):
        raise ValueError("Selected model differs from its frozen configuration")
    check_freeze(root, freeze)


def verify_v3(root, experiment_name="austria_2026_v3", save=True):
    from .data import load_local_dataset
    from .simulator import SimulatorV3
    from .solvers import solve_heuristic_v3
    from .calibration import _safe_compare
    from .vehicle import VehicleParamsV3
    root = Path(root).resolve()
    output = root / "results" / experiment_name
    bundle = read_json(output / "experiment.json")
    assert_model_freeze(root, bundle)
    checks = []
    for entry in bundle["candidates"]:
        kind = entry["candidate"]
        if read_json(output / kind / "calibration.json") != entry["report"]:
            raise ValueError(f"Candidate report differs from the saved calculation: {kind}")
        comparisons = read_json(output / kind / "trajectories.json")
        if any(comparisons[role] != entry[role] for role in ("calibration", "development")):
            raise ValueError(f"Candidate traces differ from the saved calculation: {kind}")
        data = load_local_dataset(root, kind, 5)
        vehicle = VehicleParamsV3.from_dict(entry["report"]["fitted_vehicle"])
        for role in ("calibration", "development"):
            recorded = entry[role]
            replayed = _safe_compare(data, vehicle, role=role, diagnostics=False)
            assert_trace_equal(replayed["autonomous"]["result"], recorded["autonomous"]["result"], kind+"/"+role+"/autonomous")
            for lap in data["laps"]:
                if lap["role"] != role:
                    continue
                actual = replayed["replay"]["results"][str(lap["lap_number"])]
                assert_trace_equal(actual, recorded["replay"]["results"][str(lap["lap_number"])],
                                   f"{kind}/{role}/replay/{lap['lap_number']}")
        checks.append({"candidate": kind, "all_local_traces_match": True,
                       "retained_partial_failures_match": True})
    cross = bundle["cross_session"]
    if cross["status"] == "evaluated":
        if read_json(output / "frozen_configuration_cross_session.json") != cross["freeze"]:
            raise ValueError("Saved and embedded cross-session configuration differ")
        if cross["adapted_vehicle"] != cross["freeze"]["vehicle"]:
            raise ValueError("Adapted model differs from the cross-session freeze")
        check_freeze(root, cross["freeze"])
        access = read_json(root / "data/v3/test_access.json")
        if access.get("status") != "completed" or access.get("result_sha256") != sha256(output / "test_evaluation.json"):
            raise ValueError("Saved final-test scoring record changed")
        # Saved late-lap results are integrity-checked; the raw late dataset is
        # never opened a second time by this command.
        checks.append({"cross_session_saved_score_sha256": True,
                       "test_arrays_reloaded": False})
    if not bundle["training"].get("skipped", True):
        from .learning import load_policy_v3
        data = load_local_dataset(root, bundle["candidates"][bundle["selected_candidate_index"]]["candidate"], 5)
        sim = SimulatorV3(data["track"],data["rules"],VehicleParamsV3.from_dict(bundle["selected_vehicle"]))
        for entry in bundle["training"].get("controllers", []):
            actual = sim.simulate(controller=load_policy_v3(output / "models" / entry["checkpoint"]))
            assert_trace_equal(actual, entry["result"], entry["id"])
    check_freeze(root, bundle["runtime"]["freeze"])
    result = {"schema_version": 3, "artifact_integrity_passed": True,
              "exact_local_replay_passed": True, "checks": checks,
              "accuracy_gate_passed": bool(bundle.get("benefit_claim_allowed")),
              "physical_gate_passed": bundle["gates"].get("physical", False),
              "new_session_status": cross["status"], "test_arrays_reloaded": False,
              "formal_training_run": not bundle["training"].get("skipped", True),
              "interpretation": "Exact reproduction includes retained failures; fidelity gates remain separate"}
    if save:
        write_json(output / "verification.json", result)
    return result

