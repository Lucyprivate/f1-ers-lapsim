"""Frozen v4 split and durable, cross-clone once-only late-test authorization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .artifacts import read_json, write_json, sha256, check_freeze, jsonable

PROTOCOL_FILE = "config/protocol_austria_2026_v4.json"
PROTOCOL_IDENTITY = "austria_2026_fp2_driver63_medium_conditional_transfer"
CANONICAL_LEDGER_ROOT = Path(r"D:\Software\.codex\Codex\2026-10-05\f1-chatgpt-conversation-6abe5967-06c4-83ea\audit\test-access")


class DataUnavailable(RuntimeError):
    """A real data or frozen-protocol boundary cannot be satisfied."""


def protocol(root):
    p = read_json(Path(root) / PROTOCOL_FILE)
    if (p.get("schema_version") != 4 or p.get("protocol_identity") != PROTOCOL_IDENTITY
        or p.get("local_calibration_laps") != [3, 4, 5, 6]
        or p.get("local_development_laps") != [7, 9]
        or p.get("historical_only_laps") != [10, 11, 12]):
        raise DataUnavailable("Invalid fixed v4 chronological protocol")
    ident = p.get("new_session_identity", {})
    if tuple(ident.get(k) for k in ("year", "country_name", "circuit_short_name", "session_name",
                                  "driver_number", "compound")) != (2026, "Austria", "Spielberg", "Practice 2", 63, "MEDIUM"):
        raise DataUnavailable("The authorized FP2 / driver 63 / MEDIUM identity changed")
    ledger = Path(p.get("test_access", {}).get("ledger_root", "")).resolve()
    if ledger != CANONICAL_LEDGER_ROOT.resolve() or ledger.is_relative_to(Path(root).resolve()):
        raise DataUnavailable("Test access must use the canonical external workspace ledger")
    return p


def ledger_paths(root):
    p = protocol(root)
    directory = Path(p["test_access"]["ledger_root"]).resolve()
    identity = hashlib.sha256(PROTOCOL_IDENTITY.encode("utf-8")).hexdigest()
    return {name: directory / f"{identity}.{name}.json" for name in
            ("claim", "unsealed", "score", "coordinate", "coordinate_result",
             "adaptation", "adaptation_result")}


def _exclusive_json(path, value, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except FileExistsError as exc:
        raise DataUnavailable(message) from exc


def ensure_protocol_lock(root):
    root = Path(root).resolve()
    p = protocol(root)
    path = root / "data/v4/protocol_lock.json"
    current = sha256(root / PROTOCOL_FILE)
    if path.exists():
        lock = read_json(path)
        if lock.get("protocol_sha256") != current:
            raise DataUnavailable("v4 preregistration changed after freeze")
        for entry in lock.get("frozen_inputs", []):
            if not (root / entry["path"]).is_file() or sha256(root / entry["path"]) != entry["sha256"]:
                raise DataUnavailable(f"Frozen historical input changed: {entry['path']}")
        return lock
    if (root / "data/v4/fp2/raw").exists() and any((root / "data/v4/fp2/raw").iterdir()):
        raise DataUnavailable("FP2 sources existed before v4 preregistration")
    inherited = ["config/rules_austria_2026.json", "data/processed/dataset.json", "data/processed/track.npz",
                 "data/raw/tum/Spielberg_minimum_curvature.csv", "data/raw/tum/Spielberg_centerline.csv",
                 "data/v3/local/registration.json"]
    inherited += [f"data/raw/openf1/{name}.json" for name in
        ("sessions", "drivers", "laps", "race_control", "weather", "pit", "stints",
         "car_data_63_laps3to9", "location_63_laps3to9")]
    lock = {"schema_version": 4, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_identity": p["protocol_identity"], "protocol_sha256": current,
            "frozen_before_fp2_download": True,
            "frozen_inputs": [{"path": name, "sha256": sha256(root / name)} for name in inherited]}
    write_json(path, lock)
    return lock


def require_local_freeze(root, freeze_path, gates):
    freeze_path = Path(freeze_path).resolve()
    if not freeze_path.is_relative_to(Path(root).resolve()):
        raise DataUnavailable("Local freeze must be inside the repository")
    freeze = read_json(freeze_path)
    check_freeze(root, freeze)
    required = ("convergence", "physical", "grid", "development", "data")
    if freeze.get("stage") != "local" or not all(gates.get(k) is True for k in required):
        raise DataUnavailable("All local gates must pass before FP2 acquisition or adaptation")
    if freeze.get("local_gates") != gates:
        raise DataUnavailable("Local gate results must be part of the frozen configuration")
    return freeze


def _claim_early_pass(root, local_freeze_path, phase, experiment_name):
    root, local_freeze_path = Path(root).resolve(), Path(local_freeze_path).resolve()
    freeze = read_json(local_freeze_path)
    require_local_freeze(root, local_freeze_path, freeze.get("local_gates", {}))
    claim = {"schema_version": 4, "status": "started", "protocol_identity": PROTOCOL_IDENTITY,
             "phase": phase, "experiment_name": str(experiment_name),
             "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "local_freeze_path": local_freeze_path.relative_to(root).as_posix(),
             "local_freeze_sha256": sha256(local_freeze_path), "code_commit": freeze["code_commit"],
             "protocol_sha256": sha256(root / PROTOCOL_FILE)}
    _exclusive_json(ledger_paths(root)[phase], claim,
                    f"Early {phase} pass already claimed across clones/branches; no refit or retry")
    return claim


def claim_coordinate_registration(root, local_freeze_path):
    return _claim_early_pass(root, local_freeze_path, "coordinate", "FP2 early coordinate/time fit")


def _complete_early_pass(root, result_path, phase):
    root, result_path = Path(root).resolve(), Path(result_path).resolve()
    paths = ledger_paths(root)
    if not paths[phase].is_file() or not result_path.is_relative_to(root) or not result_path.is_file():
        raise DataUnavailable(f"Concrete authorized {phase} result is required")
    claim = read_json(paths[phase])
    freeze_path = root / claim["local_freeze_path"]
    if not freeze_path.is_file() or sha256(freeze_path) != claim["local_freeze_sha256"]:
        raise DataUnavailable(f"Local freeze changed during the early {phase} pass")
    value = {"schema_version": 4, "protocol_identity": PROTOCOL_IDENTITY, "phase": phase,
             "local_freeze_sha256": claim["local_freeze_sha256"],
             "completed_at_utc": datetime.now(timezone.utc).isoformat(),
             "result_relative_path": result_path.relative_to(root).as_posix(),
             "result_sha256": sha256(result_path), "result": read_json(result_path)}
    _exclusive_json(paths[phase + "_result"], value, f"Early {phase} result already recorded")
    return value


def complete_coordinate_registration(root, registration_path):
    return _complete_early_pass(root, registration_path, "coordinate")


def claim_condition_adaptation(root, local_freeze_path, experiment_name):
    paths = ledger_paths(root)
    if not paths["coordinate_result"].is_file():
        raise DataUnavailable("Numerical condition adaptation needs the one completed coordinate fit")
    coordinate = read_json(paths["coordinate_result"])
    result_path = (Path(root).resolve() / coordinate["result_relative_path"]).resolve()
    if (coordinate.get("local_freeze_sha256") != sha256(local_freeze_path)
        or not result_path.is_relative_to(Path(root).resolve()) or not result_path.is_file()
        or sha256(result_path) != coordinate.get("result_sha256")):
        raise DataUnavailable("Coordinate fit no longer matches the frozen local model")
    return _claim_early_pass(root, local_freeze_path, "adaptation", experiment_name)


def complete_condition_adaptation(root, result_path):
    return _complete_early_pass(root, result_path, "adaptation")


def authorize_test_access(root, freeze_path, experiment_name):
    root, freeze_path = Path(root).resolve(), Path(freeze_path).resolve()
    ensure_protocol_lock(root)
    if not freeze_path.is_relative_to(root) or not freeze_path.is_file():
        raise DataUnavailable("Freeze the adapted model inside the project before late-test access")
    freeze = read_json(freeze_path)
    check_freeze(root, freeze)
    if freeze.get("stage") != "cross_session" or freeze.get("adaptation_complete") is not True:
        raise DataUnavailable("Late-test access requires a completed condition-adaptation freeze")
    paths = ledger_paths(root)
    if not paths["adaptation_result"].is_file() or not paths["adaptation"].is_file():
        raise DataUnavailable("Late-test access requires the one durable completed numerical adaptation")
    adaptation = read_json(paths["adaptation_result"])
    adaptation_path = (root / adaptation["result_relative_path"]).resolve()
    adaptation_claim = read_json(paths["adaptation"])
    parent_freeze_path = (root / adaptation_claim["local_freeze_path"]).resolve()
    if (not adaptation_path.is_relative_to(root) or not adaptation_path.is_file()
        or sha256(adaptation_path) != adaptation.get("result_sha256")
        or freeze.get("adaptation") != adaptation.get("result")
        or not parent_freeze_path.is_relative_to(root) or not parent_freeze_path.is_file()
        or sha256(parent_freeze_path) != adaptation.get("local_freeze_sha256")
        or freeze.get("parent_local_freeze") != read_json(parent_freeze_path)):
        raise DataUnavailable("Adapted freeze no longer matches its one numerical-adaptation receipt")
    if not all(freeze.get("local_gates", {}).get(k) is True for k in
               ("convergence", "physical", "grid", "development", "data")):
        raise DataUnavailable("Late-test access requires all frozen local gates")
    claim = {"schema_version": 4, "status": "started", "protocol_identity": PROTOCOL_IDENTITY,
             "experiment_name": str(experiment_name), "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "configuration_path": freeze_path.relative_to(root).as_posix(),
             "configuration_sha256": sha256(freeze_path), "protocol_sha256": sha256(root / PROTOCOL_FILE),
             "code_commit": freeze["code_commit"], "original_repository": str(root)}
    _exclusive_json(paths["claim"], claim,
                    "Late-test identity already claimed across all clones/branches; use saved scores")
    return claim


def verify_test_authorization(root):
    root = Path(root).resolve()
    paths = ledger_paths(root)
    if not paths["claim"].exists():
        raise DataUnavailable("Late-test arrays require an exclusive external-ledger claim")
    if paths["score"].exists():
        raise DataUnavailable("Late-test scoring is already completed; read its saved score")
    claim = read_json(paths["claim"])
    freeze_path = (root / claim.get("configuration_path", "")).resolve()
    if (not freeze_path.is_relative_to(root) or not freeze_path.is_file()
        or sha256(freeze_path) != claim.get("configuration_sha256")
        or sha256(root / PROTOCOL_FILE) != claim.get("protocol_sha256")
        or claim.get("protocol_identity") != PROTOCOL_IDENTITY or claim.get("status") != "started"):
        raise DataUnavailable("Late-test authorization no longer matches the adapted freeze")
    check_freeze(root, read_json(freeze_path))
    return claim


def mark_test_arrays_unsealed(root):
    claim = verify_test_authorization(root)
    value = {"schema_version": 4, "protocol_identity": PROTOCOL_IDENTITY,
             "configuration_sha256": claim["configuration_sha256"],
             "unsealed_at_utc": datetime.now(timezone.utc).isoformat()}
    _exclusive_json(ledger_paths(root)["unsealed"], value,
                    "Late-test arrays already unsealed; an interrupted run cannot retry scoring")
    return value


def record_test_score(root, score, result_path=None):
    paths = ledger_paths(root)
    claim = verify_test_authorization(root)
    if not paths["unsealed"].exists():
        raise DataUnavailable("Cannot store a late-test score before its one authorized unseal")
    value = {"schema_version": 4, "protocol_identity": PROTOCOL_IDENTITY,
             "configuration_sha256": claim["configuration_sha256"],
             "scored_at_utc": datetime.now(timezone.utc).isoformat(), "score": score}
    if result_path is not None:
        root = Path(root).resolve()
        result_path = Path(result_path).resolve()
        trajectory_path = result_path.with_name("test_trajectories.json.gz")
        if not result_path.is_relative_to(root) or not trajectory_path.is_file():
            raise DataUnavailable("Saved score and final test trajectories must exist inside the repository")
        value["result_relative_path"] = result_path.relative_to(root).as_posix()
        value["result_sha256"] = sha256(result_path)
        value["trajectory_relative_path"] = trajectory_path.relative_to(root).as_posix()
        value["trajectory_sha256"] = sha256(trajectory_path)
    canonical = json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False, sort_keys=True,
                           separators=(",", ":"))
    value["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _exclusive_json(paths["score"], value, "Late-test score already stored; rescoring is prohibited")
    return value


def read_saved_test_score(root, result_path=None):
    paths = ledger_paths(root)
    root = Path(root).resolve()
    requested_score = Path(result_path).resolve() if result_path is not None else None
    if requested_score is not None and not requested_score.is_relative_to(root):
        raise DataUnavailable("Offline final score must be inside the repository")
    exported_receipt = (requested_score.with_name("test_access_receipt.json")
                        if requested_score is not None else None)
    if paths["score"].exists():
        value = read_json(paths["score"])
        if exported_receipt is not None and exported_receipt.is_file() and read_json(exported_receipt) != value:
            raise DataUnavailable("Exported test receipt differs from the authoritative external ledger")
    elif exported_receipt is not None and exported_receipt.is_file():
        # Portable integrity verification only. This receipt is never consumed
        # by authorize_test_access or mark_test_arrays_unsealed.
        value = read_json(exported_receipt)
    else:
        return None
    if value.get("protocol_identity") != PROTOCOL_IDENTITY:
        raise DataUnavailable("Saved test score has another protocol identity")
    canonical = json.dumps({k: v for k, v in value.items() if k != "receipt_sha256"},
                           ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != value.get("receipt_sha256"):
        raise DataUnavailable("Saved test receipt integrity failed")
    score_path = requested_score if requested_score is not None else (
        root / value.get("result_relative_path", "")).resolve()
    trajectory_path = score_path.with_name("test_trajectories.json.gz")
    if (not score_path.is_relative_to(root) or not score_path.is_file()
        or sha256(score_path) != value.get("result_sha256")
        or not trajectory_path.is_file() or sha256(trajectory_path) != value.get("trajectory_sha256")
        or read_json(score_path) != value.get("score")):
        raise DataUnavailable("Saved test score/trajectory artifact changed or missing")
    return value


def complete_test_access(root, result_path):
    """Persist a completed score independently of branches and ignored results."""
    result_path = Path(result_path).resolve()
    if not result_path.is_file():
        raise DataUnavailable("A concrete saved final score is required")
    exported_receipt = result_path.with_name("test_access_receipt.json")
    if exported_receipt.exists():
        raise DataUnavailable("An exported final-test receipt already exists; do not overwrite")
    value = record_test_score(root, read_json(result_path), result_path=result_path)
    _exclusive_json(exported_receipt, value, "An exported final-test receipt already exists")
    return value
