"""Immutable preregistration and once-only held-out data access for v3."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROTOCOL_FILE = "config/protocol_austria_2026_v3.json"


class DataUnavailable(RuntimeError):
    """Real data or a provenance boundary is unavailable; no invented replacement."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(root):
    p = read_json(Path(root) / PROTOCOL_FILE)
    if p.get("schema_version") != 3 or p["local_calibration_laps"] != [3,4,5,6] or p["local_development_laps"] != [7,9]:
        raise DataUnavailable("Invalid fixed v3 chronological local split")
    ident = p["new_session_identity"]
    if (ident["year"], ident["circuit_short_name"], ident["session_name"], ident["driver_number"], ident["compound"]) != (2026,"Spielberg","Practice 2",63,"MEDIUM"):
        raise DataUnavailable("The authorized FP2 / driver 63 / MEDIUM identity changed")
    return p


def ensure_protocol_lock(root):
    root = Path(root).resolve()
    protocol(root)
    path = root / "data/v3/protocol_lock.json"
    p_hash = sha256(root / PROTOCOL_FILE)
    if path.exists():
        lock = read_json(path)
        if lock["protocol_sha256"] != p_hash:
            raise DataUnavailable("v3 preregistration changed after freeze")
        for item in lock["frozen_inputs"]:
            source = root / item["path"]
            if not source.is_file() or sha256(source) != item["sha256"]:
                raise DataUnavailable(f"Frozen inherited data changed: {item['path']}")
        return lock
    raw_dir = root / "data/v3/fp2/raw"
    if raw_dir.exists() and any(raw_dir.glob("*.json")):
        raise DataUnavailable("FP2 data already existed before preregistration")
    inherited = ["config/rules_austria_2026.json", "data/processed/dataset.json", "data/raw/tum/Spielberg_minimum_curvature.csv", "data/raw/tum/Spielberg_centerline.csv"]
    inherited += [f"data/raw/openf1/{name}.json" for name in ("sessions","drivers","laps","race_control","weather","pit","stints","car_data_63_laps3to9","location_63_laps3to9")]
    lock = {"schema_version":3,"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":p_hash,
            "frozen_before_fp2_download":True,"frozen_inputs":[{"path":name,"sha256":sha256(root/name)} for name in inherited],
            "historical_10_12_status":"Previously scored v2 historical diagnostic only; no new arrays read by local v3 loader"}
    write_json(path, lock)
    return lock


def authorize_test_access(root, freeze_path, experiment_name):
    """Atomically bind first test access to a concrete frozen model configuration."""
    root = Path(root).resolve()
    ensure_protocol_lock(root)
    freeze = Path(freeze_path).resolve()
    try:
        relative = freeze.relative_to(root).as_posix()
    except ValueError as exc:
        raise DataUnavailable("Cross-session freeze must be inside the v3 project") from exc
    if not freeze.is_file():
        raise DataUnavailable("Freeze the adapted model before accessing late laps")
    claim = {"schema_version":3,"status":"started","experiment_name":str(experiment_name),
             "started_at_utc":datetime.now(timezone.utc).isoformat(),"configuration_path":relative,
             "configuration_sha256":sha256(freeze),"protocol_sha256":sha256(root/PROTOCOL_FILE)}
    marker = root / "data/v3/test_access.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x",encoding="utf-8") as stream:
            json.dump(claim,stream,ensure_ascii=False,indent=2,allow_nan=False)
    except FileExistsError as exc:
        raise DataUnavailable("Late test laps already claimed; replay saved results instead of rescoring") from exc
    return claim


def verify_test_authorization(root):
    root = Path(root).resolve()
    marker = root / "data/v3/test_access.json"
    if not marker.exists():
        raise DataUnavailable("Late test arrays require an exclusive frozen-model authorization")
    claim = read_json(marker)
    freeze = (root / claim.get("configuration_path", "")).resolve()
    if not freeze.is_relative_to(root) or not freeze.is_file() or sha256(freeze) != claim.get("configuration_sha256"):
        raise DataUnavailable("Cross-session configuration changed after test claim")
    if sha256(root/PROTOCOL_FILE) != claim.get("protocol_sha256") or claim.get("status") != "started":
        raise DataUnavailable("Test authorization is completed, invalid or attached to another protocol")
    return claim
