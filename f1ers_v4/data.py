"""Read-only inherited local channels and sealed condition-adapted FP2 inputs.

The development loader reads only separately materialized adaptation records.
Broad FP2 telemetry is used for pre-registered timestamp coverage screening;
late performance arrays are formed only after an external exclusive claim.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np

from f1ers_v3 import data as inherited
from f1ers_v3.geometry import build_track, remap_rules
from .artifacts import read_json, write_json, sha256
from .protocol import (DataUnavailable, protocol, ensure_protocol_lock, require_local_freeze,
                       mark_test_arrays_unsealed, ledger_paths, claim_coordinate_registration,
                       complete_coordinate_registration)

epoch = inherited.epoch
previous_value = inherited.previous_value
raw_arrays = inherited.raw_arrays
coverage_quality = inherited.coverage_quality
metadata_quality = inherited.metadata_quality
select_stint_roles = inherited.select_stint_roles
_LAST_REQUEST = 0.0


def load_local_dataset(root, candidate="spline", ds_m=5.0):
    """Construct cal/development inputs without writing v2/v3 data or audits."""
    root = Path(root).resolve()
    p = protocol(root)
    if candidate != "spline":
        raise DataUnavailable("Only the fixed periodic spline candidate is calibrated in v4")
    document, metadata, arrays = inherited._local_source(root)
    registration = read_json(root / "data/v3/local/registration.json")
    if registration.get("fit_laps") != [3, 4, 5, 6] or registration.get("fit_role") != "calibration":
        raise DataUnavailable("Inherited registration is not frozen calibration-only")
    track = build_track(root, "spline", ds_m, registration)
    laps = [inherited.process_lap(lap, arrays, track, registration) for lap in metadata]
    if [(int(l["lap_number"]), l["role"]) for l in laps] != [
            (3, "calibration"), (4, "calibration"), (5, "calibration"), (6, "calibration"),
            (7, "development"), (9, "development")]:
        raise DataUnavailable("Local loader returned an unexpected historical role or order")
    audit = {"schema_version": 4, "registration": registration, "signal_audit": arrays["signal_audit"],
             "alignment_per_lap": [{"lap_number": l["lap_number"], "role": l["role"],
                                    **l["alignment_audit"]} for l in laps],
             "protocol_sha256": sha256(root / "config/protocol_austria_2026_v4.json"),
             "historical_only_laps": p["historical_only_laps"], "test_arrays_loaded": False,
             "read_only_inherited_loader": True, "geometry": track["geometry_audit"],
             "data_gate_passed": all(l["alignment_audit"]["grid_bracketed"] and
                 0.96 <= l["alignment_audit"]["projected_span_fraction"] <= 1.04 for l in laps)}
    return {"track": track, "rules": remap_rules(read_json(root / "config/rules_austria_2026.json"), track),
            "laps": laps, "audit": audit, "session": document["session"], "driver": document["driver"],
            "registration": registration}


def _bounded_fetch(url, path, minimum_request_interval_s=2.2):
    global _LAST_REQUEST
    path = Path(path)
    if path.exists():
        return read_json(path)
    delay = max(0.0, minimum_request_interval_s - (time.monotonic() - _LAST_REQUEST))
    if delay:
        time.sleep(delay)
    _LAST_REQUEST = time.monotonic()
    try:
        with urlopen(Request(url, headers={"User-Agent": "f1-ers-lab-v4-public-research/1.0"}),
                     timeout=25) as response:
            payload = response.read()
        import json
        value = json.loads(payload)
        if not isinstance(value, list):
            raise DataUnavailable("Public API response is not a records array")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
        write_json(path.with_name(path.name + ".download.json"),
                   {"url": url, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": sha256(path)})
        return value
    except HTTPError as exc:
        reason = exc.read(4096).decode("utf-8", errors="replace")
        raise DataUnavailable(f"HTTP {exc.code} from {url}: {reason}") from exc
    except DataUnavailable:
        raise
    except Exception as exc:
        raise DataUnavailable(f"Required public data unavailable: {url}: {exc}") from exc


def _pending(p, lock, reason, *, attempted=False, freeze_sha=None):
    return {"schema_version": 4, "status": "pending", "reason": reason, "query_attempted": attempted,
            "queried_at_utc": datetime.now(timezone.utc).isoformat() if attempted else None,
            "protocol_sha256": lock["protocol_sha256"], "local_freeze_sha256": freeze_sha,
            "target": p["new_session_identity"], "session_substitution": False,
            "driver_substitution": False, "compound_substitution": False, "late_test_scored": False,
            "cross_session_acceptance": "新会话验收待完成"}


def prepare_new_session(root, local_freeze_path, local_gates, allow_network=True):
    """Acquire only after all local gates; quality screening never uses pace."""
    root = Path(root).resolve()
    require_local_freeze(root, local_freeze_path, local_gates)
    lock, p = ensure_protocol_lock(root), protocol(root)
    freeze_sha = sha256(local_freeze_path)
    path = root / "data/v4/fp2/status.json"
    if path.exists():
        cached = read_json(path)
        if cached.get("local_freeze_sha256") != freeze_sha:
            raise DataUnavailable("FP2 acquisition is already bound to another local freeze")
        unqueried = cached.get("status") == "pending" and not cached.get("query_attempted")
        claimed = ledger_paths(root)["claim"].exists()
        if not (allow_network and unqueried and not claimed):
            return cached
    if not allow_network:
        value = _pending(p, lock, "Offline mode: exact FP2 public sources are not cached", freeze_sha=freeze_sha)
        write_json(path, value)
        return value
    raw = root / "data/v4/fp2/raw"
    try:
        fetch = lambda endpoint, query, filename: _bounded_fetch(
            f"https://api.openf1.org/v1/{endpoint}?" + urlencode(query), raw / filename,
            p["minimum_request_interval_s"])
        sessions = fetch("sessions", {"year": 2026, "country_name": "Austria", "session_name": "Practice 2"}, "sessions.json")
        exact = [s for s in sessions if all(s.get(k) == v for k, v in
            {"year": 2026, "country_name": "Austria", "circuit_short_name": "Spielberg", "session_name": "Practice 2"}.items())]
        if len(exact) != 1:
            raise DataUnavailable("Exact 2026 Austria FP2 metadata is absent or ambiguous")
        session, metadata = exact[0], {}
        for endpoint in ("laps", "race_control", "weather", "pit", "stints"):
            query = {"session_key": session["session_key"]}
            if endpoint in ("laps", "pit", "stints"):
                query["driver_number"] = 63
            metadata[endpoint] = fetch(endpoint, query, f"{endpoint}.json")
        laps = [l for l in metadata["laps"] if l.get("driver_number") == 63]
        medium = [s for s in metadata["stints"] if s.get("driver_number") == 63 and s.get("compound") == "MEDIUM"]
        candidates = [l for l in laps if l.get("date_start") and l.get("lap_duration") and
                      any(s["lap_start"] <= l["lap_number"] <= s["lap_end"] for s in medium)]
        if not candidates:
            raise DataUnavailable("No exact #63 MEDIUM complete-lap candidates in FP2")
        fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
        query = {"session_key": session["session_key"], "driver_number": 63,
                 "date>": fmt(min(epoch(l["date_start"]) for l in candidates) - 3),
                 "date<": fmt(max(epoch(l["date_start"]) + l["lap_duration"] for l in candidates) + 3)}
        car = fetch("car_data", query, "car_data_63_medium.json")
        location = fetch("location", query, "location_63_medium.json")
        # Coverage screening consumes dates only. No late speed/driver arrays
        # or local-model residuals are formed during stint selection.
        coverage_arrays = {"car_t": np.asarray(sorted({epoch(r["date"]) for r in car})),
                           "loc_t": np.asarray(sorted({epoch(r["date"]) for r in location}))}
        if any(len(v) < 2 for v in coverage_arrays.values()):
            raise DataUnavailable("Too few original FP2 timestamps")
        statuses = {}
        for lap in laps:
            quality = metadata_quality(lap, metadata, p["data_quality"])
            if quality["passed"]:
                quality.update(coverage_quality(lap, coverage_arrays, p["data_quality"]))
            statuses[int(lap["lap_number"])] = quality
        selected = select_stint_roles(medium, laps, statuses)
        by_number = {int(l["lap_number"]): l for l in laps}
        early = [{**by_number[n], "role": "adaptation"} for n in selected["adaptation_laps"]]
        first = min(epoch(l["date_start"]) for l in early) - 3
        last = max(epoch(l["date_start"]) + l["lap_duration"] for l in early) + 3
        cutoff = min(epoch(by_number[n]["date_start"]) for n in selected["test_laps"])
        early_car = [r for r in car if first <= epoch(r["date"]) <= last and epoch(r["date"]) < cutoff]
        early_location = [r for r in location if first <= epoch(r["date"]) <= last and epoch(r["date"]) < cutoff]
        arrays = raw_arrays(early_car, early_location)
        claim_coordinate_registration(root, local_freeze_path)
        registration = inherited.fit_registration(root, early, arrays, role="adaptation", cache_path=None)
        track = build_track(root, "spline", 5.0, registration)
        processed = [inherited.process_lap(l, arrays, track, registration, strict_signal_window=True) for l in early]
        if not all(l["alignment_audit"]["grid_bracketed"] and
                   0.96 <= l["alignment_audit"]["projected_span_fraction"] <= 1.04 and
                   bool(np.all(l["scoring_signal_valid_mask"])) for l in processed):
            raise DataUnavailable("Early adaptation projection/span/boundary checks failed; no late-informed refit")
        early_dir = raw / "adaptation"
        for name, value in (("laps.json", early), ("car_data.json", early_car), ("location.json", early_location)):
            write_json(early_dir / name, value)
        write_json(root / "data/v4/fp2/registration.json", registration)
        complete_coordinate_registration(root, root / "data/v4/fp2/registration.json")
        write_json(root / "data/v4/fp2/selection.json", selected)
        value = {"schema_version": 4, "status": "ready", "query_attempted": True,
                 "queried_at_utc": datetime.now(timezone.utc).isoformat(), "session": session,
                 "driver_number": 63, "compound": "MEDIUM", "selection": selected,
                 "quality": list(statuses.values()), "late_test_scored": False,
                 "protocol_sha256": lock["protocol_sha256"], "local_freeze_sha256": freeze_sha,
                 "registration_fit_laps": selected["adaptation_laps"],
                 "quality_selection_uses_timestamp_coverage_only": True,
                 "late_projection_checked_after_exclusive_unseal": True,
                 "raw_sources": [{"path": x.relative_to(root).as_posix(), "sha256": sha256(x)}
                    for x in sorted(raw.rglob("*.json")) if not x.name.endswith(".download.json")]}
    except (DataUnavailable, inherited.DataUnavailable) as exc:
        value = _pending(p, lock, str(exc), attempted=True, freeze_sha=freeze_sha)
    write_json(path, value)
    return value


def load_new_session(root, include_test=False, candidate="spline", ds_m=5.0, registration=None):
    root = Path(root).resolve()
    protocol(root)
    if candidate != "spline":
        raise DataUnavailable("The frozen v4 geometry must remain spline")
    report = read_json(root / "data/v4/fp2/status.json")
    if report.get("status") != "ready":
        raise DataUnavailable(report.get("reason", "FP2 is not ready"))
    selected = read_json(root / "data/v4/fp2/selection.json")
    raw = root / "data/v4/fp2/raw"
    if include_test:
        # Durable marker is written before opening late performance records.
        mark_test_arrays_unsealed(root)
        names = ("laps.json", "car_data_63_medium.json", "location_63_medium.json")
        paths = [raw / name for name in names]
    else:
        paths = [raw / "adaptation" / name for name in ("laps.json", "car_data.json", "location.json")]
    source_hash = {entry["path"]: entry["sha256"] for entry in report["raw_sources"]}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if relative not in source_hash or sha256(path) != source_hash[relative]:
            raise DataUnavailable(f"Selected FP2 source changed: {relative}")
    metadata, car, location = [read_json(path) for path in paths]
    numbers = selected["test_laps"] if include_test else selected["adaptation_laps"]
    chosen = []
    for number in numbers:
        lap = next(dict(l) for l in metadata if int(l["lap_number"]) == number and l["driver_number"] == 63)
        lap.update(role="test" if include_test else "adaptation", lap_time_s=float(lap["lap_duration"]),
                   sector_times_s=[float(lap[f"duration_sector_{i}"]) for i in (1, 2, 3)], compound="MEDIUM")
        chosen.append(lap)
    if include_test:
        first = min(epoch(l["date_start"]) for l in chosen) - 3
        last = max(epoch(l["date_start"]) + l["lap_duration"] for l in chosen) + 3
        car = [r for r in car if first <= epoch(r["date"]) <= last]
        location = [r for r in location if first <= epoch(r["date"]) <= last]
    arrays = raw_arrays(car, location)
    registration = registration or read_json(root / "data/v4/fp2/registration.json")
    if registration.get("fit_role") != "adaptation" or registration.get("fit_laps") != selected["adaptation_laps"]:
        raise DataUnavailable("FP2 coordinate/time registration used unauthorized late laps")
    track = build_track(root, "spline", ds_m, registration)
    laps = [inherited.process_lap(l, arrays, track, registration, strict_signal_window=True) for l in chosen]
    data_gate = all(l["alignment_audit"]["grid_bracketed"] and
                    0.96 <= l["alignment_audit"]["projected_span_fraction"] <= 1.04 and
                    bool(np.all(l["scoring_signal_valid_mask"])) for l in laps)
    return {"track": track, "rules": remap_rules(read_json(root / "config/rules_austria_2026.json"), track),
            "laps": laps, "session": report["session"], "registration": registration,
            "audit": {"test_arrays_loaded": bool(include_test), "conditional_transfer": True,
                      "registration_fit_laps": selected["adaptation_laps"], "selection": selected,
                      "data_gate_passed": data_gate, "early_loader_materialized_records_only": not include_test,
                      "outside_metadata_signal_is_nan": True}}
