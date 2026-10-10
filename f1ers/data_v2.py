"""Frozen chronological v2 holdout, preserving the complete legacy data tree.

Only the new #63 laps 10--18 telemetry window is fetched. Geometry, registration
and approximate aero endpoints are immutable calibration-only v1 artifacts.
The development loader does not open or expose any held-out lap arrays/labels.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from urllib.parse import urlencode

import numpy as np

from .data import (DataUnavailable, _clean_lap_check, _epoch, _fetch, _polyline,
                   _project, _read, _repair_signal, _speed_time, _write)


PROTOCOL_FILE = "config/protocol_austria_2026_v2.json"
V2_DIRECTORY = "data/v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root: Path, path: Path, **extra) -> dict:
    return dict(path=path.relative_to(root).as_posix(), sha256=_sha(path),
                bytes=path.stat().st_size, **extra)


def _verify_records(root: Path, records: list[dict]):
    for record in records:
        path = root / record["path"]
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise DataUnavailable(f"Frozen v2 provenance changed or is missing: {record['path']}")


def _protocol(root: Path) -> dict:
    path = root / PROTOCOL_FILE
    if not path.exists():
        raise DataUnavailable(f"The preregistered v2 protocol is missing: {PROTOCOL_FILE}")
    p = _read(path)
    if (p.get("session_key"), p.get("meeting_key"), p.get("driver_number")) != (11315, 1288, 63):
        raise DataUnavailable("v2 protocol does not identify the audited Austria Race/driver")
    groups = [p[k] for k in ("calibration_laps", "development_laps", "primary_test_laps", "replacement_test_laps_in_order")]
    flat = [lap for group in groups for lap in group]
    if len(flat) != len(set(flat)) or groups[0] != [3, 4, 5, 6] or groups[1] != [7, 9]:
        raise DataUnavailable("v2 roles overlap or alter the frozen calibration/development split")
    if p["required_test_laps"] != len(p["primary_test_laps"]):
        raise DataUnavailable("v2 must keep the precommitted number of complete test laps")
    if set(p["download_laps"]) != set(groups[2] + groups[3]):
        raise DataUnavailable("Download candidates must match the frozen test/replacement pool")
    return p


def _all_metadata(root: Path) -> dict:
    raw = root / "data/raw/openf1"
    return {key: _read(raw / f"{key}.json")
            for key in ("sessions", "drivers", "laps", "race_control", "weather", "pit", "stints")}


def _lap_map(metadata: dict, protocol: dict) -> dict:
    return {int(lap["lap_number"]): lap for lap in metadata["laps"]
            if lap.get("driver_number") == protocol["driver_number"]
            and lap.get("session_key") == protocol["session_key"]}


def _download_plan(protocol: dict, laps: dict) -> list[dict]:
    requested = [laps[number] for number in protocol["download_laps"]
                 if number in laps and laps[number].get("date_start") and laps[number].get("lap_duration")]
    if not requested:
        raise DataUnavailable("All candidate lap start/end metadata is missing; cannot freeze download window")
    padding = protocol["download_padding_s"]
    start = math.floor(min(_epoch(lap["date_start"]) for lap in requested) - padding)
    end = math.ceil(max(_epoch(lap["date_start"]) + lap["lap_duration"] for lap in requested) + padding)
    format_utc = lambda epoch: datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    query = {"session_key": protocol["session_key"], "driver_number": protocol["driver_number"],
             "date>": format_utc(start), "date<": format_utc(end)}
    return [dict(endpoint=endpoint, path=f"{V2_DIRECTORY}/raw/{endpoint}_63_laps10to18.json",
                 url=f"https://api.openf1.org/v1/{endpoint}?" + urlencode(query),
                 window_start_utc=format_utc(start), window_end_utc=format_utc(end))
            for endpoint in ("car_data", "location")]


def _ensure_lock(root: Path, protocol: dict, metadata: dict) -> dict:
    """Lock roles, quality rules, raw-source hashes and transform before fetching."""
    path = root / V2_DIRECTORY / "protocol_lock.json"
    protocol_hash = _sha(root / PROTOCOL_FILE)
    if path.exists():
        lock = _read(path)
        if lock["protocol_sha256"] != protocol_hash:
            raise DataUnavailable("v2 protocol changed after it was frozen; create a new protocol version")
        manifest_path = root / V2_DIRECTORY / "manifest.json"
        if manifest_path.exists() and _read(manifest_path)["protocol_lock_sha256"] != _sha(path):
            raise DataUnavailable("The v2 protocol lock changed after data preparation")
        _verify_records(root, lock["frozen_inputs"])
        return lock
    # An existing untracked telemetry cache would make the preregistration claim false.
    plans = _download_plan(protocol, _lap_map(metadata, protocol))
    if any((root / plan["path"]).exists() for plan in plans):
        raise DataUnavailable("New v2 telemetry exists without its pre-download protocol lock")
    legacy_manifest = _read(root / "data/manifest.json")
    _verify_records(root, legacy_manifest["sources"])
    _verify_records(root, legacy_manifest["processed"])
    legacy = _read(root / "data/processed/dataset.json")
    if legacy["audit"]["track_registration"]["fit_laps"] != protocol["calibration_laps"]:
        raise DataUnavailable("Inherited geometry was not fitted exclusively to calibration laps 3-6")
    if [lap["lap_number"] for lap in legacy["laps"] if lap["role"] == "calibration"] != [3, 4, 5, 6]:
        raise DataUnavailable("Inherited aero endpoint calibration identity is not verified")
    frozen_paths = {r["path"] for r in legacy_manifest["sources"] + legacy_manifest["processed"]}
    frozen_paths.add("data/manifest.json")
    lock = dict(schema_version=2, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                protocol_path=PROTOCOL_FILE, protocol_sha256=protocol_hash,
                frozen_before_new_telemetry_download=True, download_plan=plans,
                frozen_inputs=[_record(root, root / relative) for relative in sorted(frozen_paths)],
                calibration_geometry_fit_laps=[3, 4, 5, 6],
                prior_information=protocol["information_boundary"]["previously_inspected"],
                accuracy_gate=protocol["accuracy_gate"],
                test_scoring_status="Not scored by data preparation; downstream experiment must freeze its model before unsealing")
    _write(path, lock)
    return lock


def _metadata_quality(number: int, laps: dict, metadata: dict, protocol: dict, reference_stint: dict) -> dict:
    lap = laps.get(number)
    if lap is None or not lap.get("date_start") or not lap.get("lap_duration"):
        return dict(lap_number=number, passed=False, reasons=["missing_lap_start_or_duration"])
    reasons, stint, nearby = _clean_lap_check(lap, metadata["laps"], metadata["race_control"],
                                            metadata["weather"], metadata["pit"], metadata["stints"])
    quality = protocol["data_quality"]
    if quality["require_same_calibration_stint"] and (not stint or stint.get("stint_number") != reference_stint.get("stint_number")):
        reasons.append("different_or_unknown_calibration_stint")
    if quality["require_same_calibration_compound"] and (not stint or stint.get("compound") != reference_stint.get("compound")):
        reasons.append("different_or_unknown_calibration_compound")
    return dict(lap_number=number, passed=not reasons, reasons=reasons,
                stint_number=stint.get("stint_number") if stint else None,
                compound=stint.get("compound") if stint else None,
                rainfall_samples=len(nearby), max_rainfall=max((w["rainfall"] for w in nearby), default=None))


def _arrays(car: list, location: list) -> dict:
    def deduplicate(records):
        # Last record at an identical timestamp is retained; raw files remain intact.
        by_time = {_epoch(record["date"]): record for record in records}
        times = np.array(sorted(by_time), dtype=float)
        return times, [by_time[t] for t in times], len(records) - len(by_time)
    car_t, car, car_duplicates = deduplicate(car)
    loc_t, location, loc_duplicates = deduplicate(location)
    if len(car) < 2 or len(location) < 2:
        raise DataUnavailable("Telemetry window has too few real samples")
    speed = np.asarray([record["speed"] for record in car], float) / 3.6
    xy = np.asarray([[record["x"], record["y"]] for record in location], float)
    if not np.all(np.isfinite(speed)) or np.any(speed < 0) or not np.all(np.isfinite(xy)):
        raise DataUnavailable("Raw speed/location contains invalid samples; no synthetic replacement")
    throttle, throttle_bad = _repair_signal(car_t, np.asarray([record["throttle"] for record in car], float))
    brake, brake_bad = _repair_signal(car_t, np.asarray([record["brake"] for record in car], float), binary=True)
    return dict(car_t=car_t, speed=speed, throttle=throttle, brake=brake, loc_t=loc_t, loc_xy=xy,
                signal_repairs=dict(throttle_invalid=throttle_bad, brake_invalid=brake_bad,
                                    car_duplicate_timestamps=car_duplicates, location_duplicate_timestamps=loc_duplicates))


def _coverage(lap: dict, arrays: dict, quality: dict) -> dict:
    start = _epoch(lap["date_start"])
    end = start + lap["lap_duration"]
    report = dict(lap_number=int(lap["lap_number"]), passed=True, reasons=[])
    for channel, key in (("car", "car_t"), ("location", "loc_t")):
        times = arrays[key]
        inside = times[(times >= start) & (times <= end)]
        gap = float(np.max(np.diff(np.r_[start, inside, end])))
        bracketed = bool(times[0] <= start and times[-1] >= end)
        report[f"{channel}_samples"] = len(inside)
        report[f"max_{channel}_gap_s"] = gap
        report[f"{channel}_boundary_bracketed"] = bracketed
        if len(inside) < quality[f"minimum_{channel}_samples_per_lap"]:
            report["reasons"].append(f"insufficient_{channel}_samples")
        if gap > quality[f"maximum_{channel}_gap_s"]:
            report["reasons"].append(f"excessive_{channel}_gap")
        if quality["require_boundary_bracketing"] and not bracketed:
            report["reasons"].append(f"{channel}_lap_boundary_not_bracketed")
    report["passed"] = not report["reasons"]
    return report


def _process_lap(lap: dict, arrays: dict, track: dict, registration: dict, route: np.ndarray) -> tuple[dict, dict]:
    """Apply a frozen calibration transform; no fitting or model-error selection."""
    raw_s, length = _polyline(route)
    if not np.isclose(length, track["length_m"], rtol=0, atol=1e-7):
        raise DataUnavailable("Frozen raw route and prepared geometry lengths disagree")
    loc_t, loc_xy = arrays["loc_t"], arrays["loc_xy"]
    rot = np.asarray(registration["rotation"])
    shift = np.asarray(registration["translation_raw_units"])
    scale = registration["raw_location_units_per_m"]
    projected, error = _project((loc_xy - shift) @ rot.T / scale, route, raw_s)
    phase = registration["original_route_control_line_s_m"]
    direction = registration["direction"]
    mod = np.mod(direction * (projected - phase), length)
    tt, _, dist = _speed_time(lap, arrays["car_t"], arrays["speed"])
    lt = np.r_[tt[0], loc_t[(loc_t > tt[0]) & (loc_t < tt[-1])], tt[-1]]
    q = np.interp(lt, tt, dist) * length / dist[-1]
    p = np.interp(lt, loc_t, np.unwrap(mod * 2 * np.pi / length) * length / (2 * np.pi))
    p -= round((p[0] - q[0]) / length) * length
    repaired = int(np.sum(np.diff(p) < 0))
    p = np.maximum.accumulate(p)
    span = float(p[-1] - p[0])
    if span <= 0:
        raise DataUnavailable("Projected lap position has no forward circuit span")
    progress = (p - p[0]) * length / span
    s = track["s_m"]
    speed = np.interp(s, progress, np.interp(lt, arrays["car_t"], arrays["speed"]))
    throttle = np.interp(s, progress, np.interp(lt, arrays["car_t"], arrays["throttle"]))
    brake = (np.interp(s, progress, np.interp(lt, arrays["car_t"], arrays["brake"])) >= .5).astype(float)
    result = dict(s_m=s, speed_mps=speed, throttle=throttle, brake=brake,
                  t_s=np.interp(s, progress, lt - tt[0]), time_s=lt - tt[0], progress_m=progress,
                  raw_interpolated_speed_mps=np.interp(lt, arrays["car_t"], arrays["speed"]))
    in_lap = (loc_t >= tt[0]) & (loc_t <= tt[-1])
    diagnostic = dict(position_projected_span_m=span, projected_span_fraction=span / length,
                      position_monotonicity_repairs=repaired,
                      mean_position_distance_to_fixed_route_m=float(error[in_lap].mean()))
    return result, diagnostic


def _select_test_laps(protocol: dict, statuses: dict) -> tuple[list[int], list[dict]]:
    accepted, substitutions = [], []
    replacements = iter(protocol["replacement_test_laps_in_order"])
    for primary in protocol["primary_test_laps"]:
        if statuses[primary]["passed"]:
            accepted.append(primary)
            continue
        replacement = next((number for number in replacements if statuses[number]["passed"]), None)
        if replacement is None:
            raise DataUnavailable(f"Test lap {primary} failed quality and no unused quality-passing replacement remains")
        accepted.append(replacement)
        substitutions.append(dict(rejected_primary_lap=primary, replacement_lap=replacement,
                                  reasons=statuses[primary]["reasons"]))
    return sorted(accepted), substitutions


def prepare_data_v2(project_root: Path, refresh: bool = False) -> dict:
    """Freeze first, fetch only new bounded-window sources, then audit offline.

    No legacy file is written. Existing caches permit a fully offline rebuild.
    The data-preparation path calculates quality diagnostics only, never model
    residuals, controller gains or test-dependent calibration parameters.
    """
    root = Path(project_root).resolve()
    protocol = _protocol(root)
    metadata = _all_metadata(root)
    laps = _lap_map(metadata, protocol)
    lock = _ensure_lock(root, protocol, metadata)
    v2 = root / V2_DIRECTORY
    processed = v2 / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    previous_manifest = v2 / "manifest.json"
    if previous_manifest.exists():
        _verify_records(root, _read(previous_manifest)["sources"])
    for plan in lock["download_plan"]:
        path = root / plan["path"]
        if refresh and path.exists():
            # Preserve the frozen raw cache if upstream publishes a changed body.
            temporary = path.with_name(path.name + ".refresh.json")
            _fetch(plan["url"], temporary, True)
            if _sha(temporary) != _sha(path):
                raise DataUnavailable("Upstream v2 source changed; retained .refresh.json separately. Use a new protocol version.")
            temporary.unlink()
            temporary.with_name(temporary.name + ".download.json").unlink(missing_ok=True)
        else:
            _fetch(plan["url"], path, False)
    legacy = _read(root / "data/processed/dataset.json")
    shutil.copyfile(root / "data/processed/track.npz", processed / "track.npz")
    shutil.copyfile(root / "config/rules_austria_2026.json", processed / "rules.json")
    track = dict(legacy["track"])
    with np.load(processed / "track.npz", allow_pickle=False) as archive:
        geometry = dict(track, **{key: archive[key].copy() for key in archive.files})
    reference_stint = next((item for item in metadata["stints"]
                            if item["driver_number"] == protocol["driver_number"]
                            and item["lap_start"] <= 3 <= item["lap_end"]), None)
    if not reference_stint:
        raise DataUnavailable("The calibration tyre stint is missing")
    fixed = protocol["calibration_laps"] + protocol["development_laps"]
    candidates = protocol["primary_test_laps"] + protocol["replacement_test_laps_in_order"]
    checks = {number: _metadata_quality(number, laps, metadata, protocol, reference_stint) for number in fixed + candidates}
    for number in fixed:
        if not checks[number]["passed"]:
            raise DataUnavailable(f"Required calibration/development lap {number} failed: {checks[number]['reasons']}")
    new_arrays = _arrays(_read(root / lock["download_plan"][0]["path"]), _read(root / lock["download_plan"][1]["path"]))
    if any(item.get("session_key") != protocol["session_key"] or item.get("driver_number") != protocol["driver_number"]
           for plan in lock["download_plan"] for item in _read(root / plan["path"])):
        raise DataUnavailable("New telemetry includes an unexpected session or driver")
    route = np.loadtxt(root / "data/raw/tum/Spielberg_minimum_curvature.csv", delimiter=",")
    registration = legacy["audit"]["track_registration"]
    prepared, coverage = {}, {}
    quality = protocol["data_quality"]
    for number in candidates:
        if not checks[number]["passed"]:
            continue
        report = _coverage(laps[number], new_arrays, quality)
        coverage[number] = report
        if report["passed"]:
            try:
                values, position = _process_lap(laps[number], new_arrays, geometry, registration, route)
                report.update(position)
                if not quality["projected_position_span_fraction_min"] <= position["projected_span_fraction"] <= quality["projected_position_span_fraction_max"]:
                    report["reasons"].append("projected_position_does_not_cover_one_circuit")
                else:
                    prepared[number] = values
            except DataUnavailable as exc:
                report["reasons"].append("position_preparation_failed: " + str(exc))
        report["passed"] = not report["reasons"]
        checks[number]["reasons"].extend(report["reasons"])
        checks[number]["passed"] = not checks[number]["reasons"]
    selected, substitutions = _select_test_laps(protocol, checks)
    records = []
    for number in fixed + selected:
        lap = laps[number]
        role = "calibration" if number in protocol["calibration_laps"] else "development" if number in protocol["development_laps"] else "test"
        filename = f"lap_{number:02d}.npz"
        if role == "test":
            np.savez_compressed(processed / filename, **prepared[number])
        else:
            shutil.copyfile(root / "data/processed" / filename, processed / filename)
        records.append(dict(lap_number=number, driver_number=protocol["driver_number"], role=role,
                            lap_time_s=float(lap["lap_duration"]),
                            sector_times_s=[float(lap[f"duration_sector_{index}"]) for index in (1, 2, 3)],
                            date_start=lap["date_start"], compound=checks[number]["compound"], data_file=filename))
    # The development audit intentionally contains no test time/speed labels.
    audit = dict(legacy["audit"])
    audit.update(schema_version=2, real_laps=len(records), calibration_laps=4, development_laps=2, test_laps=3,
                 validation_laps=0, split_method="Chronological complete laps; prior v1 validation becomes development; test fixed before new telemetry download",
                 calibration_lap_numbers=protocol["calibration_laps"], development_lap_numbers=protocol["development_laps"],
                 primary_test_lap_numbers=protocol["primary_test_laps"], selected_test_lap_numbers=selected,
                 test_substitutions=substitutions, candidate_quality_checks=[checks[n] for n in fixed + candidates],
                 new_telemetry_coverage=[coverage[n] for n in candidates if n in coverage],
                 new_signal_repairs=new_arrays["signal_repairs"], geometry_frozen_from_v1=True,
                 geometry_fit_laps=[3, 4, 5, 6], aero_endpoint_fit_laps=[3, 4, 5, 6],
                 protocol_path=PROTOCOL_FILE, protocol_sha256=lock["protocol_sha256"],
                 protocol_lock_path=f"{V2_DIRECTORY}/protocol_lock.json", protocol_lock_sha256=_sha(v2 / "protocol_lock.json"),
                 test_scored_by_data_preparation=False, no_model_accuracy_claim=True,
                 test_scope=protocol["information_boundary"]["test_status"], prior_information=lock["prior_information"])
    # Remove the v1-specific validation/coverage count names from the main audit.
    audit["clean_lap_checks"] = [checks[n] for n in fixed + selected]
    audit["telemetry_coverage"] = [item for item in legacy["audit"]["telemetry_coverage"] if item["lap_number"] in fixed] + [coverage[n] for n in selected]
    document = dict(schema_version=2, session=legacy["session"], driver=legacy["driver"],
                    track=track, rules_file="rules.json", laps=records, audit=audit)
    _write(processed / "dataset.json", document)
    manifest = dict(schema_version=2, prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                    session_key=protocol["session_key"], meeting_key=protocol["meeting_key"],
                    protocol_sha256=lock["protocol_sha256"], protocol_lock_sha256=_sha(v2 / "protocol_lock.json"),
                    selected_test_laps=selected, test_substitutions=substitutions, test_metrics_computed=False,
                    inherited_sources=_read(root / "data/manifest.json")["sources"], frozen_inputs=lock["frozen_inputs"],
                    sources=[], processed=[])
    for plan in lock["download_plan"]:
        path = root / plan["path"]
        download = _read(path.with_name(path.name + ".download.json"))
        if download["url"] != plan["url"]:
            raise DataUnavailable("Raw cache download URL differs from the frozen acquisition plan")
        if _epoch(download["downloaded_at_utc"]) < _epoch(lock["frozen_at_utc"]):
            raise DataUnavailable("New telemetry was downloaded before the recorded protocol freeze")
        manifest["sources"].append(_record(root, path, url=plan["url"],
                                            downloaded_at_utc=download["downloaded_at_utc"],
                                            record_count=len(_read(path)),
                                            window_start_utc=plan["window_start_utc"], window_end_utc=plan["window_end_utc"]))
    for path in [processed / "track.npz", processed / "rules.json", processed / "dataset.json"] + [processed / record["data_file"] for record in records]:
        manifest["processed"].append(_record(root, path))
    _write(v2 / "manifest.json", manifest)
    return manifest


def load_dataset_v2(project_root: Path, include_test: bool = True) -> dict:
    """Load all roles, or load calibration/development without opening test NPZs.

    With include_test=False, no test lap duration/sector labels or speed/time/
    throttle/brake/progress arrays are returned. Quality audit and source hashes
    remain visible. This boundary prevents accidental leakage in caller code;
    direct manual reading of the raw/processed files is outside that boundary.
    """
    root = Path(project_root).resolve()
    processed = root / V2_DIRECTORY / "processed"
    if not (processed / "dataset.json").exists():
        prepare_data_v2(root)
    protocol = _protocol(root)
    lock = _read(root / V2_DIRECTORY / "protocol_lock.json")
    if _sha(root / PROTOCOL_FILE) != lock["protocol_sha256"]:
        raise DataUnavailable("v2 protocol was modified after freeze")
    manifest = _read(root / V2_DIRECTORY / "manifest.json")
    if _sha(root / V2_DIRECTORY / "protocol_lock.json") != manifest["protocol_lock_sha256"]:
        raise DataUnavailable("The v2 protocol lock changed after data preparation")
    _verify_records(root, manifest["frozen_inputs"])
    _verify_records(root, manifest["sources"] + manifest["processed"])
    document = _read(processed / "dataset.json")
    track = dict(document["track"])
    with np.load(processed / track.pop("data_file"), allow_pickle=False) as archive:
        track.update({key: archive[key].copy() for key in archive.files})
    laps = []
    for metadata in document["laps"]:
        if metadata["role"] == "test" and not include_test:
            continue
        lap = dict(metadata)
        with np.load(processed / lap.pop("data_file"), allow_pickle=False) as archive:
            lap.update({key: archive[key].copy() for key in archive.files})
        laps.append(lap)
    return dict(track=track, laps=laps, rules=_read(processed / document["rules_file"]),
                audit=dict(document["audit"], test_arrays_loaded=bool(include_test)),
                session=document["session"], driver=document["driver"], protocol=protocol)
