"""Role-isolated local data views; no network access and no frozen-cache writes."""
from pathlib import Path

import numpy as np

from f1ers_v3 import data as inherited
from f1ers_v3.protocol import DataUnavailable, read_json
from .geometry import baseline_spec, build_track_from_spec, remap_rules

ROLE_LAPS = {"calibration": (3, 4, 5, 6), "development": (7, 9)}


def _raw_lap(lap, arrays, track, registration):
    start = inherited.epoch(lap["date_start"])
    duration = float(lap["lap_duration"])
    car_mask = (arrays["car_t"] >= start) & (arrays["car_t"] <= start+duration)
    t = arrays["car_t"][car_mask]
    corrected_t = arrays["loc_t"]-float(registration["location_clock_offset_s"])
    location_mask = (corrected_t >= start-3) & (corrected_t <= start+duration+3)
    lt = corrected_t[location_mask]
    xy = (arrays["loc_xy"][location_mask]-np.asarray(registration["translation_raw_units"])) @ np.asarray(registration["rotation"]).T
    xy /= float(registration["raw_location_units_per_m"])
    route = track["continuous_route"]
    source, error = inherited._project(xy, route.points, route.source_s)
    mod = route.source_to_s(source)
    integral = inherited._integral(arrays, lt, start)
    progress = mod+np.round((integral-mod)/route.length_m)*route.length_m
    return {"lap_number": int(lap["lap_number"]), "role": lap["role"],
        "lap_time_s": duration, "sector_times_s": list(lap["sector_times_s"]),
        "date_start": lap["date_start"], "time_s": t-start,
        "speed_mps": arrays["speed"][car_mask].copy(),
        "throttle": arrays["throttle"][car_mask].copy(), "brake": arrays["brake"][car_mask].copy(),
        "throttle_raw_value": arrays["throttle_raw_value"][car_mask].copy(),
        "brake_raw_value": arrays["brake_raw_value"][car_mask].copy(),
        "speed_original_valid": np.ones(len(t), bool), "source_sample_original": np.ones(len(t), bool),
        "throttle_original_valid": arrays["throttle_original_valid"][car_mask].copy(),
        "brake_original_valid": arrays["brake_original_valid"][car_mask].copy(),
        "car_source": np.full(len(t), "openf1_original_car_sample"),
        "location_time_s": lt-start, "location_raw_time_s": arrays["loc_t"][location_mask]-start,
        "location_xy_m": xy, "location_source_original": np.ones(len(lt), bool),
        "location_valid_mask": np.isfinite(xy).all(axis=1) & (lt >= start) & (lt <= start+duration),
        "location_source": np.full(len(lt), "openf1_original_location_sample"),
        "progress_m": progress, "location_projection_error_m": error,
        "speed_integral_progress_m": inherited._integral(arrays, t, start),
        "location_speed_integral_progress_m": integral,
        "provenance": {"speed": "Original OpenF1 integer km/h converted to m/s",
            "car_time": "Raw timestamps relative to official lap start; no inserted boundaries",
            "location_time": "Raw timestamp minus frozen shared clock offset, relative to official lap start",
            "location": "Original samples transformed with frozen global similarity",
            "invalid_throttle": "Repaired by inherited linear interpolation, original validity retained",
            "invalid_brake": "Inherited previous-valid-value hold, original validity retained",
            "progress": "Unrepaired original projection unwrapped using unscaled speed integral",
            "speed_integral_progress_time_axis": "car time_s",
            "location_speed_integral_progress_time_axis": "location_time_s",
            "per_lap_first_sample_reset": False, "per_lap_span_stretch": False}}


def load_dataset(root, geometry_spec=None, role="calibration", ds_m=5.):
    """Return exactly one allowed role, preserving baseline v4 processed values.

    The inherited historical source contains both authorized roles.  Only the
    requested role is processed or exposed; historical diagnostic and FP2 files
    are never accessed.  The baseline adapter calls the unchanged v3 routines.
    """
    if role not in ROLE_LAPS: raise DataUnavailable("v5 loader requires calibration or development role")
    root = Path(root).resolve()
    document, metadata, arrays = inherited._local_source(root)
    registration = read_json(root/"data/v3/local/registration.json")
    if registration.get("fit_laps") != [3, 4, 5, 6] or registration.get("fit_role") != "calibration":
        raise DataUnavailable("Registration is not the frozen calibration-only transform")
    chosen = [x for x in metadata if x["role"] == role]
    if [int(x["lap_number"]) for x in chosen] != list(ROLE_LAPS[role]):
        raise DataUnavailable("Unexpected local role membership or chronological order")
    spec = geometry_spec or baseline_spec()
    track = build_track_from_spec(root, spec, ds_m, registration)
    laps = [inherited.process_lap(x, arrays, track, registration) for x in chosen]
    raw = [_raw_lap(x, arrays, track, registration) for x in chosen]
    audit = {"schema_version": 5, "role": role, "lap_numbers": list(ROLE_LAPS[role]),
        "registration": registration, "signal_audit": arrays["signal_audit"],
        "alignment_per_lap": [{"lap_number": x["lap_number"], "role": role, **x["alignment_audit"]} for x in laps],
        "test_arrays_loaded": False, "historical_only_laps": [10, 11, 12],
        "role_isolation": True, "read_only_inherited_loader": True,
        "geometry": track["geometry_audit"],
        "data_gate_passed": all(x["alignment_audit"]["grid_bracketed"] and
            .96 <= x["alignment_audit"]["projected_span_fraction"] <= 1.04 for x in laps)}
    return {"track": track, "rules": remap_rules(read_json(root/"config/rules_austria_2026.json"), track),
        "laps": laps, "raw_laps": raw, "audit": audit, "session": document["session"],
        "driver": document["driver"], "registration": registration, "role": role,
        "geometry_spec": spec,
        "boundary_reference": np.loadtxt(root/"data/raw/tum/Spielberg_centerline.csv", delimiter=",")}


load_local_dataset = load_dataset
