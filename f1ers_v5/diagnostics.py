"""Calibration-only falsification evidence, separate from official accuracy.

Intervals are deterministic sensitivity envelopes, not statistical confidence
intervals. Observed kinematics are never fed back into a prediction. Unknown
brake pressure, wheel power allocation and SOC remain unknown.
"""
from __future__ import annotations

from itertools import product
import numpy as np

from f1ers_v3.diagnostics import valid_result

CONFIG = {
    "calibration_laps": [3, 4, 5, 6], "minimum_supporting_laps": 3,
    "speed_quantization_half_width_kmh": .5,
    "derivative_windows_s": [.4, .6, .8], "minimum_derivative_samples": 3,
    "maximum_raw_car_gap_s": .35, "brake_window_s": .75,
    "minimum_brake_interval_s": 1.8, "minimum_window_samples": 3,
    "brake_utilization_drop_min": .15,
    "straight_curvature_max_1pm": .002, "full_throttle_min": .95,
    "minimum_straight_samples_per_lap": 5,
    "exit_window_m": [20., 120.], "exit_anchor_window_m": [0., 20.],
    "exit_speed_bias_lower_kmh": 5., "exit_anchor_abs_bias_max_kmh": 5.,
    "throttle_build_min_s": .2, "minimum_position_sensitivity_m": .5,
    "position_sensitivity_queries": 5,
    "mass_kg": 800., "air_density_kgpm3": 1.18, "rolling_coefficient": .012,
    "mu_grid": [1.2, 1.8, 2.4], "drag_area_grid_m2": [.6, 1.3, 2.],
    "downforce_area_grid_m2": [1.8, 3.9, 6.],
    "drag_mode_factor_grid": [.72, 1.], "downforce_mode_factor_grid": [.65, 1.],
    "zero_mguk_equivalent_ice_power_kw": 400.,
    "phase_sensitivity_half_width_m": 50., "phase_sensitivity_step_m": 5.,
    "clock_added_sensitivity_half_width_s": .3, "clock_sensitivity_step_s": .05,
    "clock_absolute_bounds_s": [-1., 1.],
    "phase_clock_near_optimal_relative_cost": .01,
    "phase_clock_near_optimal_numerical_cost_tolerance_m2": 1e-12,
    "minimum_phase_clock_location_samples_per_lap": 20,
    "similarity_template_variance_floor_m2": 1e-12,
    "phase_clock_numerical_time_tolerance_s": 1e-12,
    "minimum_phase_clock_correlation_cells": 3,
    "phase_clock_correlation_spread_floor": 1e-12,
    "uncertainty_interpretation": "Fixed sampling/quantization and original parameter-bound sensitivity envelopes; not confidence intervals",
    "window_selection": "All complete raw brake intervals and all frozen-route corners; no residual-based window selection",
}


def _finite_json(value):
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _mask(raw, key, n, default=True):
    a = np.asarray(raw.get(key, np.full(n, default)), bool)
    if a.shape != (n,):
        raise ValueError(f"Unaligned original mask: {key}")
    return a


def _raw(raw):
    arrays = {k: np.asarray(raw[k], float) for k in ("time_s", "speed_mps", "throttle", "brake")}
    t = arrays["time_s"]; n = len(t)
    if n < 3 or any(a.shape != (n,) for a in arrays.values()) or not np.all(np.diff(t) > 0):
        raise ValueError("Diagnostics require aligned strictly increasing original car samples")
    valid = np.isfinite(t) & np.isfinite(arrays["speed_mps"]) & (arrays["speed_mps"] >= 0)
    valid &= _mask(raw, "source_sample_original", n) & _mask(raw, "speed_original_valid", n)
    control_valid = valid & _mask(raw, "throttle_original_valid", n) & _mask(raw, "brake_original_valid", n)
    control_valid &= np.isfinite(arrays["throttle"]) & np.isfinite(arrays["brake"])
    control_valid &= (arrays["throttle"] >= 0) & (arrays["throttle"] <= 1)
    control_valid &= np.isclose(arrays["brake"],0) | np.isclose(arrays["brake"],1)
    arrays.update(valid=valid, control_valid=control_valid)
    return arrays


def _kinematics(t, v, valid):
    """Union local slopes over three fixed windows, including integer rounding."""
    n = len(t); low = np.full(n, np.nan); high = low.copy(); centre = low.copy()
    quant = CONFIG["speed_quantization_half_width_kmh"] / 3.6
    for i in range(n):
        slopes = []; bounds = []
        for window in CONFIG["derivative_windows_s"]:
            ix = np.flatnonzero((np.abs(t-t[i]) <= window/2 + 1e-9) & valid)
            if (len(ix) < CONFIG["minimum_derivative_samples"] or ix[0] == ix[-1]
                    or not np.all(valid[ix[0]:ix[-1]+1])
                    or np.max(np.diff(t[ix])) > CONFIG["maximum_raw_car_gap_s"]):
                continue
            dt = t[ix]-np.mean(t[ix]); den = float(dt @ dt)
            slope = float(dt @ v[ix]/den)
            intercept = float(np.mean(v[ix])); residual = v[ix]-(intercept+slope*dt)
            # A bounded perturbation of each rounded speed changes the slope
            # by sum(|dt|)/sum(dt²)*quant; local curvature adds a fixed envelope.
            margin = quant*np.sum(np.abs(dt))/den + 2*np.max(np.abs(residual))/(t[ix[-1]]-t[ix[0]])
            slopes.append(slope); bounds.extend([slope-margin, slope+margin])
        if len(slopes) >= 2:
            centre[i] = np.median(slopes); low[i] = min(bounds); high[i] = max(bounds)
    return centre, low, high


def _position(raw, arrays, dataset):
    lt = np.asarray(raw.get("location_time_s", []), float)
    p = np.asarray(raw.get("progress_m", []), float)
    if lt.shape != p.shape or len(lt) < 2 or not np.all(np.diff(lt) > 0):
        raise ValueError("Diagnostics need original registered location times/progress")
    loc_valid = _mask(raw, "location_valid_mask", len(lt)) & np.isfinite(lt) & np.isfinite(p)
    # Original location observations outside metadata bounds can bracket zero;
    # they are recorded separately rather than silently used as valid in-lap data.
    original = _mask(raw, "location_source_original", len(lt))
    available = np.isfinite(lt) & np.isfinite(p) & original
    if np.sum(available) < 2:
        raise ValueError("Insufficient original location samples")
    at = lt[available]; ap = p[available]; t = arrays["time_s"]
    s = np.interp(t, at, ap)
    bracket = (t >= at[0]) & (t <= at[-1])
    integral = np.asarray(raw.get("location_speed_integral_progress_m", []), float)
    if integral.shape == lt.shape:
        delta = p-integral; dvalid = loc_valid & np.isfinite(integral)
    else:
        integral = np.asarray(raw.get("speed_integral_progress_m", []), float)
        if integral.shape != t.shape:
            # Integrate raw speeds without resetting or stretching a position trace.
            integral = np.r_[0., np.cumsum(.5*(arrays["speed_mps"][:-1]+arrays["speed_mps"][1:])*np.diff(t))]
            integral += arrays["speed_mps"][0]*t[0]
        delta = s-integral; dvalid = arrays["valid"] & bracket
    rmse = float(np.sqrt(np.mean(delta[dvalid]**2))) if dvalid.any() else np.inf
    reg = dataset.get("registration", {})
    registered_position_u = max(CONFIG["minimum_position_sensitivity_m"], rmse, float(reg.get("rmse_m", 0.)))
    clock = .5*max(float(np.median(np.diff(t))), float(np.median(np.diff(at))))
    # Include common channel timing uncertainty in every geometry/residual
    # query, rather than accounting for it only at brake-switch windows.
    clock_position_u = float(np.max(arrays["speed_mps"][arrays["valid"]], initial=0.))*clock
    uncertainty = registered_position_u + clock_position_u
    return s, bracket, uncertainty, clock, {
        "projected_start_m": float(np.interp(0., at, ap)) if at[0] <= 0 <= at[-1] else None,
        "projection_minus_speed_integral_rmse_m": rmse,
        "projection_minus_speed_integral_signed_mean_m": float(np.mean(delta[dvalid])) if dvalid.any() else None,
        "position_sensitivity_half_width_m": uncertainty,
        "registered_position_sensitivity_half_width_m": registered_position_u,
        "clock_position_sensitivity_half_width_m": clock_position_u,
        "sampling_clock_sensitivity_half_width_s": clock,
        "frozen_global_location_clock_offset_s": reg.get("location_clock_offset_s"),
        "registration_rmse_m": reg.get("rmse_m"),
        "per_lap_origin_reset": False, "per_lap_span_stretch": False,
        "clock_interval_is_sampling_sensitivity_not_fitted_offset": True,
        "phase_clock_identifiable": False,
        "reason": "One global registration cannot separately establish metadata control-line timing, racing-line bias and clock uncertainty",
    }


def _curvature(track, s, uncertainty):
    route = track.get("continuous_route")
    length = float(track["length_m"])
    def evaluate(q):
        if route is not None:
            return np.abs(np.asarray(route.evaluate(q)["curvature_1pm"], float))
        nodes = np.asarray(track["s_m"], float); values = np.abs(np.asarray(track["curvature_1pm"], float))
        return np.interp(np.mod(q, length), np.r_[nodes, length], np.r_[values, values[0]])
    queries = np.stack([evaluate(s+x) for x in np.linspace(-uncertainty, uncertainty, CONFIG["position_sensitivity_queries"])])
    low, high = np.min(queries, axis=0), np.max(queries, axis=0)
    lower = track.get("curvature_lower_1pm"); upper = track.get("curvature_upper_1pm")
    quantified = lower is not None and upper is not None
    if quantified:
        nodes = np.asarray(track["s_m"], float)
        lo = np.asarray(lower, float); hi = np.asarray(upper, float)
        if lo.shape != nodes.shape or hi.shape != nodes.shape or np.any(lo > hi):
            raise ValueError("Unaligned geometry curvature sensitivity envelope")
        # These are signed intervals; an interval crossing zero has zero lower magnitude.
        alow = np.where((lo <= 0) & (hi >= 0), 0., np.minimum(np.abs(lo), np.abs(hi)))
        ahigh = np.maximum(np.abs(lo), np.abs(hi))
        low = np.minimum(low, np.interp(np.mod(s,length), np.r_[nodes,length], np.r_[alow,alow[0]]))
        high = np.maximum(high, np.interp(np.mod(s,length), np.r_[nodes,length], np.r_[ahigh,ahigh[0]]))
    return evaluate(s), low, high, quantified


def _forces(v, alo, ahi, klo, khi):
    m = CONFIG["mass_kg"]; rho = CONFIG["air_density_kgpm3"]; rolling = CONFIG["rolling_coefficient"]
    quant = CONFIG["speed_quantization_half_width_kmh"]/3.6
    vlo, vhi = np.maximum(0., v-quant), v+quant
    flo = m*alo+.5*rho*min(CONFIG["drag_area_grid_m2"])*min(CONFIG["drag_mode_factor_grid"])*vlo**2+rolling*m*9.81
    fhi = m*ahi+.5*rho*max(CONFIG["drag_area_grid_m2"])*vhi**2+rolling*m*9.81
    lateral_lo, lateral_hi = m*vlo**2*klo, m*vhi**2*khi
    total_lo = min(CONFIG["mu_grid"])*(m*9.81+.5*rho*min(CONFIG["downforce_area_grid_m2"])*min(CONFIG["downforce_mode_factor_grid"])*vlo**2)
    total_hi = max(CONFIG["mu_grid"])*(m*9.81+.5*rho*max(CONFIG["downforce_area_grid_m2"])*vhi**2)
    abslo = np.where((flo <= 0) & (fhi >= 0), 0., np.minimum(np.abs(flo), np.abs(fhi)))
    abshi = np.maximum(np.abs(flo), np.abs(fhi))
    return {"wheel_force_lower_n": flo, "wheel_force_upper_n": fhi,
            "lateral_force_lower_n": lateral_lo, "lateral_force_upper_n": lateral_hi,
            "total_capacity_lower_n": total_lo, "total_capacity_upper_n": total_hi,
            "joint_utilization_lower": np.hypot(abslo,lateral_lo)/total_hi,
            "joint_utilization_upper": np.hypot(abshi,lateral_hi)/total_lo,
            "pure_lateral_utilization_lower": lateral_lo/total_hi,
            "pure_lateral_utilization_upper": lateral_hi/total_lo}


def _brake_events(a, s, clock, klo, khi, alo, ahi, quantified):
    t, v, b = a["time_s"], a["speed_mps"], a["brake"]
    changes = np.flatnonzero(np.diff(b >= .5).astype(int) != 0)+1
    events = []
    for i in changes:
        if b[i] < .5:
            continue
        ends = changes[(changes > i) & (b[changes] < .5)]
        if not len(ends):
            events.append({"on_time_s":t[i], "status":"not_identifiable", "reason":"Brake release is not covered by original samples"}); continue
        j = int(ends[0]); on, off = t[i], t[j]
        early = (t >= on+clock) & (t <= on+CONFIG["brake_window_s"]-clock)
        late = (t >= off-CONFIG["brake_window_s"]+clock) & (t <= off-clock)
        finite = a["control_valid"] & np.isfinite(alo) & np.isfinite(ahi)
        early &= finite; late &= finite
        sufficient = (off-on >= CONFIG["minimum_brake_interval_s"] and
                      min(int(early.sum()),int(late.sum())) >= CONFIG["minimum_window_samples"] and
                      np.all(a["control_valid"][i:j]) and np.max(np.diff(t[i:j+1])) <= CONFIG["maximum_raw_car_gap_s"])
        row = {"on_time_s":on, "off_time_s":off, "on_position_m":s[i], "off_position_m":s[j],
               "early_samples":int(early.sum()), "late_samples":int(late.sum()),
               "geometry_sensitivity_quantified":quantified, "supports_decay":False,
               "status":"not_identifiable"}
        if not sufficient:
            row["reason"] = "Insufficient original interval/window coverage after switch-time sensitivity"; events.append(row); continue
        drops = []; feasible = 0; m = CONFIG["mass_kg"]; rho = CONFIG["air_density_kgpm3"]
        for mu, cla, cda, df, drag_factor in product(CONFIG["mu_grid"], CONFIG["downforce_area_grid_m2"], CONFIG["drag_area_grid_m2"], CONFIG["downforce_mode_factor_grid"], CONFIG["drag_mode_factor_grid"]):
            cla *= df; cda *= drag_factor
            vlo = np.maximum(0.,v-CONFIG["speed_quantization_half_width_kmh"]/3.6)
            vhi = v+CONFIG["speed_quantization_half_width_kmh"]/3.6
            total_lo = mu*(m*9.81+.5*rho*cla*vlo*vlo)
            total_hi = mu*(m*9.81+.5*rho*cla*vhi*vhi)
            caplo = np.sqrt(np.maximum(0.,total_lo**2-(m*vhi*vhi*khi)**2))
            caphi = np.sqrt(np.maximum(0.,total_hi**2-(m*vlo*vlo*klo)**2))
            active = early | late
            if np.any(caplo[active] <= 1.):
                continue
            draglo = .5*rho*cda*vlo*vlo+CONFIG["rolling_coefficient"]*m*9.81
            draghi = .5*rho*cda*vhi*vhi+CONFIG["rolling_coefficient"]*m*9.81
            demandlo = np.maximum(0.,-(m*ahi+draghi)); demandhi = np.maximum(0.,-(m*alo+draglo))
            # Parameter variants that cannot accommodate even minimum observed
            # braking demand do not establish a valid normalization reference.
            if np.any(demandlo[active] > caphi[active]):
                continue
            feasible += 1
            drops.append(float(np.median(demandlo[early]/caphi[early])-np.median(demandhi[late]/caplo[late])))
        row.update(feasible_normalization_variants=feasible,
                   worst_case_early_minus_late_utilization=min(drops) if drops else None,
                   early_acceleration_interval_mps2=[float(np.median(alo[early])),float(np.median(ahi[early]))],
                   late_acceleration_interval_mps2=[float(np.median(alo[late])),float(np.median(ahi[late]))])
        row["supports_decay"] = bool(quantified and drops and min(drops) > CONFIG["brake_utilization_drop_min"])
        row["status"] = "supported" if row["supports_decay"] else "not_identifiable"
        row["reason"] = ("Equivalent normalized brake demand declines under every physically admissible sensitivity variant" if row["supports_decay"] else
                         "No robust normalized demand decline after sampling, alignment, geometry and capability sensitivity")
        events.append(row)
    return events


def _residuals(dataset, raw, a, s, position_uncertainty, clock, comparison, quantified):
    if comparison is None:
        return {"complete_autonomous":False, "corners":[], "reason":"No complete frozen baseline supplied"}
    result = comparison.get("autonomous", {}).get("result", {})
    track = dataset["track"]; length = float(track["length_m"])
    if not valid_result(result, len(track["s_m"])):
        return {"complete_autonomous":False, "corners":[], "reason":"Partial/nonphysical autonomous output is not scored"}
    ps = np.r_[np.asarray(result["s_m"],float),length]
    pv = np.r_[np.asarray(result["speed_mps"],float),result["summary"]["speed_end_mps"]]
    corners = []
    for name, apex in track.get("corner_s_m", {}).items():
        offset = (s-float(apex)+length/2)%length-length/2
        windows = {"entry":(-250.,-80.),"apex":(-40.,20.),"exit":tuple(CONFIG["exit_window_m"]),"anchor":tuple(CONFIG["exit_anchor_window_m"])}
        groups = {}
        masks = {}
        for group, (lo,hi) in windows.items():
            mask = (offset >= lo) & (offset <= hi) & a["valid"]
            masks[group] = mask
            biases = []
            for shift in np.linspace(-position_uncertainty,position_uncertainty,CONFIG["position_sensitivity_queries"]):
                errors = (np.interp(np.mod(s[mask]+shift,length),ps,pv)-a["speed_mps"][mask])*3.6
                if len(errors):biases.append(float(np.mean(errors)))
            groups[group] = {"original_samples":int(mask.sum()),
                             "signed_bias_sensitivity_kmh":[min(biases)-.5,max(biases)+.5] if biases else None,
                             "signed_bias_kmh":float(np.mean((np.interp(s[mask],ps,pv)-a["speed_mps"][mask])*3.6)) if mask.any() else None}
        exit_mask = masks["exit"] & a["control_valid"]
        corner_controls = ((offset >= -40.) & (offset <= 120.) & a["control_valid"])
        ramp_times = a["time_s"][corner_controls & (a["brake"]<.5) & (a["throttle"]>=.2) & (a["throttle"]<CONFIG["full_throttle_min"])]
        ramp_span = float(ramp_times[-1]-ramp_times[0]) if len(ramp_times)>1 else 0.
        eb = groups["exit"]["signed_bias_sensitivity_kmh"]; ab = groups["anchor"]["signed_bias_sensitivity_kmh"]
        quality = (int(exit_mask.sum()) >= CONFIG["minimum_window_samples"] and
                   groups["anchor"]["original_samples"] >= CONFIG["minimum_window_samples"] and
                   np.all(a["control_valid"][masks["exit"] | masks["anchor"]]))
        supports = bool(quality and quantified and eb and ab and
                        eb[0] > CONFIG["exit_speed_bias_lower_kmh"] and
                        max(abs(ab[0]),abs(ab[1])) <= CONFIG["exit_anchor_abs_bias_max_kmh"] and
                        ramp_span-2*clock >= CONFIG["throttle_build_min_s"])
        corners.append({"corner":str(name),"frozen_apex_m":float(apex),"groups":groups,
                        "raw_throttle_build_interval_s":ramp_span,
                        "supports_traction_build":supports,
                        "status":"supported" if supports else "not_identifiable",
                        "reason":"Robust exit overspeed grows from a small anchor error with observed throttle establishment" if supports else
                                 "Exit residual or raw control establishment is not separable from prior-state, geometry and alignment uncertainty"})
    return {"complete_autonomous":True,"corners":corners,
            "formal_accuracy_recomputed":False,"window_metrics_are_calibration_diagnostics_only":True}


def _failure_evidence(comparison):
    if not comparison:return []
    failures = []
    for number,result in comparison.get("replay",{}).get("results",{}).items():
        if not result.get("summary",{}).get("completed_lap",False):
            failures.append({"lap_number":int(number),"scored":False,"terminal_status":result.get("terminal_status"),
                             "terminal_s_m":result.get("terminal_s_m"),"terminal_time_s":result.get("terminal_time_s"),
                             "terminal_speed_mps":result.get("terminal_speed_mps"),
                             "failure_proposal":result.get("failure_proposal"),
                             "guard_diagnostics":result.get("guard_diagnostics",[]),
                             "last_executed_wheel_force_n":list(np.asarray(result.get("wheel_force_n",[]),float)[-3:]),
                             "interpretation":"Preserved request/execution and physical termination evidence; no partial-lap accuracy"})
    return failures


def _proper_similarity_cost(template, observed):
    """One shared positive-scale, proper-rotation nuisance fit in metre units."""
    xc = template-template.mean(axis=0); yc = observed-observed.mean(axis=0)
    den = float(np.sum(xc*xc))
    if den <= CONFIG["similarity_template_variance_floor_m2"] or not np.isfinite(den):
        return np.inf, None
    u,_,vt = np.linalg.svd(xc.T @ yc)
    if np.linalg.det(u @ vt) < 0:
        u[:,-1] *= -1
    rotation = u @ vt
    scale = float(np.sum((xc @ rotation)*yc)/den)
    if not np.isfinite(scale) or scale <= 0:
        return np.inf, None
    translation = observed.mean(axis=0)-scale*template.mean(axis=0) @ rotation
    error = scale*template @ rotation+translation-observed
    return float(np.mean(np.sum(error*error,axis=1))), {
        "positive_scale":scale,"proper_rotation":rotation,"translation_m":translation,
        "rotation_determinant":float(np.linalg.det(rotation)),
    }


def _phase_clock_sensitivity(dataset):
    """Position/integral-only profile; never changes or returns new registration.

    A fixed common original-location view is used for every grid cell. Its
    shifted time must lie in that lap's original in-metadata car time window;
    neither integral extrapolation nor neighbouring-lap speed is permitted.
    """
    raws = dataset.get("raw_laps",[])
    if (len(raws)!=4 or {int(r.get("lap_number",-1)) for r in raws}!={3,4,5,6}
            or any(r.get("role")!="calibration" for r in raws)):
        raise ValueError("Global phase/clock sensitivity is calibration-only")
    route = dataset.get("track",{}).get("continuous_route")
    common = {"diagnostic_only":True,"uses_predicted_speed":False,
              "uses_observed_position_and_unscaled_speed_integral_only":True,
              "registration_changed":False,"route_shape_changed":False,
              "formal_model_re_registration_permitted":False,
              "interval_interpretation":"1% deterministic grid cost sensitivity; not a statistical confidence interval",
              "nuisance":"One global proper rotation, positive scale and translation shared by all four laps"}
    if route is None:
        return {**common,"status":"not_identifiable","reason":"Continuous frozen baseline route is unavailable"}
    if dataset.get("geometry_spec",{}).get("kind", "baseline") != "baseline":
        return {**common,"status":"not_identifiable","reason":"Phase/clock profile requires the unchanged frozen baseline route, not a selected shape candidate"}
    base_clock = float(dataset.get("registration",{}).get("location_clock_offset_s",0.))
    half = CONFIG["clock_added_sensitivity_half_width_s"]; step = CONFIG["clock_sensitivity_step_s"]
    clocks = np.round(np.arange(-half,half+step*.1,step),12)
    lo,hi = CONFIG["clock_absolute_bounds_s"]
    time_tol = CONFIG["phase_clock_numerical_time_tolerance_s"]
    clocks = clocks[(clocks+base_clock >= lo-time_tol)&(clocks+base_clock <= hi+time_tol)]
    phalf = CONFIG["phase_sensitivity_half_width_m"]; pstep = CONFIG["phase_sensitivity_step_m"]
    phases = np.arange(-phalf,phalf+pstep*.1,pstep)
    if not len(clocks):
        return {**common,"status":"not_identifiable","reason":"Frozen clock is outside the original absolute clock bounds"}
    prepared = []; coverage = []
    for raw in sorted(raws,key=lambda r:int(r["lap_number"])):
        lt = np.asarray(raw.get("location_time_s",[]),float)
        xy = np.asarray(raw.get("location_xy_m",[]),float)
        car_t = np.asarray(raw.get("time_s",[]),float)
        car_q = np.asarray(raw.get("speed_integral_progress_m",[]),float)
        loc_q = np.asarray(raw.get("location_speed_integral_progress_m",[]),float)
        if (len(lt)<2 or xy.shape!=(len(lt),2) or len(car_t)<2 or
                not np.all(np.diff(lt)>0) or not np.all(np.diff(car_t)>0)):
            return {**common,"status":"not_identifiable","reason":"Missing original location/integral time coverage"}
        # Prefer original car-axis integrals: their support remains strictly
        # inside this calibration lap even when locations have ±3 s padding.
        if car_q.shape==car_t.shape and np.isfinite(car_q).all():
            it,iq = car_t,car_q
        elif loc_q.shape==lt.shape and np.isfinite(loc_q).all():
            it,iq = lt,loc_q
        else:
            return {**common,"status":"not_identifiable","reason":"No aligned unscaled observed speed-integral view"}
        lower = max(float(it[0]),float(car_t[0])); upper = min(float(it[-1]),float(car_t[-1]))
        keep = _mask(raw,"location_valid_mask",len(lt)) & _mask(raw,"location_source_original",len(lt))
        keep &= np.isfinite(lt) & np.isfinite(xy).all(axis=1)
        keep &= (lt-float(np.max(clocks)) >= lower) & (lt-float(np.min(clocks)) <= upper)
        if int(keep.sum()) < CONFIG["minimum_phase_clock_location_samples_per_lap"]:
            return {**common,"status":"not_identifiable","reason":"Insufficient common original location coverage for every clock perturbation",
                    "failed_lap_number":int(raw["lap_number"]),"common_location_samples":int(keep.sum()),
                    "extrapolation_used":False}
        shifted = lt[keep][None,:]-clocks[:,None]
        if np.any(shifted < lower-time_tol) or np.any(shifted > upper+time_tol):
            raise ValueError("Phase/clock window cannot extrapolate observed integral")
        prepared.append((np.stack([np.interp(query,it,iq) for query in shifted]),xy[keep]))
        coverage.append({"lap_number":int(raw["lap_number"]),"common_location_samples":int(keep.sum()),
                         "integral_time_bounds_s":[lower,upper],"sample_time_bounds_s":[float(lt[keep][0]),float(lt[keep][-1])]})
    observed = np.concatenate([item[1] for item in prepared])
    costs = np.full((len(phases),len(clocks)),np.inf); best_nuisance = None
    for i,phase in enumerate(phases):
        for j,clock in enumerate(clocks):
            q = np.concatenate([item[0][j] for item in prepared])+phase
            values = route.evaluate(q)
            template = np.column_stack([values["x_m"],values["y_m"]])
            cost,nuisance = _proper_similarity_cost(template,observed)
            costs[i,j] = cost
            if nuisance is not None and cost <= np.min(costs):best_nuisance=nuisance
    if not np.isfinite(costs).any():
        return {**common,"status":"not_identifiable","reason":"Global proper similarity is singular","coverage":coverage}
    i,j = np.unravel_index(np.argmin(costs),costs.shape); minimum = float(costs[i,j])
    threshold = minimum*(1+CONFIG["phase_clock_near_optimal_relative_cost"])+CONFIG["phase_clock_near_optimal_numerical_cost_tolerance_m2"]
    ids = np.argwhere(costs <= threshold)
    near_phase,near_clock = phases[ids[:,0]],clocks[ids[:,1]]
    spread_floor = CONFIG["phase_clock_correlation_spread_floor"]
    correlation = (float(np.corrcoef(near_phase,near_clock)[0,1]) if len(ids)>=CONFIG["minimum_phase_clock_correlation_cells"] and np.std(near_phase)>spread_floor and np.std(near_clock)>spread_floor else None)
    return {**common,"status":"profile_available","phase_shift_grid_m":phases,
            "added_clock_grid_s":clocks,"absolute_clock_grid_s":base_clock+clocks,
            "frozen_clock_offset_s":base_clock,"cost_grid_m2":costs,
            "cost_definition":"Mean squared 2D position residual in registered observed metre units, after one shared proper similarity",
            "minimum_cost_m2":minimum,"minimum_position_rmse_m":np.sqrt(minimum),
            "minimum_phase_shift_m":float(phases[i]),"minimum_added_clock_s":float(clocks[j]),
            "minimum_absolute_clock_s":float(base_clock+clocks[j]),
            "near_optimal_cost_threshold_m2":threshold,"near_optimal_grid_points":len(ids),
            "near_optimal_phase_shift_range_m":[float(np.min(near_phase)),float(np.max(near_phase))],
            "near_optimal_added_clock_range_s":[float(np.min(near_clock)),float(np.max(near_clock))],
            "near_optimal_absolute_clock_range_s":[float(base_clock+np.min(near_clock)),float(base_clock+np.max(near_clock))],
            "near_optimal_phase_clock_correlation":correlation,
            "correlation_unavailable_reason":None if correlation is not None else "Too few near-optimal cells or zero phase/clock range; separate identifiability is not established",
            "best_global_similarity_nuisance":best_nuisance,"coverage":coverage,"extrapolation_used":False,
            "conclusion":"Position-only phase/clock sensitivity; do not re-align formal trajectories or infer a model improvement from this minimum"}


def diagnose(dataset, baseline_comparison=None):
    """Return JSON-safe evidence and conservative gates from race laps 3–6."""
    raws = dataset.get("raw_laps",[])
    if (len(raws)!=4 or {int(r.get("lap_number",-1)) for r in raws}!={3,4,5,6}
            or any(r.get("role")!="calibration" for r in raws)
            or any(l.get("role")!="calibration" for l in dataset.get("laps",[]))):
        raise ValueError("Falsification diagnostics only accept the four fixed calibration laps 3–6")
    if baseline_comparison is not None and baseline_comparison.get("role","calibration")!="calibration":
        raise ValueError("Development/test baseline residuals are prohibited for candidate evidence")
    if baseline_comparison is not None:
        for mode in ("autonomous", "replay"):
            metrics = baseline_comparison.get(mode,{}).get("metrics",[])
            if any(m.get("role","calibration")!="calibration" or int(m.get("lap_number",-1)) not in (3,4,5,6) for m in metrics):
                raise ValueError("Baseline metric identities must remain calibration-only")
        if any(int(number) not in (3,4,5,6) for number in baseline_comparison.get("replay",{}).get("results",{})):
            raise ValueError("Held-out replay failure arrays are prohibited for candidate evidence")
    rows = []; brake_support = []; traction_support = []
    for raw in sorted(raws,key=lambda r:int(r["lap_number"])):
        a = _raw(raw); t,v = a["time_s"],a["speed_mps"]
        s,bracket,pos_u,clock,alignment = _position(raw,a,dataset)
        a["valid"] &= bracket; a["control_valid"] &= bracket
        ac,alo,ahi = _kinematics(t,v,a["valid"])
        kc,klo,khi,quantified = _curvature(dataset["track"],s,pos_u)
        forces = _forces(v,alo,ahi,klo,khi)
        kvalid = a["valid"] & np.isfinite(alo) & np.isfinite(ahi)
        brake = _brake_events(a,s,clock,klo,khi,alo,ahi,quantified)
        residual = _residuals(dataset,raw,a,s,pos_u,clock,baseline_comparison,quantified)
        n = int(raw["lap_number"])
        if any(e.get("supports_decay",False) for e in brake):brake_support.append(n)
        if any(c.get("supports_traction_build",False) for c in residual["corners"]):traction_support.append(n)
        straight = kvalid & a["control_valid"] & (a["throttle"]>=CONFIG["full_throttle_min"]) & (a["brake"]<.5) & (khi<CONFIG["straight_curvature_max_1pm"])
        powerlo = forces["wheel_force_lower_n"]*np.maximum(0.,v-.5/3.6)/1000.
        powerhi = forces["wheel_force_upper_n"]*(v+.5/3.6)/1000.
        enough = int(straight.sum()) >= CONFIG["minimum_straight_samples_per_lap"]
        rows.append({"lap_number":n,"role":"calibration","original_car_samples":len(t),
                     "kinematic_samples":int(kvalid.sum()),"control_alignment":alignment,
                     "geometry_sensitivity_quantified":quantified,
                     "geometry_sensitivity_source":"Position-shift plus supplied curvature sensitivity" if quantified else "Position-shift only; smoothing uncertainty unquantified",
                     "kinematics":{"time_s":t,"progress_m":s,"valid":kvalid,
                                   "acceleration_mps2":ac,"acceleration_lower_mps2":alo,"acceleration_upper_mps2":ahi,
                                   "curvature_abs_1pm":kc,"curvature_abs_lower_1pm":klo,"curvature_abs_upper_1pm":khi,**forces},
                     "capability_certificate":{"certain_joint_exceedances":int(np.sum(kvalid & (forces["joint_utilization_lower"]>1.))),
                                               "possible_joint_exceedances":int(np.sum(kvalid & (forces["joint_utilization_upper"]>1.))),
                                               "certain_pure_lateral_exceedances":int(np.sum(a["valid"] & (forces["pure_lateral_utilization_lower"]>1.))),
                                               "bounds_expanded":False,"observed_state_feedback":False},
                     "brake_events":brake,"autonomous_residual_structure":residual,
                     "full_throttle_straight_power":{"status":"interval_available" if enough else "not_identifiable",
                                                    "original_samples":int(straight.sum()),
                                                    "median_equivalent_wheel_power_interval_kw":[float(np.median(powerlo[straight])),float(np.median(powerhi[straight]))] if enough else None,
                                                    "ice_ers_separation_identifiable":False,
                                                    "interpretation":"Inferred ma+drag times speed, conditional on mass/drag bounds; not measured engine power or SOC"}})
    gates = {
        "brake_decay":{"eligible":len(brake_support)>=CONFIG["minimum_supporting_laps"],"supporting_laps":brake_support},
        "traction_build":{"eligible":len(traction_support)>=CONFIG["minimum_supporting_laps"],"supporting_laps":traction_support},
    }
    for gate in gates.values():
        gate.update(required_supporting_laps=CONFIG["minimum_supporting_laps"],
                    status="supported" if gate["eligible"] else "not_identifiable")
        gate["reasons"] = [] if gate["eligible"] else ["Fewer than three original calibration laps support the hypothesis after fixed nuisance sensitivity"]
        if not all(r["geometry_sensitivity_quantified"] for r in rows):
            gate["reasons"].append("Smoothing/reconstruction curvature uncertainty is not quantified; position shifts alone cannot exclude a geometry explanation")
    if not all(r["autonomous_residual_structure"]["complete_autonomous"] for r in rows):
        gates["traction_build"]["reasons"].append("Complete physically valid frozen autonomous calibration output is required")
    phase_clock = _phase_clock_sensitivity(dataset)
    assumed_power = CONFIG["zero_mguk_equivalent_ice_power_kw"]
    power_evidence = []
    for row in rows:
        interval = row["full_throttle_straight_power"]["median_equivalent_wheel_power_interval_kw"]
        relation = "not_identifiable"
        if interval is not None:
            relation = "contains_assumed_power" if interval[0] <= assumed_power <= interval[1] else ("above_assumed_power" if interval[0] > assumed_power else "below_assumed_power")
        power_evidence.append({"lap_number":row["lap_number"],"interval_kw":interval,"relation":relation})
    above = [x["lap_number"] for x in power_evidence if x["relation"]=="above_assumed_power"]
    below = [x["lap_number"] for x in power_evidence if x["relation"]=="below_assumed_power"]
    power_mismatch = max(len(above),len(below)) >= CONFIG["minimum_supporting_laps"]
    ledger = [
        {"hypothesis":"binary_brake_mapping_missing_global_decay","status":gates["brake_decay"]["status"],
         "evidence":"Normalized early/late demand intervals, same nuisance parameters, at least three calibration laps",
         "expected_observation":"Brake-on normalized equivalent demand is robustly stronger early than late in at least three calibration laps",
         "falsification_condition":"The early-minus-late sensitivity lower bound does not exceed 0.15, or fewer than three laps have adequate original control/geometry coverage",
         "conclusion":"Eligible for a limited global decay comparison" if gates["brake_decay"]["eligible"] else "Missing-decay explanation is not identifiable; raw deceleration changes alone cannot justify it",
         "alternative_causes":"Changing speed/downforce, lateral budget, original switch sampling, geometry and clock uncertainty"},
        {"hypothesis":"autonomous_traction_establishes_too_fast","status":gates["traction_build"]["status"],
         "evidence":"Complete autonomous calibration-only exit residuals and raw throttle establishment",
         "expected_observation":"Exit signed-speed sensitivity stays above 5 km/h while anchor error remains within 5 km/h and raw throttle establishment remains longer than 0.2 seconds after timing sensitivity",
         "falsification_condition":"Exit growth is absent, already present at the anchor, or inseparable from original input/position/geometry uncertainty in at least three laps",
         "conclusion":"Eligible for a limited autonomous traction-build comparison" if gates["traction_build"]["eligible"] else "Rapid-traction explanation is not identifiable; no new control parameter is justified",
         "alternative_causes":"Approach-state error, route curvature, observed throttle already established, power/drag allocation"},
        {"hypothesis":"frozen_reference_line_or_control_line_mismatch","status":"not_identifiable",
         "evidence":"Original location versus speed-integral and frozen registration sensitivity, without residual-driven re-registration",
         "expected_observation":"Position-only phase/clock profile localizes a stable shared mismatch across fixed calibration laps",
         "falsification_condition":"Frozen phase/clock remains near-optimal or a broad/correlated sensitivity valley cannot separate metadata clock, control line and shape",
         "conclusion":"Position-only profile is diagnostic sensitivity; its minimum cannot authorize model re-registration or establish the cause of a speed residual",
         "alternative_causes":"Location noise, actual racing line, metadata boundary, correlated global clock and phase"},
        {"hypothesis":"mu_downforce_bounds_cannot_cover_observed_demand","status":"supported" if all(r["geometry_sensitivity_quantified"] for r in rows) and any(r["capability_certificate"]["certain_joint_exceedances"] for r in rows) else "not_identifiable",
         "evidence":"Observed kinematics under unexpanded bounds; noisy intervals cannot measure actual tyre friction",
         "expected_observation":"Observed joint-demand lower bound exceeds upper physical capacity after quantified geometry and kinematic sensitivity",
         "falsification_condition":"Upper capability accommodates observed-demand intervals or geometry uncertainty remains unquantified",
         "conclusion":"Any exceedance is conditional on the specified mass, aero and route sensitivity; absence of an exceedance does not identify true friction/downforce",
         "alternative_causes":"Curvature, mass, aero mode, time/position correspondence, differentiation noise"},
        {"hypothesis":"zero_MGUK_power_assumption","status":"conditional_model_power_mismatch" if power_mismatch else "not_identifiable",
         "assumed_equivalent_ice_power_kw":assumed_power,"evidence_by_lap":power_evidence,
         "expected_observation":"Zero requested MGU-K plus 400 kW equivalent ICE is consistent with full-throttle straight inferred wheel-power sensitivity intervals",
         "falsification_condition":"At least three calibration-lap intervals exclude 400 kW in the same direction after original-sample, drag, position and derivative sensitivity",
         "conclusion":"The assumed total equivalent power is conditionally inconsistent; this does not establish real MGU-K deployment or SOC" if power_mismatch else "400 kW is not falsified by broad intervals that contain it; real ICE/ERS allocation is not identifiable",
         "alternative_causes":"Unknown ICE wheel power, ERS deployment, drivetrain efficiency, drag, fuel mass, speed quantization and sampling"},
    ]
    return _finite_json({"schema_version":5,"diagnostic_only":True,"config":CONFIG,"calibration_laps":[3,4,5,6],
                         "development_used_for_model_evidence":False,"future_test_arrays_read":False,
                         "brake_decay_eligible":gates["brake_decay"]["eligible"],
                         "traction_eligible":gates["traction_build"]["eligible"],
                         "candidate_evidence_gates":gates,"hypotheses_ledger":ledger,"by_lap":rows,
                         "global_phase_clock_sensitivity":phase_clock,
                         "partial_replay_failure_evidence":_failure_evidence(baseline_comparison),
                         "unknowns":["real brake pressure","SOC","ICE/ERS power split","precise racing line","true fuel mass"],
                         "acceptance_thresholds_changed":False,"partial_lap_accuracy_computed":False})
