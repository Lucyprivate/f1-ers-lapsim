"""Original channel processing and quality checks without legacy data pipelines."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .input_io import DataUnavailable, read_json, write_json
from .geometry import polyline_length


def epoch(value):
    return datetime.fromisoformat(value.replace("Z","+00:00")).timestamp()


def previous_value(t,values,query):
    """A discrete channel never takes a future value across a transition."""
    t=np.asarray(t,float);values=np.asarray(values)
    return values[np.clip(np.searchsorted(t,np.asarray(query,float),side="right")-1,0,len(t)-1)]


def raw_arrays(car,location):
    def unique(records):
        ordered={epoch(r["date"]):r for r in records}
        keys=np.array(sorted(ordered),float)
        return keys,[ordered[t] for t in keys],len(records)-len(keys)
    ct,car,cd=unique(car);lt,location,ld=unique(location)
    if len(ct)<2 or len(lt)<2:
        raise DataUnavailable("Too few original telemetry samples")
    speed=np.array([r["speed"] for r in car],float)/3.6
    xy=np.array([[r["x"],r["y"]] for r in location],float)
    if not np.isfinite(speed).all() or (speed<0).any() or not np.isfinite(xy).all():
        raise DataUnavailable("Invalid raw speed/location values")
    channels={}
    flags={}
    for name in ("throttle","brake"):
        raw=np.asarray([r.get(name,np.nan) for r in car],float)
        good=np.isfinite(raw)&(raw>=0)&(raw<=100)
        if good.sum()<2:
            raise DataUnavailable(f"Too few valid original {name} samples")
        if name=="throttle":values=np.interp(ct,ct[good],raw[good])/100.
        else:values=previous_value(ct[good],(raw[good]>=50).astype(float),ct)
        channels[name]=values
        flags[f"{name}_original_valid"]=good
        flags[f"{name}_raw_value"]=raw
    return {"car_t":ct,"speed":speed,"loc_t":lt,"loc_xy":xy,**channels,**flags,
            "signal_audit":{"car_duplicate_timestamps":cd,"location_duplicate_timestamps":ld,
               "throttle_invalid_samples":int((~flags['throttle_original_valid']).sum()),
               "brake_invalid_samples":int((~flags['brake_original_valid']).sum()),
               "throttle_interpolation":"linear only","brake_interpolation":"zero-order hold from previous valid sample"}}


def _integral(arrays,query,reference):
    t,v=arrays["car_t"],arrays["speed"]
    distance=np.r_[0.,np.cumsum(.5*(v[:-1]+v[1:])*np.diff(t))]
    # Between samples speed is linear, so its integral is quadratic.
    def evaluate(q):
        q=np.asarray(q,float)
        ids=np.clip(np.searchsorted(t,q,side="right")-1,0,len(t)-2)
        dt=q-t[ids]
        return distance[ids]+v[ids]*dt+.5*(v[ids+1]-v[ids])/(t[ids+1]-t[ids])*dt**2
    return evaluate(query)-evaluate(reference)


def _similarity(template,observed):
    xc,yc=template-template.mean(0),observed-observed.mean(0)
    u,_,vt=np.linalg.svd(xc.T@yc)
    rotation=u@vt
    if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@vt
    scale=float(np.sum((xc@rotation)*yc)/np.sum(xc*xc))
    shift=observed.mean(0)-scale*template.mean(0)@rotation
    residual=(scale*template@rotation+shift-observed)/scale
    return residual,scale,rotation,shift


def fit_registration(root,laps,arrays,*,role="calibration",cache_path=None):
    if role not in ("calibration","adaptation") or any(l.get("role",role)!=role for l in laps):
        raise DataUnavailable("Coordinate/clock fitting may use calibration or early adaptation only")
    expected=[3,4,5,6] if role=="calibration" else None
    if expected is not None and [int(l["lap_number"]) for l in laps]!=expected:
        raise DataUnavailable("Local registration requires fixed calibration laps 3-6")
    root=Path(root)
    oldreg=read_json(root/"data/processed/dataset.json")["audit"]["track_registration"]
    route=np.loadtxt(root/"data/raw/tum/Spielberg_minimum_curvature.csv",delimiter=",")
    source_s=polyline_length(route);length=float(source_s[-1]);direction=oldreg["direction"]
    samples=[]
    for lap in laps:
        start=epoch(lap["date_start"]);end=start+lap["lap_duration"]
        inside=(arrays["loc_t"]>=start+1)&(arrays["loc_t"]<=end-1)
        samples.append((start,arrays["loc_t"][inside],arrays["loc_xy"][inside]))
    if any(len(item[1])<100 for item in samples):
        raise DataUnavailable("Insufficient calibration-only coordinate samples")
    observed=np.concatenate([x[2] for x in samples])
    def evaluate(theta):
        phase,offset=theta
        q=np.concatenate([_integral(arrays,tt-offset,start) for start,tt,_ in samples])
        u=np.mod(phase+direction*q,length)
        template=np.column_stack([np.interp(u,source_s,np.r_[route[:,j],route[0,j]]) for j in (0,1)])
        return _similarity(template,observed)
    phase0=oldreg["original_route_control_line_s_m"]
    # Three-point derivatives use a time perturbation larger than IEEE spacing
    # of epoch timestamps; default two-point differences can silently be zero.
    fit=least_squares(lambda x:evaluate(x)[0].ravel(),[phase0,0.],jac="3-point",bounds=([phase0-100,-1],[phase0+100,1]),
                      max_nfev=120,xtol=1e-7,ftol=1e-7,gtol=1e-7)
    residual,scale,rotation,shift=evaluate(fit.x)
    rmse=float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
    if not fit.success or rmse>25 or not 8<scale<12:
        raise DataUnavailable(f"Calibration-only registration failed: RMSE={rmse:.3f}m scale={scale:.3f}")
    result={"original_route_control_line_s_m":float(fit.x[0]%length),"location_clock_offset_s":float(fit.x[1]),
            "direction":direction,"raw_location_units_per_m":scale,"rotation":rotation.tolist(),
            "translation_raw_units":shift.tolist(),"rmse_m":rmse,"fit_laps":[int(l["lap_number"]) for l in laps],
            "fit_role":role,"converged":bool(fit.success),"evaluations":int(fit.nfev),
            "method":"Proper global 2D similarity + common control-line phase + common clock offset fitted to unscaled calibration speed integral",
            "per_lap_first_sample_reset":False,"per_lap_span_stretch":False,"speed_integral_scale":1.,
            "clock_offset_definition":"corrected_location_time = raw_location_time - location_clock_offset_s"}
    if cache_path is not None:write_json(cache_path,result)
    return result


def _project(points,route,source_s):
    delta=np.roll(route,-1,axis=0)-route
    norms=np.sum(delta**2,axis=1)
    result,error=[],[]
    for point in points:
        frac=np.clip(np.sum((point-route)*delta,axis=1)/norms,0,1)
        d2=np.sum((route+frac[:,None]*delta-point)**2,axis=1)
        i=int(np.argmin(d2))
        result.append(source_s[i]+frac[i]*np.sqrt(norms[i]));error.append(np.sqrt(d2[i]))
    return np.asarray(result),np.asarray(error)


def process_lap(lap,arrays,track,registration,strict_signal_window=False):
    start=epoch(lap["date_start"]);end=start+lap["lap_duration"]
    route=track["continuous_route"]
    corrected_t=arrays["loc_t"]-registration["location_clock_offset_s"]
    inside=(corrected_t>=start-3)&(corrected_t<=end+3)
    lt=corrected_t[inside];rawxy=arrays["loc_xy"][inside]
    rotation=np.asarray(registration["rotation"]);shift=np.asarray(registration["translation_raw_units"])
    xy=(rawxy-shift)@rotation.T/registration["raw_location_units_per_m"]
    source,error=_project(xy,route.points,route.source_s)
    mod=route.source_to_s(source)
    expected=_integral(arrays,lt,start)
    progress=mod+np.round((expected-mod)/route.length_m)*route.length_m
    reversals=int(np.sum(np.diff(progress)<0))
    # Retain the absolute calibrated control-line origin; only small public
    # location jitter is monotonized. No subtract-first or affine span change.
    progress=np.maximum.accumulate(progress)
    if len(progress)<2 or progress[-1]<=progress[0]:
        raise DataUnavailable("Observed position cannot be unwrapped on the frozen route")
    node_t=np.interp(track["s_m"],progress,lt)
    ct=arrays["car_t"]
    source_valid=(node_t>=max(start,ct[0])-1e-7)&(node_t<=min(end,ct[-1])+1e-7)
    raw_inside=(ct>start)&(ct<end)
    tt=np.r_[start,ct[raw_inside],end]
    original=np.r_[False,np.ones(int(raw_inside.sum()),bool),False]
    driver={"time_s":tt-start,"speed_mps":np.interp(tt,ct,arrays["speed"]),
            "throttle":np.interp(tt,ct,arrays["throttle"]),"brake":previous_value(ct,arrays["brake"],tt),
            "throttle_raw_value":np.r_[np.nan,arrays["throttle_raw_value"][raw_inside],np.nan],
            "brake_raw_value":np.r_[np.nan,arrays["brake_raw_value"][raw_inside],np.nan],
            "source_sample_original":original,"throttle_original_valid":previous_value(ct,arrays["throttle_original_valid"],tt)&original,
            "brake_original_valid":previous_value(ct,arrays["brake_original_valid"],tt)&original,
            "raw_epoch_s":tt,"provenance":{"speed_unit":"Original integer km/h converted to m/s","throttle":"Linear time interpolation",
             "brake":"Previous valid value hold, no interpolated switch","boundary_samples":"Interpolated boundary samples explicitly marked non-original",
             "end_boundary_extrapolated":bool(ct[-1]<end),
             "no_observed_ers_labels":True}}
    at_boundary=np.interp([start,end],lt,progress)
    observed_speed=np.interp(node_t,ct,arrays["speed"])
    if strict_signal_window:observed_speed=np.where(source_valid,observed_speed,np.nan)
    result={**lap,"s_m":track["s_m"].copy(),"speed_mps":observed_speed,
            "throttle":np.interp(node_t,ct,arrays["throttle"]),"brake":previous_value(ct,arrays["brake"],node_t),
            "t_s":node_t-start,"driver_trace":driver,"time_s":lt-start,"progress_m":progress,
            "speed_integral_m":float(_integral(arrays,end,start)),"initial_speed_mps":float(driver["speed_mps"][0]),
            "sector_end_s_m":track["sector_end_s_m"],"ers_validated":False,
            "scoring_signal_valid_mask":source_valid,
            "alignment_audit":{"absolute_projected_start_m":float(at_boundary[0]),"absolute_projected_end_m":float(at_boundary[1]),
              "projected_span_m":float(at_boundary[1]-at_boundary[0]),"projected_span_fraction":float((at_boundary[1]-at_boundary[0])/route.length_m),
              "speed_integral_m":float(_integral(arrays,end,start)),"projection_vs_speed_integral_rmse_m":float(np.sqrt(np.mean((progress-expected)**2))),
              "mean_route_projection_error_m":float(error.mean()),"monotonicity_repairs":reversals,
              "outside_metadata_lap_scoring_nodes":int((~source_valid).sum()),
              "per_lap_first_sample_reset":False,"per_lap_span_stretch":False,"grid_bracketed":bool(progress[0]<=0 and progress[-1]>=track["s_m"][-1])}}
    return result


def _local_source(root):
    raw=Path(root)/"data/raw/openf1"
    document=read_json(Path(root)/"data/processed/dataset.json")
    metadata=read_json(raw/"laps.json")
    chosen={int(l["lap_number"]):l for l in metadata if l["driver_number"]==63 and l["session_key"]==11315}
    selected=[]
    for old in document["laps"]:
        number=int(old["lap_number"])
        if number not in (3,4,5,6,7,9):continue
        item=dict(chosen[number]);item.update(role="calibration" if number<=6 else "development",lap_time_s=float(item["lap_duration"]),
                                              sector_times_s=[float(item[f"duration_sector_{i}"]) for i in (1,2,3)],compound=old["compound"])
        selected.append(item)
    arrays=raw_arrays(read_json(raw/"car_data_63_laps3to9.json"),read_json(raw/"location_63_laps3to9.json"))
    return document,selected,arrays


def coverage_quality(lap,arrays,quality):
    start=epoch(lap["date_start"]);end=start+lap["lap_duration"]
    reasons=[];result={"lap_number":int(lap["lap_number"])}
    for channel in ("car","location"):
        times=arrays["car_t" if channel=="car" else "loc_t"]
        inside=times[(times>=start)&(times<=end)]
        gaps=np.diff(np.r_[start,inside,end])
        result[f"{channel}_samples"]=len(inside);result[f"max_{channel}_gap_s"]=float(gaps.max())
        bracketed=bool(times[0]<=start and times[-1]>=end);result[f"{channel}_boundary_bracketed"]=bracketed
        if len(inside)<quality[f"minimum_{channel}_samples_per_lap"]:reasons.append(f"insufficient_{channel}_samples")
        if gaps.max()>quality[f"maximum_{channel}_gap_s"]:reasons.append(f"excessive_{channel}_gap")
        if quality["require_boundary_bracketing"] and not bracketed:reasons.append(f"{channel}_boundary_not_bracketed")
    return {**result,"passed":not reasons,"reasons":reasons}


def metadata_quality(lap,metadata,quality):
    # Avoid the legacy pace anomaly filter: no selection by fast/slow lap time.
    import re
    from .data_helpers import _bad_intervals
    reasons=[];number=int(lap["lap_number"])
    if not lap.get("date_start") or not lap.get("lap_duration") or lap["lap_duration"]<=0:
        return {"lap_number":number,"passed":False,"reasons":["missing_start_or_complete_duration"]}
    start=epoch(lap["date_start"]);end=start+lap["lap_duration"]
    if number<=2:reasons.append("first_two_laps")
    pits={p["lap_number"] for p in metadata["pit"] if p["driver_number"]==63}
    if lap.get("is_pit_out_lap") or number in pits or number-1 in pits:reasons.append("pit_in_out_window")
    if any(a<end and b>start for a,b in _bad_intervals(metadata["race_control"])):reasons.append("yellow_red_sc_vsc_window")
    for message in metadata["race_control"]:
        found=re.search(r"CAR\s+(\d+).*DELETED.*\bLAP\s+(\d+)",message.get("message",""))
        if found and int(found[1])==63 and int(found[2])==number:reasons.append("deleted_lap")
    weather=[w for w in metadata["weather"] if start-75<=epoch(w["date"])<=end+75]
    if not weather or any(w.get("rainfall")!=0 for w in weather):reasons.append("rain_or_weather_missing")
    sectors=[lap.get(f"duration_sector_{i}") for i in (1,2,3)]
    if any(x is None or not np.isfinite(x) or x<=0 for x in sectors) or abs(sum(x or 0 for x in sectors)-lap["lap_duration"])>.01:
        reasons.append("incomplete_or_inconsistent_sectors")
    return {"lap_number":number,"passed":not reasons,"reasons":reasons}


def select_stint_roles(stints,laps,statuses):
    """Return the first qualifying MEDIUM stint, never sort by observed pace."""
    ordered=sorted([s for s in stints if s.get("driver_number")==63 and s.get("compound")=="MEDIUM"],
                   key=lambda s:min((l["date_start"] for l in laps if s["lap_start"]<=l["lap_number"]<=s["lap_end"] and l.get("date_start")),default="9999"))
    for stint in ordered:
        passing=sorted([l for l in laps if stint["lap_start"]<=l["lap_number"]<=stint["lap_end"] and statuses.get(int(l["lap_number"]),{}).get("passed")],key=lambda l:l["date_start"])
        if len(passing)>=6:
            return {"stint_number":stint["stint_number"],"compound":"MEDIUM","adaptation_laps":[int(l["lap_number"]) for l in passing[:3]],
                    "test_laps":[int(l["lap_number"]) for l in passing[3:6]],"quality_passing_laps_in_order":[int(l["lap_number"]) for l in passing],
                    "rejected_laps":[statuses[int(l["lap_number"])] for l in laps if stint["lap_start"]<=l["lap_number"]<=stint["lap_end"] and not statuses[int(l["lap_number"])]["passed"]]}
    raise DataUnavailable("No MEDIUM stint contains six quality-passing complete laps")
