"""Public raw channels, calibrated control-line alignment and frozen FP2 transfer.

No observed SOC, deployment or brake-pressure labels are fabricated. A denied
remote source is a pending cross-session test, never a substitute real session.
"""
from __future__ import annotations

from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import time
import numpy as np
from scipy.optimize import least_squares

from .protocol import (DataUnavailable,ensure_protocol_lock,protocol,read_json,write_json,
                       sha256,verify_test_authorization,authorize_test_access)
from .geometry import build_track,remap_rules,polyline_length

_LAST_REQUEST=0.


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


def load_local_dataset(root,candidate="alignment",ds_m=5.):
    root=Path(root).resolve();lock=ensure_protocol_lock(root)
    document,metadata,arrays=_local_source(root)
    path=root/"data/v3/local/registration.json"
    if path.exists():registration=read_json(path)
    else:registration=fit_registration(root,[l for l in metadata if l["role"]=="calibration"],arrays,cache_path=path)
    if registration["fit_laps"]!=[3,4,5,6] or registration.get("fit_role")!="calibration":
        raise DataUnavailable("Cached registration is not calibration-only")
    track=build_track(root,candidate,ds_m,registration)
    laps=[process_lap(l,arrays,track,registration) for l in metadata]
    rules=remap_rules(read_json(root/"config/rules_austria_2026.json"),track)
    audit={"schema_version":3,"registration":registration,"signal_audit":arrays["signal_audit"],
           "alignment_per_lap":[{"lap_number":l["lap_number"],"role":l["role"],**l["alignment_audit"]} for l in laps],
           "protocol_sha256":lock["protocol_sha256"],"historical_only_laps":[10,11,12],"test_arrays_loaded":False,
           "data_gate_passed":all(l["alignment_audit"]["grid_bracketed"] and .96<=l["alignment_audit"]["projected_span_fraction"]<=1.04 for l in laps),
           "geometry":track["geometry_audit"],"role_selection":"Fixed chronological calibration 3–6, development 7/9; no test labels"}
    audit_path=root/f"data/v3/local/audit_{candidate}_{float(ds_m):g}m.json"
    if not audit_path.exists():write_json(audit_path,audit)
    elif read_json(audit_path)!=audit:
        raise DataUnavailable("Cached local audit changed; do not rewrite frozen inputs during replay")
    return {"track":track,"rules":rules,"laps":laps,"audit":audit,"session":document["session"],"driver":document["driver"],"registration":registration}


def _bounded_fetch(url,path):
    global _LAST_REQUEST
    path=Path(path)
    if path.exists():return read_json(path)
    delay=max(0.,2.2-(time.monotonic()-_LAST_REQUEST))
    if delay:time.sleep(delay)
    _LAST_REQUEST=time.monotonic()
    try:
        with urlopen(Request(url,headers={"User-Agent":"f1-ers-lab-v3-public-research/1.0"}),timeout=25) as response:
            payload=response.read()
        import json
        value=json.loads(payload)
        if not isinstance(value,list):raise DataUnavailable("API response is not a records array")
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        write_json(path.with_name(path.name+".download.json"),{"url":url,"downloaded_at_utc":datetime.now(timezone.utc).isoformat(),"sha256":sha256(path)})
        return value
    except HTTPError as exc:
        message=exc.read(4096).decode("utf-8",errors="replace")
        raise DataUnavailable(f"HTTP {exc.code} from {url}: {message}") from exc
    except Exception as exc:
        raise DataUnavailable(f"Required public data unavailable: {url}: {exc}") from exc


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
    from f1ers.data import _bad_intervals
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


def _new_session_experiment_frozen(root):
    """The preregistration may precede acquisition; a model freeze may not move."""
    if any((root/"data/v3"/name).exists() for name in
           ("test_access.json","test_arrays_unsealed.json","experiment_lock.json")):
        return True
    # Copied v2 experiment results have the same filename but schema 2. They
    # must remain intact without blocking a new, unfrozen v3 acquisition.
    candidates=list((root/"results").rglob("frozen_configuration*.json"))
    candidates+=list((root/"data/v3").glob("frozen_configuration*.json"))
    for candidate in candidates:
        try:
            if read_json(candidate).get("schema_version")!=2:return True
        except (OSError,ValueError,AttributeError):
            return True
    return False


def prepare_new_session(root,allow_network=True):
    root=Path(root).resolve();lock=ensure_protocol_lock(root);p=protocol(root)
    path=root/"data/v3/fp2/status.json"
    if path.exists():
        cached=read_json(path)
        raw=root/"data/v3/fp2/raw"
        offline_unqueried=(cached.get("status")=="pending"
            and cached.get("reason")=="Offline mode: exact FP2 public sources are not cached"
            and not cached.get("query_attempted",False)
            and not any(cached.get(key) for key in ("queried_at_utc","network_checked_at_utc","bounded_attempt_count"))
            and not (raw.exists() and any(raw.glob("*.json"))))
        # Only the explicit unqueried offline placeholder may advance to an
        # authorized first request. A real denial/quality result, ready cache
        # or frozen experiment remains byte-for-byte unchanged.
        if not (allow_network and offline_unqueried and not _new_session_experiment_frozen(root)):
            return cached
    if not allow_network:
        report={"schema_version":3,"status":"pending","reason":"Offline mode: exact FP2 public sources are not cached","query_attempted":False,
                "protocol_sha256":lock["protocol_sha256"],"target":p["new_session_identity"],"session_substitution":False,
                "driver_substitution":False,"compound_substitution":False,"late_test_scored":False,
                "cross_session_acceptance":"Pending real FP2 data; cannot pass transfer or release training"}
        write_json(path,report)
        return report
    raw=root/"data/v3/fp2/raw"
    queried_at=datetime.now(timezone.utc).isoformat()
    try:
        query={"year":2026,"country_name":"Austria","session_name":"Practice 2"}
        sessions=_bounded_fetch("https://api.openf1.org/v1/sessions?"+urlencode(query),raw/"sessions.json")
        exact=[s for s in sessions if s.get("year")==2026 and s.get("country_name")=="Austria" and s.get("circuit_short_name")=="Spielberg" and s.get("session_name")=="Practice 2"]
        if len(exact)!=1:raise DataUnavailable("Exact 2026 Austria FP2 session metadata is absent or ambiguous")
        session=exact[0];key=session["session_key"]
        metadata={}
        for endpoint in ("laps","race_control","weather","pit","stints"):
            query={"session_key":key}
            if endpoint in ("laps","pit","stints"):query["driver_number"]=63
            metadata[endpoint]=_bounded_fetch(f"https://api.openf1.org/v1/{endpoint}?"+urlencode(query),raw/f"{endpoint}.json")
        laps=[l for l in metadata["laps"] if l.get("driver_number")==63]
        medium=[s for s in metadata["stints"] if s.get("driver_number")==63 and s.get("compound")=="MEDIUM"]
        candidates=[l for l in laps if l.get("date_start") and l.get("lap_duration") and any(s["lap_start"]<=l["lap_number"]<=s["lap_end"] for s in medium)]
        if not candidates:raise DataUnavailable("No real #63 MEDIUM complete-lap candidates in FP2")
        start=min(epoch(l["date_start"]) for l in candidates)-3
        end=max(epoch(l["date_start"])+l["lap_duration"] for l in candidates)+3
        fmt=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat()
        query={"session_key":key,"driver_number":63,"date>":fmt(start),"date<":fmt(end)}
        car=_bounded_fetch("https://api.openf1.org/v1/car_data?"+urlencode(query),raw/"car_data_63_medium.json")
        loc=_bounded_fetch("https://api.openf1.org/v1/location?"+urlencode(query),raw/"location_63_medium.json")
        arrays=raw_arrays(car,loc);statuses={}
        for lap in laps:
            q=metadata_quality(lap,metadata,p["data_quality"])
            if q["passed"]:
                cov=coverage_quality(lap,arrays,p["data_quality"]);q.update(cov)
            statuses[int(lap["lap_number"])]=q
        selection=select_stint_roles(medium,laps,statuses)
        chosen={int(l["lap_number"]):l for l in laps}
        early=[{**chosen[n],"role":"adaptation"} for n in selection["adaptation_laps"]]
        registration=fit_registration(root,early,arrays,role="adaptation",cache_path=root/"data/v3/fp2/registration.json")
        # Span screening uses the already fitted early registration, no late
        # speed residual. If screening fails, this session is pending; no refit.
        track=build_track(root,"alignment",5.,registration)
        for number in selection["quality_passing_laps_in_order"]:
            aligned=process_lap(chosen[number],arrays,track,registration)
            span=aligned["alignment_audit"]["projected_span_fraction"]
            if not .96<=span<=1.04 or not aligned["alignment_audit"]["grid_bracketed"]:
                statuses[number]={**statuses[number],"passed":False,"reasons":["frozen_projection_span_or_boundary_failed"]}
        validated=select_stint_roles(medium,laps,statuses)
        if validated["adaptation_laps"]!=selection["adaptation_laps"] or validated["stint_number"]!=selection["stint_number"]:
            raise DataUnavailable("Projection quality changed early adaptation identity; no late-informed registration refit allowed")
        write_json(root/"data/v3/fp2/selection.json",validated)
        report={"schema_version":3,"status":"ready","query_attempted":True,"session":session,"driver_number":63,"compound":"MEDIUM",
                "queried_at_utc":queried_at,"protocol_sha256":lock["protocol_sha256"],"selection":validated,
                "quality":list(statuses.values()),"late_test_scored":False,"registration_fit_laps":selection["adaptation_laps"],
                "raw_sources":[{"path":x.relative_to(root).as_posix(),"sha256":sha256(x)} for x in sorted(raw.glob("*.json")) if not x.name.endswith(".download.json")]}
    except DataUnavailable as exc:
        report={"schema_version":3,"status":"pending","query_attempted":True,"reason":str(exc),"queried_at_utc":queried_at,
                "protocol_sha256":lock["protocol_sha256"],"target":p["new_session_identity"],
                "session_substitution":False,"driver_substitution":False,"compound_substitution":False,
                "late_test_scored":False,"cross_session_acceptance":"Pending real FP2 data; cannot pass transfer or release training"}
    write_json(path,report)
    return report


def load_new_session(root,include_test=False,candidate="alignment",ds_m=5.,registration=None):
    root=Path(root).resolve();ensure_protocol_lock(root)
    report=read_json(root/"data/v3/fp2/status.json")
    if report["status"]!="ready":raise DataUnavailable(report["reason"])
    for source in report["raw_sources"]:
        if sha256(root/source["path"])!=source["sha256"]:raise DataUnavailable("FP2 raw source changed after selection")
    if include_test:
        claim=verify_test_authorization(root)
        unseal=root/"data/v3/test_arrays_unsealed.json"
        try:
            import json
            with unseal.open("x",encoding="utf-8") as stream:
                json.dump({"configuration_sha256":claim["configuration_sha256"],"unsealed_at_utc":datetime.now(timezone.utc).isoformat()},stream,indent=2)
        except FileExistsError as exc:
            raise DataUnavailable("Late test arrays already unsealed; use saved replay results") from exc
    selected=read_json(root/"data/v3/fp2/selection.json")
    raw=root/"data/v3/fp2/raw"
    # The development path filters lap metadata before constructing arrays;
    # broad source cache is immutable and the returned dataset has no late labels.
    metadata=read_json(raw/"laps.json")
    numbers=selected["adaptation_laps"]+(selected["test_laps"] if include_test else [])
    chosen=[]
    for n in numbers:
        lap=next(dict(l) for l in metadata if int(l["lap_number"])==n and l["driver_number"]==63)
        lap.update(role="adaptation" if n in selected["adaptation_laps"] else "test",lap_time_s=float(lap["lap_duration"]),
                   sector_times_s=[float(lap[f"duration_sector_{i}"]) for i in (1,2,3)],compound="MEDIUM")
        chosen.append(lap)
    car=read_json(raw/"car_data_63_medium.json");location=read_json(raw/"location_63_medium.json")
    if not include_test:
        first=min(epoch(l["date_start"]) for l in chosen)-3
        last=max(epoch(l["date_start"])+l["lap_duration"] for l in chosen)+3
        first_test_start=min(epoch(l["date_start"]) for l in metadata if int(l["lap_number"]) in selected["test_laps"] and l["driver_number"]==63)
        # Never form an early car/position array from the reserved late lap,
        # even when broad acquisition padding reaches past the early endpoint.
        car=[r for r in car if first<=epoch(r["date"])<=last and epoch(r["date"])<first_test_start]
        location=[r for r in location if first<=epoch(r["date"])<=last and epoch(r["date"])<first_test_start]
    arrays=raw_arrays(car,location)
    registration=registration or read_json(root/"data/v3/fp2/registration.json")
    if registration["fit_role"]!="adaptation" or registration["fit_laps"]!=selected["adaptation_laps"]:
        raise DataUnavailable("FP2 coordinate fit used unauthorized late laps")
    track=build_track(root,candidate,ds_m,registration)
    laps=[process_lap(l,arrays,track,registration,strict_signal_window=True) for l in chosen]
    data_gate=all(l["alignment_audit"]["grid_bracketed"] and bool(np.all(l["scoring_signal_valid_mask"])) for l in laps)
    return {"track":track,"rules":remap_rules(read_json(root/"config/rules_austria_2026.json"),track),
            "laps":laps,"session":report["session"],
            "registration":registration,"audit":{"test_arrays_loaded":bool(include_test),"conditional_transfer":True,
              "registration_fit_laps":selected["adaptation_laps"],"selection":selected,"data_gate_passed":data_gate,
              "early_raw_cutoff_at_first_test_start":not include_test,"outside_metadata_signal_is_nan":True}}
