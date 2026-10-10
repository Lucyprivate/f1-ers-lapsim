"""Public-data preparation with explicit provenance and no invented ERS labels.

The default dataset is the 2026 Austrian Race, Russell laps 3--7 and 9. Four
chronologically earlier complete laps calibrate; two complete later laps validate.
OpenF1 location is only used for approximate circuit registration/progress.  The
TUM line is a precomputed minimum-curvature line, not a minimum-lap-time solution.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

SESSION_KEY = 11315
DRIVER_NUMBER = 63
LAP_NUMBERS = (3, 4, 5, 6, 7, 9)
TUM_COMMIT = "e59595d1f3573b30d1ded6a08984935b957688e0"
_LAST_API_REQUEST = 0.0


class DataUnavailable(RuntimeError):
    """A required real-data gate failed; never replaced with synthetic laps."""


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _source_urls():
    sources = {}
    for endpoint in ("sessions", "drivers", "laps", "race_control", "weather", "pit", "stints"):
        sources[f"openf1/{endpoint}.json"] = f"https://api.openf1.org/v1/{endpoint}?session_key={SESSION_KEY}"
    window = {"session_key": SESSION_KEY, "driver_number": DRIVER_NUMBER,
              "date>": "2026-06-28T13:05:46", "date<": "2026-06-28T13:14:12"}
    for endpoint in ("car_data", "location"):
        sources[f"openf1/{endpoint}_63_laps3to9.json"] = "https://api.openf1.org/v1/" + endpoint + "?" + urlencode(window)
    tum = f"https://raw.githubusercontent.com/TUMFTM/racetrack-database/{TUM_COMMIT}/"
    for remote, local in (("tracks/Spielberg.csv", "Spielberg_centerline.csv"),
                          ("racelines/Spielberg.csv", "Spielberg_minimum_curvature.csv"),
                          ("LICENSE", "LICENSE"), ("README.md", "README.md")):
        sources[f"tum/{local}"] = tum + remote
    sources.update({
        "fia/power_unit_information.pdf": "https://www.fia.com/system/files/decision-document/2026_austrian_grand_prix_-_power_unit_information.pdf",
        "fia/technical_regulations_issue19.pdf": "https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_19_-_2026-06-25.pdf",
        "fia/sporting_regulations_issue07.pdf": "https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_07_-_2026-06-25.pdf",
        "fia/circuit_map.pdf": "https://www.fia.com/system/files/decision-document/2026_austrian_grand_prix_-_competition_notes_-_circuit_map_pit_lane_drawing_emergency_exits_map_and_red_zone.pdf",
        "fia/competition_notes.pdf": "https://www.fia.com/system/files/decision-document/2026_austrian_grand_prix_-_race_directors_competition_notes.pdf",
    })
    return sources


def _fetch(url: str, path: Path, refresh: bool):
    global _LAST_API_REQUEST
    if path.exists() and not refresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        if "openf1.org" in url:
            time.sleep(max(0.0, 2.2 - (time.monotonic() - _LAST_API_REQUEST)))
            _LAST_API_REQUEST = time.monotonic()
        try:
            with urlopen(Request(url, headers={"User-Agent": "f1-ers-lab-public-research/1.0"}), timeout=40) as response:
                body = response.read()
            if not body:
                raise DataUnavailable(f"Empty source: {url}")
            if path.suffix == ".json":
                value = json.loads(body)
                if not isinstance(value, list) or not value:
                    raise DataUnavailable(f"No records returned by {url}")
            path.write_bytes(body)
            _write(path.with_name(path.name+".download.json"), {"url":url,"downloaded_at_utc":datetime.now(timezone.utc).isoformat()})
            return
        except Exception as exc:
            if attempt == 2:
                raise DataUnavailable(f"Cannot obtain required real source {url}: {exc}") from exc
            time.sleep(3.0 * (attempt + 1))


def _polyline(points):
    segment = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segment, axis=1)
    if np.any(lengths <= 0):
        raise DataUnavailable("Track contains duplicate adjacent points")
    return np.r_[0.0, np.cumsum(lengths)], float(lengths.sum())


def _periodic_xy(s, raw_s, points, length):
    return np.column_stack([np.interp(np.mod(s, length), raw_s, np.r_[points[:,j], points[0,j]]) for j in (0,1)])


def _similarity(template, observed):
    xc, yc = template-template.mean(axis=0), observed-observed.mean(axis=0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    scale = float(np.sum((xc @ rot)*yc)/np.sum(xc*xc))
    shift = observed.mean(axis=0) - scale * template.mean(axis=0) @ rot
    residual = float(np.sqrt(np.mean(np.sum((scale*template@rot+shift-observed)**2,axis=1)))/scale)
    return residual, scale, rot, shift


def _speed_time(lap, car_t, speed):
    start = _epoch(lap["date_start"])
    end = start + lap["lap_duration"]
    inside = (car_t > start) & (car_t < end)
    tt = np.r_[start, car_t[inside], end]
    vv = np.interp(tt, car_t, speed)
    distance = np.r_[0.0, np.cumsum((vv[1:]+vv[:-1])/2*np.diff(tt))]
    return tt, vv, distance


def _register(route, laps, car_t, speed, loc_t, loc_xy):
    raw_s, length = _polyline(route)
    progress, observed = [], []
    for lap in laps[:4]:
        tt, _, dist = _speed_time(lap, car_t, speed)
        inside = (loc_t >= tt[0]) & (loc_t <= tt[-1])
        progress.append(np.interp(loc_t[inside], tt, dist)*length/dist[-1])
        observed.append(loc_xy[inside])
    q, y = np.concatenate(progress), np.concatenate(observed)
    def score(phase, direction):
        return _similarity(_periodic_xy(phase+direction*q, raw_s, route, length), y)
    best = min((score(p,d)[0], float(p), d) for p in np.arange(0,length,10) for d in (1,-1))
    best = min((score(p,best[2])[0], float(p), best[2]) for p in np.arange(best[1]-10,best[1]+10,0.25))
    rmse, scale, rot, shift = score(best[1],best[2])
    if rmse > 25 or not 8 < scale < 12:
        raise DataUnavailable(f"Track registration failed: residual={rmse:.1f}m scale={scale:.2f}")
    return raw_s, length, {"original_route_control_line_s_m": best[1]%length, "direction":best[2],
                          "rmse_m":rmse, "raw_location_units_per_m":scale,
                          "rotation":rot.tolist(), "translation_raw_units":shift.tolist(),
                          "fit_laps":[int(a["lap_number"]) for a in laps[:4]],
                          "method":"proper 2D similarity and periodic phase fit; integrated public speed gives approximate progress; calibration laps only"}


def _project(points, route, raw_s):
    delta = np.roll(route,-1,axis=0)-route
    norm2 = np.sum(delta*delta,axis=1)
    result, error = [], []
    for point in points:
        u=np.clip(np.sum((point-route)*delta,axis=1)/norm2,0,1)
        projected=route+u[:,None]*delta
        d2=np.sum((point-projected)**2,axis=1)
        i=int(np.argmin(d2))
        result.append(float(raw_s[i]+u[i]*math.sqrt(norm2[i])))
        error.append(float(math.sqrt(d2[i])))
    return np.asarray(result), np.asarray(error)


def _bad_intervals(messages):
    active, intervals = {}, []
    for item in sorted(messages,key=lambda a:a["date"]):
        when=_epoch(item["date"])
        key=(item.get("scope"),item.get("sector"))
        flag=item.get("flag")
        message=item["message"]
        if flag in ("YELLOW","DOUBLE YELLOW","RED"):
            active.setdefault(key,when)
        elif flag == "CLEAR":
            if key in active:intervals.append((active.pop(key),when))
        elif flag == "GREEN" and item.get("scope")=="Track":
            intervals.extend((v,when) for v in active.values());active.clear()
        elif item.get("category")=="SafetyCar":
            if "DEPLOYED" in message:active.setdefault(("SafetyCar",None),when)
            elif "ENDING" in message:
                # Conservative 60s buffer after ending message.
                if ("SafetyCar",None) in active:intervals.append((active.pop(("SafetyCar",None)),when+60))
    intervals.extend((v,float("inf")) for v in active.values())
    return intervals


def _clean_lap_check(lap, all_laps, messages, weather, pits, stints):
    reasons=[]
    start=_epoch(lap["date_start"]);end=start+lap["lap_duration"]
    sectors=[lap.get(f"duration_sector_{i}") for i in (1,2,3)]
    if lap["lap_number"] <= 2:reasons.append("start_or_warmup_lap")
    if lap.get("is_pit_out_lap"):reasons.append("pit_out_lap")
    pit_numbers={p["lap_number"] for p in pits if p["driver_number"]==lap["driver_number"]}
    if lap["lap_number"] in pit_numbers or lap["lap_number"]-1 in pit_numbers:reasons.append("pit_in_or_out_window")
    if any(a<end and b>start for a,b in _bad_intervals(messages)):reasons.append("yellow_red_sc_vsc_window")
    for m in messages:
        deleted=re.search(r"CAR\s+(\d+).*DELETED.*\bLAP\s+(\d+)",m["message"])
        if deleted and int(deleted[1])==lap["driver_number"] and int(deleted[2])==lap["lap_number"]:reasons.append("deleted_lap")
    nearby=[w for w in weather if start-75<=_epoch(w["date"])<=end+75]
    if not nearby or any(w.get("rainfall")!=0 for w in nearby):reasons.append("rainfall_or_weather_missing")
    stint=next((x for x in stints if x["driver_number"]==lap["driver_number"] and x["lap_start"]<=lap["lap_number"]<=x["lap_end"]),None)
    if not stint or stint["compound"] not in ("SOFT","MEDIUM","HARD"):reasons.append("dry_compound_unverified")
    valid=[x["lap_duration"] for x in all_laps if x["driver_number"]==lap["driver_number"] and x.get("lap_duration") and not x.get("is_pit_out_lap") and x["lap_number"]>2 and x["lap_duration"]<90]
    if not valid or lap["lap_duration"]>1.08*np.median(valid):reasons.append("anomalous_lap_duration")
    if any(x is None or x<=0 for x in sectors) or abs(sum(sectors)-lap["lap_duration"])>.01:reasons.append("incomplete_or_inconsistent_sector_times")
    return reasons,stint,nearby


def _repair_signal(t, values, binary=False):
    good=np.isfinite(values)&(values>=0)&(values<=100)
    if good.sum()<2:raise DataUnavailable("Telemetry channel has too few valid samples")
    result=np.interp(t,t[good],values[good])
    if binary:result=(result>=50).astype(float)
    else:result=result/100.0
    return result,int((~good).sum())


def prepare_data(project_root: Path, refresh: bool=False) -> dict:
    """Fetch if needed, audit and prepare cached real data. No network when cached."""
    root=Path(project_root).resolve();raw=root/"data/raw";processed=root/"data/processed"
    processed.mkdir(parents=True,exist_ok=True)
    sources=_source_urls()
    for relative,url in sources.items():_fetch(url,raw/relative,refresh)
    rules=_read(root/"config/rules_austria_2026.json")
    all_laps=_read(raw/"openf1/laps.json");messages=_read(raw/"openf1/race_control.json")
    weather=_read(raw/"openf1/weather.json");pits=_read(raw/"openf1/pit.json");stints=_read(raw/"openf1/stints.json")
    sessions=_read(raw/"openf1/sessions.json")
    if len(sessions)!=1 or sessions[0]["year"]!=2026 or sessions[0]["session_type"]!="Race" or sessions[0]["circuit_short_name"]!="Spielberg":
        raise DataUnavailable("The cached session is not the intended 2026 Austria Race")
    chosen=sorted([a for a in all_laps if a["driver_number"]==DRIVER_NUMBER and a["lap_number"] in LAP_NUMBERS],key=lambda a:a["date_start"])
    if len(chosen)!=6:raise DataUnavailable("Required six complete independent real laps are unavailable")
    clean=[]
    for lap in chosen:
        reasons,stint,nearby=_clean_lap_check(lap,all_laps,messages,weather,pits,stints)
        clean.append({"lap_number":lap["lap_number"],"rejected_reasons":reasons,"compound":stint["compound"] if stint else None,
                      "rainfall_samples":len(nearby),"max_rainfall":max((w["rainfall"] for w in nearby),default=None)})
        if reasons:raise DataUnavailable(f"Selected lap {lap['lap_number']} failed clean-data gate: {reasons}")
    car=sorted(_read(raw/"openf1/car_data_63_laps3to9.json"),key=lambda a:a["date"])
    loc=sorted(_read(raw/"openf1/location_63_laps3to9.json"),key=lambda a:a["date"])
    car_t=np.asarray([_epoch(a["date"]) for a in car]);speed=np.asarray([a["speed"]/3.6 for a in car])
    loc_t=np.asarray([_epoch(a["date"]) for a in loc]);loc_xy=np.asarray([[a["x"],a["y"]] for a in loc],float)
    throttle,throttle_bad=_repair_signal(car_t,np.asarray([a["throttle"] for a in car],float))
    brake,brake_bad=_repair_signal(car_t,np.asarray([a["brake"] for a in car],float),binary=True)
    route=np.loadtxt(raw/"tum/Spielberg_minimum_curvature.csv",delimiter=",")
    centre=np.loadtxt(raw/"tum/Spielberg_centerline.csv",delimiter=",")
    raw_s,length,registration=_register(route,chosen,car_t,speed,loc_t,loc_xy)
    n=int(round(length/10.0));s=np.arange(n,dtype=float)*length/n;ds=length/n
    phase=registration["original_route_control_line_s_m"];direction=registration["direction"]
    xy=_periodic_xy(phase+direction*s,raw_s,route,length)
    # Periodic 5-point derivatives preserve closed-loop curvature at the seam.
    dxy=(np.roll(xy,2,axis=0)-8*np.roll(xy,1,axis=0)+8*np.roll(xy,-1,axis=0)-np.roll(xy,-2,axis=0))/(12*ds)
    ddxy=(-np.roll(xy,2,axis=0)+16*np.roll(xy,1,axis=0)-30*xy+16*np.roll(xy,-1,axis=0)-np.roll(xy,-2,axis=0))/(12*ds*ds)
    curvature=(dxy[:,0]*ddxy[:,1]-dxy[:,1]*ddxy[:,0])/np.maximum(np.linalg.norm(dxy,axis=1)**3,1e-9)
    nearest=np.argmin(np.sum((xy[:,None,:]-centre[None,:,:2])**2,axis=2),axis=1)
    tangent=np.roll(centre[:,:2],-1,axis=0)-np.roll(centre[:,:2],1,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normals=np.column_stack([-tangent[:,1],tangent[:,0]])
    offset=np.sum((xy-centre[nearest,:2])*normals[nearest],axis=1)
    left=centre[nearest,3]-offset;right=centre[nearest,2]+offset
    if direction<0:left,right=right,left
    if np.min(left)<=0 or np.min(right)<=0:raise DataUnavailable("Fixed minimum-curvature route is outside inferred source track widths")
    np.savez_compressed(processed/"track.npz",s_m=s,x_m=xy[:,0],y_m=xy[:,1],curvature_1pm=curvature,
                        width_left_m=left,width_right_m=right,fia_s_m=s*rules["fia_centreline_length_m"]/length)
    rot=np.asarray(registration["rotation"]);shift=np.asarray(registration["translation_raw_units"]);scale=registration["raw_location_units_per_m"]
    projected_source_s,projection_error=_project((loc_xy-shift)@rot.T/scale,route,raw_s)
    projected_mod=np.mod(direction*(projected_source_s-phase),length)
    lap_records=[];coverage=[]
    for j,lap in enumerate(chosen):
        tt,vv,dist=_speed_time(lap,car_t,speed)
        lt=np.r_[tt[0],loc_t[(loc_t>tt[0])&(loc_t<tt[-1])],tt[-1]]
        q=np.interp(lt,tt,dist)*length/dist[-1]
        # Unwrap the raw projected position with approximate speed progress.
        p=np.interp(lt,loc_t,np.unwrap(projected_mod*2*np.pi/length)*length/(2*np.pi))
        p-=round((p[0]-q[0])/length)*length
        nonmonotonic=int(np.sum(np.diff(p)<0));p=np.maximum.accumulate(p)
        span=p[-1]-p[0]
        if span<0.96*length or span>1.04*length:raise DataUnavailable(f"Lap {lap['lap_number']} position does not cover one circuit")
        progress=(p-p[0])*length/span
        v=np.interp(s,progress,np.interp(lt,car_t,speed))
        th=np.interp(s,progress,np.interp(lt,car_t,throttle))
        br=(np.interp(s,progress,np.interp(lt,car_t,brake))>=.5).astype(float)
        np.savez_compressed(processed/f"lap_{lap['lap_number']:02d}.npz",s_m=s,speed_mps=v,throttle=th,brake=br,t_s=np.interp(s,progress,lt-tt[0]),
                            time_s=lt-tt[0],progress_m=progress,raw_interpolated_speed_mps=np.interp(lt,car_t,speed))
        sel=(car_t>=tt[0])&(car_t<=tt[-1]);ls=(loc_t>=tt[0])&(loc_t<=tt[-1])
        max_gap=float(np.diff(tt).max())
        if sel.sum()<200 or ls.sum()<200 or max_gap>2.0:raise DataUnavailable(f"Lap {lap['lap_number']} telemetry coverage is insufficient")
        cov={"lap_number":lap["lap_number"],"car_samples":int(sel.sum()),"location_samples":int(ls.sum()),
             "max_car_gap_s":max_gap,"speed_integral_m":float(dist[-1]),"position_projected_span_m":float(span),
             "progress_normalisation_scale":float(length/span),"position_monotonicity_repairs":nonmonotonic,
             "mean_position_distance_to_fixed_route_m":float(projection_error[ls].mean())}
        coverage.append(cov)
        lap_records.append({"lap_number":int(lap["lap_number"]),"driver_number":DRIVER_NUMBER,
                            "role":"calibration" if j<4 else "validation","lap_time_s":float(lap["lap_duration"]),
                            "sector_times_s":[float(lap[f"duration_sector_{k}"]) for k in (1,2,3)],
                            "date_start":lap["date_start"],"data_file":f"lap_{lap['lap_number']:02d}.npz","compound":clean[j]["compound"]})
    # Corner locations are inferred from the old source line, not surveyed FIA coordinates.
    corner_windows={1:(200,330),3:(1100,1300),4:(1920,2130),6:(2440,2650),7:(2760,2930),8:(2940,3090),9:(3480,3680),10:(3670,3900)}
    corners={}
    for turn,(lo,hi) in corner_windows.items():
        ids=np.flatnonzero((s>=lo)&(s<=hi));corners[str(turn)]=float(s[ids[np.argmax(np.abs(curvature[ids]))]])
    cal_brake=np.mean([np.load(processed/a["data_file"])["brake"] for a in lap_records[:4]],axis=0)
    approx_aero=[]
    for activation,next_turn in zip(rules["aero"]["normal_grip_activation"],(1,3,4,9)):
        begin=(corners[str(activation["turn"])]+activation["offset_after_turn_m"]*length/rules["fia_centreline_length_m"])%length
        next_s=corners[str(next_turn)]
        ahead=(s-begin)%length;distance_to_corner=(next_s-begin)%length
        ids=np.flatnonzero((ahead>20)&(ahead<distance_to_corner)&(cal_brake>=.5))
        end=float(s[ids[np.argmin(ahead[ids])]]) if len(ids) else (next_s-80)%length
        approx_aero.append({"zone":activation["zone"],"start_m":float(begin),"end_m":end,
                            "official_start_offset":activation,"status":"approximate_route_projection_and_calibration_braking_end"})
    audit={"gate_passed":True,"status":"ready_with_explicit_limits","real_laps":6,"calibration_laps":4,"validation_laps":2,
           "split_method":"chronological complete laps; no random telemetry-point split", "clean_lap_checks":clean,"telemetry_coverage":coverage,
           "track_registration":registration,"original_tum_centreline_length_m":_polyline(centre[:,:2])[1],
           "fixed_minimum_curvature_route_length_m":length,"route_length_vs_fia_centreline_pct":100*(length/rules["fia_centreline_length_m"]-1),
           "track_epoch":"TUM repository snapshot e59595d at 2021-09-18T14:28:20Z; geometry may be older; not a surveyed 2026 map",
           "track_coordinate_frame":"TUM local planar metre coordinates, proper rotation fitted to approximate OpenF1 location, source orientation retained",
           "fia_distance_mapping":"s_fia=s_route*4326/route_length; explicit approximate linear mapping, not surveyed loop positions",
           "position_usage":"Approximate circuit registration and progress only. No GPS accuracy or actual racing-line claim.",
           "speed_precision":"Original OpenF1 speed is integer km/h; approximately 3.7Hz. Linear time/space interpolation does not create additional measurement precision.",
           "signal_repairs":{"throttle_out_of_0_100_range":throttle_bad,"brake_out_of_0_100_range":brake_bad,
                             "method":"Invalid values (including 104) linearly filled from valid neighbours; throttle fraction, brake boolean after 50 threshold; raw cache unchanged"},
           "observed_ers_available":False,"observed_soc_available":False,"observed_aero_state_available":False,
           "unidentifiable_from_speed":"ICE/MGU-K split, SOC and real team deployment policy cannot be uniquely recovered from speed/throttle/brake",
           "full_fia_compliance_claim_supported":False,"rule_unknowns":rules["unavailable_or_unverified"],
           "validation_limitations":["Same driver/stint/event only; no leave-one-circuit generalisation test", "Real laps include changing fuel, tyre age, traffic and possible Overtake usage; counterfactual fixed vehicle Overtake-off simulation is not the exact observed car state", "2026 drs data is null; aero states are approximate", "FIA Doc5 centreline typo/conflict preserved", "Vertical track gradient omitted in the required 2D model", "Lap date_start timestamps are approximate; official lap/sector durations are retained unchanged"]}
    dataset={"schema_version":1,"session":sessions[0],"driver":next(a for a in _read(raw/"openf1/drivers.json") if a["driver_number"]==DRIVER_NUMBER),
             "track":{"name":"Spielberg - Red Bull Ring","data_file":"track.npz","length_m":length,"ds_m":ds,"nodes":n,
                      "line_type":"TUM minimum-curvature fixed route","corner_s_m":corners,"approximate_aero_zones_m":approx_aero,
                      "sector_end_s_m":[1215*length/4326,2912*length/4326,length]},"laps":lap_records,"audit":audit}
    _write(processed/"dataset.json",dataset)
    manifest={"schema_version":1,"prepared_at_utc":datetime.now(timezone.utc).isoformat(),"session_key":SESSION_KEY,
              "meeting_key":sessions[0]["meeting_key"],"source_snapshot_tum_commit":TUM_COMMIT,"sources":[],"audit":audit}
    for relative,url in sources.items():
        path=raw/relative
        download_metadata=path.with_name(path.name+".download.json")
        if download_metadata.exists():actual_url=_read(download_metadata)["url"]
        else:actual_url=url.replace(TUM_COMMIT,"master") if relative.startswith("tum/") else url
        manifest["sources"].append({"path":"data/raw/"+relative,"url":actual_url,"reproducible_url":url,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "bytes":path.stat().st_size,"downloaded_at_utc":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()})
    extra_sources={
        "tum/commit.json":"https://api.github.com/repos/TUMFTM/racetrack-database/commits/master",
        "fia/technical_regulations_issue17.pdf":"https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_17_-_2026-04-28.pdf",
    }
    old_window={"session_key":SESSION_KEY,"driver_number":DRIVER_NUMBER,"date>":"2026-06-28T13:05:46","date<":"2026-06-28T13:12:59"}
    for endpoint in ("car_data","location"):
        extra_sources[f"openf1/{endpoint}_63_laps3to8.json"]="https://api.openf1.org/v1/"+endpoint+"?"+urlencode(old_window)
    for relative,url in extra_sources.items():
        path=raw/relative
        if path.exists():
            manifest["sources"].append({"path":"data/raw/"+relative,"url":url,"used_for_simulation":False,
                                       "note":"Retained acquisition audit snapshot; not the applicable rule profile or final telemetry window",
                                       "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size,
                                       "downloaded_at_utc":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()})
    processed_files=[processed/"track.npz",processed/"dataset.json"]+[processed/a["data_file"] for a in lap_records]
    processed_files.extend(sorted((processed/"fia_visual_evidence").glob("*.png")))
    processed_files.append(root/"config/rules_austria_2026.json")
    for path in processed_files:
        manifest.setdefault("processed",[]).append({"path":path.relative_to(root).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    _write(root/"data/manifest.json",manifest)
    (root/"third_party").mkdir(exist_ok=True)
    shutil.copyfile(raw/"tum/LICENSE",root/"third_party/LICENSE_TUM.txt")
    return manifest


def load_dataset(project_root: Path) -> dict:
    """Load prepared ndarray data; prepare offline from the raw cache if absent."""
    root=Path(project_root).resolve();processed=root/"data/processed"
    if not (processed/"dataset.json").exists():prepare_data(root)
    document=_read(processed/"dataset.json")
    track=dict(document["track"])
    with np.load(processed/track.pop("data_file"),allow_pickle=False) as archive:
        track.update({k:archive[k].copy() for k in archive.files})
    laps=[]
    for metadata in document["laps"]:
        lap=dict(metadata)
        with np.load(processed/lap.pop("data_file"),allow_pickle=False) as archive:
            lap.update({k:archive[k].copy() for k in archive.files})
        laps.append(lap)
    return {"track":track,"laps":laps,"rules":_read(root/"config/rules_austria_2026.json"),"audit":document["audit"],
            "session":document["session"],"driver":document["driver"]}
