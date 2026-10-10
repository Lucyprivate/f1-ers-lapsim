"""Precommitted observable-accuracy tests and calibration-only multistart fitting."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import minimize

from .model import VehicleParams
from .solvers_v2 import solve_heuristic_v2,valid_result

ACCURACY_TARGET={"mean_speed_rmse_kmh":15.0,"max_abs_lap_error_pct":2.0,"max_abs_sector_error_s":1.0}
BOUNDS=np.array([[1.2,2.4],[.6,2.0],[1.8,6.0]])
PARAMETERS=("mu","drag_area_m2","downforce_area_m2")


def compare_observables(dataset,result,role):
    """All samples of an entire lap contribute; sectors use actual solver dt."""
    track=dataset["track"];length=float(track["length_m"])
    expected=len(track.get("s_m",result["s_m"]))
    if (not result["summary"].get("completed_lap",True)
            or any(len(result[k])!=expected for k in ("s_m","speed_mps","dt_s"))
            or any(not np.isfinite(result[k]).all() for k in ("s_m","speed_mps","dt_s"))):
        return {"metrics":[],"reference_speed_mps":[],"role":role,
            "invalid_reason":"Incomplete or nonfinite causal trajectory; missing segments are not interpolated or scored."}
    ss=np.r_[np.asarray(result["s_m"],float),length]
    speed=np.asarray(result["speed_mps"],float)
    # Causal results expose node speeds; terminal speed is kept separately.
    terminal=float(result["summary"].get("speed_end_mps",speed[0]))
    vv=np.r_[speed,terminal]
    sim_t=np.r_[0.,np.cumsum(result["dt_s"])]
    metrics=[];references=[]
    for lap in dataset["laps"]:
        if lap["role"]!=role:continue
        s=np.asarray(lap["s_m"],float);v=np.asarray(lap["speed_mps"],float)
        predicted=np.interp(s,ss,vv)
        error=predicted-v
        real_t=float(lap["lap_time_s"]);lap_t=float(result["summary"]["lap_time_s"])
        sectors=np.asarray(lap["sector_times_s"],float)
        boundary=np.interp(np.cumsum(sectors)[:2],np.asarray(lap["t_s"]),s)
        sim_sector=np.diff(np.r_[0.,np.interp(boundary,ss,sim_t),lap_t])
        masks={"braking":np.asarray(lap["brake"])>=.5,
            "partial_throttle":(np.asarray(lap["throttle"])<.95)&(np.asarray(lap["brake"])<.5),
            "full_throttle_straight":(np.asarray(lap["throttle"])>=.95)&(np.abs(track["curvature_1pm"])<.002),
            "corner":np.abs(track["curvature_1pm"])>=.002}
        grouped={k:{"samples":int(m.sum()),"speed_rmse_kmh":float(np.sqrt(np.mean(error[m]**2))*3.6) if m.any() else None} for k,m in masks.items()}
        entry={"lap_number":lap["lap_number"],"role":role,"real_lap_time_s":real_t,"sim_lap_time_s":lap_t,
            "lap_error_s":lap_t-real_t,"lap_error_pct":100*(lap_t-real_t)/real_t,
            "speed_rmse_kmh":float(np.sqrt(np.mean(error**2))*3.6),
            "speed_mae_kmh":float(np.mean(np.abs(error))*3.6),"sample_count":len(s),
            "real_sector_times_s":sectors.tolist(),"sim_sector_times_s":sim_sector.tolist(),
            "sector_error_s":(sim_sector-sectors).tolist(),"grouped_speed_errors":grouped,
            "sector_boundary_method":"recorded telemetry time/distance interpolation; approximate"}
        metrics.append(entry);references.append(np.interp(result["s_m"],s,v))
    return {"metrics":metrics,"reference_speed_mps":np.mean(references,axis=0).tolist() if references else [],
        "reference_controller":"causal_calibration_heuristic","role":role,
        "interpretation":"Observable equivalent-vehicle check; real ERS, fuel, aero, wind and tyre state are unobserved."}


def accuracy_gate(comparison,physical=True):
    metrics=comparison["metrics"]
    if not metrics:return {"passed":False,"reasons":["no complete comparison laps"],"thresholds":ACCURACY_TARGET}
    if not all(np.isfinite([m["speed_rmse_kmh"],m["lap_error_pct"],*m["sector_error_s"]]).all() for m in metrics):
        return {"passed":False,"reasons":["nonfinite observable error"],"thresholds":dict(ACCURACY_TARGET),"precommitted":True}
    observed={"mean_speed_rmse_kmh":float(np.mean([m["speed_rmse_kmh"] for m in metrics])),
        "max_abs_lap_error_pct":max(abs(m["lap_error_pct"]) for m in metrics),
        "max_abs_sector_error_s":max(abs(e) for m in metrics for e in m["sector_error_s"])}
    reasons=[f"{k}: {v:.6g} exceeds {ACCURACY_TARGET[k]}" for k,v in observed.items() if not np.isfinite(v) or v>ACCURACY_TARGET[k]]
    if not physical:reasons.append("causal reference fails physical/boundary checks")
    return {"passed":not reasons,"reasons":reasons,"thresholds":dict(ACCURACY_TARGET),"observed":observed,
        "precommitted":True,"standard":"research acceptance target, not an FIA standard"}


def fit_vehicle_v2(dataset,previous_vehicle=None,progress=print,maxiter=200,maxfev=400):
    from .causal import CausalSimulator
    if any(l["role"]=="test" for l in dataset["laps"]):
        raise ValueError("Test arrays must not be supplied to model fitting")
    if len([l for l in dataset["laps"] if l["role"]=="calibration"])!=4:
        raise ValueError("Exactly the four precommitted calibration laps are required")
    base=VehicleParams()
    previous=previous_vehicle or base
    starts=[np.array([getattr(previous,k) for k in PARAMETERS]),np.array([getattr(base,k) for k in PARAMETERS]),BOUNDS.mean(axis=1)]
    cache={};evaluations=0
    def evaluate(x):
        nonlocal evaluations
        key=tuple(np.round(x,8))
        if key not in cache:
            p=replace(base,**dict(zip(PARAMETERS,map(float,x))))
            sim=CausalSimulator(dataset["track"],dataset["rules"],p)
            result=solve_heuristic_v2(sim)
            comparison=compare_observables(dataset,result,"calibration")
            residual=[]
            for m in comparison["metrics"]:
                residual.append((m["speed_rmse_kmh"]/15)**2)
                residual.append((m["lap_error_s"]/(.02*m["real_lap_time_s"]))**2)
                residual.append(float(np.mean(np.asarray(m["sector_error_s"])**2)))
            prior=.01*np.sum(((np.asarray(x)-np.array([1.75,1.15,3.6]))/np.array([.6,.7,2.]))**2)
            objective=float(np.mean(residual)+prior) if residual else 1e6
            if not valid_result(result):objective+=1000
            cache[key]=(objective,result,p)
            evaluations+=1
            if progress and (evaluations==1 or evaluations%25==0):
                progress(f"  calibration evaluation {evaluations}: objective {objective:.4f}")
        return cache[key]
    runs=[];converged=[]
    for i,start in enumerate(starts):
        if progress:progress(f"  calibration start {i+1}/3: {start.tolist()}")
        opt=minimize(lambda x:evaluate(x)[0],np.clip(start,BOUNDS[:,0],BOUNDS[:,1]),method="Nelder-Mead",bounds=BOUNDS,
            options={"maxiter":maxiter,"maxfev":maxfev,"xatol":1e-3,"fatol":1e-3})
        objective,result,p=evaluate(opt.x)
        record={"start":start.tolist(),"parameters":dict(zip(PARAMETERS,map(float,opt.x))),"objective":objective,
            "solver_success":bool(opt.success),"message":str(opt.message),"iterations":int(opt.nit),"function_evaluations":int(opt.nfev),"physical_valid":valid_result(result)}
        runs.append(record)
        if opt.success and valid_result(result):converged.append((objective,result,p,i))
    if converged:
        selected=min(converged,key=lambda x:x[0])
    else:
        key=min(cache,key=lambda k:cache[k][0]);obj,res,p=cache[key];selected=(obj,res,p,None)
    obj,result,p,index=selected
    calibration=compare_observables(dataset,result,"calibration")
    development=compare_observables(dataset,result,"development")
    gate=accuracy_gate(development,valid_result(result))
    success=index is not None
    report={"method":"calibration_only_three_start_bounded_Nelder_Mead","fitted_parameters":PARAMETERS,
        "fixed_assumptions":base.to_dict(),"fitted_vehicle":p.to_dict(),"multistart":runs,"calibration_evaluations":evaluations,
        "selected_converged_start":index,"solver_success":success,"calibration_metrics":calibration["metrics"],
        "development_metrics":development["metrics"],"development_gate":gate,"ready_for_training":success and gate["passed"],
        "validation_used_for_fitting":False,"test_loaded_for_fitting":False,"physical_check":result["summary"],
        "accuracy_gate_precommitted":True,"calibration_laps":[3,4,5,6],"development_laps":[7,9],
        "near_equivalent_runs":[r for r in runs if r["objective"]<=obj+.05],
        "identification_warning":"Equivalent parameters only; observed speed does not uniquely identify ICE/K split, real SOC or aero/grip.",
        "objective":"equal-weight normalized speed, lap-time and sector squared errors plus small fixed physical prior; calibration laps only"}
    return p,report,result
