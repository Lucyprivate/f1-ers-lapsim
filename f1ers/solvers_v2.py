"""Finite control-family baselines scored by the shared causal executor."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def valid_result(result):
    s=result["summary"]
    return bool(s.get("feasible")) and bool(s.get("sustainable")) and bool(s.get("execution_feasible",True))


def _maps(sim):
    neutral=sim.simulate(np.zeros(sim.n_nodes))
    brake=np.asarray(neutral["wheel_power_kw"]) < -1e-6
    fast=(np.asarray(neutral["speed_mps"])>77.0)&~brake
    return neutral,brake,fast


def solve_heuristic_v2(sim):
    neutral,brake,fast=_maps(sim)
    actions=np.where(brake,-1.0,np.where(fast,-.15,.85))
    result=sim.simulate(actions)
    result["solver_status"]={"method":"causal_distance_rule_heuristic","success":valid_result(result),
        "global_optimality_claim":False,"offline_deployment_scaling":False,
        "action_map_source":"zero-K causal reference; fixed during evaluation"}
    return result


def solve_optimal_v2(sim,initial=None,maxiter=40,n_blocks=6):
    """SLSQP finds a feasible candidate; it never relaxes the execution shield."""
    neutral,brake,fast=_maps(sim)
    initial=solve_heuristic_v2(sim) if initial is None else initial
    block=np.minimum(n_blocks-1,np.arange(sim.n_nodes)*n_blocks//sim.n_nodes)
    cache={};candidates=[r for r in (initial,neutral) if valid_result(r)]

    def expand(x):
        action=x[block].copy()
        action[fast]=-x[n_blocks+block[fast]]
        action[brake]=-1.0
        return action

    def evaluate(x):
        key=tuple(np.round(x,10))
        if key not in cache:
            cache[key]=sim.simulate(expand(x))
            if valid_result(cache[key]):candidates.append(cache[key])
        return cache[key]

    def objective(x):
        result=evaluate(x);s=result["summary"]
        if not s.get("execution_feasible",False):return 1e5+s.get("lap_time_s",1e4)
        return s["lap_time_s"]+100*max(0,s.get("periodic_speed_error_mps",0)-.02)

    opt=minimize(objective,np.r_[np.full(n_blocks,.8),np.full(n_blocks,.1)],method="SLSQP",
        bounds=[(0.,1.)]*(2*n_blocks),options={"maxiter":maxiter,"eps":2e-3,"ftol":2e-3})
    evaluate(np.clip(opt.x,0,1))
    if not candidates:raise RuntimeError("No valid causal numerical or zero-K reference")
    best=min(candidates,key=lambda r:r["summary"]["lap_time_s"])
    # Avoid mutating the caller's heuristic/zero reference.
    best=dict(best)
    best["solver_status"]={"method":"causal_SLSQP_distance_blocks","success":bool(opt.success),
        "message":str(opt.message),"iterations":int(opt.nit),"function_evaluations":len(cache),
        "n_control_variables":2*n_blocks,"n_track_nodes":sim.n_nodes,
        "selected_feasible_candidate":True,"global_optimality_claim":False,
        "offline_deployment_scaling":False,"interpretation":"best valid candidate in a limited control family"}
    return best
