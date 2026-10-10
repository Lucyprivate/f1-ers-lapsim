"""Time-domain execution with separately sampled output and fixed driving decisions.

The autonomous driver sees only fixed geometry and the current integrated state.
Observed-input replay sees raw time channels and only their initial speed.  Neither
path resets an integrated speed.  Numerical audits freeze *requested* commands,
never reconstruct requests from tyre-clipped wheel forces.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import numpy as np

from f1ers_v3.simulator import _driver_envelope
from .vehicle import IntegrationState, IntegratorConfig, VehicleParamsV4, advance_interval

PROTOCOL = "event_time_midpoint_v4"
DRIVER_DS_M = 5.0
DRIVER_GRIP_MARGIN = 0.96


def _sample_states(query_s, s, t, v):
    """Evaluate the accepted midpoint step's constant-acceleration trajectory.

    Returns boundary time/speed plus the segment index and elapsed-time fraction,
    so held DC power is integrated in time rather than interpolated in distance.
    """
    query_s=np.asarray(query_s,float);s=np.asarray(s,float);t=np.asarray(t,float);v=np.asarray(v,float)
    if len(s)<2:
        return np.full_like(query_s,t[0]),np.full_like(query_s,v[0]),np.zeros(len(query_s),int),np.zeros_like(query_s)
    ids=np.clip(np.searchsorted(s,query_s,side="right")-1,0,len(s)-2)
    dt=t[ids+1]-t[ids];a=(v[ids+1]-v[ids])/dt
    distance=np.clip(query_s-s[ids],0,s[ids+1]-s[ids])
    endpoint_speed=np.sqrt(np.maximum(0,v[ids]**2+2*a*distance))
    denominator=v[ids]+endpoint_speed
    tau=np.divide(2*distance,denominator,out=np.zeros_like(distance),where=denominator>1e-15)
    tau=np.clip(tau,0,dt)
    return t[ids]+tau,v[ids]+a*tau,ids,tau/dt


def _inside(s, a, b):
    return (a <= s < b) if a < b else (s >= a or s < b)


class ContinuousGeometry:
    """Analytic frozen curve plus half-open, explicitly transformed aero zones."""
    def __init__(self, track, rules, params):
        self.route = track.get("continuous_route")
        self.continuous_route = self.route
        if self.route is None:
            raise ValueError("v4 requires one continuous closed route")
        self.length_m = float(self.route.length_m)
        self.params = params
        route_zones = track.get("approximate_aero_zones_m", [])
        if "aero_straight_zones_m" in rules:
            official = float(rules.get("fia_centreline_length_m", self.route.source_length))
            route_zones = []
            for z in rules["aero_straight_zones_m"]:
                a, b = (z["start_m"], z["end_m"]) if isinstance(z, dict) else z[:2]
                route_zones.append({"start_m": float(self.route.legacy_to_s(a*self.route.source_length/official)),
                                    "end_m": float(self.route.legacy_to_s(b*self.route.source_length/official))})
        self.aero_zones = [{"start_m": float(z["start_m"]), "end_m": float(z["end_m"]),
                            "drag_factor": params.straight_drag_factor,
                            "downforce_factor": params.straight_downforce_factor} for z in route_zones]
        rule_zones = rules.get("alternative_power_zones_route_m", [])
        exceptions = rules.get("power_reduction_exceptions_route", [])
        self.events_m = np.unique([0., self.length_m, *[v for z in self.aero_zones for v in (z["start_m"],z["end_m"])],
                                  *[v for z in rule_zones for v in ((z["start_m"],z["end_m"]) if isinstance(z,dict) else z[:2])],
                                  *[v for z in exceptions for v in (z["start_m"],z["end_m"]) ]])

    def evaluate(self, s):
        values = self.route.evaluate(s)
        ss = np.mod(np.asarray(s, float), self.length_m)
        aero = np.zeros_like(ss, dtype=bool)
        for z in self.aero_zones:
            a,b = z["start_m"],z["end_m"]
            aero |= ((ss >= a)&(ss < b)) if a < b else ((ss >= a)|(ss < b))
        return {**values, "drag_area_m2": self.params.drag_area_m2*np.where(aero,self.params.straight_drag_factor,1.),
                "downforce_area_m2": self.params.downforce_area_m2*np.where(aero,self.params.straight_downforce_factor,1.),
                "aero_mode": aero}

    def next_space_event(self, s):
        i = np.searchsorted(self.events_m, s+1e-5, side="right")
        return float(self.events_m[i]) if i < len(self.events_m) else np.inf


class ReplayDriver:
    """Raw time adapter: throttle linear, brake previous-value hold."""
    def __init__(self, trace, params, geometry):
        self.params,self.geometry = params,geometry
        self.trace = {k: np.asarray(trace[k],float).reshape(-1).copy() for k in ("time_s","speed_mps","throttle","brake")}
        t = self.trace["time_s"]; n = len(t)
        if n < 2 or any(a.shape != (n,) or not np.isfinite(a).all() for a in self.trace.values()) or t[0] > 0 or not np.all(np.diff(t)>0):
            raise ValueError("Replay requires finite aligned raw times bracketing zero")
        if np.max(self.trace["throttle"]) > 1+1e-6:
            self.trace["throttle"] /= 100.
        if np.any((self.trace["throttle"]<0)|(self.trace["throttle"]>1)):
            raise ValueError("Throttle must be in [0,1] or [0,100]")
        if not np.all(np.isclose(self.trace["brake"],0)|np.isclose(self.trace["brake"],1)):
            raise ValueError("Brake input must be a discrete switch")
        self.initial_speed_mps = float(np.interp(0.,t,self.trace["speed_mps"]))
        self.final_time_s = float(t[-1])

    def query(self, t, s):
        times = self.trace["time_s"]
        i = min(len(times)-1,max(0,int(np.searchsorted(times,t,side="right")-1)))
        return {"power_kw": float(np.interp(t,times,self.trace["throttle"]))*self.params.equivalent_power_kw*self.params.replay_power_scale,
                "brake_fraction": float(self.trace["brake"][i]), "mode": "replay",
                "max_tractive_force_n": self.params.ice_max_tractive_force_n}

    def next_time_event(self, t):
        ts = self.trace["time_s"]
        i = np.searchsorted(ts,t+1e-8,side="right")
        return float(ts[i]) if i < len(ts) else np.inf

    def next_space_event(self,s):
        return self.geometry.next_space_event(s)


class GeometryDriver:
    """Fixed geometry controller, with explicit acceleration at 5 m decisions.

    A preregistered 4% grip margin covers curvature between decision positions.
    It is a driver safety setting, not a fitted parameter or a numerical guard.
    """
    def __init__(self, geometry, params, frozen_profile=None):
        self.geometry,self.params = geometry,params
        self.length_m = geometry.length_m
        self.decision_s = np.linspace(0,self.length_m,int(np.ceil(self.length_m/DRIVER_DS_M))+1)
        self.rows = []
        self.command = {"power_kw": params.ice_power_kw, "brake_fraction":0., "mode":"autonomous",
                        "max_tractive_force_n":params.ice_max_tractive_force_n, "target_acceleration_mps2":0.}
        self.frozen = None
        if frozen_profile is not None:
            self.frozen = {k:np.asarray(v,float).copy() for k,v in frozen_profile.items() if k in
                           ("s_m","target_acceleration_mps2","power_kw","brake_fraction","requested_dc_kw")}
            ss = self.frozen["s_m"]
            if len(ss)<1 or ss[0]!=0 or not np.all(np.diff(ss)>0) or any(v.shape!=ss.shape or not np.isfinite(v).all() for v in self.frozen.values()):
                raise ValueError("Frozen raw requests require aligned finite increasing positions starting at zero")
            if ss.shape!=self.decision_s[:-1].shape or not np.allclose(ss,self.decision_s[:-1],rtol=0,atol=1e-5):
                raise ValueError("Frozen requests must cover every original 5 m driving decision")
            self.decision_s = np.r_[ss,self.length_m]
            self.nominal_speed = None
            self.envelope_report = {"method":"frozen_raw_requests","converged":True}
            return
        ds = self.length_m/int(np.ceil(self.length_m))
        s = np.arange(int(np.ceil(self.length_m)))*ds
        g = geometry.evaluate(s)
        mu = params.mu*params.session_grip_scale*DRIVER_GRIP_MARGIN
        denom = params.mass_kg*np.abs(g["curvature_1pm"])-mu*.5*params.air_density_kgpm3*g["downforce_area_m2"]
        caps2 = np.divide(mu*params.mass_kg*9.81,denom,out=np.full(len(s),params.max_speed_mps**2),where=denom>1e-9)
        caps = np.minimum(params.max_speed_mps,np.sqrt(np.maximum(caps2,.25)))
        speed,residual,iterations,ok = _driver_envelope(caps,g["curvature_1pm"],g["drag_area_m2"],g["downforce_area_m2"],ds,
            params.mass_kg,mu,params.air_density_kgpm3,params.rolling_coefficient,1.,params.ice_power_kw,params.ice_max_tractive_force_n)
        self.nominal_speed = np.interp(self.decision_s,np.r_[s,self.length_m],np.r_[speed,speed[0]])
        self.envelope_report = {"method":"continuous_geometry_1m_forward_backward","converged":bool(ok),
                                "iterations":int(iterations),"residual_mps":float(residual),"fixed_grip_margin":DRIVER_GRIP_MARGIN}

    def decide(self,state):
        i = min(len(self.decision_s)-2,max(0,int(np.searchsorted(self.decision_s,state.s_m+1e-5,side="right")-1)))
        if self.frozen is not None:
            for k in ("target_acceleration_mps2","power_kw","brake_fraction"):
                if k in self.frozen:self.command[k] = float(self.frozen[k][i])
        else:
            ds = self.decision_s[i+1]-state.s_m
            self.command["target_acceleration_mps2"] = float((self.nominal_speed[i+1]**2-state.speed_mps**2)/(2*max(ds,1e-8)))
        self.rows.append({"s_m":float(self.decision_s[i]), **self.command})

    def query(self,t,s):
        return dict(self.command)

    def next_time_event(self,t):
        return np.inf

    def next_space_event(self,s):
        i = np.searchsorted(self.decision_s,s+1e-5,side="right")
        decision = float(self.decision_s[i]) if i<len(self.decision_s) else np.inf
        return min(decision,self.geometry.next_space_event(s))


class ERSRequests:
    """Causal requests; the shared integrator applies the physical shield."""
    def __init__(self,rules,geometry,controller=None):
        self.rules,self.geometry,self.controller = rules,geometry,controller
        self.max_dc = float(rules.get("ers_max_power_kw",350.))
        self.request_kw = 0.; self.rows = []

    def cap(self,s,v):
        vkph = v*3.6
        cap = np.clip(1800-5*vkph if vkph<340 else (6900-20*vkph if vkph<345 else 0),0,self.max_dc)
        zones = self.rules.get("alternative_power_zones_route_m",[])
        for z in zones:
            a,b = (z["start_m"],z["end_m"]) if isinstance(z,dict) else z[:2]
            threshold = z.get("threshold_kph",310) if isinstance(z,dict) else 310
            power = z.get("power_kw",250) if isinstance(z,dict) else 250
            if _inside(s%self.geometry.length_m,a,b) and vkph<threshold:cap=min(cap,power)
        return float(cap)

    def decide(self,state,i,driver):
        ctl = self.controller
        if isinstance(ctl,dict) and "raw_requested_dc_kw" in ctl:
            self.request_kw = float(ctl["raw_requested_dc_kw"][i])
        elif callable(ctl):
            obs = np.array([state.s_m/self.geometry.length_m,state.speed_mps/100.,state.energy_mj/4.,state.recharge_mj/8.,state.previous_dc_kw/self.max_dc])
            action = float(ctl(obs))
            self.request_kw = np.clip(action,-1,1)*(self.cap(state.s_m,state.speed_mps) if action>=0 else self.max_dc)
        elif ctl is not None:
            action = float(np.asarray(ctl)[i])
            self.request_kw = np.clip(action,-1,1)*(self.cap(state.s_m,state.speed_mps) if action>=0 else self.max_dc)
        else:self.request_kw=0.
        self.rows.append(self.request_kw)

    def query(self,state,command,geometry,dt_s):
        coordinate=(state.s_m+.5*state.speed_mps*dt_s)%self.geometry.length_m
        alternative_threshold=0.;alternative_cap=self.max_dc
        for z in self.rules.get("alternative_power_zones_route_m",[]):
            a,b=(z["start_m"],z["end_m"]) if isinstance(z,dict) else z[:2]
            if _inside(coordinate,a,b):
                alternative_threshold=max(alternative_threshold,float(z.get("threshold_kph",310)) if isinstance(z,dict) else 310.)
                alternative_cap=min(alternative_cap,float(z.get("power_kw",250)) if isinstance(z,dict) else 250.)
        return {"requested_dc_kw":self.request_kw,"deployment_cap_kw":self.cap(coordinate,state.speed_mps),
                "alternative_threshold_kph":alternative_threshold,"alternative_cap_kw":alternative_cap,
                "max_dc_kw":self.max_dc,"harvest_limit_mj":float(self.rules.get("harvest_max_mj_per_lap",8.)),
                "energy_floor_mj":2.,"energy_capacity_mj":float(self.rules.get("soc_window_mj",4.)),
                "ramp_rate_kw_per_s":float(self.rules.get("power_reduction_rate_kw_per_s",100.))}


class SimulatorV4:
    protocol = PROTOCOL
    def __init__(self,track,rules,vehicle,config=None):
        self.track,self.rules=track,rules
        self.vehicle = VehicleParamsV4.from_dict(vehicle) if isinstance(vehicle,dict) else vehicle
        if not isinstance(self.vehicle,VehicleParamsV4):self.vehicle=VehicleParamsV4.from_dict(self.vehicle.to_dict())
        self.params = self.vehicle
        self.config = IntegratorConfig(**config) if isinstance(config,dict) else config or IntegratorConfig()
        self.geometry = ContinuousGeometry(track,rules,self.vehicle)
        self.length_m = self.geometry.length_m
        self.s_m = np.asarray(track["s_m"],float);self.n_nodes=len(self.s_m);self.ds_m=self.length_m/self.n_nodes

    def simulate(self,mode="autonomous",driver_trace=None,controller=None,initial_speed_mps=None,
                 output_ds_m=5.,numerics=None,collect_steps=True):
        if mode not in ("replay","autonomous"):raise ValueError("Mode must be replay or autonomous")
        cfg = numerics or self.config
        if isinstance(cfg,dict):cfg=IntegratorConfig(**cfg)
        if not np.isfinite(output_ds_m) or output_ds_m<=0:raise ValueError("Output spacing must be positive")
        if mode=="replay":
            if driver_trace is None:raise ValueError("Replay requires raw driver trace")
            if controller is not None:raise ValueError("Replay does not accept ERS control")
            driver=ReplayDriver(driver_trace,self.vehicle,self.geometry);ers=None
            start=driver.initial_speed_mps if initial_speed_mps is None else float(initial_speed_mps)
            state=IntegrationState(0.,start,0.)
            envelope=None
        else:
            if driver_trace is not None:raise ValueError("Autonomous cannot see future observed driving input")
            profile=controller.get("driver_profile") if isinstance(controller,dict) else None
            driver=GeometryDriver(self.geometry,self.vehicle,profile)
            actualcontroller=controller if not isinstance(controller,dict) else ({"raw_requested_dc_kw":controller["raw_requested_dc_kw"]} if "raw_requested_dc_kw" in controller else controller.get("actions"))
            values=actualcontroller.get("raw_requested_dc_kw") if isinstance(actualcontroller,dict) else actualcontroller
            if values is not None and not callable(values):
                values=np.asarray(values,float)
                if values.shape!=(len(driver.decision_s)-1,) or not np.isfinite(values).all():
                    raise ValueError("One finite ERS request per fixed driver decision is required")
            ers=ERSRequests(self.rules,self.geometry,actualcontroller)
            start=float(driver.nominal_speed[0]) if initial_speed_mps is None and driver.nominal_speed is not None else float(initial_speed_mps)
            state=IntegrationState(0.,start,0.,2.,0.,0.)
            envelope=driver.envelope_report
        if not np.isfinite(start) or start<=0:raise ValueError("Initial speed must be finite and positive")
        initial=state;records=[];events=[];failure=None;status="accepted"
        stops=driver.decision_s[1:] if mode=="autonomous" else np.array([self.length_m])
        if envelope is not None and not envelope["converged"]:
            failure={"guard_reasons":["driver_envelope_not_converged"],"status":"numerical_fail"};status="numerical_fail"
        else:
            for i,stop in enumerate(stops):
                if mode=="autonomous":driver.decide(state);ers.decide(state,i,driver)
                state,info=advance_interval(state,driver,ers,float(stop),self.geometry,self.vehicle,cfg,collect_steps=True)
                records.extend(info.get("step_records",[]));events.extend(info.get("events",[]))
                status=info.get("status","numerical_fail")
                if status!="accepted":failure=info;break
                if mode=="replay" and state.elapsed_time_s>driver.trace["time_s"][-1]+cfg.event_t_tol:
                    failure={"status":"physical_fail","guard_reasons":["driver_trace_exhausted"]};status="physical_fail";break
        result=self._result(mode,initial,state,records,events,failure,status,float(output_ds_m),cfg)
        result["summary"].update(envelope_report=envelope,driver_decision_spacing_m=DRIVER_DS_M,
                                  raw_requests_fixed=isinstance(controller,dict) and controller.get("driver_profile") is not None)
        if mode=="autonomous":
            result["driver_profile"]={k:np.asarray([r[k] for r in driver.rows],float) for k in
                ("s_m","target_acceleration_mps2","power_kw","brake_fraction")}
            result["driver_profile"]["requested_dc_kw"]=np.asarray(ers.rows,float)
        return result

    def _result(self,mode,initial,terminal,records,events,failure,status,output_ds,cfg):
        # Accepted substeps have actual states, including any real partial terminal.
        def col(key,default=0.):return np.asarray([r.get(key,default) for r in records],float)
        ns=int(np.ceil(self.length_m/output_ds));ds=self.length_m/ns
        grid=np.arange(ns+1,dtype=float)*ds
        ss=np.r_[initial.s_m,col("s_end_m")];tt=np.r_[initial.elapsed_time_s,col("time_end_s")]
        vv=np.r_[initial.speed_mps,col("speed_next_mps")]
        if len(records) and abs(ss[-1]-terminal.s_m)>cfg.event_s_tol:
            ss=np.r_[ss,terminal.s_m];tt=np.r_[tt,terminal.elapsed_time_s];vv=np.r_[vv,terminal.speed_mps]
        complete=terminal.s_m>=self.length_m-cfg.event_s_tol
        count=ns if complete else int(np.searchsorted(grid,terminal.s_m+cfg.event_s_tol,side="right")-1)
        count=max(0,min(ns,count));boundaries=grid[:count+1]
        times,speeds,sample_ids,time_fractions=_sample_states(boundaries,ss,tt,vv)
        sample_s=boundaries[:-1];geom=self.geometry.evaluate(sample_s)
        raw_ds=col("ds_m");raw_dt=col("dt_s")
        force=col("wheel_force_n");power=col("wheel_power_kw")
        def mean_over_distance(values):
            cumulative=np.r_[0.,np.cumsum(values*raw_ds)]
            return np.diff(np.interp(boundaries,np.r_[initial.s_m,col("s_end_m")],cumulative))/ds if count else np.array([])
        def mean_over_time(values):
            cumulative=np.r_[0.,np.cumsum(values*raw_dt)]
            return np.diff(np.interp(times,np.r_[initial.elapsed_time_s,col("time_end_s")],cumulative))/np.diff(times) if count else np.array([])
        def per_output_cell(values,method):
            if not count:return np.array([])
            sums=np.zeros(ns)
            if method=="max":
                first=np.clip(np.searchsorted(grid,col("s_start_m")+1e-12,side="right")-1,0,ns-1)
                last=np.clip(np.searchsorted(grid,col("s_end_m")-1e-12,side="right")-1,0,ns-1)
                for offset in range(int(np.max(last-first,initial=0))+1):
                    active=first+offset<=last
                    np.maximum.at(sums,(first+offset)[active],np.abs(values[active]))
            else:
                cells=np.clip(np.searchsorted(grid,col("s_start_m")+.5*raw_ds,side="right")-1,0,ns-1)
                np.add.at(sums,cells,values)
            return sums[:count]
        periodic=abs(terminal.speed_mps-initial.speed_mps)
        physical=status=="accepted" and failure is None
        feasible=complete and physical and (mode=="replay" or periodic<=.02)
        reasons=[] if failure is None else [{"status":status,"guard_reasons":failure.get("guard_reasons",[status]),
                                                "s_m":terminal.s_m,"time_s":terminal.elapsed_time_s}]
        if mode=="autonomous" and complete and physical and periodic>.02:reasons.append({"guard_reasons":["periodic_speed_boundary"],"residual_mps":periodic})
        residual=col("mechanical_work_residual_mj");tire=col("tire_force_excess_n")
        result={"s_m":sample_s,"speed_mps":speeds[:-1],"speed_next_mps":speeds[1:],"time_s":times,"dt_s":np.diff(times),
                "x_m":geom["x_m"],"y_m":geom["y_m"],"ay_mps2":speeds[:-1]**2*geom["curvature_1pm"],"aero_mode":geom["aero_mode"],
                "ax_mps2":np.diff(speeds**2)/(2*ds),"wheel_force_n":mean_over_distance(force),
                "wheel_power_kw":mean_over_distance(force)*ds/np.maximum(np.diff(times),1e-30)/1000.,
                "mechanical_work_residual_mj":per_output_cell(residual,"max"),
                "mechanical_work_residual_max_abs_mj":per_output_cell(residual,"max"),
                "mechanical_work_residual_sum_mj":per_output_cell(residual,"sum"),
                "tire_force_excess_n":per_output_cell(tire,"max"),
                "mode":mode,"ers_validated":mode=="autonomous","length_m":self.length_m,"ds_m":ds,"terminal_status":status,
                "distance_completed_m":terminal.s_m,"terminal_s_m":terminal.s_m,"terminal_speed_mps":terminal.speed_mps,
                "terminal_time_s":terminal.elapsed_time_s,"event_diagnostics":events,"guard_diagnostics":reasons,
                "internal_trace":{"s_m":ss,"time_s":tt,"speed_mps":vv,"wheel_force_n":force,"wheel_power_kw":power,
                    "mechanical_work_residual_mj":residual,"tire_force_excess_n":tire},
                "summary":{"protocol":PROTOCOL,"mode":mode,"completed_lap":complete,"execution_feasible":physical,"feasible":feasible,
                    "status":status,"lap_time_s":terminal.elapsed_time_s,"speed_start_mps":initial.speed_mps,"speed_end_mps":terminal.speed_mps,
                    "periodic_speed_residual_mps":periodic,"periodic_speed_tolerance_mps":.02,"ers_validated":mode=="autonomous",
                    "observed_initial_speed_only":mode=="replay","intermediate_speed_resets":0,"future_speed_caps_used":False,
                    "equivalent_power_and_braking":mode=="replay","whole_lap_scaling_used":False,"shared_vehicle_step":True,
                    "distance_completed_m":terminal.s_m,"length_m":self.length_m,"ds_m":ds,"guard_failures":reasons,
                    "mechanical_work_max_segment_residual_mj":float(np.max(np.abs(residual))) if len(residual) else 0.,
                    "mechanical_work_total_residual_mj":float(np.sum(residual)),
                    "output_residual_aggregation":"maximum absolute internal-step residual per output cell (MJ)",
                    "output_residual_sum_assignment":"internal-step midpoint assigns each residual once; total sum preserved",
                    "output_sampling":"exact accepted-step constant-acceleration time/position relation",
                    "max_joint_tire_force_excess_n":float(tire.max()) if len(tire) else 0.,"internal_steps":len(records),
                    "integrator_config":asdict(cfg),"input_interpolation":{"throttle":"linear","brake":"previous_value_hold"},
                    "sustainable":feasible if mode=="autonomous" else None}}
        result["failure_proposal"]={k:(asdict(v) if is_dataclass(v) else v) for k,v in failure.items()
                                    if k not in ("step_records","events")} if failure is not None else None
        if mode=="replay":
            for key in ("energy_mj","recharge_mj","soc_fraction","mguk_dc_kw","mguk_mechanical_kw"):result[key]=None
            result.update(equivalent_drive_kw=np.maximum(0,result["wheel_power_kw"]),equivalent_brake_force_n=np.maximum(0,-result["wheel_force_n"]))
        else:
            energy=np.r_[initial.energy_mj,col("energy_next_mj",initial.energy_mj)]
            recharge=np.r_[initial.recharge_mj,col("recharge_next_mj",initial.recharge_mj)]
            def sample_energy(values):
                if len(values)<2:return np.full(len(boundaries),values[0])
                return values[sample_ids]+time_fractions*(values[sample_ids+1]-values[sample_ids])
            result.update(energy_mj=sample_energy(energy),recharge_mj=sample_energy(recharge),
                          mguk_dc_kw=mean_over_time(col("dc_kw")),mguk_mechanical_kw=mean_over_time(col("mguk_mechanical_kw")))
            result["soc_fraction"]=result["energy_mj"]/float(self.rules.get("soc_window_mj",4))
            result["summary"].update(energy_end_mj=terminal.energy_mj,energy_min_mj=float(energy.min()),energy_max_mj=float(energy.max()),
                                      recharge_end_mj=terminal.recharge_mj,energy_start_mj=initial.energy_mj,energy_floor_mj=2.)
        return result

    def numerical_audit(self,driver_trace=None):
        """Freeze raw autonomous decisions; output sampling never changes execution."""
        modes=["autonomous"]+(["replay"] if driver_trace is not None else [])
        out={"protocol":PROTOCOL,"passed":True,"physical_passed":True,"modes":{},"failed_trajectories":[],
             "thresholds":{"lap_time_difference_s":.10,"speed_difference_rmse_kmh":.5}}
        for mode in modes:
            kwargs={"driver_trace":driver_trace} if mode=="replay" else {}
            baseline=self.simulate(mode=mode,**kwargs)
            if mode=="autonomous":
                if not baseline["summary"]["completed_lap"] or not baseline["summary"]["execution_feasible"]:
                    out["modes"][mode]={"passed":False,"reason":"Incomplete autonomous raw request profile cannot be extended", "summaries":[baseline["summary"]]}
                    out["failed_trajectories"].append({"mode":mode,"case":"raw_request_reference","result":baseline})
                    out["passed"]=False;out["physical_passed"]=False
                    continue
                profile=baseline["driver_profile"]
                kwargs={"controller":{"driver_profile":profile,"raw_requested_dc_kw":profile["requested_dc_kw"]},
                        "initial_speed_mps":baseline["summary"]["speed_start_mps"]}
            else:kwargs["initial_speed_mps"]=baseline["summary"]["speed_start_mps"]
            runs=[self.simulate(mode=mode,numerics=replace(self.config,dt_max=dt,max_ds=ds),**kwargs) for dt,ds in ((.02,1.),(.01,.5),(.005,.25))]
            comparisons=[]
            for a,b in zip(runs,runs[1:]):
                valid=all(r["summary"]["completed_lap"] and r["summary"]["execution_feasible"] for r in (a,b))
                time_difference=abs(a["terminal_time_s"]-b["terminal_time_s"]) if valid else None
                error=float(np.sqrt(np.mean((a["speed_mps"]-b["speed_mps"])**2))*3.6) if valid else None
                comparisons.append({"valid":valid,"lap_time_difference_s":time_difference,"speed_difference_rmse_kmh":error,
                                    "passed":bool(valid and time_difference<=.10 and error<=.5)})
            outputs=[self.simulate(mode=mode,output_ds_m=ds,**kwargs) for ds in (10.,5.,2.5)]
            resampling=[]
            for a,b in zip(outputs,outputs[1:]):
                valid=all(r["summary"]["completed_lap"] and r["summary"]["execution_feasible"] for r in (a,b))
                common=np.linspace(0,self.length_m,int(np.ceil(self.length_m/.5))+1)
                av=np.interp(common,np.r_[a["s_m"],a["terminal_s_m"]],np.r_[a["speed_mps"],a["terminal_speed_mps"]]) if valid else None
                bv=np.interp(common,np.r_[b["s_m"],b["terminal_s_m"]],np.r_[b["speed_mps"],b["terminal_speed_mps"]]) if valid else None
                difference=abs(a["terminal_time_s"]-b["terminal_time_s"]) if valid else None
                error=float(np.sqrt(np.mean((av-bv)**2))*3.6) if valid else None
                resampling.append({"valid":valid,"lap_time_difference_s":difference,"speed_difference_rmse_kmh":error,
                                   "passed":bool(valid and difference<=.10 and error<=.5)})
            passed=all(r["passed"] for r in comparisons+resampling)
            physical=all(r["summary"]["completed_lap"] and r["summary"]["execution_feasible"] for r in runs+outputs)
            out["physical_passed"] &= physical
            for kind,cases in (("integration",runs),("output_resampling",outputs)):
                for index,r in enumerate(cases):
                    if not r["summary"]["completed_lap"] or not r["summary"]["execution_feasible"]:
                        out["failed_trajectories"].append({"mode":mode,"case":kind,"index":index,"result":r})
            out["modes"][mode]={"passed":passed,"integration":comparisons,"output_resampling":resampling,
                                  "summaries":[r["summary"] for r in runs],"raw_requests_frozen":True,
                                  "raw_request_sha256": hashlib.sha256(b"".join(np.asarray(v).tobytes() for v in kwargs.get("controller",{}).get("driver_profile",{}).values())).hexdigest() if mode=="autonomous" else None}
            out["passed"] &= passed
        return out
