"""Limited v5 driving models on the unmodified v4 mechanics and reporting.

No-option baseline execution delegates directly to SimulatorV4 and is exactly
identical.  Modified adapters share its geometry, certified transition,
output sampler, energy ledger, force budget and numerical audit.
"""
from __future__ import annotations
import numpy as np
from f1ers_v4.simulator import SimulatorV4, ContinuousGeometry, ERSRequests, DRIVER_DS_M
from f1ers_v4.vehicle import IntegrationState, IntegratorConfig
from .adapters import VehicleParamsV5, ReplayDriverV5, GeometryDriverV5
from .integration import advance_interval, FROZEN_V4_KERNEL_SHA256

PROTOCOL = 'falsification_adapters_frozen_midpoint_v5'


class SimulatorV5(SimulatorV4):
    protocol = PROTOCOL

    def __init__(self, track, rules, vehicle, config=None):
        if isinstance(vehicle, dict):
            vehicle = VehicleParamsV5.from_dict(vehicle)
        elif not isinstance(vehicle, VehicleParamsV5):
            vehicle = VehicleParamsV5.from_dict(vehicle.to_dict())
        super().__init__(track, rules, vehicle, config)

    def simulate(self,mode="autonomous",driver_trace=None,controller=None,initial_speed_mps=None,
                 output_ds_m=5.,numerics=None,collect_steps=True):
        if self.vehicle.frozen_v4_baseline:
            return super().simulate(mode, driver_trace, controller, initial_speed_mps, output_ds_m, numerics, collect_steps)
        if mode not in ("replay","autonomous"):raise ValueError("Mode must be replay or autonomous")
        cfg = numerics or self.config
        if isinstance(cfg,dict):cfg=IntegratorConfig(**cfg)
        if not np.isfinite(output_ds_m) or output_ds_m<=0:raise ValueError("Output spacing must be positive")
        if mode=="replay":
            if driver_trace is None:raise ValueError("Replay requires raw driver trace")
            if controller is not None:raise ValueError("Replay does not accept ERS control")
            driver=ReplayDriverV5(driver_trace,self.vehicle,self.geometry);ers=None
            start=driver.initial_speed_mps if initial_speed_mps is None else float(initial_speed_mps)
            state=IntegrationState(0.,start,0.)
            envelope=None
        else:
            if driver_trace is not None:raise ValueError("Autonomous cannot see future observed driving input")
            profile=controller.get("driver_profile") if isinstance(controller,dict) else None
            driver=GeometryDriverV5(self.geometry,self.vehicle,profile)
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
        result["summary"].update(protocol=PROTOCOL, frozen_v4_kernel_sha256=FROZEN_V4_KERNEL_SHA256,
                                 brake_mapping=self.vehicle.brake_mapping,
                                 brake_decay_tau_s=self.vehicle.brake_decay_tau_s,
                                 traction_tau_s=self.vehicle.traction_tau_s,
                                 raw_driver_requests_preserved=True)
        result["internal_trace"].update(
            raw_requested_force_n=np.asarray([r.get("raw_requested_force_n",np.nan) for r in records],float),
            requested_force_n=np.asarray([r.get("requested_force_n",np.nan) for r in records],float),
            executed_force_n=np.asarray([r.get("wheel_force_n",np.nan) for r in records],float),
            lateral_utilization=np.asarray([r.get("lateral_utilization",np.nan) for r in records],float),
            demand_clip_reasons=[r.get("demand_clip_reasons",[]) for r in records],
            brake_elapsed_s=np.asarray([np.nan if r.get("brake_elapsed_s") is None else r["brake_elapsed_s"] for r in records],float),
            traction_response_fraction=np.asarray([np.nan if r.get("traction_response_fraction") is None else r["traction_response_fraction"] for r in records],float))
        return result

    def numerical_audit(self, driver_trace=None):
        audit = super().numerical_audit(driver_trace)
        if not self.vehicle.frozen_v4_baseline:
            audit.update(protocol=PROTOCOL, frozen_v4_kernel_sha256=FROZEN_V4_KERNEL_SHA256,
                         raw_request_source='original time channels or driver decision demand, never executed force')
        return audit
