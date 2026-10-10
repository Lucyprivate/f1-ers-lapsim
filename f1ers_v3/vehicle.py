"""Shared signed wheel-force, distance-domain vehicle transition.

The force is held over a segment. Drag uses its trapezoidal mean speed, so
wheel work exactly equals kinetic-energy change plus resistance work. Driver
adapters choose demands; this kernel never reads a future trace or resets speed.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from math import sqrt
import numpy as np
from f1ers.model import VehicleParams, njit


@dataclass
class VehicleParamsV3(VehicleParams):
    brake_ratio: float = 1.0
    equivalent_power_kw: float = 600.0
    session_grip_scale: float = 1.0
    replay_power_scale: float = 1.0

    @classmethod
    def from_dict(cls, values):
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in allowed})


@dataclass(frozen=True)
class VehicleState:
    speed_mps: float
    elapsed_time_s: float = 0.0


@dataclass(frozen=True)
class WheelCommand:
    power_kw: float = 0.0
    brake_fraction: float = 0.0
    max_tractive_force_n: float = 12000.0
    # Only the autonomous geometry-driver sets this demand. Replay leaves None.
    target_speed_mps: float | None = None
    wheel_force_n: float | None = None


@dataclass(frozen=True)
class SegmentGeometry:
    ds_m: float
    curvature_1pm: float = 0.0
    drag_area_m2: float | None = None
    downforce_area_m2: float | None = None


@njit(cache=False)
def _signed_step(v0, ds, curvature, cda, cla, mass, mu, rho, rolling,
                 brake_ratio, power_kw, brake_fraction, force_cap, target, signed_force):
    lateral = v0*v0*abs(curvature)
    total = mu*(9.81 + .5*rho*cla*v0*v0/mass)
    tire = mass*sqrt(max(0., total*total-lateral*lateral))
    negative_cap = tire*brake_ratio
    low, high = 0.0, max(v0+25., 160.)
    # A bounded implicit root, no speed clipping after the transition.
    for _ in range(35):
        trial = .5*(low+high)
        mean = .5*(v0+trial)
        drag = .5*rho*cda*mean*mean+rolling*mass*9.81
        if signed_force < 1e90:
            demanded = signed_force
        elif brake_fraction > 0:
            demanded = -negative_cap*brake_fraction
        else:
            demanded = min(force_cap, max(0., power_kw)*1000./max(mean, .5))
        if target >= 0:
            target_mean = .5*(v0+target)
            target_drag = .5*rho*cda*target_mean*target_mean+rolling*mass*9.81
            target_force = mass*(target*target-v0*v0)/(2*ds)+target_drag
            demanded = min(demanded, target_force)
        wheel = max(-negative_cap, min(tire, demanded))
        needed = mass*(trial*trial-v0*v0)/(2*ds)+drag
        if needed <= wheel: low = trial
        else: high = trial
    v1 = .5*(low+high)
    mean = .5*(v0+v1)
    drag = .5*rho*cda*mean*mean+rolling*mass*9.81
    if signed_force < 1e90: demanded = signed_force
    elif brake_fraction > 0: demanded = -negative_cap*brake_fraction
    else: demanded = min(force_cap, max(0., power_kw)*1000./max(mean, .5))
    if target >= 0:
        target_mean=.5*(v0+target)
        target_drag=.5*rho*cda*target_mean*target_mean+rolling*mass*9.81
        demanded=min(demanded,mass*(target*target-v0*v0)/(2*ds)+target_drag)
    wheel=max(-negative_cap,min(tire,demanded))
    dt=ds/max(mean,1e-12)
    work=(wheel-drag)*ds-.5*mass*(v1*v1-v0*v0)
    return v1,dt,wheel,drag,tire,negative_cap,work,max(0.,lateral-total)*mass


def tire_force_capacity(state, geometry, params):
    p, g = params, geometry
    cla=p.downforce_area_m2 if g.downforce_area_m2 is None else g.downforce_area_m2
    total=p.mu*p.session_grip_scale*(9.81+.5*p.air_density_kgpm3*cla*state.speed_mps**2/p.mass_kg)
    lateral=state.speed_mps**2*abs(g.curvature_1pm)
    return p.mass_kg*sqrt(max(0.,total*total-lateral*lateral))


def vehicle_step(state: VehicleState, wheel_command: WheelCommand,
                 geometry: SegmentGeometry, params: VehicleParamsV3):
    """Pure accepted-or-rejected transition returning ``(next_state, info)``.

    Positive wheel force propels; negative force brakes. Brake ratio changes
    negative longitudinal capacity only, preserving the lateral envelope.
    """
    if isinstance(state, dict): state=VehicleState(**state)
    if isinstance(wheel_command, dict): wheel_command=WheelCommand(**wheel_command)
    if isinstance(geometry, dict): geometry=SegmentGeometry(**geometry)
    if isinstance(params, dict): params=VehicleParamsV3.from_dict(params)
    s,c,g,p=state,wheel_command,geometry,params
    values=[s.speed_mps,s.elapsed_time_s,g.ds_m,g.curvature_1pm,p.mass_kg,p.mu,
            p.session_grip_scale,p.brake_ratio,c.power_kw,c.brake_fraction,c.max_tractive_force_n]
    if not np.isfinite(values).all() or s.speed_mps<=0 or g.ds_m<=0 or p.mass_kg<=0 or p.mu<=0 or p.session_grip_scale<=0 or p.brake_ratio<=0:
        raise ValueError("Vehicle state, segment and physical parameters must be finite and positive")
    if not 0<=c.brake_fraction<=1 or c.power_kw<0 or c.max_tractive_force_n<0:
        raise ValueError("Wheel power/capacity must be nonnegative and brake fraction in [0,1]")
    if c.target_speed_mps is not None and (not np.isfinite(c.target_speed_mps) or c.target_speed_mps<0):
        raise ValueError("Target speed must be nonnegative and finite")
    if c.wheel_force_n is not None and not np.isfinite(c.wheel_force_n):
        raise ValueError("Signed force demand must be finite")
    cda=p.drag_area_m2 if g.drag_area_m2 is None else g.drag_area_m2
    cla=p.downforce_area_m2 if g.downforce_area_m2 is None else g.downforce_area_m2
    result=_signed_step(s.speed_mps,g.ds_m,g.curvature_1pm,cda,cla,p.mass_kg,
        p.mu*p.session_grip_scale,p.air_density_kgpm3,p.rolling_coefficient,p.brake_ratio,
        c.power_kw,c.brake_fraction,c.max_tractive_force_n,
        -1. if c.target_speed_mps is None else c.target_speed_mps,
        1e100 if c.wheel_force_n is None else c.wheel_force_n)
    v1,dt,wheel,drag,poscap,negcap,work,lateral_excess=map(float,result)
    failures=[]
    if v1<.5: failures.append("vehicle_stopped_before_segment_end")
    if abs(work)>1e-2: failures.append("mechanical_work")
    # Lateral overshoot is rejected, rather than introducing an unseen brake.
    lateral_end=p.mass_kg*max(0.,v1*v1*abs(g.curvature_1pm)-
        p.mu*p.session_grip_scale*(9.81+.5*p.air_density_kgpm3*cla*v1*v1/p.mass_kg))
    lateral_excess=max(lateral_excess,lateral_end)
    if lateral_excess>8.: failures.append("lateral_tire_force")
    info=dict(speed_next_mps=v1,dt_s=dt,ax_mps2=(v1*v1-s.speed_mps*s.speed_mps)/(2*g.ds_m),
        wheel_force_n=wheel,wheel_power_kw=wheel*.5*(s.speed_mps+v1)/1000.,drag_force_n=drag,
        tire_drive_capacity_n=poscap,tire_brake_capacity_n=negcap,
        tire_force_excess_n=max(0.,wheel-poscap,-wheel-negcap,lateral_excess),
        mechanical_work_residual_mj=work/1e6,step_certified=not failures,guard_reasons=failures,
        lateral_tire_force_excess_n=lateral_excess)
    return (s if failures else VehicleState(v1,s.elapsed_time_s+dt)),info
