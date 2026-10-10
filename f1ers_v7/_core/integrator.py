"""Event-aware implicit-midpoint mechanics shared by replay and autonomy.

Input discontinuities are handled by ``advance_interval``. The scalar Numba
kernel evaluates the continuous curve at all three force-budget stages. A
stopped vehicle terminates at its actual time and distance, never at the cell
boundary. Electrical power is DC power; signed wheel work is mechanical work.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace, asdict
import math
import numpy as np
from f1ers_v7._core.params import VehicleParams, njit


@dataclass
class VehicleParamsV4(VehicleParams):
    equivalent_power_kw: float = 600.0
    replay_brake_gain: float = 0.5
    session_grip_scale: float = 1.0
    replay_power_scale: float = 1.0

    @classmethod
    def from_dict(cls, values):
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})


@dataclass(frozen=True)
class IntegrationState:
    s_m: float
    speed_mps: float
    elapsed_time_s: float = 0.0
    energy_mj: float | None = None
    recharge_mj: float | None = None
    previous_dc_kw: float = 0.0


@dataclass(frozen=True)
class IntegratorConfig:
    dt_max: float = 0.02
    max_ds: float = 1.0
    dt_min: float = 1e-6
    max_subdivisions: int = 20
    v_atol: float = 1e-3
    v_rtol: float = 1e-5
    s_atol: float = 1e-4
    s_rtol: float = 1e-5
    e_atol: float = 1e-7
    e_rtol: float = 1e-6
    root_tol: float = 1e-9
    root_iterations: int = 50
    event_t_tol: float = 1e-8
    event_s_tol: float = 1e-5

    def __post_init__(self):
        numeric = [value for value in asdict(self).values()]
        if not np.isfinite(numeric).all() or any(value <= 0 for value in numeric):
            raise ValueError("Integrator settings must be finite and positive")
        if self.dt_min > self.dt_max or self.root_iterations < 1:
            raise ValueError("Invalid integrator step or root limits")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class WheelCommand:
    power_kw: float = 0.0
    brake_fraction: float = 0.0
    max_tractive_force_n: float = 12000.0
    wheel_force_n: float | None = None
    target_acceleration_mps2: float | None = None
    mode: str = "replay"
    requested_dc_kw: float = 0.0
    deployment_cap_kw: float = 350.0
    max_dc_kw: float = 350.0
    harvest_limit_mj: float = 8.0
    energy_floor_mj: float = 2.0
    energy_capacity_mj: float = 4.0
    ramp_rate_kw_per_s: float = 100.0
    minimum_ice_wheel_power_kw: float | None = None
    alternative_threshold_kph: float = 0.0
    alternative_cap_kw: float = 350.0


@njit(cache=False)
def _poly(x, coefficients, value, derivative=0):
    index = min(max(np.searchsorted(x, value, side="right") - 1, 0), len(x)-2)
    z = value-x[index]
    if derivative == 1:
        return (3*coefficients[0,index]*z+2*coefficients[1,index])*z+coefficients[2,index]
    if derivative == 2:
        return 6*coefficients[0,index]*z+2*coefficients[1,index]
    return ((coefficients[0,index]*z+coefficients[1,index])*z+coefficients[2,index])*z+coefficients[3,index]


@njit(cache=False)
def _geometry_value(s, geo, base_cda, base_cla):
    arc_x, arc_c, source_x, curve_x, curve_y, metadata, zones, fallback = geo
    if metadata[0] > 0:
        arc = (metadata[2]+metadata[3]*s) % metadata[1]
        u = _poly(arc_x, arc_c, arc)
        dx, dy = _poly(source_x, curve_x, u, 1), _poly(source_x, curve_y, u, 1)
        ddx, ddy = _poly(source_x, curve_x, u, 2), _poly(source_x, curve_y, u, 2)
        curvature = metadata[3]*(dx*ddy-dy*ddx)/max((dx*dx+dy*dy)**1.5, 1e-18)
        cda, cla = base_cda, base_cla
        q = s % metadata[1]
        for row in zones:
            start,end = row[0],row[1]
            active = start <= q < end if start <= end else (q >= start or q < end)
            if active:
                cda,cla = base_cda*row[2],base_cla*row[3]
    else:
        curvature = np.interp(s, fallback[:,0], fallback[:,1])
        cda = np.interp(s, fallback[:,0], fallback[:,2])
        cla = np.interp(s, fallback[:,0], fallback[:,3])
    return curvature,cda,cla


def _packed_geometry(geometry, state, dt, params):
    route = getattr(geometry, "route", getattr(geometry, "continuous_route", None))
    if route is None and getattr(geometry,"_curve",None) is not None:
        route = geometry
    if route is not None and getattr(route, "_curve", None) is not None:
        cache = getattr(geometry, "_v4_geometry_cache", None)
        if cache is not None:
            return cache
        zone_rows = []
        for zone in getattr(geometry,"aero_zones",[]):
            zone_rows.append([float(zone["start_m"]),float(zone["end_m"]),
                float(zone.get("drag_factor",zone.get("drag_area_factor",params.straight_drag_factor))),
                float(zone.get("downforce_factor",zone.get("downforce_area_factor",params.straight_downforce_factor)))])
        packed = (np.asarray(route._arc_to_u.x,float), np.asarray(route._arc_to_u.c,float),
                  np.asarray(route._curve.x,float), np.asarray(route._curve.c[:,:,0],float),
                  np.asarray(route._curve.c[:,:,1],float),
                  np.array([1.,route.length_m,route.phase_arc,route.direction],float),
                  np.asarray(zone_rows,float).reshape(-1,4),np.zeros((2,4),float))
        try:
            geometry._v4_geometry_cache = packed
        except (AttributeError,TypeError):
            pass
        return packed
    # Callback fixtures use a dense local interpolation. Actual production
    # ClosedRoute instances always take the analytic Numba path above.
    width = max(1., (state.speed_mps+dt*params.ice_max_tractive_force_n/params.mass_kg+5)*dt)
    positions = state.s_m+np.linspace(0,width,33)
    values = []
    for s in positions:
        value = geometry.evaluate(float(s)) if hasattr(geometry,"evaluate") else geometry(float(s))
        if isinstance(value,(int,float,np.number)):
            value = {"curvature_1pm":float(value)}
        values.append([s,float(value.get("curvature_1pm",0.)),
            float(value.get("drag_area_m2",params.drag_area_m2)),
            float(value.get("downforce_area_m2",params.downforce_area_m2))])
    return (np.array([0.,1.]),np.zeros((4,1)),np.array([0.,1.]),np.zeros((4,1)),
            np.zeros((4,1)),np.zeros(4),np.empty((0,4)),np.asarray(values,float))


@njit(cache=False)
def _public_cap(v, max_dc):
    kph = 3.6*v
    if kph < 340:
        return max(0.,min(max_dc,1800.-5.*kph))
    if kph < 345:
        return max(0.,min(max_dc,6900.-20.*kph))
    return 0.


@njit(cache=False)
def _forces(v1, dt, s0, v0, energy, recharge, previous, geo, p, c):
    mass,mu,rho,rolling,cda0,cla0,eta_d,eta_h,brake_gain,superclip = p
    power,brake,force_cap,signed,target_ax,replay,request,external_cap,max_dc,harvest_limit,floor,capacity,ramp,minimum_ice,alt_threshold,alt_cap = c
    mean = .5*(v0+v1)
    ds = mean*dt
    total0 = total1 = total2 = lateral0 = lateral1 = lateral2 = 0.
    tire = 1e100
    cda_mid = cda0
    for k in range(3):
        stage_v = v0 if k == 0 else (mean if k == 1 else v1)
        stage_s = s0+.5*k*ds
        curvature,cda,cla = _geometry_value(stage_s,geo,cda0,cla0)
        total = mu*(mass*9.81+.5*rho*cla*stage_v**2)
        lateral = mass*stage_v**2*abs(curvature)
        tire = min(tire,math.sqrt(max(0.,total**2-lateral**2)))
        if k == 0:
            total0,lateral0 = total,lateral
        elif k == 1:
            total1,lateral1 = total,lateral
        else:
            total2,lateral2 = total,lateral
        if k == 1:
            cda_mid = cda
    drag = .5*rho*cda_mid*mean*mean+rolling*mass*9.81
    braking = brake > 0 or signed < 0 or (target_ax < 1e90 and mass*target_ax+drag < 0)
    if not replay:
        power = min(power,force_cap*mean/1000.)
    dc = 0.
    if not replay and energy >= 0:
        cap = min(external_cap,_public_cap(v0,max_dc),_public_cap(v1,max_dc),max_dc)
        if min(v0,v1)*3.6 < alt_threshold:
            cap = min(cap,alt_cap)
        exempt = braking or min(v0,v1)*3.6 < 210. or cap < previous-1e-4
        lower = max(0.,previous-ramp*dt) if not exempt else 0.
        if not braking and (request > 0 or lower > 0):
            credit = max(0.,energy-floor)
            reserve_bound = -ramp*dt+math.sqrt((ramp*dt)**2+2000.*ramp*credit)
            dc = min(max(request,lower),cap,reserve_bound)
        elif request < 0:
            room = max(0.,min(capacity-energy,harvest_limit-recharge))*1000./dt
            if braking:
                dc = max(request,-max_dc,-room)
            else:
                available = max(0.,power-minimum_ice)*eta_h
                dc = max(request,-max_dc,-superclip,-available,-room)
    # Brake is an explicit driving demand. The longitudinal axes stay equal;
    # replay gain never increases physical tire capability.
    if signed < 1e90:
        demand = signed
    elif brake > 0:
        demand = -tire*brake*(brake_gain if replay else 1.)
    elif target_ax < 1e90:
        demand = mass*target_ax+drag
    else:
        mech = dc*eta_d if dc >= 0 else dc/eta_h
        demand = min(force_cap,max(0.,power+mech)*1000./max(mean,.5))
    if demand >= 0:
        mech = dc*eta_d if dc >= 0 else dc/eta_h
        available = min(force_cap,max(0.,power+mech)*1000./max(mean,.5))
        wheel = min(tire,demand,available)
    else:
        wheel = max(-tire,demand)
    wheel_kw = wheel*mean/1000.
    if not replay and energy >= 0:
        if wheel < 0:
            dc = max(min(dc,0.),wheel_kw*eta_h)
        elif dc > 0:
            dc = min(dc,wheel_kw/eta_d)
    kmec = dc*eta_d if dc >= 0 else dc/eta_h
    ice = max(0.,wheel_kw-kmec)
    friction = max(0.,kmec-wheel_kw)
    excess = max(0.,math.sqrt(wheel*wheel+lateral0**2)-total0,
                 math.sqrt(wheel*wheel+lateral1**2)-total1,
                 math.sqrt(wheel*wheel+lateral2**2)-total2)
    return wheel,drag,tire,excess,dc,kmec,ice,friction


@njit(cache=False)
def _midpoint(dt,s0,v0,energy,recharge,previous,geo,p,c,tol,iterations):
    mass = p[0]
    low,high = 0.,v0+dt*(max(c[2],p[0]*p[1]*9.81)/mass)+10.
    force0 = _forces(0.,dt,s0,v0,energy,recharge,previous,geo,p,c)
    f0 = -mass*v0-dt*(force0[0]-force0[1])
    if f0 >= 0:
        # Locate a real stopping-time root, not a fictitious zero-speed cell.
        a,b = 0.,dt
        for k in range(iterations):
            stop_dt = .5*(a+b)
            force = _forces(0.,stop_dt,s0,v0,energy,recharge,previous,geo,p,c)
            residual = -mass*v0-stop_dt*(force[0]-force[1])
            if residual >= 0:
                b = stop_dt
            else:
                a = stop_dt
            if b-a <= min(1e-10,tol/max(v0,1.)):
                break
        stop_dt = .5*(a+b)
        force = _forces(0.,stop_dt,s0,v0,energy,recharge,previous,geo,p,c)
        residual = -mass*v0-stop_dt*(force[0]-force[1])
        if abs(residual) > mass*max(tol,1e-8):
            return 0.,stop_dt,force,3
        return 0.,stop_dt,force,1
    fh_force = _forces(high,dt,s0,v0,energy,recharge,previous,geo,p,c)
    fh = mass*(high-v0)-dt*(fh_force[0]-fh_force[1])
    if not math.isfinite(fh) or fh <= 0:
        return v0,dt,fh_force,3
    converged = False
    for k in range(iterations):
        trial = .5*(low+high)
        force = _forces(trial,dt,s0,v0,energy,recharge,previous,geo,p,c)
        residual = mass*(trial-v0)-dt*(force[0]-force[1])
        if not math.isfinite(residual):
            return v0,dt,force,3
        if residual <= 0:
            low = trial
        else:
            high = trial
        if high-low <= tol:
            converged = True
            break
    v1 = .5*(low+high)
    force = _forces(v1,dt,s0,v0,energy,recharge,previous,geo,p,c)
    return v1,dt,force,0 if converged else 3


def _as_command(value):
    if isinstance(value,WheelCommand):
        return value
    allowed = {item.name for item in fields(WheelCommand)}
    return WheelCommand(**{key:value for key,value in value.items() if key in allowed})


def vehicle_step(state, command, continuous_geometry, params, dt_s, config=None):
    """One signed midpoint transition, including an actual partial stop event.

    Returns ``(next_state, info)``. Failed numerical or physical proposals return
    the unchanged accepted state and retain proposed terminal coordinates.
    """
    state = IntegrationState(**state) if isinstance(state,dict) else state
    params = VehicleParamsV4.from_dict(params) if isinstance(params,dict) else params
    command = _as_command(command)
    config = IntegratorConfig(**config) if isinstance(config,dict) else config or IntegratorConfig()
    if command.mode not in ("replay","autonomous"):
        raise ValueError("Driver mode must be replay or autonomous")
    values = [state.s_m,state.speed_mps,state.elapsed_time_s,dt_s,params.mass_kg,params.mu,
        params.session_grip_scale,command.power_kw,command.brake_fraction,command.max_tractive_force_n,
        params.air_density_kgpm3,params.rolling_coefficient,params.drag_area_m2,params.downforce_area_m2,
        params.deploy_efficiency,params.harvest_efficiency,command.requested_dc_kw,command.deployment_cap_kw,
        command.max_dc_kw,command.harvest_limit_mj,command.energy_floor_mj,command.energy_capacity_mj,
        command.ramp_rate_kw_per_s,command.alternative_threshold_kph,command.alternative_cap_kw]
    if not np.isfinite(values).all() or state.speed_mps < 0 or dt_s <= 0 or params.mass_kg <= 0 or params.mu <= 0:
        raise ValueError("Finite nonnegative speed and positive step/mass/grip required")
    if not 0 <= command.brake_fraction <= 1 or command.power_kw < 0 or command.max_tractive_force_n < 0:
        raise ValueError("Invalid wheel command")
    if not .1 <= params.replay_brake_gain <= 1. or params.session_grip_scale <= 0:
        raise ValueError("Replay brake gain must lie in [0.1,1.0]")
    if not 0 < params.deploy_efficiency <= 1 or not 0 < params.harvest_efficiency <= 1 or command.ramp_rate_kw_per_s <= 0:
        raise ValueError("Efficiencies and electrical slew must be positive")
    if command.max_dc_kw < 0 or command.max_dc_kw > 350 or command.deployment_cap_kw < 0 or command.alternative_cap_kw < 0:
        raise ValueError("Electrical requests cannot change the public 350 kW limit")
    if command.energy_floor_mj != 2. or command.energy_capacity_mj != 4. or command.ramp_rate_kw_per_s > 100.:
        raise ValueError("v4 locks the 2–4 MJ window and at most 100 kW/s reduction")
    if command.mode == "replay" and (command.target_acceleration_mps2 is not None or command.requested_dc_kw != 0):
        raise ValueError("Replay cannot receive autonomous acceleration or ERS commands")
    for value in (command.wheel_force_n,command.target_acceleration_mps2):
        if value is not None and not np.isfinite(value):
            raise ValueError("Explicit signed demands must be finite")
    if state.speed_mps == 0:
        return state,dict(status="physical_stop",step_certified=False,guard_reasons=["vehicle_stopped"],dt_s=0.,ds_m=0.)
    energy = -1. if state.energy_mj is None else state.energy_mj
    recharge = 0. if state.recharge_mj is None else state.recharge_mj
    if state.energy_mj is not None and (not np.isfinite([energy,recharge,state.previous_dc_kw]).all() or
            energy < command.energy_floor_mj-1e-8 or energy > command.energy_capacity_mj+1e-8):
        raise ValueError("Electrical state outside permitted usable window")
    p = np.array([params.mass_kg,params.mu*params.session_grip_scale,params.air_density_kgpm3,
        params.rolling_coefficient,params.drag_area_m2,params.downforce_area_m2,
        params.deploy_efficiency,params.harvest_efficiency,params.replay_brake_gain,params.superclip_max_dc_kw])
    effective_power = command.power_kw if command.mode == "replay" else min(command.power_kw,params.ice_power_kw)
    c = np.array([effective_power,command.brake_fraction,min(command.max_tractive_force_n,params.ice_max_tractive_force_n),
        1e100 if command.wheel_force_n is None else command.wheel_force_n,
        1e100 if command.target_acceleration_mps2 is None else command.target_acceleration_mps2,
        float(command.mode == "replay"),command.requested_dc_kw,command.deployment_cap_kw,
        command.max_dc_kw,command.harvest_limit_mj,command.energy_floor_mj,command.energy_capacity_mj,
        command.ramp_rate_kw_per_s,params.minimum_ice_wheel_power_kw if command.minimum_ice_wheel_power_kw is None else command.minimum_ice_wheel_power_kw,
        command.alternative_threshold_kph,command.alternative_cap_kw])
    geo = _packed_geometry(continuous_geometry,state,dt_s,params)
    v1,actual_dt,force,status = _midpoint(dt_s,state.s_m,state.speed_mps,energy,recharge,
        state.previous_dc_kw,geo,p,c,config.root_tol,config.root_iterations)
    wheel,drag,tire,excess,dc,kmec,ice,friction = map(float,force)
    ds = .5*(state.speed_mps+float(v1))*float(actual_dt)
    kinetic = .5*params.mass_kg*(float(v1)**2-state.speed_mps**2)
    work = (wheel-drag)*ds-kinetic
    next_energy = None if state.energy_mj is None else energy-dc*actual_dt/1000.
    next_recharge = None if state.energy_mj is None else recharge+max(0.,-dc)*actual_dt/1000.
    failures = []
    if status == 3:
        failures.append("velocity_root_not_converged")
    if excess > 8.:
        failures.append("combined_tire_force")
    if abs(work) > .01 or not np.isfinite([v1,actual_dt,ds,wheel,drag,work]).all():
        failures.append("mechanical_work")
    reserve = max(0.,dc)**2/(2000.*command.ramp_rate_kw_per_s)
    if next_energy is not None:
        if next_energy < command.energy_floor_mj-1e-8:
            failures.append("energy_floor")
        if next_energy > command.energy_capacity_mj+1e-8:
            failures.append("energy_capacity")
        if next_energy-command.energy_floor_mj < reserve-1e-8:
            failures.append("ramp_reserve")
        if next_recharge > command.harvest_limit_mj+1e-8:
            failures.append("recharge_limit")
        cap = min(command.deployment_cap_kw,_public_cap(state.speed_mps,command.max_dc_kw),_public_cap(float(v1),command.max_dc_kw))
        if min(state.speed_mps,float(v1))*3.6 < command.alternative_threshold_kph:
            cap = min(cap,command.alternative_cap_kw)
        exempt = min(state.speed_mps,float(v1))*3.6 < 210 or wheel < 0 or cap < state.previous_dc_kw-1e-4
        if not exempt and dc < max(0.,state.previous_dc_kw-command.ramp_rate_kw_per_s*actual_dt)-1e-4:
            failures.append("deployment_slew")
        if dc > cap+1e-4 or abs(dc) > command.max_dc_kw+1e-4:
            failures.append("electrical_power")
        if command.mode == "autonomous" and ice > min(params.ice_power_kw,
                params.ice_max_tractive_force_n*.5*(state.speed_mps+float(v1))/1000.)+1e-3:
            failures.append("ice_source_power")
    status_name = "numerical_fail" if status == 3 else ("physical_fail" if failures else ("physical_stop" if status == 1 else "accepted"))
    proposed = IntegrationState(state.s_m+ds,float(v1),state.elapsed_time_s+float(actual_dt),
        None if next_energy is None else float(next_energy),None if next_recharge is None else float(next_recharge),dc)
    info = dict(status=status_name,step_certified=not failures,guard_reasons=failures,
        s_start_m=state.s_m,s_end_m=proposed.s_m,speed_start_mps=state.speed_mps,speed_next_mps=float(v1),
        time_start_s=state.elapsed_time_s,time_end_s=proposed.elapsed_time_s,dt_s=float(actual_dt),ds_m=ds,
        ax_mps2=(float(v1)-state.speed_mps)/max(actual_dt,1e-15),wheel_force_n=wheel,drag_force_n=drag,
        tire_drive_capacity_n=tire,tire_brake_capacity_n=tire,tire_force_excess_n=excess,
        combined_tire_force_excess_n=excess,mechanical_work_residual_mj=work/1e6,
        wheel_work_mj=wheel*ds/1e6,resistance_work_mj=drag*ds/1e6,kinetic_delta_mj=kinetic/1e6,
        wheel_power_kw=wheel*.5*(state.speed_mps+float(v1))/1000.,dc_kw=dc,
        mguk_mechanical_kw=kmec,ice_mechanical_kw=ice,friction_brake_kw=friction,
        energy_start_mj=state.energy_mj,energy_next_mj=next_energy,recharge_start_mj=state.recharge_mj,
        recharge_next_mj=next_recharge,ramp_reserve_mj=reserve,
        ers_validated=command.mode == "autonomous" and state.energy_mj is not None,
        driver_power_kw=command.power_kw,driver_brake_fraction=command.brake_fraction,
        driver_target_acceleration_mps2=command.target_acceleration_mps2,
        proposed_state=asdict(proposed))
    return (state if failures else proposed),info


def _query_command(adapter,state,geometry,dt,ers_request):
    # Events bound dt before querying. Linear throttle is evaluated at the
    # true time midpoint; the held brake switch cannot cross its release.
    time = state.elapsed_time_s+.5*dt
    position = state.s_m+.5*state.speed_mps*dt
    value = adapter.query(time,position) if hasattr(adapter,"query") else adapter(time,position)
    command = _as_command(value)
    if ers_request is not None:
        request = ers_request.query(state,command,geometry,dt) if hasattr(ers_request,"query") else (
            ers_request(state,command,geometry,dt) if callable(ers_request) else ers_request)
        values = {"requested_dc_kw":float(request)} if np.isscalar(request) else dict(request)
        if "request_kw" in values:
            values["requested_dc_kw"] = values.pop("request_kw")
        names = {item.name for item in fields(WheelCommand)}
        command = replace(command,**{key:value for key,value in values.items() if key in names})
    return command


def _next_event(adapter,name,value):
    fn = getattr(adapter,name,None)
    if fn is None:
        return math.inf
    result = fn(value)
    return math.inf if result is None else float(result)


def advance_interval(state, driver_adapter, ers_request, stop_s_m, continuous_geometry,
                     params, config=None, collect_steps=True):
    """Advance to a spatial decision/finish boundary with hard input events.

    Output-grid positions are deliberately not events. Both halves evaluate
    their own actual midpoint input and causal electrical state.
    """
    config = IntegratorConfig(**config) if isinstance(config,dict) else config or IntegratorConfig()
    params = VehicleParamsV4.from_dict(params) if isinstance(params,dict) else params
    state = IntegrationState(**state) if isinstance(state,dict) else state
    if not np.isfinite(stop_s_m) or stop_s_m < state.s_m-config.event_s_tol:
        raise ValueError("Spatial interval must not move backwards")
    records,events = [],[]
    accepted_steps = rejected_steps = 0
    terminal_info = None
    while state.s_m < stop_s_m-config.event_s_tol:
        final_time = getattr(driver_adapter,"final_time_s",math.inf)
        if state.elapsed_time_s >= final_time-config.event_t_tol:
            terminal_info = dict(status="physical_fail",guard_reasons=["driver_trace_exhausted"],step_certified=False)
            break
        if state.speed_mps <= 0:
            terminal_info = dict(status="physical_stop",guard_reasons=["vehicle_stopped"],step_certified=False)
            break
        time_event = min(final_time,_next_event(driver_adapter,"next_time_event",state.elapsed_time_s))
        space_event = min(stop_s_m,_next_event(driver_adapter,"next_space_event",state.s_m),
            _next_event(continuous_geometry,"next_space_event",state.s_m),
            _next_event(ers_request,"next_space_event",state.s_m) if ers_request is not None else math.inf)
        time_room = time_event-state.elapsed_time_s
        if time_room <= config.event_t_tol:
            # An adapter must return the strictly next event; do not invent time.
            terminal_info = dict(status="numerical_fail",guard_reasons=["nonadvancing_time_event"],step_certified=False)
            break
        distance_room = space_event-state.s_m
        if distance_room <= config.event_s_tol:
            terminal_info = dict(status="numerical_fail",guard_reasons=["nonadvancing_space_event"],step_certified=False)
            break
        dt = min(config.dt_max,config.max_ds/max(state.speed_mps,1.),time_room)
        # Close spatial boundaries by a root in dt, using the same step kernel.
        command = _query_command(driver_adapter,state,continuous_geometry,dt,ers_request)
        full,full_info = vehicle_step(state,command,continuous_geometry,params,dt,config)
        if full_info.get("s_end_m",state.s_m) > space_event+config.event_s_tol:
            a,b = 0.,dt
            for _ in range(50):
                mid = .5*(a+b)
                command = _query_command(driver_adapter,state,continuous_geometry,mid,ers_request)
                candidate,proposal = vehicle_step(state,command,continuous_geometry,params,mid,config)
                if proposal.get("s_end_m",state.s_m) > space_event:
                    b = mid
                else:
                    a = mid
                if abs(proposal.get("s_end_m",state.s_m)-space_event) <= config.event_s_tol*.05:
                    a=b=mid
                    break
            dt = .5*(a+b)
        success = False
        for subdivision in range(config.max_subdivisions+1):
            command = _query_command(driver_adapter,state,continuous_geometry,dt,ers_request)
            full,full_info = vehicle_step(state,command,continuous_geometry,params,dt,config)
            if full_info["status"] == "physical_fail":
                # Reducing a step may resolve a prospective force overshoot,
                # but never repairs an already infeasible start state.
                if subdivision == config.max_subdivisions:
                    terminal_info = full_info
                    break
            half_command = _query_command(driver_adapter,state,continuous_geometry,dt/2,ers_request)
            half,first_info = vehicle_step(state,half_command,continuous_geometry,params,dt/2,config)
            second_info = None
            if first_info["status"] == "accepted":
                second_command = _query_command(driver_adapter,half,continuous_geometry,dt/2,ers_request)
                halves,second_info = vehicle_step(half,second_command,continuous_geometry,params,dt/2,config)
            else:
                halves = half
            stop_info = first_info if first_info["status"] == "physical_stop" else (
                second_info if second_info and second_info["status"] == "physical_stop" else None)
            stop_valid = stop_info is not None and stop_info["step_certified"] and full_info["status"] == "physical_stop"
            if stop_valid:
                dt_stop_error = abs(full.elapsed_time_s-halves.elapsed_time_s)
                ds_stop_error = abs(full.s_m-halves.s_m)
                de_stop_error = 0. if full.energy_mj is None else abs(full.energy_mj-halves.energy_mj)
                stop_valid = (dt_stop_error <= config.event_t_tol and
                    ds_stop_error <= config.s_atol+config.s_rtol*max(0.,halves.s_m-state.s_m) and
                    de_stop_error <= config.e_atol+config.e_rtol*(0. if halves.energy_mj is None else abs(halves.energy_mj)))
            if stop_valid:
                step_infos = [first_info] if first_info["status"] == "physical_stop" else [first_info,second_info]
                if collect_steps:
                    records.extend(step_infos)
                state = halves
                accepted_steps += len(step_infos)
                terminal_info = stop_info
                success = True
                break
            valid = full_info["status"] == "accepted" and first_info["status"] == "accepted" and second_info is not None and second_info["status"] == "accepted"
            if valid:
                dv = abs(full.speed_mps-halves.speed_mps)
                ds = abs(full.s_m-halves.s_m)
                de = 0. if full.energy_mj is None else abs(full.energy_mj-halves.energy_mj)
                vt = config.v_atol+config.v_rtol*max(state.speed_mps,halves.speed_mps)
                st = config.s_atol+config.s_rtol*max(0.,halves.s_m-state.s_m)
                et = config.e_atol+config.e_rtol*(0. if halves.energy_mj is None else abs(halves.energy_mj))
                valid = dv <= vt and ds <= st and de <= et
            if valid and halves.s_m <= space_event+config.event_s_tol:
                if collect_steps:
                    records.extend([first_info,second_info])
                state = halves
                accepted_steps += 2
                success = True
                if abs(state.s_m-space_event) <= config.event_s_tol:
                    events.append(dict(kind="space",s_m=state.s_m,time_s=state.elapsed_time_s))
                if abs(state.elapsed_time_s-time_event) <= config.event_t_tol:
                    events.append(dict(kind="time",s_m=state.s_m,time_s=state.elapsed_time_s))
                break
            rejected_steps += 1
            dt *= .5
            if dt < config.dt_min or subdivision == config.max_subdivisions:
                candidates = [full_info,first_info]+([] if second_info is None else [second_info])
                terminal_info = next((item for item in candidates if item["status"] == "physical_fail"),None)
                if terminal_info is None:
                    terminal_info = dict(status="numerical_fail",step_certified=False,
                        guard_reasons=["adaptive_step_limit"],last_proposal=full_info)
                break
        if terminal_info is not None:
            break
        if not success:
            terminal_info = dict(status="numerical_fail",step_certified=False,guard_reasons=["adaptive_step_limit"])
            break
    if terminal_info is None:
        terminal_info = dict(status="accepted",step_certified=True,guard_reasons=[])
    return state,{**terminal_info,"step_records":records,"events":events,"accepted_steps":accepted_steps,
        "rejected_steps":rejected_steps,"terminal_s_m":state.s_m,"terminal_time_s":state.elapsed_time_s,
        "distance_advanced_m":state.s_m-(records[0]["s_start_m"] if records else state.s_m)}
