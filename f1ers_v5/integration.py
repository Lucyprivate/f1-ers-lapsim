"""v5 orchestration around the byte-frozen v4 midpoint mechanics.

The orchestration is copied from v4 only to support query_state(state, ...).
Every transition, geometry packing, velocity root, stop location, electrical
shield, and error tolerance is supplied by the unmodified v4 implementation.
Extra diagnostics have no effect on certified forces or accepted state.
"""
from __future__ import annotations
from dataclasses import fields, replace
import hashlib
import math
from pathlib import Path
import numpy as np
from f1ers_v4.vehicle import (IntegrationState, IntegratorConfig, WheelCommand,
    vehicle_step as _frozen_vehicle_step, _next_event, _geometry_value, _packed_geometry)
from .adapters import VehicleParamsV5, WheelCommandV5

FROZEN_V4_KERNEL_SHA256 = hashlib.sha256(Path(__import__('f1ers_v4.vehicle', fromlist=['__file__']).__file__).read_bytes()).hexdigest()


def _query_command(adapter, state, geometry, dt, ers_request):
    time = state.elapsed_time_s+.5*dt
    position = state.s_m+.5*state.speed_mps*dt
    if hasattr(adapter, 'query_state'):
        value = adapter.query_state(state, time, position, dt)
    else:
        value = adapter.query(time, position) if hasattr(adapter, 'query') else adapter(time, position)
    command = WheelCommandV5.from_value(value)
    if ers_request is not None:
        request = ers_request.query(state, command, geometry, dt) if hasattr(ers_request, 'query') else (
            ers_request(state, command, geometry, dt) if callable(ers_request) else ers_request)
        values = {'requested_dc_kw': float(request)} if np.isscalar(request) else dict(request)
        if 'request_kw' in values:
            values['requested_dc_kw'] = values.pop('request_kw')
        names = {item.name for item in fields(WheelCommand)}
        command = replace(command, **{key:value for key,value in values.items() if key in names})
    return command


def vehicle_step(state, command, geometry, params, dt_s, config=None):
    """Unchanged v4 transition, with separately labelled request diagnostics."""
    state = IntegrationState(**state) if isinstance(state, dict) else state
    params = VehicleParamsV5.from_dict(params) if isinstance(params, dict) else params
    command = WheelCommandV5.from_value(command)
    next_state, info = _frozen_vehicle_step(state, command, geometry, params, dt_s, config)
    if 'wheel_force_n' not in info:
        return next_state, info
    mean = .5*(state.speed_mps+info['speed_next_mps'])
    if command.wheel_force_n is not None:
        # Explicit signed demand is the adapter's physical request, including
        # traction establishment.  Its pre-response metadata remains useful.
        requested = command.wheel_force_n if command.requested_force_n is None else command.requested_force_n
        raw = requested if command.raw_requested_force_n is None else command.raw_requested_force_n
    else:
        # Unfiltered target-acceleration commands are executed using midpoint
        # resistance in v4.  Adapter metadata evaluated at the accepted start
        # state must not turn a harmless drag difference into a clipping claim.
        if command.brake_fraction > 0:
            requested = -info['tire_brake_capacity_n']*command.brake_fraction*(params.replay_brake_gain if command.mode=='replay' else 1.)
        elif command.target_acceleration_mps2 is not None:
            requested = params.mass_kg*command.target_acceleration_mps2+info['drag_force_n']
        else:
            force_cap = min(command.max_tractive_force_n, params.ice_max_tractive_force_n)
            power = command.power_kw if command.mode=='replay' else min(command.power_kw,params.ice_power_kw,force_cap*mean/1000.)
            requested = min(force_cap,max(0.,power+info['mguk_mechanical_kw'])*1000./max(mean,.5))
        raw = requested
    reasons = []
    actual = info['wheel_force_n']
    tire = info['tire_brake_capacity_n']
    if abs(actual-requested)>1e-5:
        if abs(requested)>tire+1e-5:
            reasons.append('remaining_joint_tire_budget')
        if requested>0 and actual+1e-5<min(requested, tire):
            reasons.append('power_or_tractive_force')
    if command.traction_response_fraction is not None and requested>0 and requested<raw-1e-5:
        reasons.append('causal_traction_establishment')
    geo = _packed_geometry(geometry, state, info['dt_s'], params)
    stage_utilization = []
    for position, speed in ((state.s_m,state.speed_mps),
                            (.5*(state.s_m+info['s_end_m']),mean),
                            (info['s_end_m'],info['speed_next_mps'])):
        curvature, _, downforce = _geometry_value(position, geo, params.drag_area_m2, params.downforce_area_m2)
        total = params.mu*params.session_grip_scale*(params.mass_kg*9.81+.5*params.air_density_kgpm3*downforce*speed**2)
        stage_utilization.append(params.mass_kg*speed**2*abs(curvature)/max(total,1e-30))
    info.update(raw_requested_force_n=float(raw), requested_force_n=float(requested),
                executed_force_n=actual, demand_clip_reasons=reasons,
                lateral_utilization=float(max(stage_utilization)),
                traction_response_fraction=command.traction_response_fraction,
                brake_elapsed_s=command.brake_elapsed_s,
                diagnostic_force_request_only=True, frozen_v4_kernel_sha256=FROZEN_V4_KERNEL_SHA256)
    return next_state, info


def advance_interval(state, driver_adapter, ers_request, stop_s_m, continuous_geometry,
                     params, config=None, collect_steps=True):
    """Advance to a spatial decision/finish boundary with hard input events.

    Output-grid positions are deliberately not events. Both halves evaluate
    their own actual midpoint input and causal electrical state.
    """
    config = IntegratorConfig(**config) if isinstance(config,dict) else config or IntegratorConfig()
    params = VehicleParamsV5.from_dict(params) if isinstance(params,dict) else params
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
