"""Separated observed-input replay and autonomous causal ERS execution."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from f1ers.causal import CausalSimulator, CausalEnv, CausalState, CausalExecutionConfig
from f1ers.model import njit
from .vehicle import VehicleParamsV3, VehicleState, WheelCommand, SegmentGeometry, vehicle_step

PROTOCOL = "signed_vehicle_causal_v3"


@njit(cache=False)
def _driver_envelope(caps,curvature,cda,cla,ds,mass,mu,rho,rolling,ratio,ice_kw,ice_force):
    speed=caps.copy();n=len(speed);residual=1e100;converged=False
    for iteration in range(40):
        old=speed.copy()
        for i in range(n-1,-1,-1):
            j=(i+1)%n;lo=.5;hi=speed[i]
            mean=.5*(speed[i]+speed[j])
            total=mu*(9.81+.5*rho*cla[i]*speed[i]*speed[i]/mass)
            lateral=speed[i]*speed[i]*abs(curvature[i])
            tire=mass*np.sqrt(max(0.,total*total-lateral*lateral))*ratio
            drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
            wheel=mass*(speed[j]*speed[j]-speed[i]*speed[i])/(2*ds)+drag
            if wheel>=-tire:continue
            for k in range(28):
                trial=.5*(lo+hi);mean=.5*(trial+speed[j])
                total=mu*(9.81+.5*rho*cla[i]*trial*trial/mass)
                lateral=trial*trial*abs(curvature[i])
                tire=mass*np.sqrt(max(0.,total*total-lateral*lateral))*ratio
                drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
                wheel=mass*(speed[j]*speed[j]-trial*trial)/(2*ds)+drag
                if wheel>=-tire:lo=trial
                else:hi=trial
            speed[i]=min(speed[i],lo)
        for i in range(n):
            j=(i+1)%n;lo=.5;hi=speed[j]
            total=mu*(9.81+.5*rho*cla[i]*speed[i]*speed[i]/mass)
            lateral=speed[i]*speed[i]*abs(curvature[i])
            tire=mass*np.sqrt(max(0.,total*total-lateral*lateral))
            mean=.5*(speed[i]+speed[j])
            drive=min(tire,ice_force,ice_kw*1000/max(mean,.5))
            drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
            wheel=mass*(speed[j]*speed[j]-speed[i]*speed[i])/(2*ds)+drag
            if wheel<=drive:continue
            for k in range(28):
                trial=.5*(lo+hi);mean=.5*(speed[i]+trial)
                drive=min(tire,ice_force,ice_kw*1000/max(mean,.5))
                drag=.5*rho*cda[i]*mean*mean+rolling*mass*9.81
                wheel=mass*(trial*trial-speed[i]*speed[i])/(2*ds)+drag
                if wheel<=drive:lo=trial
                else:hi=trial
            speed[j]=min(speed[j],lo)
        residual=np.max(np.abs(old-speed))
        if residual<1e-7:
            converged=True;break
    return speed,residual,iteration+1,converged


class SimulatorV3(CausalSimulator):
    """Both modes call vehicle_step; only autonomous mode invokes ERS shielding.

    Results use N segment-start speed/s arrays, N segment dt/power arrays and
    N+1 boundary time/energy arrays, matching the retained legacy plotting API.
    Replay has None for all electrical fields and never claims ERS validation.
    """
    protocol=PROTOCOL

    def __init__(self,track,rules,vehicle,config=None):
        vehicle=VehicleParamsV3.from_dict(vehicle) if isinstance(vehicle,dict) else vehicle
        if not isinstance(vehicle,VehicleParamsV3):vehicle=VehicleParamsV3.from_dict(vehicle.to_dict())
        # A mesh change must transform every numeric rule coordinate using the
        # frozen route, rather than inheriting a differently sized FIA array.
        if track.get('continuous_route') is not None:
            route=track['continuous_route'];track=dict(track)
            official=float(rules.get('fia_centreline_length_m',rules.get('official_track_length_m',route.source_length)))
            track['fia_s_m']=route.s_to_legacy(np.asarray(track['s_m'],float))*official/route.source_length
        self.fixed_driver_profile=None
        super().__init__(track,rules,vehicle,config)

    def _safe_envelope(self,terminal_speed):
        p,g=self.vehicle,self.geometry;mu=p.mu*p.session_grip_scale
        effective=replace(p,mu=mu,session_grip_scale=1.)
        g.vehicle=effective
        denominator=p.mass_kg*np.abs(self.curvature)-mu*.5*p.air_density_kgpm3*g.downforce_area
        caps2=np.divide(mu*p.mass_kg*9.81,denominator,
            out=np.full(self.n_nodes,p.max_speed_mps**2),where=denominator>1e-9)
        self.speed_cap_mps=np.minimum(p.max_speed_mps,np.sqrt(np.maximum(caps2,.25)))
        g.speed_cap_mps=self.speed_cap_mps
        caps=self.speed_cap_mps.copy()
        if terminal_speed is not None:caps[0]=min(caps[0],terminal_speed)
        def envelope(dc=None,braking=None,ice_kw=p.ice_power_kw,ice_force=p.ice_max_tractive_force_n):
            speed,residual,iterations,converged=_driver_envelope(caps,self.curvature,g.drag_area,g.downforce_area,self.ds_m,
                p.mass_kg,mu,p.air_density_kgpm3,p.rolling_coefficient,p.brake_ratio,
                ice_kw,ice_force)
            report={'method':'signed_force_forward_backward','converged':bool(converged),
                'iterations':int(iterations),'maximum_iterations':40,'residual_mps':float(residual),
                'tolerance_mps':1e-7,'role':'braking_safety' if ice_kw>1e6 else 'noK_initial_guess'}
            if not hasattr(self,'envelope_reports'):self.envelope_reports=[]
            self.envelope_reports.append(report)
            if not converged:
                raise RuntimeError('Geometry driver envelope did not converge: '+str(report))
            return speed
        g._speeds=envelope
        # Braking/corner safety envelope is independent of propulsion. The
        # separate zero-K initial guess uses the actual ICE acceleration limit.
        safe=envelope(ice_kw=1e7,ice_force=1e9)
        return np.r_[safe,safe[0]]

    def set_fixed_driver_profile(self,profile):
        """Freeze distance-domain driver commands for the mesh audit."""
        if profile is None:self.fixed_driver_profile=None;return
        data={key:np.asarray(profile[key],float) for key in ('s_m','drive_fraction','brake_fraction')}
        if (len(data['s_m'])<2 or any(a.shape!=data['s_m'].shape or not np.isfinite(a).all() for a in data.values())
                or not np.all(np.diff(data['s_m'])>0)):
            raise ValueError("Fixed driver profile must have finite aligned increasing distance")
        if any(np.any((data[key]<0)|(data[key]>1)) for key in ('drive_fraction','brake_fraction')):
            raise ValueError("Driver profile fractions must be in [0,1]")
        self.fixed_driver_profile=data

    def _physical(self,state,dc_kw):
        i,p,g=state.node_index,self.vehicle,self.geometry
        segment=SegmentGeometry(self.ds_m,self.curvature[i],g.drag_area[i],g.downforce_area[i])
        kmec=dc_kw*(p.deploy_efficiency if dc_kw>=0 else 1./p.harvest_efficiency)
        if self.fixed_driver_profile is None:
            command=WheelCommand(max(0.,p.ice_power_kw+kmec),0.,p.ice_max_tractive_force_n,
                target_speed_mps=float(min(self.safe_speed_mps[i+1],self.speed_cap_mps[i])))
        else:
            prof=self.fixed_driver_profile
            drive=float(np.interp(self.s_m[i],prof['s_m'],prof['drive_fraction']))
            brake=float(np.interp(self.s_m[i],prof['s_m'],prof['brake_fraction']))
            command=WheelCommand(max(0.,p.ice_power_kw*drive+kmec) if brake==0 else 0.,brake,
                p.ice_max_tractive_force_n)
        _,info=vehicle_step(VehicleState(state.speed_mps,state.elapsed_time_s),command,segment,p)
        mean=.5*(state.speed_mps+info['speed_next_mps']);wheel=info['wheel_power_kw']
        ice=max(0.,wheel-kmec);friction=max(0.,kmec-wheel)
        ice_cap=g._ice_available_kw(mean)
        info.update(mguk_mechanical_kw=kmec,ice_mechanical_kw=ice,friction_brake_kw=friction,
            deployment_cap_kw=min(self._cap(i,state.speed_mps),self._cap(i,info['speed_next_mps'])),
            ice_power_excess_kw=max(0.,ice-ice_cap),ice_available_kw=ice_cap,
            driver_drive_fraction=min(1.,max(0.,wheel-kmec)/max(ice_cap,1e-12)),
            driver_brake_fraction=min(1.,max(0.,-info['wheel_force_n'])/max(info['tire_brake_capacity_n'],1e-12)))
        return info

    def _certificate(self,state,dc_kw,physical):
        certificate=super()._certificate(state,dc_kw,physical)
        if self.fixed_driver_profile is not None:
            # A mesh audit freezes actual driving demands. Recomputing the
            # backward-looking safety envelope at each mesh would change its
            # driver and reject tiny future-envelope overshoots even while the
            # signed wheel transition remains physically admissible. The pure
            # kernel still certifies actual start/end lateral and wheel forces;
            # all energy/source/power/slew constraints are retained here.
            physical['future_braking_envelope_excess_mps']=certificate['residuals']['next_speed_cap_mps']
            certificate['residuals']['next_speed_cap_mps']=0.
            certificate['failures']=[reason for reason in certificate['failures'] if reason!='next_speed_cap_mps']
        for reason in physical.get('guard_reasons',[]):
            if reason not in certificate['failures']:certificate['failures'].append(reason)
        certificate['certified']=not certificate['failures']
        return certificate

    def simulate(self,mode='autonomous',driver_trace=None,controller=None,initial_speed_mps=None,
                 actions=None,policy=None,sustainable=True):
        # Legacy numeric caller and teacher wrappers remain usable within v3.
        if not isinstance(mode,str):
            if actions is not None:raise ValueError("Actions supplied twice")
            actions=mode;mode='autonomous'
        if mode=='replay':
            if actions is not None or policy is not None or controller is not None:
                raise ValueError("Observed-input replay does not accept an ERS controller")
            return self._replay(driver_trace,initial_speed_mps)
        if mode!='autonomous':raise ValueError("Mode must be replay or autonomous")
        if driver_trace is not None:raise ValueError("Autonomous execution cannot receive observed future driver trace")
        if controller is not None:
            if actions is not None or policy is not None:raise ValueError("Controller supplied twice")
            if isinstance(controller,dict):
                self.set_fixed_driver_profile(controller.get('driver_profile'))
                controller=controller.get('actions',np.zeros(self.n_nodes))
            if callable(controller):policy=controller
            else:actions=controller
        if initial_speed_mps is None:
            result=super().simulate(actions=actions,policy=policy,sustainable=sustainable)
        else:
            # The fixed-initial-condition mesh audit may use a common external
            # speed. It is explicitly reported; the periodic gate still applies.
            env=self.create_env();env.reset()
            start=float(initial_speed_mps)
            if not np.isfinite(start) or start<=0:raise ValueError("Initial speed must be finite and positive")
            env.state=CausalState(0,start,self.config.energy_start_mj)
            env.states=[env.state]
            act=np.zeros(self.n_nodes) if actions is None else np.asarray(actions,float)
            if policy is None and (act.shape!=(self.n_nodes,) or not np.isfinite(act).all()):
                raise ValueError("One finite action per segment is required")
            while not env.done:env.step(policy(self.observation(env.state)) if policy else act[env.i])
            result=env.result();result['summary']['speed_start_mps']=start
            result['summary']['initial_speed_override']=True
        history=result.get('guard_diagnostics',[])
        accepted=[row for row in history if row.get('step_certified')]
        result['driver_profile']={'s_m':self.s_m[:len(accepted)].copy(),
            'drive_fraction':np.array([row['driver_drive_fraction'] for row in accepted]),
            'brake_fraction':np.array([row['driver_brake_fraction'] for row in accepted])}
        result.update(mode='autonomous',ers_validated=True)
        result.update(length_m=self.length_m,ds_m=self.ds_m,distance_completed_m=len(accepted)*self.ds_m,
            terminal_s_m=len(accepted)*self.ds_m,terminal_speed_mps=result['summary']['speed_end_mps'],
            terminal_time_s=result['summary']['lap_time_s'])
        result['summary'].update(protocol=PROTOCOL,mode='autonomous',ers_validated=True,
            shared_vehicle_step=True,driver_profile_fixed=self.fixed_driver_profile is not None,
            future_braking_envelope_gate=self.fixed_driver_profile is None,
            envelope_reports=self.envelope_reports,distance_completed_m=len(accepted)*self.ds_m,
            length_m=self.length_m,ds_m=self.ds_m)
        return result

    def _replay(self,driver_trace,initial_speed):
        if driver_trace is None:raise ValueError("Replay requires raw driver trace")
        trace={key:np.asarray(driver_trace[key],float).reshape(-1) for key in ('time_s','speed_mps','throttle','brake')}
        t=trace['time_s'];n=len(t)
        if n<2 or any(a.shape!=(n,) or not np.isfinite(a).all() for a in trace.values()) or not np.all(np.diff(t)>0) or t[0]>0:
            raise ValueError("Driver trace requires finite aligned increasing times bracketing zero")
        throttle=trace['throttle']
        if np.max(throttle)>1.+1e-6:throttle=throttle/100.
        if np.any((throttle<0)|(throttle>1)) or np.any((trace['brake']<0)|(trace['brake']>1)):
            raise ValueError("Throttle and brake switch must be in permitted ranges")
        if not np.all(np.isclose(trace['brake'],0)|np.isclose(trace['brake'],1)):
            raise ValueError("Brake trace must contain discrete switch values")
        start=float(np.interp(0.,t,trace['speed_mps'])) if initial_speed is None else float(initial_speed)
        state=VehicleState(start);states=[state];rows=[];failure=None;p=self.vehicle;g=self.geometry
        for i in range(self.n_nodes):
            if state.elapsed_time_s>t[-1]+1e-9:
                failure={'guard_reasons':['driver_trace_exhausted'],'node_index':i};break
            throttle_i=float(np.interp(state.elapsed_time_s,t,throttle))
            k=max(0,int(np.searchsorted(t,state.elapsed_time_s,side='right'))-1)
            brake_i=float(trace['brake'][k])
            command=WheelCommand(p.equivalent_power_kw*p.replay_power_scale*throttle_i,brake_i,p.ice_max_tractive_force_n)
            next_state,info=vehicle_step(state,command,SegmentGeometry(self.ds_m,self.curvature[i],
                g.drag_area[i],g.downforce_area[i]),p)
            info.update(throttle=throttle_i,brake=brake_i,node_index=i,observed_input_time_s=state.elapsed_time_s)
            info.update(equivalent_drive_kw=max(0.,info['wheel_power_kw']),
                equivalent_brake_force_n=max(0.,-info['wheel_force_n']))
            if not info['step_certified']:failure=info;break
            rows.append(info);state=next_state;states.append(state)
        count=len(rows);speed=np.array([s.speed_mps for s in states[:-1]])
        result={key:np.array([row[key] for row in rows]) for key in ('dt_s','speed_next_mps','ax_mps2',
            'wheel_force_n','wheel_power_kw','mechanical_work_residual_mj','tire_force_excess_n','throttle','brake','observed_input_time_s',
            'equivalent_drive_kw','equivalent_brake_force_n')}
        result.update(s_m=self.s_m[:count].copy(),speed_mps=speed,time_s=np.array([s.elapsed_time_s for s in states]),
            x_m=np.asarray(self.track.get('x_m',np.zeros(self.n_nodes)))[:count],
            y_m=np.asarray(self.track.get('y_m',np.zeros(self.n_nodes)))[:count],
            ay_mps2=speed**2*self.curvature[:count],aero_mode=self.aero_mode[:count],
            energy_mj=None,recharge_mj=None,soc_fraction=None,mguk_dc_kw=None,mguk_mechanical_kw=None,
            guard_diagnostics=rows+([failure] if failure else []),mode='replay',ers_validated=False,
            length_m=self.length_m,ds_m=self.ds_m,distance_completed_m=count*self.ds_m,
            terminal_s_m=count*self.ds_m,terminal_speed_mps=state.speed_mps,terminal_time_s=state.elapsed_time_s)
        residual=max((abs(row['mechanical_work_residual_mj']) for row in rows),default=0.)
        result['summary']=dict(protocol=PROTOCOL,mode='replay',ers_validated=False,
            completed_lap=count==self.n_nodes,execution_feasible=failure is None,
            feasible=count==self.n_nodes and failure is None,sustainable=None,
            lap_time_s=state.elapsed_time_s,speed_start_mps=start,speed_end_mps=state.speed_mps,
            mechanical_work_max_segment_residual_mj=residual,guard_failures=[failure] if failure else [],
            observed_initial_speed_only=True,intermediate_speed_resets=0,future_speed_caps_used=False,
            distance_completed_m=count*self.ds_m,length_m=self.length_m,ds_m=self.ds_m,
            equivalent_power_and_braking=True,whole_lap_scaling_used=False,shared_vehicle_step=True,
            input_interpolation={'throttle':'linear','brake':'previous_value_hold'})
        return result
