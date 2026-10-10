"""Calibration-only falsification evidence, kept outside formal prediction."""
import numpy as np
from .data import require_role


def audit_inputs(data, simulator):
    require_role(data,'calibration')
    reg=data['registration']
    rows=[]
    for lap in data['laps']:
        trace=lap['driver_trace']
        t,v=np.asarray(trace['time_s']),np.asarray(trace['speed_mps'])
        # Local linear slopes use real sample times. The conservative interval
        # includes half of the observed 1 km/h quantization and fit scatter.
        acceleration,uncertainty=[],[]
        for ti in t:
            mask=abs(t-ti)<=.6
            x=t[mask]-ti;y=v[mask]
            if len(x)<3 or np.ptp(x)<=0:
                acceleration.append(np.nan);uncertainty.append(np.inf);continue
            design=np.c_[np.ones(len(x)),x]
            beta=np.linalg.lstsq(design,y,rcond=None)[0]
            residual=y-design@beta
            se=np.sqrt(np.sum(residual**2)/max(1,len(x)-2)/max(np.sum((x-x.mean())**2),1e-12))
            acceleration.append(beta[1]);uncertainty.append(2*se+(1/3.6)/np.ptp(x))
        a=np.asarray(acceleration);da=np.asarray(uncertainty)
        s=np.interp(t,np.asarray(lap['t_s']),np.asarray(lap['s_m']))
        g=simulator.geometry.evaluate(s)
        p=simulator.vehicle
        drag=.5*p.air_density_kgpm3*np.asarray(g['drag_area_m2'])*v*v+p.rolling_coefficient*p.mass_kg*9.81
        force=p.mass_kg*a+drag
        lateral=p.mass_kg*v*v*abs(np.asarray(g['curvature_1pm']))
        budget=p.mu*(p.mass_kg*9.81+.5*p.air_density_kgpm3*np.asarray(g['downforce_area_m2'])*v*v)
        valid=np.isfinite(a)&np.isfinite(da)&(t>=0)&(t<=lap['lap_time_s'])
        straight=valid&(np.asarray(trace['throttle'])>.98)&(np.asarray(trace['brake'])==0)&(lateral/budget<.15)&(a>0)
        power=force*v/1000;delta=p.mass_kg*da*v/1000
        power_interval=[float(np.median((power-delta)[straight])),float(np.median((power+delta)[straight]))] if straight.any() else None
        rows.append({'lap_number':int(lap['lap_number']),'alignment':lap['alignment_audit'],
            'full_throttle_straight_samples':int(straight.sum()),'equivalent_wheel_power_interval_kw':power_interval,
            '400kw_excluded':bool(power_interval and not power_interval[0]<=400<=power_interval[1]),
            'observed_lateral_infeasible_samples':int(np.sum(lateral>budget+1e-8)),
            'combined_kinematic_infeasible_samples':int(np.sum(valid&(np.hypot(force,lateral)>budget))),
            'uncertainty_identifies_true_ICE_or_ERS':False})
    exclusions=sum(r['400kw_excluded'] for r in rows)
    return {'role':'calibration','laps':rows,'registration':reg,
        'geometry_audit':data['track']['geometry_audit'],
        'diagnostic_only':True,'integrated_state_updated_from_observations':False,
        'power_hypothesis':{'hypothesis':'Fixed 400 kW may omit an equivalent straight-line power contribution',
            'expected':'At least three calibration intervals consistently exclude 400 kW',
            'falsifier':'Uncertainty overlaps 400 kW or signs differ across sectors',
            'conclusion':'Equivalent-power mismatch merits investigation; ICE versus unknown ERS is not identifiable' if exclusions>=3 else 'Not identifiable; retain 400 kW'},
        'alignment_hypothesis':{'hypothesis':'One shared geometry/control-line registration explains corner displacement',
            'expected':'Projection and unscaled speed integral agree within sampling uncertainty',
            'falsifier':'Persistent lap-varying offset or boundary failure after one common transform',
            'conclusion':'Frozen offsets reported; no residual-based or per-lap registration is performed'},
        'braking_hypothesis':{'hypothesis':'Remaining-capacity-scaled request vanishes near the lateral limit',
            'expected':'Independent request remains negative while executed force clips to capacity',
            'falsifier':'Independent mapping leaves the same failure and residual pattern',
            'conclusion':'Requires paired candidate execution; kernel regression alone proves only the mapping'},
        'traction_hypothesis':{'hypothesis':'Instantaneous traction recovery causes positive exit residual',
            'expected':'At least three fixed calibration exit windows have a systematic positive error',
            'falsifier':'Straight underprediction, sign-changing sectors, or geometry uncertainty explains error',
            'conclusion':'No extra response-time parameter is introduced in this preregistered experiment'},
        'FP2_accessed':False}
