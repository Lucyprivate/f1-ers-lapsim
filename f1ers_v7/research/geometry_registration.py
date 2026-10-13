"""Calibration-only coordinate audit and boundary-constrained common geometry.

The 20/40/80 metre values are second-derivative regularization lengths. They
are not Gaussian smoothing standard deviations. Held-out telemetry cannot
provide a training fold's speed integration support or coordinate transform.
No model speed residual enters this module and no per-lap origin is fitted.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, lsq_linear
from scipy.sparse import coo_matrix, diags, vstack

from .._core.geometry import ClosedRoute, build_track, polyline_length
from .._core.input_channels import epoch, raw_arrays, _similarity, _project
from .._core.input_io import DataUnavailable, read_json
from .._core.provenance import digest
from .physics import segment_boundary_audit

CALIBRATION_LAPS = (3, 4, 5, 6)
REGULARIZATION_LENGTHS_M = (20., 40., 80.)
FIT_GRID_MAX_M = 2.
BOUNDARY_MARGIN_M = .05
MAX_BOUNDARY_EXTRAPOLATION_S = .5


@dataclass(frozen=True)
class CalibrationRawView:
    """Four isolated raw windows. The immutable role is checked at every fit."""
    laps: tuple
    source_hashes: dict
    session: object = None
    driver: object = None
    role: str = "calibration"

    def select(self, numbers):
        numbers = tuple(int(x) for x in numbers)
        if (self.role != "calibration" or tuple(int(x['lap_number']) for x in self.laps) != CALIBRATION_LAPS
                or any(x.get('role')!='calibration' for x in self.laps)):
            raise DataUnavailable("An exact calibration-only laps 3-6 raw view is required")
        allowed = [CALIBRATION_LAPS] + [tuple(n for n in CALIBRATION_LAPS if n != held) for held in CALIBRATION_LAPS]
        if numbers not in allowed:
            raise DataUnavailable("Only the registered full fit or one of four leave-one-out training folds is permitted")
        return [x for x in self.laps if int(x['lap_number']) in numbers]


def load_calibration_raw(data_root):
    """Materialize only in-lap 3-6 samples, including isolated channel repair.

    Historical files contain later laps, but a later value never repairs a
    missing early channel or supplies a fold's integration boundary.
    """
    root = Path(data_root)
    document = read_json(root/'data/processed/dataset.json')
    names = ('laps.json', 'car_data_63_laps3to9.json', 'location_63_laps3to9.json')
    raw = root/'data/raw/openf1'
    metadata, car, location = [read_json(raw/name) for name in names]
    chosen = {int(x['lap_number']): x for x in metadata
              if x.get('driver_number') == 63 and x.get('session_key') == 11315}
    laps = []
    for number in CALIBRATION_LAPS:
        if number not in chosen:
            raise DataUnavailable('Missing fixed calibration lap '+str(number))
        row = deepcopy(chosen[number]); start = epoch(row['date_start'])
        end = start+float(row['lap_duration'])
        # Date filtering occurs before raw_arrays repairs invalid channels.
        cars = [x for x in car if start <= epoch(x['date']) <= end]
        locations = [x for x in location if start <= epoch(x['date']) <= end]
        arrays = raw_arrays(cars, locations)
        arrays['car_t'] = arrays['car_t']-start
        arrays['loc_t'] = arrays['loc_t']-start
        row.update(role='calibration', lap_time_s=float(row['lap_duration']),
                   sector_times_s=[float(row[f'duration_sector_{i}']) for i in (1,2,3)],
                   arrays=arrays, original_epoch_start_s=start)
        laps.append(row)
    hashes = {('data/raw/openf1/'+name): digest(raw/name) for name in names}
    for name in ('data/processed/dataset.json', 'data/v3/local/registration.json',
                 'data/raw/tum/Spielberg_minimum_curvature.csv', 'data/raw/tum/Spielberg_centerline.csv'):
        hashes[name] = digest(root/name)
    return CalibrationRawView(tuple(laps), hashes, document.get('session'), document.get('driver'))


def _supported_speed_integral(lap, query):
    """Exact linear-speed integral, with explicitly limited within-lap support."""
    a = lap['arrays']; t = np.asarray(a['car_t'], float); v = np.asarray(a['speed'], float)
    duration = float(lap['lap_duration']); q = np.asarray(query, float)
    if (len(t) < 2 or not np.all(np.diff(t) > 0) or t[0] < 0 or t[-1] > duration
            or t[0] > MAX_BOUNDARY_EXTRAPOLATION_S
            or duration-t[-1] > MAX_BOUNDARY_EXTRAPOLATION_S
            or np.any(q < -1e-8) or np.any(q > duration+1e-8)):
        raise DataUnavailable('Insufficient isolated training-lap speed support')
    # Boundary values extrapolate from two in-lap original samples only. This
    # is diagnostic registration support, never an extension of replay input.
    first = v[0]+(0.-t[0])*(v[1]-v[0])/(t[1]-t[0])
    last = v[-1]+(duration-t[-1])*(v[-1]-v[-2])/(t[-1]-t[-2])
    if first < 0 or last < 0:
        raise DataUnavailable('Boundary speed extrapolation becomes negative')
    tt = np.r_[0., t[t>0], duration] if t[-1] < duration else np.r_[0.,t[t>0]]
    vv = np.r_[first, v[t>0], last] if t[-1] < duration else np.r_[first,v[t>0]]
    distance = np.r_[0.,np.cumsum(.5*(vv[:-1]+vv[1:])*np.diff(tt))]
    ids = np.clip(np.searchsorted(tt,q,side='right')-1,0,len(tt)-2)
    dt = np.clip(q,0.,duration)-tt[ids]
    return distance[ids]+vv[ids]*dt+.5*(vv[ids+1]-vv[ids])/(tt[ids+1]-tt[ids])*dt**2


def _speed_support_record(lap):
    t = np.asarray(lap['arrays']['car_t'],float)
    # Verify all endpoints, including the integration origin.
    _supported_speed_integral(lap,[0.,float(lap['lap_duration'])])
    return {'lap_number':int(lap['lap_number']), 'source_lap':int(lap['lap_number']),
            'original_samples':len(t), 'first_original_time_s':float(t[0]),
            'last_original_time_s':float(t[-1]),
            'start_extrapolation_s':float(t[0]),
            'end_extrapolation_s':float(lap['lap_duration']-t[-1]),
            'maximum_allowed_boundary_extrapolation_s':MAX_BOUNDARY_EXTRAPOLATION_S,
            'method':'linear continuation of two original samples from this same lap only',
            'foreign_lap_samples':False, 'allowed_for_formal_replay_extension':False}


def fit_registration_view(view, fit_laps, reference_points, old_registration):
    """Fit one shared proper similarity, phase and clock using a guarded fold."""
    laps = view.select(fit_laps)
    reference = np.asarray(reference_points,float); source = polyline_length(reference)
    length = float(source[-1]); direction = int(old_registration['direction'])
    phase0 = float(old_registration['original_route_control_line_s_m'])
    samples = []; support = []
    for lap in laps:
        support.append(_speed_support_record(lap))
        a = lap['arrays']; t = np.asarray(a['loc_t'],float)
        inside = (t >= 1.) & (t <= float(lap['lap_duration'])-1.)
        if inside.sum() < 100:
            raise DataUnavailable('Insufficient original coordinate samples in a training lap')
        samples.append((lap,t[inside],np.asarray(a['loc_xy'],float)[inside]))
    observed = np.concatenate([x[2] for x in samples])
    total_residual_evaluations=0
    def evaluate(theta):
        nonlocal total_residual_evaluations
        total_residual_evaluations+=1
        phase, offset = theta
        q = np.concatenate([_supported_speed_integral(lap,t-offset) for lap,t,_ in samples])
        u = np.mod(phase+direction*q,length)
        template = np.column_stack([np.interp(u,source,np.r_[reference[:,j],reference[0,j]]) for j in (0,1)])
        return _similarity(template, observed)
    optimizer_residual_evaluations=0
    def objective(theta):
        nonlocal optimizer_residual_evaluations
        if optimizer_residual_evaluations>=120:
            raise DataUnavailable('Registration optimizer residual evaluation budget exhausted: 120/120; no converged fit accepted')
        optimizer_residual_evaluations+=1
        return evaluate(theta)[0].ravel()
    fit = least_squares(objective,[phase0,0.],jac='3-point',
        bounds=([phase0-100.,-1.],[phase0+100.,1.]),max_nfev=120,
        xtol=1e-7,ftol=1e-7,gtol=1e-7)
    diagnostic_start=total_residual_evaluations
    residual, scale, rotation, shift = evaluate(fit.x)
    rmse = float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
    if not fit.success or rmse > 25. or not 8. < scale < 12.:
        raise DataUnavailable(f'Training-only registration failed: converged={fit.success} RMSE={rmse:.5f}m scale={scale:.5f}')
    perturb = np.array([.1,.001]);lower=np.array([phase0-100.,-1.]);upper=np.array([phase0+100.,1.])
    columns=[]
    for i in range(2):
        plus=fit.x.copy();minus=fit.x.copy()
        plus[i]=min(upper[i],plus[i]+perturb[i]);minus[i]=max(lower[i],minus[i]-perturb[i])
        columns.append((evaluate(plus)[0].ravel()-evaluate(minus)[0].ravel())/(plus[i]-minus[i]))
    jac = np.column_stack(columns)
    normal = jac.T@jac; rank = int(np.linalg.matrix_rank(normal))
    covariance = np.linalg.pinv(normal)*float(np.sum(residual**2)/max(1,residual.size-2))
    std = np.sqrt(np.maximum(0.,np.diag(covariance)))
    sensitivity = []
    for i, label in enumerate(('phase_m','clock_s')):
        delta = 1. if i == 0 else .05
        for sign in (-1,1):
            point=fit.x.copy();point[i]=np.clip(point[i]+sign*delta,lower[i],upper[i])
            value=evaluate(point)[0]
            sensitivity.append({'parameter':label,'delta':float(point[i]-fit.x[i]),'requested_delta':sign*delta,
                'position_rmse_m':float(np.sqrt(np.mean(np.sum(value**2,axis=1))))})
    condition=float(np.linalg.cond(normal))
    return {'original_route_control_line_s_m':float(fit.x[0]%length),
        'location_clock_offset_s':float(fit.x[1]), 'direction':direction,
        'raw_location_units_per_m':float(scale), 'rotation':rotation.tolist(),
        'translation_raw_units':shift.tolist(), 'rmse_m':rmse,
        'fit_laps':[int(x['lap_number']) for x in laps], 'fit_role':'calibration',
        'scope':'full_calibration' if len(laps)==4 else 'registered_leave_one_out_training_fold',
        'converged':bool(fit.success), 'evaluations':optimizer_residual_evaluations,
        'optimizer_nfev':int(fit.nfev), 'residual_evaluations':optimizer_residual_evaluations,
        'residual_evaluation_budget_definition':'all optimizer objective calls including 3-point numerical Jacobian residuals',
        'fixed_diagnostic_residual_evaluations':total_residual_evaluations-diagnostic_start,
        'diagnostic_calls_outside_optimizer_budget':'final-fit residual, bounded local uncertainty differences and preregistered phase/clock perturbations only; no parameter search',
        'phase_bounds_m':[phase0-100.,phase0+100.], 'clock_bounds_s':[-1.,1.],
        'units_per_m_bounds':[8.,12.], 'maximum_evaluations':120,
        'method':'proper shared 2D similarity + common phase and location clock from unscaled original speed integrals',
        'speed_support':support, 'per_lap_first_sample_reset':False,
        'per_lap_span_stretch':False, 'speed_integral_scale':1., 'model_speed_residuals_used':False,
        'clock_offset_definition':'corrected_location_time = raw_location_time - location_clock_offset_s',
        'linearized_uncertainty':{'phase_std_m':float(std[0]),'clock_std_s':float(std[1]),
            'rank':rank,'normal_condition':condition if np.isfinite(condition) else None,
            'interpretation':'local residual-based diagnostic; correlated GPS/systematic errors are not covered',
            'identifiability':'not_identified' if rank<2 else 'local_only'},
        'fixed_perturbation_sensitivity':sensitivity}


class SharedPositionRoute(ClosedRoute):
    """Periodic analytic route with a frozen nonlinear legacy-coordinate map."""
    def __init__(self,spec):
        points=np.asarray(spec['points_m'],float)
        self.reference_u_m=np.asarray(spec['source_reference_u_m'],float)
        self.reference_length_m=float(spec['source_reference_length_m'])
        if (len(self.reference_u_m)!=len(points)+1 or self.reference_u_m[0]!=0.
                or self.reference_u_m[-1]!=self.reference_length_m
                or np.any(np.diff(self.reference_u_m)<=0)):
            raise DataUnavailable('Invalid frozen legacy reference mapping')
        candidate_u=polyline_length(points)
        phase=float(np.interp(spec['control_reference_u_m']%self.reference_length_m,
                              self.reference_u_m,candidate_u))
        super().__init__(points,kind='spline',phase=phase,direction=int(spec['travel_direction']))
        self.legacy_phase=float(spec['legacy_phase_m']); self.legacy_direction=int(spec['legacy_direction'])

    def legacy_to_s(self,old_s):
        original=np.mod(self.legacy_phase+self.legacy_direction*np.asarray(old_s,float),self.reference_length_m)
        return self.source_to_s(np.interp(original,self.reference_u_m,self.source_s))

    def s_to_legacy(self,s):
        original=np.interp(self.s_to_source(s),self.source_s,self.reference_u_m)
        legacy=np.mod(self.legacy_direction*(original-self.legacy_phase),self.reference_length_m)
        return np.where(np.abs(legacy-self.reference_length_m)<=1e-7,0.,legacy)


def _registered_positions(lap,registration,route):
    a=lap['arrays']; corrected=np.asarray(a['loc_t'],float)-registration['location_clock_offset_s']
    valid=(corrected>=0.)&(corrected<=float(lap['lap_duration']))
    raw=np.asarray(a['loc_xy'],float)[valid]
    xy=(raw-np.asarray(registration['translation_raw_units']))@np.asarray(registration['rotation']).T/registration['raw_location_units_per_m']
    if len(xy)<100: raise DataUnavailable('Too few original in-lap position samples')
    source,error=_project(xy,route.points,route.source_s)
    s=route.source_to_s(source)
    gaps=np.diff(np.r_[np.sort(s),np.min(s)+route.length_m])
    if float(np.max(gaps))>max(100.,.03*route.length_m):
        raise DataUnavailable('Original positions do not cover a common closed route')
    values=route.evaluate(s); tangent=values['tangent']
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    base=np.column_stack([values['x_m'],values['y_m']])
    return {'lap_number':int(lap['lap_number']),'s_m':s,'xy_m':xy,
        'offset_m':np.sum((xy-base)*normal,axis=1),'projection_error_m':error,
        'corrected_time_s':corrected[valid],'source_original':True}


def _periodic_matrix(s,length,n):
    u=np.mod(np.asarray(s,float),length)/(length/n)
    first=np.floor(u).astype(int); fraction=u-first
    ids=np.arange(len(u))
    return coo_matrix((np.r_[1.-fraction,fraction],(np.r_[ids,ids],np.r_[first,(first+1)%n])),shape=(len(u),n)).tocsr()


def _boundary_offset_bounds(base_xy,normals,centre):
    """Node bounds from inherited nearest-segment width projections.

    Spline interpolation is not assumed to preserve these node bounds. It is
    independently subjected to continuous corridor and dense clearance gates.
    """
    centre=np.asarray(centre,float);delta=np.roll(centre[:,:2],-1,axis=0)-centre[:,:2]
    norm2=np.sum(delta**2,axis=1); tangent=delta/np.sqrt(norm2)[:,None]
    leftnormal=np.column_stack([-tangent[:,1],tangent[:,0]])
    lower=[];upper=[]
    for xy,normal in zip(base_xy,normals):
        t=np.clip(np.sum((xy-centre[:,:2])*delta,axis=1)/norm2,0.,1.)
        residual=xy-centre[:,:2]-t[:,None]*delta
        i=int(np.argmin(np.sum(residual**2,axis=1)))
        offset=float(residual[i]@leftnormal[i]);projection=float(normal@leftnormal[i])
        width=(1.-t[i])*centre[i,2:4]+t[i]*centre[(i+1)%len(centre),2:4]
        if abs(projection)<.5: raise DataUnavailable('Route normal cannot be related reliably to boundary normal')
        ends=np.array([-width[0]+BOUNDARY_MARGIN_M-offset,width[1]-BOUNDARY_MARGIN_M-offset])/projection
        lower.append(float(np.min(ends)));upper.append(float(np.max(ends)))
    if np.any(np.asarray(lower)>=np.asarray(upper)):
        raise DataUnavailable('Empty inherited boundary interval')
    return np.asarray(lower),np.asarray(upper)


def fit_route_spec(samples,route,centre,regularization_length_m,old_registration):
    """Equal-per-lap positions plus periodic second derivative, bounded solve.

    Minimize mean_lap(mean_samples(position lateral residual squared)) +
    length^4 * mean_nodes((second derivative of lateral offset)^2).
    This fixes the unit meaning of the three preregistered regularizers.
    """
    scale=float(regularization_length_m)
    if scale not in REGULARIZATION_LENGTHS_M: raise ValueError('Unregistered regularization length')
    n=int(np.ceil(route.length_m/FIT_GRID_MAX_M));ds=route.length_m/n
    s=np.arange(n)*ds;values=route.evaluate(s)
    base=np.column_stack([values['x_m'],values['y_m']]);normal=np.column_stack([-values['tangent'][:,1],values['tangent'][:,0]])
    lower,upper=_boundary_offset_bounds(base,normal,centre)
    matrices=[];targets=[]
    for sample in samples:
        weight=1./math.sqrt(len(samples)*len(sample['s_m']))
        matrices.append(_periodic_matrix(sample['s_m'],route.length_m,n)*weight)
        targets.append(np.asarray(sample['offset_m'],float)*weight)
    second=diags((np.ones(n-1),-2.*np.ones(n),np.ones(n-1)),(-1,0,1),shape=(n,n),format='lil')
    second[0,n-1]=1.;second[n-1,0]=1.
    matrices.append(second.tocsr()*(scale**2/(ds**2*math.sqrt(n))))
    targets.append(np.zeros(n))
    fit=lsq_linear(vstack(matrices,format='csr'),np.concatenate(targets),bounds=(lower,upper),
                   method='trf',lsq_solver='lsmr',tol=1e-8,lsmr_tol=1e-10,max_iter=150,lsmr_maxiter=10000)
    if not fit.success or not np.isfinite(fit.x).all():
        raise DataUnavailable(f'Boundary-constrained route solve failed: {fit.message}')
    travel_points=base+fit.x[:,None]*normal
    # Sort by original source coordinate, preserving the exact non-affine map.
    reference=route.s_to_source(s);order=np.argsort(reference)
    reference=reference[order];points=travel_points[order]
    # The uniform travel grid rarely contains original source zero. Materialize
    # it from the same periodic offset interpolation, not an extra fitted node.
    if reference[0]>1e-9:
        zero_s=float(route.source_to_s(0.));zero=route.evaluate(zero_s)
        offset=float((_periodic_matrix([zero_s],route.length_m,n)@fit.x)[0])
        tangent=np.asarray(zero['tangent']);normal0=np.array([-tangent[1],tangent[0]])
        points=np.vstack([np.array([zero['x_m'],zero['y_m']])+offset*normal0,points])
        reference=np.r_[0.,reference]
    else: reference[0]=0.
    return {'schema_version':9,'kind':'boundary_constrained_position_spline','fit_role':'calibration',
        'fit_laps':sorted(int(x['lap_number']) for x in samples),
        'method':'bounded equal-per-lap lateral residual + periodic second-derivative regularization',
        'regularization_length_m':scale,'regularization_definition':'length^4 times mean squared second spatial derivative',
        'gaussian_sigma_used':False,'fit_grid_max_spacing_m':FIT_GRID_MAX_M,
        'points_m':points.tolist(),'source_reference_u_m':np.r_[reference,route.source_length].tolist(),
        'source_reference_length_m':route.source_length,'control_reference_u_m':route.phase,
        'travel_direction':route.direction,'legacy_phase_m':float(old_registration['original_route_control_line_s_m']),
        'legacy_direction':int(old_registration['direction']), 'model_speed_residuals_used':False,
        'solver':{'converged':bool(fit.success),'iterations':int(fit.nit),'optimality':float(fit.optimality),
            'cost':float(fit.cost),'active_nodes':int(np.count_nonzero(fit.active_mask)),
            'bound_margin_m':BOUNDARY_MARGIN_M,'post_fit_clipping':False}}


def _corridor_rings(centre):
    """Explicit inner/outer width rings with adjacent-segment miter normals.

    A closed circuit is an annulus. Joining its rings into one simple polygon
    would introduce an artificial start-line boundary and reject every lap.
    """
    c=np.asarray(centre,float);delta=np.roll(c[:,:2],-1,axis=0)-c[:,:2]
    tangent=delta/np.linalg.norm(delta,axis=1)[:,None]
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    joint=normal+np.roll(normal,1,axis=0)
    denominator=np.sum(joint*normal,axis=1)
    if np.any(denominator<=1e-8): raise DataUnavailable('Boundary polygon has a reversal')
    miter=joint/denominator[:,None]
    left=c[:,:2]+c[:,3,None]*miter;right=c[:,:2]-c[:,2,None]*miter
    area=lambda p:abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))))
    return (left,right) if area(left)>area(right) else (right,left)


def _point_in_polygon(point,polygon):
    p=np.asarray(point,float);a=polygon;b=np.roll(polygon,-1,axis=0)
    crossings=((a[:,1]>p[1])!=(b[:,1]>p[1]))
    ids=np.flatnonzero(crossings)
    intercept=a[ids,0]+(p[1]-a[ids,1])*(b[ids,0]-a[ids,0])/(b[ids,1]-a[ids,1])
    return bool(np.count_nonzero(intercept>p[0])%2)


def continuous_corridor_check(route,centre):
    """Check cubic/line intersections, rather than declaring dense samples proof.

    Each spline polynomial is intersected analytically with bounding-box
    candidate polygon edges. A tangent contact is also a gate failure.
    """
    outer,inner=_corridor_rings(centre)
    a=np.vstack([outer,inner]);b=np.vstack([np.roll(outer,-1,axis=0),np.roll(inner,-1,axis=0)])
    edges=b-a; norm2=np.sum(edges**2,axis=1)
    initial=route._curve(0.)
    if not _point_in_polygon(initial,outer) or _point_in_polygon(initial,inner):
        return {'passed':False,'reason':'initial_point_outside_inherited_width_polygon','intersections':0,
                'method':'analytic cubic intersections with explicit inherited width polygon'}
    intersections=0
    for i in range(len(route._curve.x)-1):
        h=float(route._curve.x[i+1]-route._curve.x[i]); coeff=route._curve.c[:,i,:]
        # A polynomial extremum encloses the complete segment, not only knots.
        candidates=[np.array([0.,h])]
        for axis in (0,1):
            roots=np.roots([3*coeff[0,axis],2*coeff[1,axis],coeff[2,axis]])
            candidates.append(np.array([r.real for r in roots if abs(r.imag)<1e-10 and 0<r.real<h]))
        t=np.unique(np.concatenate(candidates));xy=route._curve(route._curve.x[i]+t)
        low=xy.min(axis=0)-1e-8;high=xy.max(axis=0)+1e-8
        possible=np.flatnonzero(np.all(np.maximum(a,b)>=low,axis=1)&np.all(np.minimum(a,b)<=high,axis=1))
        for j in possible:
            polynomial=coeff[:,0]*edges[j,1]-coeff[:,1]*edges[j,0]
            polynomial=polynomial.copy();polynomial[-1]-=a[j,0]*edges[j,1]-a[j,1]*edges[j,0]
            if np.max(np.abs(polynomial))<1e-12:
                intersections+=1;continue
            roots=np.roots(np.trim_zeros(polynomial,'f'))
            for r in roots:
                if abs(r.imag)>1e-7 or not -1e-8<=r.real<=h+1e-8: continue
                point=route._curve(route._curve.x[i]+np.clip(r.real,0.,h))
                projection=float((point-a[j])@edges[j]/norm2[j])
                if -1e-8<=projection<=1.+1e-8: intersections+=1
    return {'passed':intersections==0,'reason':None if intersections==0 else 'continuous_curve_touches_or_crosses_width_polygon',
            'intersections':intersections,'method':'analytic cubic intersections with explicit inherited width polygon',
            'surveyed_boundary':False,'numeric_root_imaginary_tolerance':1e-7}


def validate_route(route,centre,baseline_route):
    s=np.arange(0.,route.length_m,.5);values=route.evaluate(s)
    k=np.asarray(values['curvature_1pm']);xy=np.column_stack([values['x_m'],values['y_m']])
    boundary=segment_boundary_audit(xy,centre,route.direction)
    base=baseline_route.evaluate(np.arange(0.,baseline_route.length_m,.5))['curvature_1pm']
    noise=float(np.sqrt(np.mean((k-gaussian_filter1d(k,10.,mode='wrap'))**2)))
    baseline_noise=float(np.sqrt(np.mean((base-gaussian_filter1d(base,10.,mode='wrap'))**2)))
    coarse=route.evaluate(np.arange(0.,route.length_m,1.))['curvature_1pm']
    fine_integral=float(np.mean(np.abs(k))*route.length_m)
    coarse_integral=float(np.mean(np.abs(coarse))*route.length_m)
    relative=abs(coarse_integral-fine_integral)/max(fine_integral,1e-12)
    closure=route.evaluate([0.,route.length_m]);pos=float(np.hypot(np.diff(closure['x_m'])[0],np.diff(closure['y_m'])[0]))
    tangent=float(np.linalg.norm(np.diff(closure['tangent'],axis=0)))
    continuous=continuous_corridor_check(route,centre);reasons=[]
    if boundary['minimum_boundary_clearance_m']<=0: reasons.append('leaves_nearest_segment_inherited_boundaries')
    if not continuous['passed']: reasons.append(continuous['reason'])
    if pos>1e-8 or tangent>1e-8: reasons.append('closure_failed')
    if not np.isfinite(k).all() or float(np.max(np.abs(k)))>max(.08,2*float(np.max(np.abs(base)))):
        reasons.append('curvature_peak_unstable')
    if noise>max(.0005,1.5*baseline_noise): reasons.append('curvature_high_frequency_unstable')
    if relative>.02: reasons.append('curvature_sampling_unstable')
    return {'passed':not reasons,'reasons':reasons,'minimum_boundary_clearance_m':boundary['minimum_boundary_clearance_m'],
        'nearest_segment_outside_samples':boundary['outside_samples'],'clearance_sampling_m':.5,
        'continuous_corridor':continuous,'closure_position_error_m':pos,'closure_tangent_error':tangent,
        'maximum_abs_curvature_1pm':float(np.max(np.abs(k))), 'curvature_high_frequency_rms_1pm':noise,
        'curvature_absolute_integral_1m_vs_half_m_relative_difference':relative,
        'analytic_curvature':True,'true_arc_length':True,'arc_quadrature_spacing_m':.1}


def _position_rmse(xy,route):
    # Closest distance to the fixed polyline knots approximates the curve to
    # <=2m knot spacing; this position-only metric never uses predicted speed.
    _,error=_project(np.asarray(xy,float),route.points,route.source_s)
    return float(np.sqrt(np.mean(error**2)))


def baseline_spec():
    return {'schema_version':9,'kind':'baseline','fit_role':'calibration','fit_laps':list(CALIBRATION_LAPS),
            'method':'unchanged_frozen_v7_TUM_spline','geometry_gate_passed':False,
            'model_speed_residuals_used':False}


def audit_geometry(data_root,public_root=None,output_dir=None):
    """Four honest registration/position folds and one final calibration fit.

    A failed route returns the unchanged baseline plus explicit failure. It
    never authorizes a vehicle screening run merely because baseline exists.
    """
    output=None if output_dir is None else Path(output_dir)
    if output is not None:
        output.mkdir(parents=True,exist_ok=False)
    def save_candidate(candidate,relative):
        payload=(json.dumps(candidate,indent=2,allow_nan=False)+'\n').encode('utf-8')
        evidence={'relative_path':relative,'sha256':sha256(payload).hexdigest(),
            'artifact_saved':output is not None,'diagnostic_candidate_not_adopted':True}
        if output is not None:
            target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(payload)
        else:
            evidence['spec']=deepcopy(candidate)
        return evidence
    root=Path(data_root); view=load_calibration_raw(root)
    old=read_json(root/'data/v3/local/registration.json')
    original=read_json(root/'data/processed/dataset.json')['audit']['track_registration']
    points=np.loadtxt(root/'data/raw/tum/Spielberg_minimum_curvature.csv',delimiter=',')
    centre=np.loadtxt(root/'data/raw/tum/Spielberg_centerline.csv',delimiter=',')
    report={'schema_version':9,'role':'calibration','fit_laps':list(CALIBRATION_LAPS),
        'raw_source_hashes':view.source_hashes,'registration_baseline':old,
        'regularization_lengths_m':list(REGULARIZATION_LENGTHS_M),
        'regularization_definition':'length^4 times mean squared second spatial derivative; not Gaussian sigma',
        'held_out_in_training_registration':False,'uses_model_speed_residuals':False,
        'selection_comparator':'TUM reference route under each fold training-only registration; candidate and comparator share that transform',
        'old_frozen_comparator_used_for_selection':False,
        'per_lap_first_sample_reset':False,'per_lap_span_stretch':False,
        'folds':[],'candidates':[],'selected_kind':'baseline','geometry_gate_passed':False,
        'test_arrays_loaded':False,'sac_attempts':0}
    spec=baseline_spec()
    try:
        folds=[]
        frozen_route=ClosedRoute(points,kind='spline',phase=old['original_route_control_line_s_m'],
            direction=old['direction'],legacy_phase=original['original_route_control_line_s_m'],legacy_direction=original['direction'])
        frozen_samples=[_registered_positions(lap,old,frozen_route) for lap in view.laps]
        report['old_frozen_position_comparator']={'registration_refit':False,
            'registration':old,'position_rmse_per_lap_m':[{'lap_number':x['lap_number'],
                'rmse_m':_position_rmse(x['xy_m'],frozen_route)} for x in frozen_samples],
            'position_cv_rmse_m':float(np.sqrt(np.mean([_position_rmse(x['xy_m'],frozen_route)**2 for x in frozen_samples]))),
            'interpretation':'Preserved old transform and route, separate contextual comparator; no candidate selection by this value'}
        for held in CALIBRATION_LAPS:
            numbers=tuple(x for x in CALIBRATION_LAPS if x!=held)
            registration=fit_registration_view(view,numbers,points,old)
            route=ClosedRoute(points,kind='spline',phase=registration['original_route_control_line_s_m'],
                direction=registration['direction'],legacy_phase=original['original_route_control_line_s_m'],legacy_direction=original['direction'])
            samples=[_registered_positions(lap,registration,route) for lap in view.select(numbers)]
            held_lap=next(lap for lap in view.laps if int(lap['lap_number'])==held)
            held_sample=_registered_positions(held_lap,registration,route)
            boundary=segment_boundary_audit(held_sample['xy_m'],centre,route.direction)
            record={'held_out_lap':held,'fit_laps':list(numbers),'registration':registration,
                'baseline_position_rmse_m':_position_rmse(held_sample['xy_m'],route),
                'held_out_original_position_count':len(held_sample['xy_m']),
                'held_out_positions_outside_fraction':boundary['outside_samples']/len(held_sample['xy_m'])}
            report['folds'].append(record);folds.append((record,route,samples,held_sample))
        final_registration=fit_registration_view(view,CALIBRATION_LAPS,points,old)
        final_route=ClosedRoute(points,kind='spline',phase=final_registration['original_route_control_line_s_m'],
            direction=final_registration['direction'],legacy_phase=original['original_route_control_line_s_m'],legacy_direction=original['direction'])
        final_samples=[_registered_positions(lap,final_registration,final_route) for lap in view.laps]
        report['final_registration']=final_registration
        report['baseline_position_cv_rmse_m']=float(np.sqrt(np.mean([x[0]['baseline_position_rmse_m']**2 for x in folds])))
        report['raw_position_outside_fraction_per_lap']=[{'lap_number':x['lap_number'],
            'fraction':segment_boundary_audit(x['xy_m'],centre,final_route.direction)['outside_samples']/len(x['xy_m'])} for x in final_samples]
        eligible=[]
        for scale in REGULARIZATION_LENGTHS_M:
            row={'regularization_length_m':scale,'folds':[],'passed':False};report['candidates'].append(row)
            try:
                for fold,baseline,samples,held_sample in folds:
                    foldspec=fit_route_spec(samples,baseline,centre,scale,original)
                    foldspec['registration']=fold['registration']
                    evidence=save_candidate(foldspec,f'candidates/length_{int(scale):03d}/held_out_{fold["held_out_lap"]}.json')
                    candidate=SharedPositionRoute(foldspec)
                    row['folds'].append({'held_out_lap':fold['held_out_lap'],'fit_laps':fold['fit_laps'],
                        'position_rmse_m':_position_rmse(held_sample['xy_m'],candidate),
                        'geometry':validate_route(candidate,centre,baseline),'solver':foldspec['solver'],
                        'route_evidence':evidence})
                full=fit_route_spec(final_samples,final_route,centre,scale,original)
                full['registration']=final_registration
                row['final_route_evidence']=save_candidate(full,f'candidates/length_{int(scale):03d}/full_calibration.json')
                row['final_geometry']=validate_route(SharedPositionRoute(full),centre,final_route)
                row['final_solver']=full['solver']
                row['position_cv_rmse_m']=float(np.sqrt(np.mean([x['position_rmse_m']**2 for x in row['folds']])))
                row['passed']=row['final_geometry']['passed'] and all(x['geometry']['passed'] for x in row['folds'])
                if row['passed'] and row['position_cv_rmse_m']<=report['baseline_position_cv_rmse_m']*.99:
                    eligible.append((row,full))
            except (DataUnavailable,ValueError,FloatingPointError) as exc:
                row['reason']=str(exc)
        if eligible:
            best=min(x[0]['position_cv_rmse_m'] for x in eligible)
            selected=max((x for x in eligible if x[0]['position_cv_rmse_m']<=best*1.01),key=lambda x:x[0]['regularization_length_m'])
            chosen,spec=selected;spec['geometry_gate_passed']=True
            report.update(selected_kind=spec['kind'],geometry_gate_passed=True,
                selected_regularization_length_m=chosen['regularization_length_m'],
                selected_position_cv_rmse_m=chosen['position_cv_rmse_m'],
                reason='Position-only leave-one-out improvement and all fold/full geometry gates passed')
        else:
            report['reason']='No boundary-stable calibration route improves unbiased position CV by at least 1%'
        report['conclusion']='supported_common_reference_route' if eligible else 'not_identified; no vehicle screening authorized'
    except (DataUnavailable,ValueError,KeyError,FloatingPointError) as exc:
        report.update(reason=str(exc),conclusion='not_identified; calibration-only support or registration failed')
    if output is not None:
        (output/'geometry_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (output/'route_spec.json').write_text(json.dumps(spec,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return spec,report


def build_track_from_spec(data_root,spec,ds_m=5.):
    root=Path(data_root)
    if spec.get('kind')=='baseline': return build_track(root,'spline',ds_m,read_json(root/'data/v3/local/registration.json'))
    if (spec.get('kind')!='boundary_constrained_position_spline' or not spec.get('geometry_gate_passed')
            or spec.get('fit_role')!='calibration' or spec.get('fit_laps')!=list(CALIBRATION_LAPS)):
        raise DataUnavailable('A fully gated calibration-only frozen route is required')
    registration=spec['registration'];route=SharedPositionRoute(spec)
    baseline=build_track(root,'spline',ds_m,registration)
    centre=np.loadtxt(root/'data/raw/tum/Spielberg_centerline.csv',delimiter=',')
    check=validate_route(route,centre,baseline['continuous_route'])
    if not check['passed']: raise DataUnavailable('Frozen route fails independent geometry verification')
    track=route.sample(ds_m);xy=np.column_stack([track['x_m'],track['y_m']])
    widths=segment_boundary_audit(xy,centre,route.direction)
    old=read_json(root/'data/processed/dataset.json')['track']
    track.update(name=old['name'],width_left_m=np.asarray(widths['left_clearance_m']),width_right_m=np.asarray(widths['right_clearance_m']),
        candidate=spec['kind'],line_type='Calibration-only boundary-constrained common reference line',
        corner_s_m={k:float(route.legacy_to_s(v)) for k,v in old['corner_s_m'].items()},
        sector_end_s_m=[float(route.legacy_to_s(v)) for v in old['sector_end_s_m'][:-1]]+[route.length_m],
        approximate_aero_zones_m=[{**z,'start_m':float(route.legacy_to_s(z['start_m'])),
            'end_m':float(route.legacy_to_s(z['end_m']))} for z in old['approximate_aero_zones_m']],
        fia_s_m=route.s_to_legacy(track['s_m'])*4326./route.reference_length_m,
        geometry_audit={**check,'curve_id':route.curve_id,'length_m':route.length_m,
            'regularization_length_m':spec['regularization_length_m'],'mesh_refit':False,
            'geometry_source':'Shared calibration positions; historical inherited boundary, not surveyed 2026 racing line'})
    return track


def remap_rules(rules,track):
    route=track['continuous_route']
    if not isinstance(route,SharedPositionRoute):
        from .._core.geometry import remap_rules as baseline_remap
        return baseline_remap(rules,track)
    result=deepcopy(rules)
    convert=lambda v:float(route.legacy_to_s(float(v)*route.reference_length_m/rules['fia_centreline_length_m']))
    result['alternative_power_zones_route_m']=[[convert(a),convert(b)] for a,b in rules['alternative_power_zones_m']]
    result['power_reduction_exceptions_route']=[{**z,'start_m':convert(z['start_m']),'end_m':convert(z['end_m'])} for z in rules['power_reduction_exceptions']]
    if 'aero_straight_zones_m' in result:
        zones=result.pop('aero_straight_zones_m')
        track['approximate_aero_zones_m']=[{'start_m':convert(z['start_m'] if isinstance(z,dict) else z[0]),
            'end_m':convert(z['end_m'] if isinstance(z,dict) else z[1])} for z in zones]
    result['coordinate_transform']={'method':'FIA fraction -> frozen legacy reference -> nonlinear shared-curve true arc length',
        'route_length_m':route.length_m,'source_length_m':route.reference_length_m,'curve_id':route.curve_id,'surveyed':False}
    return result
