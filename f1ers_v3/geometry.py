"""One frozen continuous route, independent of the simulation mesh."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline, PchipInterpolator

from .protocol import DataUnavailable, read_json


def polyline_length(points):
    distances = np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    if np.any(distances <= 1e-10):
        raise DataUnavailable("A closed source route has duplicate neighbouring points")
    return np.r_[0.,np.cumsum(distances)]


class ClosedRoute:
    """Periodic source-parameter curve, transformed to actual arc-length coordinates.

    The interpolation is created exactly once from the source. Resampling never
    refits the spline and cannot change curvature/length/zone coordinates.
    """
    def __init__(self, points, *, kind="spline", phase=0., direction=1,
                 legacy_curvature=None, legacy_phase=0., legacy_direction=1):
        self.points = np.asarray(points,float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points)<4 or not np.isfinite(self.points).all():
            raise DataUnavailable("Invalid two-dimensional closed route")
        if kind not in ("alignment","spline","anisotropic") or direction not in (-1,1):
            raise ValueError("Unknown route candidate or direction")
        self.kind,self.direction = kind,direction
        self.source_s = polyline_length(self.points)
        self.source_length = float(self.source_s[-1])
        self.phase = float(phase)%self.source_length
        self.legacy_phase,self.legacy_direction = float(legacy_phase),int(legacy_direction)
        self._legacy_curvature=legacy_curvature
        if kind == "alignment":
            self._curve = None
            self._arc_s = self.source_s
            self._arc_u = self.source_s
            self.length_m = self.source_length
        else:
            self._curve = CubicSpline(self.source_s,np.vstack([self.points,self.points[0]]),bc_type="periodic",axis=0)
            # Fixed integration mesh contains every spline knot. Simpson quadrature
            # and monotone inversion are independent of all simulation meshes.
            u=np.unique(np.r_[np.linspace(0,self.source_length,int(np.ceil(self.source_length/.1))+1),self.source_s])
            derivative=self._curve(u,1)
            speed=np.linalg.norm(derivative,axis=1)
            if np.min(speed)<.1:
                raise DataUnavailable("Periodic spline is locally singular")
            self._arc_u=u
            self._arc_s=np.r_[0.,cumulative_simpson(speed,x=u)]
            self.length_m=float(self._arc_s[-1])
        self._u_to_arc=PchipInterpolator(self._arc_u,self._arc_s,extrapolate=False)
        self._arc_to_u=PchipInterpolator(self._arc_s,self._arc_u,extrapolate=False)
        self.phase_arc=float(self._u_to_arc(self.phase))
        digest=hashlib.sha256(self.points.tobytes()+str((kind,self.phase,direction)).encode())
        self.curve_id=digest.hexdigest()

    def source_to_s(self,u):
        """Original polyline parameter to travel distance after frozen control line."""
        u=np.mod(np.asarray(u,float),self.source_length)
        return np.mod(self.direction*(self._u_to_arc(u)-self.phase_arc),self.length_m)

    def s_to_source(self,s):
        arc=np.mod(self.phase_arc+self.direction*np.asarray(s,float),self.length_m)
        return self._arc_to_u(arc)

    def legacy_to_s(self,old_s):
        source=self.legacy_phase+self.legacy_direction*np.asarray(old_s,float)
        return self.source_to_s(source)

    def s_to_legacy(self,s):
        return np.mod(self.legacy_direction*(self.s_to_source(s)-self.legacy_phase),self.source_length)

    def evaluate(self,s):
        u=self.s_to_source(s)
        if self._curve is None:
            xy=np.stack([np.interp(u,self.source_s,np.r_[self.points[:,j],self.points[0,j]]) for j in (0,1)],axis=-1)
            if self._legacy_curvature is None:
                curvature=np.zeros_like(u)
            else:
                cs,cv=self._legacy_curvature
                curvature=np.interp(self.s_to_legacy(s),np.r_[cs,self.source_length],np.r_[cv,cv[0]])*self.direction/self.legacy_direction
            ids=np.minimum(np.searchsorted(self.source_s,u,side="right")-1,len(self.points)-1)
            tangent=(np.roll(self.points,-1,axis=0)-self.points)[ids]*self.direction
            tangent=tangent/np.linalg.norm(tangent,axis=-1,keepdims=True)
        else:
            xy=self._curve(u)
            first=self._curve(u,1);second=self._curve(u,2)
            norm=np.linalg.norm(first,axis=-1)
            curvature=(first[...,0]*second[...,1]-first[...,1]*second[...,0])/norm**3*self.direction
            tangent=self.direction*first/norm[...,None]
        return {"x_m":xy[...,0],"y_m":xy[...,1],"curvature_1pm":curvature,"tangent":tangent}

    def sample(self,ds_m=5.):
        if not np.isfinite(ds_m) or ds_m<=0:
            raise ValueError("Mesh distance must be finite and positive")
        n=int(np.ceil(self.length_m/ds_m))
        s=np.arange(n,dtype=float)*self.length_m/n
        values=self.evaluate(s)
        return {"s_m":s,"ds_m":self.length_m/n,"requested_ds_m":float(ds_m),"nodes":n,
                "length_m":self.length_m,"segment_ds_m":np.full(n,self.length_m/n),
                "curve_id":self.curve_id,"continuous_route":self,**values}


def _widths(root,xy,direction):
    centre=np.loadtxt(Path(root)/"data/raw/tum/Spielberg_centerline.csv",delimiter=",")
    nearest=np.argmin(np.sum((xy[:,None,:]-centre[None,:,:2])**2,axis=2),axis=1)
    tangent=np.roll(centre[:,:2],-1,axis=0)-np.roll(centre[:,:2],1,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normals=np.column_stack([-tangent[:,1],tangent[:,0]])
    offset=np.sum((xy-centre[nearest,:2])*normals[nearest],axis=1)
    left,right=centre[nearest,3]-offset,centre[nearest,2]+offset
    if direction<0:left,right=right,left
    return left,right


def build_track(root,candidate="spline",ds_m=5.,registration=None):
    root=Path(root)
    old=read_json(root/"data/processed/dataset.json")
    oldreg=old["audit"]["track_registration"]
    reg=registration or oldreg
    points=np.loadtxt(root/"data/raw/tum/Spielberg_minimum_curvature.csv",delimiter=",")
    with np.load(root/"data/processed/track.npz",allow_pickle=False) as archive:
        oldcurvature=(archive["s_m"].copy(),archive["curvature_1pm"].copy())
    route=ClosedRoute(points,kind=candidate,phase=reg["original_route_control_line_s_m"],direction=reg["direction"],
                      legacy_curvature=oldcurvature,legacy_phase=oldreg["original_route_control_line_s_m"],legacy_direction=oldreg["direction"])
    track=route.sample(ds_m)
    xy=np.column_stack([track["x_m"],track["y_m"]])
    left,right=_widths(root,xy,route.direction)
    dense=route.evaluate(np.arange(0,route.length_m,1.))
    dense_left,dense_right=_widths(root,np.column_stack([dense["x_m"],dense["y_m"]]),route.direction)
    if min(float(dense_left.min()),float(dense_right.min()))<=0:
        raise DataUnavailable("Continuous candidate leaves inherited TUM track boundaries")
    oldtrack=old["track"]
    track.update(name=oldtrack["name"],width_left_m=left,width_right_m=right,
                 line_type=f"Frozen TUM route / {candidate}",candidate=candidate,
                 corner_s_m={k:float(route.legacy_to_s(v)) for k,v in oldtrack["corner_s_m"].items()},
                 sector_end_s_m=[float(route.legacy_to_s(v)) for v in oldtrack["sector_end_s_m"][:-1]]+[route.length_m],
                 approximate_aero_zones_m=[{**z,"start_m":float(route.legacy_to_s(z["start_m"])),"end_m":float(route.legacy_to_s(z["end_m"]))} for z in oldtrack["approximate_aero_zones_m"]],
                 fia_s_m=route.s_to_legacy(track["s_m"])*4326./route.source_length,
                 geometry_audit={"curve_id":route.curve_id,"length_m":route.length_m,"source_polyline_length_m":route.source_length,
                   "analytic_curvature":candidate!="alignment","arc_quadrature_spacing_max_m":.1 if candidate!="alignment" else None,
                   "minimum_boundary_clearance_m":min(float(dense_left.min()),float(dense_right.min())),
                   "mesh_refit":False,"geometry_source":"Historical TUM minimum-curvature route; not the observed racing line"})
    return track


def remap_rules(rules,track):
    """FIA numeric zones and inherited aero endpoints use one explicit transform."""
    import copy
    result=copy.deepcopy(rules)
    route=track["continuous_route"]
    to_route=lambda value:float(route.legacy_to_s(float(value)*route.source_length/rules["fia_centreline_length_m"]))
    result["alternative_power_zones_route_m"]=[[to_route(a),to_route(b)] for a,b in rules["alternative_power_zones_m"]]
    result["power_reduction_exceptions_route"]=[{**z,"start_m":to_route(z["start_m"]),"end_m":to_route(z["end_m"])} for z in rules["power_reduction_exceptions"]]
    result["coordinate_transform"]={"method":"FIA fraction -> frozen legacy source parameter -> continuous candidate arc length",
                                     "route_length_m":route.length_m,"source_length_m":route.source_length,"curve_id":route.curve_id,
                                     "surveyed":False}
    return result
