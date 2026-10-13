"""Synthetic geometry/registration counterexamples with no historical data."""
import copy
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from f1ers_v7._core.geometry import ClosedRoute, polyline_length
from f1ers_v7._core.input_io import DataUnavailable
from f1ers_v7._core.input_channels import _similarity
from f1ers_v7.research.geometry_registration import (CalibrationRawView,
    _supported_speed_integral, fit_registration_view, fit_route_spec,
    SharedPositionRoute, continuous_corridor_check, validate_route,
    remap_rules, _boundary_offset_bounds, REGULARIZATION_LENGTHS_M, audit_geometry)


def circular_reference(radius=50.,nodes=64):
    angle=np.arange(nodes)*2*np.pi/nodes
    points=np.column_stack([radius*np.cos(angle),radius*np.sin(angle)])
    return points,np.column_stack([points,np.full(nodes,8.),np.full(nodes,8.)])


def synthetic_raw_view():
    points,_=circular_reference();source=polyline_length(points);length=source[-1]
    laps=[];duration=40.;phase=12.;clock=.2;rotation=np.array([[np.cos(.03),-np.sin(.03)],[np.sin(.03),np.cos(.03)]])
    for lap in (3,4,5,6):
        car_t=np.linspace(.1,duration-.1,401)
        speed=length/duration*(1.+.3*np.sin(2*np.pi*car_t/duration))
        row={'lap_number':lap,'role':'calibration','lap_duration':duration,'arrays':{'car_t':car_t,'speed':speed}}
        loc_t=np.linspace(1.,duration-1.,301)
        progress=_supported_speed_integral(row,loc_t-clock)
        u=np.mod(phase+progress,length)
        template=np.column_stack([np.interp(u,source,np.r_[points[:,j],points[0,j]]) for j in (0,1)])
        row['arrays'].update(loc_t=loc_t,loc_xy=10.*template@rotation+np.array([130.,-90.]))
        laps.append(row)
    old={'direction':1,'original_route_control_line_s_m':10.}
    return CalibrationRawView(tuple(laps),{}),points,old


class GeometryRegistrationTests(unittest.TestCase):
    def test_fold_registration_does_not_use_held_out_values(self):
        view,points,old=synthetic_raw_view()
        first=fit_registration_view(view,(3,4,5),points,old)
        changed=copy.deepcopy(view)
        changed.laps[3]['arrays']['loc_xy'][:]=1e8
        changed.laps[3]['arrays']['speed'][:]=1e5
        second=fit_registration_view(changed,(3,4,5),points,old)
        self.assertEqual(first,second)
        self.assertEqual(first['fit_laps'],[3,4,5])
        self.assertAlmostEqual(first['location_clock_offset_s'],.2,5)
        self.assertAlmostEqual(first['original_route_control_line_s_m'],12.,5)
        self.assertTrue(all(x['source_lap'] in (3,4,5) and not x['foreign_lap_samples'] for x in first['speed_support']))

    def test_fold_guard_cannot_be_bypassed_by_adaptation_role(self):
        view,points,old=synthetic_raw_view()
        with self.assertRaises(DataUnavailable): fit_registration_view(view,(3,4),points,old)
        bad=CalibrationRawView(view.laps,{},role='adaptation')
        with self.assertRaises(DataUnavailable): fit_registration_view(bad,(3,4,5),points,old)
        bad_laps=copy.deepcopy(view.laps);bad_laps[0]['role']='development'
        with self.assertRaises(DataUnavailable): fit_registration_view(CalibrationRawView(bad_laps,{}),(3,4,5),points,old)

    def test_registration_prior_is_frozen_but_legacy_rule_origin_is_preserved(self):
        view,points,old=synthetic_raw_view();_,centre=circular_reference()
        frozen=fit_registration_view(view,(3,4,5,6),points,old)
        legacy={'original_route_control_line_s_m':37.,'direction':1}
        calls=[]
        def fit_guard(view,numbers,reference,prior):
            calls.append(prior)
            raise DataUnavailable('stop after verifying declared prior')
        def read_guard(path):
            return frozen if str(path).replace('\\','/').endswith('data/v3/local/registration.json') else {'audit':{'track_registration':legacy}}
        def load_guard(path,delimiter):
            return centre if 'centerline' in str(path) else points
        with patch('f1ers_v7.research.geometry_registration.load_calibration_raw',return_value=view), \
                patch('f1ers_v7.research.geometry_registration.read_json',side_effect=read_guard), \
                patch('f1ers_v7.research.geometry_registration.np.loadtxt',side_effect=load_guard), \
                patch('f1ers_v7.research.geometry_registration.fit_registration_view',side_effect=fit_guard):
            _,report=audit_geometry('unused')
        self.assertEqual(calls,[frozen])
        self.assertNotEqual(calls[0]['original_route_control_line_s_m'],legacy['original_route_control_line_s_m'])
        self.assertFalse(report['geometry_gate_passed'])
        self.assertIn('old_frozen_position_comparator',report)

    def test_rank_deficient_uncertainty_is_json_safe_and_not_identified(self):
        view,points,old=synthetic_raw_view()
        with patch('f1ers_v7.research.geometry_registration.np.linalg.cond',return_value=float('inf')), \
                patch('f1ers_v7.research.geometry_registration.np.linalg.matrix_rank',return_value=0):
            result=fit_registration_view(view,(3,4,5),points,old)
        self.assertIsNone(result['linearized_uncertainty']['normal_condition'])
        self.assertEqual(result['linearized_uncertainty']['identifiability'],'not_identified')
        json.dumps(result,allow_nan=False)

    def test_failed_geometry_specs_are_saved_before_validation_and_never_overwritten(self):
        view,points,old=synthetic_raw_view();_,centre=circular_reference()
        frozen=fit_registration_view(view,(3,4,5,6),points,old)
        source=polyline_length(points)
        template={'points_m':points.tolist(),'source_reference_u_m':source.tolist(),
            'source_reference_length_m':float(source[-1]),'control_reference_u_m':12.,
            'travel_direction':1,'legacy_phase_m':10.,'legacy_direction':1,'solver':{'converged':True}}
        def fit_registration(view,numbers,reference,prior):
            return {**frozen,'fit_laps':list(numbers)}
        def fit_route(samples,*args):
            return {**copy.deepcopy(template),'fit_laps':[x['lap_number'] for x in samples]}
        def read_guard(path):
            return frozen if str(path).replace('\\','/').endswith('data/v3/local/registration.json') else {'audit':{'track_registration':old}}
        def load_guard(path,delimiter):return centre if 'centerline' in str(path) else points
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'geometry'
            with patch('f1ers_v7.research.geometry_registration.load_calibration_raw',return_value=view), \
                    patch('f1ers_v7.research.geometry_registration.read_json',side_effect=read_guard), \
                    patch('f1ers_v7.research.geometry_registration.np.loadtxt',side_effect=load_guard), \
                    patch('f1ers_v7.research.geometry_registration.fit_registration_view',side_effect=fit_registration), \
                    patch('f1ers_v7.research.geometry_registration.fit_route_spec',side_effect=fit_route), \
                    patch('f1ers_v7.research.geometry_registration.validate_route',return_value={'passed':False,'reasons':['synthetic_failure']}):
                _,report=audit_geometry('unused',output_dir=output)
                self.assertFalse(report['geometry_gate_passed'])
                for candidate in report['candidates']:
                    evidence=[x['route_evidence'] for x in candidate['folds']]+[candidate['final_route_evidence']]
                    self.assertEqual(len(evidence),5)
                    for row in evidence:
                        self.assertEqual(sha256((output/row['relative_path']).read_bytes()).hexdigest(),row['sha256'])
                        self.assertFalse(Path(row['relative_path']).is_absolute())
                with self.assertRaises(FileExistsError):audit_geometry('unused',output_dir=output)

    def test_missing_training_boundary_support_fails_instead_of_borrowing(self):
        view,points,old=synthetic_raw_view();lap=copy.deepcopy(view.laps[0])
        lap['arrays']['car_t']=lap['arrays']['car_t'][10:]
        lap['arrays']['speed']=lap['arrays']['speed'][10:]
        with self.assertRaises(DataUnavailable): _supported_speed_integral(lap,[0.,10.])

    def test_registration_cap_counts_jacobian_objective_calls(self):
        view,points,old=synthetic_raw_view()
        def excessive_objective_calls(fun,*args,**kwargs):
            for _ in range(121):fun(np.array([old['original_route_control_line_s_m'],0.]))
            self.fail('A 121st optimizer residual must be rejected')
        with patch('f1ers_v7.research.geometry_registration.least_squares',side_effect=excessive_objective_calls), \
                patch('f1ers_v7.research.geometry_registration._similarity',wraps=_similarity) as calls:
            with self.assertRaisesRegex(DataUnavailable,'budget exhausted: 120/120'):
                fit_registration_view(view,(3,4,5),points,old)
        self.assertEqual(calls.call_count,120)
        fitted=fit_registration_view(view,(3,4,5),points,old)
        self.assertLessEqual(fitted['residual_evaluations'],120)
        self.assertGreater(fitted['residual_evaluations'],fitted['optimizer_nfev'])
        self.assertEqual(fitted['fixed_diagnostic_residual_evaluations'],9)

    def test_speed_integral_has_no_origin_reset_or_span_scaling(self):
        lap={'lap_duration':10.,'arrays':{'car_t':np.array([.1,1.,9.9]),'speed':np.array([20.1,21.,29.9])}}
        # The underlying exact speed is v=20+t, hence distance=20t+t²/2.
        np.testing.assert_allclose(_supported_speed_integral(lap,[0.,2.,10.]),[0.,42.,250.],atol=1e-12)
        with self.assertRaises(DataUnavailable): _supported_speed_integral(lap,10.001)

    def test_continuous_check_rejects_crossing_between_safe_knots(self):
        points,centre=circular_reference()
        safe=ClosedRoute(points,kind='spline')
        self.assertTrue(continuous_corridor_check(safe,centre)['passed'])
        # Alternating radius knots lie inside a very narrow corridor, while
        # cubic overshoot at their adjoining intervals leaves the corridor.
        angle=np.arange(16)*2*np.pi/16
        radius=np.where(np.arange(16)%4<2,50.98,49.02)
        zig=np.column_stack([radius*np.cos(angle),radius*np.sin(angle)])
        narrow=centre.copy();narrow[:,2:4]=1.
        curve=ClosedRoute(zig,kind='spline')
        check=continuous_corridor_check(curve,narrow)
        self.assertFalse(check['passed'])
        self.assertGreater(check['intersections'],0)

    def test_bound_constrained_fit_is_not_post_fit_clipped(self):
        points,centre=circular_reference();route=ClosedRoute(points,kind='spline')
        s=np.linspace(0.,route.length_m,300,endpoint=False)
        samples=[{'lap_number':lap,'s_m':s,'offset_m':np.full(len(s),20.)} for lap in (3,4,5)]
        old={'original_route_control_line_s_m':0.,'direction':1}
        spec=fit_route_spec(samples,route,centre,20.,old)
        self.assertFalse(spec['solver']['post_fit_clipping'])
        self.assertTrue(spec['solver']['converged'])
        self.assertGreater(spec['solver']['active_nodes'],0)
        self.assertFalse(spec['gaussian_sigma_used'])
        self.assertLessEqual(spec['fit_grid_max_spacing_m'],2.)

    def test_regularity_gate_rejects_synthetic_curvature_spikes(self):
        points,centre=circular_reference();base=ClosedRoute(points,kind='spline')
        angle=np.arange(256)*2*np.pi/256
        radius=50.+.15*np.sin(64.*angle)
        noisy=ClosedRoute(np.column_stack([radius*np.cos(angle),radius*np.sin(angle)]),kind='spline')
        check=validate_route(noisy,centre,base)
        self.assertFalse(check['passed'])
        self.assertTrue(any('curvature' in x for x in check['reasons']))

    def test_shared_map_is_nonlinear_and_periodic_and_rule_coordinates_use_it(self):
        points,centre=circular_reference();route=ClosedRoute(points,kind='spline',phase=10.)
        s=np.linspace(0.,route.length_m,301,endpoint=False)
        samples=[{'lap_number':lap,'s_m':s,'offset_m':2.*np.cos(2*np.pi*s/route.length_m)} for lap in (3,4,5,6)]
        old={'original_route_control_line_s_m':10.,'direction':1}
        spec=fit_route_spec(samples,route,centre,20.,old);shared=SharedPositionRoute(spec)
        old_s=np.linspace(0.,shared.reference_length_m,17,endpoint=False)
        mapped=shared.legacy_to_s(old_s)
        recovered=shared.s_to_legacy(mapped)
        np.testing.assert_allclose(recovered,old_s,atol=1e-6)
        self.assertGreater(np.ptp(np.diff(mapped)),.01)
        self.assertLess(np.linalg.norm(np.diff(shared.evaluate([0.,shared.length_m])['tangent'],axis=0)),1e-8)
        rules={'fia_centreline_length_m':4326.,'alternative_power_zones_m':[[100.,400.]],
               'power_reduction_exceptions':[], 'aero_straight_zones_m':[[200.,600.]]}
        track={'continuous_route':shared}
        result=remap_rules(rules,track)
        self.assertAlmostEqual(result['alternative_power_zones_route_m'][0][0],
            float(shared.legacy_to_s(100.*shared.reference_length_m/4326.)),10)
        self.assertNotIn('aero_straight_zones_m',result)
        self.assertTrue(result['coordinate_transform']['method'].startswith('FIA fraction'))
        self.assertEqual(REGULARIZATION_LENGTHS_M,(20.,40.,80.))


if __name__=='__main__': unittest.main()
