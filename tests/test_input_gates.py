"""Regression scenarios for terminal raw input and independent acceptance gates."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np

from f1ers_v7._core.base_simulator import ReplayDriver, SimulatorV4
from f1ers_v7._core.integrator import (IntegrationState, IntegratorConfig,
    VehicleParamsV4, WheelCommand, advance_interval, vehicle_step)
from f1ers_v7._core.metrics import score
from f1ers_v7._core.geometry import ClosedRoute
from f1ers_v7._core.base_rules import RuleConfig
from f1ers_v7.research.gates import GateReport, model_fit_eligible
from f1ers_v7.research.physics import (sobol_viability, replay_numerical_audit,
    autonomous_physical_numerical_audit)
from f1ers_v7.research.environment import ResearchSimulator


class StraightGeometry:
    """A local analytic straight fixture; not a route-validation fixture."""
    length_m = 100.

    def evaluate(self, s):
        values = np.asarray(s, float)
        return dict(x_m=values, y_m=np.zeros_like(values),
            curvature_1pm=np.zeros_like(values), drag_area_m2=np.zeros_like(values),
            downforce_area_m2=np.zeros_like(values), aero_mode=np.zeros_like(values, bool))

    def next_space_event(self, s):
        return np.inf


class RecordingReplay(ReplayDriver):
    def __init__(self, *args):
        super().__init__(*args)
        self.queries = []

    def query(self, t, s):
        self.queries.append(float(t))
        return super().query(t, s)


def replay_trace(end=.033, **metadata):
    # Later observed speeds must have no opportunity to overwrite dynamics.
    return dict(time_s=np.array([0., end/2, end]),
        speed_mps=np.array([20., 200., 2.]), throttle=np.zeros(3),
        brake=np.zeros(3), **metadata)


class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = StraightGeometry()
        self.params = VehicleParamsV4(rolling_coefficient=0., drag_area_m2=0.,
            downforce_area_m2=0.)

    def test_input_event_exact_terminal_and_no_control_extrapolation(self):
        driver = RecordingReplay(replay_trace(), self.params, self.geometry)
        end, info = advance_interval(IntegrationState(0., 20.), driver, None,
            100., self.geometry, self.params)
        self.assertEqual(info['status'], 'input_exhausted')
        self.assertEqual(end.elapsed_time_s, driver.final_time_s)
        self.assertAlmostEqual(end.s_m, 20.*driver.final_time_s, 10)
        self.assertAlmostEqual(end.speed_mps, 20., 8)
        self.assertGreater(info['accepted_steps'], 0)
        self.assertEqual(info['step_records'][-1]['time_end_s'], driver.final_time_s)
        self.assertTrue(all(t < driver.final_time_s for t in driver.queries))
        self.assertEqual(info['events'][-1]['kind'], 'input_exhausted')
        with self.assertRaises(ValueError):
            driver.query(np.nextafter(driver.final_time_s, np.inf), end.s_m)

    def test_tiny_remaining_time_is_integrated_before_exhaustion(self):
        driver = RecordingReplay(replay_trace(.02), self.params, self.geometry)
        start = IntegrationState(0., 20., .02-5e-9)
        end, info = advance_interval(start, driver, None, 100., self.geometry, self.params)
        self.assertEqual(info['status'], 'input_exhausted')
        self.assertEqual(end.elapsed_time_s, .02)
        self.assertGreater(end.s_m, 0.)
        self.assertGreater(len(info['step_records']), 0)

    def test_declared_raw_coverage_cannot_be_extended_by_processed_boundary(self):
        trace = replay_trace(.1, coverage_end_s=.025, coverage_provenance='last original sample')
        driver = RecordingReplay(trace, self.params, self.geometry)
        end, info = advance_interval(IntegrationState(0., 20.), driver, None,
            100., self.geometry, self.params)
        self.assertEqual(info['status'], 'input_exhausted')
        self.assertEqual(end.elapsed_time_s, .025)
        self.assertEqual(driver.coverage_provenance, 'last original sample')
        with self.assertRaises(ValueError):
            ReplayDriver(replay_trace(.1, coverage_end_s=.11), self.params, self.geometry)

    def test_finish_before_input_end_remains_complete_event(self):
        driver = RecordingReplay(replay_trace(), self.params, self.geometry)
        end, info = advance_interval(IntegrationState(0., 20.), driver, None,
            .3, self.geometry, self.params)
        self.assertEqual(info['status'], 'accepted')
        self.assertAlmostEqual(end.s_m, .3, 5)
        self.assertLess(end.elapsed_time_s, driver.final_time_s)

    def test_linear_throttle_and_switch_hold_are_not_changed(self):
        trace = dict(time_s=np.array([0., .01, .02]), speed_mps=np.full(3, 20.),
            throttle=np.array([0., 1., 0.]), brake=np.array([1., 0., 0.]))
        driver = ReplayDriver(trace, self.params, self.geometry)
        before = driver.query(.005, .1)
        after = driver.query(.01, .2)
        self.assertAlmostEqual(before['power_kw'], self.params.equivalent_power_kw/2)
        self.assertEqual(before['brake_fraction'], 1.)
        self.assertEqual(after['brake_fraction'], 0.)

    def test_partial_physical_prefix_gets_no_formal_accuracy(self):
        sim = SimulatorV4.__new__(SimulatorV4)
        sim.geometry, sim.vehicle, sim.length_m = self.geometry, self.params, 100.
        sim.config = IntegratorConfig()
        result = sim.simulate(mode='replay', driver_trace=replay_trace())
        self.assertEqual(result['terminal_status'], 'input_exhausted')
        self.assertFalse(result['summary']['completed_lap'])
        self.assertTrue(result['summary']['execution_feasible'])
        self.assertFalse(result['summary']['input_coverage_passed'])
        self.assertEqual(result['terminal_time_s'], .033)
        self.assertAlmostEqual(result['internal_trace']['s_m'][-1], .66, 10)
        self.assertEqual(result['summary']['intermediate_speed_resets'], 0)
        view = {'audit':{'data_gate_passed':True},
                'laps':[{'lap_number':3, 'role':'calibration'}]}
        report = score(result, view)
        self.assertFalse(report['metrics'][0]['valid'])
        self.assertNotIn('speed_rmse_kmh', report['metrics'][0])
        gates = GateReport.from_result(result, report)
        self.assertIsNone(gates.accuracy)
        self.assertIsNone(gates.public_rules)
        self.assertFalse(gates.model_fit_eligible)

    def test_demand_capability_and_lateral_diagnostics_are_distinct(self):
        class Corner(StraightGeometry):
            def evaluate(self, s):
                return {**super().evaluate(s), 'curvature_1pm': .015}
        params = replace(self.params, independent_replay_braking=True, replay_brake_gain=1.)
        _, info = vehicle_step(IntegrationState(0., 30.), WheelCommand(brake_fraction=1.),
            Corner(), params, .02)
        self.assertEqual(info['status'], 'accepted')
        self.assertLessEqual(info['lateral_utilization'], 1.)
        self.assertGreater(abs(info['requested_wheel_force_n']), info['remaining_longitudinal_capacity_n'])
        self.assertLessEqual(abs(info['wheel_force_n']), info['remaining_longitudinal_capacity_n'])
        self.assertEqual(info['force_truncation_reason'], 'remaining_joint_tire_budget')


class IndependentGateTests(unittest.TestCase):
    def test_complete_cold_lap_can_fit_despite_control_closure_failure(self):
        gate = GateReport('autonomous', None, True, True, True,
            accuracy=False, per_lap_energy=False, endpoint_repeatability=False)
        self.assertTrue(gate.model_fit_eligible)
        self.assertFalse(gate.single_lap_control_qualified)
        self.assertFalse(gate.control_qualified)
        self.assertTrue(model_fit_eligible(dict(complete=True, physical=True,
            public_rules_passed=True, autonomous_control_qualified=False)))

    def test_public_rule_missing_or_false_cannot_enter_fit(self):
        for public in (None, False):
            gate = GateReport('autonomous', None, True, True, public)
            self.assertFalse(gate.model_fit_eligible)
            self.assertFalse(model_fit_eligible(dict(complete=True, physical=True,
                public_rules_passed=public)))
        self.assertFalse(model_fit_eligible(dict(complete=True, physical=True)))

    def test_replay_rule_gate_is_inapplicable_and_control_never_qualified(self):
        gate = GateReport('replay', True, True, True, None, accuracy=True)
        self.assertTrue(gate.model_fit_eligible)
        self.assertFalse(gate.control_qualified)
        self.assertIsNone(gate.to_dict()['public_rules'])
        with self.assertRaises(ValueError):
            GateReport('replay', True, True, True, True)

    def test_single_lap_closure_does_not_assert_six_carry_continuity(self):
        gate = GateReport('autonomous', None, True, True, True,
            per_lap_energy=True, endpoint_repeatability=True)
        self.assertTrue(gate.single_lap_control_qualified)
        self.assertFalse(gate.control_qualified)
        self.assertTrue(replace(gate, six_carry_continuity=True).control_qualified)
        self.assertFalse(replace(gate, six_carry_continuity=False).control_qualified)

    def test_fixed_screen_uses_public_physics_not_cold_control_gate(self):
        count = 0
        def evaluator(values, key):
            nonlocal count
            count += 1
            return dict(complete=True, physical=True, public_rules_passed=count<=3,
                autonomous_control_qualified=False, objective=1.)
        report = sobol_viability(evaluator)
        self.assertEqual(report['viable_points'], 3)
        self.assertTrue(report['fit_allowed'])
        self.assertEqual([r['index'] for r in report['starts']], [1, 2, 3])
        self.assertEqual(len(report['records']), 64)

    def test_replay_refinement_preserves_and_hashes_raw_coverage(self):
        sim = SimulatorV4.__new__(SimulatorV4)
        sim.geometry,sim.length_m = StraightGeometry(),100.
        sim.vehicle = VehicleParamsV4(rolling_coefficient=0.,drag_area_m2=0.,downforce_area_m2=0.)
        sim.config = IntegratorConfig()
        trace = replay_trace(.1,coverage_start_s=0.,coverage_end_s=.025,
            coverage_provenance={'source':'original raw interval'})
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            report=replay_numerical_audit(sim,trace,folder/'first')
            self.assertFalse(report['passed'])
            self.assertTrue(report['raw_coverage_frozen'])
            self.assertEqual(report['raw_coverage']['coverage_end_s'],.025)
            for summaries in report['summaries'].values():
                for summary in summaries:
                    self.assertEqual(summary['status'],'input_exhausted')
                    self.assertEqual(summary['lap_time_s'],.025)
            changed={**trace,'coverage_provenance':{'source':'different receipt'}}
            again=replay_numerical_audit(sim,changed,folder/'second')
            self.assertNotEqual(report['raw_time_channels_sha256'],again['raw_time_channels_sha256'])
        self.assertEqual(trace['coverage_provenance']['source'],'original raw interval')

    def test_model_convergence_does_not_reinstate_cold_control_closure(self):
        theta=np.linspace(0.,2*np.pi,40,endpoint=False)
        route=ClosedRoute(np.c_[10*np.cos(theta),10*np.sin(theta)])
        rules={'soc_window_mj':4.,'ers_max_power_kw':350.,'harvest_max_mj_per_lap':8.,
            'alternative_power_zones_route_m':[],'power_reduction_exceptions_route':[],
            'fia_centreline_length_m':route.source_length,
            'fia_sector_lengths_m':[route.source_length/3]*3}
        sim=ResearchSimulator(route.sample(5.),rules,rule_config=RuleConfig(tail_guard_m=1000.),
            margin_s=1.,energy_notice_s=.5)
        baseline=sim.simulate(lambda obs:0.,initial_speed_mps=5.)
        self.assertTrue(baseline['summary']['completed_lap'])
        self.assertTrue(baseline['summary']['execution_feasible'])
        self.assertTrue(baseline['summary']['public_rules_passed'])
        self.assertFalse(baseline['summary']['periodic_speed_closed'])
        self.assertFalse(baseline['summary']['feasible'])
        with tempfile.TemporaryDirectory() as name:
            report=autonomous_physical_numerical_audit(sim,baseline,Path(name)/'model')
            self.assertTrue(all(row['valid'] for rows in report['comparisons'].values() for row in rows))
            self.assertTrue(all(row['passed'] for row in report['comparisons']['integration']))
            # This short fixture accelerates steeply from its cold start: its
            # coarse output sampling really exceeds the unchanged threshold.
            # Distinguish that error from cold control closure, and do not
            # promote the entire numerical audit merely because physics passed.
            self.assertFalse(report['passed'])
            self.assertGreater(report['comparisons']['output'][0]['speed_rmse_kmh'],.5)
            self.assertFalse(report['cold_lap_control_closure_used_for_numerical_eligibility'])
            self.assertFalse(report['control_qualification_granted'])
            self.assertTrue(all(not summary['feasible'] for rows in report['summaries'].values()
                for summary in rows))


if __name__ == '__main__':
    unittest.main()
