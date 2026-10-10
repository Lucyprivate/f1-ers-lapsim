"""SI work and local authority regressions; no solver or vehicle execution."""
import copy
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from f1ers_v7.research.teacher import (casadi_proposal,
    surrogate_action_authority, surrogate_speed_squared_coefficients)


class TeacherDimensionalTests(unittest.TestCase):
    def test_mechanical_work_sets_velocity_scale_independently(self):
        # First-law oracle: kinetic-energy change must equal efficient DC work.
        # It intentionally does not repeat the helper's power/distance formula.
        mass, speed, efficiency, fraction_change = 800., 50., .95, .2
        dc_energy_mj = np.array([.25,.5,1.])
        duration = np.array([2.,4.,8.])
        response = surrogate_speed_squared_coefficients(dc_energy_mj,duration,
            speed*duration,np.full(4,speed),mass,efficiency)
        for energy, coefficient in zip(dc_energy_mj,response):
            final_speed = np.sqrt(speed**2+coefficient*fraction_change)
            kinetic_before = .5*mass*speed**2
            kinetic_after = .5*mass*final_speed**2
            expected_mechanical_work_j = efficiency*energy*fraction_change*1_000_000.
            self.assertAlmostEqual(kinetic_after-kinetic_before,
                expected_mechanical_work_j,delta=1e-8)

    def test_zero_response_normal_columns_have_no_useful_dimension(self):
        audit=surrogate_action_authority([0.,0.,0.],[['NORMAL']]*3)
        self.assertEqual(audit['free_blocks'],[1,2,3])
        self.assertEqual(audit['responsive_free_columns'],0)
        self.assertEqual(audit['terminal_speed_equality_rank'],0)
        self.assertEqual(audit['useful_free_action_dimension'],0)

    def test_one_responsive_free_column_is_fixed_by_endpoint(self):
        audit=surrogate_action_authority([0.,10.,20.],
            [['NORMAL'],['NORMAL'],['NORMAL','HOLD']])
        self.assertEqual(audit['pinned_blocks'],[3])
        self.assertEqual(audit['responsive_free_blocks'],[2])
        self.assertEqual(audit['terminal_speed_equality_rank'],1)
        self.assertEqual(audit['useful_free_action_dimension'],0)
        self.assertFalse(audit['six_carry_optimality_proven'])

    def test_two_responsive_columns_survive_endpoint_constraint(self):
        audit=surrogate_action_authority([10.,20.,30.,0.],
            [['NORMAL'],['NORMAL'],['LIMITED'],['NORMAL']])
        self.assertEqual(audit['responsive_free_blocks'],[1,2])
        self.assertEqual(audit['zero_response_free_blocks'],[4])
        self.assertEqual(audit['useful_free_action_dimension'],1)

    def test_nonconstant_redistribution_changes_interior_motion_without_endpoint_shift(self):
        coefficient=np.array([.25,.5])
        dt=np.array([2.,4.]);ds=50.*dt;vref=np.full(3,50.)
        response=surrogate_speed_squared_coefficients(coefficient,dt,ds,vref,800.,.95)
        # Removing early work and adding it later preserves total mechanical
        # energy, while changing interior speed and the modeled traversal time.
        fraction_change=np.array([-.02,.01])
        perturbation=np.r_[0.,np.cumsum(response*fraction_change)]
        velocity=np.sqrt(vref**2+perturbation)
        self.assertAlmostEqual(velocity[-1],vref[-1],places=12)
        self.assertLess(velocity[1],vref[1])
        self.assertAlmostEqual(float(coefficient@fraction_change),0.,places=14)
        duration_baseline=float(np.sum(2.*ds/(vref[:-1]+vref[1:])))
        duration_redistributed=float(np.sum(2.*ds/(velocity[:-1]+velocity[1:])))
        self.assertGreater(duration_redistributed,duration_baseline)
        self.assertEqual(surrogate_action_authority(response,
            [['NORMAL'],['NORMAL']])['useful_free_action_dimension'],1)

    def test_invalid_dimensional_inputs_are_rejected(self):
        for coefficients,dt,ds,vref,mass,eta in [
                ([-1.],[1.],[10.],[10.,10.],800.,.95),
                ([1.],[0.],[10.],[10.,10.],800.,.95),
                ([1.],[1.],[10.],[10.],800.,.95),
                ([1.],[1.],[10.],[10.,10.],0.,.95),
                ([1.],[1.],[10.],[10.,10.],800.,1.1)]:
            with self.subTest(coefficients=coefficients,dt=dt,mass=mass,eta=eta):
                with self.assertRaises(ValueError):
                    surrogate_speed_squared_coefficients(coefficients,dt,ds,vref,mass,eta)

    def test_unrecorded_phase_schedule_is_not_free_authority(self):
        with self.assertRaises(ValueError):surrogate_action_authority([1.],[[]])

    @staticmethod
    def _fixture(dc_kw,phases):
        sim=SimpleNamespace(length_m=500.,
            vehicle=SimpleNamespace(mass_kg=800.,deploy_efficiency=.95))
        state=SimpleNamespace(speed_mps=50.,energy_mj=2.)
        baseline={'internal_trace':{'s_m':[0.,250.,500.],
            'time_s':[0.,5.,10.],'speed_mps':[50.,50.,50.],
            'dc_kw':dc_kw,'dt_s':[5.,5.],'energy_mj':[2.,2.,2.]},
            'energy_viability_audit':[{'s_m':s,
                'state_after_decide':{'phase':phase}}
                for s,phase in zip([0.,250.],phases)]}
        return sim,baseline,state

    def _assert_withheld_without_solver(self,dc_kw,phases,expected_columns):
        sim,baseline,state=self._fixture(dc_kw,phases)
        before=copy.deepcopy(baseline)
        fake=types.ModuleType('casadi')
        fake.Opti=mock.Mock(side_effect=AssertionError('A zero-authority solver must not start'))
        with mock.patch.dict(sys.modules,{'casadi':fake}):
            proposal,status=casadi_proposal(sim,baseline,state,2,[0.,0.])
        self.assertIsNone(proposal)
        self.assertEqual(status['status'],'withheld_zero_action_authority')
        self.assertFalse(status['solver_invoked'])
        self.assertEqual(status['action_authority']['responsive_free_columns'],expected_columns)
        self.assertEqual(status['action_authority']['useful_free_action_dimension'],0)
        self.assertEqual(status['MJ_to_J'],1_000_000.)
        self.assertEqual(status['mechanical_power_unit'],'W')
        self.assertEqual(baseline,before)
        fake.Opti.assert_not_called()

    def test_zero_energy_response_is_withheld_before_casadi(self):
        self._assert_withheld_without_solver([0.,0.],['NORMAL','NORMAL'],0)

    def test_one_responsive_column_is_withheld_before_casadi(self):
        self._assert_withheld_without_solver([100.,0.],['NORMAL','NORMAL'],1)

    def test_pinned_power_does_not_count_as_free_authority(self):
        self._assert_withheld_without_solver([100.,100.],['HOLD','LIMITED'],0)


if __name__ == '__main__':unittest.main()
