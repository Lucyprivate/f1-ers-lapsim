import unittest
import numpy as np
from f1ers_v6.metrics import score,sector_energy,legal_time
from f1ers_v6.teacher import BlockController

class MetricsTests(unittest.TestCase):
    def case(self):
        result={'length_m':30.,'summary':{'completed_lap':True,'execution_feasible':True,'feasible':True},
            'internal_trace':{'s_m':np.array([0.,10.,20.,30.]),'time_s':np.array([0.,1.,2.,3.]),
                'speed_mps':np.full(4,20.),'dc_kw':np.array([100.,-200.,100.]),
                'energy_mj':np.array([2.,1.9,2.1,2.]),'mechanical_work_residual_mj':np.zeros(3)}}
        lap={'lap_number':3,'role':'calibration','s_m':np.array([0.,5.,10.,15.,20.,25.]),
            'speed_mps':np.array([19.,19.,18.,18.,17.,17.]),'t_s':np.arange(6)*.5,
            'lap_time_s':3.,'sector_times_s':[1.,1.,1.],'sector_boundaries_m':[10.,20.]}
        return result,{'laps':[lap],'audit':{'data_gate_passed':True}}
    def test_sector_errors_not_hidden_by_total_time(self):
        result,data=self.case(); row=score(result,data)['metrics'][0]
        np.testing.assert_allclose(row['sector_speed_rmse_kmh'],[3.6,7.2,10.8])
        self.assertEqual(row['sector_error_s'],[0.,0.,0.])
    def test_sector_energy_signed_and_conserved(self):
        result,_=self.case(); rows=sector_energy(result,[10.,20.])
        np.testing.assert_allclose([r['net_energy_mj'] for r in rows],[-.1,.2,-.1])
        np.testing.assert_allclose([r['balance_residual_mj'] for r in rows],0.,atol=1e-14)
    def test_partial_not_scored_or_extrapolated(self):
        result,data=self.case(); result['summary']['completed_lap']=False
        self.assertNotIn('speed_rmse_kmh',score(result,data)['metrics'][0])
        self.assertFalse(score(result,data)['accuracy_passed'])
    def test_bad_data_cannot_release(self):
        result,data=self.case(); data['audit']['data_gate_passed']=False
        self.assertFalse(score(result,data)['accuracy_passed'])
    def test_inherited_sector_boundary_fallback_uses_frozen_time_mapping(self):
        result,data=self.case(); del data['laps'][0]['sector_boundaries_m']
        np.testing.assert_allclose(score(result,data)['metrics'][0]['sector_speed_rmse_kmh'],[3.6,7.2,10.8])
    def test_missing_hard_certificate_cannot_win(self):
        result,_=self.case(); self.assertEqual(legal_time(result),float('inf'))
    def test_block_policy_uses_wheel_braking_not_deceleration(self):
        obs=np.zeros(31); obs[0]=.2
        self.assertEqual(BlockController([.7])(obs),.7)
        obs[11]=1.; self.assertEqual(BlockController([.7])(obs),-1.)
        self.assertEqual(BlockController([.7],'mode_aware')(obs),1.)

if __name__=='__main__':unittest.main()
