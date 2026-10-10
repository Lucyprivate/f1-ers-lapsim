import tempfile
from pathlib import Path
import unittest
import numpy as np
from f1ers_v7.research.protocol import EpisodeBudget
from f1ers_v7.research.data import require_role
from f1ers_v7.research.teacher import BlockController, deployment_coefficients


class ResearchProtocolTests(unittest.TestCase):
    def test_episode_attempt_survives_new_instance_and_duplicate_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            first=EpisodeBudget(directory)
            first.claim('numpy',2026,'cold',1)
            second=EpisodeBudget(directory)
            self.assertEqual(second.counts()['attempted'],1)
            with self.assertRaises(FileExistsError): second.claim('numpy',2026,'cold',1)
            second.claim('sb3',2026,'cold',1)
            self.assertEqual(first.counts()['per_backend'],{'numpy':1,'sb3':1})

    def test_budget_rejects_outside_frozen_case(self):
        with tempfile.TemporaryDirectory() as directory:
            budget=EpisodeBudget(directory)
            for arguments in [('other',2026,'cold',1),('numpy',2029,'cold',1),
                              ('numpy',2026,'hot',1),('numpy',2026,'cold',11)]:
                with self.assertRaises(ValueError): budget.claim(*arguments)
            self.assertEqual(budget.counts()['attempted'],0)

    def test_calibration_view_rejects_mixed_roles(self):
        with self.assertRaises(ValueError):
            require_role({'audit':{'role':'calibration'},'laps':[{'role':'development'}]},'calibration')

    def test_action_grid_uses_only_position_feature(self):
        controller=BlockController([-.8,.6])
        observation=np.zeros(36);observation[0]=.49
        self.assertEqual(controller(observation),-.8)
        observation[3:]=999
        self.assertEqual(controller(observation),-.8)
        observation[0]=.5
        self.assertEqual(controller(observation),.6)
        with self.assertRaises(ValueError): BlockController([np.nan])

    def test_surrogate_reproduces_own_deploy_integrals_at_each_start(self):
        deploy=np.array([.2,0.,.04]);harvest=np.array([0.,.3,0.])
        for start in (0.,-.25,.25):
            fraction,coefficient=deployment_coefficients(deploy,np.full(3,start))
            np.testing.assert_allclose(coefficient*fraction,deploy,rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(-coefficient*fraction+harvest,-deploy+harvest)
            np.testing.assert_array_equal(coefficient*(fraction-fraction),np.zeros(3))
        with self.assertRaises(ValueError): deployment_coefficients([.1],[-1.])


if __name__ == '__main__': unittest.main()
