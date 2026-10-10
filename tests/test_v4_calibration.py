"""Guard precommitted data roles, gate completeness and exact Sobol budget."""
import unittest
import numpy as np
from scipy.stats import qmc
from f1ers_v4.calibration import fitting_view, BOUNDS, BUDGET, PARAMETERS, parameter_sensitivity
from f1ers_v4.cli import release_gates


class CalibrationV4Tests(unittest.TestCase):
    def test_fitting_view_strips_development_and_historical(self):
        data={"laps":[{"role":r} for r in ("calibration","development","test","historical_diagnostic")]}
        self.assertEqual(fitting_view(data)["laps"],[{"role":"calibration"}])
        with self.assertRaises(ValueError):fitting_view(data,"test")

    def test_sensitivity_rejects_held_out_before_simulation(self):
        with self.assertRaises(ValueError):parameter_sensitivity({"laps":[{"role":"test"}]},None)

    def test_preregistered_five_parameter_sobol_and_bounds(self):
        points=qmc.scale(qmc.Sobol(5,scramble=True,seed=BUDGET["screening_seed"]).random_base2(6),BOUNDS[:,0],BOUNDS[:,1])
        self.assertEqual(points.shape,(64,5));self.assertEqual(len(PARAMETERS),5)
        self.assertTrue(np.all(points>=BOUNDS[:,0]) and np.all(points<=BOUNDS[:,1]))
        self.assertEqual(tuple(BOUNDS[-1]),(.1,1.))

    def test_missing_data_or_physical_evidence_cannot_release(self):
        report={"solver_success":True,"physical_valid":True};dev={"physical_valid":True,"passed":True}
        self.assertFalse(release_gates(report,{"passed":True},dev)["physical"])
        self.assertFalse(release_gates(report,{"passed":True,"physical_passed":True},dev)["data"])
        self.assertTrue(all(release_gates(report,{"passed":True,"physical_passed":True},dev,True).values()))

if __name__=="__main__":unittest.main()
