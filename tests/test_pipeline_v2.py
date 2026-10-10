"""A failed prospective gate must stop training and freeze before test access."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from f1ers import cli_v2
from f1ers.model import VehicleParams

TMP=Path(__file__).resolve().parents[3]/"work"/"test_tmp"
TMP.mkdir(parents=True,exist_ok=True)


class PipelineV2Tests(unittest.TestCase):
    def test_failed_development_never_trains_and_test_opens_after_freeze(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            root=Path(folder);output=root/"results/austria_2026_v2"
            (root/"config").mkdir();(root/"f1ers").mkdir()
            (root/"config/protocol_austria_2026_v2.json").write_text('{}')
            (root/"data/v2").mkdir(parents=True)
            (root/"data/v2/manifest.json").write_text('{}')
            old=root/"results/austria_2026";old.mkdir(parents=True)
            (old/"vehicle.json").write_text(json.dumps(VehicleParams().to_dict()))
            (old/"summary.json").write_text('{"controllers":[]}')
            n=30;s=np.arange(n,dtype=float)
            result={"s_m":s,"speed_mps":np.full(n,10),"dt_s":np.full(n,.2),"summary":{
                "lap_time_s":6,"speed_end_mps":10,"feasible":True,"sustainable":True,"execution_feasible":True}}
            track={"s_m":s,"length_m":30,"curvature_1pm":np.zeros(n)}
            lap={"s_m":s,"speed_mps":np.full(n,10),"t_s":s*.2,"throttle":np.ones(n),"brake":np.zeros(n),
                "lap_time_s":6,"sector_times_s":[2,2,2],"lap_number":7,"role":"development"}
            development={"track":track,"rules":{},"audit":{"gate_passed":True},"laps":[lap]}
            test={**development,"laps":[{**lap,"lap_number":10,"role":"test"}]}
            accesses=[]
            def load(project_root,include_test=True):
                accesses.append(include_test)
                if include_test:self.assertTrue((output/"frozen_configuration.json").exists())
                return test if include_test else development
            class FakeSimulator:
                def __init__(self,*args):
                    self.n_nodes=n;self.protocol="causal_floor_v2"
                    self.config=SimpleNamespace(to_dict=lambda:{})
                    self.boundary_report={"converged":True}
                def simulate(self,actions):return result
            calibration={"ready_for_training":False,"solver_success":True,
                "development_gate":{"passed":False,"reasons":["speed error"]}}
            with patch.object(cli_v2,"PROJECT_ROOT",root),patch("f1ers.data_v2.prepare_data_v2"),\
                    patch("f1ers.data_v2.load_dataset_v2",side_effect=load),patch("f1ers.causal.CausalSimulator",FakeSimulator),\
                    patch.object(cli_v2,"fit_vehicle_v2",return_value=(VehicleParams(),calibration,result)),\
                    patch("f1ers.learning_v2.train_and_evaluate_v2",side_effect=AssertionError("training must be gated")),\
                    patch.object(cli_v2,"render_report_v2"),patch.object(cli_v2,"export_trace_v2"):
                bundle=cli_v2.run_v2(SimpleNamespace(name="austria_2026_v2",refresh=False))
                self.assertEqual(accesses,[False,True])
                with self.assertRaisesRegex(RuntimeError,"already unsealed"):
                    cli_v2.run_v2(SimpleNamespace(name="another_name",refresh=False))
                self.assertEqual(accesses,[False,True])
                self.assertTrue(bundle["training"]["skipped"])
                self.assertFalse(bundle["gates"]["benefit_claim_allowed"])
                with self.assertRaisesRegex(RuntimeError,"already evaluated"):
                    cli_v2.run_v2(SimpleNamespace(name="austria_2026_v2",refresh=False))
                self.assertEqual(accesses,[False,True])


if __name__=="__main__":unittest.main()
