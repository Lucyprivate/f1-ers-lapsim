"""Saved-score verification and ordering must never open late arrays again."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from f1ers_v4.artifacts import sha256
from f1ers_v4.cli import score_cross_session
from f1ers_v4.verification import verify


class PipelineV4Tests(unittest.TestCase):
    def test_failed_local_gates_leave_new_session_unopened(self):
        with patch('f1ers_v4.data.prepare_new_session') as fetch:
            out=score_cross_session(Path('.'),Path('.'),None,None,{"physical":False},'alias',False)
            self.assertFalse(out['test_scored']);fetch.assert_not_called()

    def test_early_quality_failure_never_adapts_or_unseals(self):
        with patch('f1ers_v4.data.prepare_new_session',return_value={"status":"ready"}), \
             patch('f1ers_v4.data.load_new_session',return_value={"audit":{"data_gate_passed":False}}), \
             patch('f1ers_v4.cli.adapt_session') as fit, \
             patch('f1ers_v4.protocol.authorize_test_access') as claim:
            out=score_cross_session(Path('.'),Path('.'),None,None,{"data":True},'alias',False)
            self.assertFalse(out['test_scored']);fit.assert_not_called();claim.assert_not_called()

    def test_evaluated_score_verifies_wrapper_and_bytes_without_late_reload(self):
        cache=Path(__file__).resolve().parents[1]/'.cache';cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            root=Path(tmp);out=root/'results/run';out.mkdir(parents=True)
            comparisons={r:{"role":r,"replay":{"metrics":[]},"autonomous":{"metrics":[]}}
                         for r in ('calibration','development')}
            freeze={"vehicle":{}}
            cross={"status":"evaluated","freeze":{"stage":"cross_session"},"evaluation":{"passed":False}}
            bundle={"freeze_path":"results/run/frozen_configuration.json","freeze":freeze,"vehicle":{},
                    "report":{"fitted_vehicle":{}},"local":comparisons,"cross_session":cross,
                    "gates":{"physical":False},"code_provenance":{"code_commit":"fixed"}}
            for name,value in [('experiment.json',bundle),('frozen_configuration.json',freeze),('test_evaluation.json',cross)]:
                (out/name).write_text(json.dumps(value),encoding='utf-8')
            with gzip.open(out/'trajectories.json.gz','wt',encoding='utf-8') as f:json.dump(comparisons,f)
            with gzip.open(out/'test_trajectories.json.gz','wt',encoding='utf-8') as f:json.dump({"saved":True},f)
            receipt={"score":cross,"result_sha256":sha256(out/'test_evaluation.json'),
                     "trajectory_sha256":sha256(out/'test_trajectories.json.gz')}
            with patch('f1ers_v4.verification.check_freeze') as check, \
                 patch('f1ers_v4.data.load_local_dataset',return_value={}), \
                 patch('f1ers_v4.verification.compare_paths',side_effect=lambda d,v,r:comparisons[r]), \
                 patch('f1ers_v4.protocol.read_saved_test_score',return_value=receipt), \
                 patch('f1ers_v4.data.load_new_session') as late:
                result=verify(root,'run',save=False)
                self.assertTrue(result['exact_local_replay_passed']);late.assert_not_called()
                self.assertEqual(check.call_count,3)
                receipt['trajectory_sha256']='changed'
                with self.assertRaises(ValueError):verify(root,'run',save=False)

if __name__=='__main__':unittest.main()
