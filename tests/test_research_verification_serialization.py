"""Exact persisted replay equality; fixtures never run a vehicle or optimizer."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from f1ers_v7.research import experiment


class ResearchReplaySerializationTests(unittest.TestCase):
    def _verify_fixture(self, change=None):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            root=base/'source';root.mkdir()
            output=root/'results'/'fixture';(output/'baseline').mkdir(parents=True)
            ledger=base/'ledger';(ledger/experiment.IDENTITY).mkdir(parents=True)
            source_bytes=b'# immutable synthetic fixture\n'
            (root/'fixture.py').write_bytes(source_bytes)
            frozen={'code_commit':'synthetic-fixture-commit',
                'tracked_sha256':{'fixture.py':hashlib.sha256(source_bytes).hexdigest()},
                'data_sha256':{'synthetic-data':'synthetic-hash'},
                'dependencies':{'numpy':'synthetic-runtime'},
                'python':{'version':'synthetic-python'},
                'config':{'cold_initial_speed_mps':50.}}
            expected={'passed':True,'formal_objective_s':70.44079066181877,
                'lap_summaries':[{'feature_schema':['speed','energy'],
                    'integrator_config':{'dt_max':.02},
                    'validation_scope':'public-subset',
                    'unverified':['DOC-058']}],
                'within_lap_velocity_or_energy_resets':0,
                'policy_specific_boundary_shooting':False}
            qualification=output/'baseline'/'qualification.json'
            qualification.write_text(json.dumps(expected),encoding='utf-8')
            (output/'freeze.json').write_text(json.dumps(frozen),encoding='utf-8')
            (ledger/experiment.IDENTITY/'run_claim.json').write_text(json.dumps(frozen),encoding='utf-8')
            manifest={'files':{'baseline/qualification.json':experiment.digest(qualification),
                'freeze.json':experiment.digest(output/'freeze.json')}}
            (output/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            replay=copy.deepcopy(expected)
            replay['lap_summaries'][0]['feature_schema']=('speed','energy')
            if change is not None:change(replay)
            # This is the mismatch that old in-memory comparison mishandled.
            self.assertNotEqual(replay,expected)
            def saved_qualification(sim,controller,path,initial_speed):
                self.assertEqual(initial_speed,50.)
                experiment._write(Path(path)/'qualification.json',replay)
                return replay,[]
            with mock.patch.object(experiment,'CANONICAL_RESEARCH_LEDGER',ledger), \
                 mock.patch.object(experiment,'input_files',return_value=frozen['data_sha256']), \
                 mock.patch.object(experiment,'dependency_identity',return_value=frozen['dependencies']), \
                 mock.patch.object(experiment,'python_identity',return_value=frozen['python']), \
                 mock.patch.object(experiment,'load_historical',return_value={'role':'calibration'}), \
                 mock.patch.object(experiment,'make_sim',return_value=object()), \
                 mock.patch.object(experiment,'qualify',side_effect=saved_qualification), \
                 mock.patch('subprocess.check_output',return_value=source_bytes):
                report=experiment.verify_saved(root,base/'private-data',output,replay=True)
            self.assertTrue(report['runtime_identity_equal'])
            self.assertTrue(report['computational_source_hashes_equal'])
            self.assertTrue(report['frozen_git_tree_equal'])
            self.assertTrue(report['canonical_run_claim_equal'])
            self.assertFalse(report['optimization_and_training_repeated'])
            return report

    def test_tuple_feature_schema_equals_persisted_list_without_relaxation(self):
        report=self._verify_fixture()
        self.assertTrue(report['passed'])
        self.assertTrue(report['baseline_replay']['passed'])
        self.assertEqual(report['baseline_replay']['formal_objective_s'],70.44079066181877)

    def test_single_float_ulp_difference_still_fails_exact_comparison(self):
        def change(row):
            row['lap_summaries'][0]['integrator_config']['dt_max']=float(np.nextafter(.02,1.))
        report=self._verify_fixture(change)
        self.assertFalse(report['passed'])
        self.assertFalse(report['baseline_replay']['passed'])

    def test_validation_metadata_difference_is_not_removed(self):
        def change(row):row['lap_summaries'][0]['validation_scope']='different-scope'
        report=self._verify_fixture(change)
        self.assertFalse(report['passed'])
        self.assertFalse(report['baseline_replay']['passed'])


if __name__ == '__main__':unittest.main()
