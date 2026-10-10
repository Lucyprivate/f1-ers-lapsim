"""Acceptance checks for the bundled measured dataset, without network calls."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from f1ers.data import load_dataset
from f1ers.cli import validate_data_gate

ROOT=Path(__file__).resolve().parents[1]


class BundledDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset=load_dataset(ROOT)

    def test_measured_dry_laps_are_complete_and_disjoint(self):
        d=self.dataset
        validate_data_gate(d)
        self.assertEqual(d['session']['year'],2026)
        self.assertTrue(d['audit']['gate_passed'])
        train={l['lap_number'] for l in d['laps'] if l['role']=='calibration'}
        valid={l['lap_number'] for l in d['laps'] if l['role']=='validation'}
        self.assertEqual(train,{3,4,5,6})
        self.assertEqual(valid,{7,9})
        for lap in d['laps']:
            self.assertAlmostEqual(sum(lap['sector_times_s']),lap['lap_time_s'],places=3)
            self.assertTrue(np.all(np.diff(lap['t_s'])>=0))
            self.assertTrue(np.all(np.isfinite(lap['speed_mps'])))
        self.assertTrue(all(c['max_rainfall']==0 for c in d['audit']['clean_lap_checks']))

    def test_track_is_closed_uniform_and_within_boundaries(self):
        t=self.dataset['track']
        self.assertTrue(np.allclose(np.diff(t['s_m']),t['length_m']/len(t['s_m'])))
        self.assertGreater(np.min(t['width_left_m']),0)
        self.assertGreater(np.min(t['width_right_m']),0)
        xy=np.c_[t['x_m'],t['y_m']]
        self.assertLess(np.linalg.norm(xy[-1]-xy[0]),1.2*t['ds_m'])
        self.assertTrue(np.all(np.isfinite(t['curvature_1pm'])))

    def test_raw_source_checksums_and_event_rule_version(self):
        manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
        for source in manifest['sources']:
            path=ROOT/source['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),source['sha256'])
        r=self.dataset['rules']
        self.assertEqual(r['technical_issue'],19)
        self.assertEqual(r['event_date'],'2026-06-28')
        self.assertEqual(r['harvest_max_mj_per_lap'],8)
        self.assertFalse(r['full_fia_compliance_claim_supported'])


if __name__=='__main__':unittest.main()
