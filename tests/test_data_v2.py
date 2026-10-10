"""Offline v2 provenance, chronological holdout and failure-path checks.

These tests do not load the real test NPZ arrays or calculate held-out residuals.
Rebuilding and corruption tests work in a temporary project, preserving the
bundled frozen manifest for the ongoing experiment.
"""
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from f1ers.data import DataUnavailable, _epoch
from f1ers.data_v2 import (PROTOCOL_FILE, _coverage, _select_test_laps,
                           load_dataset_v2, prepare_data_v2)

ROOT = Path(__file__).resolve().parents[1]


class V2DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset_v2(ROOT, include_test=False)
        cls.protocol = cls.dataset["protocol"]
        cls.manifest = json.loads((ROOT / "data/v2/manifest.json").read_text(encoding="utf-8"))

    def _temporary_project(self):
        directory = tempfile.TemporaryDirectory(prefix="data-v2-test-", dir=ROOT / "data/v2")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for relative in ("data/raw", "data/processed", "data/v2/raw", "data/v2/processed"):
            shutil.copytree(ROOT / relative, root / relative)
        for relative in ("data/manifest.json", "data/v2/manifest.json", "data/v2/protocol_lock.json",
                         PROTOCOL_FILE, "config/rules_austria_2026.json"):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        return root

    def test_development_loader_never_opens_test_arrays(self):
        actual_load = np.load
        opened = []
        def record_load(path, *args, **kwargs):
            opened.append(Path(path).name)
            return actual_load(path, *args, **kwargs)
        with patch("f1ers.data_v2.np.load", side_effect=record_load):
            dataset = load_dataset_v2(ROOT, include_test=False)
        self.assertEqual([lap["lap_number"] for lap in dataset["laps"]], [3, 4, 5, 6, 7, 9])
        self.assertEqual({lap["role"] for lap in dataset["laps"]}, {"calibration", "development"})
        self.assertFalse(dataset["audit"]["test_arrays_loaded"])
        for number in dataset["audit"]["selected_test_lap_numbers"]:
            self.assertNotIn(f"lap_{number:02d}.npz", opened)
        # Default API can expose all roles for the final, already-frozen scoring.
        self.assertTrue(inspect.signature(load_dataset_v2).parameters["include_test"].default)

    def test_protocol_precedes_acquisition_and_checksums_match(self):
        lock = json.loads((ROOT / "data/v2/protocol_lock.json").read_text(encoding="utf-8"))
        self.assertTrue(lock["frozen_before_new_telemetry_download"])
        self.assertFalse(self.manifest["test_metrics_computed"])
        self.assertEqual(lock["protocol_sha256"], hashlib.sha256((ROOT / PROTOCOL_FILE).read_bytes()).hexdigest())
        for source in self.manifest["sources"]:
            self.assertLess(_epoch(lock["frozen_at_utc"]), _epoch(source["downloaded_at_utc"]))
        for record in self.manifest["sources"] + self.manifest["processed"] + lock["frozen_inputs"]:
            self.assertEqual(record["sha256"], hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest())
        self.assertEqual(lock["calibration_geometry_fit_laps"], [3, 4, 5, 6])

    def test_geometry_and_aero_are_immutable_calibration_artifacts(self):
        legacy = json.loads((ROOT / "data/processed/dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(self.dataset["audit"]["geometry_fit_laps"], [3, 4, 5, 6])
        self.assertEqual(self.dataset["track"]["approximate_aero_zones_m"], legacy["track"]["approximate_aero_zones_m"])
        for original, inherited in (("data/processed/track.npz", "data/v2/processed/track.npz"),
                                    ("config/rules_austria_2026.json", "data/v2/processed/rules.json")):
            self.assertEqual((ROOT / original).read_bytes(), (ROOT / inherited).read_bytes())
        self.assertFalse(self.dataset["audit"]["observed_soc_available"])
        self.assertFalse(self.dataset["audit"]["full_fia_compliance_claim_supported"])

    def test_selected_test_identity_quality_and_equal_count(self):
        audit = self.dataset["audit"]
        self.assertEqual(audit["selected_test_lap_numbers"], [10, 11, 12])
        self.assertEqual(audit["test_substitutions"], [])
        selected = set(audit["selected_test_lap_numbers"])
        fixed = {lap["lap_number"] for lap in self.dataset["laps"]}
        self.assertFalse(selected & fixed)
        for entry in audit["candidate_quality_checks"]:
            if entry["lap_number"] in selected:
                self.assertTrue(entry["passed"])
                self.assertEqual(entry["compound"], "MEDIUM")
                self.assertEqual(entry["max_rainfall"], 0)
        for coverage in audit["new_telemetry_coverage"]:
            if coverage["lap_number"] in selected:
                self.assertGreaterEqual(coverage["car_samples"], 200)
                self.assertGreaterEqual(coverage["location_samples"], 200)
                self.assertLessEqual(coverage["max_car_gap_s"], 2)
                self.assertLessEqual(coverage["max_location_gap_s"], 2)

    def test_gap_gate_rejects_large_hole_despite_sufficient_sample_count(self):
        lap = dict(lap_number=10, date_start="2026-06-28T13:14:11Z", lap_duration=72)
        start = _epoch(lap["date_start"])
        times = np.arange(start - 1, start + 74, .2)
        hole = times[(times < start + 20) | (times > start + 24)]
        quality = self.protocol["data_quality"]
        for key, reason in (("car_t", "excessive_car_gap"), ("loc_t", "excessive_location_gap")):
            arrays = dict(car_t=times, loc_t=times)
            arrays[key] = hole
            report = _coverage(lap, arrays, quality)
            self.assertFalse(report["passed"])
            self.assertIn(reason, report["reasons"])
            self.assertGreaterEqual(report["car_samples"], 200)
            self.assertGreaterEqual(report["location_samples"], 200)

    def test_replacement_uses_only_quality_and_order_or_fails(self):
        numbers = self.protocol["primary_test_laps"] + self.protocol["replacement_test_laps_in_order"]
        statuses = {number: dict(passed=True, reasons=[]) for number in numbers}
        statuses[11] = dict(passed=False, reasons=["excessive_location_gap"])
        statuses[13] = dict(passed=False, reasons=["yellow_red_sc_vsc_window"])
        selected, replacements = _select_test_laps(self.protocol, statuses)
        self.assertEqual(selected, [10, 12, 14])
        self.assertEqual(replacements[0]["rejected_primary_lap"], 11)
        self.assertEqual(replacements[0]["replacement_lap"], 14)
        for number in self.protocol["replacement_test_laps_in_order"]:
            statuses[number] = dict(passed=False, reasons=["insufficient_car_samples"])
        with self.assertRaisesRegex(DataUnavailable, "no unused quality-passing replacement"):
            _select_test_laps(self.protocol, statuses)

    def test_cached_rebuild_is_offline_and_preserves_legacy_bytes(self):
        root = self._temporary_project()
        legacy_before = hashlib.sha256((root / "data/manifest.json").read_bytes()).hexdigest()
        processed_before = hashlib.sha256((root / "data/processed/dataset.json").read_bytes()).hexdigest()
        with patch("f1ers.data.urlopen", side_effect=AssertionError("Offline rebuild attempted network")):
            rebuilt = prepare_data_v2(root)
        self.assertEqual(rebuilt["selected_test_laps"], [10, 11, 12])
        self.assertEqual(legacy_before, hashlib.sha256((root / "data/manifest.json").read_bytes()).hexdigest())
        self.assertEqual(processed_before, hashlib.sha256((root / "data/processed/dataset.json").read_bytes()).hexdigest())
        self.assertEqual([lap["lap_number"] for lap in load_dataset_v2(root, include_test=False)["laps"]], [3, 4, 5, 6, 7, 9])

    def test_tampered_source_and_protocol_cannot_be_silently_rebuilt(self):
        root = self._temporary_project()
        source = root / self.manifest["sources"][0]["path"]
        source.write_bytes(source.read_bytes() + b" ")
        with self.assertRaisesRegex(DataUnavailable, "provenance changed"):
            load_dataset_v2(root, include_test=False)
        with self.assertRaisesRegex(DataUnavailable, "provenance changed"):
            prepare_data_v2(root)
        shutil.copyfile(ROOT / self.manifest["sources"][0]["path"], source)
        protocol = root / PROTOCOL_FILE
        protocol.write_text(protocol.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DataUnavailable, "modified after freeze"):
            load_dataset_v2(root, include_test=False)


if __name__ == "__main__":
    unittest.main()
