import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from f1ers_v5.calibration import (candidates, candidate_bounds, require_role,
                                 fit_candidate, select_candidate, huber_objective)
from f1ers_v5.transfer import score_cross_session
from f1ers_v5.verification import validate_experiment


class V5PipelineTests(unittest.TestCase):
    def test_role_views_reject_development_fitting_and_mixed_laps(self):
        development = {"laps": [{"lap_number": n, "role": "development"} for n in (7, 9)]}
        require_role(development, "development")
        with self.assertRaises(ValueError): require_role(development, "calibration")
        mixed = {"laps": [{"lap_number": n, "role": "calibration"} for n in (3, 4, 5, 9)]}
        with self.assertRaises(ValueError): require_role(mixed, "calibration")

    def test_conditional_parameters_require_explicit_evidence(self):
        baseline = candidates({"kind": "baseline"}, {"brake_decay_eligible": False, "traction_eligible": False})
        self.assertEqual(len(baseline), 2)
        self.assertTrue(all(len(candidate_bounds(s)[0]) == 5 for s in baseline))
        expanded = candidates({"kind": "shared_positions"}, {"brake_decay_eligible": True, "traction_eligible": True})
        self.assertEqual(len(expanded), 7)
        self.assertEqual(candidate_bounds(expanded[-1])[1][-2:].tolist(), [[.1, 5.], [.05, 1.]])

    def test_no_complete_starts_preserves_64_failures_and_never_optimizes(self):
        dataset = {"laps": [{"lap_number": n, "role": "calibration"} for n in (3, 4, 5, 6)]}
        comparison = {"role": "calibration", "physical_valid": False, "normalized_objective": None,
                      "huber_objective": None, "passed": False,
                      "replay": {"metrics": [], "results": {}}, "autonomous": {"metrics": [], "result": {}}}
        with tempfile.TemporaryDirectory() as folder, patch("f1ers_v5.calibration.compare_paths", return_value=comparison), \
                patch("f1ers_v5.calibration.minimize") as optimize:
            _, report, _ = fit_candidate(dataset, candidates({"kind": "baseline"}, {})[0], folder,
                                         progress=lambda *a, **k: None)
            optimize.assert_not_called()
            self.assertEqual(len(report["screening"]), 64)
            self.assertEqual(len(list((Path(folder) / "failures").glob("*.json.gz"))), 64)
            self.assertFalse(report["solver_success"])
            self.assertTrue(report["diagnostic_selection"])
            self.assertEqual(report["huber_sensitivity"]["multistart"], [])

    def test_development_tie_prefers_fewer_parameters_then_fixed_order(self):
        def row(index, score, count, converged=True, physical=True):
            return {"index": index, "report": {"solver_success": converged, "parameter_order": [None]*count},
                    "development": {"physical_valid": physical, "normalized_objective": score}}
        rows = [row(0, 1., 6), row(1, 1.005, 5), row(2, 1.006, 5), row(3, .1, 5, False)]
        self.assertIs(select_candidate(rows), rows[1])
        self.assertIsNone(select_candidate([row(0, 1, 5, False), row(1, 1, 5, True, False)]))

    def test_partial_replay_never_gets_huber_objective_or_accuracy(self):
        lap = {"lap_number": 3, "s_m": np.array([0., 5.]), "speed_mps": np.array([10., 10.])}
        trace = {"s_m": np.array([0., 5.]), "speed_mps": np.array([12., 14.]), "summary": {"speed_end_mps": 14.}}
        metric = {"valid": True, "lap_error_s": 0., "real_lap_time_s": 1., "sector_error_s": [0., 0., 0.]}
        comparison = {"physical_valid": False, "autonomous": {"result": trace, "metrics": [metric]},
                      "replay": {"results": {"3": {}}, "metrics": [{"valid": False}]}}
        objective = huber_objective({"track": {"length_m": 10.}, "laps": [lap]}, comparison)
        self.assertIsNone(objective)
        self.assertGreater(metric["speed_median_absolute_error_kmh"], 0)
        self.assertNotIn("speed_median_absolute_error_kmh", comparison["replay"]["metrics"][0])

    def test_huber_is_speed_only_and_keeps_all_tail_points(self):
        lap = {"lap_number": 3, "s_m": np.array([0., 5.]), "speed_mps": np.zeros(2)}
        trace = {"s_m": np.array([0., 5.]), "speed_mps": np.array([0., 100./3.6]), "summary": {"speed_end_mps": 100./3.6}}
        metric = {"valid": True, "lap_error_s": .02, "real_lap_time_s": 1., "sector_error_s": [1., 1., 1.]}
        comparison = {"physical_valid": True,
                      "autonomous": {"result": trace, "metrics": [copy.deepcopy(metric)]},
                      "replay": {"results": {"3": trace}, "metrics": [copy.deepcopy(metric)]}}
        value = huber_objective({"track": {"length_m": 10.}, "laps": [lap]}, comparison)
        self.assertAlmostEqual(value, ((2*(100/15)-1)/2+1+1)/3)
        self.assertGreater(comparison["autonomous"]["metrics"][0]["speed_p95_absolute_error_kmh"], 90.)

    def test_failed_local_gate_does_not_touch_fp2_or_ledger(self):
        gates = {"data": True, "convergence": False, "physical": False, "grid": False, "development": False}
        with patch("f1ers_v5.transfer.inherited.prepare_new_session") as load, \
                patch("f1ers_v5.transfer.claim_condition_adaptation") as claim:
            result = score_cross_session(Path("."), Path("."), None, {"kind": "baseline"}, None, gates, "test", False)
            self.assertEqual(result["status"], "pending")
            self.assertFalse(result["query_attempted"])
            load.assert_not_called(); claim.assert_not_called()

    def test_transfer_fit_role_is_bound_to_original_selected_ids(self):
        data = {"laps": [{"lap_number": n, "role": "adaptation"} for n in (14, 15, 16)],
                "audit": {"selection": {"adaptation_laps": [14, 15, 16], "test_laps": [17, 18, 19]}}}
        require_role(data, "adaptation")
        with self.assertRaises(ValueError): require_role(data, "test")

    def test_report_gate_or_geometry_tampering_is_rejected(self):
        gates = {"data": True, "convergence": False, "physical": False, "grid": False, "development": False}
        candidate = {"name": "baseline_remaining", "geometry": {"kind": "baseline"}, "brake_mapping": "remaining"}
        certificate = {"local_gates": gates, "vehicle": {"mu": 1.8}, "geometry_spec": {"kind": "baseline"},
                       "selected_candidate": candidate, "code_commit": "fixed-commit", "dependencies": {"python": "3.12"}}
        experiment = {"schema_version": 5, "freeze": copy.deepcopy(certificate), "vehicle": {"mu": 1.8},
                      "code_provenance": {"code_commit": "fixed-commit", "dependencies": {"python": "3.12"}},
                      "report": {"fitted_vehicle": {"mu": 1.8}}, "geometry_spec": {"kind": "baseline"},
                      "gates": {**gates, "new_test": False}, "displayed_diagnostic_candidate": "baseline_remaining",
                      "selected_candidate": None, "candidate_plan": [candidate], "cross_session": {"status": "pending"},
                      "training": {"skipped": True, "sac_expanded": False}, "benefit_claim_allowed": False}
        validate_experiment(experiment, certificate, {"candidates": [candidate]})
        for key, value in (("gates", {**gates, "development": True, "new_test": False}),
                           ("geometry_spec", {"kind": "other"}), ("selected_candidate", "baseline_remaining"),
                           ("code_provenance", {"code_commit": "changed-commit", "dependencies": {"python": "3.12"}}),
                           ("code_provenance", {"code_commit": "fixed-commit", "dependencies": {"python": "changed"}})):
            altered = {**experiment, key: value}
            with self.assertRaises(ValueError): validate_experiment(altered, certificate, {"candidates": [candidate]})
        with self.assertRaises(ValueError):
            validate_experiment({**experiment, "gates": {**gates, "new_test": True}}, certificate, {"candidates": [candidate]})

    def test_new_session_final_gate_retains_quality_check(self):
        # Exercise the conditional late branch with quality-failing observations;
        # no real data or external access ledger is used.
        gates = {k: True for k in ("data", "convergence", "physical", "grid", "development")}
        from f1ers_v5.adapters import VehicleParamsV5
        early = {"audit": {"data_gate_passed": True}}
        late = {"audit": {"data_gate_passed": False}}
        complete = {"physical_valid": True, "normalized_objective": 0., "passed": True,
                    "replay": {}, "autonomous": {}}
        from types import SimpleNamespace
        optimum = SimpleNamespace(success=True, x=[1., 1.], nit=1, nfev=1, message="fixture")
        with tempfile.TemporaryDirectory() as temp:
            with patch("f1ers_v5.transfer.check_freeze"), patch("f1ers_v5.transfer.inherited.prepare_new_session", return_value={"status": "ready"}), \
                 patch("f1ers_v5.transfer.load_view", side_effect=[early, late]), patch("f1ers_v5.transfer.require_role"), \
                 patch("f1ers_v5.transfer.claim_condition_adaptation"), patch("f1ers_v5.transfer.complete_condition_adaptation"), \
                 patch("f1ers_v5.transfer.authorize_test_access"), patch("f1ers_v5.transfer.complete_test_access"), \
                 patch("f1ers_v5.transfer.read_json", return_value={}), patch("f1ers_v5.transfer.freeze", return_value=(Path(temp)/"freeze.json", {})), \
                 patch("f1ers_v5.transfer.minimize", return_value=optimum) as fit, patch("f1ers_v5.transfer.compare_paths", side_effect=[copy.deepcopy(complete), copy.deepcopy(complete)]):
                result = score_cross_session(Path(temp), Path(temp), VehicleParamsV5(equivalent_power_kw=700.), {"kind": "baseline"},
                                             Path(temp)/"freeze.json", gates, "fixture", True)
                self.assertFalse(result["evaluation"]["passed"])
                self.assertFalse(result["test_quality"]["data_gate_passed"])
                self.assertEqual(fit.call_args.kwargs["bounds"].tolist(), [[.85, 1.15], [.85, 1.15]])

    def test_diagnose_cannot_overwrite_an_interrupted_formal_run(self):
        from f1ers_v5.cli import diagnose_command
        for marker in ("frozen_configuration.json", "candidates"):
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "results" / "interrupted"
                output.mkdir(parents=True)
                if marker == "candidates": (output / marker).mkdir()
                else: (output / marker).write_text("{}")
                with patch("f1ers_v5.cli.diagnostic_stage") as stage:
                    with self.assertRaises(RuntimeError): diagnose_command(temp, "interrupted")
                    stage.assert_not_called()


if __name__ == "__main__": unittest.main()
