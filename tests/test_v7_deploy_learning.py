"""Focused synthetic tests; no racing vehicle or training experiment needed."""
from pathlib import Path
import inspect
import json
import numpy as np
import unittest
import tempfile

from f1ers.learning import clone_behaviour
from f1ers_v7.deploy_only import FEATURES, PROTOCOL
from f1ers_v7.learning import _bind, save_policy, load_policy
from f1ers_v7.deploy_learning import analytical_zero_policy, require_exact_zero_mean, train_sac
from f1ers_v7.deployment_experiment import episode_counts, prior_episode_budget, validate_config, ConstantZeroController



class DeployLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp_path = Path(self.temp.name)

    def test_legacy_zero_budget_requires_withheld_and_absent_training_evidence(self):
        (self.tmp_path/'rounds.json').write_text('[{"round":"S5"},{"round":"S6"}]')
        receipt = {'training_status': 'withheld_no_s1_physical_teacher'}
        self.assertEqual(prior_episode_budget(self.tmp_path, receipt)['episodes_completed'], 0)
        (self.tmp_path/'sac_history.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            prior_episode_budget(self.tmp_path, receipt)

    def test_missing_unknown_or_partial_budget_never_defaults_to_zero(self):
        for receipt in ({}, {'training_status':'completed'}, {'actual_new_SAC_episodes_requested':0}):
            with self.assertRaises(ValueError):
                prior_episode_budget(self.tmp_path, receipt)

    def test_explicit_budget_requires_bounded_integer_counts(self):
        valid={'actual_new_SAC_episodes_requested':120, 'actual_new_SAC_episodes_completed':17}
        self.assertEqual(prior_episode_budget(self.tmp_path, valid)['episodes_completed'],17)
        for requested,completed in ((0,1),(121,120),(20,-1),(True,0),(20,1.5)):
            with self.assertRaises(ValueError):
                prior_episode_budget(self.tmp_path, {'actual_new_SAC_episodes_requested':requested,
                    'actual_new_SAC_episodes_completed':completed})

    def test_analytical_zero_head_survives_supervised_log_std_updates(self):
        states = np.random.default_rng(4).normal(size=(17, len(FEATURES)))
        initial = analytical_zero_policy(states, seed=2026)
        require_exact_zero_mean(initial, states)
        before_std = initial.network.weights[-1][:, 1].copy()
        policy, history = clone_behaviour(states, np.zeros(len(states)), epochs=3, initial_policy=initial)
        _bind(policy, FEATURES, 'mode_aware', PROTOCOL)
        require_exact_zero_mean(policy, states)
        assert all(row['action_rmse'] == 0. for row in history)
        assert np.array_equal(policy.network.weights[-1][:, 0], np.zeros(32))
        assert all(policy(state) == 0. for state in states)
        assert not np.array_equal(policy.network.weights[-1][:, 1], before_std)


    def test_mean_head_contract_rejects_small_error(self):
        policy = analytical_zero_policy(np.zeros((2, len(FEATURES))))
        policy.network.biases[-1][0] = 1e-15
        with self.assertRaises(RuntimeError):
            require_exact_zero_mean(policy)


    def test_schema7_exact_zero_roundtrip(self):
        policy = analytical_zero_policy(np.ones((2, len(FEATURES))))
        path = self.tmp_path/'policy.npz'
        save_policy(policy, path)
        loaded = load_policy(path, expected_features=FEATURES, expected_action_space='mode_aware',
            expected_execution_protocol=PROTOCOL)
        require_exact_zero_mean(loaded)
        assert all(np.array_equal(a, b) for a, b in zip(policy.network.parameters(), loaded.network.parameters()))


    def test_episode_accounting_does_not_count_failed_attempt_as_completed(self):
        for index in (1, 2):
            (self.tmp_path/f'episode_{index:03d}_started.json').write_text('{}')
        from f1ers_v6.learning import _write
        _write(self.tmp_path/'training_episode_001.json.gz', {'result': {'summary': {'completed_lap': True}}})
        (self.tmp_path/'sac_history.json').write_text(json.dumps([{'episode': 0}, {'episode': 1}, {'episode': 2, 'failure': 'test'}]))
        count = episode_counts(self.tmp_path)
        assert count['episodes_attempted'] == 2
        assert count['episodes_completed'] == 1
        assert count['physical_full_laps_completed'] == 1


    def test_D_disallows_periodic_shooting_before_simulator_access(self):
        assert inspect.signature(train_sac).parameters['periodic_evaluation'].default is False
        with self.assertRaisesRegex(ValueError, 'shooting'):
            train_sac(None, None, Path('.'), periodic_evaluation=True)


    def test_constant_zero_controller_raw_action(self):
        assert ConstantZeroController()(np.zeros(len(FEATURES))) == 0.


    def test_D_frozen_budget_and_state_contract(self):
        path = Path(__file__).resolve().parents[1]/'config/v7/deployment.json'
        config = json.loads(path.read_text())
        validate_config(config)
        config['new_SAC_episode_maximum'] = 121
        with self.assertRaisesRegex(ValueError, 'new_SAC_episode_maximum'):
            validate_config(config)
