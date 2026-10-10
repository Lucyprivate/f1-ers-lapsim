"""Small causal fixtures verify v2 protocol, paired ablation and diagnostics."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from f1ers.learning import GaussianPolicy, ReplayBuffer
from f1ers.learning_v2 import (
    DiagnosticSAC, PROTOCOL, _rollout, _state_parity, critic_calibration,
    _summary_rollout,
    _sac_run,
    evaluate_policy_v2, load_policy_v2, save_policy_v2, select_ablation_group,
    train_and_evaluate_v2,
)


class CausalLearningTests(unittest.TestCase):
    def test_checkpoint_schema_roundtrip_and_legacy_rejection(self):
        policy = GaussianPolicy(13, hidden=(8,), seed=3)
        observations = np.random.default_rng(9).normal(size=(7, 13))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path = Path(folder)/"v2.npz"
            save_policy_v2(policy, path)
            restored = load_policy_v2(path)
            np.testing.assert_array_equal([policy(x) for x in observations],
                                          [restored(x) for x in observations])
            legacy = Path(folder)/"legacy.npz"
            GaussianPolicy(12).save(legacy)
            with self.assertRaisesRegex(ValueError, "legacy"):
                load_policy_v2(legacy)
            with self.assertRaises(ValueError):
                save_policy_v2(GaussianPolicy(12), path)
            with np.load(path) as arrays:
                changed = {key: arrays[key].copy() for key in arrays.files}
            changed["protocol"] = np.asarray("legacy_offline_projection")
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, "protocol"):
                load_policy_v2(path)

    def test_fixed_alpha_is_constant_automatic_alpha_updates(self):
        observations = np.random.default_rng(7).normal(size=(16, 13))
        batch = {"obs": observations, "next_obs": observations+0.01,
                 "actions": np.zeros((16, 1)), "rewards": np.zeros((16, 1)),
                 "dones": np.ones((16, 1))}
        fixed = DiagnosticSAC(GaussianPolicy(13, seed=6), seed=6, temperature_mode="fixed")
        automatic = DiagnosticSAC(GaussianPolicy(13, seed=6), seed=6, temperature_mode="auto")
        # Identical initial networks and random streams isolate temperature mode.
        np.testing.assert_array_equal(fixed.q1.weights[0], automatic.q1.weights[0])
        for _ in range(4):
            fm, am = fixed.update(batch), automatic.update(batch)
        self.assertAlmostEqual(fixed.alpha, 0.001, places=12)
        self.assertNotEqual(automatic.alpha, fixed.alpha)
        self.assertEqual(fixed.gamma, 1.0)
        self.assertIn("actor_gradient_norm", fm)
        self.assertTrue(all(np.isfinite(value) for value in am.values()))

    def test_terminal_bellman_target_contains_no_bootstrapped_entropy(self):
        learner = DiagnosticSAC(GaussianPolicy(13), seed=0, temperature_mode="fixed")
        learner.target_q1.biases[-1][0] = 10000
        learner.target_q2.biases[-1][0] = 10000
        batch = {"obs": np.zeros((4, 13)), "next_obs": np.ones((4, 13)),
                 "actions": np.zeros((4, 1)), "rewards": np.full((4, 1), -0.7),
                 "dones": np.ones((4, 1))}
        metrics = learner.update(batch)
        self.assertAlmostEqual(metrics["bellman_target_min"], -0.7)
        self.assertAlmostEqual(metrics["bellman_target_max"], -0.7)

    def test_ablation_expansion_threshold_legality_completeness_and_tie(self):
        records = [_gate_record(seed, mode, 0.06) for mode in ("fixed", "auto") for seed in (0, 1, 2)]
        gate = select_ablation_group(records, (0, 1, 2))
        self.assertEqual(gate["winner"], "fixed")
        broken = deepcopy(records)
        broken[0]["training_failure"] = "nonfinite gradient"
        self.assertEqual(select_ablation_group(broken, (0, 1, 2))["winner"], "auto")
        for record in records:
            record["paired_improvement_over_bc_s"] = 0.049
        self.assertIsNone(select_ablation_group(records, (0, 1, 2))["winner"])
        records[0]["paired_improvement_over_bc_s"] = 1.0
        records[0]["best_summary"]["feasible"] = False
        self.assertIsNone(select_ablation_group(records, (0, 1, 2))["winner"])

    def test_deterministic_causal_step_and_simulate_state_parity(self):
        sim = _CausalFixture()
        policy = GaussianPolicy(13, seed=12)
        official = evaluate_policy_v2(sim, policy)
        self.assertTrue(_state_parity(sim, policy, official, seed=0)["passed"])
        altered = deepcopy(official)
        altered["observations"][1, 5] += 0.1
        self.assertFalse(_state_parity(sim, policy, altered, seed=0)["passed"])

    def test_mc_diagnostic_matches_known_terminal_returns(self):
        learner = DiagnosticSAC(GaussianPolicy(13), seed=0, temperature_mode="fixed")
        for network in (learner.q1, learner.q2):
            for parameter in network.parameters():
                parameter[...] = 0.0
        trajectory = {"observations": np.zeros((3, 13)), "actions": np.zeros(3),
                      "rewards": np.asarray([1., 2., 3.]), "log_probabilities": np.zeros(3)}
        metrics = critic_calibration(learner, trajectory)
        self.assertAlmostEqual(metrics["mc_reward_return_mean"], 14/3)
        self.assertAlmostEqual(metrics["q_minus_soft_mc_mean"], -14/3)
        self.assertAlmostEqual(metrics["q_vs_soft_mc_rmse"], np.sqrt(70/3))

    def test_failed_guard_keeps_terminal_transition_and_reward_diagnostics(self):
        class FailedEnvironment(_FixtureEnvironment):
            def step(self, action):
                observation, reward, done, info = super().step(action)
                info.update(step_certified=False, reward_failure_s=-100.,
                            guard_reasons=["synthetic force-limit rejection"])
                return observation, reward-100., True, info

        class FailedSimulator(_CausalFixture):
            def create_env(self):
                return FailedEnvironment(self.n_nodes)

        replay = ReplayBuffer(20, 13)
        trajectory = _rollout(FailedSimulator(), GaussianPolicy(13), seed=0, replay=replay)
        self.assertFalse(trajectory["completed_lap"])
        self.assertEqual(replay.size, 1)
        self.assertEqual(replay.dones[0, 0], 1.)
        metrics = _summary_rollout(trajectory)
        self.assertAlmostEqual(metrics["reward_decomposition_residual_s"], 0.)
        self.assertEqual(metrics["failure_reward_s"], -100.)
        self.assertEqual(metrics["terminal_info"]["guard_reasons"], ["synthetic force-limit rejection"])

    def test_small_training_preserves_legal_bc_and_exports_reproducible_controllers(self):
        sim = _CausalFixture()
        teacher = sim.simulate(actions=np.zeros(sim.n_nodes))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            models = Path(folder)/"models"
            report = train_and_evaluate_v2(sim, teacher, models, seeds=(0,),
                                           bc_epochs=3, diagnostic_episodes=1, formal_episodes=1)
            metadata = report["metadata"]
            self.assertEqual(metadata["protocol"], PROTOCOL)
            self.assertEqual(len(metadata["ablation_records"]), 2)
            self.assertFalse(metadata["formal_run_performed"])
            self.assertEqual(metadata["actual_training_episode_budget"], 2)
            bc = metadata["bc_records"][0]
            loaded = load_policy_v2(Path(folder)/bc["bc_checkpoint"])
            reproduced = evaluate_policy_v2(sim, loaded)
            self.assertEqual(reproduced["summary"], bc["bc_summary"])
            for record in metadata["ablation_records"]:
                self.assertEqual(record["best_episode"], 0)
                self.assertTrue(record["state_parity"]["passed"])
            self.assertEqual(len(report["controllers"]), 6)
            for controller in report["controllers"]:
                if controller["checkpoint"] is not None:
                    replayed = evaluate_policy_v2(sim, load_policy_v2(Path(folder)/controller["checkpoint"]))
                    self.assertEqual(replayed["summary"], controller["result"]["summary"])
            self.assertTrue((models/"training_summary.json").is_file())
            self.assertFalse(hasattr(loaded, "teacher_actions"))

    def test_old_teacher_schema_blocks_before_writing_training(self):
        sim = _CausalFixture()
        teacher = sim.simulate(actions=np.zeros(sim.n_nodes))
        teacher["observations"] = teacher["observations"][:, :12]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            with self.assertRaisesRegex(ValueError, "13-feature"):
                train_and_evaluate_v2(sim, teacher, Path(folder)/"models", bc_epochs=1,
                                      diagnostic_episodes=1, formal_episodes=1)
            self.assertFalse((Path(folder)/"models").exists())

    def test_nonfinite_actor_failure_saves_incumbent_and_diagnostic_metadata(self):
        sim = _CausalFixture()
        policy = GaussianPolicy(13, seed=9)
        baseline = evaluate_policy_v2(sim, policy)
        trajectory = _rollout(sim, policy, seed=0)

        def corrupt_actor(learner, batch):
            learner.policy.network.weights[0][0, 0] = np.nan
            raise FloatingPointError("injected nonfinite actor")

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            models = Path(folder)/"models"
            directory = models/"seed_0"/"diagnostic_fixed"
            with patch.object(DiagnosticSAC, "update", corrupt_actor):
                metadata, candidates, _, _ = _sac_run(
                    sim, policy, baseline, trajectory, 0, "fixed", 10,
                    directory, models, progress=None)
            self.assertIn("injected nonfinite actor", metadata["training_failure"])
            self.assertIsNone(metadata["last_checkpoint"])
            self.assertFalse(candidates["last"]["summary"]["feasible"])
            self.assertIsNone(candidates["last"]["summary"]["lap_time_s"])
            self.assertEqual(metadata["best_episode"], 0)
            self.assertTrue(metadata["state_parity"]["passed"])
            self.assertTrue((directory/"metadata.json").exists())
            self.assertTrue((directory/"last_candidate_failure.json").exists())
            checkpoint = Path(folder)/metadata["best_checkpoint"]
            self.assertEqual(evaluate_policy_v2(sim, load_policy_v2(checkpoint))["summary"], baseline["summary"])


def _gate_record(seed, mode, improvement):
    return {"seed": seed, "temperature_mode": mode, "episodes_completed": 20,
            "episodes_requested": 20, "training_failure": None,
            "paired_improvement_over_bc_s": improvement,
            "best_summary": {"feasible": True, "sustainable": True,
                             "coupling_converged": True, "periodic_speed_error_mps": 0.}}


class _FixtureEnvironment:
    def __init__(self, count):
        self.count = count

    def _observation(self):
        observation = np.zeros(13)
        observation[0] = self.index/self.count
        observation[3] = 0.8
        observation[5] = self.energy/4.
        observation[6] = 1.
        observation[9] = 1.-self.index/self.count
        observation[12] = (self.energy-2.)/4.
        return observation

    def reset(self, seed=None):
        self.index, self.energy = 0, 2.
        return self._observation()

    def step(self, action):
        shielded = min(float(action), max(0., self.energy-2.))
        self.energy -= shielded*0.01
        dt = 1.+0.1*float(action)**2
        self.index += 1
        done = self.index == self.count
        info = {"dt_s": dt, "energy_mj": self.energy, "recharge_mj": 0.,
                "periodic_speed_error_mps": 0., "step_certified": True,
                "reward_time_s": 1.-dt, "reward_terminal_energy_s": 0.,
                "reward_terminal_speed_s": 0., "requested_action": float(action),
                "shielded_action": shielded, "executed_action": shielded,
                "reserve_reduction_kw": max(0., float(action)-shielded)*350}
        return self._observation(), 1.-dt, done, info


class _CausalFixture:
    n_nodes = 8

    def create_env(self):
        return _FixtureEnvironment(self.n_nodes)

    def simulate(self, actions=None, policy=None):
        environment = self.create_env()
        observation = environment.reset()
        observations, requested, executed = [], [], []
        elapsed = 0.
        for index in range(self.n_nodes):
            observations.append(observation.copy())
            action = policy(observation) if policy is not None else float(actions[index])
            requested.append(action)
            observation, reward, done, info = environment.step(action)
            elapsed += info["dt_s"]
            executed.append(info["executed_action"])
        return {"observations": np.asarray(observations), "requested_actions": np.asarray(requested),
                "executed_actions": np.asarray(executed), "summary": {
                    "protocol": PROTOCOL, "lap_time_s": elapsed, "feasible": True,
                    "sustainable": True, "coupling_converged": True, "projection_alpha": 1.,
                    "energy_end_mj": environment.energy, "periodic_speed_error_mps": 0.,
                    "sustainability_deficit_mj": 0.}}


if __name__ == "__main__":
    unittest.main()
