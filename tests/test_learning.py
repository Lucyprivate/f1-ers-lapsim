"""Numerical and lifecycle tests for the dependency-free learning implementation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from f1ers.learning import (
    Adam, GaussianPolicy, MLP, ReplayBuffer, SAC, clone_behaviour,
    evaluate_policy, load_policy, train_and_evaluate,
)


class LearningTests(unittest.TestCase):
    def test_mlp_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(12)
        network = MLP(3, 2, hidden=(4,), rng=rng)
        x = rng.normal(size=(5, 3))
        target = rng.normal(size=(5, 2))
        output, cache = network.forward(x, cache=True)
        input_grad, gradients = network.backward(cache, 2 * (output - target) / output.size)

        def loss():
            return float(np.mean((network.forward(x) - target) ** 2))

        epsilon = 1e-6
        for parameter, gradient in zip(network.parameters(), gradients):
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                upper = loss()
                parameter[index] = original - epsilon
                lower = loss()
                parameter[index] = original
                numerical[index] = (upper - lower) / (2 * epsilon)
            np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)
        original = x[0, 0]
        x[0, 0] = original + epsilon
        upper = loss()
        x[0, 0] = original - epsilon
        lower = loss()
        x[0, 0] = original
        self.assertAlmostEqual(input_grad[0, 0], (upper - lower) / (2 * epsilon), places=7)

    def test_reparameterised_sac_actor_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(13)
        policy = GaussianPolicy(3, hidden=(4,), seed=2)
        sac = SAC(policy, seed=3)
        # Keep Q1 the minimum throughout finite differencing.
        sac.q2.biases[-1][0] += 20
        observations = rng.normal(size=(5, 3))
        gaussian_noise = rng.normal(size=(5, 1))
        _, gradients, _ = sac.actor_loss_and_gradients(observations, gaussian_noise)
        epsilon = 1e-6
        for parameter, gradient in zip(policy.network.parameters(), gradients):
            numerical = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                upper = sac.actor_loss_and_gradients(observations, gaussian_noise)[0]
                parameter[index] = original - epsilon
                lower = sac.actor_loss_and_gradients(observations, gaussian_noise)[0]
                parameter[index] = original
                numerical[index] = (upper - lower) / (2 * epsilon)
            np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)

    def test_behaviour_cloning_learns_state_function(self):
        rng = np.random.default_rng(8)
        observations = rng.uniform(-1, 1, size=(160, 3))
        actions = 0.5 * np.tanh(1.1 * observations[:, 0] - 0.6 * observations[:, 1])
        policy, history = clone_behaviour(observations, actions, epochs=100, hidden=(16, 16))
        self.assertLess(history[-1]["action_rmse"], 0.025)
        self.assertLess(history[-1]["action_rmse"], history[0]["action_rmse"] * 0.3)
        # New states, not teacher indices, must drive the controller.
        held_out = rng.uniform(-1, 1, size=(30, 3))
        expected = 0.5 * np.tanh(1.1 * held_out[:, 0] - 0.6 * held_out[:, 1])
        predicted = np.asarray([policy(s) for s in held_out])
        self.assertLess(float(np.sqrt(np.mean((predicted - expected) ** 2))), 0.04)

    def test_policy_checkpoint_roundtrip_and_action_probability(self):
        policy = GaussianPolicy(5, hidden=(7, 7), seed=4)
        states = np.random.default_rng(1).normal(size=(20, 5))
        states[0] *= 10000  # Saturation must not make tanh log density non-finite.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.npz"
            policy.save(path)
            restored = load_policy(path)
            np.testing.assert_array_equal([policy(s) for s in states], [restored(s) for s in states])
            actions, log_prob, _ = restored.sample(states, np.random.default_rng(2))
            self.assertTrue(np.all(np.abs(actions) <= 1))
            self.assertTrue(np.isfinite(log_prob).all())

    def test_replay_capacity_sampling_and_requested_action_validation(self):
        replay = ReplayBuffer(4, 3)
        for i in range(7):
            replay.add(np.full(3, i), i / 10, i, np.full(3, i + 1), i == 6)
        self.assertEqual(replay.size, 4)
        sample = replay.sample(100, np.random.default_rng(2))
        self.assertEqual(sample["obs"].shape, (100, 3))
        self.assertGreaterEqual(float(sample["obs"].min()), 3)
        self.assertLessEqual(float(sample["obs"].max()), 6)
        np.testing.assert_array_equal(sample["next_obs"] - sample["obs"], np.ones((100, 3)))
        with self.assertRaises(ValueError):
            replay.add(np.zeros(3), 1.1, 0, np.zeros(3), False)
        with self.assertRaises(ValueError):
            ReplayBuffer(4, 3).sample(1, np.random.default_rng())

    def test_sac_updates_networks_and_reduces_terminal_critic_error(self):
        rng = np.random.default_rng(41)
        policy = GaussianPolicy(3, hidden=(8, 8), seed=6)
        sac = SAC(policy, seed=6, learning_rate=1e-3)
        observations = rng.uniform(-1, 1, size=(64, 3))
        actions = rng.uniform(-1, 1, size=(64, 1))
        rewards = -((actions - 0.4 * observations[:, :1]) ** 2)
        batch = {"obs": observations, "next_obs": observations + 0.01,
                 "actions": actions, "rewards": rewards, "dones": np.ones((64, 1))}
        before_actor = [p.copy() for p in policy.network.parameters()]
        before_target = [p.copy() for p in sac.target_q1.parameters()]
        first = sac.update(batch)
        for _ in range(200):
            last = sac.update(batch)
        self.assertLess(last["q1_loss"], first["q1_loss"] * 0.4)
        self.assertLess(last["q2_loss"], first["q2_loss"] * 0.4)
        self.assertTrue(any(np.any(a != b) for a, b in zip(before_actor, policy.network.parameters())))
        self.assertTrue(any(np.any(a != b) for a, b in zip(before_target, sac.target_q1.parameters())))
        self.assertGreater(sac.alpha, 0)
        self.assertTrue(all(np.isfinite(v) for v in last.values()))

    def test_adam_rejects_nonfinite_gradient_before_mutation(self):
        parameter = np.asarray([1.0])
        optimiser = Adam([parameter])
        with self.assertRaises(FloatingPointError):
            optimiser.step([np.asarray([np.nan])])
        self.assertEqual(parameter[0], 1.0)

    def test_training_delivers_reproducible_feedback_checkpoint(self):
        sim = _FeedbackSimulator()
        teacher = sim.simulate(actions=np.linspace(-0.4, 0.4, sim.n_nodes))
        with tempfile.TemporaryDirectory() as folder:
            report = train_and_evaluate(sim, teacher, folder, seeds=(0,), episodes=2,
                                        bc_epochs=5, evaluate_every=1, critic_warmup_updates=2,
                                        aggregation_rounds=0,
                                        batch_size=4)
            metadata = report["metadata"]
            self.assertEqual(metadata["feasible_seed_count"], 1)
            self.assertGreater(metadata["records"][0]["sac_updates"], 0)
            path = Path(folder).parent / metadata["selected_policy"]
            result = evaluate_policy(sim, load_policy(path))
            self.assertEqual(result["summary"], report["results"]["0"]["selected"]["summary"])
            self.assertTrue((Path(folder) / "training_summary.json").is_file())

    def test_infeasible_learning_does_not_replace_optimizer_baseline(self):
        sim = _FeedbackSimulator()
        teacher = sim.simulate(actions=np.zeros(sim.n_nodes))
        sim.reject_policies = True
        with tempfile.TemporaryDirectory() as folder:
            result = train_and_evaluate(sim, teacher, folder, seeds=(0,), episodes=0,
                                       bc_epochs=2, critic_warmup_updates=0, aggregation_rounds=0)
            self.assertIsNone(result["metadata"]["selected_policy"])
            self.assertEqual(result["metadata"]["feasible_seed_count"], 0)
            self.assertFalse((Path(folder) / "seed_0" / "best_feasible_policy.npz").exists())

    def test_dagger_collects_feedback_states_and_keeps_runtime_policy_numeric(self):
        sim = _FeedbackSimulator()
        teacher = sim.simulate(actions=np.linspace(-0.4, 0.4, sim.n_nodes))
        with tempfile.TemporaryDirectory() as folder:
            result = train_and_evaluate(sim, teacher, folder, seeds=(0,), episodes=0,
                                       bc_epochs=3, critic_warmup_updates=0,
                                       aggregation_rounds=2, aggregation_epochs=5)
            record = result["metadata"]["records"][0]
            self.assertEqual(len(record["aggregation_history"]), 2)
            self.assertEqual([r["samples"] for r in record["aggregation_history"]], [16, 24])
            loaded = load_policy(Path(folder) / "seed_0" / "bc_policy.npz")
            self.assertEqual(loaded.network.sizes[-1], 2)
            self.assertFalse(hasattr(loaded, "teacher_actions"))
            self.assertTrue(-1 <= loaded(np.asarray([0.32, 0.41, 1.0])) <= 1)


class _FeedbackEnvironment:
    def __init__(self, count):
        self.count = count

    def reset(self, seed=None):
        self.index, self.soc = 0, 0.5
        return np.asarray([0.0, self.soc, 1.0])

    def step(self, action):
        self.soc = np.clip(self.soc - 0.01 * action, 0, 1)
        self.index += 1
        observation = np.asarray([self.index / self.count, self.soc, 1.0])
        reward = -(1.0 + 0.1 * (action - 0.2) ** 2)
        done = self.index == self.count
        return observation, reward, done, {"soc": float(self.soc)}


class _FeedbackSimulator:
    n_nodes = 8
    reject_policies = False

    def create_env(self):
        return _FeedbackEnvironment(self.n_nodes)

    def simulate(self, actions=None, policy=None, sustainable=True):
        environment = self.create_env()
        observation = environment.reset()
        observations, requested = [], []
        total = 0
        for i in range(self.n_nodes):
            observations.append(observation.copy())
            action = policy(observation) if policy is not None else float(actions[i])
            requested.append(action)
            observation, reward, done, _ = environment.step(action)
            total -= reward
        return {"observations": np.asarray(observations), "requested_actions": np.asarray(requested),
                "summary": {"lap_time_s": total, "feasible": not (self.reject_policies and policy),
                            "sustainable": not (self.reject_policies and policy)}}


if __name__ == "__main__":
    unittest.main()
