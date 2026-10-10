"""Reward invariants and versioned learning lifecycle, using a small executor."""
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
import numpy as np

from f1ers.learning import GaussianPolicy
from f1ers_v6.reward import RewardConfig, transition_reward
from f1ers_v6.learning import (DiagnosticSAC, _bind, _rollout, evaluate_policy, is_legal,
                              load_policy, save_policy, monte_carlo_returns,
                              warmstart_critics, train_bc, train_sac, RolloutFailure)


@dataclass(frozen=True)
class State:
    elapsed_time_s: float
    energy_mj: float
    speed_mps: float = 40.0


class ToyEnv:
    observation_dim = 2
    features = ("phase", "energy")

    def __init__(self, sim, reward_config, initial_speed_mps):
        self.sim = sim
        self.config = reward_config or RewardConfig()
        self.initial_speed = 40.0 if initial_speed_mps is None else initial_speed_mps
        self.reset()

    def reset(self, seed=None):
        self.i = 0
        self.state = State(0.0, 2.0, self.initial_speed)
        self.observations, self.actions, self.times = [], [], [0.0]
        self.done = False
        return np.array([0.0, 0.5])

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode requires reset")
        if self.sim.raise_at is not None and self.i == self.sim.raise_at:
            raise RuntimeError("Injected numerical failure")
        before = self.state
        obs = np.array([self.i/self.sim.n_nodes, 0.5])
        dt = 1.0+0.2*(float(action)-0.25)**2
        self.state = State(before.elapsed_time_s+dt, 2.0, self.initial_speed)
        self.i += 1
        self.done = self.i == self.sim.n_nodes
        reward, terms = transition_reward(before, self.state, 1.0, "accepted", self.done,
                                           self.done, self.config, 0.0)
        self.observations.append(obs)
        self.actions.append(action)
        self.times.append(self.state.elapsed_time_s)
        return np.array([self.i/self.sim.n_nodes, 0.5]), reward, self.done, {
            "dt_s": dt, "step_certified": True, "projection_fraction": 0.0,
            "reward_terms": terms, "status": "accepted"}

    def result(self):
        n = len(self.actions)
        return {"observations": np.asarray(self.observations), "requested_actions": np.asarray(self.actions),
                "speed_mps": np.full(n, self.initial_speed), "time_s": np.asarray(self.times),
                "energy_mj": np.full(n+1, 2.0), "summary": {
                    "completed_lap": self.done, "execution_feasible": True, "feasible": self.done,
                    "sustainable": True, "public_rules_passed": True, "periodic_speed_closed": True,
                    "dc_boundary_closed": True, "rule_state_closed": True,
                    "periodic_speed_residual_mps": 0.0, "energy_start_mj": 2.0, "energy_end_mj": 2.0,
                    "speed_start_mps": self.initial_speed, "speed_end_mps": self.initial_speed,
                    "lap_time_s": self.state.elapsed_time_s}}


class ToySimulator:
    features = ToyEnv.features
    observation_dim = 2
    protocol = "synthetic_v6_lifecycle"
    n_nodes = 4
    length_m = 20.0

    def __init__(self):
        self.raise_at = None
        self.periodic_calls = 0
        self.fixed_calls = 0

    def create_env(self, action_space="signed", reward_config=None, initial_speed_mps=None,
                   carried_rule_state=None):
        return ToyEnv(self, reward_config, initial_speed_mps)

    def simulate(self, controller, action_space="signed", reward_config=None, initial_speed_mps=None,
                 carried_rule_state=None):
        self.fixed_calls += 1
        env = self.create_env(action_space, reward_config, initial_speed_mps, carried_rule_state)
        obs = env.reset()
        while not env.done:
            action = controller(obs) if callable(controller) else controller[env.i]
            obs, _, _, _ = env.step(action)
        return env.result()

    def periodic_simulate(self, **kwargs):
        self.periodic_calls += 1
        result = self.simulate(**kwargs)
        result["summary"]["shooting"] = {"converged": True, "full_lap_trials": 1}
        return result


class RewardTests(unittest.TestCase):
    def test_surplus_charge_has_no_terminal_bonus(self):
        config = RewardConfig(potential_seconds_per_mj=3.0)
        states = [State(0, 2), State(1, 2.5), State(2, 3), State(3, 4)]
        rewards = [transition_reward(a, b, 0, "accepted", i == 2, i == 2, config)[0]
                   for i, (a, b) in enumerate(zip(states, states[1:]))]
        self.assertAlmostEqual(sum(rewards), -3.0)

    def test_failed_prefix_cannot_keep_energy_potential(self):
        config = RewardConfig(potential_seconds_per_mj=20.0)
        first = transition_reward(State(0, 2), State(1, 4), 1, "accepted", False, False, config)[0]
        second, terms = transition_reward(State(1, 4), State(1.2, 4), 1, "numerical_fail", True, False, config)
        self.assertAlmostEqual(first+second, -100.2)
        self.assertEqual(terms["potential_after_s"], 0)
        self.assertAlmostEqual(terms["reward_time_s"], -.2)

    def test_unfinished_accepted_termination_is_failure(self):
        reward, terms = transition_reward(State(0, 2), State(.1, 2), 1, "accepted", True, False)
        self.assertLess(reward, -100)
        self.assertTrue(terms["failed_episode"])
        with self.assertRaises(ValueError):
            transition_reward(State(0, 2), State(.1, 2), 1, "physical_fail", False, False)

    def test_projection_is_diagnostic_and_units_stay_seconds(self):
        args = (State(0, 2), State(.2, 2), .3, "accepted", True, True)
        a, terms = transition_reward(*args, projection_fraction=1.0)
        b, _ = transition_reward(*args, projection_fraction=0.0)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, .1)
        self.assertEqual(terms["reward_unit"], "seconds")
        self.assertEqual(terms["discount_factor"], 1)


class LearningTests(unittest.TestCase):
    def policy(self):
        return _bind(GaussianPolicy(2, hidden=(4,), seed=1), ToyEnv.features, "signed", ToySimulator.protocol)

    def test_checkpoint_feature_order_action_mapping_and_legacy_rejection(self):
        policy = self.policy()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"v6.npz"
            save_policy(policy, path)
            loaded = load_policy(path, expected_features=ToyEnv.features, expected_action_space="signed")
            np.testing.assert_array_equal([policy(np.array([x, .5])) for x in (0, .3, .7)],
                                           [loaded(np.array([x, .5])) for x in (0, .3, .7)])
            with self.assertRaises(ValueError):
                load_policy(path, expected_features=tuple(reversed(ToyEnv.features)))
            with self.assertRaises(ValueError):
                load_policy(path, expected_action_space="mode_aware")
            policy.save(Path(folder)/"legacy.npz")
            with self.assertRaises(ValueError):
                load_policy(Path(folder)/"legacy.npz")

    def test_missing_physical_or_fsm_certificate_cannot_pass(self):
        sim = ToySimulator()
        result = evaluate_policy(sim, self.policy())
        self.assertTrue(is_legal(result))
        for key in ("public_rules_passed", "rule_state_closed", "dc_boundary_closed", "completed_lap"):
            changed = {"summary": dict(result["summary"])}
            changed["summary"].pop(key)
            self.assertFalse(is_legal(changed))
        for key, value in (("energy_end_mj", 1.9), ("periodic_speed_residual_mps", .021), ("lap_time_s", np.nan)):
            changed = {"summary": dict(result["summary"], **{key: value})}
            self.assertFalse(is_legal(changed))

    def test_formal_evaluation_shoots_and_fixed_reload_matches_same_start(self):
        sim = ToySimulator()
        policy = self.policy()
        formal = evaluate_policy(sim, policy)
        self.assertEqual(sim.periodic_calls, 1)
        fixed = evaluate_policy(sim, policy, periodic=False,
                                initial_speed_mps=formal["summary"]["speed_start_mps"])
        np.testing.assert_array_equal(formal["observations"], fixed["observations"])
        self.assertEqual(formal["summary"]["lap_time_s"], fixed["summary"]["lap_time_s"])

    def test_monte_carlo_returns_stop_at_episode_boundaries_and_shift_entropy(self):
        rewards = [1, 2, 10, 20]
        dones = [False, True, False, True]
        np.testing.assert_array_equal(monte_carlo_returns(rewards, dones), [3, 2, 30, 20])
        np.testing.assert_array_equal(monte_carlo_returns(rewards, dones, log_probabilities=[-5, -2, -9, -4], alpha=.5),
                                      [4, 2, 32, 20])
        with self.assertRaises(ValueError):
            monte_carlo_returns([1, 2], [False, False])

    def test_mc_initialisation_improves_q_without_moving_actor(self):
        sim, policy = ToySimulator(), self.policy()
        trajectory = _rollout(sim, policy)
        sac = DiagnosticSAC(policy, seed=3)
        before_actor = [v.copy() for v in policy.network.parameters()]
        labels = monte_carlo_returns(trajectory["rewards"], trajectory["dones"],
                                     log_probabilities=trajectory["log_probabilities"], alpha=sac.alpha)
        inputs = np.column_stack((policy.normalise(trajectory["observations"]), trajectory["actions"]))
        before = np.mean((sac.q1.forward(inputs)[:, 0]-labels)**2)
        warmstart_critics(sac, trajectory, iterations=120, batch_size=8, seed=2, monte_carlo=True)
        after = np.mean((sac.q1.forward(inputs)[:, 0]-labels)**2)
        self.assertLess(after, before*.3)
        for original, current in zip(before_actor, policy.network.parameters()):
            np.testing.assert_array_equal(original, current)
        for current, target in zip(sac.q1.parameters(), sac.target_q1.parameters()):
            np.testing.assert_array_equal(current, target)

    def test_sac_actor_gradient_and_fixed_temperature(self):
        policy = self.policy()
        sac = DiagnosticSAC(policy, seed=5, temperature_mode="fixed")
        sac.q2.biases[-1][0] += 20
        obs = np.array([[.1, .5], [.3, .5], [.8, .5]])
        noise = np.array([[.2], [-.4], [.1]])
        _, gradients, _ = sac.actor_loss_and_gradients(obs, noise)
        original = policy.network.weights[0][0, 0]
        epsilon = 1e-6
        policy.network.weights[0][0, 0] = original+epsilon
        upper = sac.actor_loss_and_gradients(obs, noise)[0]
        policy.network.weights[0][0, 0] = original-epsilon
        lower = sac.actor_loss_and_gradients(obs, noise)[0]
        policy.network.weights[0][0, 0] = original
        self.assertAlmostEqual(gradients[0][0, 0], (upper-lower)/(2*epsilon), places=7)
        batch = {"obs": obs, "next_obs": obs, "actions": np.zeros((3, 1)),
                 "rewards": np.ones((3, 1)), "dones": np.ones((3, 1))}
        alpha = sac.alpha
        sac.update(batch)
        self.assertEqual(sac.alpha, alpha)
        self.assertEqual(sac.gamma, 1.0)

    def test_bc_fixed_label_dagger_and_distinct_checkpoint_sources(self):
        sim = ToySimulator()
        teacher = sim.simulate(np.full(4, .25))
        with tempfile.TemporaryDirectory() as folder:
            trained = train_bc(teacher, sim, folder, seed=0, epochs=10, dagger_rounds=1, dagger_epochs=10)
            self.assertTrue(trained["metadata"]["legal_controller_found"])
            self.assertIn("no state-dependent expert replanning", trained["metadata"]["expert"])
            self.assertNotEqual(trained["metadata"]["selected_checkpoint"], trained["metadata"]["last_checkpoint"])
            self.assertGreaterEqual(sim.periodic_calls, 2)
            self.assertEqual(len(trained["metadata"]["aggregation"]), 1)
            self.assertTrue(trained["metadata"]["checkpoint_roundtrip_passed"])

    def test_sac_actual_budget_and_bc_fallback_are_explicit(self):
        sim, policy = ToySimulator(), self.policy()
        with tempfile.TemporaryDirectory() as folder:
            trained = train_sac(sim, policy, folder, episodes=2, critic_warmup_updates=2,
                                batch_size=4, evaluate_every=1, mc_warmstart=True)
            self.assertEqual(trained["metadata"]["episodes_requested"], 2)
            self.assertEqual(trained["metadata"]["episodes_completed"], 2)
            self.assertNotEqual(trained["metadata"]["selected_checkpoint"], trained["metadata"]["last_checkpoint"])
            self.assertTrue(trained["metadata"]["checkpoint_roundtrip_passed"])
            self.assertIn(trained["metadata"]["best_episode"], (0, 1, 2))
            self.assertFalse(trained["metadata"]["real_vehicle_benefit_claim"])
            self.assertGreaterEqual(sim.periodic_calls, 4)
            self.assertIn("q_vs_mc_rmse_s", trained["history"][1]["critic"])

    def test_numerical_exception_retains_real_prefix(self):
        sim = ToySimulator()
        sim.raise_at = 2
        with self.assertRaises(RolloutFailure) as caught:
            _rollout(sim, self.policy())
        self.assertEqual(len(caught.exception.trace["actions"]), 2)
        self.assertFalse(caught.exception.trace["accepted_trajectory_invented"])
        self.assertEqual(len(caught.exception.trace["infos"]), 2)


if __name__ == "__main__":
    unittest.main()
