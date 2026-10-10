"""Value conditioning, action gradients and immutable carried boundaries."""
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
import numpy as np

from f1ers_v7._core.networks import GaussianPolicy
from f1ers_v7._core.learning_base import DiagnosticSAC, save_policy as save_v6
from f1ers_v7._core.reward import RewardConfig, transition_reward
from f1ers_v7.learning import (ReturnScaledSAC, _bind, _rollout, adopt_policy, save_policy,
    load_policy, warmstart_critics, critic_diagnostics, evaluate_policy, is_legal, train_sac,
    monte_carlo_returns, RolloutFailure)


@dataclass(frozen=True)
class ToyState:
    elapsed_time_s: float
    energy_mj: float
    speed_mps: float


class ToyEnv:
    features = ("phase", "energy", "reference", "reduction")
    observation_dim = len(features)

    def __init__(self, sim, reward_config=None, initial_speed_mps=None,
                 initial_energy_mj=2., carried_rule_state=None):
        self.sim, self.config = sim, reward_config or RewardConfig()
        self.speed = 40. if initial_speed_mps is None else initial_speed_mps
        self.energy, self.carry = initial_energy_mj, deepcopy(carried_rule_state)
        self.reset()

    def reset(self, seed=None):
        self.i, self.done = 0, False
        self.state = ToyState(0., self.energy, self.speed)
        self.actions, self.observations, self.times = [], [], [0.]
        return self.get_obs()

    def get_obs(self):
        carry = self.carry or {}
        return np.array([self.i/self.sim.n_nodes, self.energy/4.,
                         carry.get("reference", 0.)/350., carry.get("reduction", 0.)/700.])

    def step(self, action):
        if self.done:
            raise RuntimeError("No accepted step after terminal")
        if self.sim.raise_at is not None and self.i == self.sim.raise_at:
            raise RuntimeError("Injected failure")
        obs, before = self.get_obs(), self.state
        dt = 1+.2*(float(action)-.25)**2
        self.state = ToyState(before.elapsed_time_s+dt, self.energy, self.speed)
        self.i += 1
        self.done = self.i == self.sim.n_nodes
        if self.done and self.sim.terminal_energy_drop:
            self.state = ToyState(self.state.elapsed_time_s, self.energy-self.sim.terminal_energy_drop, self.speed)
        reward_status = "terminal_constraint_fail" if self.done and self.state.energy_mj<self.energy-1e-8 else "accepted"
        reward, terms = transition_reward(before, self.state, 0., reward_status, self.done,
                                           self.done, self.config, 1.)
        self.actions.append(action)
        self.observations.append(obs)
        self.times.append(self.state.elapsed_time_s)
        return self.get_obs(), reward, self.done, {"reward_terms": terms, "step_certified": True,
            "projection_fraction": 1., "requested_dc_kw": 350*float(action), "dc_kw": 0.,
            "status": "accepted", "reward_status": reward_status, "dt_s": dt}

    def result(self):
        return {"observations": np.array(self.observations), "requested_actions": np.array(self.actions),
                "speed_mps": np.full(len(self.actions)+1, self.speed), "time_s": np.array(self.times),
                "energy_mj": np.array([self.energy]*len(self.actions)+[self.state.energy_mj]), "summary": {
                    "completed_lap": self.done, "execution_feasible": True,
                    "feasible": self.done and self.state.energy_mj>=self.energy-1e-8,
                    "sustainable": self.state.energy_mj>=self.energy-1e-8,
                    "public_rules_passed": True, "periodic_speed_closed": True,
                    "dc_boundary_closed": True, "rule_state_closed": True, "request_proxy_consistent": True,
                    "periodic_speed_residual_mps": 0., "energy_start_mj": self.energy,
                    "energy_end_mj": self.state.energy_mj, "speed_start_mps": self.speed,
                    "speed_end_mps": self.speed, "lap_time_s": self.state.elapsed_time_s}}


class ToySimulator:
    protocol = "v7_reachable_control"
    features, observation_dim = ToyEnv.features, ToyEnv.observation_dim
    n_nodes, length_m = 4, 20.

    def __init__(self):
        self.raise_at, self.periodic_calls = None, 0
        self.terminal_energy_drop = 0.
        self.boundaries = []

    def create_env(self, action_space="signed", **kwargs):
        self.boundaries.append(deepcopy(kwargs))
        return ToyEnv(self, **kwargs)

    def simulate(self, controller, action_space="signed", **kwargs):
        env = self.create_env(action_space=action_space, **kwargs)
        obs = env.reset()
        while not env.done:
            obs, _, _, _ = env.step(controller(obs))
        return env.result()

    def periodic_simulate(self, **kwargs):
        self.periodic_calls += 1
        result = self.simulate(**kwargs)
        result["summary"]["shooting"] = {"converged": True, "full_lap_trials": 1}
        return result


class LearningTests(unittest.TestCase):
    def policy(self, sim=None):
        sim = sim or ToySimulator()
        return _bind(GaussianPolicy(sim.observation_dim, hidden=(4,), seed=2), sim.features,
                     "signed", sim.protocol)

    def test_initial_q_and_bellman_target_preserve_seconds_objective(self):
        p, scale = self.policy(), 73.084831
        plain = DiagnosticSAC(p.copy(), seed=4, initial_alpha=.001)
        scaled = ReturnScaledSAC(p.copy(), seed=4, return_scale_s=scale, initial_alpha=.001)
        obs = np.random.default_rng(3).normal(size=(5, p.observation_dim))
        inputs = np.column_stack((p.normalise(obs), np.linspace(-.4, .4, 5)))
        np.testing.assert_allclose(scaled.q1.forward(inputs)*scale, plain.q1.forward(inputs), atol=1e-15)
        batch = {"obs": obs, "next_obs": obs+.1, "actions": np.zeros((5, 1)),
                 "rewards": np.arange(5)[:, None]*-.1, "dones": np.array([[0], [0], [1], [0], [1]])}
        action = np.ones((5, 1))*.2
        logp = np.ones((5, 1))*-2.
        next_inputs = np.column_stack((plain.policy.normalise(batch["next_obs"]), action))
        expected = batch["rewards"]+(1-batch["dones"])*(
            np.minimum(plain.target_q1.forward(next_inputs), plain.target_q2.forward(next_inputs))-plain.alpha*logp)
        np.testing.assert_allclose(scaled.bellman_target(batch, next_actions=action,
            next_log_probabilities=logp)*scale, expected, atol=1e-15)
        self.assertEqual(scaled.alpha, plain.alpha)
        self.assertEqual(scaled.gamma, 1.)

    def test_all_actor_derivatives_match_finite_differences_and_seconds_gradient(self):
        p, scale = self.policy(), 73.084831
        plain = DiagnosticSAC(p.copy(), seed=7)
        scaled = ReturnScaledSAC(p.copy(), seed=7, return_scale_s=scale)
        plain.q2.biases[-1][0] += 20
        scaled.q2.biases[-1][0] += 20/scale
        obs = np.random.default_rng(5).normal(size=(4, p.observation_dim))
        epsilon = np.random.default_rng(6).normal(size=(4, 1))
        loss, gradients, _ = scaled.actor_loss_and_gradients(obs, epsilon)
        seconds_loss, seconds_gradients, _ = plain.actor_loss_and_gradients(obs, epsilon)
        self.assertAlmostEqual(loss*scale, seconds_loss, places=12)
        for a, b in zip(gradients, seconds_gradients):
            np.testing.assert_allclose(a*scale, b, atol=1e-14)
        step = 1e-6
        for parameter, gradient in zip(scaled.policy.network.parameters(), gradients):
            numerical = np.empty_like(parameter)
            for i in np.ndindex(parameter.shape):
                original = parameter[i]
                parameter[i] = original+step
                upper = scaled.actor_loss_and_gradients(obs, epsilon)[0]
                parameter[i] = original-step
                lower = scaled.actor_loss_and_gradients(obs, epsilon)[0]
                parameter[i] = original
                numerical[i] = (upper-lower)/(2*step)
            np.testing.assert_allclose(gradient, numerical, atol=2e-10, rtol=2e-5)

    def test_fixed_alpha_does_not_mutate_temperature_optimizer(self):
        sac = ReturnScaledSAC(self.policy(), temperature_mode="fixed", initial_alpha=.001)
        obs = np.zeros((4, ToyEnv.observation_dim))
        batch = {"obs": obs, "next_obs": obs, "actions": np.zeros((4, 1)),
                 "rewards": np.ones((4, 1))*-.1, "dones": np.ones((4, 1))}
        alpha = sac.alpha
        result = sac.update(batch)
        self.assertEqual(sac.alpha, alpha)
        self.assertEqual(sac.alpha_optimiser.steps, 0)
        self.assertEqual(result["q1_loss"], result["q1_loss_normalized"]*sac.return_scale_s**2)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_large_return_mc_conditioning_improves_value_without_actor_movement(self):
        n, dim = 860, 4
        phase = np.arange(n)/n
        trajectory = {"observations": np.column_stack((phase, np.ones((n, dim-1)))),
                      "next_observations": np.column_stack((phase+1/n, np.ones((n, dim-1)))),
                      "actions": np.zeros(n), "rewards": np.ones(n)*(-73.084831/n),
                      "dones": np.arange(n)==n-1, "log_probabilities": np.ones(n)*1.}
        plain = DiagnosticSAC(self.policy(), seed=2026)
        scaled = ReturnScaledSAC(self.policy(), seed=2026)
        actor = [a.copy() for a in scaled.policy.network.parameters()]
        from f1ers_v7._core.learning_base import warmstart_critics as unscaled_warmstart
        unscaled_warmstart(plain, trajectory, iterations=120, batch_size=64, seed=2026, monte_carlo=True)
        record = warmstart_critics(scaled, trajectory, iterations=120, batch_size=64, seed=2026)
        labels = monte_carlo_returns(trajectory["rewards"], trajectory["dones"],
                                    log_probabilities=trajectory["log_probabilities"], alpha=plain.alpha)
        inputs = np.column_stack((plain.policy.normalise(trajectory["observations"]), trajectory["actions"]))
        unscaled_error = np.sqrt(np.mean((np.minimum(plain.q1.forward(inputs), plain.q2.forward(inputs))[:, 0]-labels)**2))
        diagnostics = critic_diagnostics(scaled, trajectory)
        self.assertLess(diagnostics["q_vs_mc_rmse_s"], unscaled_error*.7)
        self.assertLess(max(v["critic_gradient_norm"] for v in record["history"]), 10.)
        for a, b in zip(actor, scaled.policy.network.parameters()):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(scaled.q1.parameters(), scaled.target_q1.parameters()):
            np.testing.assert_array_equal(a, b)

    def test_feature_schema_checkpoint_and_explicit_historical_adoption(self):
        sim, p = ToySimulator(), self.policy()
        with tempfile.TemporaryDirectory() as folder:
            own, old = Path(folder)/"v7.npz", Path(folder)/"v6.npz"
            save_policy(p, own)
            restored = load_policy(own, expected_features=sim.features, expected_execution_protocol=sim.protocol)
            np.testing.assert_array_equal([p(np.zeros(4))], [restored(np.zeros(4))])
            with self.assertRaises(ValueError):
                load_policy(own, expected_features=tuple(reversed(sim.features)))
            save_v6(p, old)
            with self.assertRaises(ValueError):
                load_policy(old)
            adopted = adopt_policy(p, sim)
            self.assertIsNot(adopted, p)
            other = ToySimulator()
            other.features, other.observation_dim = tuple(f"f{i}" for i in range(33)), 33
            p33 = self.policy(other)
            self.assertEqual(p33.observation_dim, 33)
            with self.assertRaises(ValueError):
                adopt_policy(p, other)

    def test_reachable_gate_is_mandatory(self):
        result = evaluate_policy(ToySimulator(), self.policy())
        self.assertTrue(is_legal(result))
        for gate in (False, None, "true", 1):
            result["summary"]["request_proxy_consistent"] = gate
            self.assertFalse(is_legal(result))
        result["summary"].pop("request_proxy_consistent")
        self.assertFalse(is_legal(result))

    def test_pool_carries_energy_speed_and_rule_state_without_shooting_or_mutation(self):
        sim, p = ToySimulator(), self.policy()
        pool = [{"speed_mps": 41., "energy_mj": 3.2, "carried_rule_state": {"reference": 100., "reduction": 30.}},
                {"speed_mps": 42., "energy_mj": 4., "carried_rule_state": {"reference": 150., "reduction": 60.}}]
        original = deepcopy(pool)
        with tempfile.TemporaryDirectory() as folder:
            result = train_sac(sim, p, folder, episodes=2, batch_size=4, critic_warmup_updates=2,
                evaluate_every=1, initial_state_pool=pool, evaluation_initial_state=pool[1], retain_rollouts=True)
        self.assertEqual(pool, original)
        self.assertEqual(sim.periodic_calls, 0)
        self.assertEqual([v["initial_boundary_index"] for v in result["history"][1:]], [0, 1])
        self.assertEqual([v["result"]["summary"]["energy_start_mj"] for v in result["rollouts"]], [3.2, 4.])
        self.assertEqual([v["result"]["summary"]["speed_start_mps"] for v in result["rollouts"]], [41., 42.])
        self.assertEqual(result["metadata"]["bc_summary"]["energy_start_mj"], 4.)
        self.assertTrue(result["metadata"]["checkpoint_roundtrip_passed"])
        self.assertEqual(result["metadata"]["replay_action"].split(";")[0], "raw requested scalar")
        with self.assertRaises(ValueError):
            evaluate_policy(sim, p, initial_state=pool[0], periodic=True)

    def test_raw_action_not_substituted_with_executed_dc_and_failure_prefix_kept(self):
        sim, p = ToySimulator(), self.policy()
        from f1ers_v7._core.networks import ReplayBuffer
        replay = ReplayBuffer(20, sim.observation_dim)
        trajectory = _rollout(sim, p, replay=replay, initial_energy_mj=3.)
        np.testing.assert_array_equal(replay.actions[:4, 0], trajectory["actions"])
        self.assertTrue(np.any(replay.actions[:4, 0] != 0.))
        self.assertTrue(all(i["dc_kw"]==0. for i in trajectory["infos"]))
        sim.raise_at = 2
        with self.assertRaises(RolloutFailure) as caught:
            _rollout(sim, p, initial_energy_mj=4.)
        self.assertEqual(len(caught.exception.trace["actions"]), 2)
        self.assertEqual(caught.exception.trace["last_accepted_state"]["energy_mj"], 4.)
        self.assertFalse(caught.exception.trace["accepted_trajectory_invented"])

    def test_warm_energy_potential_still_telescopes_to_zero(self):
        sim, p = ToySimulator(), self.policy()
        config = RewardConfig(potential_seconds_per_mj=3., energy_start_mj=2.)
        trajectory = _rollout(sim, p, reward_config=config, initial_energy_mj=4.)
        self.assertAlmostEqual(trajectory["rewards"].sum(), -trajectory["result"]["summary"]["lap_time_s"])
        self.assertAlmostEqual(sum(i["reward_terms"]["reward_potential_s"] for i in trajectory["infos"]), 0.)
        self.assertEqual(config.energy_start_mj, 2.)

    def test_completed_energy_depletion_receives_once_only_terminal_failure_penalty(self):
        sim, p = ToySimulator(), self.policy()
        sim.terminal_energy_drop = .1
        trajectory = _rollout(sim, p, initial_energy_mj=4.,
            reward_config=RewardConfig(potential_seconds_per_mj=3.))
        summary = trajectory["result"]["summary"]
        self.assertTrue(summary["completed_lap"])
        self.assertTrue(summary["execution_feasible"])
        self.assertFalse(summary["sustainable"])
        self.assertFalse(is_legal(trajectory["result"]))
        self.assertEqual(len(trajectory["actions"]), sim.n_nodes)
        self.assertEqual(trajectory["infos"][-1]["status"], "accepted")
        self.assertEqual(trajectory["infos"][-1]["reward_status"], "terminal_constraint_fail")
        self.assertAlmostEqual(trajectory["rewards"].sum(), -summary["lap_time_s"]-100.)
        self.assertAlmostEqual(sum(i["reward_terms"]["reward_failure_s"] for i in trajectory["infos"]), -100.)
        self.assertAlmostEqual(sum(i["reward_terms"]["reward_potential_s"] for i in trajectory["infos"]), 0.)
        sac = ReturnScaledSAC(p, initial_alpha=.001)
        batch = {"obs": trajectory["observations"][-1:],
                 "next_obs": trajectory["next_observations"][-1:],
                 "actions": trajectory["actions"][-1:, None],
                 "rewards": trajectory["rewards"][-1:, None], "dones": np.ones((1, 1))}
        target = sac.bellman_target(batch, next_actions=np.zeros((1, 1)),
                                    next_log_probabilities=np.ones((1, 1))*-100.)
        np.testing.assert_allclose(target*sac.return_scale_s, batch["rewards"], atol=1e-14)

    def test_invalid_boundary_and_budget_rejected(self):
        sim, p = ToySimulator(), self.policy()
        for energy in (1.99, 4.01, np.nan):
            with self.assertRaises(ValueError):
                _rollout(sim, p, initial_energy_mj=energy)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                train_sac(sim, p, folder, episodes=21)
            with self.assertRaises(ValueError):
                train_sac(sim, p, folder, initial_state_pool=[])


if __name__ == "__main__":
    unittest.main()
