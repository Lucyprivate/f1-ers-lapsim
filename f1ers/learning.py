"""Small, auditable NumPy implementation of behaviour cloning and continuous SAC.

The controller receives the simulator's *current* 12-dimensional observation.
It never indexes a stored racing-line action sequence during evaluation.  The
teacher's trajectory is used only as an offline supervised training dataset.

SAC here means a reparameterised tanh-Gaussian actor, two independent Q networks,
entropy-regularised Bellman targets, replay, Polyak target updates, and learned
temperature.  All network derivatives are implemented explicitly in NumPy; no
deep-learning runtime or external training service is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHECKPOINT_VERSION = 1
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class MLP:
    """Tanh hidden layers with a linear output, operating on row batches."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: tuple[int, ...] = (32, 32),
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng(0)
        self.sizes = (int(input_dim), *hidden, int(output_dim))
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0 / (n_in + n_out))
            self.weights.append(rng.normal(0.0, scale, (n_in, n_out)))
            self.biases.append(np.zeros(n_out, dtype=np.float64))

    def parameters(self) -> list[np.ndarray]:
        return [p for pair in zip(self.weights, self.biases) for p in pair]

    def forward(self, x: np.ndarray, cache: bool = False) -> Any:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != self.sizes[0]:
            raise ValueError(f"Expected {self.sizes[0]} input features, got {x.shape[1]}")
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.tanh(x)
            activations.append(x)
        return (x, activations) if cache else x

    def backward(
        self, activations: list[np.ndarray], output_grad: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        grad = np.asarray(output_grad, dtype=np.float64)
        grads: list[np.ndarray] = [np.empty(0)] * (2 * len(self.weights))
        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:
                grad = grad * (1.0 - activations[i + 1] ** 2)
            grads[2 * i] = activations[i].T @ grad
            grads[2 * i + 1] = grad.sum(axis=0)
            grad = grad @ self.weights[i].T
        return grad, grads

    def copy(self) -> "MLP":
        result = MLP(self.sizes[0], self.sizes[-1], self.sizes[1:-1])
        for target, source in zip(result.parameters(), self.parameters()):
            target[...] = source
        return result

    def polyak_update(self, source: "MLP", tau: float) -> None:
        for target, current in zip(self.parameters(), source.parameters()):
            target *= 1.0 - tau
            target += tau * current


class Adam:
    """Adam with global gradient-norm clipping for numerical robustness."""

    def __init__(self, parameters: Iterable[np.ndarray], lr: float = 3e-4) -> None:
        self.parameters = list(parameters)
        self.lr = lr
        self.first = [np.zeros_like(p) for p in self.parameters]
        self.second = [np.zeros_like(p) for p in self.parameters]
        self.steps = 0

    def step(self, gradients: list[np.ndarray], max_norm: float = 10.0) -> None:
        if len(gradients) != len(self.parameters):
            raise ValueError("One gradient is required for every parameter")
        if not all(np.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Non-finite gradient; checkpoint remains unchanged")
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        scale = min(1.0, max_norm / max(norm, 1e-12))
        self.steps += 1
        for p, g, m, v in zip(self.parameters, gradients, self.first, self.second):
            g = g * scale
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= self.lr * (m / (1.0 - 0.9**self.steps)) / (
                np.sqrt(v / (1.0 - 0.999**self.steps)) + 1e-8
            )


class GaussianPolicy:
    """One continuous signed ERS request in [-1, 1], evaluated from state."""

    def __init__(
        self,
        observation_dim: int,
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        hidden: tuple[int, ...] = (32, 32),
        seed: int = 0,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.mean = np.zeros(observation_dim) if mean is None else np.asarray(mean).copy()
        self.scale = np.ones(observation_dim) if scale is None else np.asarray(scale).copy()
        if self.mean.shape != (observation_dim,) or self.scale.shape != (observation_dim,):
            raise ValueError("Observation normalisation must match observation dimension")
        if np.any(self.scale <= 0) or not np.isfinite(self.scale).all():
            raise ValueError("Observation scales must be finite and positive")
        self.network = MLP(observation_dim, 2, hidden, np.random.default_rng(seed))
        self.network.weights[-1] *= 0.05
        self.network.biases[-1][1] = -2.5

    def normalise(self, observations: np.ndarray) -> np.ndarray:
        observations = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        if observations.shape[1] != self.observation_dim or not np.isfinite(observations).all():
            raise ValueError("Invalid current simulator observation")
        return np.clip((observations - self.mean) / self.scale, -8.0, 8.0)

    def __call__(self, observation: np.ndarray) -> float:
        values = self.network.forward(self.normalise(observation))
        return float(np.tanh(values[0, 0]))

    def sample(
        self,
        observations: np.ndarray,
        rng: np.random.Generator,
        epsilon: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        values, network_cache = self.network.forward(self.normalise(observations), cache=True)
        mu = values[:, :1]
        raw_log_std = values[:, 1:2]
        log_std = np.clip(raw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = np.exp(log_std)
        epsilon = rng.normal(size=mu.shape) if epsilon is None else np.asarray(epsilon)
        z = mu + std * epsilon
        actions = np.tanh(z)
        # Stable log(1 - tanh(z)^2); no arbitrary probability floor is needed.
        log_det = 2.0 * (np.log(2.0) - z - np.logaddexp(0.0, -2.0 * z))
        log_prob = -0.5 * epsilon**2 - log_std - 0.5 * np.log(2.0 * np.pi) - log_det
        cache = {
            "network": network_cache,
            "actions": actions,
            "std_epsilon": std * epsilon,
            "log_std_mask": (raw_log_std > LOG_STD_MIN) & (raw_log_std < LOG_STD_MAX),
        }
        return actions, log_prob, cache

    def copy(self) -> "GaussianPolicy":
        result = GaussianPolicy(
            self.observation_dim, self.mean, self.scale, self.network.sizes[1:-1]
        )
        result.network = self.network.copy()
        return result

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "version": np.asarray(CHECKPOINT_VERSION),
            "sizes": np.asarray(self.network.sizes, dtype=np.int64),
            "observation_mean": self.mean,
            "observation_scale": self.scale,
        }
        for i, (w, b) in enumerate(zip(self.network.weights, self.network.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, **arrays)


def load_policy(path: Path | str) -> GaussianPolicy:
    """Load a portable numeric checkpoint without pickle or executable payloads."""
    with np.load(path, allow_pickle=False) as arrays:
        if int(arrays["version"]) != CHECKPOINT_VERSION:
            raise ValueError("Unsupported policy checkpoint version")
        sizes = tuple(int(n) for n in arrays["sizes"])
        policy = GaussianPolicy(
            sizes[0], arrays["observation_mean"], arrays["observation_scale"], sizes[1:-1]
        )
        if sizes[-1] != 2:
            raise ValueError("Checkpoint is not a scalar tanh-Gaussian controller")
        for i, (w, b) in enumerate(zip(policy.network.weights, policy.network.biases)):
            if arrays[f"w{i}"].shape != w.shape or arrays[f"b{i}"].shape != b.shape:
                raise ValueError("Checkpoint parameter shape mismatch")
            w[...] = arrays[f"w{i}"]
            b[...] = arrays[f"b{i}"]
        if not all(np.isfinite(p).all() for p in policy.network.parameters()):
            raise ValueError("Non-finite checkpoint weights")
        return policy


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.observations = np.zeros((capacity, observation_dim))
        self.next_observations = np.zeros((capacity, observation_dim))
        self.actions = np.zeros((capacity, 1))
        self.rewards = np.zeros((capacity, 1))
        self.dones = np.zeros((capacity, 1))

    def add(
        self, observation: np.ndarray, action: float, reward: float,
        next_observation: np.ndarray, done: bool,
    ) -> None:
        if not np.isfinite(action) or not -1.0 <= action <= 1.0:
            raise ValueError("Replay stores requested actions within [-1, 1]")
        if not np.isfinite(reward):
            raise ValueError("Replay reward must be finite")
        self.observations[self.position] = observation
        self.next_observations[self.position] = next_observation
        self.actions[self.position, 0] = action
        self.rewards[self.position, 0] = reward
        self.dones[self.position, 0] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.observations[indices],
            "next_obs": self.next_observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }


def clone_behaviour(
    observations: np.ndarray,
    actions: np.ndarray,
    seed: int = 0,
    epochs: int = 120,
    hidden: tuple[int, ...] = (32, 32),
    batch_size: int = 128,
    initial_policy: GaussianPolicy | None = None,
) -> tuple[GaussianPolicy, list[dict[str, float]]]:
    """Supervised teacher imitation with true backpropagation, never action lookup."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    if observations.ndim != 2 or observations.shape[0] != actions.shape[0]:
        raise ValueError("Teacher observations and requested actions must have equal length")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("Teacher trajectory contains non-finite values")
    if np.any(np.abs(actions) > 1.0000001):
        raise ValueError("Teacher requested actions must be normalised to [-1, 1]")
    rng = np.random.default_rng(seed)
    if initial_policy is None:
        policy = GaussianPolicy(
            observations.shape[1], observations.mean(axis=0),
            np.maximum(observations.std(axis=0), 0.05), hidden, seed,
        )
    else:
        if initial_policy.observation_dim != observations.shape[1]:
            raise ValueError("Warm-start policy dimension differs from imitation dataset")
        # Preserve normalisation and weights when aggregating learner states.
        policy = initial_policy.copy()
    optimiser = Adam(policy.network.parameters(), lr=1e-3)
    features = policy.normalise(observations)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(observations))
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            prediction, cache = policy.network.forward(features[selected], cache=True)
            deployed = np.tanh(prediction[:, :1])
            grad = np.zeros_like(prediction)
            grad[:, :1] = 2.0 * (deployed - actions[selected]) * (1.0 - deployed**2) / len(selected)
            # A moderately concentrated exploration distribution starts SAC near BC.
            grad[:, 1:2] = 0.02 * (prediction[:, 1:2] + 2.5) / len(selected)
            _, gradients = policy.network.backward(cache, grad)
            optimiser.step(gradients)
        prediction = np.tanh(policy.network.forward(features)[:, :1])
        history.append({"epoch": epoch + 1, "action_rmse": float(np.sqrt(np.mean((prediction - actions) ** 2)))})
    return policy, history


class SAC:
    """Continuous-action Soft Actor-Critic, with automatic entropy temperature."""

    def __init__(
        self,
        policy: GaussianPolicy,
        seed: int = 0,
        gamma: float = 0.999,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
        initial_alpha: float = 0.001,
    ) -> None:
        self.policy = policy
        self.rng = np.random.default_rng(seed + 7919)
        self.gamma = gamma
        self.tau = tau
        self.q1 = MLP(policy.observation_dim + 1, 1, (32, 32), self.rng)
        self.q2 = MLP(policy.observation_dim + 1, 1, (32, 32), self.rng)
        self.target_q1 = self.q1.copy()
        self.target_q2 = self.q2.copy()
        self.actor_optimiser = Adam(policy.network.parameters(), learning_rate)
        self.q1_optimiser = Adam(self.q1.parameters(), learning_rate)
        self.q2_optimiser = Adam(self.q2.parameters(), learning_rate)
        self.log_alpha = np.asarray([np.log(initial_alpha)], dtype=np.float64)
        self.alpha_optimiser = Adam([self.log_alpha], learning_rate)
        self.target_entropy = -1.0
        self.updates = 0

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha[0]))

    def actor_loss_and_gradients(
        self, observations: np.ndarray, epsilon: np.ndarray | None = None
    ) -> tuple[float, list[np.ndarray], np.ndarray]:
        actions, log_prob, cache = self.policy.sample(observations, self.rng, epsilon)
        features = self.policy.normalise(observations)
        q_input = np.concatenate([features, actions], axis=1)
        q1, c1 = self.q1.forward(q_input, cache=True)
        q2, c2 = self.q2.forward(q_input, cache=True)
        choose_first = q1 <= q2
        q1_input_grad, _ = self.q1.backward(c1, choose_first.astype(float))
        q2_input_grad, _ = self.q2.backward(c2, (~choose_first).astype(float))
        dq_da = (q1_input_grad + q2_input_grad)[:, -1:]
        action_derivative = 1.0 - actions**2
        # Reparameterisation includes the tanh change-of-variable derivative.
        mu_grad = self.alpha * 2.0 * actions - dq_da * action_derivative
        log_std_grad = self.alpha * (-1.0 + 2.0 * actions * cache["std_epsilon"])
        log_std_grad -= dq_da * action_derivative * cache["std_epsilon"]
        log_std_grad *= cache["log_std_mask"]
        grad = np.concatenate([mu_grad, log_std_grad], axis=1) / len(features)
        _, parameter_gradients = self.policy.network.backward(cache["network"], grad)
        loss = float(np.mean(self.alpha * log_prob - np.minimum(q1, q2)))
        return loss, parameter_gradients, log_prob

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        next_action, next_log_prob, _ = self.policy.sample(batch["next_obs"], self.rng)
        next_features = self.policy.normalise(batch["next_obs"])
        next_input = np.concatenate([next_features, next_action], axis=1)
        next_q = np.minimum(self.target_q1.forward(next_input), self.target_q2.forward(next_input))
        target = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * (
            next_q - self.alpha * next_log_prob
        )
        features = self.policy.normalise(batch["obs"])
        q_input = np.concatenate([features, batch["actions"]], axis=1)
        critic_losses = []
        for network, optimiser in [(self.q1, self.q1_optimiser), (self.q2, self.q2_optimiser)]:
            predicted, cache = network.forward(q_input, cache=True)
            error = predicted - target
            critic_losses.append(float(np.mean(error**2)))
            _, gradients = network.backward(cache, 2.0 * error / len(error))
            optimiser.step(gradients)
        actor_loss, actor_gradients, log_prob = self.actor_loss_and_gradients(batch["obs"])
        self.actor_optimiser.step(actor_gradients)
        temperature_gradient = -float(np.mean(log_prob + self.target_entropy))
        self.alpha_optimiser.step([np.asarray([temperature_gradient])])
        self.log_alpha[...] = np.clip(self.log_alpha, np.log(1e-5), np.log(0.5))
        self.target_q1.polyak_update(self.q1, self.tau)
        self.target_q2.polyak_update(self.q2, self.tau)
        self.updates += 1
        return {
            "q1_loss": critic_losses[0], "q2_loss": critic_losses[1],
            "actor_loss": actor_loss, "alpha": self.alpha,
            "mean_log_probability": float(np.mean(log_prob)),
        }


def evaluate_policy(sim: Any, policy: GaussianPolicy) -> dict[str, Any]:
    """Formal whole-lap evaluation uses the simulator's periodic-speed solver."""
    return sim.simulate(policy=policy, sustainable=True)


def _valid_result(result: dict[str, Any]) -> bool:
    """An invalid evaluation can never displace a feasible incumbent."""
    summary = result["summary"]
    return (bool(summary.get("feasible", False)) and bool(summary.get("sustainable", False))
            and bool(summary.get("coupling_converged", True)))


def _score(result: dict[str, Any]) -> float:
    return float(result["summary"].get("lap_time_s", np.inf)) if _valid_result(result) else np.inf


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def train_and_evaluate(
    sim: Any,
    teacher_result: dict[str, Any],
    output_dir: Path | str,
    seeds: tuple[int, ...] = (0, 1, 2),
    episodes: int = 60,
    bc_epochs: int = 120,
    batch_size: int = 64,
    evaluate_every: int = 5,
    update_every: int = 2,
    critic_warmup_updates: int = 120,
    aggregation_rounds: int = 3,
    aggregation_epochs: int = 600,
    progress: Any = None,
) -> dict[str, Any]:
    """Imitate the optimiser, fine-tune SAC, and retain the best valid controller.

    All seed evaluations share the supplied simulator, initial SOC, and the same
    periodic-speed solver.  Invalid or slower candidates remain in the training
    history, rather than silently replacing the saved best controller.  If no
    learned controller is sustainable, ``selected_policy`` is null and the
    caller must retain its previously saved optimal-control baseline.

    Training reward is the simulator's seconds saved relative to no-K control.
    The physical simulator owns projection, terminal sustainability penalties,
    and periodic-speed penalties; no future teacher action enters SAC reward.
    """
    if episodes < 0 or bc_epochs < 1 or not seeds:
        raise ValueError("Require at least one seed, positive BC epochs, nonnegative SAC episodes")
    if update_every < 1 or evaluate_every < 1:
        raise ValueError("Update and evaluation intervals must be positive")
    if not 0 <= aggregation_rounds <= 3 or aggregation_epochs < 1:
        raise ValueError("Require 0 to 3 DAgger rounds and positive aggregation epochs")
    if not _valid_result(teacher_result):
        raise ValueError("Teacher must pass physics and sustainable-lap audit before training")
    observations = np.asarray(teacher_result["observations"], dtype=float)
    teacher_actions = np.asarray(teacher_result["requested_actions"], dtype=float)
    teacher_start_speed = (float(teacher_result["speed_mps"][0])
                           if "speed_mps" in teacher_result else None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    selected_checkpoint: str | None = None
    selected_score = np.inf

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"Seed {seed}: behaviour cloning")
        policy, bc_history = clone_behaviour(observations, teacher_actions, seed, bc_epochs)
        policy.save(seed_dir / "raw_bc_policy.npz")
        raw_bc_result = evaluate_policy(sim, policy)
        bc_result, best_bc_policy = raw_bc_result, policy.copy()
        best_bc_score = _score(raw_bc_result)
        selected_bc_round = 0
        aggregation_history = []
        imitation_observations = [observations]
        imitation_actions = [teacher_actions]
        # DAgger with a fixed, position-indexed single-track expert. The expert
        # labels *only training* learner states, never deployed policy calls.
        # This corrects the teacher/learner SOC and speed distribution mismatch;
        # it does not replan the expert or establish cross-track generalisation.
        for round_index in range(1, aggregation_rounds + 1):
            learner_rollout = evaluate_policy(sim, policy)
            learner_observations = np.asarray(learner_rollout["observations"], dtype=float)
            if learner_observations.shape != observations.shape:
                raise ValueError("DAgger requires equal, ordered simulator track nodes")
            if not np.isfinite(learner_observations).all():
                raise ValueError("Cannot aggregate non-finite learner states")
            imitation_observations.append(learner_observations)
            imitation_actions.append(teacher_actions.copy())
            combined_observations = np.vstack(imitation_observations)
            combined_actions = np.concatenate(imitation_actions)
            policy, round_curve = clone_behaviour(
                combined_observations, combined_actions,
                seed=seed + round_index * 1009, epochs=aggregation_epochs,
                initial_policy=policy,
            )
            candidate = evaluate_policy(sim, policy)
            candidate_score = _score(candidate)
            improved = candidate_score < best_bc_score - 1e-9
            if improved:
                best_bc_policy, bc_result = policy.copy(), candidate
                best_bc_score, selected_bc_round = candidate_score, round_index
            aggregation_history.append({
                "round": round_index, "samples": len(combined_observations),
                "epochs": aggregation_epochs,
                "learner_state_source_summary": learner_rollout["summary"],
                "aggregate_action_rmse": round_curve[-1]["action_rmse"],
                "evaluation": candidate["summary"], "selected_as_best": improved,
            })
            _write_json(seed_dir / f"dagger_round_{round_index}_curve.json", round_curve)
            policy.save(seed_dir / f"dagger_round_{round_index}_candidate.npz")
            if progress:
                label = f"{candidate_score:.4f} s" if np.isfinite(candidate_score) else "infeasible"
                progress(f"Seed {seed}, DAgger round {round_index}/{aggregation_rounds}: {label}")
        policy = best_bc_policy
        policy.save(seed_dir / "bc_policy.npz")
        bc_action_rmse = float(np.sqrt(np.mean((
            np.tanh(policy.network.forward(policy.normalise(observations))[:, 0]) - teacher_actions
        ) ** 2)))
        best_result = bc_result
        best_score = _score(bc_result)
        best_policy = policy.copy() if np.isfinite(best_score) else None
        best_episode = 0 if best_policy is not None else None
        sac = SAC(policy, seed=seed)
        replay = ReplayBuffer(max(20000, len(observations) * (episodes + 1)), policy.observation_dim)
        rng = np.random.default_rng(seed + 104729)
        environment = sim.create_env()

        # Seed replay with a BC rollout, using *actual* feedback and transitions.
        # Teacher states/actions are not inserted as fictitious environment data.
        reset_kwargs = ({"start_speed_mps": teacher_start_speed}
                        if teacher_start_speed is not None else {})
        observation = np.asarray(environment.reset(seed=seed, **reset_kwargs), dtype=float)
        for _ in range(int(sim.n_nodes) + 1):
            action = policy(observation)
            next_observation, reward, done, info = environment.step(action)
            replay.add(observation, action, float(reward), next_observation, done)
            observation = np.asarray(next_observation, dtype=float)
            if done:
                break

        # Fit critics before the actor moves away from its supervised solution.
        # Use the same entropy Bellman target, but freeze actor/temperature here.
        for _ in range(critic_warmup_updates):
            batch = replay.sample(batch_size, rng)
            next_action, log_prob, _ = policy.sample(batch["next_obs"], sac.rng)
            q_next = np.concatenate([policy.normalise(batch["next_obs"]), next_action], axis=1)
            target = batch["rewards"] + sac.gamma * (1.0 - batch["dones"]) * (
                np.minimum(sac.target_q1.forward(q_next), sac.target_q2.forward(q_next)) - sac.alpha * log_prob
            )
            q_input = np.concatenate([policy.normalise(batch["obs"]), batch["actions"]], axis=1)
            for network, optimiser in [(sac.q1, sac.q1_optimiser), (sac.q2, sac.q2_optimiser)]:
                predicted, cache = network.forward(q_input, cache=True)
                _, gradients = network.backward(cache, 2.0 * (predicted - target) / len(predicted))
                optimiser.step(gradients)
            sac.target_q1.polyak_update(sac.q1, sac.tau)
            sac.target_q2.polyak_update(sac.q2, sac.tau)

        history: list[dict[str, Any]] = [{
            "episode": 0, "stage": "bc", "evaluation": bc_result["summary"],
            "selected_as_best": best_policy is not None,
        }]
        transition_count = 0
        failed = None
        for episode in range(1, episodes + 1):
            observation = np.asarray(environment.reset(seed=seed + episode, **reset_kwargs), dtype=float)
            episode_reward = 0.0
            update_metrics = []
            info = {}
            try:
                for _ in range(int(sim.n_nodes) + 1):
                    actions, _, _ = policy.sample(observation, rng)
                    action = float(actions[0, 0])
                    next_observation, reward, done, info = environment.step(action)
                    replay.add(observation, action, float(reward), next_observation, done)
                    observation = np.asarray(next_observation, dtype=float)
                    episode_reward += float(reward)
                    transition_count += 1
                    if replay.size >= batch_size and transition_count % update_every == 0:
                        update_metrics.append(sac.update(replay.sample(batch_size, rng)))
                    if done:
                        break
            except (FloatingPointError, ValueError) as error:
                failed = f"{type(error).__name__}: {error}"
                history.append({"episode": episode, "stage": "sac", "failure": failed})
                break
            record: dict[str, Any] = {
                "episode": episode, "stage": "sac", "training_reward_seconds_saved": episode_reward,
                "updates": sac.updates, "alpha": sac.alpha, "terminal_info": info,
            }
            if update_metrics:
                record["losses"] = {k: float(np.mean([m[k] for m in update_metrics])) for k in update_metrics[0]}
            if episode % evaluate_every == 0 or episode == episodes:
                evaluation = evaluate_policy(sim, policy)
                candidate_score = _score(evaluation)
                improved = candidate_score < best_score - 1e-9
                record["evaluation"] = evaluation["summary"]
                record["selected_as_best"] = improved
                if improved:
                    best_policy, best_result = policy.copy(), evaluation
                    best_score, best_episode = candidate_score, episode
                if progress:
                    label = f"{candidate_score:.4f} s" if np.isfinite(candidate_score) else "infeasible"
                    progress(f"Seed {seed}, episode {episode}/{episodes}: deterministic candidate {label}")
            history.append(record)

        policy.save(seed_dir / "last_candidate.npz")
        last_result = evaluate_policy(sim, policy)
        if best_policy is not None:
            best_path = seed_dir / "best_feasible_policy.npz"
            best_policy.save(best_path)
            # Reload before reporting: the delivered checkpoint must reproduce it.
            best_result = evaluate_policy(sim, load_policy(best_path))
            best_score = _score(best_result)
            if best_score < selected_score:
                selected_score = best_score
                selected_checkpoint = best_path.relative_to(output_dir.parent).as_posix()
        else:
            best_path = None
        _write_json(seed_dir / "bc_curve.json", bc_history)
        _write_json(seed_dir / "dagger_summary.json", aggregation_history)
        _write_json(seed_dir / "sac_curve.json", history)
        metadata = {
            "seed": seed, "bc_action_rmse": bc_action_rmse,
            "raw_bc_action_rmse": bc_history[-1]["action_rmse"],
            "raw_bc_summary": raw_bc_result["summary"],
            "selected_bc_aggregation_round": selected_bc_round,
            "aggregation_history": aggregation_history,
            "bc_summary": bc_result["summary"], "last_summary": last_result["summary"],
            "best_summary": best_result["summary"] if best_policy is not None else None,
            "best_episode": best_episode,
            "best_checkpoint": best_path.relative_to(output_dir.parent).as_posix() if best_path else None,
            "sac_updates": sac.updates, "replay_size": replay.size, "training_failure": failed,
            "episodes_completed": max((r["episode"] for r in history if "failure" not in r), default=0),
        }
        _write_json(seed_dir / "metadata.json", metadata)
        records.append(metadata)
        results[str(seed)] = {"bc": bc_result, "sac_last": last_result}
        if best_policy is not None:
            results[str(seed)]["selected"] = best_result

    successful_times = [float(r["best_summary"]["lap_time_s"]) for r in records if r["best_summary"] is not None]
    teacher_time = float(teacher_result["summary"]["lap_time_s"])
    metadata = {
        "algorithm": "NumPy behaviour cloning / DAgger + continuous Soft Actor-Critic",
        "observation_source": "current simulator state; no oracle future actions",
        "observation_dim": int(observations.shape[1]), "actor_hidden_layers": [32, 32],
        "critic_hidden_layers": [32, 32], "seeds": list(seeds), "episodes_requested": episodes,
        "bc_epochs": bc_epochs, "reward_scale": 1.0, "gamma": 0.999, "tau": 0.005,
        "aggregation_rounds": aggregation_rounds, "aggregation_epochs": aggregation_epochs,
        "aggregation_expert": "Frozen single-track teacher requested action at current node; labels are used only for training",
        "aggregation_dataset": "Teacher states plus learner feedback rollout states, retaining every round",
        "bc_selection": "Fastest physically feasible BC/DAgger checkpoint, rather than final aggregation round",
        "training_start_speed_mps": teacher_start_speed,
        "training_start_speed_source": "Numerical teacher's periodic start speed",
        "batch_size": batch_size, "update_every": update_every,
        "critic_warmup_updates": critic_warmup_updates, "initial_alpha": 0.001,
        "target_entropy": -1.0, "learning_rate": 3e-4,
        "evaluation_protocol": "identical simulator / initial SOC / periodic speed solver for every seed",
        "teacher_lap_time_s": teacher_time, "records": records,
        "feasible_seed_count": len(successful_times), "requested_seed_count": len(seeds),
        "selected_policy": selected_checkpoint,
        "checkpoint_paths_relative_to": "experiment directory (parent of the models directory)",
        "mean_selected_lap_time_s": float(np.mean(successful_times)) if successful_times else None,
        "std_selected_lap_time_s": float(np.std(successful_times)) if successful_times else None,
        "mean_gap_to_teacher_s": float(np.mean(successful_times) - teacher_time) if successful_times else None,
        "selection_rule": "best feasible deterministic learned policy per seed; keep optimiser baseline if none",
        "claim": "single-track numerical experiment, not a globally optimal strategy or verified real SOC",
    }
    _write_json(output_dir / "training_summary.json", metadata)
    return {"metadata": metadata, "results": results}
