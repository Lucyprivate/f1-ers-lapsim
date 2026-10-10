"""Versioned causal-controller training and a bounded SAC temperature ablation.

Legacy training/checkpoints are deliberately kept separate.  This module reuses
the tested NumPy neural-network primitives, but scores every controller with the
causal v2 simulator, its common certified speed boundary, and its energy reserve.
No retrospective deployment scaling or terminal SOC repair enters selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .learning import (
    Adam, GaussianPolicy, LOG_STD_MAX, LOG_STD_MIN, MLP, ReplayBuffer, SAC,
    _write_json, clone_behaviour,
)
from .causal import PROTOCOL


OBSERVATION_DIM = 13
CHECKPOINT_SCHEMA = 2
TEMPERATURE_MODES = ("fixed", "auto")


def save_policy_v2(policy: GaussianPolicy, path: Path | str) -> None:
    """Store schema/protocol *inside* the numeric archive; pickle stays disabled."""
    if policy.observation_dim != OBSERVATION_DIM:
        raise ValueError("Causal v2 requires 13 observation features")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": np.asarray(CHECKPOINT_SCHEMA),
        "protocol": np.asarray(PROTOCOL),
        "observation_dim": np.asarray(OBSERVATION_DIM),
        "sizes": np.asarray(policy.network.sizes, dtype=np.int64),
        "observation_mean": policy.mean,
        "observation_scale": policy.scale,
    }
    for index, (weight, bias) in enumerate(zip(policy.network.weights, policy.network.biases)):
        arrays[f"w{index}"] = weight
        arrays[f"b{index}"] = bias
    np.savez_compressed(path, **arrays)


def load_policy_v2(path: Path | str) -> GaussianPolicy:
    """Reject legacy archives instead of silently changing their state meaning."""
    with np.load(path, allow_pickle=False) as arrays:
        required = {"schema_version", "protocol", "observation_dim", "sizes",
                    "observation_mean", "observation_scale"}
        if not required.issubset(arrays.files):
            raise ValueError("Not a versioned causal v2 checkpoint; legacy policies are unsupported")
        if (int(arrays["schema_version"]) != CHECKPOINT_SCHEMA
                or str(arrays["protocol"]) != PROTOCOL
                or int(arrays["observation_dim"]) != OBSERVATION_DIM):
            raise ValueError("Checkpoint schema or execution protocol differs from causal v2")
        sizes = tuple(int(size) for size in arrays["sizes"])
        if len(sizes) < 2 or sizes[0] != OBSERVATION_DIM or sizes[-1] != 2:
            raise ValueError("Invalid causal v2 scalar-Gaussian network shape")
        policy = GaussianPolicy(sizes[0], arrays["observation_mean"],
                                arrays["observation_scale"], sizes[1:-1])
        if not np.isfinite(policy.mean).all():
            raise ValueError("Checkpoint observation mean must be finite")
        for index, (weight, bias) in enumerate(zip(policy.network.weights, policy.network.biases)):
            source_weight, source_bias = arrays[f"w{index}"], arrays[f"b{index}"]
            if source_weight.shape != weight.shape or source_bias.shape != bias.shape:
                raise ValueError("Checkpoint parameter shape mismatch")
            weight[...] = source_weight
            bias[...] = source_bias
        if not all(np.isfinite(parameter).all() for parameter in policy.network.parameters()):
            raise ValueError("Checkpoint parameters must be finite")
        return policy


def _relative(path: Path, models_dir: Path) -> str:
    return path.relative_to(models_dir.parent).as_posix()


def evaluate_policy_v2(sim: Any, policy: GaussianPolicy) -> dict[str, Any]:
    if policy.observation_dim != OBSERVATION_DIM:
        raise ValueError("Policy observation schema differs from causal v2")
    return sim.simulate(policy=policy)


def _legal(result: dict[str, Any]) -> bool:
    summary = result["summary"]
    return (bool(summary.get("feasible", False))
            and bool(summary.get("sustainable", False))
            and bool(summary.get("coupling_converged", True))
            and abs(float(summary.get("projection_alpha", 1.0)) - 1.0) < 1e-12
            and float(summary.get("sustainability_deficit_mj", 0.0)) <= 1e-6
            and float(summary.get("periodic_speed_error_mps", 0.0)) <= 0.02
            and np.isfinite(float(summary.get("lap_time_s", 0.0)))
            and np.isfinite(float(summary.get("energy_end_mj", 0.0))))


def _score(result: dict[str, Any]) -> float:
    return float(result["summary"]["lap_time_s"]) if _legal(result) else np.inf


def _policy_statistics(policy: GaussianPolicy, observations: np.ndarray) -> dict[str, float]:
    standardised = (observations - policy.mean) / policy.scale
    output = policy.network.forward(policy.normalise(observations))
    actions = np.tanh(output[:, 0])
    standard_deviation = np.exp(np.clip(output[:, 1], LOG_STD_MIN, LOG_STD_MAX))
    return {
        "pre_tanh_sigma_mean": float(standard_deviation.mean()),
        "pre_tanh_sigma_min": float(standard_deviation.min()),
        "pre_tanh_sigma_max": float(standard_deviation.max()),
        "deterministic_action_mean": float(actions.mean()),
        "deterministic_action_abs_mean": float(np.abs(actions).mean()),
        "deterministic_action_negative_fraction": float(np.mean(actions < 0.0)),
        "normalisation_clipped_fraction": float(np.mean(np.abs(standardised) > 8.0)),
    }


class DiagnosticSAC(SAC):
    """The tested SAC derivative with temperature choice and bounded diagnostics."""

    def __init__(self, policy: GaussianPolicy, seed: int, temperature_mode: str) -> None:
        if temperature_mode not in TEMPERATURE_MODES:
            raise ValueError("Temperature must be fixed or auto")
        super().__init__(policy, seed=seed, gamma=1.0, initial_alpha=0.001)
        self.temperature_mode = temperature_mode

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        next_action, next_log_probability, _ = self.policy.sample(batch["next_obs"], self.rng)
        next_input = np.column_stack((self.policy.normalise(batch["next_obs"]), next_action))
        target_value = np.minimum(self.target_q1.forward(next_input), self.target_q2.forward(next_input))
        target = batch["rewards"] + (1.0 - batch["dones"]) * (
            target_value - self.alpha * next_log_probability
        )
        critic_input = np.column_stack((self.policy.normalise(batch["obs"]), batch["actions"]))
        losses = []
        for network, optimiser in [(self.q1, self.q1_optimiser), (self.q2, self.q2_optimiser)]:
            prediction, cache = network.forward(critic_input, cache=True)
            _, gradients = network.backward(cache, 2.0 * (prediction - target) / len(prediction))
            losses.append(float(np.mean((prediction - target)**2)))
            optimiser.step(gradients)
        actor_loss, actor_gradients, log_probability = self.actor_loss_and_gradients(batch["obs"])
        actor_gradient_norm = np.sqrt(sum(float(np.sum(g*g)) for g in actor_gradients))
        self.actor_optimiser.step(actor_gradients)
        if self.temperature_mode == "auto":
            gradient = -float(np.mean(log_probability + self.target_entropy))
            self.alpha_optimiser.step([np.asarray([gradient])])
            self.log_alpha[...] = np.clip(self.log_alpha, np.log(1e-5), np.log(0.5))
        self.target_q1.polyak_update(self.q1, self.tau)
        self.target_q2.polyak_update(self.q2, self.tau)
        self.updates += 1
        current_q = np.minimum(self.q1.forward(critic_input), self.q2.forward(critic_input))
        return {
            "q1_loss": losses[0], "q2_loss": losses[1], "actor_loss": actor_loss,
            "alpha": self.alpha, "actor_gradient_norm": float(actor_gradient_norm),
            "mean_log_probability": float(log_probability.mean()),
            "bellman_target_min": float(target.min()), "bellman_target_max": float(target.max()),
            "q_min": float(current_q.min()), "q_max": float(current_q.max()),
        }

    def warmup_critics(self, replay: ReplayBuffer, rng: np.random.Generator,
                       iterations: int = 120, batch_size: int = 64) -> None:
        for _ in range(iterations):
            batch = replay.sample(batch_size, rng)
            next_action, log_probability, _ = self.policy.sample(batch["next_obs"], self.rng)
            next_input = np.column_stack((self.policy.normalise(batch["next_obs"]), next_action))
            target = batch["rewards"] + (1.0 - batch["dones"]) * (
                np.minimum(self.target_q1.forward(next_input), self.target_q2.forward(next_input))
                - self.alpha * log_probability
            )
            inputs = np.column_stack((self.policy.normalise(batch["obs"]), batch["actions"]))
            for network, optimiser in [(self.q1, self.q1_optimiser), (self.q2, self.q2_optimiser)]:
                prediction, cache = network.forward(inputs, cache=True)
                _, gradients = network.backward(cache, 2.0 * (prediction - target) / len(prediction))
                optimiser.step(gradients)
            self.target_q1.polyak_update(self.q1, self.tau)
            self.target_q2.polyak_update(self.q2, self.tau)


def _returns(rewards: np.ndarray) -> np.ndarray:
    return np.cumsum(rewards[::-1])[::-1]


def critic_calibration(sac: DiagnosticSAC, trajectory: dict[str, Any]) -> dict[str, float | str]:
    observations = np.asarray(trajectory["observations"])
    actions = np.asarray(trajectory["actions"]).reshape(-1, 1)
    rewards = np.asarray(trajectory["rewards"])
    log_probability = np.asarray(trajectory["log_probabilities"])
    inputs = np.column_stack((sac.policy.normalise(observations), actions))
    q1, q2 = sac.q1.forward(inputs)[:, 0], sac.q2.forward(inputs)[:, 0]
    q = np.minimum(q1, q2)
    plain_mc = _returns(rewards)
    # Q(s_t,a_t) includes future entropy bonuses, excluding entropy of a_t.
    shifted_entropy = np.r_[-sac.alpha * log_probability[1:], 0.0]
    soft_mc = plain_mc + _returns(shifted_entropy)
    return {
        "mc_reward_return_mean": float(plain_mc.mean()),
        "mc_soft_return_mean": float(soft_mc.mean()),
        "q_mean": float(q.mean()), "q_min": float(q.min()), "q_max": float(q.max()),
        "q1_q2_mean_abs_disagreement": float(np.abs(q1-q2).mean()),
        "q_minus_plain_mc_mean": float((q-plain_mc).mean()),
        "q_minus_soft_mc_mean": float((q-soft_mc).mean()),
        "q_vs_soft_mc_rmse": float(np.sqrt(np.mean((q-soft_mc)**2))),
        "interpretation": "Realised online stochastic rollout returns; actor changes during rollout and soft MC uses current alpha; diagnostic rather than an unbiased convergence proof",
    }


def _rollout(
    sim: Any, policy: GaussianPolicy, seed: int,
    rng: np.random.Generator | None = None,
    replay: ReplayBuffer | None = None,
    sac: DiagnosticSAC | None = None,
    batch_size: int = 64,
    update_every: int = 2,
) -> dict[str, Any]:
    env = sim.create_env()
    observation = np.asarray(env.reset(seed=seed), dtype=float)
    observations, next_observations, actions = [], [], []
    rewards, dones, log_probabilities, infos, losses = [], [], [], [], []
    for step in range(int(sim.n_nodes)):
        if rng is None:
            action, log_probability = policy(observation), 0.0
        else:
            sampled, logp, _ = policy.sample(observation, rng)
            action, log_probability = float(sampled[0, 0]), float(logp[0, 0])
        next_observation, reward, done, info = env.step(action)
        observations.append(observation.copy())
        next_observations.append(np.asarray(next_observation).copy())
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        log_probabilities.append(log_probability)
        infos.append(info)
        if replay is not None:
            replay.add(observation, action, float(reward), next_observation, done)
        if (sac is not None and replay is not None and replay.size >= batch_size
                and (step + 1) % update_every == 0):
            losses.append(sac.update(replay.sample(batch_size, rng)))
        observation = np.asarray(next_observation, dtype=float)
        if done:
            break
    if not dones or not dones[-1]:
        raise RuntimeError("Causal environment failed to terminate within one lap")
    return {"observations": np.asarray(observations), "next_observations": np.asarray(next_observations),
            "actions": np.asarray(actions), "rewards": np.asarray(rewards),
            "dones": np.asarray(dones), "log_probabilities": np.asarray(log_probabilities),
            "infos": infos, "losses": losses,
            "completed_lap": len(actions) == int(sim.n_nodes)
                             and all(bool(i.get("step_certified", True)) for i in infos)}


def _summary_rollout(trajectory: dict[str, Any]) -> dict[str, Any]:
    infos = trajectory["infos"]
    requested = np.asarray(trajectory["actions"])
    shielded = np.asarray([i.get("shielded_action", a) for i, a in zip(infos, requested)])
    executed = np.asarray([i.get("executed_action", a) for i, a in zip(infos, shielded)])
    terminal = infos[-1]
    reward_time = sum(float(i.get("reward_time_s", 0.0)) for i in infos)
    energy_penalty = sum(float(i.get("reward_terminal_energy_s", 0.0)) for i in infos)
    speed_penalty = sum(float(i.get("reward_terminal_speed_s", 0.0)) for i in infos)
    failure_penalty = sum(float(i.get("reward_failure_s", 0.0)) for i in infos)
    total = float(np.sum(trajectory["rewards"]))
    return {
        "total_reward_s": total, "time_reward_s": reward_time,
        "terminal_energy_reward_s": energy_penalty, "terminal_speed_reward_s": speed_penalty,
        "failure_reward_s": failure_penalty,
        "reward_decomposition_residual_s": total-reward_time-energy_penalty-speed_penalty-failure_penalty,
        "completed_lap": trajectory["completed_lap"],
        "requested_to_shielded_mean_abs": float(np.mean(np.abs(requested-shielded))),
        "shielded_to_executed_mean_abs": float(np.mean(np.abs(shielded-executed))),
        "shielded_fraction": float(np.mean(np.abs(requested-shielded) > 1e-6)),
        "reserve_reduction_kw_sum": float(sum(i.get("reserve_reduction_kw", 0.0) for i in infos)),
        "steps_certified": all(bool(i.get("step_certified", True)) for i in infos),
        "terminal_info": terminal,
    }


def _state_parity(sim: Any, policy: GaussianPolicy, official: dict[str, Any], seed: int) -> dict[str, Any]:
    trajectory = _rollout(sim, policy, seed)
    observed = np.asarray(official["observations"])
    maximum_state_error = float(np.max(np.abs(observed-trajectory["observations"])))
    lap_time = float(sum(i["dt_s"] for i in trajectory["infos"]))
    energy = float(trajectory["infos"][-1]["energy_mj"])
    official_energy = float(official["summary"]["energy_end_mj"])
    return {
        "maximum_observation_difference": maximum_state_error,
        "lap_time_difference_s": lap_time-float(official["summary"]["lap_time_s"]),
        "terminal_energy_difference_mj": energy-official_energy,
        "passed": maximum_state_error <= 1e-7
                  and abs(lap_time-float(official["summary"]["lap_time_s"])) <= 1e-7
                  and abs(energy-official_energy) <= 1e-7,
    }


def _seed_replay(trajectory: dict[str, Any], n_nodes: int, episodes: int) -> ReplayBuffer:
    replay = ReplayBuffer(max(20000, n_nodes * (episodes+1)), OBSERVATION_DIM)
    for values in zip(trajectory["observations"], trajectory["actions"], trajectory["rewards"],
                      trajectory["next_observations"], trajectory["dones"]):
        replay.add(*values)
    return replay


def _failed_candidate(sim: Any, error: str) -> dict[str, Any]:
    """No accepted trajectory can be invented for a policy evaluation exception."""
    scalar_trace_keys = (
        "s_m", "x_m", "y_m", "speed_mps", "speed_next_mps", "dt_s", "ax_mps2", "ay_mps2",
        "mguk_dc_kw", "mguk_mechanical_kw", "ice_mechanical_kw", "friction_brake_kw", "wheel_power_kw",
        "deployment_cap_kw", "requested_actions", "shielded_actions", "executed_actions", "aero_mode",
        "reserve_mj", "deployable_credit_mj", "requested_dc_kw", "reserve_reduction_kw",
        "power_lower_kw", "power_upper_kw", "slew_lower_kw", "mechanical_work_residual_mj",
        "energy_balance_residual_mj",
    )
    result: dict[str, Any] = {key: np.empty(0) for key in scalar_trace_keys}
    result.update(observations=np.empty((0, OBSERVATION_DIM)), time_s=np.asarray([0.]),
                  energy_mj=np.empty(0), recharge_mj=np.empty(0), soc_fraction=np.empty(0),
                  energy_floor_mj=np.empty(0), operating_mode=np.empty(0, dtype=str),
                  guard_reasons=[], guard_diagnostics=[{"policy_evaluation_exception": error}],
                  slew_exemptions=[], step_certified=np.empty(0, dtype=bool))
    result["summary"] = {
        "protocol": PROTOCOL, "lap_time_s": None, "energy_end_mj": None,
        "feasible": False, "sustainable": False, "execution_feasible": False,
        "completed_lap": False, "coupling_converged": False,
        "periodic_speed_error_mps": None, "sustainability_deficit_mj": None,
        "projection_alpha": 1.0, "whole_lap_scaling_used": False,
        "evaluation_exception": error,
        "interpretation": "Policy evaluation failed; no accepted lap or replacement of the feasible incumbent",
    }
    return result


def _sac_run(
    sim: Any, bc_policy: GaussianPolicy, bc_result: dict[str, Any],
    initial_trajectory: dict[str, Any], seed: int, mode: str, episodes: int,
    directory: Path, models_dir: Path, progress: Any,
) -> tuple[dict[str, Any], dict[str, Any], GaussianPolicy, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    policy = bc_policy.copy()
    sac = DiagnosticSAC(policy, seed, mode)
    rng = np.random.default_rng(seed + 104729)
    replay = _seed_replay(initial_trajectory, int(sim.n_nodes), episodes)
    sac.warmup_critics(replay, rng)
    best_policy, best_result, best_episode = policy.copy(), bc_result, 0
    best_score = _score(best_result)
    history = [{"episode": 0, "stage": "bc", "evaluation": bc_result["summary"],
                "selected_as_best": np.isfinite(best_score)}]
    failure = None
    last_result = bc_result
    for episode in range(1, episodes+1):
        try:
            trajectory = _rollout(sim, policy, seed+episode, rng=rng, replay=replay,
                                  sac=sac, batch_size=64, update_every=2)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            failure = f"{type(error).__name__}: {error}"
            history.append({"episode": episode, "failure": failure})
            break
        record = {"episode": episode, "stage": "sac", "alpha": sac.alpha,
                  "updates": sac.updates, "rollout": _summary_rollout(trajectory),
                  "policy_statistics": _policy_statistics(policy, trajectory["observations"]),
                  "critic_calibration": critic_calibration(sac, trajectory)}
        if not trajectory["completed_lap"]:
            failure = "Causal guard terminated episode before a certified complete lap"
            record["failure"] = failure
            record["guard_diagnostics"] = trajectory["infos"]
            history.append(record)
            break
        if trajectory["losses"]:
            record["losses"] = {key: float(np.mean([loss[key] for loss in trajectory["losses"]]))
                                 for key in trajectory["losses"][0]}
        if episode % 5 == 0 or episode == episodes:
            last_result = evaluate_policy_v2(sim, policy)
            record["evaluation"] = last_result["summary"]
            improved = _score(last_result) < best_score-1e-9
            record["selected_as_best"] = improved
            if improved:
                best_result, best_policy, best_episode = last_result, policy.copy(), episode
                best_score = _score(last_result)
            if progress:
                value = f"{_score(last_result):.5f} s" if _legal(last_result) else "infeasible"
                progress(f"v2 {mode}, seed {seed}, episode {episode}/{episodes}: {value}")
        history.append(record)
    # A non-finite actor may have caused the caught rollout exception.  Retrying
    # it must not abort failure reporting or discard the known-good incumbent.
    try:
        last_result = evaluate_policy_v2(sim, policy)
    except (FloatingPointError, ValueError, RuntimeError) as error:
        evaluation_error = f"{type(error).__name__}: {error}"
        failure = failure or evaluation_error
        last_result = _failed_candidate(sim, evaluation_error)
        _write_json(directory/"last_candidate_failure.json", last_result["summary"])
    final_parameters_finite = all(np.isfinite(p).all() for p in policy.network.parameters())
    last_checkpoint = directory/"last_candidate.npz" if final_parameters_finite else None
    if last_checkpoint is not None:
        save_policy_v2(policy, last_checkpoint)
    checkpoint = directory/"best_feasible_policy.npz"
    save_policy_v2(best_policy, checkpoint)
    reloaded = evaluate_policy_v2(sim, load_policy_v2(checkpoint))
    reproducible = abs(float(reloaded["summary"]["lap_time_s"])-float(best_result["summary"]["lap_time_s"])) <= 1e-8
    if not reproducible:
        raise RuntimeError("Reloaded causal v2 checkpoint does not reproduce its selected lap")
    best_result = reloaded
    parity = _state_parity(sim, best_policy, best_result, seed)
    if not parity["passed"]:
        raise RuntimeError("Formal evaluation and step environment differ; training result rejected")
    metadata = {
        "seed": seed, "temperature_mode": mode, "episodes_requested": episodes,
        "episodes_completed": max((r["episode"] for r in history if "failure" not in r), default=0),
        "best_episode": best_episode, "best_checkpoint": _relative(checkpoint, models_dir),
        "last_checkpoint": _relative(last_checkpoint, models_dir) if last_checkpoint else None,
        "best_summary": best_result["summary"], "last_summary": last_result["summary"],
        "paired_improvement_over_bc_s": float(bc_result["summary"]["lap_time_s"])-float(best_result["summary"]["lap_time_s"]),
        "checkpoint_roundtrip_passed": reproducible, "state_parity": parity,
        "sac_updates": sac.updates, "training_failure": failure,
    }
    _write_json(directory/"sac_curve.json", history)
    _write_json(directory/"metadata.json", metadata)
    return metadata, {"last": last_result, "selected": best_result}, best_policy, trajectory if episodes and failure is None else initial_trajectory


def select_ablation_group(records: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    """A preregistered threshold bounds work; failed runs cannot pass by BC fallback."""
    groups = []
    for mode in TEMPERATURE_MODES:
        candidates = [r for r in records if r["temperature_mode"] == mode]
        complete = len(candidates) == len(seeds) and {r["seed"] for r in candidates} == set(seeds)
        all_legal = complete and all(_legal({"summary": r["best_summary"]}) for r in candidates)
        all_finished = complete and all(r["episodes_completed"] == r["episodes_requested"]
                                        and r["training_failure"] is None for r in candidates)
        improvement = float(np.mean([r["paired_improvement_over_bc_s"] for r in candidates])) if candidates else None
        groups.append({"temperature_mode": mode, "all_seeds_legal": all_legal,
                       "all_runs_completed": all_finished, "mean_paired_improvement_s": improvement,
                       "eligible": bool(all_legal and all_finished and improvement >= 0.05)})
    eligible = [group for group in groups if group["eligible"]]
    # Iteration order makes a tie choose fixed temperature.
    winner = max(eligible, key=lambda g: g["mean_paired_improvement_s"])["temperature_mode"] if eligible else None
    return {"winner": winner, "groups": groups, "threshold_s": 0.05,
            "rule": "All requested seeds legally evaluated and completed; mean paired best-checkpoint gain >=0.05s; exact tie chooses fixed"}


def train_and_evaluate_v2(
    sim: Any, teacher_result: dict[str, Any], output_dir: Path | str,
    seeds: tuple[int, ...] = (0, 1, 2), bc_epochs: int = 600,
    diagnostic_episodes: int = 20, formal_episodes: int = 60, progress: Any = None,
) -> dict[str, Any]:
    """BC/DAgger -> fixed/auto 20-lap diagnostics -> conditional 60-lap rerun.

    A formal run restarts from each seed's BC policy, not from its diagnostic
    winner, so the two phases have unambiguous budgets.  Every final candidate
    competes against the same feasible BC incumbent; failed training keeps BC.
    Caller owns the public-data/calibration gate and must pass it before calling.
    """
    if not seeds or len(set(seeds)) != len(seeds) or bc_epochs < 1:
        raise ValueError("Require distinct nonempty seeds and positive BC epochs")
    if not 1 <= diagnostic_episodes <= 20 or not 1 <= formal_episodes <= 60:
        raise ValueError("Bounded v2 budget: 1..20 diagnostic episodes and 1..60 formal episodes")
    if not _legal(teacher_result):
        raise ValueError("Causal teacher fails physical, terminal or protocol audit")
    observations = np.asarray(teacher_result["observations"], dtype=float)
    teacher_actions = np.asarray(teacher_result["requested_actions"], dtype=float)
    if observations.shape != (int(sim.n_nodes), OBSERVATION_DIM):
        raise ValueError("Causal v2 teacher must contain one ordered 13-feature state per node")
    if teacher_actions.shape != (int(sim.n_nodes),):
        raise ValueError("Causal teacher action trajectory has an incompatible shape")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bc_records, ablation_records, formal_records = [], [], []
    results, bc_policies, bc_results, bc_trajectories = {}, {}, {}, {}
    for seed in seeds:
        seed_dir = output_dir/f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"v2 seed {seed}: BC {bc_epochs} epochs, three feedback aggregation rounds")
        policy, history = clone_behaviour(observations, teacher_actions, seed=seed, epochs=bc_epochs)
        raw_result = evaluate_policy_v2(sim, policy)
        save_policy_v2(policy, seed_dir/"raw_bc_policy.npz")
        best_policy, best_result, selected_round = policy.copy(), raw_result, 0
        datasets, labels = [observations], [teacher_actions]
        aggregation = []
        for round_index in range(1, 4):
            rollout = evaluate_policy_v2(sim, policy)
            state = np.asarray(rollout["observations"], dtype=float)
            if state.ndim != 2 or state.shape[1] != OBSERVATION_DIM or len(state) > len(observations) or not np.isfinite(state).all():
                raise RuntimeError("Causal learner rollout has invalid states")
            # A rejected physical proposal may end a rollout early. Accepted
            # states remain an ordered track prefix, and can still be labelled.
            if len(state):
                datasets.append(state)
                labels.append(teacher_actions[:len(state)].copy())
            policy, curve = clone_behaviour(np.vstack(datasets), np.concatenate(labels),
                                             seed=seed+round_index*1009, epochs=600,
                                             initial_policy=policy)
            candidate = evaluate_policy_v2(sim, policy)
            improved = _score(candidate) < _score(best_result)-1e-9
            if improved:
                best_policy, best_result, selected_round = policy.copy(), candidate, round_index
            aggregation.append({"round": round_index, "samples": len(np.vstack(datasets)),
                                "aggregate_action_rmse": curve[-1]["action_rmse"],
                                "evaluation": candidate["summary"], "selected_as_best": improved})
            _write_json(seed_dir/f"dagger_round_{round_index}_curve.json", curve)
            save_policy_v2(policy, seed_dir/f"dagger_round_{round_index}_candidate.npz")
            if progress:
                value = f"{_score(candidate):.5f}s" if _legal(candidate) else "infeasible"
                progress(f"v2 seed {seed}, DAgger round {round_index}/3: {value}")
        if not _legal(best_result):
            _write_json(seed_dir/"bc_failed.json", {"raw": raw_result["summary"], "aggregation": aggregation})
            raise RuntimeError(f"No legal BC/DAgger controller for seed {seed}; SAC experiment blocked")
        checkpoint = seed_dir/"bc_policy.npz"
        save_policy_v2(best_policy, checkpoint)
        best_result = evaluate_policy_v2(sim, load_policy_v2(checkpoint))
        parity = _state_parity(sim, best_policy, best_result, seed)
        if not parity["passed"]:
            raise RuntimeError("Causal BC checkpoint fails environment/evaluation parity")
        record = {"seed": seed, "selected_aggregation_round": selected_round,
                  "raw_action_rmse": history[-1]["action_rmse"], "raw_summary": raw_result["summary"],
                  "bc_summary": best_result["summary"], "bc_checkpoint": _relative(checkpoint, output_dir),
                  "aggregation": aggregation, "state_parity": parity,
                  "policy_statistics": _policy_statistics(best_policy, observations)}
        _write_json(seed_dir/"bc_curve.json", history)
        _write_json(seed_dir/"bc_metadata.json", record)
        bc_records.append(record)
        bc_policies[seed], bc_results[seed] = best_policy, best_result
        bc_trajectories[seed] = _rollout(sim, best_policy, seed)
        results[str(seed)] = {"bc": best_result, "selected": best_result}
    for mode in TEMPERATURE_MODES:
        for seed in seeds:
            record, candidates, _, _ = _sac_run(
                sim, bc_policies[seed], bc_results[seed], bc_trajectories[seed], seed, mode,
                diagnostic_episodes, output_dir/f"seed_{seed}"/f"diagnostic_{mode}", output_dir, progress)
            ablation_records.append(record)
            results[str(seed)][f"diagnostic_{mode}_last"] = candidates["last"]
            results[str(seed)][f"diagnostic_{mode}_selected"] = candidates["selected"]
    gate = select_ablation_group(ablation_records, seeds)
    if gate["winner"] is not None:
        for seed in seeds:
            record, candidates, _, _ = _sac_run(
                sim, bc_policies[seed], bc_results[seed], bc_trajectories[seed], seed, gate["winner"],
                formal_episodes, output_dir/f"seed_{seed}"/"formal", output_dir, progress)
            formal_records.append(record)
            results[str(seed)]["formal_last"] = candidates["last"]
            results[str(seed)]["selected"] = candidates["selected"]
    selected_times = [results[str(seed)]["selected"]["summary"]["lap_time_s"] for seed in seeds]
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA, "protocol": PROTOCOL,
        "algorithm": "NumPy BC/DAgger plus bounded continuous SAC temperature ablation",
        "observation_dim": OBSERVATION_DIM, "seeds": list(seeds), "bc_epochs": bc_epochs,
        "aggregation_rounds": 3, "aggregation_epochs": 600,
        "aggregation_expert": "Frozen same-node teacher requested actions for training labels only; deployment remains numeric MLP",
        "diagnostic_episodes": diagnostic_episodes, "formal_episodes": formal_episodes,
        "actual_training_episode_budget": len(seeds)*len(TEMPERATURE_MODES)*diagnostic_episodes
                                          + (len(seeds)*formal_episodes if gate["winner"] is not None else 0),
        "diagnostic_temperature_modes": list(TEMPERATURE_MODES), "initial_alpha": 0.001,
        "gamma": 1.0, "tau": 0.005, "batch_size": 64, "update_every": 2,
        "learning_rate": 3e-4, "critic_warmup_updates": 120, "evaluation_every": 5,
        "paired_protocol": "Per seed identical BC actor, critics, seeded BC replay, common certified speed boundary and initial SOC; stochastic streams start identically",
        "reward_protocol": "Undiscounted simulator reward; terminal constraints separately audited",
        "bc_records": bc_records, "ablation_records": ablation_records, "formal_records": formal_records,
        "expansion_gate": gate, "formal_run_performed": gate["winner"] is not None,
        "checkpoint_paths_relative_to": "experiment directory (parent of models directory)",
        "teacher_lap_time_s": float(teacher_result["summary"]["lap_time_s"]),
        "mean_selected_lap_time_s": float(np.mean(selected_times)),
        "std_selected_lap_time_s": float(np.std(selected_times)),
        "selection_rule": "Per-seed feasible best checkpoint including BC episode zero; no replacement by infeasible or slower candidates",
        "claim": "Single-track public-data numerical experiment; neither real SOC validation nor guaranteed SAC improvement",
    }
    _write_json(output_dir/"training_summary.json", metadata)
    controllers = []
    labels = {
        "bc": "因果 BC / DAgger", "selected": "因果保留最佳",
        "diagnostic_fixed_last": "SAC 固定温度诊断末次",
        "diagnostic_fixed_selected": "SAC 固定温度诊断保留",
        "diagnostic_auto_last": "SAC 自动温度诊断末次",
        "diagnostic_auto_selected": "SAC 自动温度诊断保留",
        "formal_last": "SAC 正式训练末次",
    }
    for seed, candidates in results.items():
        seed_value = int(seed)
        bc_record = next(record for record in bc_records if record["seed"] == seed_value)
        diagnostic_records = {record["temperature_mode"]: record for record in ablation_records
                              if record["seed"] == seed_value}
        formal_record = next((record for record in formal_records if record["seed"] == seed_value), None)
        checkpoints = {"bc": bc_record["bc_checkpoint"],
                       "selected": formal_record["best_checkpoint"] if formal_record else bc_record["bc_checkpoint"]}
        for mode, record in diagnostic_records.items():
            checkpoints[f"diagnostic_{mode}_last"] = record["last_checkpoint"]
            checkpoints[f"diagnostic_{mode}_selected"] = record["best_checkpoint"]
        if formal_record:
            checkpoints["formal_last"] = formal_record["last_checkpoint"]
        for stage, candidate in candidates.items():
            controllers.append({"id": f"{stage}_seed_{seed}",
                                "label": f"{labels[stage]} · seed {seed}", "result": candidate,
                                "checkpoint": checkpoints[stage]})
    return {"metadata": metadata, "results": results, "controllers": controllers}
