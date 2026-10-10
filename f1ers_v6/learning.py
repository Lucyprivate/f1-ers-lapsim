"""Versioned state-feedback BC/DAgger and bounded NumPy SAC experiments.

Only numerical network primitives are reused from frozen historical packages.
All rollouts, replay transitions and selection use the v6 executor.  The
same-position DAgger expert is disclosed as a fixed-label single-track expert;
it is never queried by the deployed network.
"""
from __future__ import annotations

from importlib import import_module
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import gzip
import json
import numpy as np

from f1ers.learning import GaussianPolicy, MLP, ReplayBuffer, SAC, clone_behaviour
from .reward import RewardConfig

CHECKPOINT_SCHEMA = 6
POLICY_PROTOCOL = "v6_state_feedback_gaussian_v1"
ACTION_SPACES = ("signed", "mode_aware")


class RolloutFailure(RuntimeError):
    def __init__(self, error, trace):
        super().__init__(str(error))
        self.trace = trace


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(_jsonable(value), ensure_ascii=False, indent=2)
    if path.suffix == ".gz":
        # A reproducible header also makes archive hashes stable.
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(serialised.encode("utf-8"))
    else:
        path.write_text(serialised, encoding="utf-8")


def _features(sim, env=None):
    names = getattr(sim, "features", None) or getattr(env, "features", None)
    if names is None:
        names = getattr(import_module(type(sim).__module__), "FEATURES", None)
    if names is None:
        raise ValueError("Simulator must publish an explicit ordered v6 feature schema")
    names = tuple(str(v) for v in names)
    dimension = int(getattr(env, "observation_dim", getattr(sim, "observation_dim", len(names))))
    if len(names) != dimension or not names or len(set(names)) != len(names):
        raise ValueError("Feature names must uniquely match the current observation dimension")
    return names


def _bind(policy, features, action_space, execution_protocol="v6"):
    if action_space not in ACTION_SPACES:
        raise ValueError("Unknown action-space schema")
    features = tuple(features)
    if policy.observation_dim != len(features):
        raise ValueError("Network dimension differs from its feature schema")
    policy.v6_features = features
    policy.v6_action_space = action_space
    policy.v6_execution_protocol = str(execution_protocol)
    return policy


def _copy(policy):
    return _bind(policy.copy(), policy.v6_features, policy.v6_action_space, policy.v6_execution_protocol)


def save_policy(policy, path, *, features=None, action_space=None, execution_protocol=None):
    features = tuple(features or getattr(policy, "v6_features", ()))
    action_space = action_space or getattr(policy, "v6_action_space", None)
    execution_protocol = execution_protocol or getattr(policy, "v6_execution_protocol", "v6")
    _bind(policy, features, action_space, execution_protocol)
    if len(set(features)) != len(features) or not all(features):
        raise ValueError("Checkpoint requires unique nonempty feature names")
    if not np.isfinite(policy.mean).all() or not all(np.isfinite(p).all() for p in policy.network.parameters()):
        raise ValueError("Checkpoint parameters must be finite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": np.asarray(CHECKPOINT_SCHEMA),
        "protocol": np.asarray(POLICY_PROTOCOL),
        "execution_protocol": np.asarray(execution_protocol),
        "action_space": np.asarray(action_space),
        "features": np.asarray(features, dtype=str),
        "sizes": np.asarray(policy.network.sizes, dtype=np.int64),
        "observation_mean": policy.mean,
        "observation_scale": policy.scale,
    }
    for index, (weight, bias) in enumerate(zip(policy.network.weights, policy.network.biases)):
        arrays[f"w{index}"] = weight
        arrays[f"b{index}"] = bias
    np.savez_compressed(path, **arrays)


def load_policy(path, *, expected_features=None, expected_action_space=None,
                expected_execution_protocol=None):
    with np.load(path, allow_pickle=False) as arrays:
        required = {"schema_version", "protocol", "execution_protocol", "action_space", "features",
                    "sizes", "observation_mean", "observation_scale"}
        if not required.issubset(arrays.files):
            raise ValueError("Missing v6 checkpoint schema; legacy policies cannot be reinterpreted")
        if int(arrays["schema_version"]) != CHECKPOINT_SCHEMA or str(arrays["protocol"]) != POLICY_PROTOCOL:
            raise ValueError("Unsupported policy schema or protocol")
        features = tuple(str(v) for v in arrays["features"])
        space = str(arrays["action_space"])
        execution = str(arrays["execution_protocol"])
        if expected_features is not None and features != tuple(expected_features):
            raise ValueError("Checkpoint feature order differs from this executor")
        if expected_action_space is not None and space != expected_action_space:
            raise ValueError("Checkpoint action mapping differs from this experiment")
        if expected_execution_protocol is not None and execution != expected_execution_protocol:
            raise ValueError("Checkpoint execution protocol differs")
        sizes = tuple(int(v) for v in arrays["sizes"])
        if len(sizes) < 2 or sizes[0] != len(features) or sizes[-1] != 2 or any(v < 1 for v in sizes):
            raise ValueError("Invalid scalar Gaussian network shape")
        if len(set(features)) != len(features) or not all(features):
            raise ValueError("Checkpoint feature schema is invalid")
        policy = GaussianPolicy(sizes[0], arrays["observation_mean"], arrays["observation_scale"], sizes[1:-1])
        for index, (weight, bias) in enumerate(zip(policy.network.weights, policy.network.biases)):
            if arrays[f"w{index}"].shape != weight.shape or arrays[f"b{index}"].shape != bias.shape:
                raise ValueError("Checkpoint parameter shape mismatch")
            weight[...] = arrays[f"w{index}"]
            bias[...] = arrays[f"b{index}"]
        if not np.isfinite(policy.mean).all() or not all(np.isfinite(p).all() for p in policy.network.parameters()):
            raise ValueError("Checkpoint contains non-finite parameters")
        return _bind(policy, features, space, execution)


def is_legal(result):
    """A missing certificate is a rejection, never an implicit pass."""
    summary = result.get("summary", {})
    flags = ("completed_lap", "execution_feasible", "feasible", "sustainable", "public_rules_passed",
             "periodic_speed_closed", "dc_boundary_closed", "rule_state_closed")
    if not all(isinstance(summary.get(key), (bool, np.bool_)) and bool(summary[key]) for key in flags):
        return False
    if "shooting" in summary and not bool(summary["shooting"].get("converged", False)):
        return False
    try:
        lap_time = float(summary["lap_time_s"])
        start, end = float(summary["energy_start_mj"]), float(summary["energy_end_mj"])
        residual = float(summary["periodic_speed_residual_mps"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite([lap_time, start, end, residual]).all() and lap_time > 0
                and start >= 0 and end >= start-1e-8 and 0 <= residual <= .02)


def _score(result):
    return float(result["summary"]["lap_time_s"]) if is_legal(result) else np.inf


def _env(sim, action_space, reward_config, initial_speed_mps, carried_rule_state):
    return sim.create_env(action_space=action_space, reward_config=reward_config,
                          initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)


def _rollout(sim, policy, *, seed=0, action_space="signed", reward_config=None,
             initial_speed_mps=None, carried_rule_state=None, rng=None, replay=None,
             sac=None, batch_size=64, update_every=2):
    env = _env(sim, action_space, reward_config, initial_speed_mps, carried_rule_state)
    features = _features(sim, env)
    if tuple(getattr(policy, "v6_features", ())) != features or getattr(policy, "v6_action_space", None) != action_space:
        raise ValueError("Policy and executor feature/action schemas differ")
    if getattr(policy, "v6_execution_protocol", None) != str(getattr(sim, "protocol", "v6")):
        raise ValueError("Policy and executor execution protocols differ")
    observation = np.asarray(env.reset(seed=seed), float)
    rows = {key: [] for key in ("observations", "next_observations", "actions", "rewards", "dones", "log_probabilities")}
    infos, losses = [], []
    max_steps = int(getattr(sim, "n_decisions", getattr(sim, "n_nodes", 0))) + 2
    if max_steps < 3:
        max_steps = int(np.ceil(float(sim.length_m)/5.0)) + 2
    def failure(error, proposal=None):
        trace = {key: np.asarray(values) for key, values in rows.items()}
        trace.update(infos=list(infos), last_accepted_state=_jsonable(getattr(env, "state", None)),
                     error=f"{type(error).__name__}: {error}", failure_proposal=proposal,
                     accepted_trajectory_invented=False)
        return RolloutFailure(trace["error"], trace)
    for index in range(max_steps):
        try:
            if rng is None:
                action = policy(observation)
                _, logp, _ = policy.sample(observation, np.random.default_rng(0), epsilon=np.zeros((1, 1)))
                log_probability = float(logp[0, 0])
            else:
                actions, logp, _ = policy.sample(observation, rng)
                action, log_probability = float(actions[0, 0]), float(logp[0, 0])
            next_observation, reward, done, info = env.step(action)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            raise failure(error) from error
        next_observation = np.asarray(next_observation, float)
        if observation.shape != (len(features),) or next_observation.shape != observation.shape:
            raise failure(ValueError("Executor observation changed its shape"), info)
        if not np.isfinite(observation).all() or not np.isfinite(next_observation).all() or not np.isfinite(reward):
            raise failure(FloatingPointError("Non-finite accepted state or reward"), info)
        values = (observation.copy(), next_observation.copy(), action, float(reward), bool(done), log_probability)
        for key, value in zip(rows, values):
            rows[key].append(value)
        infos.append(info)
        if replay is not None:
            replay.add(observation, action, float(reward), next_observation, done)
        if sac is not None and replay is not None and replay.size >= batch_size and (index+1) % update_every == 0:
            try:
                losses.append(sac.update(replay.sample(batch_size, rng)))
            except (FloatingPointError, ValueError, RuntimeError) as error:
                raise failure(error) from error
        observation = next_observation
        if done:
            break
    if not rows["dones"] or not rows["dones"][-1]:
        raise failure(RuntimeError("Executor failed to terminate within the declared lap"))
    trajectory = {key: np.asarray(values) for key, values in rows.items()}
    trajectory.update(infos=infos, losses=losses, result=env.result(), features=features,
                      action_space=action_space, stochastic=rng is not None)
    return trajectory


def evaluate_policy(sim, policy, *, action_space=None, reward_config=None, initial_speed_mps=None,
                    carried_rule_state=None, seed=0, periodic=True):
    space = action_space or getattr(policy, "v6_action_space", None)
    if tuple(getattr(policy, "v6_features", ())) != _features(sim) or getattr(policy, "v6_action_space", None) != space:
        raise ValueError("Policy and executor feature/action schemas differ")
    if getattr(policy, "v6_execution_protocol", None) != str(getattr(sim, "protocol", "v6")):
        raise ValueError("Policy and executor execution protocols differ")
    kwargs = {"controller": policy, "action_space": space, "reward_config": reward_config,
              "initial_speed_mps": initial_speed_mps, "carried_rule_state": carried_rule_state}
    if periodic:
        if not hasattr(sim, "periodic_simulate"):
            raise ValueError("Formal policy evaluation requires the v6 periodic shooting solver")
        return sim.periodic_simulate(**kwargs)
    return sim.simulate(**kwargs)


def trajectory_diagnostics(trajectory):
    infos = trajectory["infos"]
    summary = trajectory["result"].get("summary", {})
    decompositions = [info.get("reward_terms", {}) for info in infos]
    rewards = np.asarray(trajectory["rewards"], float)
    projection = [float(info.get("projection_fraction", 0.0)) for info in infos]
    return {
        "reward_total_s": float(rewards.sum()),
        "reward_time_s": sum(float(t.get("reward_time_s", 0.0)) for t in decompositions),
        "reward_potential_s": sum(float(t.get("reward_potential_s", 0.0)) for t in decompositions),
        "reward_failure_s": sum(float(t.get("reward_failure_s", 0.0)) for t in decompositions),
        "mean_projection_fraction": float(np.mean(projection)) if projection else 0.0,
        "projected_step_fraction": float(np.mean(np.asarray(projection) > 1e-8)) if projection else 0.0,
        "accepted_step_fraction": float(np.mean([bool(i.get("step_certified", False)) for i in infos])),
        "step_count": len(rewards), "legal": is_legal(trajectory["result"]),
        "summary": summary,
    }


def _teacher_arrays(teacher_rollouts, features):
    if isinstance(teacher_rollouts, dict):
        teacher_rollouts = [teacher_rollouts]
    teachers = list(teacher_rollouts)
    observations, actions = [], []
    for teacher in teachers:
        result = teacher.get("result", teacher)
        if not is_legal(result):
            raise ValueError("BC teacher requires a complete legal cyclic sustainable lap")
        states = np.asarray(teacher.get("observations", result.get("observations")), float)
        labels = np.asarray(teacher.get("actions", result.get("requested_actions")), float).reshape(-1)
        if states.ndim != 2 or states.shape != (len(labels), len(features)) or not len(states):
            raise ValueError("Teacher states and requested actions must follow the current feature schema")
        if not np.isfinite(states).all() or not np.isfinite(labels).all() or np.max(np.abs(labels)) > 1+1e-8:
            raise ValueError("Teacher observations/actions must be finite and normalised")
        observations.append(states)
        actions.append(labels)
    if not teachers:
        raise ValueError("At least one teacher rollout is required")
    return teachers, observations, actions


def train_bc(teacher_rollouts, simulator, output_dir, *, seed=0, epochs=600, dagger_rounds=3,
             dagger_epochs=200, action_space="signed", reward_config=None, initial_speed_mps=None,
             carried_rule_state=None, progress=None):
    if not 1 <= epochs <= 600 or not 0 <= dagger_rounds <= 3 or not 1 <= dagger_epochs <= 200:
        raise ValueError("Bounded BC budget: at most 600 epochs and three 200-epoch aggregation rounds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = _features(simulator)
    teachers, datasets, labels = _teacher_arrays(teacher_rollouts, features)
    protocol = str(getattr(simulator, "protocol", "v6"))
    policy, curve = clone_behaviour(np.vstack(datasets), np.concatenate(labels), seed=seed, epochs=epochs)
    _bind(policy, features, action_space, protocol)
    save_policy(policy, output_dir/"raw_bc_policy.npz")
    raw = _rollout(simulator, policy, seed=seed, action_space=action_space, reward_config=reward_config,
                   initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
    raw_result = evaluate_policy(simulator, policy, action_space=action_space, reward_config=reward_config,
                                 initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
    best_score = _score(raw_result)
    best_policy = _copy(policy) if np.isfinite(best_score) else None
    best_result, selected_round = raw_result, 0 if best_policy is not None else None
    trajectories, aggregation = [raw], []
    # Explicit fixed same-position labels from the first frozen numerical
    # teacher.  The executor remains authoritative for energy/rule protection.
    fixed_labels = labels[0]
    for round_index in range(1, dagger_rounds+1):
        learner = _rollout(simulator, policy, seed=seed, action_space=action_space, reward_config=reward_config,
                           initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
        learner_states = learner["observations"]
        if len(learner_states) > len(fixed_labels):
            raise ValueError("Same-position expert cannot label beyond its certified route")
        datasets.append(learner_states)
        labels.append(fixed_labels[:len(learner_states)].copy())
        policy, round_curve = clone_behaviour(np.vstack(datasets), np.concatenate(labels),
                                               seed=seed+1009*round_index, epochs=dagger_epochs,
                                               initial_policy=policy)
        _bind(policy, features, action_space, protocol)
        candidate = _rollout(simulator, policy, seed=seed, action_space=action_space, reward_config=reward_config,
                             initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
        formal_result = evaluate_policy(simulator, policy, action_space=action_space, reward_config=reward_config,
                                        initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
        score = _score(formal_result)
        improved = score < best_score-1e-9
        if improved:
            best_score, best_result, best_policy, selected_round = score, formal_result, _copy(policy), round_index
        save_policy(policy, output_dir/f"dagger_round_{round_index}_candidate.npz")
        _write(output_dir/f"dagger_round_{round_index}_curve.json", round_curve)
        _write(output_dir/f"dagger_round_{round_index}_trajectory.json.gz", candidate)
        aggregation.append({"round": round_index, "samples": sum(len(v) for v in datasets),
                            "action_rmse": round_curve[-1]["action_rmse"],
                            "training_rollout": candidate["result"]["summary"],
                            "evaluation": formal_result["summary"], "selected_as_best": improved})
        _write(output_dir/f"dagger_round_{round_index}_formal_result.json.gz", formal_result)
        trajectories.append(candidate)
        if progress:
            progress(f"BC seed {seed}, aggregation {round_index}: {'legal' if is_legal(formal_result) else 'rejected'}")
    save_policy(policy, output_dir/"last_bc_candidate.npz")
    selected_checkpoint = None
    if best_policy is not None:
        selected_checkpoint = "best_bc_policy.npz"
        save_policy(best_policy, output_dir/selected_checkpoint)
        restored = load_policy(output_dir/selected_checkpoint, expected_features=features, expected_action_space=action_space)
        reproduced = evaluate_policy(simulator, restored, reward_config=reward_config,
                                      initial_speed_mps=float(best_result["summary"]["speed_start_mps"]),
                                      carried_rule_state=carried_rule_state, seed=seed, periodic=False)
        _require_reproducible(best_result, reproduced)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA, "algorithm": "NumPy BC / fixed-label same-track DAgger",
        "features": features, "action_space": action_space, "seed": seed, "epochs": epochs,
        "dagger_rounds": dagger_rounds, "dagger_epochs": dagger_epochs,
        "expert": "Frozen numerical teacher requested action at the same ordered route decision; no state-dependent expert replanning",
        "deployed_controller": "Current-state numeric MLP; no teacher action table lookup",
        "raw_action_rmse": curve[-1]["action_rmse"], "raw_summary": raw_result["summary"],
        "raw_training_rollout_summary": raw["result"]["summary"],
        "aggregation": aggregation, "best_summary": best_result["summary"] if best_policy is not None else None,
        "selected_round": selected_round, "selected_checkpoint": selected_checkpoint,
        "last_checkpoint": "last_bc_candidate.npz", "last_summary": formal_result["summary"] if dagger_rounds else raw_result["summary"],
        "legal_controller_found": best_policy is not None,
        "checkpoint_roundtrip_passed": best_policy is not None,
        "formal_evaluation": "Policy-specific periodic speed shooting with the same physical executor",
        "checkpoint_verification": "Fixed initial speed of the selected final shooting trajectory; compare identical initial states",
        "global_optimality_claim": False,
    }
    _write(output_dir/"bc_curve.json", curve)
    _write(output_dir/"bc_metadata.json", metadata)
    _write(output_dir/"raw_bc_trajectory.json.gz", raw)
    return {"policy": best_policy, "result": best_result if best_policy is not None else None,
            "last_policy": policy, "last_result": formal_result if dagger_rounds else raw_result,
            "metadata": metadata, "rollouts": trajectories}


def _require_reproducible(reference, result):
    if is_legal(reference) != is_legal(result):
        raise RuntimeError("Checkpoint reload changed legality")
    for key in ("lap_time_s", "energy_end_mj", "periodic_speed_residual_mps"):
        a, b = reference["summary"].get(key), result["summary"].get(key)
        if a is None or b is None or abs(float(a)-float(b)) > 1e-8:
            raise RuntimeError(f"Checkpoint reload changed {key}")
    for key in ("observations", "requested_actions", "speed_mps", "time_s", "energy_mj"):
        if key in reference and key in result and not np.allclose(reference[key], result[key], rtol=0, atol=1e-8):
            raise RuntimeError(f"Checkpoint reload changed {key}")


class DiagnosticSAC(SAC):
    """Undiscounted bounded SAC with separately controlled entropy temperature."""
    def __init__(self, policy, seed=0, temperature_mode="fixed", initial_alpha=.001):
        if temperature_mode not in ("fixed", "auto"):
            raise ValueError("Temperature mode must be fixed or auto")
        if not np.isfinite(initial_alpha) or not 1e-5 <= initial_alpha <= .5:
            raise ValueError("Entropy coefficient must stay within the declared seconds-based range")
        super().__init__(policy, seed=seed, gamma=1.0, initial_alpha=initial_alpha)
        self.temperature_mode = temperature_mode

    def update(self, batch):
        before = self.log_alpha.copy()
        metrics = super().update(batch)
        if self.temperature_mode == "fixed":
            self.log_alpha[...] = before
            metrics["alpha"] = self.alpha
        metrics["gamma"] = self.gamma
        features = self.policy.normalise(batch["obs"])
        inputs = np.column_stack((features, batch["actions"]))
        q1, q2 = self.q1.forward(inputs), self.q2.forward(inputs)
        metrics.update(q_min=float(np.minimum(q1, q2).min()), q_max=float(np.maximum(q1, q2).max()),
                       q_disagreement=float(np.mean(np.abs(q1-q2))))
        return metrics


def monte_carlo_returns(rewards, dones, *, log_probabilities=None, alpha=0.0):
    """Undiscounted realised returns, resetting at every terminal boundary.

    Q(s_t,a_t) includes entropy of future actions, excluding action a_t.
    These labels are training diagnostics, never deployed input features.
    """
    rewards = np.asarray(rewards, float).reshape(-1)
    dones = np.asarray(dones, bool).reshape(-1)
    if rewards.shape != dones.shape or not len(rewards) or not dones[-1] or not np.isfinite(rewards).all():
        raise ValueError("Monte Carlo labels require finite complete terminal episodes")
    logp = np.zeros_like(rewards) if log_probabilities is None else np.asarray(log_probabilities, float).reshape(-1)
    if logp.shape != rewards.shape or not np.isfinite(logp).all() or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Invalid entropy-return labels")
    output = np.empty_like(rewards)
    following = 0.0
    for index in range(len(rewards)-1, -1, -1):
        if dones[index]:
            following = 0.0
        entropy = -alpha*logp[index+1] if index+1 < len(rewards) and not dones[index] else 0.0
        output[index] = rewards[index] + following + entropy
        following = output[index]
    return output


def warmstart_critics(sac, trajectory, *, iterations=120, batch_size=64, seed=0, monte_carlo=False):
    if iterations < 0 or batch_size < 1:
        raise ValueError("Invalid critic warmstart budget")
    obs, actions = np.asarray(trajectory["observations"]), np.asarray(trajectory["actions"]).reshape(-1, 1)
    rng = np.random.default_rng(seed+32452843)
    targets = monte_carlo_returns(trajectory["rewards"], trajectory["dones"],
                                  log_probabilities=trajectory.get("log_probabilities"), alpha=sac.alpha)
    replay = ReplayBuffer(max(1, len(obs)), obs.shape[1])
    for values in zip(obs, trajectory["actions"], trajectory["rewards"], trajectory["next_observations"], trajectory["dones"]):
        replay.add(*values)
    history = []
    for _ in range(iterations):
        indices = rng.integers(0, len(obs), size=batch_size)
        if monte_carlo:
            target = targets[indices, None]
            inputs = np.column_stack((sac.policy.normalise(obs[indices]), actions[indices]))
        else:
            batch = replay.sample(batch_size, rng)
            next_action, logp, _ = sac.policy.sample(batch["next_obs"], sac.rng)
            next_input = np.column_stack((sac.policy.normalise(batch["next_obs"]), next_action))
            target = batch["rewards"] + (1-batch["dones"])*(
                np.minimum(sac.target_q1.forward(next_input), sac.target_q2.forward(next_input))-sac.alpha*logp)
            inputs = np.column_stack((sac.policy.normalise(batch["obs"]), batch["actions"]))
        loss = []
        for network, optimiser in ((sac.q1, sac.q1_optimiser), (sac.q2, sac.q2_optimiser)):
            prediction, cache = network.forward(inputs, cache=True)
            error = prediction-target
            _, gradients = network.backward(cache, 2*error/len(inputs))
            optimiser.step(gradients)
            loss.append(float(np.mean(error**2)))
        if monte_carlo:
            # MC initialisation estimates the same value function in both live
            # and target critics; do not leave a random target after fitting.
            for target_network, current in ((sac.target_q1, sac.q1), (sac.target_q2, sac.q2)):
                for target_parameter, parameter in zip(target_network.parameters(), current.parameters()):
                    target_parameter[...] = parameter
        else:
            sac.target_q1.polyak_update(sac.q1, sac.tau)
            sac.target_q2.polyak_update(sac.q2, sac.tau)
        history.append({"q1_loss": loss[0], "q2_loss": loss[1]})
    return {"method": "realised Monte Carlo critic initialisation" if monte_carlo else "Bellman bootstrap critic warmup",
            "iterations": iterations, "future_returns_only_training_labels": True, "history": history}


def critic_diagnostics(sac, trajectory):
    obs, actions = trajectory["observations"], np.asarray(trajectory["actions"]).reshape(-1, 1)
    inputs = np.column_stack((sac.policy.normalise(obs), actions))
    q1, q2 = sac.q1.forward(inputs)[:, 0], sac.q2.forward(inputs)[:, 0]
    plain = monte_carlo_returns(trajectory["rewards"], trajectory["dones"])
    soft = monte_carlo_returns(trajectory["rewards"], trajectory["dones"],
                               log_probabilities=trajectory["log_probabilities"], alpha=sac.alpha)
    q = np.minimum(q1, q2)
    return {"q_mean": float(q.mean()), "q_min": float(q.min()), "q_max": float(q.max()),
            "plain_mc_mean_s": float(plain.mean()), "soft_mc_mean_s": float(soft.mean()),
            "q_minus_mc_mean_s": float((q-soft).mean()), "q_vs_mc_rmse_s": float(np.sqrt(np.mean((q-soft)**2))),
            "q1_q2_disagreement_s": float(np.mean(np.abs(q1-q2))),
            "interpretation": "Realised stochastic trajectory and current alpha; changing actor makes this diagnostic, not a convergence proof"}


def train_sac(simulator, bc_policy, output_dir, *, episodes=20, seed=0, temperature_mode="fixed",
              action_space=None, reward_config=None, mc_warmstart=False, initial_alpha=.001,
              initial_speed_mps=None, carried_rule_state=None, critic_warmup_updates=120,
              batch_size=64, update_every=2, evaluate_every=5, progress=None, retain_rollouts=False):
    if not 0 <= episodes <= 20 or batch_size < 1 or update_every < 1 or evaluate_every < 1:
        raise ValueError("A bounded SAC research round uses at most 20 episodes")
    space = action_space or getattr(bc_policy, "v6_action_space", None)
    features = _features(simulator)
    if tuple(getattr(bc_policy, "v6_features", ())) != features or getattr(bc_policy, "v6_action_space", None) != space:
        raise ValueError("BC warm-start schema differs from SAC experiment")
    config = RewardConfig(**reward_config) if isinstance(reward_config, dict) else reward_config or RewardConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = _copy(bc_policy)
    initial = _rollout(simulator, policy, seed=seed, action_space=space, reward_config=config,
                       initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
    initial_formal = evaluate_policy(simulator, policy, action_space=space, reward_config=config,
                                     initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
    save_policy(policy, output_dir/"initial_bc_policy.npz")
    best_score = _score(initial_formal)
    best_policy = _copy(policy) if np.isfinite(best_score) else None
    best_result, best_episode = initial_formal, 0 if best_policy is not None else None
    sac = DiagnosticSAC(policy, seed=seed, temperature_mode=temperature_mode, initial_alpha=initial_alpha)
    warmup = warmstart_critics(sac, initial, iterations=critic_warmup_updates, batch_size=batch_size,
                               seed=seed, monte_carlo=mc_warmstart)
    replay = ReplayBuffer(max(20000, len(initial["actions"])*(episodes+1)), len(features))
    for values in zip(initial["observations"], initial["actions"], initial["rewards"], initial["next_observations"], initial["dones"]):
        replay.add(*values)
    rng = np.random.default_rng(seed+104729)
    history = [{"episode": 0, "stage": "BC incumbent", "evaluation": initial_formal["summary"],
                "selected_as_best": best_policy is not None}]
    failures, rollouts = [], []
    completed_episodes = 0
    for episode in range(1, episodes+1):
        try:
            trajectory = _rollout(simulator, policy, seed=seed+episode, action_space=space, reward_config=config,
                                   initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state,
                                   rng=rng, replay=replay, sac=sac, batch_size=batch_size, update_every=update_every)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            failure = {"episode": episode, "failure": f"{type(error).__name__}: {error}",
                       "selected_as_best": False, "accepted_trajectory_invented": False}
            failures.append(failure)
            history.append(failure)
            _write(output_dir/f"episode_{episode:03d}_exception.json", failure)
            if isinstance(error, RolloutFailure):
                _write(output_dir/f"episode_{episode:03d}_exception_trace.json.gz", error.trace)
            break
        if retain_rollouts:
            rollouts.append(trajectory)
        completed_episodes = episode
        record = {"episode": episode, "stage": "SAC", "alpha": sac.alpha, "updates": sac.updates,
                  "rollout": trajectory_diagnostics(trajectory), "critic": critic_diagnostics(sac, trajectory),
                  "selected_as_best": False}
        if trajectory["losses"]:
            record["losses"] = {key: float(np.mean([v[key] for v in trajectory["losses"]])) for key in trajectory["losses"][0]}
        if not trajectory["result"]["summary"].get("completed_lap", False):
            failure = {"episode": episode, "failure": "Executor terminated before the lap boundary",
                       "terminal_info": trajectory["infos"][-1]}
            failures.append(failure)
            record["failure"] = failure["failure"]
            _write(output_dir/f"episode_{episode:03d}_failed_trajectory.json.gz", trajectory)
        if episode % evaluate_every == 0 or episode == episodes:
            evaluated = evaluate_policy(simulator, policy, action_space=space, reward_config=config,
                                         initial_speed_mps=initial_speed_mps, carried_rule_state=carried_rule_state)
            score = _score(evaluated)
            improved = score < best_score-1e-9
            record["evaluation"], record["selected_as_best"] = evaluated["summary"], improved
            _write(output_dir/f"evaluation_{episode:03d}_trajectory.json.gz", evaluated)
            if improved:
                best_score, best_result, best_policy, best_episode = score, evaluated, _copy(policy), episode
            if progress:
                progress(f"SAC {temperature_mode} seed {seed}, episode {episode}: {'legal' if is_legal(evaluated) else 'rejected'}")
        history.append(record)
        _write(output_dir/"sac_history.json", history)
    last_checkpoint = None
    last_result = None
    if all(np.isfinite(p).all() for p in policy.network.parameters()):
        last_checkpoint = "last_sac_candidate.npz"
        save_policy(policy, output_dir/last_checkpoint)
        try:
            last_result = evaluate_policy(simulator, policy, reward_config=config, initial_speed_mps=initial_speed_mps,
                                          carried_rule_state=carried_rule_state, seed=seed)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            failures.append({"stage": "last deterministic evaluation", "failure": f"{type(error).__name__}: {error}"})
    selected_checkpoint = None
    if best_policy is not None:
        selected_checkpoint = "best_feasible_policy.npz"
        save_policy(best_policy, output_dir/selected_checkpoint)
        loaded = load_policy(output_dir/selected_checkpoint, expected_features=features, expected_action_space=space)
        result = evaluate_policy(simulator, loaded, reward_config=config,
                                  initial_speed_mps=float(best_result["summary"]["speed_start_mps"]),
                                  carried_rule_state=carried_rule_state, seed=seed, periodic=False)
        _require_reproducible(best_result, result)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA, "algorithm": "NumPy twin-Q continuous SAC",
        "seed": seed, "features": features, "action_space": space, "temperature_mode": temperature_mode,
        "reward_config": config.to_dict(), "reward_units": "seconds", "gamma": 1.0,
        "episodes_requested": episodes, "episodes_completed": completed_episodes,
        "all_training_episodes_completed": completed_episodes == episodes and not failures,
        "critic_initialisation": warmup, "batch_size": batch_size, "update_every": update_every,
        "learning_rate": 3e-4, "initial_alpha": initial_alpha, "final_alpha": sac.alpha,
        "sac_updates": sac.updates, "replay_size": replay.size,
        "best_episode": best_episode, "selected_checkpoint": selected_checkpoint,
        "last_checkpoint": last_checkpoint, "best_summary": best_result["summary"] if best_policy is not None else None,
        "last_summary": last_result["summary"] if last_result is not None else None,
        "bc_summary": initial_formal["summary"],
        "bc_training_rollout_summary": initial["result"]["summary"],
        "gain_over_bc_s": float(initial_formal["summary"]["lap_time_s"])-best_score
                          if best_policy is not None and is_legal(initial_formal) else None,
        "training_failures": failures, "sac_improvement_found": best_episode not in (None, 0),
        "checkpoint_roundtrip_passed": best_policy is not None,
        "formal_evaluation": "Policy-specific periodic speed shooting with the same physical executor",
        "training_boundary": "Fixed training initial speed; any nonclosed training lap is diagnostic and cannot itself be selected",
        "checkpoint_verification": "Fixed initial speed of selected final shooting trajectory; no shooting residual is labelled environment parity",
        "selection_rule": "Fastest deterministic publicly certified cyclic sustainable checkpoint, with BC episode zero separately identified",
        "global_optimality_claim": False, "real_vehicle_benefit_claim": False,
    }
    _write(output_dir/"sac_history.json", history)
    _write(output_dir/"sac_metadata.json", metadata)
    _write(output_dir/"initial_bc_trajectory.json.gz", initial)
    return {"policy": best_policy, "result": best_result if best_policy is not None else None,
            "last_policy": policy, "last_result": last_result, "metadata": metadata,
            "history": history, "rollouts": rollouts}
