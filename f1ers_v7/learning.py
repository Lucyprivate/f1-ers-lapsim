"""Independent v7 learning with internally scaled values and carried boundaries.

Rewards and entropy temperature remain in seconds, gamma is one, and the
physical executor remains authoritative.  Critic outputs are divided by a
fixed, preregistered time scale.  Raw actions are retained in replay: averaged
executed DC power is an outcome of a stateful shield, not this actor's action.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np

from f1ers.learning import GaussianPolicy, SAC, ReplayBuffer
from f1ers_v6 import learning as previous
from f1ers_v6.reward import RewardConfig

CHECKPOINT_SCHEMA = 7
POLICY_PROTOCOL = "v7_state_feedback_gaussian_v1"
ACTION_SPACES = previous.ACTION_SPACES
RolloutFailure = previous.RolloutFailure
monte_carlo_returns = previous.monte_carlo_returns
_features, _write = previous._features, previous._write


def _bind(policy, features, action_space, execution_protocol="v7_reachable_control"):
    # The network primitives and historical environment use these attribute
    # names; the v7 checkpoint header and execution binding remain independent.
    return previous._bind(policy, features, action_space, execution_protocol)


def _copy(policy):
    return _bind(policy.copy(), policy.v6_features, policy.v6_action_space,
                 policy.v6_execution_protocol)


def adopt_policy(policy, simulator, *, action_space=None):
    """Explicitly copy a historical network into the declared v7 execution.

    This is an execution re-evaluation, not evidence of training under v7.
    Feature order must already agree; no old checkpoint is silently accepted.
    """
    features = _features(simulator)
    if tuple(getattr(policy, "v6_features", ())) != features:
        raise ValueError("Historical checkpoint feature order differs")
    space = action_space or getattr(policy, "v6_action_space", None)
    return _bind(policy.copy(), features, space, str(simulator.protocol))


def save_policy(policy, path):
    features = tuple(getattr(policy, "v6_features", ()))
    space = getattr(policy, "v6_action_space", None)
    execution = getattr(policy, "v6_execution_protocol", None)
    _bind(policy, features, space, execution)
    if (not features or len(set(features)) != len(features) or not all(features)
            or not execution or not np.isfinite(policy.mean).all()
            or not np.isfinite(policy.scale).all() or np.any(policy.scale <= 0)
            or not all(np.isfinite(p).all() for p in policy.network.parameters())):
        raise ValueError("Invalid finite bound policy")
    arrays = {"schema_version": np.asarray(CHECKPOINT_SCHEMA),
              "protocol": np.asarray(POLICY_PROTOCOL),
              "execution_protocol": np.asarray(execution),
              "action_space": np.asarray(space), "features": np.asarray(features, dtype=str),
              "sizes": np.asarray(policy.network.sizes, dtype=np.int64),
              "observation_mean": policy.mean, "observation_scale": policy.scale}
    for i, (w, b) in enumerate(zip(policy.network.weights, policy.network.biases)):
        arrays[f"w{i}"], arrays[f"b{i}"] = w, b
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_policy(path, *, expected_features=None, expected_action_space=None,
                expected_execution_protocol=None):
    with np.load(path, allow_pickle=False) as arrays:
        required = {"schema_version", "protocol", "execution_protocol", "action_space", "features",
                    "sizes", "observation_mean", "observation_scale"}
        if not required.issubset(arrays.files):
            raise ValueError("Missing v7 checkpoint schema")
        if int(arrays["schema_version"]) != CHECKPOINT_SCHEMA or str(arrays["protocol"]) != POLICY_PROTOCOL:
            raise ValueError("Historical checkpoint requires explicit adopt_policy")
        features = tuple(str(v) for v in arrays["features"])
        space, execution = str(arrays["action_space"]), str(arrays["execution_protocol"])
        if expected_features is not None and features != tuple(expected_features):
            raise ValueError("Checkpoint feature order differs")
        if expected_action_space is not None and space != expected_action_space:
            raise ValueError("Checkpoint action mapping differs")
        if expected_execution_protocol is not None and execution != expected_execution_protocol:
            raise ValueError("Checkpoint execution protocol differs")
        sizes = tuple(int(v) for v in arrays["sizes"])
        if len(sizes) < 2 or sizes[0] != len(features) or sizes[-1] != 2 or any(v < 1 for v in sizes):
            raise ValueError("Invalid scalar Gaussian network")
        policy = GaussianPolicy(sizes[0], arrays["observation_mean"], arrays["observation_scale"], sizes[1:-1])
        for i, (w, b) in enumerate(zip(policy.network.weights, policy.network.biases)):
            if arrays[f"w{i}"].shape != w.shape or arrays[f"b{i}"].shape != b.shape:
                raise ValueError("Checkpoint parameter shape differs")
            w[...], b[...] = arrays[f"w{i}"], arrays[f"b{i}"]
        if not np.isfinite(policy.mean).all() or not all(np.isfinite(p).all() for p in policy.network.parameters()):
            raise ValueError("Checkpoint contains nonfinite parameters")
        return _bind(policy, features, space, execution)


class ReturnScaledSAC(SAC):
    """Twin-Q SAC in fixed normalized value units; alpha stays in seconds.

    The normalized objective is the seconds objective divided by return_scale_s.
    The change also conditions optimization; it does not add any reward term.
    Initial critic values in seconds match the unscaled critic initialization.
    """
    def __init__(self, policy, seed=0, *, return_scale_s=73.084831,
                 temperature_mode="fixed", initial_alpha=.001):
        if not np.isfinite(return_scale_s) or return_scale_s <= 0:
            raise ValueError("Fixed positive return time scale required")
        if temperature_mode not in ("fixed", "auto"):
            raise ValueError("Temperature mode must be fixed or auto")
        if not np.isfinite(initial_alpha) or not 1e-5 <= initial_alpha <= .5:
            raise ValueError("Entropy coefficient is specified in seconds")
        super().__init__(policy, seed=seed, gamma=1., initial_alpha=initial_alpha)
        self.return_scale_s = float(return_scale_s)
        self.temperature_mode = temperature_mode
        # Preserve the original seconds-space initial Q, rather than turning
        # a random order-one normalized output into a random ~73-second value.
        for network in (self.q1, self.q2):
            network.weights[-1] /= self.return_scale_s
            network.biases[-1] /= self.return_scale_s
        self.target_q1, self.target_q2 = self.q1.copy(), self.q2.copy()

    @property
    def normalized_alpha(self):
        return self.alpha / self.return_scale_s

    def actor_loss_and_gradients(self, observations, epsilon=None):
        actions, logp, cache = self.policy.sample(observations, self.rng, epsilon)
        features = self.policy.normalise(observations)
        inputs = np.column_stack((features, actions))
        q1, c1 = self.q1.forward(inputs, cache=True)
        q2, c2 = self.q2.forward(inputs, cache=True)
        choose = q1 <= q2
        g1, _ = self.q1.backward(c1, choose.astype(float))
        g2, _ = self.q2.backward(c2, (~choose).astype(float))
        dq = (g1+g2)[:, -1:]
        da = 1-actions**2
        alpha = self.normalized_alpha
        mu_grad = 2*alpha*actions-dq*da
        std_grad = alpha*(-1+2*actions*cache["std_epsilon"])-dq*da*cache["std_epsilon"]
        std_grad *= cache["log_std_mask"]
        _, gradients = self.policy.network.backward(cache["network"],
            np.column_stack((mu_grad, std_grad))/len(features))
        return float(np.mean(alpha*logp-np.minimum(q1, q2))), gradients, logp

    def bellman_target(self, batch, *, next_actions=None, next_log_probabilities=None):
        if next_actions is None:
            next_actions, next_log_probabilities, _ = self.policy.sample(batch["next_obs"], self.rng)
        if next_log_probabilities is None:
            raise ValueError("Next-action log probabilities required")
        inputs = np.column_stack((self.policy.normalise(batch["next_obs"]), next_actions))
        next_q = np.minimum(self.target_q1.forward(inputs), self.target_q2.forward(inputs))
        return batch["rewards"]/self.return_scale_s + (1-batch["dones"])*(
            next_q-self.normalized_alpha*next_log_probabilities)

    def update(self, batch):
        target = self.bellman_target(batch)
        inputs = np.column_stack((self.policy.normalise(batch["obs"]), batch["actions"]))
        critic_losses, gradient_norms = [], []
        for network, optimiser in ((self.q1, self.q1_optimiser), (self.q2, self.q2_optimiser)):
            predicted, cache = network.forward(inputs, cache=True)
            error = predicted-target
            _, gradients = network.backward(cache, 2*error/len(error))
            gradient_norms.append(float(np.sqrt(sum(np.sum(g*g) for g in gradients))))
            optimiser.step(gradients)
            critic_losses.append(float(np.mean(error**2)))
        loss, gradients, logp = self.actor_loss_and_gradients(batch["obs"])
        self.actor_optimiser.step(gradients)
        if self.temperature_mode == "auto":
            temperature_gradient = -float(np.mean(logp+self.target_entropy))
            self.alpha_optimiser.step([np.asarray([temperature_gradient])])
            self.log_alpha[...] = np.clip(self.log_alpha, np.log(1e-5), np.log(.5))
        self.target_q1.polyak_update(self.q1, self.tau)
        self.target_q2.polyak_update(self.q2, self.tau)
        self.updates += 1
        q1, q2 = self.q1.forward(inputs), self.q2.forward(inputs)
        scale = self.return_scale_s
        return {"q1_loss": critic_losses[0]*scale**2, "q2_loss": critic_losses[1]*scale**2,
                "q1_loss_normalized": critic_losses[0], "q2_loss_normalized": critic_losses[1],
                "actor_loss": loss*scale, "actor_loss_normalized": loss,
                "alpha": self.alpha, "normalized_alpha": self.normalized_alpha,
                "gamma": self.gamma, "return_scale_s": scale,
                "mean_log_probability": float(logp.mean()),
                "critic_gradient_norm": float(np.mean(gradient_norms)),
                "critic_clipped_fraction": float(np.mean(np.asarray(gradient_norms)>10)),
                "q_min": float(min(q1.min(), q2.min())*scale),
                "q_max": float(max(q1.max(), q2.max())*scale),
                "q_disagreement": float(np.mean(np.abs(q1-q2))*scale)}


def warmstart_critics(sac, trajectory, *, iterations=120, batch_size=64, seed=0, monte_carlo=True):
    if iterations < 0 or batch_size < 1:
        raise ValueError("Invalid critic warmstart budget")
    obs, actions = np.asarray(trajectory["observations"]), np.asarray(trajectory["actions"]).reshape(-1, 1)
    if len(obs) != len(actions):
        raise ValueError("Trajectory state/action count differs")
    labels = monte_carlo_returns(trajectory["rewards"], trajectory["dones"],
        log_probabilities=trajectory.get("log_probabilities"), alpha=sac.alpha)/sac.return_scale_s
    rng = np.random.default_rng(seed+32452843)
    replay = ReplayBuffer(max(1, len(obs)), obs.shape[1])
    for values in zip(obs, trajectory["actions"], trajectory["rewards"], trajectory["next_observations"], trajectory["dones"]):
        replay.add(*values)
    history = []
    for _ in range(iterations):
        if monte_carlo:
            indices = rng.integers(0, len(obs), size=batch_size)
            target = labels[indices, None]
            inputs = np.column_stack((sac.policy.normalise(obs[indices]), actions[indices]))
        else:
            batch = replay.sample(batch_size, rng)
            target = sac.bellman_target(batch)
            inputs = np.column_stack((sac.policy.normalise(batch["obs"]), batch["actions"]))
        losses, norms = [], []
        for network, optimiser in ((sac.q1, sac.q1_optimiser), (sac.q2, sac.q2_optimiser)):
            prediction, cache = network.forward(inputs, cache=True)
            error = prediction-target
            _, gradients = network.backward(cache, 2*error/len(inputs))
            norms.append(float(np.sqrt(sum(np.sum(g*g) for g in gradients))))
            optimiser.step(gradients)
            losses.append(float(np.mean(error**2)))
        for target_network, current in ((sac.target_q1, sac.q1), (sac.target_q2, sac.q2)):
            if monte_carlo:
                for target_parameter, parameter in zip(target_network.parameters(), current.parameters()):
                    target_parameter[...] = parameter
            else:
                target_network.polyak_update(current, sac.tau)
        history.append({"q1_loss": losses[0]*sac.return_scale_s**2,
                        "q2_loss": losses[1]*sac.return_scale_s**2,
                        "q1_loss_normalized": losses[0], "q2_loss_normalized": losses[1],
                        "critic_gradient_norm": float(np.mean(norms)),
                        "critic_clipped_fraction": float(np.mean(np.asarray(norms)>10))})
    return {"method": "return-scaled realised Monte Carlo critic initialisation" if monte_carlo else
            "return-scaled Bellman bootstrap critic warmup", "iterations": iterations,
            "return_scale_s": sac.return_scale_s, "future_returns_only_training_labels": True,
            "actor_moved": False, "history": history}


def critic_diagnostics(sac, trajectory):
    inputs = np.column_stack((sac.policy.normalise(trajectory["observations"]), trajectory["actions"]))
    q1, q2 = sac.q1.forward(inputs)[:, 0]*sac.return_scale_s, sac.q2.forward(inputs)[:, 0]*sac.return_scale_s
    plain = monte_carlo_returns(trajectory["rewards"], trajectory["dones"])
    soft = monte_carlo_returns(trajectory["rewards"], trajectory["dones"],
        log_probabilities=trajectory["log_probabilities"], alpha=sac.alpha)
    q = np.minimum(q1, q2)
    infos = trajectory.get("infos", [])
    projection = np.asarray([i.get("projection_fraction", 0.) for i in infos])
    return {"q_mean": float(q.mean()), "q_min": float(q.min()), "q_max": float(q.max()),
            "plain_mc_mean_s": float(plain.mean()), "soft_mc_mean_s": float(soft.mean()),
            "q_minus_mc_mean_s": float((q-soft).mean()),
            "q_vs_mc_rmse_s": float(np.sqrt(np.mean((q-soft)**2))),
            "q_vs_mc_rmse_normalized": float(np.sqrt(np.mean((q-soft)**2))/sac.return_scale_s),
            "q1_q2_disagreement_s": float(np.mean(np.abs(q1-q2))),
            "raw_action_replay": True,
            "projected_step_fraction": float(np.mean(projection>1e-8)) if len(projection) else 0.,
            "return_scale_s": sac.return_scale_s,
            "interpretation": "Realised on-policy return at current alpha; changing actor and replay distribution prevent a convergence claim"}


def _boundary(value=None, *, initial_speed_mps=None, initial_energy_mj=2., carried_rule_state=None):
    value = deepcopy(value or {})
    speed = value.get("speed_mps", value.get("initial_speed_mps", initial_speed_mps))
    energy = float(value.get("energy_mj", value.get("initial_energy_mj", initial_energy_mj)))
    carry = value.get("carried_rule_state", value.get("rule_state", carried_rule_state))
    if speed is not None and (not np.isfinite(speed) or float(speed)<=0):
        raise ValueError("Initial speed must be finite and positive")
    if not np.isfinite(energy) or not 2.<=energy<=4.:
        raise ValueError("Initial energy must stay in declared 2–4 MJ interval")
    if carry is not None and not isinstance(carry, dict):
        raise ValueError("Carried rule state must be an explicit boundary dictionary")
    if carry is not None and speed is None:
        raise ValueError("Carried boundary requires its physical speed")
    return {"speed_mps": None if speed is None else float(speed), "energy_mj": energy,
            "carried_rule_state": deepcopy(carry), "source": value.get("source", "caller_frozen_boundary")}


class _BoundaryExecutor:
    """Inject an immutable whole initial boundary into the existing rollout.

    Every episode starts a new diagnostic simulation. Physical multi-lap
    acceptance is performed separately by simulate_laps, carrying actual states.
    """
    def __init__(self, simulator, boundary):
        self.simulator, self.boundary = simulator, deepcopy(boundary)

    def __getattr__(self, key):
        return getattr(self.simulator, key)

    def create_env(self, **kwargs):
        config = kwargs.get("reward_config")
        config = RewardConfig(**config) if isinstance(config, dict) else config or RewardConfig()
        # A warm episode must start with zero shaping potential too.  Keeping
        # a hard-coded 2 MJ reference at 4 MJ would add a −2*lambda constant
        # and compare episodes with different complete-return objectives.
        kwargs["reward_config"] = replace(config, energy_start_mj=self.boundary["energy_mj"])
        kwargs.update(initial_speed_mps=self.boundary["speed_mps"],
                      initial_energy_mj=self.boundary["energy_mj"],
                      carried_rule_state=deepcopy(self.boundary["carried_rule_state"]))
        return self.simulator.create_env(**kwargs)


def _rollout(simulator, policy, *, initial_speed_mps=None, initial_energy_mj=2.,
             carried_rule_state=None, initial_state=None, **kwargs):
    boundary = _boundary(initial_state, initial_speed_mps=initial_speed_mps,
                         initial_energy_mj=initial_energy_mj, carried_rule_state=carried_rule_state)
    return previous._rollout(_BoundaryExecutor(simulator, boundary), policy, **kwargs)


def is_legal(result):
    if not previous.is_legal(result):
        return False
    # Mandatory extra v7 engineering gate is not represented as full FIA proof.
    gate = result["summary"].get("request_proxy_consistent")
    return isinstance(gate, (bool, np.bool_)) and bool(gate)


def trajectory_diagnostics(trajectory):
    diagnostics = previous.trajectory_diagnostics(trajectory)
    diagnostics["legal"] = is_legal(trajectory["result"])
    return diagnostics


def _score(result):
    return float(result["summary"]["lap_time_s"]) if is_legal(result) else np.inf


def evaluate_policy(simulator, policy, *, action_space=None, reward_config=None,
                    initial_speed_mps=None, initial_energy_mj=2., carried_rule_state=None,
                    initial_state=None, periodic=None, seed=0):
    space = action_space or getattr(policy, "v6_action_space", None)
    if tuple(getattr(policy, "v6_features", ())) != _features(simulator) or policy.v6_action_space != space:
        raise ValueError("Policy and executor feature/action schemas differ")
    if getattr(policy, "v6_execution_protocol", None) != str(simulator.protocol):
        raise ValueError("Policy execution protocol differs")
    boundary = _boundary(initial_state, initial_speed_mps=initial_speed_mps,
                         initial_energy_mj=initial_energy_mj, carried_rule_state=carried_rule_state)
    if periodic is None:
        periodic = boundary["carried_rule_state"] is None
    if periodic and boundary["carried_rule_state"] is not None:
        raise ValueError("A physical carried boundary cannot be replaced by speed shooting")
    config = RewardConfig(**reward_config) if isinstance(reward_config, dict) else reward_config or RewardConfig()
    config = replace(config, energy_start_mj=boundary["energy_mj"])
    executor = simulator.periodic_simulate if periodic else simulator.simulate
    return executor(controller=policy, action_space=space, reward_config=config,
                    initial_speed_mps=boundary["speed_mps"], initial_energy_mj=boundary["energy_mj"],
                    carried_rule_state=deepcopy(boundary["carried_rule_state"]))


def train_sac(simulator, bc_policy, output_dir, *, episodes=20, seed=0, temperature_mode="fixed",
              action_space=None, reward_config=None, mc_warmstart=True, initial_alpha=.001,
              initial_speed_mps=None, initial_energy_mj=2., carried_rule_state=None,
              initial_state_pool=None, evaluation_initial_state=None, periodic_evaluation=None,
              return_scale_s=73.084831, critic_warmup_updates=120, batch_size=64,
              update_every=2, evaluate_every=5, progress=None, retain_rollouts=False):
    if not 0<=episodes<=20 or batch_size<1 or update_every<1 or evaluate_every<1:
        raise ValueError("Each preregistered v7 SAC case permits at most 20 episodes")
    space = action_space or getattr(bc_policy, "v6_action_space", None)
    features = _features(simulator)
    if tuple(getattr(bc_policy, "v6_features", ())) != features or bc_policy.v6_action_space != space:
        raise ValueError("BC warm-start feature/action schema differs")
    if bc_policy.v6_execution_protocol != str(simulator.protocol):
        raise ValueError("Historical policy needs explicit adoption before v7 training")
    default_boundary = _boundary(initial_speed_mps=initial_speed_mps,
        initial_energy_mj=initial_energy_mj, carried_rule_state=carried_rule_state)
    pool = [_boundary(v) for v in initial_state_pool] if initial_state_pool is not None else [default_boundary]
    if not pool:
        raise ValueError("At least one frozen initial boundary required")
    evaluation_boundary = _boundary(evaluation_initial_state) if evaluation_initial_state is not None else default_boundary
    config = RewardConfig(**reward_config) if isinstance(reward_config, dict) else reward_config or RewardConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = _copy(bc_policy)
    initial = _rollout(simulator, policy, seed=seed, action_space=space, reward_config=config, initial_state=pool[0])
    def evaluate(candidate):
        return evaluate_policy(simulator, candidate, action_space=space, reward_config=config,
                               initial_state=evaluation_boundary, periodic=periodic_evaluation)
    initial_formal = evaluate(policy)
    save_policy(policy, output_dir/"initial_bc_policy.npz")
    best_score = _score(initial_formal)
    best_policy = _copy(policy) if np.isfinite(best_score) else None
    best_result, best_episode = initial_formal, 0 if best_policy is not None else None
    sac = ReturnScaledSAC(policy, seed=seed, return_scale_s=return_scale_s,
                         temperature_mode=temperature_mode, initial_alpha=initial_alpha)
    before = critic_diagnostics(sac, initial)
    warmup = warmstart_critics(sac, initial, iterations=critic_warmup_updates, batch_size=batch_size,
                               seed=seed, monte_carlo=mc_warmstart)
    warmup.update(before=before, after=critic_diagnostics(sac, initial))
    replay = ReplayBuffer(max(20000, len(initial["actions"])*(episodes+1)), len(features))
    for values in zip(initial["observations"], initial["actions"], initial["rewards"], initial["next_observations"], initial["dones"]):
        replay.add(*values)
    rng = np.random.default_rng(seed+104729)
    history = [{"episode": 0, "stage": "BC incumbent", "evaluation": initial_formal["summary"],
                "selected_as_best": best_policy is not None}]
    failures, rollouts, completed = [], [], 0
    for episode in range(1, episodes+1):
        boundary_index = (episode-1)%len(pool)
        boundary = pool[boundary_index]
        try:
            trajectory = _rollout(simulator, policy, seed=seed+episode, action_space=space, reward_config=config,
                initial_state=boundary, rng=rng, replay=replay, sac=sac, batch_size=batch_size, update_every=update_every)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            failure = {"episode": episode, "failure": f"{type(error).__name__}: {error}",
                       "selected_as_best": False, "accepted_trajectory_invented": False,
                       "initial_boundary_index": boundary_index, "initial_boundary": boundary}
            failures.append(failure)
            history.append(failure)
            _write(output_dir/f"episode_{episode:03d}_exception.json", failure)
            if isinstance(error, RolloutFailure):
                _write(output_dir/f"episode_{episode:03d}_exception_trace.json.gz", error.trace)
            break
        completed = episode
        if retain_rollouts:
            rollouts.append(trajectory)
        record = {"episode": episode, "stage": "return-scaled SAC", "alpha": sac.alpha,
                  "updates": sac.updates, "initial_boundary_index": boundary_index,
                  "initial_boundary": boundary, "rollout": trajectory_diagnostics(trajectory),
                  "critic": critic_diagnostics(sac, trajectory), "selected_as_best": False}
        if trajectory["losses"]:
            record["losses"] = {k: float(np.mean([v[k] for v in trajectory["losses"]])) for k in trajectory["losses"][0]}
        if not trajectory["result"]["summary"].get("completed_lap", False):
            record["failure"] = "Executor terminated before lap boundary"
            failures.append({"episode": episode, "failure": record["failure"], "terminal_info": trajectory["infos"][-1]})
            _write(output_dir/f"episode_{episode:03d}_failed_trajectory.json.gz", trajectory)
        if episode%evaluate_every==0 or episode==episodes:
            evaluated = evaluate(policy)
            score = _score(evaluated)
            improved = score<best_score-1e-9
            record["evaluation"], record["selected_as_best"] = evaluated["summary"], improved
            _write(output_dir/f"evaluation_{episode:03d}_trajectory.json.gz", evaluated)
            if improved:
                best_score, best_result, best_policy, best_episode = score, evaluated, _copy(policy), episode
            if progress:
                progress(f"v7 SAC {temperature_mode} seed {seed}, episode {episode}: {'legal' if is_legal(evaluated) else 'rejected'}")
        history.append(record)
        _write(output_dir/"sac_history.json", history)
    last_result, last_checkpoint = None, None
    if all(np.isfinite(p).all() for p in policy.network.parameters()):
        last_checkpoint = "last_sac_candidate.npz"
        save_policy(policy, output_dir/last_checkpoint)
        try:
            last_result = evaluate(policy)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            failures.append({"stage": "last deterministic evaluation", "failure": f"{type(error).__name__}: {error}"})
    selected_checkpoint = None
    if best_policy is not None:
        selected_checkpoint = "best_feasible_policy.npz"
        save_policy(best_policy, output_dir/selected_checkpoint)
        loaded = load_policy(output_dir/selected_checkpoint, expected_features=features,
                             expected_action_space=space, expected_execution_protocol=str(simulator.protocol))
        fixed_boundary = deepcopy(evaluation_boundary)
        fixed_boundary["speed_mps"] = float(best_result["summary"]["speed_start_mps"])
        reproduced = evaluate_policy(simulator, loaded, action_space=space, reward_config=config,
                                     initial_state=fixed_boundary, periodic=False)
        previous._require_reproducible(best_result, reproduced)
    metadata = {"schema_version": 7, "algorithm": "NumPy return-scaled twin-Q continuous SAC",
        "seed": seed, "features": features, "action_space": space, "execution_protocol": str(simulator.protocol),
        "temperature_mode": temperature_mode, "reward_config": config.to_dict(), "reward_units": "seconds",
        "potential_reference": "Actual frozen initial energy of each episode; initial and terminal potentials zero",
        "gamma": 1., "return_scale_s": sac.return_scale_s,
        "scale_source": "Fixed preregistered frozen neutral model time, independent of observed speed residuals",
        "critic_output_units": "seconds divided by fixed return_scale_s",
        "alpha_units": "seconds; divided exactly once in normalized value objective",
        "episodes_requested": episodes, "episodes_completed": completed,
        "all_training_episodes_completed": completed==episodes and not failures,
        "critic_initialisation": warmup, "batch_size": batch_size, "update_every": update_every,
        "learning_rate": 3e-4, "initial_alpha": initial_alpha, "final_alpha": sac.alpha,
        "sac_updates": sac.updates, "replay_size": replay.size, "best_episode": best_episode,
        "selected_checkpoint": selected_checkpoint, "last_checkpoint": last_checkpoint,
        "best_summary": best_result["summary"] if best_policy is not None else None,
        "last_summary": last_result["summary"] if last_result is not None else None,
        "bc_summary": initial_formal["summary"], "bc_training_rollout_summary": initial["result"]["summary"],
        "gain_over_bc_s": float(initial_formal["summary"]["lap_time_s"])-best_score if best_policy is not None and is_legal(initial_formal) else None,
        "training_failures": failures, "sac_improvement_found": best_episode not in (None, 0),
        "checkpoint_roundtrip_passed": best_policy is not None,
        "initial_state_pool": pool, "initial_pool_schedule": "deterministic round-robin frozen physical boundaries",
        "evaluation_initial_state": evaluation_boundary,
        "formal_evaluation": "Fixed full boundary when carried; otherwise policy-specific periodic speed shooting",
        "training_boundary": "Frozen cold or physics-carried initial boundaries; no within-lap speed/energy/FSM reset",
        "replay_action": "raw requested scalar; execution/projection are outcomes, not substituted actor inputs",
        "selection_rule": "Fastest deterministic certified cyclic sustainable checkpoint; BC episode zero explicitly retained",
        "physical_multilap_release_verified_here": False,
        "global_optimality_claim": False, "real_vehicle_benefit_claim": False}
    _write(output_dir/"sac_history.json", history)
    _write(output_dir/"sac_metadata.json", metadata)
    _write(output_dir/"initial_bc_trajectory.json.gz", initial)
    return {"policy": best_policy, "result": best_result if best_policy is not None else None,
            "last_policy": policy, "last_result": last_result, "metadata": metadata,
            "history": history, "rollouts": rollouts}
