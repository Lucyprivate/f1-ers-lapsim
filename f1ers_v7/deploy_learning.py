"""Independent D learning, with an explicitly analytical constant-action clone.

The zero mean head is preregistered from the constant raw-zero teacher. BC
still runs the bounded supervised and aggregation budgets; its deterministic
actor cannot improve on this analytical clone. SAC is allowed to move the
mean head later. Frozen v6/v7 routines and checkpoints are not rewritten.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from f1ers.learning import GaussianPolicy, clone_behaviour
from f1ers_v6.learning import _write, _teacher_arrays, _require_reproducible
from f1ers_v7.learning import (_bind, _copy, _features, _rollout, _score,
    evaluate_policy, is_legal, save_policy, load_policy)
from f1ers_v7.deploy_only import FEATURES, PROTOCOL, require_policy_contract


def analytical_zero_policy(observations, *, seed=0):
    observations = np.asarray(observations, float)
    if (observations.ndim != 2 or observations.shape[1] != len(FEATURES)
            or not len(observations) or not np.isfinite(observations).all()):
        raise ValueError('An analytical D clone requires finite 34-feature states')
    policy = GaussianPolicy(len(FEATURES), observations.mean(axis=0),
        np.maximum(observations.std(axis=0), .05), seed=seed)
    policy.network.weights[-1][:, 0] = 0.
    policy.network.biases[-1][0] = 0.
    return _bind(policy, FEATURES, 'mode_aware', PROTOCOL)


def require_exact_zero_mean(policy, observations=None):
    require_policy_contract(policy)
    if (np.any(policy.network.weights[-1][:, 0] != 0.)
            or policy.network.biases[-1][0] != 0.):
        raise RuntimeError('Analytical mean head changed during zero-label BC')
    if observations is not None:
        values = policy.network.forward(policy.normalise(observations))[:, 0]
        if np.any(values != 0.):
            raise RuntimeError('Zero mean head did not execute exact raw zero')
    return {'mean_head_weights_exact_zero': True, 'mean_head_bias_exact_zero': True,
        'deterministic_raw_action_exact_zero_for_every_finite_observation': True,
        'constant_clone_is_analytical_not_a_learned_performance_gain': True}


def exact_execution(reference, result):
    keys = ('requested_actions', 'mapped_actions', 'speed_mps', 'time_s', 'energy_mj')
    if any(not np.array_equal(np.asarray(reference[k]), np.asarray(result[k]), equal_nan=True)
            for k in keys):
        return False
    first, second = reference['internal_trace'], result['internal_trace']
    return set(first) == set(second) and all(np.array_equal(np.asarray(first[k]),
        np.asarray(second[k]), equal_nan=True) for k in first)


def train_bc(teacher_rollouts, simulator, output_dir, *, seed=0, epochs=600,
        dagger_rounds=3, dagger_epochs=200, action_space='mode_aware',
        reward_config=None, initial_speed_mps=None, progress=None):
    """Run BC with exact-zero analytical initialization, retaining every round.

    The teacher must be six newly integrated D laps. Old raw-one braking labels
    or 33-feature arrays are rejected rather than silently translated.
    """
    if not 1 <= epochs <= 600 or not 0 <= dagger_rounds <= 3 or not 1 <= dagger_epochs <= 200:
        raise ValueError('BC budget exceeds the frozen 600 + 3 x 200 limit')
    if tuple(_features(simulator)) != FEATURES or simulator.protocol != PROTOCOL or action_space != 'mode_aware':
        raise ValueError('The analytical clone requires the D executor and schema')
    teachers = list(teacher_rollouts)
    if len(teachers) != 6 or not all(is_legal(t.get('result', t)) for t in teachers):
        raise ValueError('Six independently qualified physical D teacher laps are required')
    for teacher in teachers:
        result = teacher.get('result', teacher)
        if (tuple(result['summary'].get('feature_schema', ())) != FEATURES
                or result['summary'].get('protocol') != PROTOCOL):
            raise ValueError('Teacher result must carry the exact new D schema/protocol')
    teachers, datasets, labels = _teacher_arrays(teachers, FEATURES)
    if any(np.any(values != 0.) for values in labels):
        raise ValueError('D teacher raw labels must be exact zero, not old mapped full-regeneration actions')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy = analytical_zero_policy(np.vstack(datasets), seed=seed)
    zero_initial = require_exact_zero_mean(policy, np.vstack(datasets))
    save_policy(policy, output/'analytical_zero_policy.npz')
    analytic = evaluate_policy(simulator, policy, periodic=False, initial_speed_mps=initial_speed_mps,
        reward_config=reward_config)
    _write(output/'analytical_zero_formal_result.json.gz', analytic)
    policy, curve = clone_behaviour(np.vstack(datasets), np.concatenate(labels), seed=seed,
        epochs=epochs, initial_policy=policy)
    _bind(policy, FEATURES, action_space, PROTOCOL)
    certificates = [require_exact_zero_mean(policy, np.vstack(datasets))]
    save_policy(policy, output/'raw_bc_policy.npz')
    raw = _rollout(simulator, policy, seed=seed, action_space=action_space,
        reward_config=reward_config, initial_speed_mps=initial_speed_mps)
    _write(output/'raw_bc_trajectory.json.gz', raw)
    raw_result = evaluate_policy(simulator, policy, periodic=False, initial_speed_mps=initial_speed_mps,
        reward_config=reward_config)
    _write(output/'raw_bc_formal_result.json.gz', raw_result)
    if not exact_execution(analytic, raw_result):
        raise RuntimeError('Analytical initialization and zero-label BC changed deterministic physical execution')
    best_score = _score(raw_result)
    best_policy = _copy(policy) if np.isfinite(best_score) else None
    best_result, selected_round = raw_result, 0 if best_policy is not None else None
    trajectories, aggregation = [raw], []
    formal_result = raw_result
    for round_index in range(1, dagger_rounds+1):
        learner = _rollout(simulator, policy, seed=seed, action_space=action_space,
            reward_config=reward_config, initial_speed_mps=initial_speed_mps)
        _write(output/f'dagger_round_{round_index}_learner_trajectory.json.gz', learner)
        learner_states = learner['observations']
        if len(learner_states) > len(labels[0]):
            raise ValueError('The fixed-label expert cannot label beyond its certified route')
        datasets.append(learner_states)
        labels.append(np.zeros(len(learner_states)))
        policy, round_curve = clone_behaviour(np.vstack(datasets), np.concatenate(labels),
            seed=seed+1009*round_index, epochs=dagger_epochs, initial_policy=policy)
        _bind(policy, FEATURES, action_space, PROTOCOL)
        certificates.append(require_exact_zero_mean(policy, np.vstack(datasets)))
        save_policy(policy, output/f'dagger_round_{round_index}_candidate.npz')
        _write(output/f'dagger_round_{round_index}_curve.json', round_curve)
        candidate = _rollout(simulator, policy, seed=seed, action_space=action_space,
            reward_config=reward_config, initial_speed_mps=initial_speed_mps)
        _write(output/f'dagger_round_{round_index}_trajectory.json.gz', candidate)
        formal_result = evaluate_policy(simulator, policy, periodic=False, initial_speed_mps=initial_speed_mps,
            reward_config=reward_config)
        _write(output/f'dagger_round_{round_index}_formal_result.json.gz', formal_result)
        if not exact_execution(analytic, formal_result):
            raise RuntimeError('DAgger changed the analytical clone deterministic trace')
        value = _score(formal_result)
        improved = value < best_score-1e-9
        if improved:
            best_score, best_result, best_policy, selected_round = value, formal_result, _copy(policy), round_index
        aggregation.append({'round': round_index, 'samples': sum(len(v) for v in datasets),
            'action_rmse': round_curve[-1]['action_rmse'], 'training_rollout': candidate['result']['summary'],
            'evaluation': formal_result['summary'], 'selected_as_best': improved,
            'analytical_trace_exact': True})
        _write(output/'aggregation.json', aggregation)
        trajectories.append(candidate)
        if progress:
            progress(f'D analytical clone seed {seed}, DAgger {round_index}: exact zero mean preserved')
    save_policy(policy, output/'last_bc_candidate.npz')
    selected_checkpoint = None
    if best_policy is not None:
        selected_checkpoint = 'best_bc_policy.npz'
        save_policy(best_policy, output/selected_checkpoint)
        loaded = load_policy(output/selected_checkpoint, expected_features=FEATURES,
            expected_action_space=action_space, expected_execution_protocol=PROTOCOL)
        require_exact_zero_mean(loaded)
        reproduced = evaluate_policy(simulator, loaded, periodic=False,
            initial_speed_mps=best_result['summary']['speed_start_mps'], reward_config=reward_config)
        _require_reproducible(best_result, reproduced)
        if not exact_execution(best_result, reproduced):
            raise RuntimeError('D BC schema7 save/load altered accepted physical trace')
    metadata = {'schema_version': 7, 'algorithm': 'Analytical constant-mean clone with executed BC / DAgger budgets',
        'seed': seed, 'features': FEATURES, 'action_space': action_space, 'execution_protocol': PROTOCOL,
        'epochs': epochs, 'dagger_rounds': dagger_rounds, 'dagger_epochs': dagger_epochs,
        'initialization': 'Mean output column and bias exactly zero; log standard deviation normally initialized and supervised',
        'initialization_source': 'Preregistered constant raw-zero numerical D teacher, never development residuals',
        'analytical_clone_certificate': zero_initial, 'post_training_certificates': certificates,
        'raw_action_rmse': curve[-1]['action_rmse'], 'aggregation': aggregation,
        'raw_summary': raw_result['summary'], 'analytical_summary': analytic['summary'],
        'best_summary': best_result['summary'] if best_policy is not None else None,
        'selected_round': selected_round, 'selected_checkpoint': selected_checkpoint,
        'last_checkpoint': 'last_bc_candidate.npz', 'last_summary': formal_result['summary'],
        'deterministic_analytic_clone_trace_exact': True, 'checkpoint_roundtrip_passed': best_policy is not None,
        'learning_performance_gain_claimed': False, 'BC_lap_time_gain_over_analytical_clone_s': 0.,
        'log_std_regularization': 'Supervised shrinkage toward log_std -2.5; no empirical variance identification',
        'policy_specific_periodic_shooting': False,
        'expert': 'Exact raw-zero teacher under new deployment-only execution, fixed labels on learner states',
        'all_BC_checkpoints_retained': True, 'physical_multilap_release_verified_here': False,
        'global_optimality_claim': False, 'real_vehicle_benefit_claim': False}
    _write(output/'bc_curve.json', curve)
    _write(output/'bc_metadata.json', metadata)
    return {'policy': best_policy, 'result': best_result if best_policy is not None else None,
        'last_policy': policy, 'last_result': formal_result, 'metadata': metadata, 'rollouts': trajectories}


# This independently versioned D SAC body preserves frozen v7 objective, units
# and optimizer, while flushing accepted trajectories/checkpoints immediately.
# It deliberately does not promise recovery of unsaved intermediate eval weights.
from copy import deepcopy
from f1ers.learning import ReplayBuffer
from f1ers_v6.reward import RewardConfig
from f1ers_v6.learning import RolloutFailure
from f1ers_v6 import learning as previous
from f1ers_v7.learning import (_boundary, ReturnScaledSAC, warmstart_critics,
    critic_diagnostics, trajectory_diagnostics)


def train_sac(simulator, bc_policy, output_dir, *, episodes=20, seed=0, temperature_mode="fixed",
              action_space=None, reward_config=None, mc_warmstart=True, initial_alpha=.001,
              initial_speed_mps=None, initial_energy_mj=2., carried_rule_state=None,
              initial_state_pool=None, evaluation_initial_state=None, periodic_evaluation=False,
              return_scale_s=73.084831, critic_warmup_updates=120, batch_size=64,
              update_every=2, evaluate_every=5, progress=None, retain_rollouts=False):
    if not 0<=episodes<=20 or batch_size<1 or update_every<1 or evaluate_every<1:
        raise ValueError("Each preregistered v7 SAC case permits at most 20 episodes")
    if periodic_evaluation is not False:
        raise ValueError("D fixes every formal boundary; periodic speed shooting is prohibited")
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
    _write(output_dir/"initial_bc_trajectory.json.gz", initial)
    initial_formal = evaluate(policy)
    save_policy(policy, output_dir/"initial_bc_policy.npz")
    best_score = _score(initial_formal)
    best_policy = _copy(policy) if np.isfinite(best_score) else None
    best_result, best_episode = initial_formal, 0 if best_policy is not None else None
    if best_policy is not None:
        save_policy(best_policy, output_dir/"best_feasible_policy.npz")
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
        _write(output_dir/f"episode_{episode:03d}_started.json", {"episode": episode, "initial_boundary_index": boundary_index, "initial_boundary": boundary})
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
            _write(output_dir/"sac_history.json", history)
            break
        completed = episode
        # Save the accepted complete/prefix trajectory before any diagnostic or evaluation can fail.
        _write(output_dir/f"training_episode_{episode:03d}.json.gz", trajectory)
        save_policy(policy, output_dir/"last_sac_candidate.npz")
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
                save_policy(best_policy, output_dir/"best_feasible_policy.npz")
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
        "formal_evaluation": "Fixed full cold or physics-carried boundary; speed shooting prohibited",
        "policy_specific_periodic_shooting": False,
        "training_boundary": "Frozen cold or physics-carried initial boundaries; no within-lap speed/energy/FSM reset",
        "replay_action": "raw requested scalar; execution/projection are outcomes, not substituted actor inputs",
        "selection_rule": "Fastest deterministic certified cyclic sustainable checkpoint; BC episode zero explicitly retained",
        "physical_multilap_release_verified_here": False,
        "complete_or_physical_failure_episode_trajectories_saved_immediately": True,
        "periodic_evaluation_policy_weights_retained": False,
        "retained_SAC_checkpoints": ["initial_bc_policy.npz", "best_feasible_policy.npz", "last_sac_candidate.npz"],
        "all_intermediate_checkpoint_qualification_claimed": False,
        "global_optimality_claim": False, "real_vehicle_benefit_claim": False}
    _write(output_dir/"sac_history.json", history)
    _write(output_dir/"sac_metadata.json", metadata)
    _write(output_dir/"initial_bc_trajectory.json.gz", initial)
    return {"policy": best_policy, "result": best_result if best_policy is not None else None,
            "last_policy": policy, "last_result": last_result, "metadata": metadata,
            "history": history, "rollouts": rollouts}
