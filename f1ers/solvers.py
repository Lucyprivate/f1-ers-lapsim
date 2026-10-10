"""Reproducible heuristic and constrained numerical control baselines."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from .model import Simulator


def solve_heuristic(sim: Simulator) -> dict:
    neutral = sim.simulate(np.zeros(sim.n_nodes), sustainable=False)
    braking = neutral["wheel_power_kw"] < 0
    speed = neutral["speed_mps"]
    # Full brake regen; deploy below the high-speed taper, harvest gently while
    # ICE is power-on on the fastest parts. All commands still pass the actuator.
    actions = np.where(braking, -1.0, np.where(speed > 80.0, -0.15, 0.85))
    result = sim.simulate(actions, sustainable=True)
    result["solver_status"] = dict(method="distance_rule_heuristic", success=True,
                                    sustainable_projection_disclosed=True,
                                    global_optimality_claim=False)
    return result


def solve_optimal(sim: Simulator, initial: dict | np.ndarray | None = None,
                  maxiter: int = 16, n_blocks: int = 6) -> dict:
    """Find a feasible numerical comparison, without an oracle/global claim.

    SLSQP chooses deployment and high-speed harvest amplitudes in fixed distance
    blocks. Brake regeneration is fixed at full request. The low-dimensional
    search is deliberate: no unverified direct-collocation oracle is reported.
    """
    if initial is None:
        initial = solve_heuristic(sim)
    initial_actions = np.asarray(initial["requested_actions"] if isinstance(initial, dict) else initial)
    best = sim.simulate(initial_actions, sustainable=True)
    neutral = sim.simulate(np.zeros(sim.n_nodes), sustainable=False)
    brake = neutral["wheel_power_kw"] < 0
    fast = (neutral["speed_mps"] > 77.0) & ~brake
    block = np.minimum(n_blocks - 1, np.arange(sim.n_nodes) * n_blocks // sim.n_nodes)
    x0 = np.r_[np.full(n_blocks, best["summary"]["projection_alpha"] * 0.85), np.full(n_blocks, 0.15)]
    cache = {}
    feasible_candidates = [best] if best["summary"]["feasible"] and best["summary"]["sustainable"] else []
    if not feasible_candidates and neutral["summary"]["feasible"] and neutral["summary"]["sustainable"]:
        feasible_candidates.append(neutral)
    evaluations = 0

    def expand(x):
        actions = x[block].copy()
        actions[fast] = -x[n_blocks + block[fast]]
        actions[brake] = -1.0
        return actions

    def evaluate(x):
        nonlocal evaluations
        key = tuple(np.round(x, 12))
        if key not in cache:
            result = sim.simulate(expand(x), sustainable=False)
            cache[key] = result
            evaluations += 1
            if result["summary"]["feasible"] and result["summary"]["sustainable"]:
                feasible_candidates.append(result)
        return cache[key]

    def objective(x):
        result = evaluate(x)
        # Physical defects are a separate penalty even when numerical termination
        # is nominally successful; they never become accepted reference laps.
        violations = result["summary"]["violations"]
        physics_penalty = 0.03 * violations["ice_power_excess_kw"] + 0.001 * violations["tire_force_excess_n"]
        return result["summary"]["lap_time_s"] + physics_penalty

    def energy_constraint(x):
        result = evaluate(x)
        return result["summary"]["energy_end_mj"] - sim.vehicle.energy_start_mj

    optimization = minimize(objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * (2 * n_blocks),
                            constraints=[{"type": "ineq", "fun": energy_constraint}],
                            options={"maxiter": maxiter, "ftol": 2e-3, "eps": 2e-3, "disp": False})
    candidate = sim.simulate(expand(np.clip(optimization.x, 0.0, 1.0)), sustainable=True)
    if candidate["summary"]["feasible"] and candidate["summary"]["sustainable"]:
        feasible_candidates.append(candidate)
    if not feasible_candidates:
        raise RuntimeError("No physically feasible sustainable numerical baseline was found")
    best = min(feasible_candidates, key=lambda item: item["summary"]["lap_time_s"])
    best["solver_status"] = dict(method="scipy_SLSQP_distance_blocks", success=bool(optimization.success),
                                  message=str(optimization.message), iterations=int(optimization.nit),
                                  function_evaluations=evaluations, n_control_variables=2 * n_blocks,
                                  n_track_nodes=sim.n_nodes, distance_step_m=sim.ds_m,
                                  energy_constraint_mj=float(energy_constraint(optimization.x)),
                                  selected_feasible_candidate=True,
                                  selected_lap_feasible=bool(best["summary"]["feasible"]),
                                  global_optimality_claim=False,
                                  interpretation="best feasible candidate found; limited block control family",
                                  final_deployment_projection_alpha=float(best["summary"]["projection_alpha"]))
    return best
