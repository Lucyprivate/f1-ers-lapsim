"""Seconds-based objectives; feasibility is certified by the executor.

Energy potentials telescope over a complete episode.  The terminal potential
is always zero, including failed episodes, so storing surplus charge or
stopping before the finish cannot retain a shaping bonus.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math


@dataclass(frozen=True)
class RewardConfig:
    potential_seconds_per_mj: float = 0.0
    energy_start_mj: float = 2.0
    failure_penalty_s: float = 100.0

    def __post_init__(self):
        values = (self.potential_seconds_per_mj, self.energy_start_mj, self.failure_penalty_s)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Reward configuration must be finite")
        if self.potential_seconds_per_mj < 0 or self.energy_start_mj < 0 or self.failure_penalty_s <= 0:
            raise ValueError("Require nonnegative potential/initial energy and positive failure penalty")

    def to_dict(self):
        return asdict(self)


def _get(state: Any, key: str, default=None):
    return state.get(key, default) if isinstance(state, dict) else getattr(state, key, default)


def _potential(state, config):
    energy = _get(state, "energy_mj")
    if energy is None or not math.isfinite(float(energy)):
        raise ValueError("Electrical reward requires a finite accepted energy state")
    return config.potential_seconds_per_mj * (float(energy) - config.energy_start_mj)


def transition_reward(before, after, ref_dt_s, status, done, completed,
                      config=None, projection_fraction=0.0):
    """Return reward and an exact seconds-based decomposition.

    Future telemetry, action lookup tables and post-hoc energy corrections do
    not enter this calculation.  Projection is diagnostic, not an added
    preference that could change the time objective.  An interrupted interval
    receives its elapsed-time cost but no full-interval reference-time credit.
    """
    config = RewardConfig(**config) if isinstance(config, dict) else config or RewardConfig()
    if not isinstance(config, RewardConfig):
        raise TypeError("config must be RewardConfig or its dictionary")
    before_t = _get(before, "elapsed_time_s", _get(before, "time_s", 0.0))
    after_t = _get(after, "elapsed_time_s", _get(after, "time_s", before_t))
    dt = float(after_t) - float(before_t)
    reference = float(ref_dt_s)
    projection = float(projection_fraction)
    if not all(math.isfinite(v) for v in (dt, reference, projection)) or dt < -1e-10 or reference < 0:
        raise ValueError("Reward time and projection diagnostics must be finite and time nonnegative")
    dt = max(0.0, dt)
    failed = status != "accepted" or (bool(done) and not bool(completed))
    if failed and not done:
        raise ValueError("A failed interval must terminate its episode")
    potential_before = _potential(before, config)
    potential_after = 0.0 if done else _potential(after, config)
    time_reward = (reference if not failed else 0.0) - dt
    shaping_reward = potential_after - potential_before
    failure_reward = -config.failure_penalty_s if failed else 0.0
    reward = time_reward + shaping_reward + failure_reward
    return float(reward), {
        "reward_time_s": float(time_reward),
        "reward_potential_s": float(shaping_reward),
        "reward_failure_s": float(failure_reward),
        "potential_before_s": float(potential_before),
        "potential_after_s": float(potential_after),
        "terminal_potential_zero": bool(done),
        "failed_episode": bool(failed),
        "projection_fraction": projection,
        "reward_unit": "seconds",
        "discount_factor": 1.0,
        "reward_total_s": float(reward),
        "reward_decomposition_residual_s": float(reward-time_reward-shaping_reward-failure_reward),
    }
