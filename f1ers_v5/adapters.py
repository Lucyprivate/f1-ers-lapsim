"""Limited, causal driving adapters; frozen v4 mechanics remain authoritative.

Independent braking requests a force from total load before the mechanical
kernel allocates the remaining combined tyre budget.  The optional autonomous
traction response acts on signed wheel demand, not on observed replay throttle.
All query methods are pure: adaptive retries cannot advance an adapter clock.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import numpy as np

from f1ers_v4.vehicle import VehicleParamsV4, WheelCommand
from f1ers_v4.simulator import ReplayDriver, GeometryDriver


@dataclass
class VehicleParamsV5(VehicleParamsV4):
    brake_mapping: str = "remaining"
    brake_decay_tau_s: float | None = None
    traction_tau_s: float | None = None

    def __post_init__(self):
        if self.brake_mapping not in ("remaining", "independent"):
            raise ValueError("Brake mapping must be remaining or independent")
        if not np.isfinite(self.replay_brake_gain) or not .1 <= self.replay_brake_gain <= 1.:
            raise ValueError("Replay brake gain must lie in [0.1,1.0]")
        for name, bounds in (("brake_decay_tau_s", (.1, 5.)), ("traction_tau_s", (.05, 1.))):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not bounds[0] <= value <= bounds[1]):
                raise ValueError(f"{name} must lie in {bounds} or be None")
        if self.brake_decay_tau_s is not None and self.brake_mapping != "independent":
            raise ValueError("Brake decay requires independent demand mapping")

    @property
    def frozen_v4_baseline(self):
        return self.brake_mapping == "remaining" and self.brake_decay_tau_s is None and self.traction_tau_s is None


@dataclass(frozen=True)
class WheelCommandV5(WheelCommand):
    """Diagnostic metadata is ignored by the unmodified v4 step kernel."""
    raw_requested_force_n: float | None = None
    requested_force_n: float | None = None
    traction_response_fraction: float | None = None
    brake_elapsed_s: float | None = None

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, WheelCommand):
            value = {f.name: getattr(value, f.name) for f in fields(value)}
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in names})


def _geometry(geometry, s):
    value = geometry.evaluate(s) if hasattr(geometry, "evaluate") else geometry(s)
    return {"curvature_1pm": float(value)} if np.isscalar(value) else value


class ReplayDriverV5(ReplayDriver):
    """Raw-time input with a single global equivalent negative-force request.

    Brake onset is found only among nodes at or before the requested time.
    In particular the next release never determines a decay duration.
    """
    def brake_elapsed(self, time_s):
        times = self.trace["time_s"]
        i = min(len(times)-1, max(0, int(np.searchsorted(times, time_s, side="right")-1)))
        if not self.trace["brake"][i]:
            return None
        onset = i
        while onset > 0 and self.trace["brake"][onset-1] != 0:
            onset -= 1
        return max(0., float(time_s-times[onset]))

    def query_state(self, state, time_s, s_m, dt_s):
        command = super().query(time_s, s_m)
        elapsed = self.brake_elapsed(time_s)
        command["brake_elapsed_s"] = elapsed
        if command["brake_fraction"] and self.params.brake_mapping == "independent":
            # Load is evaluated at the actual current accepted integration
            # state, not at an observed or prospective future speed.
            g = _geometry(self.geometry, state.s_m)
            load = self.params.mass_kg*9.81 + .5*self.params.air_density_kgpm3*float(
                g.get("downforce_area_m2", self.params.downforce_area_m2))*state.speed_mps**2
            raw = -self.params.replay_brake_gain*self.params.mu*self.params.session_grip_scale*load
            decay = 1. if self.params.brake_decay_tau_s is None else math.exp(-elapsed/self.params.brake_decay_tau_s)
            demand = raw*decay
            command.update(wheel_force_n=demand, raw_requested_force_n=raw, requested_force_n=demand)
        return command


class GeometryDriverV5(GeometryDriver):
    """Causal first-order positive-demand establishment at fixed decisions.

    The response fraction follows the held positive-demand indicator.  It is
    initialized to one at the lap boundary so no arbitrary standing-start
    transient is introduced.  Negative requests pass through unchanged.
    """
    def __init__(self, geometry, params, frozen_profile=None):
        super().__init__(geometry, params, frozen_profile)
        self._response_fraction = 1.
        self._response_target = 1.
        self._response_time_s = None

    def _fraction_at(self, time_s):
        if self.params.traction_tau_s is None or self._response_time_s is None:
            return 1.
        elapsed = max(0., time_s-self._response_time_s)
        return float(np.clip(self._response_target + (self._response_fraction-self._response_target)*
                             math.exp(-elapsed/self.params.traction_tau_s), 0., 1.))

    def _raw_force(self, state, s_m, command):
        if command.get("wheel_force_n") is not None:
            return float(command["wheel_force_n"])
        g = _geometry(self.geometry, s_m)
        if command.get("brake_fraction", 0.) > 0:
            total = self.params.mu*self.params.session_grip_scale*(self.params.mass_kg*9.81+
                .5*self.params.air_density_kgpm3*float(g.get("downforce_area_m2", self.params.downforce_area_m2))*state.speed_mps**2)
            lateral = self.params.mass_kg*state.speed_mps**2*abs(float(g.get("curvature_1pm", 0.)))
            return -float(command["brake_fraction"])*math.sqrt(max(0.,total**2-lateral**2))
        drag = .5*self.params.air_density_kgpm3*float(g.get("drag_area_m2", self.params.drag_area_m2))*state.speed_mps**2
        drag += self.params.rolling_coefficient*self.params.mass_kg*9.81
        if command.get("target_acceleration_mps2") is not None:
            return self.params.mass_kg*float(command["target_acceleration_mps2"])+drag
        return min(self.params.ice_max_tractive_force_n, float(command["power_kw"])*1000./max(state.speed_mps, .5))

    def decide(self, state):
        # The decision advances the response once.  Queries, including rejected
        # full/half-step proposals, never modify it.
        fraction = self._fraction_at(state.elapsed_time_s)
        super().decide(state)
        raw = self._raw_force(state, state.s_m, self.command)
        self._response_fraction = fraction
        self._response_target = float(raw > 0)
        self._response_time_s = state.elapsed_time_s

    def query_state(self, state, time_s, s_m, dt_s):
        command = super().query(time_s, s_m)
        raw = self._raw_force(state, state.s_m, command)
        fraction = self._fraction_at(time_s)
        demand = raw*fraction if raw > 0 else raw
        command.update(raw_requested_force_n=raw, requested_force_n=demand,
                       traction_response_fraction=fraction)
        if self.params.traction_tau_s is not None and raw > 0:
            # Explicit signed force has priority in the frozen kernel.  Thus
            # target acceleration cannot bypass the positive demand response.
            command["wheel_force_n"] = demand
        return command
