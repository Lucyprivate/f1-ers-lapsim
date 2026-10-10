"""Distance-domain closed-lap vehicle and DC-energy model.

This is a transparent engineering approximation, not a replica of a team's PU
or the FIA Standard ECU.  Positive K power is DC energy drawn from the store;
negative K power is DC energy delivered to it. All energies are in MJ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Callable
from math import sqrt
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function

OBS_DIM = 12


@njit(cache=False)
def _tire_value(speed, curvature, downforce_area, mass, mu, density):
    total = mu * (9.81 + 0.5 * density * downforce_area * speed**2 / mass)
    lateral = speed**2 * abs(curvature)
    return sqrt(max(0.0, total**2 - lateral**2))


@njit(cache=False)
def _velocity_kernel(caps, curvature, drag_area, downforce_area, pdc, braking,
                     ds, mass, mu, density, rolling, ice_kw, ice_force, eta_d, eta_h):
    speed = caps.copy()
    n = len(speed)
    for iteration in range(32):
        old = speed.copy()
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            mean = 0.5 * (speed[i] + speed[j])
            drag = 0.5 * density * drag_area[i] * mean**2 + rolling * mass * 9.81
            tire = _tire_value(speed[i], curvature[i], downforce_area[i], mass, mu, density)
            wheel = mass * (speed[j]**2 - speed[i]**2) / (2.0 * ds) + drag
            if wheel < -mass * tire:
                low, high = 2.0, speed[i]
                for k in range(25):
                    trial = 0.5 * (low + high)
                    mean = 0.5 * (trial + speed[j])
                    drag = 0.5 * density * drag_area[i] * mean**2 + rolling * mass * 9.81
                    tire = _tire_value(trial, curvature[i], downforce_area[i], mass, mu, density)
                    wheel = mass * (speed[j]**2 - trial**2) / (2.0 * ds) + drag
                    if wheel >= -mass * tire:
                        low = trial
                    else:
                        high = trial
                speed[i] = low
        for i in range(n):
            j = (i + 1) % n
            kmec = pdc[i] * (eta_d if pdc[i] >= 0 else 1.0 / eta_h)
            if braking[i] and kmec < 0:
                kmec = 0.0
            mean = 0.5 * (speed[i] + speed[j])
            available_ice = min(ice_kw, ice_force * max(mean, 1.0) / 1000.0)
            power = max(0.0, available_ice + kmec) * 1000.0
            tire = _tire_value(speed[i], curvature[i], downforce_area[i], mass, mu, density)
            wheel_force = min(power / max(mean, 2.0), mass * tire)
            drag = 0.5 * density * drag_area[i] * mean**2 + rolling * mass * 9.81
            required_force = mass * (speed[j]**2 - speed[i]**2) / (2.0 * ds) + drag
            if required_force > wheel_force:
                low, high = 2.0, speed[j]
                for k in range(25):
                    trial = 0.5 * (low + high)
                    mean = 0.5 * (speed[i] + trial)
                    available_ice = min(ice_kw, ice_force * max(mean, 1.0) / 1000.0)
                    power = max(0.0, available_ice + kmec) * 1000.0
                    available_force = min(power / max(mean, 2.0), mass * tire)
                    drag = 0.5 * density * drag_area[i] * mean**2 + rolling * mass * 9.81
                    required_force = mass * (trial**2 - speed[i]**2) / (2.0 * ds) + drag
                    if required_force <= available_force:
                        low = trial
                    else:
                        high = trial
                speed[j] = low
        if np.max(np.abs(speed - old)) < 2e-8:
            break
    return speed


@njit(cache=False)
def _dc_value(request, dt, speed, wheel_kw, energy, recharge, previous, deploy_cap,
              max_dc, max_sc, ice_kw, ice_force, minimum_wheel_kw, eta_d, eta_h,
              capacity, harvest_limit, ramp_rate):
    mode = 0
    if wheel_kw < -1e-6:
        bound = min(max_dc, -wheel_kw * eta_h)
        dc = max(-bound, min(0.0, request))
        mode = 1 if dc < 0 else 2
    elif request < 0:
        available = max(0.0, min(ice_kw, ice_force * max(speed, 1.0) / 1000.0) - minimum_wheel_kw)
        bound = min(max_dc, max_sc, available * eta_h)
        dc = max(request, -bound)
        mode = 3 if dc < 0 else 0
    else:
        dc = min(request, deploy_cap)
        if speed * 3.6 >= 210.0 and dc >= 100.0 and previous > dc and deploy_cap >= previous:
            dc = max(dc, previous - ramp_rate * dt)
        dc = min(dc, max(0.0, wheel_kw) / eta_d)
        mode = 4 if dc > 0 else 0
    if dc >= 0:
        dc = min(dc, energy * 1000.0 / dt)
    else:
        room = min(capacity - energy, harvest_limit - recharge)
        dc = max(dc, -max(0.0, room) * 1000.0 / dt)
    return dc, mode


@njit(cache=False)
def _energy_kernel(actions, caps, dt, speed, wheel, scale, start_energy, max_dc, max_sc,
                   ice_kw, ice_force, minimum_wheel_kw, eta_d, eta_h, capacity, harvest_limit, ramp_rate):
    n = len(actions)
    energy = np.empty(n + 1)
    recharge = np.zeros(n + 1)
    power = np.empty(n)
    executed = np.empty(n)
    modes = np.empty(n, np.int64)
    energy[0] = start_energy
    previous = 0.0
    for i in range(n):
        request = actions[i] * (caps[i] if actions[i] >= 0 else max_dc)
        if request > 0:
            request *= scale
        dc, mode = _dc_value(request, dt[i], speed[i], wheel[i], energy[i], recharge[i], previous,
                             caps[i], max_dc, max_sc, ice_kw, ice_force, minimum_wheel_kw, eta_d,
                             eta_h, capacity, harvest_limit, ramp_rate)
        power[i] = dc
        modes[i] = mode
        energy[i + 1] = energy[i] - dc * dt[i] / 1000.0
        recharge[i + 1] = recharge[i] + max(0.0, -dc) * dt[i] / 1000.0
        denominator = caps[i] if dc >= 0 else max_dc
        executed[i] = dc / max(denominator, 1e-9)
        previous = dc
    return energy, recharge, power, executed, modes


@njit(cache=False)
def _segment_velocity(start_speed, upper_speed, curvature, drag_area, downforce_area,
                      pdc, ds, mass, mu, density, rolling, ice_kw, ice_force, eta_d, eta_h):
    """Implicit power-to-speed step with exact trapezoidal-distance work."""
    tire_force = mass * _tire_value(start_speed, curvature, downforce_area, mass, mu, density)
    kmec = pdc * (eta_d if pdc >= 0 else 1.0 / eta_h)
    low, high = 2.0, max(2.0, upper_speed)
    for k in range(28):
        trial = 0.5 * (low + high)
        mean = 0.5 * (start_speed + trial)
        ice = min(ice_kw, ice_force * max(mean, 1.0) / 1000.0)
        drive_force = min(max(0.0, ice + kmec) * 1000.0 / max(mean, 2.0), tire_force)
        drag = 0.5 * density * drag_area * mean**2 + rolling * mass * 9.81
        needed = mass * (trial**2 - start_speed**2) / (2.0 * ds) + drag
        if needed <= drive_force:
            low = trial
        else:
            high = trial
    next_speed = low
    # Friction + regenerative braking shares one GGV force limit.
    mean = 0.5 * (start_speed + next_speed)
    drag = 0.5 * density * drag_area * mean**2 + rolling * mass * 9.81
    needed = mass * (next_speed**2 - start_speed**2) / (2.0 * ds) + drag
    if needed < -tire_force:
        low, high = next_speed, max(start_speed, next_speed)
        for k in range(28):
            trial = 0.5 * (low + high)
            mean = 0.5 * (start_speed + trial)
            drag = 0.5 * density * drag_area * mean**2 + rolling * mass * 9.81
            needed = mass * (trial**2 - start_speed**2) / (2.0 * ds) + drag
            if needed >= -tire_force:
                high = trial
            else:
                low = trial
        next_speed = high
    return next_speed


@dataclass
class VehicleParams:
    mass_kg: float = 800.0
    ice_power_kw: float = 400.0
    ice_max_tractive_force_n: float = 12000.0
    mu: float = 1.75
    drag_area_m2: float = 1.15
    downforce_area_m2: float = 3.6
    straight_drag_factor: float = 0.72
    straight_downforce_factor: float = 0.65
    air_density_kgpm3: float = 1.18
    rolling_coefficient: float = 0.012
    deploy_efficiency: float = 0.95
    harvest_efficiency: float = 0.90
    energy_start_mj: float = 2.0
    max_speed_mps: float = 103.0
    superclip_max_dc_kw: float = 350.0
    minimum_ice_wheel_power_kw: float = 25.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "VehicleParams":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in allowed})


def _rule(rules: dict, names: tuple[str, ...], default: float) -> float:
    for name in names:
        if name in rules:
            return float(rules[name])
    return float(default)


class Simulator:
    observation_dim = OBS_DIM

    def __init__(self, track: dict, rules: dict, vehicle: VehicleParams | dict):
        self.track = track
        self.rules = rules
        self.vehicle = VehicleParams.from_dict(vehicle) if isinstance(vehicle, dict) else vehicle
        self.params = self.vehicle
        self.s_m = np.asarray(track["s_m"], dtype=float)
        self.curvature = np.asarray(track["curvature_1pm"], dtype=float)
        self.n_nodes = len(self.s_m)
        self.length_m = float(track["length_m"])
        self.ds_m = self.length_m / self.n_nodes
        self.rules_distance_m = np.asarray(track.get("fia_s_m", track.get("s_fia_m", self.s_m *
            float(rules.get("official_track_length_m", rules.get("fia_centreline_length_m", self.length_m))) / self.length_m)))
        if self.n_nodes < 8 or len(self.curvature) != self.n_nodes:
            raise ValueError("Track must have >=8 equally spaced, non-repeated closed-lap nodes")
        if not np.all(np.isfinite(self.curvature)):
            raise ValueError("Track curvature must be finite")
        if not np.allclose(np.diff(self.s_m), self.ds_m, rtol=1e-4, atol=1e-4):
            raise ValueError("Track nodes must be equally spaced")
        p = self.vehicle
        if p.mass_kg <= 0 or p.mu <= 0 or not 0 < p.deploy_efficiency <= 1 or not 0 < p.harvest_efficiency <= 1:
            raise ValueError("Invalid mass, grip or efficiencies")
        self.max_dc_kw = _rule(rules, ("ers_max_power_kw", "mguk_max_dc_kw"), 350.0)
        self.energy_capacity_mj = _rule(rules, ("soc_window_mj", "energy_window_mj"), 4.0)
        self.harvest_limit_mj = _rule(rules, ("harvest_max_mj_per_lap", "recharge_limit_mj"), 8.0)
        if self.max_dc_kw <= 0 or self.energy_capacity_mj <= 0 or self.harvest_limit_mj <= 0:
            raise ValueError("Power, energy window and lap harvest budget must be positive")
        if not 0 <= p.energy_start_mj <= self.energy_capacity_mj:
            raise ValueError("Initial energy lies outside the permitted window")
        self.ramp_rate = _rule(rules, ("power_reduction_rate_kw_per_s",), 100.0)
        self.aero_mode = self._aero_modes()
        self.drag_area = p.drag_area_m2 * np.where(self.aero_mode, p.straight_drag_factor, 1.0)
        self.downforce_area = p.downforce_area_m2 * np.where(self.aero_mode, p.straight_downforce_factor, 1.0)
        denominator = p.mass_kg * np.abs(self.curvature) - p.mu * 0.5 * p.air_density_kgpm3 * self.downforce_area
        caps2 = np.divide(p.mu * p.mass_kg * 9.81, denominator,
                          out=np.full(self.n_nodes, p.max_speed_mps**2), where=denominator > 1e-9)
        self.speed_cap_mps = np.minimum(p.max_speed_mps, np.sqrt(np.maximum(caps2, 4.0)))
        self._base_speed = None

    def _aero_modes(self) -> np.ndarray:
        zones = self.rules.get("aero_straight_zones_m", self.track.get("approximate_aero_zones_m",
                              self.rules.get("aero", {}).get("approximate_zones_m", [])))
        zone_coordinate = self.rules_distance_m if "aero_straight_zones_m" in self.rules else self.s_m
        if zones:
            modes = np.zeros(self.n_nodes, dtype=bool)
            for zone in zones:
                start, end = zone[:2] if isinstance(zone, (list, tuple)) else (zone["start_m"], zone["end_m"])
                if start > end:
                    modes |= (zone_coordinate >= start) | (zone_coordinate <= end)
                else:
                    modes |= (zone_coordinate >= start) & (zone_coordinate <= end)
            return modes
        # Explicit approximation if no official track activation zones are mapped.
        return np.abs(self.curvature) < 0.0012

    def deployment_limits_kw(self, speed_mps: np.ndarray) -> np.ndarray:
        """Race standard DC curve, Overtake off, with Austrian Alt1 zones."""
        vkph = np.asarray(speed_mps) * 3.6
        caps = np.where(vkph < 340.0, np.minimum(self.max_dc_kw, 1800.0 - 5.0 * vkph),
                        np.where(vkph < 345.0, 6900.0 - 20.0 * vkph, 0.0))
        caps = np.clip(caps, 0.0, self.max_dc_kw)
        for zone in self.rules.get("alternative_power_zones_m", []):
            if isinstance(zone, dict):
                start, end = zone["start_m"], zone["end_m"]
                alternative = float(zone.get("power_kw", 250.0))
                threshold = float(zone.get("threshold_kph", 310.0))
            else:
                start, end = zone[:2]
                alternative, threshold = 250.0, 310.0
            mask = (self.rules_distance_m >= start) & (self.rules_distance_m <= end) & (vkph < threshold)
            caps[mask] = np.minimum(caps[mask], alternative)
        return caps

    def _drag(self, i: int, speed: float) -> float:
        p = self.vehicle
        return 0.5 * p.air_density_kgpm3 * self.drag_area[i] * speed**2 + p.rolling_coefficient * p.mass_kg * 9.81

    def _tire_accel(self, i: int, speed: float) -> float:
        p = self.vehicle
        total = p.mu * (9.81 + 0.5 * p.air_density_kgpm3 * self.downforce_area[i] * speed**2 / p.mass_kg)
        lateral = speed**2 * abs(self.curvature[i])
        return sqrt(max(0.0, total**2 - lateral**2))

    def _ice_available_kw(self, speed: float) -> float:
        return min(self.vehicle.ice_power_kw, self.vehicle.ice_max_tractive_force_n * max(speed, 1.0) / 1000.0)

    def _speeds(self, pdc: np.ndarray, braking: np.ndarray) -> np.ndarray:
        """Circular forward/backward acceleration envelope, periodic by construction."""
        p = self.vehicle
        return _velocity_kernel(self.speed_cap_mps, self.curvature, self.drag_area, self.downforce_area,
                                pdc, braking, self.ds_m, p.mass_kg, p.mu, p.air_density_kgpm3,
                                p.rolling_coefficient, p.ice_power_kw, p.ice_max_tractive_force_n,
                                p.deploy_efficiency, p.harvest_efficiency)

    def _kinematics(self, speed: np.ndarray):
        next_speed = np.roll(speed, -1)
        mean_speed = 0.5 * (speed + next_speed)
        dt = self.ds_m / np.maximum(mean_speed, 0.5)
        ax = (next_speed**2 - speed**2) / (2.0 * self.ds_m)
        drag = 0.5 * self.vehicle.air_density_kgpm3 * self.drag_area * mean_speed**2 + self.vehicle.rolling_coefficient * self.vehicle.mass_kg * 9.81
        wheel_power = (self.vehicle.mass_kg * ax + drag) * mean_speed / 1000.0
        return dt, ax, wheel_power, wheel_power < -1e-6

    def observation(self, i: int, speed: float, energy: float, recharge: float, previous_dc: float,
                    previous_ax: float = 0.0) -> np.ndarray:
        i %= self.n_nodes
        phase = 2.0 * np.pi * i / self.n_nodes
        ahead = max(2, int(150.0 / self.ds_m))
        upcoming = self.speed_cap_mps[(i + np.arange(1, ahead + 1)) % self.n_nodes].min()
        return np.array([np.sin(phase), np.cos(phase), abs(self.curvature[i]) / 0.03,
                         speed / 100.0, self.speed_cap_mps[i] / 100.0,
                         energy / self.energy_capacity_mj,
                         max(0.0, 1.0 - recharge / self.harvest_limit_mj), previous_dc / self.max_dc_kw,
                         float(self.aero_mode[i]), 1.0 - i / self.n_nodes,
                         upcoming / 100.0, previous_ax / 20.0], dtype=np.float64)

    def _constrain_dc(self, i: int, request: float, dt: float, speed: float, wheel_kw: float,
                      energy: float, recharge: float, previous: float, deploy_cap: float) -> tuple[float, str]:
        p = self.vehicle
        dc, mode = _dc_value(request, dt, speed, wheel_kw, energy, recharge, previous, deploy_cap,
                              self.max_dc_kw, p.superclip_max_dc_kw, p.ice_power_kw,
                              p.ice_max_tractive_force_n, p.minimum_ice_wheel_power_kw,
                              p.deploy_efficiency, p.harvest_efficiency, self.energy_capacity_mj,
                              self.harvest_limit_mj, self.ramp_rate)
        return float(dc), ("powered", "brake_regen", "friction_brake", "superclip", "deploy")[mode]

    def _energy_pass(self, speed: np.ndarray, actions: np.ndarray | None, policy: Callable | None,
                     deployment_scale: float) -> dict:
        dt, ax, wheel, braking = self._kinematics(speed)
        caps = self.deployment_limits_kw(speed)
        if policy is None:
            p = self.vehicle
            requested = np.clip(actions, -1.0, 1.0)
            energy, recharge, power, executed, mode_indices = _energy_kernel(
                requested, caps, dt, speed, wheel, deployment_scale, p.energy_start_mj,
                self.max_dc_kw, p.superclip_max_dc_kw, p.ice_power_kw, p.ice_max_tractive_force_n,
                p.minimum_ice_wheel_power_kw, p.deploy_efficiency, p.harvest_efficiency,
                self.energy_capacity_mj, self.harvest_limit_mj, self.ramp_rate)
            phases = 2.0 * np.pi * np.arange(self.n_nodes) / self.n_nodes
            ahead = max(2, int(150.0 / self.ds_m))
            upcoming = np.minimum.reduce([np.roll(self.speed_cap_mps, -j) for j in range(1, ahead + 1)])
            previous_dc = np.r_[0.0, power[:-1]]
            observations = np.column_stack((np.sin(phases), np.cos(phases), abs(self.curvature)/0.03,
                speed/100.0, self.speed_cap_mps/100.0, energy[:-1]/self.energy_capacity_mj,
                np.maximum(0.0, 1.0-recharge[:-1]/self.harvest_limit_mj), previous_dc/self.max_dc_kw,
                self.aero_mode.astype(float), 1.0-np.arange(self.n_nodes)/self.n_nodes,
                upcoming/100.0, np.r_[0.0, ax[:-1]]/20.0))
            modes = np.array(["powered", "brake_regen", "friction_brake", "superclip", "deploy"])[mode_indices]
            return dict(speed_mps=speed, dt_s=dt, ax_mps2=ax, wheel_power_kw=wheel, braking=braking,
                        energy_mj=energy, recharge_mj=recharge, requested_actions=requested,
                        executed_actions=executed, mguk_dc_kw=power, observations=observations,
                        operating_mode=modes, deployment_cap_kw=caps)
        energy = np.empty(self.n_nodes + 1)
        recharge = np.zeros(self.n_nodes + 1)
        energy[0] = self.vehicle.energy_start_mj
        requested = np.empty(self.n_nodes)
        executed = np.empty(self.n_nodes)
        power = np.empty(self.n_nodes)
        observations = np.empty((self.n_nodes, OBS_DIM))
        modes = []
        previous = 0.0
        for i in range(self.n_nodes):
            observations[i] = self.observation(i, speed[i], energy[i], recharge[i], previous,
                                               ax[i - 1] if i else 0.0)
            raw = float(policy(observations[i])) if policy is not None else float(actions[i])
            if not np.isfinite(raw):
                raise ValueError("Policy returned non-finite action")
            if abs(raw) > 1.0 + 1e-8:
                raise ValueError("Policy action must lie in [-1,1]")
            action = float(np.clip(raw, -1.0, 1.0))
            requested[i] = action
            request_kw = action * (caps[i] if action >= 0 else self.max_dc_kw)
            if request_kw > 0:
                request_kw *= deployment_scale
            power[i], mode = self._constrain_dc(i, request_kw, dt[i], speed[i], wheel[i],
                                               energy[i], recharge[i], previous, caps[i])
            energy[i + 1] = energy[i] - power[i] * dt[i] / 1000.0
            recharge[i + 1] = recharge[i] + max(0.0, -power[i]) * dt[i] / 1000.0
            denominator = caps[i] if power[i] >= 0 else self.max_dc_kw
            executed[i] = power[i] / max(denominator, 1e-9)
            previous = power[i]
            modes.append(mode)
        return dict(speed_mps=speed, dt_s=dt, ax_mps2=ax, wheel_power_kw=wheel, braking=braking,
                    energy_mj=energy, recharge_mj=recharge, requested_actions=requested,
                    executed_actions=executed, mguk_dc_kw=power, observations=observations,
                    operating_mode=np.array(modes), deployment_cap_kw=caps)

    def _coupled(self, actions: np.ndarray | None, policy: Callable | None, scale: float) -> dict:
        if self._base_speed is None:
            self._base_speed = self._speeds(np.zeros(self.n_nodes), np.zeros(self.n_nodes, bool))
        speed = self._base_speed.copy()
        previous_power = np.zeros(self.n_nodes)
        converged = False
        # Damping resolves switching between braking and full-throttle harvesting.
        for iteration in range(60):
            result = self._energy_pass(speed, actions, policy, scale)
            new_speed = self._speeds(result["mguk_dc_kw"], result["braking"])
            dv = np.max(np.abs(new_speed - speed))
            dp = np.max(np.abs(result["mguk_dc_kw"] - previous_power))
            if dv < 0.012 and dp < 0.4:
                speed = new_speed
                converged = True
                break
            speed = 0.55 * new_speed + 0.45 * speed
            previous_power = result["mguk_dc_kw"].copy()
        result = self._energy_pass(speed, actions, policy, scale)
        final_speed_residual = float(np.max(np.abs(self._speeds(result["mguk_dc_kw"], result["braking"]) - speed)))
        result["coupling_iterations"] = iteration + 1
        result["coupling_converged"] = converged and final_speed_residual < 0.02
        result["coupling_speed_residual_mps"] = final_speed_residual
        return result

    def simulate(self, actions: np.ndarray | None = None, policy: Callable | None = None,
                 sustainable: bool = True) -> dict:
        if actions is not None and policy is not None:
            raise ValueError("Supply actions or policy, not both")
        if actions is None and policy is None:
            actions = np.zeros(self.n_nodes)
        if actions is not None:
            actions = np.asarray(actions, dtype=float).reshape(-1)
            if actions.shape != (self.n_nodes,) or not np.all(np.isfinite(actions)):
                raise ValueError("Actions must be finite and have one value per track node")
            if np.max(np.abs(actions)) > 1.0 + 1e-8:
                raise ValueError("Normalized actions must lie in [-1,1]")
        result = self._coupled(actions, policy, 1.0)
        scale = 1.0
        requested_end = float(result["energy_mj"][-1])
        if sustainable and requested_end < self.vehicle.energy_start_mj - 1e-7:
            # A disclosed actuator budget projection; never adds terminal energy.
            low, high = 0.0, 1.0
            best = self._coupled(actions, policy, 0.0)
            for _ in range(13):
                middle = (low + high) / 2.0
                trial = self._coupled(actions, policy, middle)
                if trial["energy_mj"][-1] >= self.vehicle.energy_start_mj:
                    low, best = middle, trial
                else:
                    high = middle
            scale, result = low, best
        return self._finish(result, scale, requested_end, sustainable)

    def _finish(self, result: dict, scale: float, requested_end: float, sustainable: bool) -> dict:
        p = self.vehicle
        speed, dc, dt = result["speed_mps"], result["mguk_dc_kw"], result["dt_s"]
        energy = result["energy_mj"]
        kmec = np.where(dc >= 0, dc * p.deploy_efficiency, dc / p.harvest_efficiency)
        wheel = result["wheel_power_kw"]
        ice = np.maximum(0.0, wheel - kmec)
        mechanical_brake = np.maximum(0.0, kmec - wheel)
        mean_speed = 0.5 * (speed + np.roll(speed, -1))
        drive_force = wheel * 1000.0 / np.maximum(mean_speed, 1.0)
        tire_limit = np.array([p.mass_kg * self._tire_accel(i, float(v)) for i, v in enumerate(speed)])
        ice_caps = np.array([self._ice_available_kw(float(v)) for v in mean_speed])
        drag = 0.5 * p.air_density_kgpm3 * self.drag_area * mean_speed**2 + p.rolling_coefficient * p.mass_kg * 9.81
        mechanical_expected_j = 0.5 * p.mass_kg * (np.roll(speed, -1)**2 - speed**2) + drag * self.ds_m
        mechanical_residual_mj = (wheel * 1000.0 * dt - mechanical_expected_j) / 1e6
        balance = float(energy[-1] - energy[0] + np.sum(dc * dt) / 1000.0)
        tire_excess = float(np.max(np.maximum(0.0, np.abs(drive_force) - tire_limit)))
        ice_excess = float(np.max(np.maximum(0.0, ice - ice_caps)))
        sustainability_deficit = max(0.0, float(energy[0] - energy[-1]))
        violations = dict(
            soc_below_min_mj=max(0.0, float(-energy.min())),
            soc_above_max_mj=max(0.0, float(energy.max() - self.energy_capacity_mj)),
            recharge_excess_mj=max(0.0, float(result["recharge_mj"][-1] - self.harvest_limit_mj)),
            dc_power_excess_kw=max(0.0, float(np.max(np.abs(dc)) - self.max_dc_kw)),
            deployment_curve_excess_kw=max(0.0, float(np.max(dc - result["deployment_cap_kw"]))),
            tire_force_excess_n=tire_excess,
            ice_power_excess_kw=ice_excess,
            negative_ice_power_kw=max(0.0, float(-ice.min())),
            energy_balance_residual_mj=abs(balance),
            mechanical_work_residual_mj=float(np.max(np.abs(mechanical_residual_mj))),
            sustainability_deficit_mj=sustainability_deficit if sustainable else 0.0,
        )
        tolerances = {k: 1e-6 for k in violations}
        tolerances.update(tire_force_excess_n=8.0, ice_power_excess_kw=0.8, negative_ice_power_kw=0.8)
        feasible = bool(result["coupling_converged"]) and all(value <= tolerances[key] for key, value in violations.items())
        lap_time = float(np.sum(dt))
        result.update(s_m=self.s_m.copy(),
                      x_m=np.asarray(self.track.get("x_m", np.zeros(self.n_nodes))),
                      y_m=np.asarray(self.track.get("y_m", np.zeros(self.n_nodes))),
                      time_s=np.r_[0.0, np.cumsum(dt)], soc_fraction=energy / self.energy_capacity_mj,
                      ay_mps2=speed**2 * self.curvature, aero_mode=self.aero_mode.astype(int),
                      mguk_mechanical_kw=kmec, ice_mechanical_kw=ice,
                      friction_brake_kw=mechanical_brake)
        result["summary"] = dict(lap_time_s=lap_time, energy_start_mj=float(energy[0]),
                                 energy_end_mj=float(energy[-1]), recharge_mj=float(result["recharge_mj"][-1]),
                                 deploy_mj=float(np.sum(np.maximum(dc, 0.0) * dt) / 1000.0),
                                 energy_balance_residual_mj=balance,
                                 mechanical_work_residual_mj=float(np.sum(mechanical_residual_mj)),
                                 mechanical_work_max_segment_residual_mj=float(np.max(np.abs(mechanical_residual_mj))),
                                 periodic_speed_residual_mps=0.0, periodic_speed_error_mps=0.0,
                                 sustainable=sustainability_deficit <= 1e-6,
                                 sustainability_deficit_mj=sustainability_deficit,
                                 feasible=feasible, violations=violations,
                                 coupling_converged=bool(result["coupling_converged"]),
                                 coupling_iterations=int(result["coupling_iterations"]),
                                 coupling_speed_residual_mps=float(result["coupling_speed_residual_mps"]),
                                 deployment_projection_scale=float(scale),
                                 projection_alpha=float(scale),
                                 unprojected_energy_end_mj=requested_end,
                                 action_projection_mean_abs=float(np.mean(np.abs(result["requested_actions"] - result["executed_actions"]))),
                                 public_rule_subset_only=True,
                                 unimplemented_rule_states=["DOC058_standard_ECU_states", "PU_reduction_start_and_hold_state_machine"])
        return result

    def create_env(self):
        return DistanceEnv(self)


def build_observations(sim: Simulator, result: dict) -> np.ndarray:
    if "observations" in result:
        return np.asarray(result["observations"], dtype=float).copy()
    return np.stack([sim.observation(i, result["speed_mps"][i], result["energy_mj"][i],
                                    result["recharge_mj"][i], result["mguk_dc_kw"][i-1] if i else 0.0)
                     for i in range(sim.n_nodes)])


class DistanceEnv:
    """Causal segment dynamics for RL; official scoring uses periodic simulate.

    Reward is seconds saved relative to noK segment time. Terminal penalties are
    5 s per MJ of energy deficit and 0.5 s per m/s of periodic speed mismatch.
    This episode never rewrites SOC or applies a post-hoc budget projection.
    """
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.observation_dim = OBS_DIM
        self.action_dim = 1
        if sim._base_speed is None:
            sim._base_speed = sim._speeds(np.zeros(sim.n_nodes), np.zeros(sim.n_nodes, bool))
        self.base = sim._base_speed.copy()
        p = sim.vehicle
        # An unlimited engine produces only the geometric/GGV safety envelope;
        # actual propulsion power is applied separately on every causal step.
        self.safe = _velocity_kernel(sim.speed_cap_mps, sim.curvature, sim.drag_area,
            sim.downforce_area, np.zeros(sim.n_nodes), np.zeros(sim.n_nodes, bool), sim.ds_m,
            p.mass_kg, p.mu, p.air_density_kgpm3, p.rolling_coefficient,
            1e7, 1e9, p.deploy_efficiency, p.harvest_efficiency)
        self.reset()

    def reset(self, seed=None, start_speed_mps=None):
        self.i = 0
        self.speed = float(self.base[0] if start_speed_mps is None else start_speed_mps)
        if not np.isfinite(self.speed) or self.speed < 2 or self.speed > self.sim.speed_cap_mps[0] + 1e-6:
            raise ValueError("Initial environment speed exceeds its local physical cap")
        self.start_speed = self.speed
        self.energy = self.sim.vehicle.energy_start_mj
        self.recharge = 0.0
        self.previous_dc = 0.0
        self.previous_ax = 0.0
        self.elapsed = 0.0
        return self.sim.observation(0, self.speed, self.energy, self.recharge, 0.0)

    def step(self, action):
        sim, p, i = self.sim, self.sim.vehicle, self.i
        if i >= sim.n_nodes:
            raise RuntimeError("reset is required after an episode ends")
        action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        j = (i + 1) % sim.n_nodes
        cap = float(sim.deployment_limits_kw(np.full(sim.n_nodes, self.speed))[i])
        dc_request = action * (cap if action >= 0 else sim.max_dc_kw)
        # The geometric/backward envelope is an exogenous safety horizon. Its
        # acceleration portions are not used as an ERS-independent speed trace.
        actual_dc = dc_request
        for _ in range(20):
            next_speed = float(_segment_velocity(self.speed, self.safe[j], sim.curvature[i],
                    sim.drag_area[i], sim.downforce_area[i], actual_dc, sim.ds_m, p.mass_kg, p.mu,
                    p.air_density_kgpm3, p.rolling_coefficient, p.ice_power_kw,
                    p.ice_max_tractive_force_n, p.deploy_efficiency, p.harvest_efficiency))
            mean_speed = 0.5 * (self.speed + next_speed)
            dt = sim.ds_m / mean_speed
            ax = (next_speed**2 - self.speed**2) / (2 * sim.ds_m)
            drag = sim._drag(i, mean_speed)
            wheel = (p.mass_kg * ax + drag) * mean_speed / 1000.0
            dc, mode = sim._constrain_dc(i, dc_request, dt, mean_speed, wheel,
                                         self.energy, self.recharge, self.previous_dc, cap)
            if abs(dc - actual_dc) < 1e-5:
                break
            actual_dc = dc
        self.energy -= dc * dt / 1000.0
        self.recharge += max(0.0, -dc) * dt / 1000.0
        self.elapsed += dt
        self.speed, self.previous_dc, self.previous_ax = next_speed, dc, ax
        self.i += 1
        done = self.i == sim.n_nodes
        baseline_dt = 2 * sim.ds_m / (self.base[i] + self.base[j])
        reward = baseline_dt - dt
        deficit = max(0.0, p.energy_start_mj - self.energy)
        mismatch = abs(self.speed - self.start_speed)
        if done:
            reward -= 5.0 * deficit + 0.5 * mismatch
        info = dict(dt_s=dt, elapsed_time_s=self.elapsed, energy_mj=self.energy,
                    recharge_mj=self.recharge, mguk_dc_kw=dc, operating_mode=mode,
                    executed_action=dc / max(cap if dc >= 0 else sim.max_dc_kw, 1e-9),
                    sustainability_deficit_mj=deficit, periodic_speed_error_mps=mismatch if done else 0.0,
                    terminal_penalty_applied=done, terminal_energy_penalty_s=5.0 * deficit if done else 0.0,
                    reward_unit="seconds_saved_relative_to_noK")
        kinetic_change_j = 0.5 * p.mass_kg * (next_speed**2 - (next_speed**2 - 2 * sim.ds_m * ax))
        info["mechanical_work_residual_mj"] = (wheel * 1000.0 * dt - kinetic_change_j - drag * sim.ds_m) / 1e6
        obs = sim.observation(self.i, self.speed, self.energy, self.recharge, dc, ax)
        return obs, float(reward), done, info
