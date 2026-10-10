"""v2 single-lap causal execution with a certified conservative energy floor.

Training, optimisation and scoring use the same implicit segment transition.
Only the known route and the fixed no-K boundary are looked ahead. Future actor
actions and predicted harvest are never used to release energy credit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import sqrt
from typing import Callable
import numpy as np

from .model import Simulator, VehicleParams, _segment_velocity, _velocity_kernel

OBS_DIM = 13
PROTOCOL = "causal_floor_v2"


@dataclass(frozen=True)
class CausalExecutionConfig:
    energy_start_mj: float = 2.0
    energy_floor_mj: float = 2.0
    energy_capacity_mj: float = 4.0
    ramp_rate_kw_per_s: float = 100.0
    boundary_max_cycles: int = 20
    boundary_tolerance_mps: float = 1e-4
    finish_tolerance_mps: float = 0.02
    guard_bisection_iterations: int = 30
    energy_tolerance_mj: float = 1e-8
    power_tolerance_kw: float = 1e-4
    force_tolerance_n: float = 8.0
    terminal_energy_penalty_s_per_mj: float = 5.0
    terminal_speed_penalty_s_per_mps: float = 0.5

    def to_dict(self):
        return asdict(self)


ExecutionConfig = CausalExecutionConfig


@dataclass(frozen=True)
class CausalState:
    node_index: int
    speed_mps: float
    energy_mj: float
    recharge_mj: float = 0.0
    previous_dc_kw: float = 0.0
    previous_ax_mps2: float = 0.0
    elapsed_time_s: float = 0.0


class CausalSimulator:
    observation_dim = OBS_DIM
    protocol = PROTOCOL

    def __init__(self, track: dict, rules: dict, vehicle: VehicleParams | dict,
                 config: CausalExecutionConfig | dict | None = None):
        self.config = (CausalExecutionConfig(**config) if isinstance(config, dict)
                       else config or CausalExecutionConfig())
        c = self.config
        if not 0 <= c.energy_floor_mj <= c.energy_start_mj <= c.energy_capacity_mj:
            raise ValueError("Require 0 <= energy floor <= initial energy <= capacity")
        if c.energy_floor_mj != c.energy_start_mj:
            raise ValueError("v2 sustainable single-lap protocol locks floor to initial energy")
        if c.ramp_rate_kw_per_s <= 0 or c.boundary_max_cycles < 1 or c.guard_bisection_iterations < 10:
            raise ValueError("Invalid slew rate or numerical iteration limits")
        self.vehicle = VehicleParams.from_dict(vehicle) if isinstance(vehicle, dict) else vehicle
        self.params = self.vehicle
        self.track, self.rules = track, rules
        self.geometry = Simulator(track, rules, replace(self.vehicle, energy_start_mj=c.energy_start_mj))
        g = self.geometry
        self.n_nodes, self.ds_m, self.length_m = g.n_nodes, g.ds_m, g.length_m
        self.s_m, self.curvature = g.s_m, g.curvature
        self.aero_mode, self.speed_cap_mps = g.aero_mode, g.speed_cap_mps
        self.max_dc_kw, self.harvest_limit_mj = g.max_dc_kw, g.harvest_limit_mj
        self.energy_capacity_mj = c.energy_capacity_mj
        if abs(c.energy_capacity_mj - g.energy_capacity_mj) > 1e-9:
            raise ValueError("Execution capacity must match the public usable-energy window")
        if c.ramp_rate_kw_per_s > g.ramp_rate + 1e-9:
            raise ValueError("Execution slew cannot be less conservative than the public bound")
        self.safe_speed_mps = self._safe_envelope(None)
        guess = float(g._speeds(np.zeros(self.n_nodes), np.zeros(self.n_nodes, bool))[0])
        cycle_errors = []
        boundary_converged = False
        for cycle in range(c.boundary_max_cycles):
            state = CausalState(0, guess, c.energy_start_mj)
            for i in range(self.n_nodes):
                state, info = self.advance(state, 0.0)
                if not info["step_certified"]:
                    raise RuntimeError("No-K boundary reference is physically infeasible: " + str(info["guard_reasons"]))
            difference = abs(state.speed_mps - guess)
            cycle_errors.append(difference)
            if difference <= c.boundary_tolerance_mps:
                boundary_converged = True
                break
            guess = state.speed_mps
        self.boundary_speed_mps = guess
        self.boundary_report = dict(method="causal_noK_periodic_fixed_point", converged=boundary_converged,
            cycles=len(cycle_errors), residual_mps=cycle_errors[-1], cycle_errors_mps=cycle_errors,
            maximum_cycles=c.boundary_max_cycles, tolerance_mps=c.boundary_tolerance_mps)
        self.safe_speed_mps = self._safe_envelope(guess)
        if self.safe_speed_mps[0] < guess - c.finish_tolerance_mps:
            raise RuntimeError("Frozen boundary is outside the conservative braking envelope")
        # Before zero_result exists, reward reference is unused by advance.
        self.zero_result = self.simulate(np.zeros(self.n_nodes))
        self.reference_speed_mps = np.r_[self.zero_result["speed_mps"], self.zero_result["speed_next_mps"][-1]]
        self._base_speed = self.reference_speed_mps[:-1]

    def _safe_envelope(self, terminal_speed):
        p, g = self.vehicle, self.geometry
        caps = self.speed_cap_mps.copy()
        if terminal_speed is not None:
            caps[0] = min(caps[0], terminal_speed)
        safe = _velocity_kernel(caps, self.curvature, g.drag_area, g.downforce_area,
            np.zeros(self.n_nodes), np.zeros(self.n_nodes, bool), self.ds_m, p.mass_kg,
            p.mu, p.air_density_kgpm3, p.rolling_coefficient, 1e7, 1e9,
            p.deploy_efficiency, p.harvest_efficiency)
        return np.r_[safe, safe[0]]

    def reserve_mj(self, dc_kw):
        return max(0.0, dc_kw)**2 / (2000.0 * self.config.ramp_rate_kw_per_s)

    def deployment_limits_kw(self, speed_mps):
        return self.geometry.deployment_limits_kw(speed_mps)

    def _cap(self, i, speed):
        vkph = 3.6 * speed
        cap = max(0.0, min(self.max_dc_kw, 1800.0 - 5.0 * vkph)) if vkph < 340 else (
            max(0.0, 6900.0 - 20.0 * vkph) if vkph < 345 else 0.0)
        for zone in self.rules.get("alternative_power_zones_m", []):
            if isinstance(zone, dict):
                start, end = zone["start_m"], zone["end_m"]
                threshold, alternative = zone.get("threshold_kph", 310.0), zone.get("power_kw", 250.0)
            else:
                start, end = zone[:2]
                threshold, alternative = 310.0, 250.0
            if start <= self.geometry.rules_distance_m[i] <= end and vkph < threshold:
                cap = min(cap, alternative)
        return float(cap)

    def observation(self, state):
        i = min(state.node_index, self.n_nodes) % self.n_nodes
        old = self.geometry.observation(i, state.speed_mps, state.energy_mj, state.recharge_mj,
                                         state.previous_dc_kw, state.previous_ax_mps2)
        credit = max(0.0, state.energy_mj - self.config.energy_floor_mj - self.reserve_mj(state.previous_dc_kw))
        return np.r_[old, credit / self.energy_capacity_mj]

    def _physical(self, state, dc_kw):
        i, p, g = state.node_index, self.vehicle, self.geometry
        next_speed = float(_segment_velocity(state.speed_mps, self.safe_speed_mps[i + 1],
            self.curvature[i], g.drag_area[i], g.downforce_area[i], dc_kw, self.ds_m,
            p.mass_kg, p.mu, p.air_density_kgpm3, p.rolling_coefficient,
            p.ice_power_kw, p.ice_max_tractive_force_n, p.deploy_efficiency, p.harvest_efficiency))
        mean_speed = 0.5 * (state.speed_mps + next_speed)
        dt = self.ds_m / mean_speed
        ax = (next_speed**2 - state.speed_mps**2) / (2.0 * self.ds_m)
        drag = g._drag(i, mean_speed)
        wheel_kw = (p.mass_kg * ax + drag) * mean_speed / 1000.0
        kmec = dc_kw * (p.deploy_efficiency if dc_kw >= 0 else 1.0 / p.harvest_efficiency)
        ice_kw, friction_kw = max(0.0, wheel_kw - kmec), max(0.0, kmec - wheel_kw)
        cap = min(self._cap(i, state.speed_mps), self._cap(i, next_speed))
        kinetic = 0.5 * p.mass_kg * (next_speed**2 - state.speed_mps**2)
        work_residual = (wheel_kw * 1000.0 * dt - kinetic - drag * self.ds_m) / 1e6
        force_excess = max(0.0, abs(wheel_kw * 1000.0 / mean_speed) -
                           p.mass_kg * g._tire_accel(i, state.speed_mps))
        ice_cap = g._ice_available_kw(mean_speed)
        return dict(speed_next_mps=next_speed, dt_s=dt, ax_mps2=ax, wheel_power_kw=wheel_kw,
                    mguk_mechanical_kw=kmec, ice_mechanical_kw=ice_kw, friction_brake_kw=friction_kw,
                    deployment_cap_kw=cap, mechanical_work_residual_mj=work_residual,
                    tire_force_excess_n=force_excess, ice_power_excess_kw=max(0.0, ice_kw - ice_cap),
                    ice_available_kw=ice_cap)

    def _slew_exempt(self, state, physical):
        if physical["wheel_power_kw"] < -1e-6:
            return "actual_braking_negative_wheel_demand"
        if min(state.speed_mps, physical["speed_next_mps"]) * 3.6 < float(self.rules.get("power_reduction_low_speed_exception_kph", 210.0)):
            return "low_speed"
        if physical["deployment_cap_kw"] < state.previous_dc_kw - self.config.power_tolerance_kw:
            return "public_speed_power_curve"
        return "none"

    def _certificate(self, state, dc_kw, physical):
        c, p = self.config, self.vehicle
        energy_next = state.energy_mj - dc_kw * physical["dt_s"] / 1000.0
        recharge_next = state.recharge_mj + max(0.0, -dc_kw) * physical["dt_s"] / 1000.0
        reserve = self.reserve_mj(dc_kw)
        exemption = self._slew_exempt(state, physical)
        lower = max(0.0, state.previous_dc_kw - c.ramp_rate_kw_per_s * physical["dt_s"]) if state.previous_dc_kw > 0 and exemption == "none" else -self.max_dc_kw
        if dc_kw >= 0:
            demand_excess = max(0.0, dc_kw * p.deploy_efficiency - max(0.0, physical["wheel_power_kw"]))
            harvest_excess = 0.0
        elif physical["wheel_power_kw"] < -1e-6:
            demand_excess = 0.0
            harvest_excess = max(0.0, -dc_kw + physical["wheel_power_kw"] * p.harvest_efficiency)
        else:
            bound = min(self.max_dc_kw, p.superclip_max_dc_kw,
                        max(0.0, physical["ice_available_kw"] - p.minimum_ice_wheel_power_kw) * p.harvest_efficiency)
            demand_excess, harvest_excess = 0.0, max(0.0, -dc_kw - bound)
        residuals = dict(energy_floor_mj=max(0.0, c.energy_floor_mj - energy_next),
            energy_capacity_mj=max(0.0, energy_next - self.energy_capacity_mj),
            reserve_certificate_mj=max(0.0, reserve - (energy_next - c.energy_floor_mj)),
            recharge_limit_mj=max(0.0, recharge_next - self.harvest_limit_mj),
            absolute_power_kw=max(0.0, abs(dc_kw) - self.max_dc_kw),
            power_curve_kw=max(0.0, dc_kw - physical["deployment_cap_kw"]),
            slew_kw=max(0.0, lower - dc_kw),
            propulsion_demand_kw=demand_excess, harvest_source_kw=harvest_excess,
            ice_power_kw=physical["ice_power_excess_kw"], tire_force_n=physical["tire_force_excess_n"],
            next_speed_cap_mps=max(0.0, physical["speed_next_mps"] - self.safe_speed_mps[state.node_index + 1]),
            mechanical_work_mj=abs(physical["mechanical_work_residual_mj"]))
        failures = []
        for key, value in residuals.items():
            tolerance = c.energy_tolerance_mj if key.endswith("mj") else (
                c.force_tolerance_n if key == "tire_force_n" else (
                0.001 if key == "ice_power_kw" else c.power_tolerance_kw))
            if value > tolerance or not np.isfinite(value):
                failures.append(key)
        return dict(certified=not failures, failures=failures, residuals=residuals,
                    energy_next_mj=energy_next, recharge_next_mj=recharge_next,
                    reserve_mj=reserve, slew_lower_kw=lower, slew_exemption=exemption)

    def _guard_candidate_ok(self, certificate, ignore_slew=False):
        """Use strict energy/source bounds when searching, tolerance only for
        floating-point physical certification of a finally selected candidate."""
        strict=("energy_floor_mj","energy_capacity_mj","reserve_certificate_mj","recharge_limit_mj",
                "absolute_power_kw","power_curve_kw","propulsion_demand_kw","harvest_source_kw")
        if any(certificate["residuals"][key]>0 for key in strict):
            return False
        return not [key for key in certificate["failures"] if not (ignore_slew and key=="slew_kw")]

    def advance(self, state: CausalState, action: float):
        """Pure transition: choose and verify power together with its actual dt."""
        if state.node_index >= self.n_nodes:
            raise RuntimeError("No transition exists beyond the lap boundary")
        action = float(np.asarray(action).reshape(-1)[0])
        if not np.isfinite(action) or abs(action) > 1.0 + 1e-8:
            raise ValueError("Normalized action must be finite and lie in [-1,1]")
        i, c = state.node_index, self.config
        requested_kw = action * (self._cap(i, state.speed_mps) if action >= 0 else self.max_dc_kw)
        zero = self._physical(state, 0.0)
        lower = 0.0
        if state.previous_dc_kw > 0 and self._slew_exempt(state, zero) == "none":
            lo, hi = 0.0, max(state.previous_dc_kw, self.max_dc_kw)
            for _ in range(c.guard_bisection_iterations):
                candidate = 0.5 * (lo + hi)
                physical = self._physical(state, candidate)
                if candidate + c.ramp_rate_kw_per_s * physical["dt_s"] >= state.previous_dc_kw:
                    hi = candidate
                else:
                    lo = candidate
            lower = hi
        reasons = []
        if requested_kw >= 0 or lower > 1e-8:
            desired=max(requested_kw,lower,0.0)
            physical=self._physical(state,desired)
            certificate=self._certificate(state,desired,physical)
            if lower == 0.0 and (state.energy_mj <= c.energy_floor_mj or zero["wheel_power_kw"] < 0.0):
                selected=upper=0.0
            elif self._guard_candidate_ok(certificate,ignore_slew=True):
                selected=desired
                upper=max(desired,physical["deployment_cap_kw"])
            else:
                # Find the maximal feasible power no larger than the request;
                # its implicit dt is recomputed for every candidate.
                lo,hi=0.0,min(desired,self.max_dc_kw)
                for _ in range(c.guard_bisection_iterations):
                    candidate=0.5*(lo+hi)
                    physical=self._physical(state,candidate)
                    certificate=self._certificate(state,candidate,physical)
                    if self._guard_candidate_ok(certificate,ignore_slew=True):
                        lo=candidate
                    else:
                        hi=candidate
                selected=upper=lo
            if requested_kw > upper + c.power_tolerance_kw:
                reasons.append("deployment_credit_reserve_or_public_cap")
            if requested_kw < lower - c.power_tolerance_kw:
                reasons.append("conservative_slew_hold")
        else:
            # Find maximum harvesting with actual ICE/brake source and energy
            # window. No prospective harvest enters the positive credit bound.
            physical=self._physical(state,requested_kw)
            certificate=self._certificate(state,requested_kw,physical)
            if self._guard_candidate_ok(certificate):
                selected=requested_kw
            else:
                lo, hi = requested_kw, 0.0
                for _ in range(c.guard_bisection_iterations):
                    candidate = 0.5 * (lo + hi)
                    physical = self._physical(state, candidate)
                    certificate = self._certificate(state, candidate, physical)
                    if self._guard_candidate_ok(certificate):
                        hi = candidate
                    else:
                        lo = candidate
                selected=hi
            upper=self._cap(i,state.speed_mps)
            if selected > requested_kw + c.power_tolerance_kw:
                reasons.append("harvest_source_window_or_lap_budget")
        physical = self._physical(state, selected)
        certificate = self._certificate(state, selected, physical)
        if lower > upper + c.power_tolerance_kw:
            certificate["certified"] = False
            certificate["failures"].append("empty_power_intersection")
        info = {**physical, "mguk_dc_kw":selected, "requested_dc_kw":requested_kw,
            "requested_action":action, "shielded_action":selected / max(self._cap(i, state.speed_mps) if selected >= 0 else self.max_dc_kw, 1e-12),
            "step_certified":certificate["certified"], "guard_reasons":reasons + certificate["failures"],
            "constraint_residuals":certificate["residuals"], "reserve_mj":certificate["reserve_mj"],
            "deployable_credit_mj":max(0.0,state.energy_mj-c.energy_floor_mj-self.reserve_mj(state.previous_dc_kw)),
            "reserve_reduction_kw":max(0.0,requested_kw-selected) if requested_kw > 0 else 0.0,
            "power_lower_kw":lower, "power_upper_kw":upper,
            "slew_lower_kw":certificate["slew_lower_kw"], "slew_exemption":certificate["slew_exemption"],
            "guard_iterations":c.guard_bisection_iterations, "inner_converged":True}
        if not certificate["certified"]:
            # Preserve the rejected proposal; do not turn it into a fake accepted
            # energy update or silently violate slew with a later SOC clipping.
            info.update(executed_action=0.0, energy_mj=state.energy_mj, recharge_mj=state.recharge_mj,
                        energy_balance_residual_mj=0.0, execution_stopped=True)
            return state, info
        next_state = CausalState(i+1, physical["speed_next_mps"],certificate["energy_next_mj"],
            certificate["recharge_next_mj"],selected,physical["ax_mps2"],state.elapsed_time_s+physical["dt_s"])
        mode = "deploy" if selected > 1e-8 else ("brake_regen" if selected < -1e-8 and physical["wheel_power_kw"] < -1e-6 else (
            "superclip" if selected < -1e-8 else ("friction_brake" if physical["wheel_power_kw"] < -1e-6 else "powered")))
        info.update(executed_action=info["shielded_action"],energy_mj=next_state.energy_mj,
            recharge_mj=next_state.recharge_mj,operating_mode=mode,execution_stopped=False,
            energy_balance_residual_mj=next_state.energy_mj-state.energy_mj+selected*physical["dt_s"]/1000.0)
        return next_state, info

    def create_env(self):
        return CausalEnv(self)

    def simulate(self, actions=None, policy: Callable | None = None, sustainable=True):
        if not sustainable:
            raise ValueError("v2 always uses its conservative sustainable execution layer")
        if actions is not None and policy is not None:
            raise ValueError("Supply actions or policy, not both")
        if actions is None and policy is None:
            actions = np.zeros(self.n_nodes)
        if actions is not None:
            actions = np.asarray(actions, float).reshape(-1)
            if actions.shape != (self.n_nodes,) or not np.isfinite(actions).all() or np.max(abs(actions)) > 1+1e-8:
                raise ValueError("One bounded finite action is required per track segment")
        env = self.create_env()
        obs, done = env.reset(), False
        while not done:
            action = policy(obs) if policy is not None else actions[env.i]
            obs, reward, done, info = env.step(action)
        return env.result()


class CausalEnv:
    observation_dim = OBS_DIM
    action_dim = 1

    def __init__(self, sim):
        self.sim = sim
        self.reset()

    def reset(self, seed=None, start_speed_mps=None):
        sim, c = self.sim, self.sim.config
        if start_speed_mps is not None and abs(float(start_speed_mps)-sim.boundary_speed_mps) > 1e-7:
            raise ValueError("All v2 policies use the same frozen no-K boundary speed")
        self.state = CausalState(0, sim.boundary_speed_mps, c.energy_start_mj)
        self.history, self.states, self.observations = [], [self.state], []
        self.failed, self.done = False, False
        return sim.observation(self.state)

    @property
    def i(self): return self.state.node_index
    @property
    def energy(self): return self.state.energy_mj
    @property
    def recharge(self): return self.state.recharge_mj
    @property
    def speed(self): return self.state.speed_mps
    @property
    def start_speed(self): return self.sim.boundary_speed_mps
    @property
    def elapsed(self): return self.state.elapsed_time_s

    def step(self, action):
        if self.done:
            raise RuntimeError("reset required after the episode ends")
        sim, c, state = self.sim, self.sim.config, self.state
        observation = sim.observation(state)
        next_state, info = sim.advance(state, action)
        self.failed |= not info["step_certified"]
        self.done = self.failed or next_state.node_index == sim.n_nodes
        if hasattr(sim, "zero_result"):
            reference_dt = sim.zero_result["dt_s"][state.node_index]
        else:
            reference_dt = info["dt_s"]
        reward_time = reference_dt - info["dt_s"] if info["step_certified"] else 0.0
        deficit = max(0.0,c.energy_start_mj-next_state.energy_mj)
        mismatch = abs(next_state.speed_mps-sim.boundary_speed_mps)
        reward_energy = -c.terminal_energy_penalty_s_per_mj*deficit if self.done else 0.0
        reward_speed = -c.terminal_speed_penalty_s_per_mps*mismatch if self.done else 0.0
        failure_penalty = -100.0 if self.failed else 0.0
        reward = reward_time + reward_energy + reward_speed + failure_penalty
        info.update(reward_time_s=reward_time,reward_terminal_energy_s=reward_energy,
            reward_terminal_speed_s=reward_speed,reward_failure_s=failure_penalty,reward_unit="seconds_saved_relative_to_common_noK",
            terminal_penalty_applied=self.done,sustainability_deficit_mj=deficit,
            periodic_speed_error_mps=mismatch if self.done else 0.0,elapsed_time_s=next_state.elapsed_time_s)
        self.observations.append(observation)
        self.history.append(info)
        self.state = next_state
        self.states.append(next_state)
        return sim.observation(next_state),float(reward),self.done,info

    def result(self):
        if not self.done:
            raise RuntimeError("A result is available only after an episode terminates")
        sim,c=self.sim,self.sim.config
        # Failed proposal remains in detailed guard diagnostics, not the accepted
        # state/energy trajectory. Completed valid laps always have N segments.
        accepted=[(state,obs,info) for state,obs,info in zip(self.states[:-1],self.observations,self.history) if info["step_certified"]]
        rows=[row[2] for row in accepted]
        states=[row[0] for row in accepted]
        n=len(rows)
        arrays={key:np.array([row.get(key,0.0) for row in rows]) for key in (
            "dt_s","speed_next_mps","ax_mps2","wheel_power_kw","mguk_dc_kw","mguk_mechanical_kw",
            "ice_mechanical_kw","friction_brake_kw","deployment_cap_kw","reserve_mj","deployable_credit_mj",
            "requested_dc_kw","reserve_reduction_kw","power_lower_kw","power_upper_kw","slew_lower_kw",
            "mechanical_work_residual_mj","energy_balance_residual_mj")}
        speed=np.array([state.speed_mps for state in states])
        energy=np.array([state.energy_mj for state in states]+[self.state.energy_mj])
        recharge=np.array([state.recharge_mj for state in states]+[self.state.recharge_mj])
        arrays.update(s_m=sim.s_m[:n].copy(),x_m=np.asarray(sim.track.get("x_m",np.zeros(sim.n_nodes)))[:n],
            y_m=np.asarray(sim.track.get("y_m",np.zeros(sim.n_nodes)))[:n],speed_mps=speed,
            time_s=np.r_[0.0,np.cumsum(arrays["dt_s"])],energy_mj=energy,recharge_mj=recharge,
            soc_fraction=energy/sim.energy_capacity_mj,ay_mps2=speed**2*sim.curvature[:n],
            aero_mode=sim.aero_mode[:n].astype(int),energy_floor_mj=np.full(n+1,c.energy_floor_mj),
            observations=np.stack([row[1] for row in accepted]) if n else np.empty((0,OBS_DIM)),
            requested_actions=np.array([row["requested_action"] for row in rows]),
            shielded_actions=np.array([row["shielded_action"] for row in rows]),
            executed_actions=np.array([row["executed_action"] for row in rows]),
            operating_mode=np.array([row["operating_mode"] for row in rows]),
            guard_reasons=[row["guard_reasons"] for row in rows],
            slew_exemptions=[row["slew_exemption"] for row in rows],
            step_certified=np.ones(n,dtype=bool),guard_diagnostics=self.history)
        violations={}
        for row in self.history:
            for key,value in row["constraint_residuals"].items():
                violations[key]=max(violations.get(key,0.0),float(value))
        speed_error=abs(self.state.speed_mps-sim.boundary_speed_mps)
        completed=self.state.node_index==sim.n_nodes
        execution_feasible=not self.failed and sim.boundary_report["converged"]
        sustainable=self.state.energy_mj>=c.energy_start_mj-c.energy_tolerance_mj
        periodic=speed_error<=c.finish_tolerance_mps
        dc,dt=arrays["mguk_dc_kw"],arrays["dt_s"]
        arrays["summary"]=dict(protocol=PROTOCOL,lap_time_s=self.state.elapsed_time_s,
            speed_start_mps=sim.boundary_speed_mps,speed_end_mps=self.state.speed_mps,
            energy_start_mj=c.energy_start_mj,energy_end_mj=self.state.energy_mj,
            recharge_mj=self.state.recharge_mj,deploy_mj=float(np.sum(np.maximum(dc,0)*dt)/1000.0),
            energy_balance_residual_mj=float(self.state.energy_mj-c.energy_start_mj+np.sum(dc*dt)/1000.0),
            mechanical_work_residual_mj=float(np.sum(arrays["mechanical_work_residual_mj"])),
            mechanical_work_max_segment_residual_mj=float(np.max(abs(arrays["mechanical_work_residual_mj"]))) if n else 0.0,
            periodic_speed_error_mps=speed_error,periodic_speed_residual_mps=speed_error,
            boundary_speed_mps=sim.boundary_speed_mps,boundary_report=sim.boundary_report,
            sustainable=sustainable,sustainability_deficit_mj=max(0.0,c.energy_start_mj-self.state.energy_mj),
            execution_feasible=execution_feasible,completed_lap=completed,
            feasible=execution_feasible and completed and sustainable and periodic,
            violations=violations,coupling_converged=execution_feasible,
            coupling_speed_residual_mps=sim.boundary_report["residual_mps"],
            projection_alpha=1.0,deployment_projection_scale=1.0,whole_lap_scaling_used=False,
            action_projection_mean_abs=float(np.mean(abs(arrays["requested_actions"]-arrays["executed_actions"]))) if n else 0.0,
            shield_fraction=float(np.mean(abs(arrays["requested_actions"]-arrays["executed_actions"])>1e-5)) if n else 0.0,
            guard_failures=[row["guard_reasons"] for row in self.history if not row["step_certified"]],
            public_rule_subset_only=True,causal_execution_config=c.to_dict(),
            unimplemented_rule_states=["DOC058_standard_ECU_states","PU_reduction_start_and_hold_state_machine",
                                     "gearshift_and_theoretical_power_slew_exemptions"])
        return arrays
