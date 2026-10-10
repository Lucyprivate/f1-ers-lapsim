"""Causal public-rule shield for the declared dry Austrian race subset.

The FIA's actual pending/limited-period transitions live in non-public DOC-058.
This module declares its own period boundary (genuine negative wheel demand),
and never claims to implement the Standard ECU. Mechanical driver-demand
reduction and electrical DC reduction are deliberately separate quantities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import inf, pi, sqrt
from typing import Mapping

import numpy as np


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _inside(s, start, end):
    return start <= s < end if start <= end else s >= start or s < end


@dataclass(frozen=True)
class RuleConfig:
    energy_floor_mj: float = 2.0
    energy_capacity_mj: float = 4.0
    initial_hold_s: float = 1.0
    general_initial_reduction_kw: float = 150.0
    mechanical_reduction_rate_kw_per_s: float = 100.0
    electrical_reduction_rate_kw_per_s: float = 100.0
    total_mechanical_reduction_kw: float = 700.0
    crankshaft_lower_bound_rpm: float = 7500.0
    mechanical_torque_limit_nm: float = 500.0
    deploy_efficiency: float = 0.95
    harvest_efficiency: float = 0.90
    ecu_efficiency_correction: float = 0.97
    future_hold_reserve_s: float = 1.0
    auxiliary_sensitivity_w: float = 50.0
    tail_guard_m: float = 350.0
    tolerance: float = 1e-6

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Rule settings must be finite")
        if self.energy_floor_mj != 2.0 or self.energy_capacity_mj != 4.0:
            raise ValueError("The frozen mechanics require the declared 2–4 MJ window")
        if self.initial_hold_s < 1.0 or self.future_hold_reserve_s < self.initial_hold_s:
            raise ValueError("At least one second of hold and its energy reserve are required")
        if not 0 < self.general_initial_reduction_kw <= 150.0:
            raise ValueError("The ordinary initial reduction cannot exceed 150 kW")
        if not 0 < self.mechanical_reduction_rate_kw_per_s <= 100.0 or not 0 < self.electrical_reduction_rate_kw_per_s <= 100.0:
            raise ValueError("The Austrian reduction rates cannot exceed 100 kW/s")
        if not 0 < self.total_mechanical_reduction_kw <= 700.0:
            raise ValueError("Total mechanical reduction cannot exceed 700 kW")
        if self.crankshaft_lower_bound_rpm != 7500.0 or self.mechanical_torque_limit_nm > 500.0 or self.mechanical_torque_limit_nm <= 0:
            raise ValueError("v6 declares a 7500 rpm lower bound and at most 500 Nm")
        if not 0 < self.deploy_efficiency <= 1 or not 0 < self.harvest_efficiency <= 1 or not 0 < self.ecu_efficiency_correction <= 1:
            raise ValueError("Electrical/mechanical efficiencies must lie in (0,1]")
        if self.auxiliary_sensitivity_w < 0 or self.tail_guard_m < 0 or self.tolerance <= 0:
            raise ValueError("Sensitivity load must be nonnegative and tolerance positive")


@dataclass(frozen=True)
class RuleState:
    phase: str = "NORMAL"
    elapsed_time_s: float = 0.0
    hold_until_s: float = 0.0
    reduction_mechanical_kw: float = 0.0
    target_reduction_mechanical_kw: float = 0.0
    reference_mechanical_kw: float = 0.0
    previous_dc_kw: float = 0.0
    requested_dc_kw: float = 0.0
    braking: bool = False
    period_number: int = 0
    lap_number: int = 0
    lap_recharge_origin_mj: float = 0.0

    def __post_init__(self):
        if self.phase not in ("NORMAL", "HOLD", "LIMITED"):
            raise ValueError("Unknown declared reduction phase")
        numeric = [v for v in asdict(self).values() if isinstance(v, (float, int))]
        if (not np.isfinite(numeric).all() or self.reduction_mechanical_kw < 0 or self.reference_mechanical_kw < 0
                or self.target_reduction_mechanical_kw < 0 or self.period_number < 0 or self.lap_number < 0):
            raise ValueError("Invalid rule boundary state")


class RuleShield:
    """Pure proposal queries and explicit commit of accepted integration states.

    The reference demand is the accepted positive K-power peak in the current
    declared positive-wheel-demand period. Before a reduction starts, deployment
    may increase. A reduction freezes that reference until negative wheel demand.
    This is a transparent approximation to a demand map, not observed ECU logic.
    """

    def __init__(self, rules, geometry, config=None):
        self.rules = dict(rules)
        self.geometry = geometry
        self.config = RuleConfig(**config) if isinstance(config, dict) else config or RuleConfig()
        if self.rules.get("overtake_enabled", False) or self.rules.get("boost_enabled", False):
            raise ValueError("The declared research subset disables Overtake and Boost")
        self.max_dc_kw = min(350.0, float(self.rules.get("ers_max_power_kw", 350.0)))
        self.harvest_limit_mj = float(self.rules.get("harvest_max_mj_per_lap", 8.0))
        if not 0 < self.max_dc_kw <= 350 or not 0 < self.harvest_limit_mj <= 8:
            raise ValueError("The dry race subset allows at most 350 kW and 8 MJ/lap")
        self.length_m = float(_get(geometry, "length_m", self.rules.get("fia_centreline_length_m", 4326.0)))
        if self.length_m <= 0:
            raise ValueError("A positive continuous route length is required")
        self.rate_dc = min(self.config.electrical_reduction_rate_kw_per_s, float(self.rules.get("power_reduction_rate_kw_per_s", 100.0)))
        self.rate_mechanical = min(self.config.mechanical_reduction_rate_kw_per_s, float(self.rules.get("power_reduction_rate_kw_per_s", 100.0)))
        if self.rate_dc <= 0 or self.rate_mechanical <= 0:
            raise ValueError("Positive reduction rates are required")
        self.state = RuleState()
        self.raw_action = 0.0
        self.action_space = "signed"
        self.request_kw = 0.0
        self.decisions = []
        self.violations = []
        self._accepted_records = 0
        self._elapsed_accumulated_s = 0.0
        self._initial_boundary = asdict(self.state)

    def _zones(self, route_key, official_key):
        if route_key in self.rules:
            return self.rules[route_key]
        # This fallback is explicitly a uniform official/route coordinate map;
        # processed datasets normally supply the frozen non-affine route map.
        scale = self.length_m / float(self.rules.get("fia_centreline_length_m", self.length_m))
        result = []
        for z in self.rules.get(official_key, []):
            if isinstance(z, dict):
                result.append({**z, "start_m": float(z["start_m"])*scale, "end_m": float(z["end_m"])*scale})
            else:
                result.append([float(z[0])*scale, float(z[1])*scale])
        return result

    def _cap(self, s, speed_mps):
        kph = float(speed_mps)*3.6
        standard = 1800.0-5.0*kph if kph < 340.0 else (6900.0-20.0*kph if kph < 345.0 else 0.0)
        cap = float(np.clip(standard, 0.0, self.max_dc_kw))
        for z in self._zones("alternative_power_zones_route_m", "alternative_power_zones_m"):
            a, b = (z["start_m"], z["end_m"]) if isinstance(z, dict) else z[:2]
            if _inside(float(s)%self.length_m, a, b) and kph < (float(z.get("threshold_kph", 310)) if isinstance(z, dict) else 310.0):
                cap = min(cap, float(z.get("power_kw", 250)) if isinstance(z, dict) else 250.0)
        return min(cap, self._torque_cap(False))

    def _torque_cap(self, braking):
        mechanical = self.config.mechanical_torque_limit_nm*self.config.crankshaft_lower_bound_rpm*2*pi/60.0/1000.0
        if braking:
            return min(self.max_dc_kw, mechanical*self.config.harvest_efficiency, mechanical*self.config.ecu_efficiency_correction)
        return min(self.max_dc_kw, mechanical/self.config.deploy_efficiency, mechanical/self.config.ecu_efficiency_correction)

    @staticmethod
    def _braking(command):
        signed = _get(command, "wheel_force_n")
        return (signed is not None and float(signed) < 0.0) or float(_get(command, "brake_fraction", 0.0)) > 0.0

    def reset(self, state, carried_state=None):
        t = float(_get(state, "elapsed_time_s", 0.0))
        dc = float(_get(state, "previous_dc_kw", 0.0))
        recharge = float(_get(state, "recharge_mj", 0.0) or 0.0)
        energy = _get(state, "energy_mj")
        if not np.isfinite([t, dc, recharge]).all() or abs(dc) > self.max_dc_kw+self.config.tolerance or recharge < 0:
            raise ValueError("Invalid accepted electrical boundary state")
        if energy is not None and not self.config.energy_floor_mj-self.config.tolerance <= float(energy) <= self.config.energy_capacity_mj+self.config.tolerance:
            raise ValueError("Initial energy is outside the declared usable window")
        if carried_state is None:
            self.state = RuleState(elapsed_time_s=t, previous_dc_kw=dc, reference_mechanical_kw=max(0., dc)*self.config.deploy_efficiency,
                                   lap_recharge_origin_mj=recharge)
        else:
            carry = RuleState(**carried_state) if isinstance(carried_state, dict) else carried_state
            if not isinstance(carry, RuleState):
                raise ValueError("Carried rule state must be a RuleState or its serialized fields")
            if carry.reduction_mechanical_kw > self.config.total_mechanical_reduction_kw+self.config.tolerance or abs(carry.requested_dc_kw) > self.max_dc_kw+self.config.tolerance:
                raise ValueError("Carried reduction or request is outside the declared rule subset")
            if abs(dc-carry.previous_dc_kw) > self.config.tolerance:
                raise ValueError("Control-line crossing cannot reset previous DC power")
            remaining = max(0.0, carry.hold_until_s-carry.elapsed_time_s)
            self.state = replace(carry, elapsed_time_s=t, hold_until_s=t+remaining, lap_number=carry.lap_number+1,
                                 lap_recharge_origin_mj=recharge)
        self.request_kw = self.state.requested_dc_kw
        self.raw_action = 0.0
        self._initial_boundary = asdict(self.state)
        return self.state

    def _tail_mask(self, s):
        return self.config.tail_guard_m > 0 and float(s)%self.length_m >= max(0., self.length_m-self.config.tail_guard_m)

    def _initial_limit(self, s):
        limit = self.config.general_initial_reduction_kw
        for z in self._zones("power_reduction_exceptions_route", "power_reduction_exceptions"):
            a, b = (z["start_m"], z["end_m"]) if isinstance(z, dict) else z[:2]
            if _inside(float(s)%self.length_m, a, b):
                limit = max(limit, min(350.0, float(z.get("initial_reduction_max_kw", z.get("max_kw", 350.0))) if isinstance(z, dict) else 350.0))
        return min(limit, self.config.total_mechanical_reduction_kw)

    def _reduction_at(self, t):
        r = self.state.reduction_mechanical_kw
        if self.state.phase == "NORMAL":
            return 0.0
        start = max(self.state.elapsed_time_s, self.state.hold_until_s) if self.state.phase == "HOLD" else self.state.elapsed_time_s
        # A K-only mechanical reduction also changes electrical demand.
        effective_rate = min(self.rate_mechanical, self.config.deploy_efficiency*self.rate_dc)
        advance = effective_rate*max(0.0, float(t)-start)
        return min(self.config.total_mechanical_reduction_kw, self.state.target_reduction_mechanical_kw, r+advance)

    def decide(self, state, command, raw_action, action_space="signed"):
        raw = np.asarray(raw_action, float).reshape(-1)
        if raw.size != 1 or not np.isfinite(raw[0]) or abs(raw[0]) > 1+self.config.tolerance:
            raise ValueError("Exactly one finite action in [-1,1] is required")
        if action_space not in ("signed", "mode_aware", "reachable"):
            raise ValueError("Unknown v6 action-space mapping")
        t = float(_get(state, "elapsed_time_s", 0.0))
        if t < self.state.elapsed_time_s-self.config.tolerance:
            raise ValueError("Accepted driving decisions cannot move time backwards")
        self.raw_action = float(np.clip(raw[0], -1, 1))
        self.action_space = action_space
        braking = self._braking(command)
        cap = self._cap(_get(state, "s_m", 0.0), _get(state, "speed_mps", 0.0))
        reasons = []
        if action_space in ("mode_aware", "reachable"):
            magnitude = 0.5*(self.raw_action+1.0)
            request = -magnitude*self._torque_cap(True) if braking else magnitude*cap
        else:
            request = self.raw_action*(self._torque_cap(True) if self.raw_action < 0 else cap)
            if braking and request > 0:
                request = 0.0; reasons.append("positive_deployment_during_braking")
            if not braking and request < 0:
                request = 0.0; reasons.append("positive_wheel_superclip_disabled")
        if not braking and self._tail_mask(_get(state, "s_m", 0.0)):
            request = 0.0
            reasons.append("declared_tail_deployment_guard")
        r = self._reduction_at(t)
        phase = "LIMITED" if self.state.phase == "HOLD" and t >= self.state.hold_until_s-self.config.tolerance else self.state.phase
        st = replace(self.state, elapsed_time_s=t, reduction_mechanical_kw=r, phase=phase)
        if braking:
            # Declared approximation: genuine negative wheel demand terminates
            # the power-limited period. Merely negative acceleration does not.
            st = replace(st, phase="NORMAL", hold_until_s=t, reduction_mechanical_kw=0., target_reduction_mechanical_kw=0.,
                         reference_mechanical_kw=0., requested_dc_kw=request, braking=True)
        else:
            if st.braking:
                st = replace(st, phase="NORMAL", reference_mechanical_kw=0., reduction_mechanical_kw=0., target_reduction_mechanical_kw=0.,
                             hold_until_s=t, period_number=st.period_number+1)
            target = max(0., st.reference_mechanical_kw-max(0., request)*self.config.deploy_efficiency)
            if st.phase == "NORMAL" and target > self.config.tolerance:
                # An allowed initial PU reduction is not an extra DC slew
                # exemption: at high speed select a permitted zero reduction.
                r = min(target, self._initial_limit(_get(state, "s_m", 0.0))) if _get(state, "speed_mps", 0.0)*3.6 < 210. else 0.0
                st = replace(st, phase="HOLD", hold_until_s=t+self.config.initial_hold_s, reduction_mechanical_kw=r,
                             target_reduction_mechanical_kw=min(target, self.config.total_mechanical_reduction_kw))
            elif st.phase != "NORMAL":
                target = max(st.reduction_mechanical_kw, target)
                st = replace(st, target_reduction_mechanical_kw=min(target, self.config.total_mechanical_reduction_kw))
            st = replace(st, requested_dc_kw=max(0., request), braking=False)
        self.state = st
        self.request_kw = request
        info = {"raw_action": self.raw_action, "action_space": action_space, "requested_dc_kw": request,
                "guard_reasons": reasons, "rule_phase": st.phase, "reduction_mechanical_kw": st.reduction_mechanical_kw}
        self.decisions.append({"time_s": t, "s_m": float(_get(state, "s_m", 0.0)), **info})
        return info

    def _reserve(self, power_kw, hold_s):
        p = max(0.0, float(power_kw))
        return p*max(0., hold_s)/1000.0+p*p/(2000.0*self.rate_dc)

    def _hold_reserve_s(self, t):
        if self.state.phase == "NORMAL":
            return self.config.future_hold_reserve_s
        return max(0., self.state.hold_until_s-float(t)) if self.state.phase == "HOLD" else 0.0

    def query(self, state, command, geometry, dt_s):
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("A positive finite proposal time is required")
        t = float(_get(state, "elapsed_time_s", 0.0))
        speed = float(_get(state, "speed_mps", 0.0))
        s = (float(_get(state, "s_m", 0.0))+.5*speed*float(dt_s))%self.length_m
        cap = self._cap(s, speed)
        braking = self._braking(command)
        request = self.state.requested_dc_kw
        if braking:
            request = min(0.0, request)
            request = max(request, -self._torque_cap(True))
        else:
            request = max(0.0, request)
            if self._tail_mask(s) and self.state.phase == "NORMAL":
                request = min(request, max(0., float(_get(state, "previous_dc_kw", 0.0))))
            if self.state.phase != "NORMAL":
                request = max(0., self.state.reference_mechanical_kw-self._reduction_at(t+dt_s))/self.config.deploy_efficiency
            energy = float(_get(state, "energy_mj", self.config.energy_floor_mj) or self.config.energy_floor_mj)
            credit = max(0., energy-self.config.energy_floor_mj)
            hold = self._hold_reserve_s(t)
            duration = dt_s+hold
            reserve_bound = -self.rate_dc*duration+sqrt((self.rate_dc*duration)**2+2000.0*self.rate_dc*credit)
            request = min(request, cap, max(0., reserve_bound))
        alternative_threshold = 0.0
        alternative_cap = self.max_dc_kw
        for z in self._zones("alternative_power_zones_route_m", "alternative_power_zones_m"):
            a, b = (z["start_m"], z["end_m"]) if isinstance(z, dict) else z[:2]
            if _inside(s, a, b):
                alternative_threshold = max(alternative_threshold, float(z.get("threshold_kph", 310)) if isinstance(z, dict) else 310.)
                alternative_cap = min(alternative_cap, float(z.get("power_kw", 250)) if isinstance(z, dict) else 250.)
        return {"requested_dc_kw": float(request), "deployment_cap_kw": float(cap), "alternative_threshold_kph": alternative_threshold,
                "alternative_cap_kw": alternative_cap, "max_dc_kw": self.max_dc_kw, "harvest_limit_mj": self.harvest_limit_mj,
                "energy_floor_mj": self.config.energy_floor_mj, "energy_capacity_mj": self.config.energy_capacity_mj,
                "ramp_rate_kw_per_s": self.rate_dc}

    def next_time_event(self, t):
        return self.state.hold_until_s if self.state.phase == "HOLD" and self.state.hold_until_s > float(t)+1e-10 else inf

    def next_space_event(self, s):
        candidates = [self.length_m]
        if 0 < self.config.tail_guard_m < self.length_m:
            candidates.append(self.length_m-self.config.tail_guard_m)
        for route, official in (("alternative_power_zones_route_m", "alternative_power_zones_m"),
                                ("power_reduction_exceptions_route", "power_reduction_exceptions")):
            for z in self._zones(route, official):
                a, b = (z["start_m"], z["end_m"]) if isinstance(z, dict) else z[:2]
                candidates.extend((float(a), float(b)))
        return min((x for x in candidates if x > float(s)+1e-5), default=inf)

    def features(self, state, command):
        t = float(_get(state, "elapsed_time_s", 0.0))
        p = max(0., float(_get(state, "previous_dc_kw", self.state.previous_dc_kw)))
        return {"deployment_cap_kw": self._cap(_get(state, "s_m", 0.), _get(state, "speed_mps", 0.)),
                "reserve_mj": self._reserve(p, self._hold_reserve_s(t)),
                "reduction_active": float(self.state.phase != "NORMAL"), "reduction_pending": float(self.state.phase == "HOLD"),
                "hold_remaining_s": max(0., self.state.hold_until_s-t) if self.state.phase == "HOLD" else 0.,
                "reduction_total_kw": self._reduction_at(t), "low_speed_exempt": float(_get(state, "speed_mps", 0.)*3.6 < 210.),
                "torque_cap_kw": self._torque_cap(self._braking(command)),
                "harvested_mj": float(_get(state, "recharge_mj", 0.) or 0.), "requested_dc_kw": self.state.requested_dc_kw,
                "deployment_tail_mask": float(self._tail_mask(_get(state, "s_m", 0.))),
                "positive_wheel_superclip_enabled": 0.0, "full_fia_verified": 0.0}

    def accept(self, state, records=None):
        t = float(_get(state, "elapsed_time_s", 0.0))
        if t < self.state.elapsed_time_s-self.config.tolerance:
            raise ValueError("Accepted integration cannot move time backwards")
        previous = self.state.previous_dc_kw
        observed_peak = self.state.reference_mechanical_kw
        for rec in records or []:
            if rec.get("status", "accepted") not in ("accepted", "physical_stop"):
                continue
            dc = float(rec.get("dc_kw", rec.get("mguk_dc_kw", 0.0)))
            v0 = float(rec.get("speed_start_mps", _get(state, "speed_mps", 0.0)))
            v1 = float(rec.get("speed_next_mps", v0))
            s0 = float(rec.get("s_start_m", _get(state, "s_m", 0.0)))
            s1 = float(rec.get("s_end_m", s0))
            dt = float(rec.get("dt_s", rec.get("time_end_s", t)-rec.get("time_start_s", t)))
            wheel = float(rec.get("wheel_force_n", 0.0))
            cap = min(self._cap(s0, v0), self._cap(s1, v1))
            checks = []
            if abs(dc) > self.max_dc_kw+self.config.tolerance: checks.append("absolute_dc_power")
            if dc > cap+1e-4: checks.append("deployment_curve")
            if dc < -self.config.tolerance and wheel >= 0: checks.append("positive_wheel_superclip")
            theoretical = previous-self.rate_dc*max(0., dt)
            exempt = wheel < 0 or min(v0, v1)*3.6 < 210 or cap < previous-1e-4 or theoretical < 100
            if previous > 0 and not exempt and dc < theoretical-1e-4: checks.append("electrical_reduction_slew")
            physical_mech = dc*self.config.deploy_efficiency if dc >= 0 else dc/self.config.harvest_efficiency
            policing_mech = dc*self.config.ecu_efficiency_correction if dc >= 0 else dc/self.config.ecu_efficiency_correction
            omega = self.config.crankshaft_lower_bound_rpm*2*pi/60.0
            if abs(physical_mech)*1000/omega > self.config.mechanical_torque_limit_nm+1e-4: checks.append("engineering_mechanical_torque")
            if abs(policing_mech)*1000/omega > self.config.mechanical_torque_limit_nm+1e-4: checks.append("ecu_policing_torque")
            energy = rec.get("energy_next_mj")
            if energy is not None and not self.config.energy_floor_mj-1e-8 <= float(energy) <= self.config.energy_capacity_mj+1e-8: checks.append("declared_energy_window")
            recharge = rec.get("recharge_next_mj")
            if recharge is not None and float(recharge) > self.harvest_limit_mj+1e-8: checks.append("lap_recharge")
            if checks:
                self.violations.append({"s_m": s1, "time_s": float(rec.get("time_end_s", t)), "reasons": checks})
            previous = dc
            if self.state.phase == "NORMAL" and not self.state.braking:
                observed_peak = max(observed_peak, max(0., dc)*self.config.deploy_efficiency)
            self._accepted_records += 1
        r = self._reduction_at(t)
        phase = "LIMITED" if self.state.phase == "HOLD" and t >= self.state.hold_until_s-self.config.tolerance else self.state.phase
        actual_dc = float(_get(state, "previous_dc_kw", previous))
        reference = self.state.reference_mechanical_kw
        if phase == "NORMAL" and not self.state.braking:
            reference = max(observed_peak, max(0., actual_dc)*self.config.deploy_efficiency)
        self._elapsed_accumulated_s += max(0., t-self.state.elapsed_time_s)
        self.state = replace(self.state, elapsed_time_s=t, previous_dc_kw=actual_dc, reduction_mechanical_kw=r, phase=phase,
                             reference_mechanical_kw=reference)
        return self.state

    def certificate(self):
        initial = self._initial_boundary
        terminal = asdict(self.state)
        closed = initial["phase"] == terminal["phase"] and initial["braking"] == terminal["braking"] and all(
            abs(float(initial[k])-float(terminal[k])) <= self.config.tolerance
            for k in ("reduction_mechanical_kw", "target_reduction_mechanical_kw", "reference_mechanical_kw", "previous_dc_kw", "requested_dc_kw"))
        initial_hold = max(0., initial["hold_until_s"]-initial["elapsed_time_s"])
        terminal_hold = max(0., terminal["hold_until_s"]-terminal["elapsed_time_s"])
        closed = closed and abs(initial_hold-terminal_hold) <= self.config.tolerance
        return {"public_rules_passed": not self.violations, "public_subset_passed": not self.violations, "full_fia_verified": False,
                "physical_scope_label": "declared conservative public-rule subset; not FIA Standard ECU certification",
                "event_technical_issue": 19, "event_sporting_issue": 7, "event_date": "2026-06-28",
                "latest_reviewed_technical_issue": 20, "latest_reviewed_sporting_issue": 9,
                "boost_enabled": False, "overtake_enabled": False, "positive_wheel_superclip_enabled": False,
                "accepted_records_checked": self._accepted_records, "violations": list(self.violations),
                "initial_boundary_state": initial, "terminal_boundary_state": terminal, "terminal_state": terminal,
                "rule_state_closed": bool(closed), "config": asdict(self.config),
                "assumptions": {"period_start": "accepted positive K peak followed by lower mechanical demand",
                    "period_end": "genuine negative wheel demand; declared surrogate for non-public DOC-058 transitions",
                    "reference_demand": "actual reachable positive K mechanical peak in declared period; not a known FIA driver maximum demand map",
                    "initial_reduction": "at >=210 km/h choose permitted zero initial mechanical reduction; below 210 use at most150/exception350 kW; separate DC slew still applies",
                    "mechanical_reduction_rate": "minimum of event mechanical rate and engineering deploy efficiency times electrical rate; default95 kW/s",
                    "control_line": "carry reduction state and previous DC; reset lap recharge counter only",
                    "crankshaft_rpm": "declared 7500 rpm lower bound; calibration observed minimum 7617, autonomous transmission unverified",
                    "torque_scope": "both engineering and .97-policing conversion under declared rpm bound",
                    "energy": "ideal DC-equivalent energy in project 2–4 MJ window; actual ES SOC/losses unknown",
                    "tail_guard": "experimental narrow control family: prohibit new deployment in final tail_guard_m metres; reductions still follow hold/slew; not a FIA rule",
                    "auxiliary": "50 W sensitivity only, not measured or subtracted from actual SOC"},
                "auxiliary_energy_sensitivity_mj": self.config.auxiliary_sensitivity_w*self._elapsed_accumulated_s/1e6,
                "unverified": ["FIA-F1-DOC-058 period transitions and driver torque map", "actual engine speed and transmission",
                               "actual ES capacity/SOC, battery losses and auxiliary consumption", "precise aero-zone state", "fuel flow and ICE efficiency map"]}
