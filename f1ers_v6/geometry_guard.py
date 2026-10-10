"""Advance deployment reduction using the fixed known geometry.

This experimental layer limits new requests. The existing public-subset FSM
still implements every actual reduction, hold and accepted-state commit; the
frozen v4 vehicle kernel is reused without alteration. No observed future
velocity, local residual mask or new policy observation channel is used.
"""
from __future__ import annotations

import numpy as np

from .environment import CausalEnvV6
from .rules import RuleShield
from .simulator import SimulatorV6


def build_geometry_map(driver, params):
    """Mechanical kinetic work plus resistance along the 1 m safety profile."""
    s = np.asarray(driver.safe_s[:-1], float)
    speed = np.asarray(driver.safe_speed[:-1], float)
    ds = driver.length_m / len(s)
    next_v = np.roll(speed, -1)
    g = driver.geometry.evaluate(s + .5 * ds)
    vmid = .5 * (speed + next_v)
    drag = (.5 * params.air_density_kgpm3 * np.asarray(g['drag_area_m2']) * vmid**2
            + params.rolling_coefficient * params.mass_kg * 9.81)
    force = params.mass_kg * (next_v**2 - speed**2) / (2 * ds) + drag
    wheel_kw = force * vmid / 1000.
    # Fixed 350 kW DC at the frozen engineering efficiency .95.
    positive_low = (wheel_kw > 0.) & (wheel_kw < 332.5)
    return {'s_m': s, 'safe_speed_mps': speed, 'ds_m': ds,
            'wheel_force_n': force, 'wheel_power_kw': wheel_kw,
            'positive_low_mask': positive_low, 'negative_wheel_mask': force < 0.,
            'hazard_s_m': s[positive_low], 'length_m': driver.length_m}


class GeometryAdvanceRuleShield(RuleShield):
    """Restrict reachable new deployment early; never clip actual accepted DC."""

    def __init__(self, rules, geometry, config, geometry_map, margin_s):
        super().__init__(rules, geometry, config)
        self.geometry_map = geometry_map
        self.margin_s = float(margin_s)
        if not np.isfinite(self.margin_s) or self.margin_s < 0:
            raise ValueError('One finite nonnegative global margin is required')
        self.guard_audit = []

    def hazard(self, state):
        m = self.geometry_map
        positions = m['hazard_s_m']
        if len(positions) == 0:
            return {'hazard_distance_m': None, 'hazard_s_m': None,
                    'arrival_lower_bound_s': None,
                    'guard_dc_request_upper_kw': self.max_dc_kw}
        s = float(state.s_m) % self.length_m
        cell = int(np.searchsorted(m['s_m'], s, side='right') - 1)
        cell = max(0, min(len(m['s_m'])-1, cell))
        if m['positive_low_mask'][cell]:
            target, distance = s, 0.
        else:
            i = int(np.searchsorted(positions, s, side='left'))
            target = float(positions[i]) if i < len(positions) else float(positions[0]) + self.length_m
            distance = target - s
        unwrapped = np.r_[m['s_m'], m['s_m'] + self.length_m]
        vv = np.r_[m['safe_speed_mps'], m['safe_speed_mps']]
        on_path = (unwrapped >= s) & (unwrapped <= target + m['ds_m'])
        # Conservative notice within the known geometric safety profile. This
        # is an experimental forecast, not a universal viability proof.
        known_v_bound = max(float(state.speed_mps),
                           float(np.max(vv[on_path], initial=0.)))
        arrival = distance / max(known_v_bound, 1e-9)
        request_upper = self.rate_dc * max(0., arrival - self.config.initial_hold_s - self.margin_s)
        return {'hazard_distance_m': distance,
                'hazard_s_m': target % self.length_m,
                'arrival_lower_bound_s': arrival,
                'arrival_speed_upper_mps': known_v_bound,
                'guard_dc_request_upper_kw': min(self.max_dc_kw, request_upper)}

    def decide(self, state, command, raw_action, action_space='signed'):
        raw_array = np.asarray(raw_action, float).reshape(-1)
        if raw_array.size != 1 or not np.isfinite(raw_array[0]) or abs(raw_array[0]) > 1. + self.config.tolerance:
            raise ValueError('One finite actor action in [-1,1] is required')
        raw = float(raw_array[0])
        hazard = self.hazard(state)
        cap = self._cap(state.s_m, state.speed_mps)
        original_request = (max(0., .5 * (raw + 1.)) * cap
                            if action_space == 'mode_aware' else max(0., raw) * cap)
        filtered, guarded = raw, original_request
        braking = self._braking(command)
        if not braking:
            guarded = min(original_request, hazard['guard_dc_request_upper_kw'])
            magnitude = guarded / cap if cap > 0. else 0.
            filtered = 2. * magnitude - 1. if action_space == 'mode_aware' else magnitude
        info = super().decide(state, command, filtered, action_space=action_space)
        audit = {'s_m': float(state.s_m), 'time_s': float(state.elapsed_time_s),
            'actual_braking': braking, 'actor_action': raw, 'guarded_action': filtered,
            'actor_positive_request_kw': original_request,
            'guarded_positive_request_kw': guarded,
            'previous_accepted_dc_kw': float(state.previous_dc_kw),
            'rule_phase_after_decide': self.state.phase,
            'hold_remaining_s': max(0., self.state.hold_until_s-self.state.elapsed_time_s),
            'global_margin_s': self.margin_s, **hazard}
        self.guard_audit.append(audit)
        info['geometry_guard_audit'] = audit
        return info

    def certificate(self):
        certificate = super().certificate()
        certificate['experimental_geometry_advance_guard'] = {
            'margin_s': self.margin_s, 'minimum_hold_s': self.config.initial_hold_s,
            'fixed_geometric_power_threshold_kw': 332.5,
            'origin': '1m periodic geometry safety-speed kinetic work and resistance',
            'future_observed_velocity_used': False,
            'new_fia_rule_claimed': False,
            'original_feature_schema_unchanged': True,
            'post_proposal_or_accepted_state_clipping': False}
        return certificate


class GeometryGuardEnv(CausalEnvV6):
    def reset(self, seed=None, **kwargs):
        super().reset(seed, **kwargs)
        self.shield = GeometryAdvanceRuleShield(self.sim.rules, self.sim.geometry,
            self.sim.rule_config, self.sim.geometry_guard_map, self.sim.margin_s)
        self.shield.reset(self.state, self.carried_rule_state)
        self.driver.shield = self.shield
        return self.get_obs()

    def result(self):
        result = super().result()
        result['geometry_guard_audit'] = self.shield.guard_audit
        result['summary']['geometry_advance_guard_enabled'] = True
        result['summary']['geometry_guard_margin_s'] = self.sim.margin_s
        return result


class GeometryGuardSimulator(SimulatorV6):
    def __init__(self, *args, margin_s=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin_s = float(margin_s)
        self.geometry_guard_map = build_geometry_map(self.driver_template, self.vehicle)

    def create_env(self, action_space='signed', reward_config=None, initial_speed_mps=None,
                   carried_rule_state=None, output_ds_m=5., numerics=None, initial_energy_mj=2.):
        if isinstance(numerics, dict):
            from f1ers_v4.vehicle import IntegratorConfig
            numerics = IntegratorConfig(**numerics)
        return GeometryGuardEnv(self, action_space, reward_config, initial_speed_mps,
            carried_rule_state, output_ds_m, numerics, initial_energy_mj)
