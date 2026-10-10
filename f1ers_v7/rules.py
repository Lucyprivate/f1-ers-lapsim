"""Energy-viable deployment requests for the declared public-rule experiment.

The maximum driver power-demand map and DOC-058 transitions remain unknown.
This module repairs the project's reachable K-request surrogate, not that
unknown FIA map.  Requests, its ceiling, and actual supplied K power remain
separate.  The frozen v6 public checks and v4 vehicle kernel are unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from math import sqrt

import numpy as np

from f1ers_v7._core.integrator import IntegrationState
from f1ers_v7._core.geometry_guard import GeometryAdvanceRuleShield
from f1ers_v7._core.base_rules import _get


class EnergyViableRuleShield(GeometryAdvanceRuleShield):
    """Start a causal reduction before energy clips a historical peak.

    A new positive request reserves an entire future hold, a reduction to zero,
    and one frozen decision notice interval.  If that available request falls
    below the accepted period peak, it requests a reduction all the way to zero
    through the inherited HOLD/LIMITED state machine.  Targets cannot retreat
    before true negative wheel demand ends that declared period.

    The notice is a tested engineering assumption, not a universal reachability
    proof.  A positive NORMAL decision longer than that notice is rejected.
    Proposal queries never mutate the state or conceal impossible intersections;
    accepted records retain and classify any shortfall for rejection by the env.
    """

    def __init__(self, rules, geometry, config, geometry_map, margin_s,
                 energy_notice_s=.25):
        super().__init__(rules, geometry, config, geometry_map, margin_s)
        self.energy_notice_s = float(energy_notice_s)
        if not np.isfinite(self.energy_notice_s) or self.energy_notice_s <= 0:
            raise ValueError('A finite positive energy decision notice is required')
        self.energy_audit = []
        self.engineering_violations = []
        self.max_positive_normal_decision_dt_s = 0.
        self._last_accepted_state = None

    def reset(self, state, carried_state=None):
        restored = super().reset(state, carried_state)
        self._last_accepted_state = state
        self.energy_audit.clear()
        self.engineering_violations.clear()
        self.max_positive_normal_decision_dt_s = 0.
        return restored

    def _energy_bound(self, state, duration_s):
        credit = max(0., float(_get(state, 'energy_mj', 2.) or 2.)
                     - self.config.energy_floor_mj)
        duration = max(0., float(duration_s))
        # Stable positive root of p*duration+p^2/(2*rate) = credit*1000.
        a = self.rate_dc * duration
        root = sqrt(a*a + 2000.*self.rate_dc*credit)
        return 2000.*self.rate_dc*credit / (root+a) if root+a > 0. else 0.

    def decide(self, state, command, raw_action, action_space='signed'):
        raw_array = np.asarray(raw_action, float).reshape(-1)
        if (raw_array.size != 1 or not np.isfinite(raw_array[0])
                or abs(raw_array[0]) > 1.+self.config.tolerance):
            raise ValueError('One finite actor action in [-1,1] is required')
        if action_space not in ('signed', 'mode_aware'):
            raise ValueError('Unknown v7 action-space mapping')
        raw = float(raw_array[0])
        braking = self._braking(command)
        t = float(_get(state, 'elapsed_time_s', 0.))
        cap = self._cap(_get(state, 's_m', 0.), _get(state, 'speed_mps', 0.))
        original_positive = (max(0., .5*(raw+1.))*cap if action_space == 'mode_aware'
                             else max(0., raw)*cap)
        previous = self.state
        reference_dc = previous.reference_mechanical_kw/self.config.deploy_efficiency
        current_ceiling_dc = max(0., previous.reference_mechanical_kw
                                 - self._reduction_at(t))/self.config.deploy_efficiency
        normal_bound = self._energy_bound(
            state, self.config.future_hold_reserve_s+self.energy_notice_s)
        remaining_bound = self._energy_bound(
            state, self._hold_reserve_s(t)+self.energy_notice_s)
        energy_reduction = False
        filtered = raw
        if not braking:
            if previous.phase == 'NORMAL' and normal_bound < reference_dc-self.config.tolerance:
                # A tiny request reduction would merely stop the ramp near the
                # old peak.  Reserve exhaustion requires committing to zero.
                energy_reduction = True
                positive = 0.
            elif (previous.phase != 'NORMAL' and remaining_bound
                  < current_ceiling_dc-self.config.tolerance):
                energy_reduction = True
                positive = 0.
            else:
                positive = min(original_positive, normal_bound) if previous.phase == 'NORMAL' else original_positive
            magnitude = positive/cap if cap > 0. else 0.
            filtered = 2.*magnitude-1. if action_space == 'mode_aware' else magnitude
        info = super().decide(state, command, filtered, action_space)
        if not braking and previous.phase != 'NORMAL' and not previous.braking:
            # Freeze the chosen terminal reduction, not only its achieved value.
            # Reopening the actor demand cannot undo a pending ramp.
            target = max(previous.target_reduction_mechanical_kw,
                         self.state.target_reduction_mechanical_kw)
            self.state = replace(self.state, target_reduction_mechanical_kw=target)
        if energy_reduction:
            self.state = replace(self.state, target_reduction_mechanical_kw=min(
                self.state.reference_mechanical_kw, self.config.total_mechanical_reduction_kw))
            info['guard_reasons'].append('energy_viability_advance_reduction_to_zero')
        elif not braking and filtered != raw:
            info['guard_reasons'].append('future_hold_ramp_and_notice_energy_reserve')
        audit = {'s_m': float(_get(state, 's_m', 0.)), 'time_s': t,
                 'actual_braking': braking, 'actor_action': raw,
                 'energy_filtered_action': filtered,
                 'actor_positive_request_kw': original_positive,
                 'energy_normal_new_request_bound_dc_kw': normal_bound,
                 'energy_remaining_period_bound_dc_kw': remaining_bound,
                 'historical_reachable_peak_dc_kw': reference_dc,
                 'declared_request_ceiling_dc_kw': current_ceiling_dc,
                 'energy_triggered_reduction_to_zero': energy_reduction,
                 'energy_credit_mj': max(0., float(_get(state, 'energy_mj', 2.))-2.),
                 'decision_notice_s': self.energy_notice_s,
                 'state_after_decide': asdict(self.state)}
        self.energy_audit.append(audit)
        info['energy_viability_audit'] = audit
        return info

    def query(self, state, command, geometry, dt_s):
        query = super().query(state, command, geometry, dt_s)
        t = float(_get(state, 'elapsed_time_s', 0.))
        braking = self._braking(command)
        phase = self.state.phase
        if phase == 'HOLD' and t >= self.state.hold_until_s-self.config.tolerance:
            phase = 'LIMITED'
        declared = max(0., self.state.reference_mechanical_kw
                       - self._reduction_at(t+float(dt_s)))/self.config.deploy_efficiency
        public = float(query['deployment_cap_kw'])
        effective = max(0., float(query['requested_dc_kw']))
        # Public caps can independently reduce availability.  This extra gate
        # checks only unexplained energy clipping below the declared surrogate.
        gap = max(0., min(declared, public)-effective) if not braking else 0.
        bound = self._energy_bound(state, float(dt_s)+self._hold_reserve_s(t))
        query.update(declared_proxy_dc_kw=declared, energy_bound_dc_kw=bound,
                     effective_positive_request_dc_kw=effective,
                     request_proxy_shortfall_dc_kw=gap,
                     request_proxy_phase=phase,
                     constraint_intersection_feasible=bool(gap <= self.config.tolerance),
                     full_FIA_maximum_demand_map_claimed=False)
        return query

    def accept(self, state, records=None):
        records = records or []
        initial = self._last_accepted_state
        if initial is not None:
            decision_dt = float(_get(state, 'elapsed_time_s', 0.))-float(
                _get(initial, 'elapsed_time_s', 0.))
            if (self.state.phase == 'NORMAL' and not self.state.braking
                    and max(self.state.reference_mechanical_kw,
                            self.state.requested_dc_kw) > self.config.tolerance):
                self.max_positive_normal_decision_dt_s = max(
                    self.max_positive_normal_decision_dt_s, decision_dt)
                if decision_dt > self.energy_notice_s+self.config.tolerance:
                    self.engineering_violations.append({
                        's_m': float(_get(state, 's_m', 0.)),
                        'time_s': float(_get(state, 'elapsed_time_s', 0.)),
                        'reasons': ['energy_decision_notice_exceeded'],
                        'decision_dt_s': decision_dt, 'frozen_notice_s': self.energy_notice_s})
            energy = float(_get(initial, 'energy_mj', 2.))
            recharge = float(_get(initial, 'recharge_mj', 0.) or 0.)
            previous_dc = float(_get(initial, 'previous_dc_kw', 0.))
            for record in records:
                if record.get('status', 'accepted') not in ('accepted', 'physical_stop'):
                    continue
                dt = float(record.get('dt_s', 0.))
                if dt <= 0.:
                    continue
                before = IntegrationState(
                    float(record.get('s_start_m', _get(initial, 's_m', 0.))),
                    float(record.get('speed_start_mps', _get(initial, 'speed_mps', 0.))),
                    float(record.get('time_start_s', _get(initial, 'elapsed_time_s', 0.))),
                    energy, recharge, previous_dc)
                query = self.query(before, {'wheel_force_n': record.get('wheel_force_n', 0.),
                                           'brake_fraction': 0.}, self.geometry, dt)
                if not query['constraint_intersection_feasible']:
                    self.engineering_violations.append({
                        's_m': before.s_m, 'time_s': before.elapsed_time_s,
                        'reasons': ['reachable_reference_energy_request_intersection_empty'],
                        'rule_phase': query['request_proxy_phase'],
                        'declared_proxy_dc_kw': query['declared_proxy_dc_kw'],
                        'energy_bound_dc_kw': query['energy_bound_dc_kw'],
                        'effective_positive_request_dc_kw': query['effective_positive_request_dc_kw'],
                        'shortfall_dc_kw': query['request_proxy_shortfall_dc_kw']})
                energy = float(record.get('energy_next_mj', energy))
                recharge = float(record.get('recharge_next_mj', recharge))
                previous_dc = float(record.get('dc_kw', previous_dc))
        accepted = super().accept(state, records)
        self._last_accepted_state = state
        return accepted

    def certificate(self):
        certificate = super().certificate()
        certificate['reachable_request_gate_passed'] = not self.engineering_violations
        certificate['engineering_violations'] = list(self.engineering_violations)
        certificate['energy_viability_guard'] = {
            'decision_notice_s': self.energy_notice_s,
            'maximum_positive_NORMAL_decision_duration_s': self.max_positive_normal_decision_dt_s,
            'new_deployment_reserve': 'future hold + full DC ramp to zero + frozen decision notice',
            'energy_trigger': 'start/extend monotonic reduction to zero before historical peak becomes unavailable',
            'pure_proposals': True, 'actual_power_overwritten': False,
            'universal_reachability_proof': False, 'additional_engineering_gate': True,
            'FIA_C5_12_4_actual_K_constant_requirement_claimed': False}
        certificate['assumptions']['target_reduction'] = (
            'target and attained reduction are both nondecreasing until genuine negative wheel demand')
        return certificate
