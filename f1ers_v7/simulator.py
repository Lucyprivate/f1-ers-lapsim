"""Versioned rule-state observations with explicit engineering rejection."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from f1ers_v4.vehicle import IntegratorConfig
from f1ers_v6.environment import CausalEnvV6, FEATURES as OLD_FEATURES
from f1ers_v6.geometry_guard import GeometryGuardSimulator
from f1ers_v6.frozen_demands import FrozenDemandDriver
from f1ers_v6.proxy_audit import audit_proxy_consistency
from f1ers_v6.reward import transition_reward
from f1ers_v6.r7 import label_elapsed_time
from f1ers_v6.learning import _write
from .rules import EnergyViableRuleShield

FEATURES = tuple(OLD_FEATURES) + (
    'declared_reference_mechanical_kw_over_332_5',
    'target_reduction_mechanical_kw_over_700',
)


class ReachableEnv(CausalEnvV6):
    features = FEATURES

    def reset(self, seed=None, **kwargs):
        super().reset(seed, **kwargs)
        self.shield = EnergyViableRuleShield(self.sim.rules, self.sim.geometry,
            self.sim.rule_config, self.sim.geometry_guard_map, self.sim.margin_s,
            energy_notice_s=self.sim.energy_notice_s)
        self.shield.reset(self.state, self.carried_rule_state)
        self.driver.shield = self.shield
        self.observation_dim = len(FEATURES)
        return self.get_obs()

    def get_obs(self):
        base = super().get_obs()
        state = self.shield.state
        return np.r_[base, state.reference_mechanical_kw / 332.5,
            state.target_reduction_mechanical_kw / 700.]

    def step(self, action):
        before = self.state
        obs, reward, done, info = super().step(action)
        if self.shield.engineering_violations and self.status == 'accepted':
            self.done = True
            self.status = 'physical_fail'
            self.failure = {'guard_reasons': ['reachable_request_intersection'],
                'classification': 'added_engineering_boundary_rejection',
                'frozen_vehicle_kernel_status': 'accepted',
                'violations': list(self.shield.engineering_violations)}
            completed = self.state.s_m >= self.sim.length_m-self.config.event_s_tol
            reward, terms = transition_reward(before, self.state, 0., self.status,
                True, completed, self.reward_config, info['projection_fraction'])
            info.update(status=self.status, step_certified=False, terminal=True,
                guard_reasons=self.failure['guard_reasons'], reward_terms=terms,
                rejection_classification=self.failure['classification'])
            self.rewards[-1] = float(reward)
            return self.get_obs(), float(reward), True, info
        if done and info['completed_lap']:
            certificate = self.shield.certificate()
            reasons = []
            if self.state.energy_mj < self.initial.energy_mj-1e-8:
                reasons.append('terminal_energy_deficit')
            if abs(self.state.speed_mps-self.initial.speed_mps) > .02:
                reasons.append('terminal_speed_not_closed')
            if abs(self.state.previous_dc_kw-self.initial.previous_dc_kw) > .1:
                reasons.append('terminal_dc_not_closed')
            if not certificate['rule_state_closed']:
                reasons.append('terminal_rule_state_not_closed')
            if not certificate['public_rules_passed']:
                reasons.append('terminal_public_subset_rejected')
            info['terminal_constraint_reasons'] = reasons
            if reasons:
                # A complete physical lap can still be outside the sustainable
                # control domain. Penalize that terminal transition without
                # inventing a physical stop, changing SOC, or resetting speed.
                reward, terms = transition_reward(before, self.state, 0.,
                    'terminal_constraint_fail', True, True,
                    self.reward_config, info['projection_fraction'])
                info.update(reward_terms=terms, terminal_constraint_failed=True)
                self.rewards[-1] = float(reward)
                return obs, float(reward), done, info
        return obs, reward, done, info

    def result(self):
        # The inherited serializer uses the v6 dimension; serialize first, then
        # restore the independently versioned observation matrix.
        saved = self.observations
        self.observations = [np.asarray(o)[:len(OLD_FEATURES)] for o in saved]
        try:
            result = super().result()
        finally:
            self.observations = saved
        result['observations'] = np.asarray(saved, float).reshape(-1, len(FEATURES))
        result['geometry_guard_audit'] = self.shield.guard_audit
        result['energy_viability_audit'] = self.shield.energy_audit
        result['summary'].update(protocol=self.sim.protocol, observation_dim=len(FEATURES),
            feature_schema=FEATURES, geometry_guard_margin_s=self.sim.margin_s,
            energy_notice_s=self.sim.energy_notice_s,
            reachable_request_gate_passed=not self.shield.engineering_violations,
            request_proxy_consistent=not self.shield.engineering_violations,
            terminal_constraint_reasons=self.history[-1].get('terminal_constraint_reasons', []) if self.history else [],
            ers_validation_scope='declared_public_subset_and_added_engineering_gate',
            added_engineering_gate_is_fia_certification=False)
        result['summary']['feasible'] = bool(result['summary']['feasible'] and
            result['summary']['reachable_request_gate_passed'])
        return label_elapsed_time(result)


class ReachableSimulator(GeometryGuardSimulator):
    protocol = 'v7_reachable_control'
    features = FEATURES
    obs_dim = observation_dim = len(FEATURES)

    def __init__(self, *args, energy_notice_s=.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_notice_s = float(energy_notice_s)

    def create_env(self, action_space='signed', reward_config=None, initial_speed_mps=None,
            carried_rule_state=None, output_ds_m=5., numerics=None, initial_energy_mj=2.):
        if isinstance(numerics, dict):
            numerics = IntegratorConfig(**numerics)
        return ReachableEnv(self, action_space, reward_config, initial_speed_mps,
            carried_rule_state, output_ds_m, numerics, initial_energy_mj)


def proxy_audit(sim, result):
    shield = EnergyViableRuleShield(sim.rules, sim.geometry, sim.rule_config,
        sim.geometry_guard_map, sim.margin_s, energy_notice_s=sim.energy_notice_s)
    audit = audit_proxy_consistency(result, sim, shield)
    audit['engineering_violations'] = list(shield.engineering_violations)
    audit['reachable_request_gate_passed'] = not shield.engineering_violations
    audit['passed'] = bool(audit['reachable_request_gate_passed'] and
        audit['additional_surrogate_hold_and_limited_request_gate_passed'])
    return audit


def freeze_simulator(sim, profile):
    immutable = {k: np.asarray(profile[k], float).copy() for k in ('s_m', 'wheel_force_n')}
    for value in immutable.values():
        value.setflags(write=False)

    class FixedEnv(ReachableEnv):
        def reset(self, *args, **kwargs):
            super().reset(*args, **kwargs)
            self.driver = FrozenDemandDriver(self.sim.geometry, self.sim.vehicle, immutable)
            self.driver.shield = self.shield
            return self.get_obs()

    class FixedSimulator(ReachableSimulator):
        def create_env(self, action_space='signed', reward_config=None, initial_speed_mps=None,
                carried_rule_state=None, output_ds_m=5., numerics=None, initial_energy_mj=2.):
            if isinstance(numerics, dict):
                numerics = IntegratorConfig(**numerics)
            return FixedEnv(self, action_space, reward_config, initial_speed_mps,
                carried_rule_state, output_ds_m, numerics, initial_energy_mj)

    fixed = FixedSimulator(sim.track, sim.rules, sim.vehicle, sim.config, sim.rule_config,
        margin_s=sim.margin_s, energy_notice_s=sim.energy_notice_s)
    fixed.geometry_guard_map = sim.geometry_guard_map
    return fixed


def numerical_audit(sim, result, output):
    fixed = freeze_simulator(sim, result['driver_profile'])
    actions = np.asarray(result['requested_actions'], float).copy()
    common = np.linspace(0., sim.length_m, int(np.ceil(sim.length_m/.5))+1)
    comparisons, groups = {}, {}
    for group, settings in (
        ('output', [(10., .02, 1.), (5., .02, 1.), (2.5, .02, 1.)]),
        ('integration', [(5., .02, 1.), (5., .01, .5), (5., .005, .25)])):
        runs, proxies = [], []
        for i, (ds, dt, max_ds) in enumerate(settings):
            run = fixed.simulate(actions, initial_speed_mps=result['summary']['speed_start_mps'],
                initial_energy_mj=result['summary']['energy_start_mj'],
                carried_rule_state=result['summary']['initial_rule_state'],
                action_space=result['summary']['raw_action_space'], output_ds_m=ds,
                numerics=replace(sim.config, dt_max=dt, max_ds=max_ds))
            proxy = proxy_audit(fixed, run)
            _write(output/f'{group}_{i}.json.gz', run)
            _write(output/f'{group}_{i}_proxy.json', proxy)
            runs.append(run)
            proxies.append(proxy)
        rows = []
        for i, (a, b) in enumerate(zip(runs[:-1], runs[1:])):
            valid = all(r['summary']['feasible'] for r in (a, b)) and all(p['passed'] for p in proxies[i:i+2])
            def velocity(r):
                if group == 'integration':
                    return np.interp(common, r['internal_trace']['s_m'], r['internal_trace']['speed_mps'])
                return np.interp(common, np.r_[r['s_m'], sim.length_m],
                    np.r_[r['speed_mps'], r['terminal_speed_mps']])
            delta = abs(a['terminal_time_s']-b['terminal_time_s']) if valid else None
            rmse = float(np.sqrt(np.mean((velocity(a)-velocity(b))**2))*3.6) if valid else None
            rows.append({'valid': valid, 'lap_time_difference_s': delta, 'speed_rmse_kmh': rmse,
                'passed': bool(valid and delta <= .1 and rmse <= .5)})
        comparisons[group] = rows
        groups[group] = [r['summary'] for r in runs]
    return {'passed': all(row['passed'] for rows in comparisons.values() for row in rows),
        'comparisons': comparisons, 'summaries': groups,
        'fixed_raw_actions_and_original_5m_wheel_demands': True,
        'same_initial_state_and_causal_v7_shield': True,
        'output_resampling_and_internal_convergence_separate': True}
