"""Deployment-only actor with bounded regeneration on genuine wheel braking.

The extra action mapping is an engineering experiment, not a new FIA rule.
The actor keeps its raw mode-aware scalar.  Only a negative signed wheel
request maps that scalar to full requested regeneration.  Actual allocation
continues through the frozen combined tire budget, energy room, lap recharge,
torque, DC power and accepted-record public-rule checks.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np

from f1ers_v4.vehicle import IntegratorConfig
from f1ers_v6.frozen_demands import FrozenDemandDriver
from f1ers_v6.learning import _write
from f1ers_v6.proxy_audit import audit_proxy_consistency
from f1ers_v6.rules import _get
from f1ers_v7.rules import EnergyViableRuleShield
from f1ers_v7.learning import load_policy
from f1ers_v7.simulator import ReachableEnv, ReachableSimulator, FEATURES as OLD_FEATURES

PROTOCOL = 'v7_deployment_only_control'
FEATURES = tuple(OLD_FEATURES)+('episode_initial_energy_mj_over_4',)


def require_policy_contract(policy):
    if (tuple(getattr(policy, 'v6_features', ())) != FEATURES
            or getattr(policy, 'v6_action_space', None) != 'mode_aware'
            or getattr(policy, 'v6_execution_protocol', None) != PROTOCOL
            or getattr(policy, 'observation_dim', None) != len(FEATURES)):
        raise ValueError('Deployment-only policy requires the exact 34-feature schema and protocol')
    return policy


def load_deploy_policy(path):
    return require_policy_contract(load_policy(path, expected_features=FEATURES,
        expected_action_space='mode_aware', expected_execution_protocol=PROTOCOL))


class DeployOnlyShield(EnergyViableRuleShield):
    """Fix negative-wheel mapped demand without changing signed wheel force."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deployment_only_audit = []

    @staticmethod
    def _braking(command):
        force = _get(command, 'wheel_force_n')
        if force is None or not np.isfinite(float(force)):
            raise ValueError('Deployment-only execution requires a finite signed wheel demand')
        return float(force) < 0.

    def reset(self, state, carried_state=None):
        restored = super().reset(state, carried_state)
        self.deployment_only_audit.clear()
        return restored

    def decide(self, state, command, raw_action, action_space='mode_aware'):
        if action_space != 'mode_aware':
            raise ValueError('Deployment-only execution accepts only the declared mode-aware scalar')
        raw_array = np.asarray(raw_action, float).reshape(-1)
        if (raw_array.size != 1 or not np.isfinite(raw_array[0])
                or abs(raw_array[0]) > 1.+self.config.tolerance):
            raise ValueError('One finite actor action in [-1,1] is required')
        raw = float(np.clip(raw_array[0], -1., 1.))
        braking = self._braking(command)
        mapped = 1. if braking else raw
        info = super().decide(state, command, mapped, action_space)
        # The FSM demand remains mapped/guarded; the actor's action is preserved.
        self.raw_action = raw
        info['raw_action'] = raw
        info['mapped_action'] = mapped
        self.decisions[-1].update(raw_action=raw, mapped_action=mapped,
            deployment_only_mapping=braking)
        audit = {'s_m': float(_get(state, 's_m', 0.)),
            'time_s': float(_get(state, 'elapsed_time_s', 0.)),
            'actor_raw_action': raw, 'mapped_action': mapped,
            'signed_wheel_demand_n': float(_get(command, 'wheel_force_n')),
            'genuine_negative_wheel_demand': braking,
            'full_regeneration_requested': braking,
            'effective_request_after_existing_shields_kw': info['requested_dc_kw'],
            'original_wheel_demand_unchanged': True,
            'actual_allocation_clipped_by_frozen_kernel': True}
        self.deployment_only_audit.append(audit)
        info['deployment_only_mapping_audit'] = audit
        return info

    def certificate(self):
        result = super().certificate()
        result['deployment_only_mapping'] = {
            'actor_raw_scalar_preserved': True,
            'action_space': 'mode_aware',
            'negative_signed_wheel_demand_maps_to_one': True,
            'positive_wheel_raw_mapping_unchanged': True,
            'new_fia_rule_claimed': False,
            'independent_regeneration_or_braking_budget': False,
            'braking_inputs_changed': False,
            'positive_wheel_superclip_enabled': False}
        return result


class DeployOnlyEnv(ReachableEnv):
    features = FEATURES

    def reset(self, seed=None, **kwargs):
        super().reset(seed, **kwargs)
        self.shield = DeployOnlyShield(self.sim.rules, self.sim.geometry,
            self.sim.rule_config, self.sim.geometry_guard_map, self.sim.margin_s,
            energy_notice_s=self.sim.energy_notice_s)
        self.shield.reset(self.state, self.carried_rule_state)
        self.driver.shield = self.shield
        self.observation_dim = len(FEATURES)
        return self.get_obs()

    def get_obs(self):
        obs = np.r_[super().get_obs(), self.initial_energy_mj/4.]
        if obs.shape != (len(FEATURES),) or not np.isfinite(obs).all():
            raise RuntimeError('Nonfinite deployment-only 34-feature observation')
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        mapping = self.shield.deployment_only_audit[-1]
        info.update(mapped_action=mapping['mapped_action'],
            deployment_only_mapping_audit=mapping,
            episode_initial_energy_mj=self.initial_energy_mj)
        return obs, reward, done, info

    def result(self):
        # Preserve the frozen serializer's execution fields, then explicitly
        # restore this independently bound 34-column observation matrix.
        saved = self.observations
        self.observations = [np.asarray(o)[:len(OLD_FEATURES)] for o in saved]
        try:
            result = super().result()
        finally:
            self.observations = saved
        result['observations'] = np.asarray(saved, float).reshape(-1, len(FEATURES))
        result['mapped_actions'] = np.asarray([row['mapped_action']
            for row in self.shield.deployment_only_audit], float)
        result['deployment_only_mapping_audit'] = list(self.shield.deployment_only_audit)
        result['summary'].update(protocol=PROTOCOL, feature_schema=FEATURES,
            observation_dim=len(FEATURES), episode_initial_energy_mj=self.initial_energy_mj,
            actor_controls_positive_deployment_only=True,
            regeneration_requested_maximum_only_under_negative_signed_wheel_demand=True,
            actor_raw_actions_preserved=True, brake_wheel_demand_modified=False)
        return result


class DeployOnlySimulator(ReachableSimulator):
    protocol = PROTOCOL
    features = FEATURES
    obs_dim = observation_dim = len(FEATURES)

    def create_env(self, action_space='mode_aware', reward_config=None,
            initial_speed_mps=None, carried_rule_state=None, output_ds_m=5.,
            numerics=None, initial_energy_mj=2.):
        if action_space != 'mode_aware':
            raise ValueError('Deployment-only execution requires mode_aware')
        if isinstance(numerics, dict):
            numerics = IntegratorConfig(**numerics)
        return DeployOnlyEnv(self, action_space, reward_config, initial_speed_mps,
            carried_rule_state, output_ds_m, numerics, initial_energy_mj)

    def simulate(self, controller=None, initial_speed_mps=None,
                 action_space='mode_aware', **kwargs):
        if hasattr(controller, 'network'):
            require_policy_contract(controller)
        return super().simulate(controller, initial_speed_mps, action_space=action_space, **kwargs)


def proxy_audit(sim, result):
    """Replay original raw actor actions through this same mapping class."""
    if (result['summary'].get('protocol') != PROTOCOL
            or tuple(result['summary'].get('feature_schema', ())) != FEATURES):
        raise ValueError('A deployment-only result must retain its exact executor/schema')
    shield = DeployOnlyShield(sim.rules, sim.geometry, sim.rule_config,
        sim.geometry_guard_map, sim.margin_s, energy_notice_s=sim.energy_notice_s)
    audit = audit_proxy_consistency(result, sim, shield)
    mapping = shield.deployment_only_audit
    saved = result.get('deployment_only_mapping_audit', [])
    raw = np.asarray(result['requested_actions'], float)
    mapped = np.asarray(result.get('mapped_actions', []), float)
    exact = (len(mapping) == len(saved) == len(raw) == len(mapped)
             and mapping == saved
             and all(row['actor_raw_action'] == float(value)
                     for row, value in zip(mapping, raw))
             and all(row['mapped_action'] == float(value)
                     for row, value in zip(mapping, mapped)))
    audit['deployment_only_mapping_reconstruction_exact'] = bool(exact)
    audit['actor_raw_actions_replayed_without_substitution'] = True
    audit['engineering_violations'] = list(shield.engineering_violations)
    audit['reachable_request_gate_passed'] = not shield.engineering_violations
    audit['passed'] = bool(exact and audit['reachable_request_gate_passed']
        and audit['additional_surrogate_hold_and_limited_request_gate_passed'])
    return audit


def freeze_simulator(sim, profile):
    immutable = {key: np.asarray(profile[key], float).copy()
                 for key in ('s_m', 'wheel_force_n')}
    for value in immutable.values():
        value.setflags(write=False)

    class FixedEnv(DeployOnlyEnv):
        def reset(self, *args, **kwargs):
            super().reset(*args, **kwargs)
            self.driver = FrozenDemandDriver(self.sim.geometry, self.sim.vehicle, immutable)
            self.driver.shield = self.shield
            return self.get_obs()

    class FixedSimulator(DeployOnlySimulator):
        def create_env(self, action_space='mode_aware', reward_config=None,
                initial_speed_mps=None, carried_rule_state=None, output_ds_m=5.,
                numerics=None, initial_energy_mj=2.):
            if action_space != 'mode_aware':
                raise ValueError('Frozen deployment-only executor requires mode_aware')
            if isinstance(numerics, dict):
                numerics = IntegratorConfig(**numerics)
            return FixedEnv(self, action_space, reward_config, initial_speed_mps,
                carried_rule_state, output_ds_m, numerics, initial_energy_mj)

    fixed = FixedSimulator(sim.track, sim.rules, sim.vehicle, sim.config, sim.rule_config,
        margin_s=sim.margin_s, energy_notice_s=sim.energy_notice_s)
    fixed.geometry_guard_map = sim.geometry_guard_map
    return fixed


def numerical_audit(sim, result, output):
    output = Path(output)
    fixed = freeze_simulator(sim, result['driver_profile'])
    actions = np.asarray(result['requested_actions'], float).copy()
    common = np.linspace(0., sim.length_m, int(np.ceil(sim.length_m/.5))+1)
    comparisons, groups = {}, {}
    for group, settings in (
        ('output', [(10., .02, 1.), (5., .02, 1.), (2.5, .02, 1.)]),
        ('integration', [(5., .02, 1.), (5., .01, .5), (5., .005, .25)])):
        runs, proxies = [], []
        for index, (ds, dt, max_ds) in enumerate(settings):
            run = fixed.simulate(actions, initial_speed_mps=result['summary']['speed_start_mps'],
                initial_energy_mj=result['summary']['energy_start_mj'],
                carried_rule_state=result['summary']['initial_rule_state'],
                action_space=result['summary']['raw_action_space'], output_ds_m=ds,
                numerics=replace(sim.config, dt_max=dt, max_ds=max_ds))
            proxy = proxy_audit(fixed, run)
            _write(output/f'{group}_{index}.json.gz', run)
            _write(output/f'{group}_{index}_proxy.json', proxy)
            runs.append(run)
            proxies.append(proxy)
        rows = []
        for index, (a, b) in enumerate(zip(runs[:-1], runs[1:])):
            valid = all(r['summary']['feasible'] for r in (a, b)) and all(
                p['passed'] for p in proxies[index:index+2])
            def velocity(run):
                if group == 'integration':
                    return np.interp(common, run['internal_trace']['s_m'], run['internal_trace']['speed_mps'])
                return np.interp(common, np.r_[run['s_m'], sim.length_m],
                    np.r_[run['speed_mps'], run['terminal_speed_mps']])
            delta = abs(a['terminal_time_s']-b['terminal_time_s']) if valid else None
            rmse = float(np.sqrt(np.mean((velocity(a)-velocity(b))**2))*3.6) if valid else None
            rows.append({'valid': valid, 'lap_time_difference_s': delta, 'speed_rmse_kmh': rmse,
                'passed': bool(valid and delta <= .1 and rmse <= .5)})
        comparisons[group], groups[group] = rows, [run['summary'] for run in runs]
    return {'passed': all(row['passed'] for rows in comparisons.values() for row in rows),
        'comparisons': comparisons, 'summaries': groups,
        'fixed_raw_actions_and_original_5m_wheel_demands': True,
        'same_initial_state_and_causal_deployment_only_shield': True,
        'deployment_only_mapping_and_34_feature_schema_retained': True,
        'output_resampling_and_internal_convergence_separate': True}
