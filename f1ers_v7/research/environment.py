"""Explicit research boundaries over the unchanged ideal-energy executor.

The observation extension represents episode closure targets. It does not
claim a fully observed arbitrary FIA ECU: only the approved NORMAL, exhausted
hold, zero-reference/reduction/request boundary family is supported.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from ..deploy_only import (DeployOnlyEnv, DeployOnlySimulator, FEATURES as OLD_FEATURES,
    PROTOCOL as OLD_PROTOCOL, proxy_audit as old_proxy, numerical_audit as old_numerical)
from ..simulator import ReachableSimulator
from .._core.integrator import IntegratorConfig

PROTOCOL = 'v7_opensource_ideal_control'
FEATURES = tuple(OLD_FEATURES) + ('episode_initial_speed_mps_over_100',
                                 'episode_initial_dc_kw_over_350')
FEATURES36 = FEATURES


def require_policy_contract(policy):
    if (tuple(getattr(policy, 'v6_features', ())) != FEATURES
            or getattr(policy, 'v6_action_space', None) != 'mode_aware'
            or getattr(policy, 'v6_execution_protocol', None) != PROTOCOL
            or getattr(policy, 'observation_dim', None) != len(FEATURES)):
        raise ValueError('Research policy requires the exact 36-feature execution contract')
    return policy


def validate_boundary(value):
    """Validate a copy; caller authorization comes from its frozen teacher receipt."""
    value = deepcopy(value)
    speed = float(value['speed_mps'])
    energy = float(value['energy_mj'])
    carry = value.get('carried_rule_state')
    if not np.isfinite(speed) or speed <= 0 or not np.isfinite(energy) or not 2 <= energy <= 4:
        raise ValueError('Unsupported finite cold/warm speed or energy boundary')
    if carry is not None:
        if not isinstance(carry, dict) or carry.get('phase') != 'NORMAL' or carry.get('braking', False):
            raise ValueError('Only approved NORMAL nonbraking initial FSM states are supported')
        for name in ('reference_mechanical_kw', 'reduction_mechanical_kw',
                     'target_reduction_mechanical_kw', 'requested_dc_kw', 'previous_dc_kw'):
            if not np.isfinite(float(carry.get(name, 0.))) or abs(float(carry.get(name, 0.))) > 1e-6:
                raise ValueError('Unobserved nonzero initial FSM closure target: '+name)
        hold = float(carry.get('hold_until_s', 0.))-float(carry.get('elapsed_time_s', 0.))
        if not np.isfinite(hold) or hold > 1e-6:
            raise ValueError('An initial active hold is outside the approved boundary pool')
    return {'speed_mps': speed, 'energy_mj': energy, 'carried_rule_state': carry}


class ResearchEnv(DeployOnlyEnv):
    features = FEATURES

    def reset(self, seed=None, **kwargs):
        obs = super().reset(seed=seed, **kwargs)
        self.observation_dim = len(FEATURES)
        return obs

    def get_obs(self):
        return np.r_[super().get_obs(), self.initial.speed_mps/100.,
                     self.initial.previous_dc_kw/350.]

    def result(self):
        saved = self.observations
        self.observations = [np.asarray(o)[:len(OLD_FEATURES)] for o in saved]
        try:
            result = super().result()
        finally:
            self.observations = saved
        result['observations'] = np.asarray(saved, float).reshape(-1, len(FEATURES))
        result['summary'].update(protocol=PROTOCOL, feature_schema=FEATURES,
            observation_dim=len(FEATURES), research_energy_model='unchanged ideal DC-equivalent energy',
            episode_initial_speed_mps=self.initial.speed_mps,
            episode_initial_dc_kw=self.initial.previous_dc_kw,
            arbitrary_initial_FSM_observability_proven=False,
            policy_specific_periodic_shooting=False)
        return result


class ResearchSimulator(DeployOnlySimulator):
    protocol = PROTOCOL
    features = FEATURES
    obs_dim = observation_dim = len(FEATURES)
    env_class = ResearchEnv

    def create_env(self, action_space='mode_aware', reward_config=None,
                   initial_speed_mps=None, carried_rule_state=None, output_ds_m=5.,
                   numerics=None, initial_energy_mj=2.):
        if action_space != 'mode_aware':
            raise ValueError('Research execution retains the deployment-only scalar mapping')
        if isinstance(numerics, dict):
            numerics = IntegratorConfig(**numerics)
        speed = float(self.driver_template.safety_speed(0)) if initial_speed_mps is None else initial_speed_mps
        validate_boundary({'speed_mps': speed, 'energy_mj': initial_energy_mj,
                           'carried_rule_state': carried_rule_state})
        return self.env_class(self, action_space, reward_config, initial_speed_mps,
                           deepcopy(carried_rule_state), output_ds_m, numerics, initial_energy_mj)

    def simulate(self, controller=None, initial_speed_mps=None, action_space='mode_aware', **kwargs):
        if kwargs.get('periodic_shooting', False):
            raise ValueError('Research compares fixed complete boundaries, without speed shooting')
        if hasattr(controller, 'network'):
            require_policy_contract(controller)
        # Skip only the historical 34-dimensional policy guard, not its physics.
        return ReachableSimulator.simulate(self, controller, initial_speed_mps,
                                          action_space=action_space, **kwargs)

    def periodic_simulate(self, *args, **kwargs):
        raise ValueError('Research requires an immutable approved boundary; shooting is forbidden')


def _physical_view(result):
    if (result['summary'].get('protocol') != PROTOCOL
            or tuple(result['summary'].get('feature_schema', ())) != FEATURES):
        raise ValueError('Research physical audit requires an explicitly bound 36-feature result')
    value = dict(result)
    value['summary'] = dict(result['summary'], protocol=OLD_PROTOCOL,
                            feature_schema=OLD_FEATURES, observation_dim=len(OLD_FEATURES))
    value['observations'] = np.asarray(result['observations'])[:, :len(OLD_FEATURES)]
    return value


def proxy_audit(sim, result):
    audit = old_proxy(sim, _physical_view(result))
    return {**audit, 'research_observation_extension_only': True,
            'new_checkpoint_reinterpreted_as_old': False}


def numerical_audit(sim, result, output):
    audit = old_numerical(sim, _physical_view(result), output)
    return {**audit, 'research_observation_extension_only': True,
            'same_original_raw_demands_and_actions': True}


try:
    import gymnasium as gym
except ImportError:
    gym = None


class GymResearchEnv(gym.Env if gym is not None else object):
    """Gymnasium semantics over a fixed, caller-approved boundary pool.

    Numerical interruptions are truncated, never bootstrap-training samples.
    Physical failures and completed infeasible laps are absorbing terminations.
    """
    metadata = {'render_modes': []}

    def __init__(self, simulator, approved_boundaries, *, reward_config=None):
        if gym is None:
            raise ImportError('gymnasium is required for the isolated research adapter')
        self.simulator = simulator
        self.boundaries = tuple(validate_boundary(b) for b in approved_boundaries)
        if not self.boundaries:
            raise ValueError('An approved physical boundary pool is required')
        self.reward_config = reward_config
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(FEATURES),), dtype=np.float64)
        self.next_boundary_index = 0
        self.env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and set(options) != {'boundary_id'}:
            raise ValueError('Reset accepts only a preapproved boundary_id')
        index = self.next_boundary_index % len(self.boundaries) if not options else options['boundary_id']
        if isinstance(index,bool) or not isinstance(index,int) or not 0 <= index < len(self.boundaries):
            raise ValueError('Boundary_id must select an existing approved boundary')
        self.next_boundary_index += 1
        boundary = deepcopy(self.boundaries[index])
        self.env = self.simulator.create_env(initial_speed_mps=boundary['speed_mps'],
            initial_energy_mj=boundary['energy_mj'], carried_rule_state=boundary['carried_rule_state'],
            reward_config=self.reward_config)
        obs = np.asarray(self.env.get_obs(), np.float64)
        return obs, {'initial_boundary_index': index, 'initial_boundary': boundary,
                     'within_lap_reset': False, 'ideal_energy_model': True}

    def step(self, action):
        if self.env is None:
            raise RuntimeError('Reset the approved episode before stepping')
        action = np.asarray(action, float).reshape(-1)
        if action.shape != (1,) or not np.isfinite(action).all() or abs(action[0]) > 1:
            raise ValueError('One bounded finite raw actor action is required')
        before = self.env.get_obs().copy()
        obs, reward, done, info = self.env.step(float(action[0]))
        numeric = bool(done and info.get('status') == 'numerical_fail')
        info = {**info, 'raw_requested_action': float(action[0]),
                'actor_active_pre_transition': bool(before[10] >= 0.),
                'trainable_transition': not numeric,
                'bootstrap_allowed': not done,
                'numerical_interrupt_requires_stop': numeric}
        if done:
            info['terminal_observation'] = np.asarray(obs, float).copy()
        return np.asarray(obs, np.float64), float(reward), bool(done and not numeric), numeric, info
