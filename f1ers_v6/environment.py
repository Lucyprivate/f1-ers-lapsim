"""One accepted time-domain execution path for learning and evaluation."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from f1ers_v4.vehicle import IntegrationState, advance_interval
from .driver import PowerResponsiveDriver

FEATURES = ('s_fraction', 'phase_sin', 'phase_cos', 'speed_scaled',
    'energy_fraction', 'deployable_credit_fraction', 'recharge_fraction',
    'previous_dc_fraction', 'curvature_scaled', 'lateral_utilization',
    'wheel_demand_fraction', 'actual_braking', 'safe_speed_fraction',
    'safe_100m_fraction', 'safe_250m_fraction', 'safe_500m_fraction',
    'drag_fraction', 'aero_straight', 'sector_1', 'sector_2', 'sector_3',
    'deployment_cap_fraction', 'reserve_fraction', 'reduction_active',
    'reduction_pending', 'hold_remaining_fraction', 'reduction_total_fraction',
    'low_speed_exempt', 'torque_cap_fraction', 'requested_dc_fraction',
    'deployment_tail_mask')


class CausalEnvV6:
    features = FEATURES
    def __init__(self, sim, action_space='signed', reward_config=None,
                 initial_speed_mps=None, carried_rule_state=None, output_ds_m=5.,
                 numerics=None,initial_energy_mj=2.):
        if action_space not in ('signed', 'mode_aware'):
            raise ValueError('Unknown v6 action space')
        self.sim, self.action_space, self.reward_config = sim, action_space, reward_config
        self.initial_speed_mps, self.carried_rule_state = initial_speed_mps, carried_rule_state
        self.initial_energy_mj=float(initial_energy_mj)
        if not np.isfinite(self.initial_energy_mj) or not 2.<=self.initial_energy_mj<=4.:
            raise ValueError('Initial energy must lie in the declared 2–4 MJ window')
        self.output_ds_m, self.config = float(output_ds_m), numerics or sim.config
        self.driver = sim.driver_template
        self.observation_dim = len(FEATURES)
        self.reset()

    def reset(self, seed=None, **kwargs):
        from .rules import RuleShield
        self.driver = PowerResponsiveDriver(self.sim.geometry, self.sim.vehicle)
        start = float(self.driver.safety_speed(0)) if self.initial_speed_mps is None else float(self.initial_speed_mps)
        if not np.isfinite(start) or start <= 0:
            raise ValueError('Initial speed must be finite and positive')
        carried = self.carried_rule_state
        previous = float(carried.get('previous_dc_kw',0.)) if isinstance(carried,dict) else float(getattr(carried,'previous_dc_kw',0.))
        self.state = IntegrationState(0., start, 0., self.initial_energy_mj, 0., previous)
        self.initial = self.state
        self.shield = RuleShield(self.sim.rules, self.sim.geometry,
            config=getattr(self.sim,'rule_config',None))
        self.shield.reset(self.state, self.carried_rule_state)
        self.driver.shield=self.shield
        self.i = 0
        self.done = not self.driver.envelope_report['converged']
        self.status = 'numerical_fail' if self.done else 'accepted'
        self.failure = {'guard_reasons': ['driver_envelope_not_converged']} if self.done else None
        self.records, self.events, self.history, self.states, self.observations = [], [], [], [self.state], []
        self.raw_actions, self.rewards = [], []
        return self.get_obs()

    def get_obs(self):
        state, p = self.state, self.sim.vehicle
        g = self.sim.geometry.evaluate(state.s_m)
        demand = 0. if self.done else self.driver.demand(state)
        total = p.mu*p.session_grip_scale*(p.mass_kg*9.81+
            .5*p.air_density_kgpm3*float(g['downforce_area_m2'])*state.speed_mps**2)
        lateral = p.mass_kg*state.speed_mps**2*abs(float(g['curvature_1pm']))
        features = self.shield.features
        if callable(features):
            features = features(state,{'wheel_force_n':demand,'brake_fraction':0.})
        features = features or {}
        def rule(name, default=0.): return float(features.get(name, default))
        phase = state.s_m/self.sim.length_m
        bounds = self.sim.sector_boundaries_m
        sector = int(np.searchsorted(bounds, state.s_m, side='right'))
        reserve = max(0., state.previous_dc_kw)**2 / 200000.
        drag = .5*p.air_density_kgpm3*float(g['drag_area_m2'])*state.speed_mps**2
        obs = np.array([phase, np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), state.speed_mps/100.,
            state.energy_mj/4., max(0., state.energy_mj-2.-reserve)/4., state.recharge_mj/8.,
            state.previous_dc_kw/350., float(g['curvature_1pm'])*100., lateral/max(total,1e-9),
            demand/12000., float(demand<0), float(self.driver.safety_speed(state.s_m))/100.,
            float(self.driver.safety_speed(state.s_m+100))/100.,
            float(self.driver.safety_speed(state.s_m+250))/100.,
            float(self.driver.safety_speed(state.s_m+500))/100., drag/12000., float(g.get('aero_mode',False)),
            float(sector==0),float(sector==1),float(sector==2),
            rule('deployment_cap_kw',350.)/350., rule('reserve_mj',reserve)/4.,
            rule('reduction_active'),rule('reduction_pending'),rule('hold_remaining_s'),
            rule('reduction_total_kw')/700.,rule('low_speed_exempt',float(state.speed_mps*3.6<210)),
            rule('torque_cap_kw',350.)/350.,rule('requested_dc_kw')/350.,
            rule('deployment_tail_mask')],float)
        if obs.shape != (len(FEATURES),) or not np.isfinite(obs).all():
            raise RuntimeError('Nonfinite v6 observation')
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError('No step after terminal state')
        values = np.asarray(action,float).reshape(-1)
        if values.size != 1:
            raise ValueError('Exactly one scalar action is required')
        raw = float(values[0])
        if not np.isfinite(raw):
            raise ValueError('Finite scalar action required')
        raw = float(np.clip(raw, -1., 1.))
        before = self.state
        obs = self.get_obs()
        command = self.driver.decide(before)
        decision = self.shield.decide(before, command, raw, action_space=self.action_space)
        stop = float(self.driver.decision_s[self.i+1])
        self.state, execution = advance_interval(before, self.driver, self.shield,
            stop, self.sim.geometry, self.sim.vehicle, self.config, collect_steps=True)
        records = execution.get('step_records',[])
        self.records.extend(records); self.events.extend(execution.get('events',[]))
        self.shield.accept(self.state, records)
        self.status = execution['status']
        self.done = self.status != 'accepted' or self.state.s_m >= self.sim.length_m-self.config.event_s_tol
        if self.status != 'accepted': self.failure = execution
        dt = self.state.elapsed_time_s-before.elapsed_time_s
        requested = float(decision.get('requested_dc_kw',getattr(self.shield,'request_kw',0.))) if isinstance(decision,dict) else float(getattr(self.shield,'request_kw',0.))
        dc = sum(r['dc_kw']*r['dt_s'] for r in records)/dt if dt>0 else 0.
        projection = abs(requested-dc)/350.
        from .reward import transition_reward
        reference = getattr(self.sim,'reference_dt_s',None)
        ref_dt = float(reference[self.i]) if reference is not None and self.i<len(reference) else 0.
        completed = self.state.s_m >= self.sim.length_m-self.config.event_s_tol
        reward, terms = transition_reward(before, self.state, ref_dt, self.status,
            self.done, completed, self.reward_config, projection)
        info = {'status':self.status,'step_certified':self.status=='accepted',
            'guard_reasons':execution.get('guard_reasons',[]),'dt_s':dt,
            'ds_m':self.state.s_m-before.s_m,'s_m':self.state.s_m,
            'speed_mps':self.state.speed_mps,'energy_mj':self.state.energy_mj,
            'recharge_mj':self.state.recharge_mj,'raw_action':raw,'requested_action':raw,
            'requested_dc_kw':requested,'dc_kw':dc,'projection_fraction':projection,
            'reward_terms':terms,'completed_lap':completed,'terminal':self.done,
            'actual_braking':command['wheel_force_n']<0,
            'wheel_demand_n':command['wheel_force_n'],
            'wheel_work_mj':sum(r['wheel_work_mj'] for r in records),
            'deploy_mj':sum(max(0.,r['dc_kw'])*r['dt_s']/1000 for r in records),
            'harvest_mj':sum(max(0.,-r['dc_kw'])*r['dt_s']/1000 for r in records)}
        self.history.append(info); self.states.append(self.state); self.observations.append(obs)
        self.raw_actions.append(raw); self.rewards.append(float(reward)); self.i += 1
        return self.get_obs(), float(reward), self.done, info

    def result(self):
        result = self.sim._result('autonomous',self.initial,self.state,self.records,
            self.events,self.failure,self.status,self.output_ds_m,self.config)
        certificate = self.shield.certificate
        if callable(certificate):
            try: certificate=certificate(self.state)
            except TypeError: certificate=certificate()
        certificate = dict(certificate or {})
        dc_work = sum(r['dc_kw']*r['dt_s']/1000 for r in self.records)
        energy_residual = self.state.energy_mj-self.initial.energy_mj+dc_work
        sustainable = self.state.energy_mj>=self.initial.energy_mj-1e-8
        speed_closed = abs(self.state.speed_mps-self.initial.speed_mps)<=.02
        dc_closed = abs(self.state.previous_dc_kw-self.initial.previous_dc_kw)<=.1
        public = bool(certificate.get('public_rules_passed',False))
        closed = bool(certificate.get('rule_state_closed',False))
        summary = result['summary']
        summary.update(protocol='power_responsive_hard_boundary_v6',
            sustainable=sustainable, public_rules_passed=public,
            periodic_speed_closed=speed_closed,dc_boundary_closed=dc_closed,
            rule_state_closed=closed,energy_balance_residual_mj=energy_residual,
            deploy_mj=sum(h['deploy_mj'] for h in self.history),
            harvest_mj=sum(h['harvest_mj'] for h in self.history),
            raw_action_space=self.action_space,observation_dim=len(FEATURES),
            initial_rule_state=self.carried_rule_state,rule_certificate=certificate,
            intermediate_speed_resets=0,whole_lap_scaling_used=False,
            frozen_vehicle_kernel=True,global_optimality_claim=False,
            full_fia_compliance_claim_supported=False,
            envelope_report=self.driver.envelope_report,
            terminal_dc_kw=self.state.previous_dc_kw,initial_dc_kw=self.initial.previous_dc_kw,
            terminal_rule_state=certificate.get('terminal_boundary_state'),
            deployment_tail_guard_m=getattr(self.shield.config,'tail_guard_m',350.))
        summary['feasible']=bool(summary['completed_lap'] and summary['execution_feasible'] and
            sustainable and speed_closed and dc_closed and closed and public and abs(energy_residual)<=1e-7)
        result.update(observations=np.asarray(self.observations,float).reshape(-1,len(FEATURES)),
            requested_actions=np.asarray(self.raw_actions,float),rewards=np.asarray(self.rewards,float),
            decision_info=self.history,rule_certificate=certificate,
            driver_profile={'s_m':np.asarray([r['s_m'] for r in self.driver.rows]),
                'wheel_force_n':np.asarray([r['wheel_force_n'] for r in self.driver.rows])},
            terminal_state=asdict(self.state))
        result['internal_trace'].update(dc_kw=np.array([r['dc_kw'] for r in self.records]),
            energy_mj=np.array([self.initial.energy_mj]+[r['energy_next_mj'] for r in self.records]),
            recharge_mj=np.array([self.initial.recharge_mj]+[r['recharge_next_mj'] for r in self.records]),
            ice_mechanical_kw=np.array([r['ice_mechanical_kw'] for r in self.records]),
            mguk_mechanical_kw=np.array([r['mguk_mechanical_kw'] for r in self.records]),
            friction_brake_kw=np.array([r['friction_brake_kw'] for r in self.records]),
            dt_s=np.array([r['dt_s'] for r in self.records]))
        return result
