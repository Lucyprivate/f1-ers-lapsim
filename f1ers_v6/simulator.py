"""Causal v6 policy execution reusing the frozen v4 time-domain kernel."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from f1ers_v4.simulator import SimulatorV4
from f1ers_v4.vehicle import VehicleParamsV4, IntegratorConfig
from .driver import PowerResponsiveDriver
from .environment import CausalEnvV6, FEATURES


class SimulatorV6(SimulatorV4):
    protocol='power_responsive_hard_boundary_v6'
    observation_dim=len(FEATURES)
    obs_dim=len(FEATURES)
    features=FEATURES

    def __init__(self,track,rules,vehicle=None,config=None,rule_config=None):
        if vehicle is None:
            vehicle=VehicleParamsV4(mu=1.8,drag_area_m2=1.3,downforce_area_m2=3.9,
                equivalent_power_kw=541.25,replay_brake_gain=.55)
        super().__init__(track,rules,vehicle,config)
        self.rule_config=rule_config
        self.driver_template=PowerResponsiveDriver(self.geometry,self.vehicle)
        official=float(rules.get('fia_centreline_length_m',4326.))
        sectors=np.asarray(rules.get('fia_sector_lengths_m',[1215.,1697.,1414.]),float)
        route=self.geometry.route
        self.sector_boundaries_m=np.asarray(route.legacy_to_s(np.cumsum(sectors)[:2]*route.source_length/official),float)

    def create_env(self,action_space='signed',reward_config=None,initial_speed_mps=None,
                   carried_rule_state=None,output_ds_m=5.,numerics=None,initial_energy_mj=2.):
        if isinstance(numerics,dict): numerics=IntegratorConfig(**numerics)
        return CausalEnvV6(self,action_space,reward_config,initial_speed_mps,
            carried_rule_state,output_ds_m,numerics,initial_energy_mj)

    def simulate(self,controller=None,initial_speed_mps=None,action_space='signed',
                 reward_config=None,carried_rule_state=None,output_ds_m=5.,
                 numerics=None,collect_steps=True,mode='autonomous',initial_energy_mj=2.,
                 periodic_shooting=False,**kwargs):
        if mode!='autonomous':
            raise ValueError('v6 policy simulator only accepts autonomous mode')
        if kwargs:
            raise TypeError('Unknown v6 execution options: '+', '.join(kwargs))
        if periodic_shooting:
            return self.periodic_simulate(controller,action_space,initial_speed_mps,
                reward_config=reward_config,carried_rule_state=carried_rule_state,
                output_ds_m=output_ds_m,numerics=numerics,initial_energy_mj=initial_energy_mj)
        env=self.create_env(action_space,reward_config,initial_speed_mps,
            carried_rule_state,output_ds_m,numerics,initial_energy_mj)
        actions=controller
        if isinstance(controller,dict):actions=controller.get('actions',controller.get('controller'))
        if actions is not None and not callable(actions):
            actions=np.asarray(actions,float)
            if actions.shape!=(len(env.driver.decision_s)-1,) or not np.isfinite(actions).all():
                raise ValueError('One finite action per fixed 5 m decision is required')
        while not env.done:
            action=actions(env.get_obs()) if callable(actions) else (float(actions[env.i]) if actions is not None else
                (-1. if action_space=='mode_aware' else 0.))
            env.step(action)
        return env.result()

    def periodic_simulate(self,controller=None,action_space='signed',initial_speed_mps=None,
                          max_cycles=12,tolerance_mps=1e-4,**kwargs):
        """Policy-specific shooting. Every trial is a full open-loop integration."""
        guess=float(self.driver_template.safety_speed(0)) if initial_speed_mps is None else float(initial_speed_mps)
        residuals=[];result=None
        for cycle in range(max_cycles):
            result=self.simulate(controller,guess,action_space=action_space,**kwargs)
            if not result['summary']['completed_lap'] or not result['summary']['execution_feasible']:
                break
            end=result['summary']['speed_end_mps']; residual=float(end-guess)
            residuals.append(residual)
            if abs(residual)<=tolerance_mps: break
            guess=float(end)
        result['summary']['shooting']={'method':'policy_specific_periodic_fixed_point',
            'full_lap_trials':cycle+1,'residuals_mps':residuals,
            'converged':bool(residuals and abs(residuals[-1])<=tolerance_mps),
            'tolerance_mps':tolerance_mps,'maximum_cycles':max_cycles,
            'within_lap_velocity_resets':0}
        return result

    def observation(self,state,env=None):
        if env is not None:return env.get_obs()
        probe=self.create_env(initial_speed_mps=state.speed_mps)
        probe.state=state
        return probe.get_obs()

    get_obs=observation

    def simulate_laps(self,controller=None,n_laps=3,initial_speed_mps=None,
                      initial_energy_mj=2.,carried_rule_state=None,**kwargs):
        """Cross the physical control line carrying speed, energy and ECU state.

        Only the regulated per-lap recharge counter restarts. This method never
        invokes shooting between successive physical laps.
        """
        if n_laps<1:raise ValueError('At least one physical lap is required')
        if kwargs.get('periodic_shooting',False):
            raise ValueError('Physical consecutive laps cannot shoot a new boundary between laps')
        laps=[];speed=initial_speed_mps;energy=float(initial_energy_mj);carry=carried_rule_state
        for i in range(int(n_laps)):
            result=self.simulate(controller,speed,initial_energy_mj=energy,
                carried_rule_state=carry,**kwargs)
            result['summary']['physical_lap_index']=i
            laps.append(result)
            if not result['summary']['completed_lap'] or not result['summary']['execution_feasible']:break
            speed=result['terminal_speed_mps'];energy=result['summary']['energy_end_mj']
            carry=result['rule_certificate']['terminal_boundary_state']
        crossing=[]
        for a,b in zip(laps,laps[1:]):
            crossing.append({'speed_difference_mps':b['summary']['speed_start_mps']-a['summary']['speed_end_mps'],
                'energy_difference_mj':b['summary']['energy_start_mj']-a['summary']['energy_end_mj'],
                'dc_difference_kw':b['summary']['initial_dc_kw']-a['summary']['terminal_dc_kw'],
                'rule_state_carried':b['summary']['initial_rule_state']==a['rule_certificate']['terminal_boundary_state']})
        return {'laps':laps,'crossings':crossing,'physical_control_line_continuity':
            bool(len(laps)==n_laps and all(abs(c['speed_difference_mps'])<1e-10 and
                abs(c['energy_difference_mj'])<1e-10 and abs(c['dc_difference_kw'])<1e-10 and
                c['rule_state_carried'] for c in crossing)),
            'within_lap_velocity_resets':0,'within_lap_energy_resets':0,
            'lap_recharge_counter_reset_only':True}
