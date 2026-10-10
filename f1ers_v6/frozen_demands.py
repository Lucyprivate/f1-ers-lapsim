"""Hold original wheel demands and actions fixed for integrator comparisons."""
from __future__ import annotations
import numpy as np
from .driver import PowerResponsiveDriver
from .simulator import SimulatorV6


class FrozenDemandDriver(PowerResponsiveDriver):
    def __init__(self,geometry,params,profile):
        super().__init__(geometry,params)
        self.frozen_s=np.asarray(profile['s_m'],float)
        self.frozen_force=np.asarray(profile['wheel_force_n'],float)
        if len(self.frozen_s)!=len(self.decision_s)-1 or not np.isfinite(self.frozen_force).all():
            raise ValueError('Original full 5m wheel demand profile required')
    def demand(self,state):
        return float(self.frozen_force[self.decision_index(state.s_m)])


def freeze_simulator(sim,profile):
    class FrozenSimulator(SimulatorV6):
        def create_env(self,*args,**kwargs):
            env=super().create_env(*args,**kwargs)
            env.driver=FrozenDemandDriver(self.geometry,self.vehicle,profile)
            # Forward rule hold events to the frozen integrator.
            env.driver.shield=env.shield
            return env
    return FrozenSimulator(sim.track,sim.rules,sim.vehicle,sim.config,sim.rule_config)
