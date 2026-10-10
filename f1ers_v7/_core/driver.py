"""Geometry safety without an ICE-only velocity tracking target.

The precomputed closed route is known to a driver; observed future velocity is
never available.  Positive demand is a geometric ceiling, not an ICE-only
acceleration reference. Actual acceleration follows ICE and executed K power.
"""
from __future__ import annotations

import numpy as np
from f1ers_v7._core.envelope import _driver_envelope
from f1ers_v7._core.base_simulator import DRIVER_DS_M, DRIVER_GRIP_MARGIN


class PowerResponsiveDriver:
    def __init__(self, geometry, params):
        self.geometry, self.params = geometry, params
        self.length_m = float(geometry.length_m)
        self.decision_s = np.linspace(0., self.length_m,
            int(np.ceil(self.length_m / DRIVER_DS_M)) + 1)
        ns = int(np.ceil(self.length_m))
        s = np.arange(ns) * self.length_m / ns
        g = geometry.evaluate(s)
        mu = params.mu * params.session_grip_scale * DRIVER_GRIP_MARGIN
        denominator = (params.mass_kg * np.abs(g['curvature_1pm']) -
            mu * .5 * params.air_density_kgpm3 * g['downforce_area_m2'])
        cap2 = np.divide(mu * params.mass_kg * 9.81, denominator,
            out=np.full(ns, params.max_speed_mps ** 2), where=denominator > 1e-9)
        caps = np.minimum(params.max_speed_mps, np.sqrt(np.maximum(.25, cap2)))
        # Infinite propulsion deliberately removes the forward power ceiling.
        # This is only a braking/curvature safety upper bound. The v4 kernel
        # always applies actual finite ICE and K supply and actual tyre limits.
        safe, residual, iterations, ok = _driver_envelope(caps,
            g['curvature_1pm'], g['drag_area_m2'], g['downforce_area_m2'],
            self.length_m / ns, params.mass_kg, mu,
            params.air_density_kgpm3, params.rolling_coefficient, 1., 1e7, 1e9)
        self.safe_s = np.r_[s, self.length_m]
        self.safe_speed = np.r_[safe, safe[0]]
        self.envelope_report = {'method': 'geometry_braking_only',
            'converged': bool(ok), 'residual_mps': float(residual),
            'iterations': int(iterations), 'fixed_grip_margin': DRIVER_GRIP_MARGIN,
            'finite_power_execution': True, 'observed_future_speed_used': False}
        self.command = None
        self.rows = []

    def safety_speed(self, s):
        return np.interp(np.asarray(s) % self.length_m, self.safe_s, self.safe_speed)

    def decision_index(self, s):
        return min(len(self.decision_s)-2, max(0,
            int(np.searchsorted(self.decision_s, s+1e-5, side='right')-1)))

    def demand(self, state):
        i = self.decision_index(state.s_m)
        ds = max(1e-8, float(self.decision_s[i+1]-state.s_m))
        next_safe = float(self.safety_speed(self.decision_s[i+1]))
        g = self.geometry.evaluate(state.s_m)
        drag = (.5 * self.params.air_density_kgpm3 * float(g['drag_area_m2']) *
            state.speed_mps**2 + self.params.rolling_coefficient *
            self.params.mass_kg * 9.81)
        requested = self.params.mass_kg * (next_safe**2-state.speed_mps**2)/(2*ds) + drag
        return float(min(self.params.ice_max_tractive_force_n, requested))

    def decide(self, state):
        self.command = {'mode': 'autonomous', 'power_kw': self.params.ice_power_kw,
            'brake_fraction': 0., 'max_tractive_force_n': self.params.ice_max_tractive_force_n,
            'wheel_force_n': self.demand(state), 'target_acceleration_mps2': None}
        self.rows.append({'s_m': float(state.s_m), **self.command})
        return dict(self.command)

    def query(self, t, s):
        return dict(self.command)

    def next_time_event(self, t):
        shield=getattr(self,'shield',None)
        event=shield.next_time_event(t) if shield is not None else np.inf
        return float(event) if event-float(t)>1e-8 else np.inf

    def next_space_event(self, s):
        i = np.searchsorted(self.decision_s, s+1e-5, side='right')
        decision = float(self.decision_s[i]) if i < len(self.decision_s) else np.inf
        return min(decision, self.geometry.next_space_event(s))
