"""Vehicle parameters and optional Numba decorator; extracted unchanged."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function


@dataclass
class VehicleParams:
    mass_kg: float = 800.0
    ice_power_kw: float = 400.0
    ice_max_tractive_force_n: float = 12000.0
    mu: float = 1.75
    drag_area_m2: float = 1.15
    downforce_area_m2: float = 3.6
    straight_drag_factor: float = 0.72
    straight_downforce_factor: float = 0.65
    air_density_kgpm3: float = 1.18
    rolling_coefficient: float = 0.012
    deploy_efficiency: float = 0.95
    harvest_efficiency: float = 0.90
    energy_start_mj: float = 2.0
    max_speed_mps: float = 103.0
    superclip_max_dc_kw: float = 350.0
    minimum_ice_wheel_power_kw: float = 25.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "VehicleParams":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in allowed})
