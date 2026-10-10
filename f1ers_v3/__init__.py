"""Stage-three public-data equivalent-vehicle experiments.

Legacy f1ers is a byte-preserved reference dependency.  v3 public APIs live in
this package and use a separate execution/checkpoint protocol.
"""

__version__ = "0.3.0"

from .vehicle import VehicleParamsV3, VehicleState, WheelCommand, SegmentGeometry, vehicle_step
from .simulator import SimulatorV3

__all__ = ["VehicleParamsV3", "VehicleState", "WheelCommand", "SegmentGeometry",
           "vehicle_step", "SimulatorV3"]

