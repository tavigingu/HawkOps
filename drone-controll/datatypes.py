from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

class MissionStatus(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class DroneState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"

@dataclass
class Position:
    x: float
    y: float
    z: float
    yaw: float = 0.0

@dataclass
class MissionResult:
    success: bool
    message: str
    data: Optional[Dict] = None
    execution_time: float = 0.0

@dataclass
class DroneStatus:
    state: DroneState
    position: Position
    battery_level: float = 0.0
    is_flying: bool = False
    altitude: float = 0.0

@dataclass
class MissionConfig:
    name: str
    description: str
    parameters: Dict = None
    max_altitude: float = 10.0
    timeout: float = 300.0 