# thermal_game/engine/state.py
from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional
import datetime as dt

BatteryAct = Literal[-1, 0, 1]  # discharge, idle, charge

@dataclass
class Action:
    # -1..1: slider biases the setpoint (cool..heat). The engine clamps.
    hvac: float
    battery: BatteryAct

@dataclass
class GameState:
    t: float                  # sim time (s)
    T_inside: float           # °C
    T_outside: float          # °C (copied from inputs each step)
    soc: float                # 0..1
    kwh_used: float           # cumulative grid import (kWh)
    ts: Optional[dt.datetime] = None  # OPTIONAL: wall-clock timestamp from feed
    occupied: Optional[int] = None    # OPTIONAL: 1/0 occupancy from feed

@dataclass
class StepResult:
    state: GameState
    metrics: Dict[str, Any]   # allow non-floats (e.g., arrays, dicts) if needed
