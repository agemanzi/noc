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
    t: float
    T_inside: float
    T_outside: float
    soc: float
    kwh_used: float
    ts: Optional[dt.datetime] = None
    occupied: Optional[int] = None

    # NEW: cumulative scores (start at 0.0)
    cumulative_reward: float = 0.0
    cumulative_financial: float = 0.0
    cumulative_comfort: float = 0.0

@dataclass
class StepResult:
    state: GameState
    metrics: Dict[str, Any]   # allow non-floats (e.g., arrays, dicts) if needed
