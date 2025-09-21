from dataclasses import dataclass
from typing import Literal, Dict

BatteryAct = Literal[-1, 0, 1]  # discharge, idle, charge

@dataclass
class Action:
    hvac: float          # -1..1 (cool..heat); clamp in engine
    battery: BatteryAct  # -1,0,1

@dataclass
class GameState:
    t: float               # sim time (s)
    T_inside: float        # °C
    T_outside: float       # °C (exogenous; could come from scenario)
    soc: float             # 0..1 state of charge
    kwh_used: float        # cumulative electricity

@dataclass
class StepResult:
    state: GameState
    metrics: Dict[str, float]  # e.g. {"electricity": dkwh, "comfort_penalty": x}
