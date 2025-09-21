from .state import GameState, Action, StepResult
from .physics import hvac_power_kw, thermal_step, battery_step

class SimulationEngine:
    def __init__(self, dt: float = 1.0):
        self.dt = dt

    def step(self, s: GameState, a: Action) -> StepResult:
        hvac = max(-1.0, min(1.0, a.hvac))
        hvac_kw = abs(hvac_power_kw(hvac))
        # energy use: HVAC + (optional) battery charge/discharge efficiency
        dkwh = hvac_kw * (self.dt/3600.0)

        next_Ti = thermal_step(s.T_inside, s.T_outside, hvac_power_kw(hvac), self.dt)
        next_soc = battery_step(s.soc, a.battery, self.dt)

        ns = GameState(
            t=s.t + self.dt,
            T_inside=next_Ti,
            T_outside=s.T_outside,
            soc=next_soc,
            kwh_used=s.kwh_used + dkwh,
        )
        metrics = {
            "electricity": dkwh,
            "T_inside": ns.T_inside,
            "soc": ns.soc,
        }
        return StepResult(state=ns, metrics=metrics)
