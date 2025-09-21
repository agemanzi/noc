import pandas as pd
from .state import GameState, Action, StepResult

class GameRecorder:
    def __init__(self):
        self.df = pd.DataFrame(columns=[
            "t","T_inside","T_outside","soc","kwh_used","hvac","battery",
            "electricity"
        ])

    def append(self, s: GameState, a: Action, r: StepResult):
        row = dict(
            t=r.state.t, T_inside=r.state.T_inside, T_outside=r.state.T_outside,
            soc=r.state.soc, kwh_used=r.state.kwh_used,
            hvac=a.hvac, battery=a.battery, electricity=r.metrics["electricity"]
        )
        self.df.loc[len(self.df)] = row

    def export_csv(self, path: str):
        self.df.to_csv(path, index=False)
