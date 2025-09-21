# -*- coding: utf-8 -*-
import math
import pandas as pd
from .state import GameState, Action, StepResult

COLUMNS = [
    # time/state
    "t","T_inside","T_outside","soc","kwh_used",
    # actions
    "hvac","battery","pv_on",
    # engine metrics
    "electricity","hvac_kw","battery_kw","pv_kw",
    # placeholders / analytics
    "price","occupancy","comfort_penalty"
]

class GameRecorder:
    """
    Keeps a tidy pandas.DataFrame of the whole run.
    Missing values are allowed and will be filled as features come online.
    """
    def __init__(self):
        self.df = pd.DataFrame(columns=COLUMNS)

    def append(self, prev_state: GameState, action: Action, step: StepResult, extra: dict | None = None):
        m = dict(step.metrics)
        if extra:
            m.update(extra)

        def _get(name, default=None):
            # pull from new state, metrics, or default
            if hasattr(step.state, name):
                return getattr(step.state, name)
            return m.get(name, default)

        row = {
            "t": step.state.t,
            "T_inside": _get("T_inside"),
            "T_outside": _get("T_outside"),
            "soc": _get("soc"),
            "kwh_used": _get("kwh_used"),
            "hvac": getattr(action, "hvac", None),
            "battery": getattr(action, "battery", None),
            "pv_on": m.get("pv_on"),
            "electricity": m.get("electricity"),
            "hvac_kw": m.get("hvac_kw"),
            "battery_kw": m.get("battery_kw"),
            "pv_kw": m.get("pv_kw"),
            "price": m.get("price"),
            "occupancy": m.get("occupancy"),
            "comfort_penalty": m.get("comfort_penalty"),
        }
        self.df.loc[len(self.df)] = row

    def export_csv(self, path: str = "run.csv"):
        self.df.to_csv(path, index=False)
        return path
