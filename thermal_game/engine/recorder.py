# -*- coding: utf-8 -*-
import pandas as pd
from .state import GameState, Action, StepResult

COLUMNS = [
    # timing
    "timestamp", "t",
    # state
    "T_inside", "T_outside", "soc", "kwh_used",
    # actions
    "hvac", "battery", "pv_on",
    # prices & context
    "price", "occupancy", "comfort_penalty",
    # metered/power
    "pv_kw", "battery_kw", "hvac_kw", "other_kw", "total_kw",
    # energy & extras
    "electricity", "solar",
    # --- NEW ---
    "opex_cost", "reward",
]

class GameRecorder:
    """
    Keeps a tidy pandas.DataFrame of the whole run.
    Columns are predeclared; missing values are fine and will be filled as features land.
    """
    def __init__(self):
        self.df = pd.DataFrame(columns=COLUMNS)

    def append(self, prev_state: GameState, action: Action, step: StepResult, extra: dict | None = None):
        m = dict(step.metrics)
        if extra:
            m.update(extra)

        def _get(name, default=None):
            # prefer new state's attribute; otherwise metrics; otherwise default
            if hasattr(step.state, name):
                return getattr(step.state, name)
            return m.get(name, default)

        # signed convention (as used by charts):
        #   pv_kw: + generation
        #   battery_kw: + discharge to house, - charge from grid
        #   hvac_kw: positive draw (charts negate internally)
        #   other_kw: negative = load
        pv_kw     = m.get("pv_kw")
        batt_kw   = m.get("battery_kw")
        hvac_kw   = m.get("hvac_kw")
        other_kw  = m.get("other_kw")

        # derive total_kw if not provided
        total_kw = m.get("total_kw")
        if total_kw is None:
            try:
                # charts compute TOTAL = pv + batt + (-hvac_kw) + other_kw
                total_kw = float(pv_kw or 0.0) \
                           + float(batt_kw or 0.0) \
                           - float(hvac_kw or 0.0) \
                           + float(other_kw or 0.0)
            except Exception:
                total_kw = None

        row = {
            "timestamp": m.get("timestamp"),        # datetime if provided by GUI
            "t": step.state.t,                      # seconds since start
            "T_inside": _get("T_inside"),
            "T_outside": _get("T_outside"),
            "soc": _get("soc"),
            "kwh_used": _get("kwh_used"),
            "hvac": getattr(action, "hvac", None),
            "battery": getattr(action, "battery", None),
            "pv_on": m.get("pv_on"),
            "price": m.get("price"),
            "occupancy": m.get("occupancy"),
            "comfort_penalty": m.get("comfort_penalty"),
            "pv_kw": pv_kw,
            "battery_kw": batt_kw,
            "hvac_kw": hvac_kw,
            "other_kw": other_kw,
            "total_kw": total_kw,
            "electricity": m.get("electricity"),
            "solar": m.get("solar"),
        }

        self.df.loc[len(self.df)] = row

    def export_csv(self, path: str = "run.csv"):
        self.df.to_csv(path, index=False)
        return path
