# -*- coding: utf-8 -*-
import pandas as pd
from .state import GameState, Action, StepResult

# Base columns we expect; the DataFrame will auto-extend if we pass more keys.
COLUMNS = [
    # timing
    "timestamp", "t",
    # state
    "T_inside", "T_outside", "soc", "kwh_used",
    # actions
    "hvac", "battery", "pv_on",
    # prices & context
    "price", "occupancy", "comfort_penalty",
    # metered/power (chart conventions)
    "pv_kw", "battery_kw", "hvac_kw", "other_kw", "total_kw",
    # energy & extras
    "electricity", "solar",

    # --- economics / reward split ---
    "import_kwh", "export_kwh",
    "import_cost", "export_credit", "net_opex", "opex_cost",  # opex_cost kept as alias
    "reward_fin", "reward_comf", "reward",                    # total reward in 'reward'
]

class GameRecorder:
    """
    Keeps a tidy pandas.DataFrame of the whole run.
    Columns are predeclared; missing values are fine and will be filled as features land.
    If additional keys are passed, the DataFrame will auto-extend to include them.
    """
    def __init__(self):
        self.df = pd.DataFrame(columns=COLUMNS)

    def _ensure_columns(self, keys):
        """Add any missing columns so we can safely assign the row."""
        missing = [k for k in keys if k not in self.df.columns]
        if missing:
            for k in missing:
                self.df[k] = pd.Series(dtype="float64") if k not in ("timestamp", "pv_on") else pd.Series(dtype="object")

    def append(self, prev_state: GameState, action: Action, step: StepResult, extra: dict | None = None):
        m = dict(step.metrics)
        if extra:
            m.update(extra)

        def _from_state(name, default=None):
            return getattr(step.state, name, default)

        # Prefer GUI-provided timestamp; otherwise use state's ts if present
        timestamp = m.get("timestamp", _from_state("ts", None))
        occupancy = m.get("occupancy", _from_state("occupied", None))

        # Signed convention (as used by charts):
        #   pv_kw: + generation
        #   battery_kw: + discharge to house, - charge from grid
        #   hvac_kw: positive draw (charts negate internally)
        #   other_kw: negative = load
        pv_kw     = m.get("pv_kw")
        batt_kw   = m.get("battery_kw")
        hvac_kw   = m.get("hvac_kw")
        other_kw  = m.get("other_kw")

        # Derive total_kw if not provided
        total_kw = m.get("total_kw")
        if total_kw is None:
            try:
                total_kw = float(pv_kw or 0.0) \
                           + float(batt_kw or 0.0) \
                           - float(hvac_kw or 0.0) \
                           + float(other_kw or 0.0)
            except Exception:
                total_kw = None

        # Economics & reward (all may or may not be present on early frames)
        import_kwh    = m.get("import_kwh")
        export_kwh    = m.get("export_kwh")
        import_cost   = m.get("import_cost")
        export_credit = m.get("export_credit")
        net_opex      = m.get("net_opex")
        # keep legacy alias too
        opex_cost     = m.get("opex_cost", net_opex)

        reward_fin  = m.get("reward_fin")
        reward_comf = m.get("reward_comf")
        reward_tot  = m.get("reward")  # total reward is stored under 'reward'

        row = {
            # timing
            "timestamp": timestamp,
            "t": step.state.t,

            # state
            "T_inside": getattr(step.state, "T_inside", m.get("T_inside")),
            "T_outside": getattr(step.state, "T_outside", m.get("T_outside")),
            "soc": getattr(step.state, "soc", m.get("soc")),
            "kwh_used": getattr(step.state, "kwh_used", m.get("kwh_used")),

            # actions
            "hvac": getattr(action, "hvac", None),
            "battery": getattr(action, "battery", None),
            "pv_on": m.get("pv_on"),

            # prices & context
            "price": m.get("price"),
            "occupancy": occupancy,
            "comfort_penalty": m.get("comfort_penalty"),

            # metered/power
            "pv_kw": pv_kw,
            "battery_kw": batt_kw,
            "hvac_kw": hvac_kw,
            "other_kw": other_kw,
            "total_kw": total_kw,

            # energy & extras
            "electricity": m.get("electricity"),
            "solar": m.get("solar"),

            # economics / reward split
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "import_cost": import_cost,
            "export_credit": export_credit,
            "net_opex": net_opex,
            "opex_cost": opex_cost,   # alias to net_opex for back-compat

            # rewards
            "reward_fin": reward_fin,
            "reward_comf": reward_comf,
            "reward": reward_tot,     # total
        }

        # Allow forwards-compatible additions: make sure df has any new keys.
        self._ensure_columns(row.keys())
        self.df.loc[len(self.df)] = row

    def export_csv(self, path: str = "run.csv"):
        self.df.to_csv(path, index=False)
        return path
