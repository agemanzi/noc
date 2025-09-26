# thermal_game/rl/rl_sac_replayer.py
from __future__ import annotations

# --- path bootstrap (run directly, no install needed) ---
import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]          # .../noc
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# --------------------------------------------------------

import os
import datetime as dt
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

# Engine deps
from thermal_game.engine.simulation import SimulationEngine
from thermal_game.engine.state import GameState, Action
from thermal_game.engine.settings import GameSettings
from thermal_game.engine.datafeed import DataFeed
from thermal_game.engine.reward import RewardConfig

# =========================
#         CONFIG
# =========================
PKG_DIR   = REPO_ROOT / "thermal_game"
DATA_DIR  = PKG_DIR / "data"   # e.g. C:\_projekty_repo\noc\thermal_game\data
MODEL_DIR = REPO_ROOT / "models"

WEATHER_CSV_NAME = "_2ndweekXX_prices_weather_seasons_FROM_2023_RELABELED_TO_2025.csv"  # Match runner
LOAD_CSV_NAME    = "load_profile.csv"

# We will attempt, in order:
# 1) models\sac_thermal_game.zip
# 2) models\sac_thermal_game (folder)  -> fallback to downloads zip
# 3) C:\Users\kollmann.marek\Downloads\sac_thermal_game.zip
MODEL_BASENAME   = "sac_thermal_game"
FALLBACK_ZIP     = Path(r"C:\Users\kollmann.marek\Downloads\sac_thermal_game.zip")

# Episode setup (match training)
START_DATE        = dt.date(2025, 3, 1)
STEPS_PER_EPISODE = 4 * 24 * 7              # 1 week of 15-min steps
PV_ON             = True
ALLOW_GRID_CHARGE = True
SEED              = 0

# Output
REPLAYS_DIR = REPO_ROOT / "outputs" / "replays"
REPLAY_CSV  = REPLAYS_DIR / "ghost_run.csv"


# =========================
#     Minimal Gym Env
# =========================
class ThermalGameEnv(gym.Env):
    """
    action: [hvac in [-1,1], battery in [-1,1] -> {-1,0,1} via deadband]
    obs:    [T_in, T_out, soc, price, pv_kw, base_kw, sin_hour, cos_hour]
    reward: engine step.metrics["reward"]
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        weather_csv: str | Path,
        load_csv: str | Path,
        *,
        start_date: dt.date = dt.date(2025, 1, 1),
        steps_per_episode: int = 4 * 24 * 7,
        pv_on: bool = True,
        allow_grid_charge: bool = True,
        reward_cfg: RewardConfig | None = None,
        settings: GameSettings | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.rng = np.random.RandomState(seed or 0)

        self.settings = settings or GameSettings()
        self.settings.start_date = start_date
        self.engine = SimulationEngine(dt=900, player_controls_hvac=True)
        self.engine.allow_grid_charge = allow_grid_charge
        self.reward_cfg = reward_cfg or RewardConfig(
            comfort_target_C=self.settings.comfort_target_C,
            comfort_tolerance_occupied_C=getattr(self.settings, "comfort_tolerance_occupied_C", 0.5),
            comfort_tolerance_unoccupied_C=getattr(self.settings, "comfort_tolerance_unoccupied_C", 1.0),
            comfort_weight=self.settings.comfort_anchor_eur_per_deg2_hour * self.engine.dt_h,
            export_tariff_ratio=self.settings.export_tariff_ratio,
            **({"comfort_inside_bonus": getattr(self.settings, "comfort_inside_bonus_eur_per_step", 0.0)}
               if "comfort_inside_bonus" in getattr(RewardConfig, "__dataclass_fields__", {}) else {})
        )

        self.feed = DataFeed(str(weather_csv), str(load_csv))
        self.feed.set_anchor_date(self.settings.start_date)

        self.pv_on = pv_on
        self.steps_per_episode = int(steps_per_episode)

        # Engine persistence state tracking for realism features
        self._prev_hvac_kw = 0.0
        self._prev_Tin_measured = None
        self._solar_pop_remaining = 0
        self._window_event_remaining = 0
        self._step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        high = np.array([60.0, 60.0, 1.0, 2.0, 100.0, 20.0, 1.0, 1.0], dtype=np.float32)
        low  = np.array([-40.0,-40.0, 0.0, 0.0,   0.0,  0.0,-1.0,-1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state: GameState | None = None
        self._steps = 0

    def _obs(self, row, _m) -> np.ndarray:
        t = int(self.state.t)
        minutes = (t // 60) % (24 * 60)
        ang = 2 * np.pi * (minutes / (24 * 60))
        sin_hour, cos_hour = np.sin(ang), np.cos(ang)

        T_out = float(row.t_out_c)
        price = float(row.price_eur_per_kwh)
        pv_kwp = float(row.solar_gen_kw_per_kwp)
        pv_kw = pv_kwp * float(self.settings.pv_size_kw) if self.pv_on else 0.0
        base_kw = float(row.base_load_kw)
        soc = float(self.state.soc)
        
        # Use measured temperature (with sensor lag) if available, else true temperature
        T_in = float(self._prev_Tin_measured if self._prev_Tin_measured is not None
                     else self.state.T_inside)

        return np.array([T_in, T_out, soc, price, pv_kw, base_kw, sin_hour, cos_hour], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.state = GameState(
            t=0.0,
            T_inside=22.0 + self.rng.randn()*0.5,
            T_outside=15.0,
            soc=float(np.clip(0.5 + 0.1*self.rng.randn(), 0.1, 0.9)),
            kwh_used=0.0,
            T_envelope=22.0  # Add envelope temperature for R3C2 model
        )
        self.feed.set_anchor_date(self.settings.start_date)
        self._steps = 0
        
        # Reset engine persistence states
        self._prev_hvac_kw = 0.0
        self._prev_Tin_measured = None  # will be set after first step based on Tin
        self._solar_pop_remaining = 0
        self._window_event_remaining = 0
        self._step_count = 0
        
        row = self.feed.by_time(self.state.t)
        obs = self._obs(row, None)
        return obs, {}

    def step(self, action: np.ndarray):
        self._steps += 1
        self._step_count += 1
        
        a = np.asarray(action, dtype=np.float32)
        hvac = float(np.clip(a[0], -1.0, 1.0))
        b_raw = float(np.clip(a[1], -1.0, 1.0))
        battery_cmd = 1 if b_raw > 0.33 else (-1 if b_raw < -0.33 else 0)

        row = self.feed.by_time(self.state.t)
        occupied = int(bool(row.occupied_home))
        price = float(row.price_eur_per_kwh)
        T_outside = float(row.t_out_c)
        pv_kwp_yield = float(row.solar_gen_kw_per_kwp)
        pv_kw = pv_kwp_yield * float(self.settings.pv_size_kw) if self.pv_on else 0.0
        base_load_kw = float(row.base_load_kw)

        # Recompute reward config with dynamic tolerance based on occupancy
        tol = (self.settings.comfort_tolerance_occupied_C if occupied
               else self.settings.comfort_tolerance_unoccupied_C)
        self.reward_cfg = RewardConfig(
            comfort_target_C=self.settings.comfort_target_C,
            comfort_tolerance_occupied_C=self.settings.comfort_tolerance_occupied_C,
            comfort_tolerance_unoccupied_C=self.settings.comfort_tolerance_unoccupied_C,
            comfort_weight=self.settings.comfort_anchor_eur_per_deg2_hour * self.engine.dt_h,
            export_tariff_ratio=self.settings.export_tariff_ratio,
            **({"comfort_inside_bonus": getattr(self.settings, "comfort_inside_bonus_eur_per_step", 0.0)}
               if "comfort_inside_bonus" in getattr(RewardConfig, "__dataclass_fields__", {}) else {})
        )

        # Enhanced engine inputs with persistence and realism hooks
        engine_inputs = {
            "ts": row.ts,
            "T_outside": T_outside,
            "price": price,
            "pv_kw": pv_kw,
            "base_load_kw": base_load_kw,
            "battery_cmd": battery_cmd,
            "settings": self.settings,
            "occupied_home": occupied,
            "q_internal_kw": 0.7 if occupied else 0.3,
            "q_solar_kw": 0.0,
            "reward_cfg": self.reward_cfg,

            # NEW: persistence & realism hooks
            "prev_hvac_kw": self._prev_hvac_kw,
            "prev_Tin_measured": (self._prev_Tin_measured
                                  if self._prev_Tin_measured is not None
                                  else float(self.state.T_inside)),
            "solar_pop_remaining": self._solar_pop_remaining,
            "window_event_remaining": self._window_event_remaining,
            "step_count": self._step_count,
        }
        
        step = self.engine.step(self.state, Action(hvac=hvac, battery=battery_cmd), engine_inputs)
        self.state = step.state
        m = step.metrics

        # Pull back persistence state for next step
        self._prev_hvac_kw = float(m.get("prev_hvac_kw", m.get("hvac_kw", 0.0)))
        self._prev_Tin_measured = float(m.get("Tin_measured", self.state.T_inside))
        self._solar_pop_remaining = int(m.get("solar_pop_remaining", 0))
        self._window_event_remaining = int(m.get("window_event_remaining", 0))

        reward = float(m.get("reward", 0.0))
        terminated = False
        truncated = self._steps >= self.steps_per_episode

        obs = self._obs(row, m)
        info = {
            "financial": float(m.get("reward_fin", 0.0)),
            "comfort": float(m.get("reward_comf", 0.0)),
            "import_kwh": float(m.get("import_kwh", 0.0)),
            "export_kwh": float(m.get("export_kwh", 0.0)),
            "soc": float(self.state.soc),
            "hvac_kw": float(m.get("hvac_kw", 0.0)),
            "battery_kw": float(m.get("battery_kw", 0.0)),
            "pv_kw": float(m.get("pv_kw", 0.0)),
            "other_kw": float(m.get("other_kw", -base_load_kw)),
        }
        return obs, reward, terminated, truncated, info


def _load_sac_model(model_dir: Path, base_name: str, fallback_zip: Path) -> SAC:
    """
    Try loading:
      1) model_dir / (base_name + '.zip')
      2) if model_dir / base_name is a directory -> try fallback_zip
      3) fallback_zip directly
    """
    zip_path = model_dir / (base_name + ".zip")
    folder_path = model_dir / base_name

    # 1) Preferred: zip sitting in models/
    if zip_path.exists():
        print(f">> loading model zip: {zip_path}")
        return SAC.load(str(zip_path))

    # 2) If they pass only a folder, SB3 can't load it; use fallback zip.
    if folder_path.exists():
        print(f">> found model folder (not loadable by SB3): {folder_path}")
        if fallback_zip.exists():
            print(f">> trying fallback zip: {fallback_zip}")
            return SAC.load(str(fallback_zip))
        raise FileNotFoundError(
            f"Found folder at {folder_path}, but no zip at {zip_path} "
            f"and fallback zip not found at {fallback_zip}."
        )

    # 3) Last chance: direct fallback
    if fallback_zip.exists():
        print(f">> loading fallback zip: {fallback_zip}")
        return SAC.load(str(fallback_zip))

    raise FileNotFoundError(
        f"No model found. Tried:\n  {zip_path}\n  {folder_path}\n  {fallback_zip}"
    )


# =========================
#         REPLAYER
# =========================
def main():
    weather_csv = DATA_DIR / WEATHER_CSV_NAME
    load_csv    = DATA_DIR / LOAD_CSV_NAME

    if not weather_csv.exists() or not load_csv.exists():
        raise FileNotFoundError(f"Missing data files:\n  {weather_csv}\n  {load_csv}")

    # Robust model loading as requested
    model = _load_sac_model(MODEL_DIR, MODEL_BASENAME, FALLBACK_ZIP)

    env = ThermalGameEnv(
        str(weather_csv), str(load_csv),
        start_date=START_DATE,
        steps_per_episode=STEPS_PER_EPISODE,
        pv_on=PV_ON,
        allow_grid_charge=ALLOW_GRID_CHARGE,
        seed=SEED,
    )

    obs, _ = env.reset()
    done = trunc = False
    ep_r = 0.0
    rows = []

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        # keep the exact action + mapped discrete command
        hvac = float(np.clip(action[0], -1.0, 1.0))
        braw = float(np.clip(action[1], -1.0, 1.0))
        batt_cmd = 1 if braw > 0.33 else (-1 if braw < -0.33 else 0)

        # exogenous used this step (before stepping)
        row = env.feed.by_time(env.state.t)
        T_out = float(row.t_out_c)
        price = float(row.price_eur_per_kwh)
        base_kw = float(row.base_load_kw)
        pv_kw = float(row.solar_gen_kw_per_kwp) * float(env.settings.pv_size_kw) if env.pv_on else 0.0

        obs, r, done, trunc, info = env.step(action)
        ep_r += float(r)

        # state after step
        st = env.state

        rows.append({
            # timing
            "t": float(st.t),
            "ts": getattr(st, "ts", None),

            # exogenous inputs
            "T_outside": T_out,
            "price": price,
            "pv_kw": pv_kw,
            "base_load_kw": base_kw,

            # actions
            "hvac": hvac,
            "battery_cmd": int(batt_cmd),
            "action_raw_0": float(action[0]),
            "action_raw_1": float(action[1]),

            # key outputs
            "T_inside": float(st.T_inside),
            "soc": float(st.soc),
            "reward": float(info.get("financial", 0.0) + info.get("comfort", 0.0)),
            "reward_fin": float(info.get("financial", 0.0)),
            "reward_comf": float(info.get("comfort", 0.0)),
            "import_kwh": float(info.get("import_kwh", 0.0)),
            "export_kwh": float(info.get("export_kwh", 0.0)),
            "hvac_kw": float(info.get("hvac_kw", 0.0)),
            "battery_kw": float(info.get("battery_kw", 0.0)),
            "pv_kw_seen": float(info.get("pv_kw", pv_kw)),
            "other_kw": float(info.get("other_kw", -base_kw)),
        })

    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPLAY_CSV, index=False)
    print(f">> episode reward: {ep_r:.3f}")
    print(f">> saved replay CSV → {REPLAY_CSV}")


if __name__ == "__main__":
    main()
