# rl_sac_runner.py
from __future__ import annotations

# --- run-from-file friendly path bootstrap (no install needed) ---
import sys
from pathlib import Path
_THIS_FILE = Path(__file__).resolve()
# repo root: .../noc
REPO_ROOT = _THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ---------------------------------------------------------------

import os
import datetime as dt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Engine deps (available under thermal_game/)
from thermal_game.engine.simulation import SimulationEngine
from thermal_game.engine.state import GameState, Action
from thermal_game.engine.settings import GameSettings
from thermal_game.engine.datafeed import DataFeed
from thermal_game.engine.reward import RewardConfig

# =========================
#   CONFIG (edit if needed)
# =========================
PKG_DIR   = REPO_ROOT / "thermal_game"
DATA_DIR  = PKG_DIR / "data"   # <-- C:\_projekty_repo\noc\thermal_game\data
MODEL_DIR = REPO_ROOT / "models"

WEATHER_CSV_NAME = "_2ndweekXX_prices_weather_seasons_FROM_2023_RELABELED_TO_2025.csv"  # Match GUI data file
LOAD_CSV_NAME    = "load_profile.csv"

START_DATE        = dt.date(2025, 3, 1)
# TIMESTEPS         = 500_000
TIMESTEPS         = 5_000

N_ENVS            = 4
STEPS_PER_EPISODE = 4 * 24 * 7
PV_ON             = True
ALLOW_GRID_CHARGE = True
SEED              = 0
SKIP_TRAINING     = False  # Set to True to skip training and just evaluate existing model

# =========================
#   Gymnasium Environment
# =========================
class ThermalGameEnv(gym.Env):
    """
    action: [hvac in [-1,1], battery in [-1,1] -> {-1,0,1} via deadband]
    obs: [T_in, T_out, soc, price, pv_kw, base_kw, sin_hour, cos_hour]
    reward: engine step.metrics["reward"] (financial + comfort)
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        weather_csv: str | Path,
        load_csv: str | Path,
        *,
        start_date: dt.date = dt.date(2025, 3, 1),
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

    def _obs(self, row, engine_step_metrics) -> np.ndarray:
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
        obs = self._obs(row, engine_step_metrics=None)
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
        m = step.metrics
        self.state = step.state

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
        }
        return obs, reward, terminated, truncated, info

# =========================
#   Helpers
# =========================
def make_env(weather_csv: str, load_csv: str, *, seed: int = 0, **env_kwargs):
    from stable_baselines3.common.monitor import Monitor
    def _f():
        env = ThermalGameEnv(weather_csv=weather_csv, load_csv=load_csv, seed=seed, **env_kwargs)
        return Monitor(env)
    return _f

def run():
    # Build canonical paths to your real data dir
    weather_csv = DATA_DIR / WEATHER_CSV_NAME
    load_csv    = DATA_DIR / LOAD_CSV_NAME
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Nice error if files are missing
    if not weather_csv.exists() or not load_csv.exists():
        raise FileNotFoundError(
            f"Missing data:\n  {weather_csv}\n  {load_csv}\n"
            f"Edit DATA_DIR/filenames at top of {Path(__file__).name}."
        )

    # Print current settings for debugging
    settings = GameSettings()
    settings.start_date = START_DATE
    print("\n>> Current GameSettings:")
    print(f"   comfort_target_C: {settings.comfort_target_C}")
    print(f"   comfort_tolerance_occupied_C: {getattr(settings, 'comfort_tolerance_occupied_C', 'NOT SET')}")
    print(f"   comfort_tolerance_unoccupied_C: {getattr(settings, 'comfort_tolerance_unoccupied_C', 'NOT SET')}")
    print(f"   comfort_anchor_eur_per_deg2_hour: {settings.comfort_anchor_eur_per_deg2_hour}")
    print(f"   export_tariff_ratio: {settings.export_tariff_ratio}")
    print(f"   pv_size_kw: {settings.pv_size_kw}")
    print(f"   hvac_size_kw: {settings.hvac_size_kw}")
    print(f"   batt_size_kwh: {settings.batt_size_kwh}")
    print(f"   start_date: {settings.start_date}")
    print(f"   Data file: {weather_csv.name}")
    
    # Debug print to confirm defaults
    print(">> Using defaults — hvac_size_kw:", settings.hvac_size_kw,
          " batt_size_kwh:", settings.batt_size_kwh,
          " pv_size_kw:", settings.pv_size_kw)
    print()

    # Consistent model path with .zip extension
    model_path = MODEL_DIR / "sac_thermal_game.zip"

    if not SKIP_TRAINING:
        print(">> training SAC…")
        vec_env = make_vec_env(
            make_env(str(weather_csv), str(load_csv),
                     start_date=START_DATE,
                     steps_per_episode=STEPS_PER_EPISODE,
                     pv_on=PV_ON,
                     allow_grid_charge=ALLOW_GRID_CHARGE),
            n_envs=N_ENVS, seed=SEED
        )
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=256,
            tau=0.02,
            gamma=0.995,
            train_freq=64,
            gradient_steps=64,
            target_update_interval=1,
            buffer_size=200_000,
            ent_coef="auto",
        )
        model.learn(total_timesteps=TIMESTEPS)
        model.save(str(model_path))
        print(f">> saved model → {model_path}")
    else:
        print(">> skipping training, loading existing model…")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Set SKIP_TRAINING=False to train first.")

    print(">> evaluating one episode…")
    env = ThermalGameEnv(
        str(weather_csv), str(load_csv),
        start_date=START_DATE,
        steps_per_episode=STEPS_PER_EPISODE,
        pv_on=PV_ON,
        allow_grid_charge=ALLOW_GRID_CHARGE,
        seed=SEED
    )
    model = SAC.load(str(model_path))
    obs, info = env.reset()
    done = False
    trunc = False
    ep_r = 0.0
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        ep_r += r
    print(">> episode reward:", ep_r, "last info:", info)

if __name__ == "__main__":
    run()
