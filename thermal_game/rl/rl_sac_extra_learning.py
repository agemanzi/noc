# rl_sac_extra_learning.py
from __future__ import annotations

# Prevent thread oversubscription (env workers × BLAS threads)
# Set NUM_* threads before importing numpy/torch so each worker uses 1 BLAS thread
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")   # macOS
os.environ.setdefault("BLIS_NUM_THREADS", "1")

# --- run-from-file friendly path bootstrap (no install needed) ---
import sys
from pathlib import Path
_THIS_FILE = Path(__file__).resolve()
# repo root: .../noc
REPO_ROOT = _THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ---------------------------------------------------------------

import argparse
import datetime as dt
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import LinearSchedule

# Engine deps (available under thermal_game/)
from thermal_game.engine.simulation import SimulationEngine
from thermal_game.engine.state import GameState, Action
from thermal_game.engine.settings import GameSettings
from thermal_game.engine.datafeed import DataFeed
from thermal_game.engine.reward import RewardConfig

# =========================
#   CONFIG (minimal)
# =========================
PKG_DIR   = REPO_ROOT / "thermal_game"
DATA_DIR  = PKG_DIR / "data"
MODEL_DIR = REPO_ROOT / "models"

WEATHER_CSV_NAME = "_2ndweekXX_prices_weather_seasons_FROM_2023_RELABELED_TO_2025.csv"
LOAD_CSV_NAME    = "load_profile.csv"

N_ENVS            = 4
GAME_DAYS         = 4
STEPS_PER_EPISODE = 4 * 24 * GAME_DAYS
PV_ON             = True
ALLOW_GRID_CHARGE = True
SEED              = 0

MODEL_PATH = MODEL_DIR / "sac_thermal_game.zip"   # same as the runner

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
        start_date: dt.date | None = None,
        steps_per_episode: int = 4 * 24 * GAME_DAYS,
        pv_on: bool = True,
        allow_grid_charge: bool = True,
        reward_cfg: RewardConfig | None = None,
        settings: GameSettings | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.rng = np.random.RandomState(seed or 0)

        self.settings = settings or GameSettings()
        if start_date is not None:
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

        # persistence for realism
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
            T_envelope=22.0
        )
        self.feed.set_anchor_date(self.settings.start_date)
        self._steps = 0
        self._prev_hvac_kw = 0.0
        self._prev_Tin_measured = None
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

        self.reward_cfg = RewardConfig(
            comfort_target_C=self.settings.comfort_target_C,
            comfort_tolerance_occupied_C=self.settings.comfort_tolerance_occupied_C,
            comfort_tolerance_unoccupied_C=self.settings.comfort_tolerance_unoccupied_C,
            comfort_weight=self.settings.comfort_anchor_eur_per_deg2_hour * self.engine.dt_h,
            export_tariff_ratio=self.settings.export_tariff_ratio,
            **({"comfort_inside_bonus": getattr(self.settings, "comfort_inside_bonus_eur_per_step", 0.0)}
               if "comfort_inside_bonus" in getattr(RewardConfig, "__dataclass_fields__", {}) else {})
        )

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
    def _f():
        env = ThermalGameEnv(weather_csv=weather_csv, load_csv=load_csv, seed=seed, **env_kwargs)
        return Monitor(env)
    return _f

def continue_training(extra_steps: int, aggressive: bool, save_suffix: str | None, update_main: bool = True):
    # Build canonical paths
    weather_csv = DATA_DIR / WEATHER_CSV_NAME
    load_csv    = DATA_DIR / LOAD_CSV_NAME
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not weather_csv.exists() or not load_csv.exists():
        raise FileNotFoundError(
            f"Missing data:\n  {weather_csv}\n  {load_csv}\n"
            f"Edit filenames at top of {Path(__file__).name}."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train once with rl_sac_runner.py first.")

    settings = GameSettings()  # use defaults (including start_date)
    print("\n>> Extra learning setup")
    print(f"   start_date: {settings.start_date}  pv={settings.pv_size_kw}kW  hvac={settings.hvac_size_kw}kW  batt={settings.batt_size_kwh}kWh")
    print(f"   model: {MODEL_PATH.name}")
    print(f"   profile: {'AGGRESSIVE' if aggressive else 'baseline'}")
    print(f"   extra_steps: {extra_steps:,}\n")

    # Vec env with normalization (same as runner)
    vec_env = make_vec_env(
        make_env(str(weather_csv), str(load_csv),
                 start_date=settings.start_date,
                 steps_per_episode=STEPS_PER_EPISODE,
                 pv_on=PV_ON,
                 allow_grid_charge=ALLOW_GRID_CHARGE),
        n_envs=N_ENVS, seed=SEED
    )
    
    # Load normalization stats if they exist
    vecnorm_path = MODEL_DIR / "vecnorm_stats.pkl"
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True  # Keep training mode for continued learning
        vec_env.norm_reward = False
        print(f"   loaded normalization stats from {vecnorm_path.name}")
    else:
        # Apply normalization like in the runner if stats don't exist yet
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,        # keep true € scale
            clip_obs=10.0
        )
        print("   applied fresh normalization (no existing stats)")

    # --- Load the existing model and optionally override some hyperparams ---
    # Update the parameters to match the improved runner defaults
    if aggressive:
        # Even more aggressive than runner defaults
        lr_schedule = LinearSchedule(2e-3, 5e-4, extra_steps)
        # DO NOT change policy_kwargs when loading from existing model
        custom = {
            "learning_rate": lr_schedule,
            "train_freq": 256,              # More frequent updates
            "gradient_steps": 512,          # More gradient steps
            "batch_size": 1024,             # Larger batches
            "tau": 0.01,
            "gamma": 0.995,
            "ent_coef": "auto_0.2",         # Higher exploration than runner
            "buffer_size": 1_500_000,       # Larger buffer
            "learning_starts": 15_000,
            "sde_sample_freq": 2,           # Harmless even if use_sde=False
            # DO NOT pass "policy_kwargs" when loading from existing .zip
        }
    else:
        # Match the runner's improved baseline parameters
        lr_schedule = LinearSchedule(1e-3, 3e-4, extra_steps)
        custom = {
            "learning_rate": lr_schedule,
            "train_freq": 256,              # Match runner's baseline
            "gradient_steps": 512,          # Match runner's baseline
            "batch_size": 512,
            "tau": 0.01,
            "gamma": 0.995,
            "ent_coef": "auto_0.1",         # Match runner's lower entropy
            "buffer_size": 1_000_000,
            "learning_starts": 10_000,
            "sde_sample_freq": 4,           # Harmless even if use_sde=False
            # DO NOT pass "policy_kwargs" when loading from existing .zip
        }

    model = SAC.load(
        str(MODEL_PATH),
        env=vec_env,
        custom_objects=custom,  # ← applies improved parameters
        device="auto",
        print_system_info=False,
    )

    # IMPORTANT: continue counting timesteps instead of resetting to 0
    model.learn(total_timesteps=int(extra_steps), reset_num_timesteps=False)

    # Save (either overwrite, or with suffix/timestamp)
    if save_suffix:
        out_path = MODEL_DIR / f"sac_thermal_game_{save_suffix}.zip"
        vecnorm_out = MODEL_DIR / f"vecnorm_stats_{save_suffix}.pkl"
    else:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = MODEL_DIR / f"sac_thermal_game_cont_{ts}.zip"
        vecnorm_out = MODEL_DIR / f"vecnorm_stats_cont_{ts}.pkl"

    model.save(str(out_path))
    
    # Save updated normalization stats
    vec_env.save(vecnorm_out)
    print(f">> extra learning complete → {out_path}")
    print(f">> updated normalization stats → {vecnorm_out}")
    
    # If no suffix provided and we want to update the main model, also save to default names
    if not save_suffix and update_main:
        main_model_path = MODEL_DIR / "sac_thermal_game.zip"
        main_vecnorm_path = MODEL_DIR / "vecnorm_stats.pkl"
        model.save(str(main_model_path))
        vec_env.save(main_vecnorm_path)
        print(f">> also updated main model → {main_model_path}")
        print(f">> also updated main vecnorm → {main_vecnorm_path}")

def main():
    p = argparse.ArgumentParser(description="Continue SAC training from existing model with improved parameters.")
    p.add_argument("--steps", type=int, default=100_000, help="extra timesteps to learn")
    p.add_argument("--aggressive", action="store_true", 
                  help="use aggressive learning profile (larger network, higher LR, more updates)")
    p.add_argument("--save-suffix", type=str, default="", 
                  help="filename suffix; if empty, timestamp is used and main model is also updated")
    p.add_argument("--no-update-main", action="store_true", 
                  help="don't update the main model files when using timestamp")
    args = p.parse_args()

    suffix = args.save_suffix.strip() or None
    continue_training(
        extra_steps=args.steps, 
        aggressive=args.aggressive, 
        save_suffix=suffix,
        update_main=not args.no_update_main
    )

if __name__ == "__main__":
    main()
