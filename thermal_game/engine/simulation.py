# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import Dict, Optional
import numpy as np

from .state import GameState, Action, StepResult
from .settings import GameSettings
from .reward import step_reward, RewardConfig


@dataclass
class SimulationEngine:
    # core step size (s) — GUI uses 900 (15 min)
    dt: float = 900.0
    player_controls_hvac: bool = True

    # -------- (legacy 1R1C kept for back-compat defaults, made leakier) --------
    tau_h: float = 4.0                 # faster heat loss
    cap_kwh_per_K: float = 2.5         # less thermal mass

    # -------- R3C2 (air + envelope; 3 resistances: air↔env, env↔amb, air↔amb) --------
    # Much leakier defaults for aggressive drift (τ ≈ 2–3h)
    Ci_kwh_per_K: float = 2.0          # lighter air/effective zone → faster response
    Ce_kwh_per_K: float = 12.0         # lighter envelope → shorter memory
    Rie_degC_per_kW: float = 8.0       # tighter air↔envelope coupling (less buffering)
    Rea_degC_per_kW: float = 10.0      # more loss to ambient through walls
    Ria_degC_per_kW: float = 15.0      # much leakier infiltration path
    clip_temp_c: tuple[float, float] = (-40.0, 60.0)

    # -------- Infiltration model (dynamic leakiness) --------
    # Effective ambient↔air conductance G_total = 1/Ria + G_infil(ACH, wind, windows)
    house_volume_m3: float = 250.0         # typical small house/apartment
    ach_base: float = 0.5                   # air changes per hour at 0 m/s wind, closed windows
    ach_per_mps: float = 0.35               # wind-driven ACH slope
    window_ach_at_full_open: float = 6.0    # extra ACH when windows fully open
    bridge_factor: float = 1.25             # multiplies envelope conductance (thermal bridges etc.)

    allow_grid_charge: bool = True

    # -------- Battery defaults (slightly less efficient) --------
    batt_eta_ch: float = 0.92
    batt_eta_dis: float = 0.93
    batt_min_soc: float = 0.15
    batt_max_soc: float = 0.85
    batt_c_rate: float = 0.4

    # -------- HVAC realism & constraints --------
    hvac_ramp_kw_per_step: float = 0.5    # max change in HVAC power per step
    hvac_defrost_temp_threshold_c: float = 2.0  # defrost penalties below this
    hvac_defrost_cop_factor: float = 0.7  # COP multiplier when defrosting
    hvac_defrost_cap_factor: float = 0.8  # power capacity multiplier when defrosting

    # Ramp helper: proportional boost scales with remaining gap to target
    hvac_ramp_frac_per_step: float = 0.35   # fraction of target gap you can close each step (0..1)

    # -------- HVAC Carnot params (a bit worse) --------
    hvac_sink_temp_C: float = 35.0
    hvac_source_approach_K: float = 5.0
    hvac_sink_approach_K: float = 8.0
    hvac_eta_carnot: float = 0.4
    hvac_cop_min: float = 1.0
    hvac_cop_max: float = 8.0

    # Controller defaults
    Tin_set_C: float = 22.0
    ctrl_aggr: float = 0.35
    
    # Controller responsiveness
    ctrl_deadband_C: float = 0.25              # no extra aggressiveness inside this band
    ctrl_boost_gain_per_K: float = 0.6         # extra aggressiveness per °C of error beyond deadband (unitless)
    ctrl_max_aggr: float = 1.6                 # clamp for total aggressiveness multiplier

    # Dynamic ramp boost (lets power ramp faster when far from setpoint)
    hvac_ramp_slew_kw_per_K: float = 1.2       # extra kW-per-step of ramp per °C error beyond deadband
    
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)

    # -------- Disturbances & realism --------
    # Internal gains noise (kW stdev when occupied)
    internal_gains_noise_std: float = 0.2
    
    # Solar pop parameters (short noon bursts)
    solar_pop_temp_threshold: float = 5.0    # °C outdoor temp for solar events
    solar_pop_min_kw: float = 0.5
    solar_pop_max_kw: float = 1.5
    solar_pop_duration_steps: int = 3        # how many steps a solar pop lasts
    
    # Window event parameters
    window_event_probability: float = 0.02   # chance per step of window opening
    window_open_min_frac: float = 0.2
    window_open_max_frac: float = 0.5
    window_event_duration_steps: int = 3     # steps to keep window open
    
    # Sensor lag (measurement filtering)
    temp_sensor_lag_factor: float = 0.3      # Tin_meas = (1-α)*Tin_prev + α*Tin_true
    
    # Difficulty presets
    difficulty_mode: str = "normal"          # "easy", "normal", "hard"

    @property
    def dt_h(self) -> float:
        return float(self.dt) / 3600.0

    def apply_difficulty_preset(self, mode: str = "normal"):
        """Apply one of the difficulty presets: easy, normal, hard"""
        self.difficulty_mode = mode
        if mode == "easy":
            # Almost today's parameters
            self.Ci_kwh_per_K = 3.0
            self.Ce_kwh_per_K = 20.0
            self.Rie_degC_per_kW = 10.0
            self.Rea_degC_per_kW = 20.0
            self.Ria_degC_per_kW = 60.0
            self.hvac_ramp_kw_per_step = 999.0  # no limit
            self.hvac_defrost_cop_factor = 1.0  # no penalty
            self.hvac_defrost_cap_factor = 1.0
            self.bridge_factor = 1.0
            
        elif mode == "normal":
            self.Ci_kwh_per_K = 2.0
            self.Ce_kwh_per_K = 12.0
            self.Rie_degC_per_kW = 8.0
            self.Rea_degC_per_kW = 10.0
            self.Ria_degC_per_kW = 15.0
            self.hvac_ramp_kw_per_step = 2.0          # was 0.5
            self.hvac_ramp_slew_kw_per_K = 3.0        # added for faster boost
            self.hvac_defrost_cop_factor = 0.7
            self.hvac_defrost_cap_factor = 0.8
            self.bridge_factor = 1.25
                    
        elif mode == "hard":
            # Must preheat/pre-cool
            self.Ci_kwh_per_K = 1.5
            self.Ce_kwh_per_K = 10.0
            self.Rie_degC_per_kW = 6.0
            self.Rea_degC_per_kW = 8.0
            self.Ria_degC_per_kW = 10.0
            self.hvac_ramp_kw_per_step = 0.5
            self.hvac_defrost_cop_factor = 0.7
            self.hvac_defrost_cap_factor = 0.8
            self.bridge_factor = 1.4

    # ---------- Disturbance generators ----------
    def _generate_disturbances(self, inputs: Dict[str, float], step_count: int = 0) -> Dict[str, float]:
        """Generate realistic disturbances to make the game more challenging"""
        disturbances = {}
        
        # Internal gains noise (only when occupied)
        occupied = bool(inputs.get("occupied_home", 0))
        if occupied and self.internal_gains_noise_std > 0:
            disturbances["q_internal_noise_kw"] = np.random.normal(0, self.internal_gains_noise_std)
        else:
            disturbances["q_internal_noise_kw"] = 0.0
        
        # Solar pops (short noon bursts on warm days)
        T_out = float(inputs.get("T_outside", 15.0))
        solar_pop_active = inputs.get("solar_pop_remaining", 0) > 0
        if not solar_pop_active and T_out > self.solar_pop_temp_threshold and np.random.random() < 0.1:
            # Start a new solar pop
            disturbances["solar_pop_kw"] = np.random.uniform(self.solar_pop_min_kw, self.solar_pop_max_kw)
            disturbances["solar_pop_remaining"] = self.solar_pop_duration_steps
        elif solar_pop_active:
            # Continue existing solar pop
            disturbances["solar_pop_kw"] = inputs.get("solar_pop_kw", 0.0)
            disturbances["solar_pop_remaining"] = max(0, inputs.get("solar_pop_remaining", 0) - 1)
        else:
            disturbances["solar_pop_kw"] = 0.0
            disturbances["solar_pop_remaining"] = 0
        
        # Window events (occasional opening)
        window_event_active = inputs.get("window_event_remaining", 0) > 0
        if not window_event_active and np.random.random() < self.window_event_probability:
            # Start new window event
            disturbances["window_open_frac"] = np.random.uniform(
                self.window_open_min_frac, self.window_open_max_frac
            )
            disturbances["window_event_remaining"] = self.window_event_duration_steps
        elif window_event_active:
            # Continue existing window event
            disturbances["window_open_frac"] = inputs.get("window_open_frac", 0.0)
            disturbances["window_event_remaining"] = max(0, inputs.get("window_event_remaining", 0) - 1)
        else:
            disturbances["window_open_frac"] = 0.0
            disturbances["window_event_remaining"] = 0
            
        return disturbances
    def _carnot_heat(self, ambient_temp_C: float) -> Dict[str, float]:
        source_C = float(ambient_temp_C)
        sink_C   = float(self.hvac_sink_temp_C)
        T_cold_K = (source_C - self.hvac_source_approach_K) + 273.15
        T_hot_K  = (sink_C   + self.hvac_sink_approach_K)   + 273.15
        T_cold_K = max(1.0, T_cold_K)
        T_hot_K  = max(T_cold_K + 1e-3, T_hot_K)
        dT_K = T_hot_K - T_cold_K
        cop_carnot = T_hot_K / dT_K
        cop = self.hvac_eta_carnot * cop_carnot
        cop = float(np.clip(cop, self.hvac_cop_min, self.hvac_cop_max))
        return {"cop": cop, "cop_carnot": float(cop_carnot), "T_source_C": source_C, "T_sink_C": sink_C}

    def _carnot_cool(self, ambient_temp_C: float) -> Dict[str, float]:
        source_C = float(ambient_temp_C)
        sink_C   = float(self.hvac_sink_temp_C)
        T_cold_K = (source_C - self.hvac_source_approach_K) + 273.15
        T_hot_K  = (sink_C   + self.hvac_sink_approach_K)   + 273.15
        T_cold_K = max(1.0, T_cold_K)
        T_hot_K  = max(T_cold_K + 1e-3, T_hot_K)
        dT_K = T_hot_K - T_cold_K
        eer_carnot = T_cold_K / dT_K
        eer = self.hvac_eta_carnot * eer_carnot
        eer = float(np.clip(eer, self.hvac_cop_min, self.hvac_cop_max))
        return {"cop": eer, "cop_carnot": float(eer_carnot), "T_source_C": source_C, "T_sink_C": sink_C}

    # ---------- Infiltration helpers ----------
    @staticmethod
    def _air_cp_kwh_per_m3K() -> float:
        # ~1.2 kg/m3 * 1.005 kJ/kg-K = 1.206 kJ/m3-K = 0.000335 kWh/m3-K
        return 0.000335

    def _infiltration_conductance_kw_per_K(self, ach: float) -> float:
        # G_infil = cp_air * Vdot = cp * (ACH * Volume)  [kW/°C]
        ach = max(0.0, float(ach))
        return self._air_cp_kwh_per_m3K() * ach * max(0.0, float(self.house_volume_m3))

    def _effective_ria(self, base_ria_degC_per_kW: float, inputs: Dict[str, float]) -> float:
        wind = float(inputs.get("wind_mps", 0.0))
        win_open = float(np.clip(inputs.get("window_open_frac", 0.0), 0.0, 1.0))
        # total ACH = base + wind-driven + window-driven
        ach = self.ach_base + self.ach_per_mps * wind + self.window_ach_at_full_open * win_open
        G_base = 1.0 / max(1e-12, base_ria_degC_per_kW)
        G_infil = self._infiltration_conductance_kw_per_K(ach)
        G_total = G_base + G_infil
        return 1.0 / max(1e-12, G_total)

    # ---------- HVAC dispatch ----------
    def _apply_defrost_penalty(self, ambient_temp_C: float, cop: float, max_capacity_kw: float) -> tuple[float, float]:
        """Apply defrost penalties when outdoor temp is too low"""
        if ambient_temp_C < self.hvac_defrost_temp_threshold_c:
            cop_adjusted = cop * self.hvac_defrost_cop_factor
            capacity_adjusted = max_capacity_kw * self.hvac_defrost_cap_factor
            return cop_adjusted, capacity_adjusted
        return cop, max_capacity_kw

    def _hvac_dispatch_heat(self, ambient_temp_C: float, heat_need_kW: float,
                            max_in_kw: float, prev_hvac_kw: float = 0.0) -> Dict[str, float]:
        if max_in_kw <= 0 or heat_need_kW <= 0:
            return {"hvac_elec_kW": 0.0, "hvac_heat_kW": 0.0, "cop": float("nan"),
                    "cop_carnot": float("nan"), "T_source_C": float("nan"), "T_sink_C": float("nan")}
        
        ci = self._carnot_heat(ambient_temp_C)
        cop = ci["cop"]
        
        # Apply defrost penalty
        cop, max_capacity = self._apply_defrost_penalty(ambient_temp_C, cop, max_in_kw)
        
        # Apply ramp rate limiting
        max_ramp_up = prev_hvac_kw + self.hvac_ramp_kw_per_step
        max_capacity = min(max_capacity, max_ramp_up)
        
        needed_in_kW = heat_need_kW / max(cop, 1e-9)
        hvac_elec_kW = min(max_capacity, needed_in_kW)
        hvac_heat_kW = hvac_elec_kW * cop
        
        return {"hvac_elec_kW": hvac_elec_kW, "hvac_heat_kW": hvac_heat_kW,
                "cop": cop, "cop_carnot": ci["cop_carnot"],
                "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}

    def _hvac_dispatch_cool(self, ambient_temp_C: float, cool_need_kW: float,
                            max_in_kw: float, prev_hvac_kw: float = 0.0) -> Dict[str, float]:
        if max_in_kw <= 0 or cool_need_kW <= 0:
            return {"hvac_elec_kW": 0.0, "hvac_heat_kW": 0.0, "cop": float("nan"),
                    "cop_carnot": float("nan"), "T_source_C": float("nan"), "T_sink_C": float("nan")}
        
        ci = self._carnot_cool(ambient_temp_C)
        eer = ci["cop"]
        
        # Apply defrost penalty (less relevant for cooling but consistent)
        eer, max_capacity = self._apply_defrost_penalty(ambient_temp_C, eer, max_in_kw)
        
        # Apply ramp rate limiting
        max_ramp_up = prev_hvac_kw + self.hvac_ramp_kw_per_step
        max_capacity = min(max_capacity, max_ramp_up)
        
        needed_in_kW = cool_need_kW / max(eer, 1e-9)
        hvac_elec_kW = min(max_capacity, needed_in_kW)
        hvac_cool_kW = hvac_elec_kW * eer
        
        return {"hvac_elec_kW": hvac_elec_kW, "hvac_heat_kW": -hvac_cool_kW,
                "cop": eer, "cop_carnot": ci["cop_carnot"],
                "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}

    def _responsive_q_need(self, e_C: float, q_env_kw: float) -> float:
        """
        Compute more-responsive heat/cool demand.
        e_C > 0 => need heating; e_C < 0 => need cooling.
        """
        dt_h = max(self.dt_h, 1e-12)
        # Base proportional term scaled like your original q_to_reach
        base_aggr = float(np.clip(self.ctrl_aggr, 0.0, 1.0))
        K_base = (self.cap_kwh_per_K / dt_h) * base_aggr

        # Gain scheduling: increase aggressiveness outside a deadband
        mag = abs(e_C)
        boost = max(0.0, mag - float(self.ctrl_deadband_C))
        aggr_eff = float(np.clip(base_aggr + self.ctrl_boost_gain_per_K * boost, 0.0, self.ctrl_max_aggr))
        K_eff = (self.cap_kwh_per_K / dt_h) * aggr_eff

        q_to_reach_kw = K_eff * e_C
        return q_env_kw + q_to_reach_kw
    def _thermal_step_r3c2(
        self,
        Ti: float,
        Te: float,
        Ta: float,
        q_hvac_kw: float,
        q_solar_kw: float = 0.0,
        q_internal_kw: float = 0.0,
        Ria_override: Optional[float] = None,
        Rea_override: Optional[float] = None,
    ) -> tuple[float, float, dict]:
        """
        3R2C (air + envelope):
          Ci * dTi/dt = (Te - Ti)/Rie + (Ta - Ti)/Ria + (q_solar + q_internal + q_hvac)
          Ce * dTe/dt = (Ti - Te)/Rie + (Ta - Te)/Rea
        """
        Ci = max(1e-12, float(self.Ci_kwh_per_K))
        Ce = max(1e-12, float(self.Ce_kwh_per_K))
        Rie = max(1e-12, float(self.Rie_degC_per_kW))
        Ria = max(1e-12, float(Ria_override if Ria_override is not None else self.Ria_degC_per_kW))
        # envelope conductance gets a multiplier to emulate thermal bridges
        Rea_raw = float(Rea_override if Rea_override is not None else self.Rea_degC_per_kW)
        Rea = max(1e-12, Rea_raw / max(1e-6, self.bridge_factor))
        dt_h = self.dt_h

        # conductive/advective heat flows (kW, positive into node)
        q_ie = (Te - Ti) / Rie     # env -> air
        q_ea = (Ta - Te) / Rea     # amb -> env
        q_ia = (Ta - Ti) / Ria     # amb -> air (ventilation/infiltration)
        q_gains = float(q_solar_kw) + float(q_internal_kw) + float(q_hvac_kw)

        # derivatives (°C/h)
        dTi_dt = (q_ie + q_ia + q_gains) / Ci
        dTe_dt = ((Ti - Te) / Rie + q_ea) / Ce

        # forward Euler
        Ti_next = float(np.clip(Ti + dt_h * dTi_dt, self.clip_temp_c[0], self.clip_temp_c[1]))
        Te_next = float(np.clip(Te + dt_h * dTe_dt, self.clip_temp_c[0], self.clip_temp_c[1]))

        diag = {
            "q_ie_kw": q_ie,
            "q_ea_kw": q_ea,
            "q_ia_kw": q_ia,
            "q_hvac_kw": float(q_hvac_kw),
            "q_solar_kw": float(q_solar_kw),
            "q_internal_kw": float(q_internal_kw),
            "dTi": dt_h * dTi_dt,
            "dTe": dt_h * dTe_dt,
            "Ria_eff": Ria,
            "Rea_eff": Rea,
        }
        return Ti_next, Te_next, diag

    # ---------- Battery step (bounded) ----------
    def _battery_step(self, soc_prev: float, cmd: int, settings: GameSettings) -> Dict[str, float]:
        cap_kwh = max(0.0, float(settings.batt_size_kwh))
        if cap_kwh <= 0.0 or cmd == 0:
            return {"battery_kw": 0.0, "soc_next": float(np.clip(soc_prev, 0.0, 1.0))}
        p_lim = self.batt_c_rate * cap_kwh
        soc_min = self.batt_min_soc
        soc_max = self.batt_max_soc
        soc = float(np.clip(soc_prev, 0.0, 1.0))
        if cmd > 0:
            max_dis_kwh = (soc - soc_min) * cap_kwh
            max_dis_kw = max(0.0, max_dis_kwh / max(self.dt_h, 1e-12))
            p_kw = min(p_lim, max_dis_kw)
            battery_kw = +p_kw * self.batt_eta_dis
            soc_next = soc - (p_kw * self.dt_h) / cap_kwh
        else:
            max_chg_kwh = (soc_max - soc) * cap_kwh
            max_chg_kw = max(0.0, max_chg_kwh / max(self.dt_h, 1e-12))
            p_kw = min(p_lim, max_chg_kw)
            battery_kw = -p_kw / self.batt_eta_ch
            soc_next = soc + (p_kw * self.dt_h) / cap_kwh
        return {"battery_kw": float(battery_kw), "soc_next": float(np.clip(soc_next, 0.0, 1.0))}

    # ---------- Main step ----------
    def step(self, prev: GameState, action: Action, inputs: Optional[Dict[str, float]] = None) -> StepResult:
        inputs = inputs or {}
        settings: GameSettings = inputs.get("settings") or GameSettings()
        cfg: RewardConfig = inputs.get("reward_cfg", self.reward_cfg)

        # Get previous HVAC power for ramp limiting (stored in inputs or default to 0)
        prev_hvac_kw = float(inputs.get("prev_hvac_kw", 0.0))

        # Generate disturbances
        disturbances = self._generate_disturbances(inputs, step_count=int(inputs.get("step_count", 0)))
        
        # Update inputs with disturbances
        inputs.update(disturbances)

        # exogenous inputs (including disturbances)
        T_out = float(inputs.get("T_outside", getattr(prev, "T_outside", 15.0)))
        pv_kw = float(inputs.get("pv_kw", 0.0))
        base_load_kw = float(inputs.get("base_load_kw", 0.0))
        price = float(inputs.get("price", 0.0))

        # Diagnostics (legacy)
        q_env_kw = (T_out - prev.T_inside) * (self.cap_kwh_per_K / max(self.tau_h, 1e-12))

        # HVAC electrical cap from settings
        hvac_elec_cap_kw = max(0.0, float(settings.hvac_size_kw))

        if self.player_controls_hvac:
            u = float(np.clip(getattr(action, "hvac", 0.0), -1.0, 1.0))
            target_elec_kw = abs(u) * hvac_elec_cap_kw

            # --- proportional ramp: close a fraction of the remaining gap each step ---
            gap = target_elec_kw - prev_hvac_kw
            base = self.hvac_ramp_kw_per_step
            prop = self.hvac_ramp_frac_per_step * abs(gap)          # scales with how far we are
            extra = self.hvac_ramp_slew_kw_per_K * max(
                0.0, abs(float(inputs.get("Tin_set_C", self.Tin_set_C)) - prev.T_inside) - self.ctrl_deadband_C
            )
            ramp_up   = base + prop + extra
            ramp_down = base + prop + extra  # symmetric; make smaller if you want slower spin-down

            # one-step update toward target (bounded)
            if gap >= 0:
                elec_in_kw = prev_hvac_kw + min(gap, ramp_up)
            else:
                elec_in_kw = prev_hvac_kw + max(gap, -ramp_down)

            # now elec_in_kw ramps quickly toward hvac_elec_cap_kw when u=±1, independent of COP
            if u > 0.0:
                ci = self._carnot_heat(T_out)
                cop, _ = self._apply_defrost_penalty(T_out, ci["cop"], hvac_elec_cap_kw)
                hvac_kw   = elec_in_kw
                q_hvac_kw = hvac_kw * cop
                hv = {"cop": cop, "cop_carnot": ci["cop_carnot"],
                      "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}
            elif u < 0.0:
                ci = self._carnot_cool(T_out)
                eer, _ = self._apply_defrost_penalty(T_out, ci["cop"], hvac_elec_cap_kw)
                hvac_kw   = elec_in_kw
                q_hvac_kw = -hvac_kw * eer
                hv = {"cop": eer, "cop_carnot": ci["cop_carnot"],
                      "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}
            else:
                hvac_kw = 0.0
                q_hvac_kw = 0.0
                hv = {"cop": float("nan"), "cop_carnot": float("nan"),
                      "T_source_C": float("nan"), "T_sink_C": float("nan")}
            q_need_kw = 0.0
        else:
            Tin_set = float(inputs.get("Tin_set_C", self.Tin_set_C))
            # legacy environment term retained (your q_env_kw above)
            e = Tin_set - prev.T_inside
            q_need_kw = self._responsive_q_need(e, q_env_kw)

            # Temporarily boost the ramp limit when far from setpoint
            dead = float(self.ctrl_deadband_C)
            extra_ramp = self.hvac_ramp_slew_kw_per_K * max(0.0, abs(e) - dead)
            _old_ramp = self.hvac_ramp_kw_per_step
            try:
                self.hvac_ramp_kw_per_step = max(self.hvac_ramp_kw_per_step, extra_ramp)

                if q_need_kw > 0:
                    di = self._hvac_dispatch_heat(T_out, q_need_kw, hvac_elec_cap_kw, prev_hvac_kw)
                elif q_need_kw < 0:
                    di = self._hvac_dispatch_cool(T_out, -q_need_kw, hvac_elec_cap_kw, prev_hvac_kw)
                else:
                    di = {"hvac_elec_kW": 0.0, "hvac_heat_kW": 0.0,
                          "cop": float("nan"), "cop_carnot": float("nan"),
                          "T_source_C": float("nan"), "T_sink_C": float("nan")}
            finally:
                # Restore original ramp limit
                self.hvac_ramp_kw_per_step = _old_ramp

            hvac_kw   = di["hvac_elec_kW"]
            q_hvac_kw = di["hvac_heat_kW"]
            hv = {"cop": di["cop"], "cop_carnot": di["cop_carnot"],
                  "T_source_C": di["T_source_C"], "T_sink_C": di["T_sink_C"]}

        # Battery — load-following
        battery_cmd = int(inputs.get("battery_cmd", getattr(action, "battery", 0)))
        soc_prev = float(prev.soc)
        cap_kwh = max(0.0, float(settings.batt_size_kwh))
        p_lim   = self.batt_c_rate * cap_kwh

        net_no_batt_kw = (hvac_kw + base_load_kw) - pv_kw
        if battery_cmd > 0:
            want_kw = max(0.0, net_no_batt_kw)
        elif battery_cmd < 0:
            want_kw = -p_lim if self.allow_grid_charge else -max(0.0, -net_no_batt_kw)
        else:
            want_kw = 0.0

        tmp_cmd = 1 if want_kw > 1e-9 else (-1 if want_kw < -1e-9 else 0)
        batt_tmp = self._battery_step(soc_prev, tmp_cmd, settings)
        max_feasible_kw = batt_tmp["battery_kw"]

        if want_kw >= 0:
            battery_kw = min(max_feasible_kw, want_kw, p_lim)
        else:
            battery_kw = -min(abs(max_feasible_kw), abs(want_kw), p_lim)

        if battery_kw >= 0:
            p_kw = battery_kw / max(self.batt_eta_dis, 1e-9)
            soc_next = soc_prev - (p_kw * self.dt_h) / cap_kwh if cap_kwh > 0 else soc_prev
        else:
            p_kw = -battery_kw * self.batt_eta_ch
            soc_next = soc_prev + (p_kw * self.dt_h) / cap_kwh if cap_kwh > 0 else soc_prev
        soc_next = float(np.clip(soc_next, 0.0, 1.0))

        other_kw = -base_load_kw
        total_kw = pv_kw + battery_kw + (-hvac_kw) + other_kw
        grid_export_kw = max(0.0,  total_kw)
        grid_import_kw = max(0.0, -total_kw)
        import_kwh = grid_import_kw * self.dt_h
        export_kwh = grid_export_kw * self.dt_h

        # -------- R3C2 thermal update (with dynamic infiltration & bridges) --------
        # Base solar and internal gains
        q_solar_kw = float(inputs.get("q_solar_kw", 0.0))
        q_internal_kw = float(inputs.get("q_internal_kw", 0.0))
        
        # Add disturbances
        q_solar_kw += disturbances.get("solar_pop_kw", 0.0)
        q_internal_kw += disturbances.get("q_internal_noise_kw", 0.0)
        
        T_env_prev = float(getattr(prev, "T_envelope", prev.T_inside))

        # dynamic leakiness from wind & windows
        Ria_eff = self._effective_ria(self.Ria_degC_per_kW, inputs)
        # optional per-step override for Rea via bridge_factor is inside _thermal_step_r3c2

        T_in_next, T_env_next, therm_diag = self._thermal_step_r3c2(
            Ti=float(prev.T_inside),
            Te=T_env_prev,
            Ta=T_out,
            q_hvac_kw=q_hvac_kw,
            q_solar_kw=q_solar_kw,
            q_internal_kw=q_internal_kw,
            Ria_override=Ria_eff,
            Rea_override=None,
        )
        
        # Apply sensor lag for "measured" temperature (for control, if using auto control)
        prev_tin_meas = float(inputs.get("prev_Tin_measured", prev.T_inside))
        Tin_measured = (1.0 - self.temp_sensor_lag_factor) * prev_tin_meas + self.temp_sensor_lag_factor * T_in_next

        # Reward
        occupied = int(inputs.get("occupied_home", getattr(prev, "occupied", 0) or 0))
        complaint_count = int(inputs.get("complaint_count", 0))
        
        reward_bits = step_reward(
            Tin_C=T_in_next,
            occupied=occupied,
            import_kwh=import_kwh,
            export_kwh=export_kwh,
            price_eur_per_kwh=price,
            cfg=cfg,
            complaint_count=complaint_count,
        )

        cum_reward_prev  = float(getattr(prev, "cumulative_reward", 0.0))
        cum_fin_prev     = float(getattr(prev, "cumulative_financial", 0.0))
        cum_comf_prev    = float(getattr(prev, "cumulative_comfort", 0.0))
        cum_reward_next  = cum_reward_prev + float(reward_bits["reward_total"])
        cum_fin_next     = cum_fin_prev    + float(reward_bits["financial_score"])
        cum_comf_next    = cum_comf_prev   + float(reward_bits["comfort_score"])

        extra_fields = {}
        if hasattr(prev, "T_envelope"):
            extra_fields["T_envelope"] = T_env_next

        state = replace(
            prev,
            t=float(prev.t) + self.dt,
            T_inside=T_in_next,
            T_outside=T_out,
            soc=soc_next,
            kwh_used=float(prev.kwh_used) + import_kwh,
            ts=inputs.get("ts", getattr(prev, "ts", None)),
            occupied=int(inputs.get("occupied_home", getattr(prev, "occupied", 0) or 0)),
            cumulative_reward=cum_reward_next,
            cumulative_financial=cum_fin_next,
            cumulative_comfort=cum_comf_next,
            **extra_fields,
        )

        metrics = {
            "T_inside": state.T_inside,
            "T_envelope": T_env_next,
            "T_outside": T_out,
            "pv_kw": pv_kw,
            "battery_kw": battery_kw,
            "hvac_kw": hvac_kw,
            "other_kw": other_kw,
            "total_kw": total_kw,
            "electricity": hvac_kw * self.dt_h,
            "price": price,

            # HVAC introspection
            "hvac_cop": hv["cop"],
            "hvac_cop_carnot": hv["cop_carnot"],
            "hvac_T_source_C": hv["T_source_C"],
            "hvac_T_sink_C": hv["T_sink_C"],

            # controller diagnostics
            "q_need_kw": q_need_kw,
            "q_env_kw": q_env_kw,
            "q_hvac_kw": q_hvac_kw,

            # thermal diag (R3C2)
            "q_ie_kw": therm_diag["q_ie_kw"],
            "q_ea_kw": therm_diag["q_ea_kw"],
            "q_ia_kw": therm_diag["q_ia_kw"],
            "q_solar_kw": therm_diag["q_solar_kw"],
            "q_internal_kw": therm_diag["q_internal_kw"],
            "Ria_eff": therm_diag["Ria_eff"],
            "Rea_eff": therm_diag["Rea_eff"],

            # disturbances
            "solar_pop_kw": disturbances.get("solar_pop_kw", 0.0),
            "window_open_frac": disturbances.get("window_open_frac", 0.0),
            "internal_noise_kw": disturbances.get("q_internal_noise_kw", 0.0),
            "Tin_measured": Tin_measured,
            "prev_hvac_kw": hvac_kw,  # for next step's ramp limiting

            # economics / rewards
            "comfort_penalty":  reward_bits["comfort_penalty"],
            "complaint_penalty": reward_bits["complaint_penalty"],
            "complaint_count": reward_bits["new_complaint_count"],
            "import_kwh":       import_kwh,
            "export_kwh":       export_kwh,
            "import_cost":      reward_bits["import_cost"],
            "export_credit":    reward_bits["export_credit"],
            "net_opex":         reward_bits["net_opex"],
            "reward_fin":       reward_bits["financial_score"],
            "reward_comf":      reward_bits["comfort_score"],
            "reward":           reward_bits["reward_total"],
            "opex_cost":        reward_bits["net_opex"],

            # cumulative
            "cum_reward":       cum_reward_next,
            "cum_financial":    cum_fin_next,
            "cum_comfort":      cum_comf_next,
            
            # persistent disturbance states (for next step)
            "solar_pop_remaining": disturbances.get("solar_pop_remaining", 0),
            "window_event_remaining": disturbances.get("window_event_remaining", 0),
        }

        return StepResult(state=state, metrics=metrics)
