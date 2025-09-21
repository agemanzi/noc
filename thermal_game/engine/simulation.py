# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional
import numpy as np

from .state import GameState, Action, StepResult
from .settings import GameSettings
from .reward import step_reward, RewardConfig
@dataclass
class ThermalParams:
    C_th_kwh_per_degC: float   # thermal capacitance [kWh/°C]
    U_kw_per_degC: float       # overall conductance [kW/°C] (≈ 1/Rth)
    dt_h: float                # step size in hours
    clip_temp_c: tuple[float, float] = (-40.0, 60.0)
    
@dataclass
class SimulationEngine:
    # core step size (s) — GUI uses 900 (15 min)
    dt: float = 900.0
    player_controls_hvac: bool = True
    # -------- Building thermal (simple 1R1C) --------
    tau_h: float = 6.0              # h, envelope time constant
    cap_kwh_per_K: float = 3.0      # kWh/K, effective thermal capacity

    # -------- Battery defaults (scaled by GameSettings) --------
    batt_eta_ch: float = 0.95
    batt_eta_dis: float = 0.95
    batt_min_soc: float = 0.10
    batt_max_soc: float = 0.90
    batt_c_rate: float = 0.5        # max |P| = C * capacity (kW)

    # -------- HVAC Carnot params (match your sketch) --------
    hvac_sink_temp_C: float = 35.0
    hvac_source_approach_K: float = 3.0
    hvac_sink_approach_K: float = 5.0
    hvac_eta_carnot: float = 0.5
    hvac_cop_min: float = 1.0
    hvac_cop_max: float = 10.0

    # Controller defaults
    Tin_set_C: float = 22.0         # comfort setpoint
    ctrl_aggr: float = 0.35         # 0..1, how aggressively we chase the setpoint this step

    @property
    def dt_h(self) -> float:
        return float(self.dt) / 3600.0

    # ---------- Carnot helpers ----------
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
        # mirror of heating but for cooling (EER-ish)
        source_C = float(ambient_temp_C)
        sink_C   = float(self.hvac_sink_temp_C)

        T_cold_K = (source_C - self.hvac_source_approach_K) + 273.15
        T_hot_K  = (sink_C   + self.hvac_sink_approach_K)   + 273.15
        T_cold_K = max(1.0, T_cold_K)
        T_hot_K  = max(T_cold_K + 1e-3, T_hot_K)

        dT_K = T_hot_K - T_cold_K
        eer_carnot = T_cold_K / dT_K  # “cooling COP”
        eer = self.hvac_eta_carnot * eer_carnot
        eer = float(np.clip(eer, self.hvac_cop_min, self.hvac_cop_max))
        return {"cop": eer, "cop_carnot": float(eer_carnot), "T_source_C": source_C, "T_sink_C": sink_C}

    # ---------- Your HVAC dispatch pattern (heating) ----------
    def _hvac_dispatch_heat(self, ambient_temp_C: float, heat_need_kW: float,
                            max_in_kw: float) -> Dict[str, float]:
        if max_in_kw <= 0 or heat_need_kW <= 0:
            return {"hvac_elec_kW": 0.0, "hvac_heat_kW": 0.0, "cop": float("nan"),
                    "cop_carnot": float("nan"), "T_source_C": float("nan"), "T_sink_C": float("nan")}

        ci = self._carnot_heat(ambient_temp_C)
        cop = ci["cop"]
        needed_in_kW = heat_need_kW / max(cop, 1e-9)
        hvac_elec_kW = min(max_in_kw, needed_in_kW)
        hvac_heat_kW = hvac_elec_kW * cop

        return {"hvac_elec_kW": hvac_elec_kW, "hvac_heat_kW": hvac_heat_kW,
                "cop": cop, "cop_carnot": ci["cop_carnot"],
                "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}

    # ---------- Cooling version (removes heat, returns negative thermal) ----------
    def _hvac_dispatch_cool(self, ambient_temp_C: float, cool_need_kW: float,
                            max_in_kw: float) -> Dict[str, float]:
        if max_in_kw <= 0 or cool_need_kW <= 0:
            return {"hvac_elec_kW": 0.0, "hvac_heat_kW": 0.0, "cop": float("nan"),
                    "cop_carnot": float("nan"), "T_source_C": float("nan"), "T_sink_C": float("nan")}

        ci = self._carnot_cool(ambient_temp_C)
        eer = ci["cop"]
        needed_in_kW = cool_need_kW / max(eer, 1e-9)
        hvac_elec_kW = min(max_in_kw, needed_in_kW)
        hvac_cool_kW = hvac_elec_kW * eer  # positive magnitude of removed heat
        return {"hvac_elec_kW": hvac_elec_kW, "hvac_heat_kW": -hvac_cool_kW,
                "cop": eer, "cop_carnot": ci["cop_carnot"],
                "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}

    # ---------- Simple 1R1C building step ----------
    # def _thermal_step(self, T_in: float, T_out: float, q_hvac_kw: float) -> float:
    #     # dT/dt = (T_out - T_in)/tau + q_hvac / Cth
    #     if self.tau_h <= 0.0 or self.cap_kwh_per_K <= 0.0:
    #         return T_in + self.dt_h * (0.2 * (T_out - T_in) + 0.5 * q_hvac_kw)
    #     dT_dt = (T_out - T_in) / self.tau_h + (q_hvac_kw / self.cap_kwh_per_K)
    #     return T_in + dT_dt * self.dt_h
    def _thermal_step(self, T_in: float, T_out: float, q_hvac_kw: float,
                    q_solar_kw: float = 0.0, q_internal_kw: float = 0.0) -> tuple[float, dict]:
        """
        1R1C room update with optional gains.
        T_in, T_out in °C
        q_hvac_kw:  + adds heat to zone,  - removes heat (cooling)
        q_solar_kw: solar gains to zone (>=0)
        q_internal_kw: internal gains (people, appliances) (>=0)
        Returns (T_next, diagnostics)
        """
        # Map existing engine fields to a param bundle
        th = ThermalParams(
            C_th_kwh_per_degC = float(self.cap_kwh_per_K),
            U_kw_per_degC     = float(self.cap_kwh_per_K / max(self.tau_h, 1e-12)),  # U = C/tau
            dt_h              = float(self.dt_h),
            clip_temp_c       = (-40.0, 60.0),
        )

        # conductive exchange (negative when T_in > T_out → heat loss)
        q_cond_kw = th.U_kw_per_degC * (T_out - T_in)
        # net heat flow into the zone
        q_net_kw  = q_cond_kw + float(q_hvac_kw) + float(q_solar_kw) + float(q_internal_kw)

        dT = (th.dt_h / max(th.C_th_kwh_per_degC, 1e-12)) * q_net_kw
        T_next = float(np.clip(T_in + dT, th.clip_temp_c[0], th.clip_temp_c[1]))

        diag = {
            "q_cond_kw": q_cond_kw,
            "q_hvac_kw": float(q_hvac_kw),
            "q_solar_kw": float(q_solar_kw),
            "q_internal_kw": float(q_internal_kw),
            "q_net_kw": q_net_kw,
            "dT": dT,
        }
        return T_next, diag


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
            # discharge to house
            max_dis_kwh = (soc - soc_min) * cap_kwh
            max_dis_kw = max(0.0, max_dis_kwh / max(self.dt_h, 1e-12))
            p_kw = min(p_lim, max_dis_kw)
            battery_kw = +p_kw * self.batt_eta_dis
            soc_next = soc - (p_kw * self.dt_h) / cap_kwh
        else:
            # charge from grid
            max_chg_kwh = (soc_max - soc) * cap_kwh
            max_chg_kw = max(0.0, max_chg_kwh / max(self.dt_h, 1e-12))
            p_kw = min(p_lim, max_chg_kw)
            battery_kw = -p_kw / self.batt_eta_ch
            soc_next = soc + (p_kw * self.dt_h) / cap_kwh

        return {"battery_kw": float(battery_kw), "soc_next": float(np.clip(soc_next, 0.0, 1.0))}

    # ---------- Main step ----------
    def step(self, prev: GameState, action: Action, inputs: Optional[Dict[str, float]] = None) -> StepResult:
        """
        inputs (optional):
          T_outside (°C), price (€/kWh), pv_kw (+), base_load_kw (+cons),
          battery_cmd (-1/0/+1), settings (GameSettings),
          Tin_set_C, ctrl_aggr (0..1)
        """
        inputs = inputs or {}
        settings: GameSettings = inputs.get("settings") or GameSettings()

        # exogenous
        T_out = float(inputs.get("T_outside", getattr(prev, "T_outside", 15.0)))
        pv_kw = float(inputs.get("pv_kw", 0.0))
        base_load_kw = float(inputs.get("base_load_kw", 0.0))
        price = float(inputs.get("price", 0.0))

        # Diagnostics you always can compute (envelope exchange, independent of control)
        q_env_kw = (T_out - prev.T_inside) * (self.cap_kwh_per_K / max(self.tau_h, 1e-12))

        # HVAC electrical cap from settings
        hvac_elec_cap_kw = max(0.0, float(settings.hvac_size_kw))

        if self.player_controls_hvac:
            # ---- DIRECT PLAYER CONTROL (−1..1) ----
            u = float(np.clip(getattr(action, "hvac", 0.0), -1.0, 1.0))
            elec_in_kw = abs(u) * hvac_elec_cap_kw

            if u > 0.0:
                ci = self._carnot_heat(T_out)
                hvac_kw   = elec_in_kw
                q_hvac_kw = hvac_kw * ci["cop"]          # (+) adds heat
                hv = {"cop": ci["cop"], "cop_carnot": ci["cop_carnot"],
                      "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}
            elif u < 0.0:
                ci = self._carnot_cool(T_out)
                hvac_kw   = elec_in_kw
                q_hvac_kw = -hvac_kw * ci["cop"]         # (−) removes heat
                hv = {"cop": ci["cop"], "cop_carnot": ci["cop_carnot"],
                      "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}
            else:
                hvac_kw = 0.0
                q_hvac_kw = 0.0
                hv = {"cop": float("nan"), "cop_carnot": float("nan"),
                      "T_source_C": float("nan"), "T_sink_C": float("nan")}

            # make sure this exists for metrics
            q_need_kw = 0.0

        else:
            # ---- COMFORT CONTROLLER ----
            Tin_set = float(inputs.get("Tin_set_C", self.Tin_set_C))
            aggr = float(np.clip(inputs.get("ctrl_aggr", self.ctrl_aggr), 0.0, 1.0))
            q_to_reach_kw = (Tin_set - prev.T_inside) * (self.cap_kwh_per_K / max(self.dt_h, 1e-12)) * aggr
            q_need_kw = q_env_kw + q_to_reach_kw  # + = heat, − = cool

            if q_need_kw > 0:
                di = self._hvac_dispatch_heat(T_out, q_need_kw, hvac_elec_cap_kw)
            elif q_need_kw < 0:
                di = self._hvac_dispatch_cool(T_out, abs(q_need_kw), hvac_elec_cap_kw)
            else:
                di = {"hvac_elec_kW": 0.0, "hvac_heat_kW": 0.0,
                      "cop": float("nan"), "cop_carnot": float("nan"),
                      "T_source_C": float("nan"), "T_sink_C": float("nan")}

            hvac_kw   = di["hvac_elec_kW"]
            q_hvac_kw = di["hvac_heat_kW"]
            hv = {"cop": di["cop"], "cop_carnot": di["cop_carnot"],
                  "T_source_C": di["T_source_C"], "T_sink_C": di["T_sink_C"]}

        # Battery
        battery_cmd = int(inputs.get("battery_cmd", getattr(action, "battery", 0)))
        batt = self._battery_step(float(prev.soc), battery_cmd, settings)
        battery_kw = batt["battery_kw"]
        soc_next   = batt["soc_next"]

        # Electrical balance in chart convention
        other_kw = -base_load_kw
        total_kw = pv_kw + battery_kw + (-hvac_kw) + other_kw
        grid_import_kw = max(0.0, total_kw)
        kwh_from_grid = grid_import_kw * self.dt_h

        # Thermal update (optional gains)
        q_solar_kw = float(inputs.get("q_solar_kw", 0.0))
        q_internal_kw = float(inputs.get("q_internal_kw", 0.0))
        T_in_next, therm_diag = self._thermal_step(float(prev.T_inside), T_out, q_hvac_kw,
                                                   q_solar_kw=q_solar_kw,
                                                   q_internal_kw=q_internal_kw)

        # Reward calculation
        occupied = int(inputs.get("occupied_home", getattr(prev, "occupied", 0) or 0))
        reward_bits = step_reward(
            Tin_C=T_in_next,
            occupied=occupied,
            grid_kwh=kwh_from_grid,
            price_eur_per_kwh=price
        )

        # --- If your GameState does NOT have ts/occupied fields, use this simpler replace: ---
        state = replace(
            prev,
            t=float(prev.t) + self.dt,
            T_inside=T_in_next,
            T_outside=T_out,
            soc=soc_next,
            kwh_used=float(prev.kwh_used) + kwh_from_grid,
        )

        metrics = {
            "T_inside": state.T_inside,
            "T_outside": T_out,
            "pv_kw": pv_kw,
            "battery_kw": battery_kw,
            "hvac_kw": hvac_kw,           # (+) device draw; charts negate it
            "other_kw": other_kw,
            "total_kw": total_kw,
            "electricity": hvac_kw * self.dt_h,
            "price": price,
            # HVAC introspection
            "hvac_cop": hv["cop"],
            "hvac_cop_carnot": hv["cop_carnot"],
            "hvac_T_source_C": hv["T_source_C"],
            "hvac_T_sink_C": hv["T_sink_C"],
            # diagnostics (now always defined)
            "q_need_kw": q_need_kw,
            "q_env_kw": q_env_kw,
            "q_hvac_kw": q_hvac_kw,
            # thermal diag
            "q_cond_kw": therm_diag["q_cond_kw"],
            "q_solar_kw": therm_diag["q_solar_kw"],
            "q_internal_kw": therm_diag["q_internal_kw"],
            "q_net_kw": therm_diag["q_net_kw"],
            # reward metrics
            "comfort_penalty": reward_bits["comfort_penalty"],
            "opex_cost": reward_bits["opex_cost"],
            "reward": reward_bits["reward"],
        }
        
        return StepResult(state=state, metrics=metrics)
