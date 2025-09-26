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

    # -------- (legacy 1R1C kept for back-compat defaults) --------
    tau_h: float = 6.0
    cap_kwh_per_K: float = 3.0

    # -------- R3C2 (air + envelope; 3 resistances: air↔env, env↔amb, air↔amb) --------
    Ci_kwh_per_K: float = 3.0       # indoor air/effective zone capacitance
    Ce_kwh_per_K: float = 20.0      # envelope capacitance (5–10x Ci is typical)
    Rie_degC_per_kW: float = 10.0   # air <-> envelope
    Rea_degC_per_kW: float = 20.0   # envelope <-> ambient (transmission)
    Ria_degC_per_kW: float = 60.0   # air <-> ambient (ventilation/infiltration)
    clip_temp_c: tuple[float, float] = (-40.0, 60.0)

    allow_grid_charge: bool = True

    # -------- Battery defaults (scaled by GameSettings) --------
    batt_eta_ch: float = 0.95
    batt_eta_dis: float = 0.95
    batt_min_soc: float = 0.10
    batt_max_soc: float = 0.90
    batt_c_rate: float = 0.5        # max |P| = C * capacity (kW)

    # -------- HVAC Carnot params --------
    hvac_sink_temp_C: float = 35.0
    hvac_source_approach_K: float = 3.0
    hvac_sink_approach_K: float = 5.0
    hvac_eta_carnot: float = 0.5
    hvac_cop_min: float = 1.0
    hvac_cop_max: float = 10.0

    # Controller defaults
    Tin_set_C: float = 22.0
    ctrl_aggr: float = 0.35
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)

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

    # ---------- HVAC dispatch ----------
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

    def _hvac_dispatch_cool(self, ambient_temp_C: float, cool_need_kW: float,
                            max_in_kw: float) -> Dict[str, float]:
        if max_in_kw <= 0 or cool_need_kW <= 0:
            return {"hvac_elec_kW": 0.0, "hvac_heat_kW": 0.0, "cop": float("nan"),
                    "cop_carnot": float("nan"), "T_source_C": float("nan"), "T_sink_C": float("nan")}
        ci = self._carnot_cool(ambient_temp_C)
        eer = ci["cop"]
        needed_in_kW = cool_need_kW / max(eer, 1e-9)
        hvac_elec_kW = min(max_in_kw, needed_in_kW)
        hvac_cool_kW = hvac_elec_kW * eer
        return {"hvac_elec_kW": hvac_elec_kW, "hvac_heat_kW": -hvac_cool_kW,
                "cop": eer, "cop_carnot": ci["cop_carnot"],
                "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}

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

    # ---------- R3C2 building step ----------
    def _thermal_step_r3c2(
        self,
        Ti: float,
        Te: float,
        Ta: float,
        q_hvac_kw: float,
        q_solar_kw: float = 0.0,
        q_internal_kw: float = 0.0,
    ) -> tuple[float, float, dict]:
        """
        3R2C grey-box (air + envelope):
          Ci * dTi/dt = (Te - Ti)/Rie + (Ta - Ti)/Ria + (q_solar + q_internal + q_hvac)
          Ce * dTe/dt = (Ti - Te)/Rie + (Ta - Te)/Rea
        Returns (Ti_next, Te_next, diagnostics)
        """
        Ci = max(1e-12, float(self.Ci_kwh_per_K))
        Ce = max(1e-12, float(self.Ce_kwh_per_K))
        Rie = max(1e-12, float(self.Rie_degC_per_kW))
        Rea = max(1e-12, float(self.Rea_degC_per_kW))
        Ria = max(1e-12, float(self.Ria_degC_per_kW))
        dt_h = self.dt_h

        # conductive/advective heat flows (kW, positive into node)
        q_ie = (Te - Ti) / Rie        # env -> air
        q_ea = (Ta - Te) / Rea        # amb -> env
        q_ia = (Ta - Ti) / Ria        # amb -> air (ventilation)
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
        }
        return Ti_next, Te_next, diag

    # ---------- Main step ----------
    def step(self, prev: GameState, action: Action, inputs: Optional[Dict[str, float]] = None) -> StepResult:
        inputs = inputs or {}
        settings: GameSettings = inputs.get("settings") or GameSettings()
        cfg: RewardConfig = inputs.get("reward_cfg", self.reward_cfg)

        # exogenous
        T_out = float(inputs.get("T_outside", getattr(prev, "T_outside", 15.0)))
        pv_kw = float(inputs.get("pv_kw", 0.0))
        base_load_kw = float(inputs.get("base_load_kw", 0.0))
        price = float(inputs.get("price", 0.0))

        # Diagnostics you always can compute (legacy)
        q_env_kw = (T_out - prev.T_inside) * (self.cap_kwh_per_K / max(self.tau_h, 1e-12))

        # HVAC electrical cap from settings
        hvac_elec_cap_kw = max(0.0, float(settings.hvac_size_kw))

        if self.player_controls_hvac:
            u = float(np.clip(getattr(action, "hvac", 0.0), -1.0, 1.0))
            elec_in_kw = abs(u) * hvac_elec_cap_kw
            if u > 0.0:
                ci = self._carnot_heat(T_out)
                hvac_kw   = elec_in_kw
                q_hvac_kw = hvac_kw * ci["cop"]
                hv = {"cop": ci["cop"], "cop_carnot": ci["cop_carnot"],
                      "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}
            elif u < 0.0:
                ci = self._carnot_cool(T_out)
                hvac_kw   = elec_in_kw
                q_hvac_kw = -hvac_kw * ci["cop"]
                hv = {"cop": ci["cop"], "cop_carnot": ci["cop_carnot"],
                      "T_source_C": ci["T_source_C"], "T_sink_C": ci["T_sink_C"]}
            else:
                hvac_kw = 0.0
                q_hvac_kw = 0.0
                hv = {"cop": float("nan"), "cop_carnot": float("nan"),
                      "T_source_C": float("nan"), "T_sink_C": float("nan")}
            q_need_kw = 0.0
        else:
            Tin_set = float(inputs.get("Tin_set_C", self.Tin_set_C))
            aggr = float(np.clip(inputs.get("ctrl_aggr", self.ctrl_aggr), 0.0, 1.0))
            q_to_reach_kw = (Tin_set - prev.T_inside) * (self.cap_kwh_per_K / max(self.dt_h, 1e-12)) * aggr
            q_need_kw = q_env_kw + q_to_reach_kw
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

        # -------- R3C2 thermal update --------
        q_solar_kw = float(inputs.get("q_solar_kw", 0.0))
        q_internal_kw = float(inputs.get("q_internal_kw", 0.0))
        T_env_prev = float(getattr(prev, "T_envelope", prev.T_inside))
        T_in_next, T_env_next, therm_diag = self._thermal_step_r3c2(
            Ti=float(prev.T_inside),
            Te=T_env_prev,
            Ta=T_out,
            q_hvac_kw=q_hvac_kw,
            q_solar_kw=q_solar_kw,
            q_internal_kw=q_internal_kw,
        )

        # Reward
        occupied = int(inputs.get("occupied_home", getattr(prev, "occupied", 0) or 0))
        reward_bits = step_reward(
            Tin_C=T_in_next,
            occupied=occupied,
            import_kwh=import_kwh,
            export_kwh=export_kwh,
            price_eur_per_kwh=price,
            cfg=cfg,
        )

        cum_reward_prev  = float(getattr(prev, "cumulative_reward", 0.0))
        cum_fin_prev     = float(getattr(prev, "cumulative_financial", 0.0))
        cum_comf_prev    = float(getattr(prev, "cumulative_comfort", 0.0))
        cum_reward_next  = cum_reward_prev + float(reward_bits["reward_total"])
        cum_fin_next     = cum_fin_prev    + float(reward_bits["financial_score"])
        cum_comf_next    = cum_comf_prev   + float(reward_bits["comfort_score"])

        # try to carry T_envelope if GameState supports it
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

            # economics / rewards
            "comfort_penalty":  reward_bits["comfort_penalty"],
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
        }

        return StepResult(state=state, metrics=metrics)