# thermal_game/engine/reward.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class RewardConfig:
    # comfort
    comfort_target_C: float = 22.0
    comfort_tolerance_C: float = 1.0
    # NOTE: units are €/deg²·step (the GUI converts from €/deg²·hour via dt_h)
    comfort_weight: float = 0.5

    # tariffs
    export_tariff_ratio: float = 0.4  # export gets 40% of spot by default

def comfort_penalty(T_in: float, occupied: int,
                    target_C: float, tol_C: float) -> float:
    """Squared penalty outside the comfort band; zero if unoccupied."""
    if not occupied:
        return 0.0
    diff = abs(T_in - target_C) - tol_C
    return float(max(0.0, diff)**2)

def opex_terms(import_kwh: float, export_kwh: float,
               import_price: float, export_price: float) -> dict:
    """Return import cost (>0), export credit (>0), and net opex (=cost-credit)."""
    imp_cost = max(0.0, import_kwh) * max(0.0, import_price)
    exp_cred = max(0.0, export_kwh) * max(0.0, export_price)
    return {
        "import_cost": imp_cost,
        "export_credit": exp_cred,
        "net_opex": imp_cost - exp_cred,
    }

def step_reward(*,
                Tin_C: float,
                occupied: int,
                import_kwh: float,
                export_kwh: float,
                price_eur_per_kwh: float,
                cfg: Optional[RewardConfig] = None) -> dict:
    """
    Returns: comfort_penalty, import_cost, export_credit, net_opex,
             comfort_score, financial_score, reward_total, reward(=alias)

    Conventions (higher is better):
      financial_score = +export_credit - import_cost = -net_opex
      comfort_score   = -comfort_weight * comfort_penalty
      reward_total    = financial_score + comfort_score
    """
    cfg = cfg or RewardConfig()  # safe default (new instance)

    cpen = comfort_penalty(Tin_C, occupied, cfg.comfort_target_C, cfg.comfort_tolerance_C)
    terms = opex_terms(import_kwh, export_kwh,
                       import_price=price_eur_per_kwh,
                       export_price=price_eur_per_kwh * cfg.export_tariff_ratio)

    financial_score = -terms["net_opex"]
    comfort_score   = -cfg.comfort_weight * cpen
    reward_total    = financial_score + comfort_score

    return {
        **terms,
        "comfort_penalty": cpen,
        "financial_score": financial_score,
        "comfort_score":   comfort_score,
        "reward_total":    reward_total,
        "reward":          reward_total,  # back-compat alias
    }
