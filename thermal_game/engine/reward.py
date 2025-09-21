# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RewardConfig:
    # comfort
    comfort_target_C: float = 22.0
    comfort_tolerance_C: float = 1.0          # no penalty within ±tol
    comfort_weight: float = 0.5               # €/ (°C over tol)^2 per step

    # tariffs
    export_tariff_ratio: float = 0.4          # export gets 40% of spot by default

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
                cfg: RewardConfig = RewardConfig()) -> dict:
    """
    Returns a dict with:
      comfort_penalty, import_cost, export_credit, net_opex, reward
    Convention: reward = +export_credit - import_cost - comfort_weight * comfort_penalty
               = -net_opex - comfort_weight * comfort_penalty
    """
    cpen = comfort_penalty(Tin_C, occupied, cfg.comfort_target_C, cfg.comfort_tolerance_C)
    terms = opex_terms(import_kwh, export_kwh,
                       import_price=price_eur_per_kwh,
                       export_price=price_eur_per_kwh * cfg.export_tariff_ratio)
    reward = -terms["net_opex"] - cfg.comfort_weight * cpen
    return {**terms, "comfort_penalty": cpen, "reward": reward}
