# thermal_game/engine/reward.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# HACK: Bulgarian scaling constants to match financial units
BULGARIAN_KWH_SCALING = 75.0       # 1.0 means no change; 100.0 = shift 2 decimal places
BULGARIAN_EXPORT_SCALING = 1.0      # use if export needs different scaling (optional)

@dataclass
class RewardConfig:
    # comfort - tighter bands and higher penalties for challenge
    comfort_target_C: float = 22.0
    comfort_tolerance_occupied_C: float = 0.5    # ±0.5°C when occupied
    comfort_tolerance_unoccupied_C: float = 1.0  # ±1.0°C when unoccupied  
    # NOTE: units are €/deg²·step (the GUI converts from €/deg²·hour via dt_h)
    comfort_weight: float = 1.5  # increased from 0.5 to make comfort more important

    # NEW: positive reward when inside the comfort band
    # Paid at the *center* of the band and tapers to 0 at the band edge.
    # Set to 0.0 to disable (default keeps backward compatibility).
    comfort_inside_bonus: float = 0.0  # €/step at center

    # Complaint spikes - extra penalty for persistent discomfort
    complaint_threshold_C: float = 1.0    # trigger complaint if outside this band
    complaint_penalty_per_step: float = 1.0  # extra penalty per step of complaint
    complaint_duration_threshold: int = 2     # steps before complaint triggers

    # tariffs
    export_tariff_ratio: float = 0.4  # export gets 40% of spot by default


def comfort_penalty(T_in: float, occupied: int,
                    target_C: float, tol_occupied_C: float, tol_unoccupied_C: float = None) -> float:
    """Squared penalty outside the comfort band; different tolerances for occupied vs unoccupied."""
    if tol_unoccupied_C is None:
        tol_unoccupied_C = tol_occupied_C
    
    tol_C = tol_occupied_C if occupied else tol_unoccupied_C
    diff = abs(T_in - target_C) - tol_C
    return float(max(0.0, diff)**2)


def comfort_bonus(T_in: float, occupied: int,
                  target_C: float, tol_occupied_C: float, tol_unoccupied_C: float,
                  bonus_at_center: float) -> float:
    """
    Positive reward when inside the band. Linear taper:
      dist = |T_in - target|
      if dist >= tol -> 0
      else -> bonus_at_center * (1 - dist/tol)
    Zero if unoccupied or tol <= 0 or bonus_at_center == 0.
    Uses different tolerances for occupied vs unoccupied.
    """
    if not occupied or bonus_at_center <= 0.0:
        return 0.0
    
    tol_C = tol_occupied_C if occupied else tol_unoccupied_C
    if tol_C <= 0.0:
        return 0.0
    
    dist = abs(T_in - target_C)
    if dist >= tol_C:
        return 0.0
    frac = 1.0 - (dist / tol_C)        # 1 at center, 0 at edge
    return float(bonus_at_center * max(0.0, min(1.0, frac)))


def complaint_penalty(T_in: float, occupied: int, target_C: float, 
                     complaint_threshold_C: float, complaint_count: int,
                     duration_threshold: int, penalty_per_step: float) -> tuple[float, int]:
    """
    Track consecutive steps outside complaint threshold and apply extra penalty.
    Returns (penalty, new_complaint_count)
    """
    if not occupied:
        return 0.0, 0
    
    outside_complaint_zone = abs(T_in - target_C) > complaint_threshold_C
    
    if outside_complaint_zone:
        new_complaint_count = complaint_count + 1
        if new_complaint_count >= duration_threshold:
            return penalty_per_step, new_complaint_count
        else:
            return 0.0, new_complaint_count
    else:
        return 0.0, 0  # reset complaint counter when back in zone


def opex_terms(import_kwh: float, export_kwh: float,
               import_price: float, export_price: float) -> dict:
    """Return import cost (>0), export credit (>0), and net opex (=cost-credit)."""
    imp_cost = max(0.0, import_kwh) * max(0.0, import_price) * BULGARIAN_KWH_SCALING
    exp_cred = max(0.0, export_kwh) * max(0.0, export_price) * BULGARIAN_KWH_SCALING * BULGARIAN_EXPORT_SCALING
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
                cfg: Optional[RewardConfig] = None,
                complaint_count: int = 0) -> dict:
    """
    Returns: comfort_penalty, comfort_bonus, complaint_penalty, import_cost, export_credit, net_opex,
             comfort_score, financial_score, reward_total, reward(=alias), new_complaint_count

    Conventions (higher is better):
      financial_score = +export_credit - import_cost = -net_opex
      comfort_score   = -comfort_weight * comfort_penalty + comfort_bonus - complaint_penalty
      reward_total    = financial_score + comfort_score
    """
    cfg = cfg or RewardConfig()  # safe default (new instance)

    cpen = comfort_penalty(Tin_C, occupied, cfg.comfort_target_C, 
                          cfg.comfort_tolerance_occupied_C, cfg.comfort_tolerance_unoccupied_C)
    cbon = comfort_bonus(Tin_C, occupied,
                         cfg.comfort_target_C, 
                         cfg.comfort_tolerance_occupied_C, cfg.comfort_tolerance_unoccupied_C,
                         cfg.comfort_inside_bonus)
    
    # Handle complaint penalty
    complaint_pen, new_complaint_count = complaint_penalty(
        Tin_C, occupied, cfg.comfort_target_C,
        cfg.complaint_threshold_C, complaint_count,
        cfg.complaint_duration_threshold, cfg.complaint_penalty_per_step
    )

    terms = opex_terms(import_kwh, export_kwh,
                       import_price=price_eur_per_kwh,
                       export_price=price_eur_per_kwh * cfg.export_tariff_ratio)

    financial_score = -terms["net_opex"]
    comfort_score   = -cfg.comfort_weight * cpen + cbon - complaint_pen
    reward_total    = financial_score + comfort_score

    return {
        **terms,
        "comfort_penalty": cpen,
        "comfort_bonus":   cbon,
        "complaint_penalty": complaint_pen,
        "new_complaint_count": new_complaint_count,
        "financial_score": financial_score,
        "comfort_score":   comfort_score,
        "reward_total":    reward_total,
        "reward":          reward_total,  # back-compat alias
    }

# # thermal_game/engine/reward.py
# # -*- coding: utf-8 -*-
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Optional

# @dataclass
# class RewardConfig:
#     # comfort
#     comfort_target_C: float = 22.0
#     comfort_tolerance_C: float = 1.0
#     # NOTE: units are €/deg²·step (the GUI converts from €/deg²·hour via dt_h)
#     comfort_weight: float = 0.5

#     # tariffs
#     export_tariff_ratio: float = 0.4  # export gets 40% of spot by default

# def comfort_penalty(T_in: float, occupied: int,
#                     target_C: float, tol_C: float) -> float:
#     """Squared penalty outside the comfort band; zero if unoccupied."""
#     if not occupied:
#         return 0.0
#     diff = abs(T_in - target_C) - tol_C
#     return float(max(0.0, diff)**2)

# def opex_terms(import_kwh: float, export_kwh: float,
#                import_price: float, export_price: float) -> dict:
#     """Return import cost (>0), export credit (>0), and net opex (=cost-credit)."""
#     imp_cost = max(0.0, import_kwh) * max(0.0, import_price)
#     exp_cred = max(0.0, export_kwh) * max(0.0, export_price)
#     return {
#         "import_cost": imp_cost,
#         "export_credit": exp_cred,
#         "net_opex": imp_cost - exp_cred,
#     }

# def step_reward(*,
#                 Tin_C: float,
#                 occupied: int,
#                 import_kwh: float,
#                 export_kwh: float,
#                 price_eur_per_kwh: float,
#                 cfg: Optional[RewardConfig] = None) -> dict:
#     """
#     Returns: comfort_penalty, import_cost, export_credit, net_opex,
#              comfort_score, financial_score, reward_total, reward(=alias)

#     Conventions (higher is better):
#       financial_score = +export_credit - import_cost = -net_opex
#       comfort_score   = -comfort_weight * comfort_penalty
#       reward_total    = financial_score + comfort_score
#     """
#     cfg = cfg or RewardConfig()  # safe default (new instance)

#     cpen = comfort_penalty(Tin_C, occupied, cfg.comfort_target_C, cfg.comfort_tolerance_C)
#     terms = opex_terms(import_kwh, export_kwh,
#                        import_price=price_eur_per_kwh,
#                        export_price=price_eur_per_kwh * cfg.export_tariff_ratio)

#     financial_score = -terms["net_opex"]
#     comfort_score   = -cfg.comfort_weight * cpen
#     reward_total    = financial_score + comfort_score

#     return {
#         **terms,
#         "comfort_penalty": cpen,
#         "financial_score": financial_score,
#         "comfort_score":   comfort_score,
#         "reward_total":    reward_total,
#         "reward":          reward_total,  # back-compat alias
#     }
