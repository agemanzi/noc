# thermal_game/engine/settings.py
from dataclasses import dataclass, field
import datetime as dt

@dataclass
class GameSettings:
    pv_size_kw: float = 4.0
    hvac_size_kw: float = 5.0
    batt_size_kwh: float = 6.0
    start_date: dt.date = field(default_factory=lambda: dt.date(2025, 1, 1))

    # Comfort + economics
    comfort_target_C: float = 22.0
    comfort_tolerance_C: float = 3.0
    # NEW: “price tag” for comfort error (€/deg²·hour)
    comfort_anchor_eur_per_deg2_hour: float = 0.50
    comfort_weight: float = 0.5  # will be overwritten from anchor; keep for back-compat if needed
    export_tariff_ratio: float = 0.4
