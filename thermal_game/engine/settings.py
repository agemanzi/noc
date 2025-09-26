from dataclasses import dataclass, field
import datetime as dt

@dataclass
class GameSettings:
    pv_size_kw: float = 10.0
    hvac_size_kw: float = 5
    batt_size_kwh: float = 15.0
    start_date: dt.date = field(default_factory=lambda: dt.date(2025, 3, 1))

    # Comfort + economics
    comfort_target_C: float = 21.0
    comfort_tolerance_occupied_C: float = 1.5
    comfort_tolerance_unoccupied_C: float = 3
    comfort_anchor_eur_per_deg2_hour: float = 1.5
    comfort_weight: float = 1.5
    export_tariff_ratio: float = 0.2
    comfort_inside_bonus_eur_per_step: float = 4.5

    # --- Back-compat alias (old code reads .comfort_tolerance_C) ---
    @property
    def comfort_tolerance_C(self) -> float:
        # default to the stricter occupied band
        return self.comfort_tolerance_occupied_C

    # useful helper
    def tolerance_for(self, occupied: bool) -> float:
        return (self.comfort_tolerance_occupied_C
                if occupied else self.comfort_tolerance_unoccupied_C)
