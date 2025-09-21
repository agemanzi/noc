from dataclasses import dataclass, field
import datetime as dt

@dataclass
class GameSettings:
    pv_size_kw: float = 4.0
    hvac_size_kw: float = 3.0
    batt_size_kwh: float = 6.0
    start_date: dt.date = field(default_factory=lambda: dt.date(2025, 1, 1))
    # start_date: dt.date = dt.date.today()

    export_tariff_ratio: float = 0.4   # new: feed-in vs spot
