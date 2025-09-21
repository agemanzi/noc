from dataclasses import dataclass
import datetime as dt

@dataclass
class GameSettings:
    pv_size_kw: float = 4.0
    hvac_size_kw: float = 3.0
    batt_size_kwh: float = 10.0
    start_date: dt.date = dt.date.today()
