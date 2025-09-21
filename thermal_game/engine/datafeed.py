# -*- coding: utf-8 -*-
from __future__ import annotations
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

@dataclass
class DataRow:
    ts: dt.datetime
    t_index: int               # 0,1,2... (15-min steps)
    t_out_c: float             # outside temp (°C)
    price_eur_per_kwh: float   # €/kWh
    solar_gen_kw_per_kwp: float# PV gen per kWp (kW/kWp) at this step
    base_load_kw: float        # house "other" load (kW), positive=consumption

class DataFeed:
    """
    Provides aligned 15-minute rows from:
      - week01_prices_weather_seasons_2025-01-01.csv (comma-sep, dot decimals)
      - load_profile.csv (semicolon-sep, comma decimals)
    Joins on timestamp (minute precision). Assumes 15-min cadence.
    """

    def __init__(self, weather_csv: Path, load_csv: Path):
        self.rows: List[DataRow] = []
        self._read(weather_csv, load_csv)
        if not self.rows:
            raise RuntimeError("DataFeed: no rows parsed")
        self.start_ts = self.rows[0].ts
        self.dt_seconds = 15 * 60

    # ---------- public ----------
    def by_index(self, idx: int) -> DataRow:
        if idx < 0:
            idx = 0
        if idx >= len(self.rows):
            idx = len(self.rows) - 1
        return self.rows[idx]

    def by_time(self, t_seconds: float) -> DataRow:
        idx = int(round(t_seconds / self.dt_seconds))
        return self.by_index(idx)

    # ---------- internals ----------
    def _read(self, weather_csv: Path, load_csv: Path):
        # 1) weather/price/solar file
        wmap: Dict[dt.datetime, Tuple[float, float, float]] = {}
        with open(weather_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # timestamp like "2025-01-01 00:00:00"
                ts = dt.datetime.fromisoformat(row["timestamp"])
                t_out = float(row["t_out_c"])
                price = float(row["price_eur_per_kwh"])
                solar_per_kwp = float(row["solar_gen_kw_per_kwp"])
                wmap[ts] = (t_out, price, solar_per_kwp)

        # 2) base load file (semicolon delim, comma decimals)
        lmap: Dict[dt.datetime, float] = {}
        with open(load_csv, "r", newline="", encoding="utf-8") as f:
            # first line often: t;datetime;consumption
            header = f.readline().strip().split(";")
            # robust reader for the rest
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(";")
                if len(parts) < 3:
                    continue
                # e.g. "01.01.2025 0:15" with EU commas in number
                dts = parts[1]
                cons_kwh = float(parts[2].replace(",", "."))  # kWh per 15min
                ts = dt.datetime.strptime(dts, "%d.%m.%Y %H:%M")
                base_load_kw = cons_kwh / 0.25  # kWh over 15min -> kW average
                lmap[ts] = base_load_kw

        # 3) join on timestamp (only those present in weather)
        ts_sorted = sorted(wmap.keys())
        for i, ts in enumerate(ts_sorted):
            t_out, price, solar_per_kwp = wmap[ts]
            base_kw = lmap.get(ts, 0.0)
            self.rows.append(
                DataRow(
                    ts=ts,
                    t_index=i,
                    t_out_c=t_out,
                    price_eur_per_kwh=price,
                    solar_gen_kw_per_kwp=solar_per_kwp,
                    base_load_kw=base_kw,
                )
            )
