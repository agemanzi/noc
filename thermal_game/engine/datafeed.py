# -*- coding: utf-8 -*-
from __future__ import annotations
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- atomic row ----------
@dataclass
class DataRow:
    ts: dt.datetime
    t_index: int                # 0,1,2... (15-min steps)
    t_out_c: float              # outside temp (°C)
    price_eur_per_kwh: float    # €/kWh
    solar_gen_kw_per_kwp: float # PV gen per kWp (kW/kWp) at this step
    base_load_kw: float         # house "other" load (kW), +consumption
    occupied_home: int          # 1 = occupied, 0 = away (derived from in_work_hours)

# ---------- window for plotting “forecast” (future rows from CSV) ----------
@dataclass(frozen=True)
class ForecastWindow:
    ts: List[dt.datetime]               # timestamps for each future step
    t_out_c: List[float]
    price_eur_per_kwh: List[float]
    solar_gen_kw_per_kwp: List[float]
    base_load_kw: List[float]
    occupied_home: List[int]            # 1/0 per step
    start_index: int                    # index of first element in rows
    end_index: int                      # exclusive
    dt_seconds: int                     # cadence (900)

    @property
    def size(self) -> int:
        return self.end_index - self.start_index

class DataFeed:
    """
    Provides aligned 15-minute rows from:
      - week01_prices_weather_seasons_2025-01-01.csv (comma-sep, dot decimals)
      - load_profile.csv (semicolon-sep, comma decimals)
    Joins on timestamp (minute precision). Assumes 15-min cadence.

    Also exposes a future window (ForecastWindow) so the GUI can plot
    a “forecast” (really just future rows from the same dataset).
    """

    def __init__(self, weather_csv: Path, load_csv: Path):
        self.rows: List[DataRow] = []
        self._read(weather_csv, load_csv)
        if not self.rows:
            raise RuntimeError("DataFeed: no rows parsed")
        self.start_ts = self.rows[0].ts
        self.dt_seconds = 15 * 60
        # ---- Anchor where t=0 maps to (default: first row) -----------------
        self._ts_list = [r.ts for r in self.rows]   # sorted by _read()
        self._anchor_index = 0
        self._anchor_ts = self._ts_list[0]

    # ---- Anchor controls ----------------------------------------------------
    def set_anchor_datetime(self, anchor_ts: dt.datetime) -> int:
        """Move t=0 to the first row >= anchor_ts. Returns the anchor index."""
        from bisect import bisect_left
        i = bisect_left(self._ts_list, anchor_ts)
        if i < 0:
            i = 0
        if i >= len(self._ts_list):
            i = len(self._ts_list) - 1
        self._anchor_index = i
        self._anchor_ts = self._ts_list[i]
        return i

    def set_anchor_date(self, anchor_date: dt.date) -> int:
        """Convenience: anchor to YYYY-MM-DD 00:00."""
        return self.set_anchor_datetime(dt.datetime.combine(anchor_date, dt.time(0, 0)))

    @property
    def anchor_ts(self) -> dt.datetime:
        return self._anchor_ts

    # ---------- point access ----------
    def by_index(self, idx: int) -> DataRow:
        if idx < 0:
            idx = 0
        if idx >= len(self.rows):
            idx = len(self.rows) - 1
        return self.rows[idx]

    def by_time(self, t_seconds: float) -> DataRow:
        # Offset by anchor so t=0 corresponds to the anchor row
        idx = self._anchor_index + int(round(t_seconds / self.dt_seconds))
        return self.by_index(idx)

    # ---------- future windows for plotting ----------
    def window_by_index(self, idx: int, horizon_steps: int = 48) -> ForecastWindow:
        """Return a forward-looking slice (up to horizon) from idx (inclusive)."""
        n = len(self.rows)
        start = max(0, idx)
        end = min(n, start + max(0, horizon_steps))
        rows = self.rows[start:end]
        return ForecastWindow(
            ts=[r.ts for r in rows],
            t_out_c=[r.t_out_c for r in rows],
            price_eur_per_kwh=[r.price_eur_per_kwh for r in rows],
            solar_gen_kw_per_kwp=[r.solar_gen_kw_per_kwp for r in rows],
            base_load_kw=[r.base_load_kw for r in rows],
            occupied_home=[r.occupied_home for r in rows],
            start_index=start,
            end_index=end,
            dt_seconds=self.dt_seconds,
        )

    def window_by_time(self, t_seconds: float, horizon_steps: int = 48) -> ForecastWindow:
        """Same as window_by_index, but anchored at current sim time in seconds."""
        idx = self._anchor_index + int(round(t_seconds / self.dt_seconds))
        return self.window_by_index(idx, horizon_steps)

    # ---------- optional: whole-series arrays (useful for static plots/QA) ----------
    def all_series(self) -> ForecastWindow:
        """Return the entire dataset as a ForecastWindow (for convenience)."""
        return self.window_by_index(0, len(self.rows))

    # ---------- internals ----------
    def _read(self, weather_csv: Path, load_csv: Path):
        # 1) weather/price/solar/occupancy file
        #    Keep raw 'in_work_hours' (0/1) and map to home occupancy: occupied = 1 - in_work_hours
        wmap: Dict[dt.datetime, Tuple[float, float, float, int]] = {}
        with open(weather_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # timestamp like "2025-01-01 00:00:00"
                ts = dt.datetime.fromisoformat(row["timestamp"])
                t_out = float(row["t_out_c"])
                price = float(row["price_eur_per_kwh"])
                solar_per_kwp = float(row["solar_gen_kw_per_kwp"])
                in_work = int(row.get("in_work_hours", "0"))  # 1 = at work, 0 = not at work
                occupied = 1 - in_work                        # 1 = home occupied
                wmap[ts] = (t_out, price, solar_per_kwp, occupied)

        # 2) base load file (semicolon delim, comma decimals)
        lmap: Dict[dt.datetime, float] = {}
        with open(load_csv, "r", newline="", encoding="utf-8") as f:
            # first line often: t;datetime;consumption
            header = f.readline().strip().split(";")  # not used; keeps format explicit
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
            t_out, price, solar_per_kwp, occupied = wmap[ts]
            base_kw = lmap.get(ts, 0.0)
            self.rows.append(
                DataRow(
                    ts=ts,
                    t_index=i,
                    t_out_c=t_out,
                    price_eur_per_kwh=price,
                    solar_gen_kw_per_kwp=solar_per_kwp,
                    base_load_kw=base_kw,
                    occupied_home=occupied,
                )
            )
