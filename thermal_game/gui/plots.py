# -*- coding: utf-8 -*-
import collections
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl

class Charts:
    """
    Dashboard charts:
      1) Electricity profile: stacked areas for PV (+), Battery (+discharge/-charge),
         HVAC (- draw), Other load (-), plus Total balance line.
      2) Player actions: HVAC action & Battery action (-1..1) + Price on r-axis.
      3) T_inside: line with comfort band (tolerance).
      4) Weather: T_outside (°C) + Solar irradiance (W/m^2) on twin axes.
    """
    def __init__(self, root, max_points=1200, comfort=(21.0, 23.0)):
        self.max_points = max_points
        self.comfort = comfort

        # data buffers
        self.buf = {
            "t": collections.deque(maxlen=max_points),

            # electricity components (kW)
            "pv_kw": collections.deque(maxlen=max_points),
            "batt_kw": collections.deque(maxlen=max_points),   # +discharge to house, -charge from grid
            "hvac_kw": collections.deque(maxlen=max_points),   # negative = draw
            "other_kw": collections.deque(maxlen=max_points),  # negative = draw
            "total_kw": collections.deque(maxlen=max_points),  # balance = sum above

            # actions & price
            "hvac_act": collections.deque(maxlen=max_points),  # -1..1
            "batt_act": collections.deque(maxlen=max_points),  # -1..1
            "price": collections.deque(maxlen=max_points),     # price per kWh

            # temps & weather
            "Tin": collections.deque(maxlen=max_points),
            "Tout": collections.deque(maxlen=max_points),
            "solar": collections.deque(maxlen=max_points),     # W/m^2
        }

        # ---- figure/axes ----------------------------------------------------
        self.fig = Figure(figsize=(9.5, 6.8), dpi=100)

        # 2 x 2 layout
        self.ax_elec   = self.fig.add_subplot(221, title="Electricity Profile (kW)")
        self.ax_elec.axhline(0, lw=0.8, alpha=0.6)  # zero baseline
        self.ax_actions = self.fig.add_subplot(222, title="Player Actions & Price")
        self.ax_temp   = self.fig.add_subplot(223, title="Indoor Temperature (°C)")
        self.ax_weather = self.fig.add_subplot(224, title="Weather: T_out / Solar")

        # secondary axis for actions/price & weather solar
        self.ax_price = self.ax_actions.twinx()
        self.ax_solar = self.ax_weather.twinx()

        # --- fixed colors/palette for electricity stacks + total line
        palette = mpl.rcParams['axes.prop_cycle'].by_key().get(
            'color',
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8dd3c7"]
        )
        self.colors = {
            "total":    palette[0],
            "pv":       palette[1],
            "batt_pos": palette[2],
            "hvac":     palette[3],
            "other":    palette[4],
            "batt_chg": palette[5],
        }

        # total balance line (fixed color, legend handled via proxy)
        (self.l_total,) = self.ax_elec.plot([], [], lw=2, color=self.colors["total"], label="_nolegend_", zorder=5)

        # other chart lines (unchanged)
        (self.l_hvac_act,) = self.ax_actions.plot([], [], lw=1.8, label="HVAC action")
        (self.l_batt_act,) = self.ax_actions.plot([], [], lw=1.2, label="Battery action")
        (self.l_price,)    = self.ax_price.plot([], [], lw=1.5, linestyle="--", label="Price")
        (self.l_Tin,)      = self.ax_temp.plot([], [], lw=2, label="T_inside")
        (self.l_Tout,)     = self.ax_weather.plot([], [], lw=2, label="T_outside")
        (self.l_solar,)    = self.ax_solar.plot([], [], lw=1.6, linestyle="--", label="Solar (W/m²)")

        # static proxy legend for electricity panel — fixed order, fixed colors
        elec_handles = [
            Line2D([], [], lw=2, color=self.colors["total"], label="Total balance"),
            Patch(alpha=0.35, facecolor=self.colors["pv"],       label="PV (+)"),
            Patch(alpha=0.35, facecolor=self.colors["batt_pos"], label="Battery (+disch)"),
            Patch(alpha=0.35, facecolor=self.colors["hvac"],     label="HVAC (draw)"),
            Patch(alpha=0.35, facecolor=self.colors["other"],    label="Other load"),
            Patch(alpha=0.35, facecolor=self.colors["batt_chg"], label="Battery (charge)"),
        ]
        self.ax_elec.legend(handles=elec_handles, loc="upper left")

        # style basics
        for ax in (self.ax_elec, self.ax_actions, self.ax_temp, self.ax_weather):
            ax.grid(True, alpha=0.25)

        self.ax_actions.set_ylim(-1.05, 1.05)
        self.ax_actions.set_ylabel("Action (-1..1)")
        self.ax_price.set_ylabel("Price (per kWh)")
        self.ax_temp.set_ylabel("°C")
        self.ax_weather.set_ylabel("°C")
        self.ax_solar.set_ylabel("W/m²")

        # legends for other axes (unchanged)
        self.ax_actions.legend(loc="upper left")
        self.ax_price.legend(loc="upper right")
        self.ax_temp.legend(loc="upper left")
        self.ax_weather.legend(loc="upper left")
        self.ax_solar.legend(loc="upper right")

        # embed
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # nice time axes everywhere
        locator = mdates.AutoDateLocator()
        fmt = mdates.ConciseDateFormatter(locator)
        for ax in (self.ax_elec, self.ax_actions, self.ax_temp, self.ax_weather):
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(fmt)
    
        # keep last stacked polys so we can remove before replot
        self._elec_fills = []

    # --------------- public API ----------------------------------------------
    def update(self, step_or_state, metrics: dict):
        s = getattr(step_or_state, "state", step_or_state)
        if getattr(s, "t", None) is None:
            return

        B = self.buf

        ts = metrics.get("timestamp")
        if ts is not None:
            B["t"].append(mdates.date2num(ts))
        else:
            # fallback: treat elapsed seconds as offset from 00:00 today
            B["t"].append(mdates.date2num(mdates.num2date(0)) + float(getattr(s, "t", 0.0)) / 86400.0)

        # helper: sanitize missing -> np.nan
        def nz(x):
            return np.nan if x is None else float(x)
        # electricity pieces
        pv   = nz(metrics.get("pv_kw", 0.0))
        batt = nz(metrics.get("battery_kw", 0.0))
        hvac = -nz(metrics.get("hvac_kw", 0.0))   # store as negative draw
        other= nz(metrics.get("other_kw", 0.0))

        total = pv + batt + hvac + other

        B["pv_kw"].append(pv)
        B["batt_kw"].append(batt)
        B["hvac_kw"].append(hvac)
        B["other_kw"].append(other)
        B["total_kw"].append(total)

        B["hvac_act"].append(nz(metrics.get("hvac_act")))
        B["batt_act"].append(nz(metrics.get("batt_act")))
        B["price"].append(nz(metrics.get("price")))

        B["Tin"].append(getattr(s, "T_inside", metrics.get("T_inside")))
        B["Tout"].append(metrics.get("T_outside", getattr(s, "T_outside", None)))
        B["solar"].append(metrics.get("solar"))

        self._draw_electricity()
        self._draw_actions()
        self._draw_temperature()
        self._draw_weather()

        self.canvas.draw_idle()

    # --------------- drawers -------------------------------------------------
    def _draw_electricity(self):
        # remove previous fills
        for poly in self._elec_fills:
            try: poly.remove()
            except Exception: pass
        self._elec_fills.clear()

        import numpy as np
        t = np.asarray(self.buf["t"], dtype=float) 
        if t.size < 2: 
            return

        pv    = np.asarray(self.buf["pv_kw"],   dtype=float)          # + generation
        batt  = np.asarray(self.buf["batt_kw"], dtype=float)          # + discharge, - charge
        hvac  = np.asarray(self.buf["hvac_kw"], dtype=float)          # <= 0 (draw)
        other = np.asarray(self.buf["other_kw"],dtype=float)          # <= 0 (draw)
        total = np.asarray(self.buf["total_kw"],dtype=float)

        # split battery into discharge (source) vs charge (load)
        batt_pos = np.clip(batt, 0, None)          # +discharge
        batt_neg = -np.clip(batt, None, 0)         # charging magnitude as positive

        # positive stack (sources): PV + Battery discharge
        src1 = pv
        src2 = batt_pos
        s0 = np.zeros_like(t)
        s1 = s0 + src1
        s2 = s1 + src2

        # negative stack (loads): HVAC + Other + Battery charge (as loads)
        hvac_load  = -np.clip(hvac,  None, 0)      # make positive magnitudes
        other_load = -np.clip(other, None, 0)
        batt_chg   = batt_neg

        l0 = np.zeros_like(t)
        l1 = -(l0 + hvac_load)
        l2 = -(hvac_load + other_load)
        l3 = -(hvac_load + other_load + batt_chg)

        # draw stacked areas with fixed colors; suppress dynamic legend with label="_nolegend_"
        self._elec_fills += [
            self.ax_elec.fill_between(t, s0, s1, alpha=0.35, color=self.colors["pv"],       label="_nolegend_"),
            self.ax_elec.fill_between(t, s1, s2, alpha=0.35, color=self.colors["batt_pos"], label="_nolegend_"),
            self.ax_elec.fill_between(t, l0, l1, alpha=0.35, color=self.colors["hvac"],     label="_nolegend_"),
            self.ax_elec.fill_between(t, l1, l2, alpha=0.35, color=self.colors["other"],    label="_nolegend_"),
            self.ax_elec.fill_between(t, l2, l3, alpha=0.35, color=self.colors["batt_chg"], label="_nolegend_"),
        ]

        # total balance line (net export/import)
        self.l_total.set_data(t, total)

        # autoscale; do NOT rebuild legend here
        self.ax_elec.relim(); self.ax_elec.autoscale_view()

    def _draw_actions(self):
        t = np.asarray(self.buf["t"], dtype=float)
        if t.size < 2: return
        hvac_act = np.asarray(self.buf["hvac_act"], dtype=float)
        batt_act = np.asarray(self.buf["batt_act"], dtype=float)
        price    = np.asarray(self.buf["price"], dtype=float)

        self.l_hvac_act.set_data(t, hvac_act)
        self.l_batt_act.set_data(t, batt_act)
        self.l_price.set_data(t, price)

        self.ax_actions.set_ylim(-1.05, 1.05)
        
        self.ax_actions.relim(); self.ax_actions.autoscale_view(scalex=True, scaley=False)
        self.ax_price.relim();   self.ax_price.autoscale_view(scalex=True, scaley=True)
        self.ax_actions.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_actions.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    def _draw_temperature(self):
        t = np.asarray(self.buf["t"], dtype=float)
        if t.size < 2:
            return
        Tin = np.asarray(self.buf["Tin"], dtype=float)
        self.l_Tin.set_data(t, Tin)

        # comfort band
        lo, hi = self.comfort
        self._temp_band = getattr(self, "_temp_band", None)
        if self._temp_band:
            try:
                self._temp_band.remove()
            except Exception:
                pass
        self._temp_band = self.ax_temp.fill_between(
            t, lo, hi, alpha=0.15, step=None, label=f"Comfort {lo:g}–{hi:g}°C"
        )

        self.ax_temp.relim(); self.ax_temp.autoscale_view()
        self.ax_temp.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    def _draw_weather(self):
        t = np.asarray(self.buf["t"], dtype=float)
        if t.size < 2:
            return
        Tout = np.asarray(self.buf["Tout"], dtype=float)
        solar= np.asarray(self.buf["solar"], dtype=float)

        self.l_Tout.set_data(t, Tout)
        self.l_solar.set_data(t, solar)

        self.ax_weather.relim(); self.ax_weather.autoscale_view(scalex=True, scaley=True)
        self.ax_solar.relim();   self.ax_solar.autoscale_view(scalex=True, scaley=True)
        self.ax_weather.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_weather.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))