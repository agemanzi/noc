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
    Dashboard charts with triple reward lines on the T plot:
      - reward_fin  (€/step)  : financial score
      - reward_comf (€/step)  : comfort score (usually ≤ 0)
      - reward      (€/step)  : total = fin + comf
    """
    def __init__(self, root, max_points=1200, comfort=(21.0, 23.0)):
        self.max_points = max_points
        self.comfort = comfort
        self.occ_temp_style = "ribbon"   # or "hatch" or "dots"
        self._occ_temp_artists = []      # holder for cleanup
        # live buffers
        self.buf = {
            "t": collections.deque(maxlen=max_points),
            "pv_kw": collections.deque(maxlen=max_points),
            "batt_kw": collections.deque(maxlen=max_points),
            "hvac_kw": collections.deque(maxlen=max_points),
            "other_kw": collections.deque(maxlen=max_points),
            "total_kw": collections.deque(maxlen=max_points),
            "hvac_act": collections.deque(maxlen=max_points),
            "batt_act": collections.deque(maxlen=max_points),
            "price": collections.deque(maxlen=max_points),
            "occupied": collections.deque(maxlen=max_points),
            "Tin": collections.deque(maxlen=max_points),
            "Te": collections.deque(maxlen=max_points),
            "Tout": collections.deque(maxlen=max_points),
            "solar": collections.deque(maxlen=max_points),
            # rewards (all €/step)
            "reward_fin": collections.deque(maxlen=max_points),
            "reward_comf": collections.deque(maxlen=max_points),
            "reward": collections.deque(maxlen=max_points),
            #SOC
            "soc": collections.deque(maxlen=max_points),
        }

        # figure & axes
        self.fig = Figure(figsize=(9.5, 6.8), dpi=100, constrained_layout=True)
        self.ax_elec    = self.fig.add_subplot(221, title="Electricity Profile (kW)")
        self.ax_actions = self.fig.add_subplot(222, title="Player Actions & Price", sharex=self.ax_elec)
        self.ax_temp    = self.fig.add_subplot(223, title="Indoor Temperature (°C)", sharex=self.ax_elec)
        self.ax_weather = self.fig.add_subplot(224, title="Weather: T_out / Solar", sharex=self.ax_elec)
        self.ax_elec.axhline(0, lw=0.8, alpha=0.6)

        # twin axes
        self.ax_price  = self.ax_actions.twinx()
        self.ax_solar  = self.ax_weather.twinx()
        self.ax_reward = self.ax_temp.twinx()
        self.ax_reward.set_frame_on(True)
        self.ax_reward.patch.set_visible(False)
        self.ax_temp.set_zorder(2); self.ax_reward.set_zorder(3); self.ax_temp.patch.set_visible(False)

        # colors
        palette = mpl.rcParams['axes.prop_cycle'].by_key().get(
            'color', ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8dd3c7"]
        )
        self.colors = {
            "total": palette[0], "pv": palette[1], "batt_pos": palette[2],
            "hvac": palette[3], "other": palette[4], "batt_chg": palette[5],
            "forecast": "#444444", "occ_band": "#94a3b8",
            "solar_area": "#fbbf24", "comfort_band": "#60a5fa",
            # reward lines
            "reward_fin": "#16a34a",    # green-ish
            "reward_comf": "#ef4444",   # red-ish
            "reward_tot": "#1f2937",    # dark gray
            "soc": "#7c3aed",          # purple for SOC
        }

        # live lines
        (self.l_total,)    = self.ax_elec.plot([], [], lw=2, color=self.colors["total"], label="_nolegend_", zorder=5)
        (self.l_hvac_act,) = self.ax_actions.plot([], [], lw=1.8, label="HVAC action")
        (self.l_batt_act,) = self.ax_actions.plot([], [], lw=1.2, label="Battery action")
        (self.l_price,)    = self.ax_price.plot([], [], lw=1.5, color="green", label="Price")
        (self.l_Tin,)      = self.ax_temp.plot([], [], lw=2, label="T_inside")
        (self.l_Te,)       = self.ax_temp.plot([], [], lw=1.5, linestyle="--", alpha=0.8, label="T_envelope")

        (self.l_soc,)    = self.ax_actions.plot(
            [], [], linestyle="--", alpha=0.8, lw=1.2,
            color=self.colors["soc"], label="SOC (0..1)"
        )


        # reward lines on right axis (three of them)
        (self.l_reward_fin,)  = self.ax_reward.plot([], [], lw=1.6, label="Financial", color=self.colors["reward_fin"])
        (self.l_reward_comf,) = self.ax_reward.plot([], [], lw=1.6, label="Comfort",   color=self.colors["reward_comf"])
        (self.l_reward_tot,)  = self.ax_reward.plot([], [], lw=2.0, label="Total",     color=self.colors["reward_tot"])

        (self.l_Tout,)    = self.ax_weather.plot([], [], lw=2, label="T_outside")
        (self.l_solar,)   = self.ax_solar.plot([], [], lw=1.6, linestyle="--",
                                               color=self.colors["solar_area"], label="Solar (W/m²)")
        self._solar_fill = None
        self._temp_band  = None
        self._occ_fill   = None
        self._occ_fill_fc= None

        # forecast lines
        (self.l_price_fc,) = self.ax_price.plot([], [], lw=1.2, linestyle=":", alpha=0.9,
                                                color=self.colors["forecast"], label="Price (next 12h)")
        (self.l_Tout_fc,)  = self.ax_weather.plot([], [], lw=1.2, linestyle=":", alpha=0.9,
                                                color=self.colors["forecast"], label="T_out (next 12h)")
        (self.l_solar_fc,) = self.ax_solar.plot([], [], lw=1.1, linestyle=":", alpha=0.9,
                                                color=self.colors["forecast"], label="Solar (next 12h)")

        # electricity legend + initial y range
        elec_handles = [
            Line2D([], [], lw=2, color=self.colors["total"], label="Total balance"),
            Patch(alpha=0.35, facecolor=self.colors["pv"],       label="PV (+)"),
            Patch(alpha=0.35, facecolor=self.colors["batt_pos"], label="Battery (+disch)"),
            Patch(alpha=0.35, facecolor=self.colors["hvac"],     label="HVAC (draw)"),
            Patch(alpha=0.35, facecolor=self.colors["other"],    label="Other load"),
            Patch(alpha=0.35, facecolor=self.colors["batt_chg"], label="Battery (charge)"),
        ]
        self.ax_elec.legend(handles=elec_handles, loc="upper left")
        self.ax_elec.set_ylim(-10, 10)

        # styles & labels
        for ax in (self.ax_elec, self.ax_actions, self.ax_temp, self.ax_weather):
            ax.grid(True, alpha=0.25)
        self.ax_actions.set_ylim(-1.05, 1.05)
        self.ax_temp.set_ylim(15, 30)
        self.ax_actions.set_ylabel("Action (-1..1)")
        self.ax_price.set_ylabel("Price (€/kWh)")
        self.ax_temp.set_ylabel("°C")
        self.ax_reward.set_ylabel("Reward (€/step)")
        self.ax_weather.set_ylabel("°C")
        self.ax_solar.set_ylabel("W/m²")
        self.ax_reward.tick_params(axis="y", labelcolor="#374151")

        # legends
        self.ax_actions.legend(loc="upper left")
        self.ax_price.legend(loc="upper right")
        self.ax_temp.legend(loc="upper left")
        self.ax_reward.legend(loc="upper right")
        self.ax_weather.legend(loc="upper left")
        self.ax_solar.legend(loc="upper right")

        # embed + time axes
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        locator = mdates.AutoDateLocator(); fmt = mdates.ConciseDateFormatter(locator)
        for ax in (self.ax_elec, self.ax_actions, self.ax_temp, self.ax_weather):
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(fmt)

        self._elec_fills = []
        self._show_rewards = True  # default

        # ---- X-axis windowing (2-day chunks that step by 1 day) ----
        self.window_days = 2               # visible width
        self.window_step_days = 1          # jump size when window is "full"
        self.snap_window_to_midnight = True
        self._xwin_start = None            # matplotlib date number of left edge

    def set_show_rewards(self, show: bool):
        self._show_rewards = bool(show)

        # toggle lines
        for ln in (self.l_reward_fin, self.l_reward_comf, self.l_reward_tot):
            ln.set_visible(self._show_rewards)

        # toggle right axis visuals (ticks, label, spine, legend)
        self.ax_reward.yaxis.set_visible(self._show_rewards)
        self.ax_reward.set_ylabel("Reward (€/step)" if self._show_rewards else "")
        self.ax_reward.spines["right"].set_visible(self._show_rewards)

        # remove legend if hidden (avoid empty box)
        leg = self.ax_reward.get_legend()
        if leg:
            leg.set_visible(self._show_rewards)
            if not self._show_rewards:
                # Matplotlib sometimes still reserves space; removing is safer
                try:
                    leg.remove()
                except Exception:
                    pass
        elif self._show_rewards:
            # restore legend when re-enabled
            self.ax_reward.legend(loc="upper right")

        # redraw now
        self.canvas.draw_idle()

    def _days_to_num(self, d):
        import matplotlib.dates as mdates
        return d / 1.0  # already in date-num units (kept for clarity)

    def _floor_to_midnight_num(self, dnum):
        import matplotlib.dates as mdates
        dt = mdates.num2date(dnum).replace(hour=0, minute=0, second=0, microsecond=0)
        return mdates.date2num(dt)

    def _set_all_xlim(self, x0, x1):
        # set on primary axes (sharex takes care of some, but be explicit)
        for ax in (self.ax_elec, self.ax_actions, self.ax_temp, self.ax_weather,
                   self.ax_price, self.ax_solar, self.ax_reward):
            try:
                ax.set_xlim(x0, x1)
            except Exception:
                pass

    def _maybe_update_xwindow(self, latest_dnum):
        """Ensure a fixed 2-day window that steps forward by 1 day when needed."""
        import matplotlib.dates as mdates
        if latest_dnum is None:
            return

        # Initialize left edge on first sample
        if self._xwin_start is None:
            start = latest_dnum
            if self.snap_window_to_midnight:
                start = self._floor_to_midnight_num(latest_dnum)
            self._xwin_start = start
            self._set_all_xlim(self._xwin_start, self._xwin_start + self.window_days)
            return

        # Advance window in 1-day hops if latest point is past current window
        right_edge = self._xwin_start + self.window_days
        step = self.window_step_days
        while latest_dnum >= right_edge:
            self._xwin_start += step
            right_edge = self._xwin_start + self.window_days

        # Apply limits (no autoscale jitter)
        self._set_all_xlim(self._xwin_start, right_edge)

    def reset_time_axes(self, clear_buffers: bool = True):
        """
        Clear cached x/y data and force Matplotlib to recompute limits/formatters.
        Call this whenever the datafeed anchor changes (i.e., start date changes).
        """
        # 1) clear buffers so we don't mix old and new dates
        if clear_buffers:
            for k in self.buf:
                self.buf[k].clear()

        # 1.5) reset window state
        self._xwin_start = None

        # 2) clear all line/patch artists' data
        for ln in (
            self.l_total, self.l_hvac_act, self.l_batt_act, self.l_price, self.l_soc,
            self.l_Tin, self.l_Tout, self.l_solar,
            self.l_price_fc, self.l_Tout_fc, self.l_solar_fc,
            self.l_reward_fin, self.l_reward_comf, self.l_reward_tot
        ):
            ln.set_data([], [])

        # remove fills/bands if present
        for poly in getattr(self, "_elec_fills", []):
            try: poly.remove()
            except Exception: pass
        self._elec_fills = []

        if getattr(self, "_solar_fill", None):
            try: self._solar_fill.remove()
            except Exception: pass
        self._solar_fill = None

        if getattr(self, "_temp_band", None):
            try: self._temp_band.remove()
            except Exception: pass
        self._temp_band = None

        if getattr(self, "_occ_fill", None):
            try: self._occ_fill.remove()
            except Exception: pass
        self._occ_fill = None

        if getattr(self, "_occ_fill_fc", None):
            try: self._occ_fill_fc.remove()
            except Exception: pass
        self._occ_fill_fc = None

        for art in getattr(self, "_occ_temp_artists", []):
            try: art.remove()
            except Exception: pass
        self._occ_temp_artists = []

        # 3) hard reset limits so mpl recomputes from fresh data
        for ax in (self.ax_elec, self.ax_actions, self.ax_temp, self.ax_weather,
                   self.ax_price, self.ax_solar, self.ax_reward):
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)

        # re-apply any preferred y ranges after reset
        self.ax_actions.set_ylim(-1.05, 1.05)
        self.ax_temp.set_ylim(15, 30)
        self.ax_elec.set_ylim(-10, 10)

        # ensure date locator/formatter is still set - use daily ticks for 2-day window
        day_locator = mdates.DayLocator(interval=1)
        day_fmt = mdates.ConciseDateFormatter(day_locator)
        for ax in (self.ax_elec, self.ax_actions, self.ax_temp, self.ax_weather):
            ax.xaxis.set_major_locator(day_locator)
            ax.xaxis.set_major_formatter(day_fmt)

        # keep reward visibility state after reset
        self.set_show_rewards(self._show_rewards)

        # redraw
        self.canvas.draw_idle()

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
            B["t"].append(mdates.date2num(mdates.num2date(0)) + float(getattr(s, "t", 0.0)) / 86400.0)

        # Update time window after appending timestamp
        self._maybe_update_xwindow(B["t"][-1] if B["t"] else None)

        def nz(x):  # sanitize None -> nan, else float
            return np.nan if x is None else float(x)

        # electricity pieces (chart convention)
        pv   = nz(metrics.get("pv_kw", 0.0))
        batt = nz(metrics.get("battery_kw", 0.0))
        hvac = -nz(metrics.get("hvac_kw", 0.0))   # store as negative draw
        other= nz(metrics.get("other_kw", 0.0))
        total = pv + batt + hvac + other

        B["pv_kw"].append(pv); B["batt_kw"].append(batt)
        B["hvac_kw"].append(hvac); B["other_kw"].append(other)
        B["total_kw"].append(total)

        B["hvac_act"].append(nz(metrics.get("hvac_act")))
        B["batt_act"].append(nz(metrics.get("batt_act")))
        B["price"].append(nz(metrics.get("price")))
        B["occupied"].append(nz(metrics.get("occupied")))
        B["soc"].append(float(getattr(s, "soc", metrics.get("soc", np.nan))))


        B["Tin"].append(getattr(s, "T_inside", metrics.get("T_inside")))
        B["Te"].append(getattr(s, "T_envelope", metrics.get("T_envelope", getattr(s, "T_inside", 22.0))))
        B["Tout"].append(metrics.get("T_outside", getattr(s, "T_outside", None)))
        B["solar"].append(metrics.get("solar"))

        # rewards: accept either explicit values or fallbacks
        # (fallback for fin: -net_opex; for comf: reward - fin)
        r_tot = metrics.get("reward", np.nan)
        r_fin = metrics.get("reward_fin", -float(metrics.get("net_opex", 0.0)) if r_tot is not np.nan else np.nan)
        r_com = metrics.get("reward_comf",
                            (float(r_tot) + float(metrics.get("net_opex", 0.0))) if r_tot is not np.nan else np.nan)

        B["reward"].append(nz(r_tot))
        B["reward_fin"].append(nz(r_fin))
        B["reward_comf"].append(nz(r_com))

        # optional forecast payload
        self._forecast = metrics.get("forecast", None)

        self._draw_electricity()
        self._draw_actions()
        self._draw_temperature()
        self._draw_weather()
        self.canvas.draw_idle()

    # --------------- drawers -------------------------------------------------
    def _draw_electricity(self):
        for poly in self._elec_fills:
            try: poly.remove()
            except Exception: pass
        self._elec_fills.clear()

        t = np.asarray(self.buf["t"], dtype=float)
        if t.size < 2:
            return

        pv    = np.asarray(self.buf["pv_kw"],   dtype=float)
        batt  = np.asarray(self.buf["batt_kw"], dtype=float)
        hvac  = np.asarray(self.buf["hvac_kw"], dtype=float)
        other = np.asarray(self.buf["other_kw"],dtype=float)
        total = np.asarray(self.buf["total_kw"],dtype=float)

        batt_pos = np.clip(batt, 0, None)
        batt_neg = -np.clip(batt, None, 0)

        s0 = np.zeros_like(t); s1 = s0 + pv; s2 = s1 + batt_pos
        hvac_load  = -np.clip(hvac,  None, 0)
        other_load = -np.clip(other, None, 0)
        batt_chg   = batt_neg
        l0 = np.zeros_like(t)
        l1 = -(l0 + hvac_load)
        l2 = -(hvac_load + other_load)
        l3 = -(hvac_load + other_load + batt_chg)

        self._elec_fills += [
            self.ax_elec.fill_between(t, s0, s1, alpha=0.35, color=self.colors["pv"],       label="_nolegend_"),
            self.ax_elec.fill_between(t, s1, s2, alpha=0.35, color=self.colors["batt_pos"], label="_nolegend_"),
            self.ax_elec.fill_between(t, l0, l1, alpha=0.35, color=self.colors["hvac"],     label="_nolegend_"),
            self.ax_elec.fill_between(t, l1, l2, alpha=0.35, color=self.colors["other"],    label="_nolegend_"),
            self.ax_elec.fill_between(t, l2, l3, alpha=0.35, color=self.colors["batt_chg"], label="_nolegend_"),
        ]
        self.l_total.set_data(t, total)

        self.ax_elec.relim(); self.ax_elec.autoscale_view()
        ymin, ymax = self.ax_elec.get_ylim()
        self.ax_elec.set_ylim(min(ymin, -10), max(ymax, 10))

    def _draw_actions(self):
        t = np.asarray(self.buf["t"], dtype=float)
        if t.size < 2:
            return
        hvac_act = np.asarray(self.buf["hvac_act"], dtype=float)
        batt_act = np.asarray(self.buf["batt_act"], dtype=float)
        price    = np.asarray(self.buf["price"], dtype=float)
        occ      = np.asarray(self.buf["occupied"], dtype=float)
        soc      = np.asarray(self.buf["soc"], dtype=float)  # NEW
        self.l_hvac_act.set_data(t, hvac_act)
        self.l_batt_act.set_data(t, batt_act)
        self.l_price.set_data(t, price)
        self.l_soc.set_data(t, soc)  # NEW
        y0, y1 = -1.05, -0.95
        if self._occ_fill:
            try: self._occ_fill.remove()
            except Exception: pass
            self._occ_fill = None
        self._occ_fill = self.ax_actions.fill_between(t, y0, y1, where=(occ >= 0.5),
                                                      alpha=0.25, color=self.colors["occ_band"],
                                                      label="_nolegend_")

        if self._forecast:
            try:
                tf = mdates.date2num(np.asarray(self._forecast["ts"]))
            except Exception:
                tf = np.asarray(self._forecast.get("ts", []), dtype=float)
            price_f = np.asarray(self._forecast.get("price", []), dtype=float)
            occ_f   = np.asarray(self._forecast.get("occupied", []), dtype=float)
            self.l_price_fc.set_data(tf, price_f)

            if self._occ_fill_fc:
                try: self._occ_fill_fc.remove()
                except Exception: pass
                self._occ_fill_fc = None
            if tf.size and occ_f.size:
                self._occ_fill_fc = self.ax_actions.fill_between(tf, y0, y1, where=(occ_f >= 0.5),
                                                                 alpha=0.15, color=self.colors["occ_band"],
                                                                 label="_nolegend_")
        else:
            self.l_price_fc.set_data([], [])
            if self._occ_fill_fc:
                try: self._occ_fill_fc.remove()
                except Exception: pass
                self._occ_fill_fc = None

        self.ax_actions.set_ylim(-1.05, 1.05)
        self.ax_actions.relim(); self.ax_actions.autoscale_view(scalex=True, scaley=False)
        self.ax_price.relim();   self.ax_price.autoscale_view(scalex=True, scaley=True)

    def _draw_temperature(self):
        t = np.asarray(self.buf["t"], dtype=float)
        if t.size < 2:
            return

        # Buffers
        Tin  = np.asarray(self.buf["Tin"],          dtype=float)
        r_fin= np.asarray(self.buf["reward_fin"],   dtype=float)
        r_com= np.asarray(self.buf["reward_comf"],  dtype=float)
        r_tot= np.asarray(self.buf["reward"],       dtype=float)
        occ  = np.asarray(self.buf["occupied"],     dtype=float)

        # Align series by time length
        n  = min(t.size, Tin.size)
        tt, Tin_t = t[-n:], Tin[-n:]
        self.l_Tin.set_data(tt, Tin_t)
        
        # T_envelope
        Te   = np.asarray(self.buf["Te"], dtype=float)
        n_te = min(t.size, Te.size)
        tt_te, Te_t = t[-n_te:], Te[-n_te:]
        self.l_Te.set_data(tt_te, Te_t)

        # One-time attrs
        if not hasattr(self, "_styled_reward_lines"):
            self._styled_reward_lines = False
        if not hasattr(self, "occ_temp_expand"):
            self.occ_temp_expand = 1.0  # °C widen vs comfort band

        # --- OCCUPANCY CUE ON Tin AXIS -------------------------------------------
        # cleanup previous artists
        for art in getattr(self, "_occ_temp_artists", []):
            try: art.remove()
            except Exception: pass
        self._occ_temp_artists = []

        lo, hi = self.comfort
        band_lo = lo - float(self.occ_temp_expand)
        band_hi = hi + float(self.occ_temp_expand)

        occ_t = occ[-n:] if occ.size else np.array([])

        if occ_t.size:
            if self.occ_temp_style == "ribbon":
                # thin ribbon near the top of the axis
                ymin, ymax = self.ax_temp.get_ylim()
                span = ymax - ymin if ymax > ymin else 1.0
                y0 = ymax - 0.10 * span
                y1 = ymax - 0.03 * span
                band = self.ax_temp.fill_between(
                    tt, y0, y1, where=(occ_t >= 0.5),
                    alpha=0.25, color=self.colors["occ_band"], edgecolor="none",
                    zorder=0.25, label="_nolegend_"
                )
                self._occ_temp_artists.append(band)

            elif self.occ_temp_style == "hatch":
                # hatch overlay only where occupied, on top of comfort band
                hatch = self.ax_temp.fill_between(
                    tt, band_lo, band_hi, where=(occ_t >= 0.5),
                    facecolor="none", edgecolor=self.colors["occ_band"],
                    hatch="////", linewidth=0.0, alpha=0.35,
                    zorder=0.35, label="_nolegend_"
                )
                self._occ_temp_artists.append(hatch)

            elif self.occ_temp_style == "dots":
                # dot rail slightly above axis bottom
                ymin, ymax = self.ax_temp.get_ylim()
                rail_y = ymin + 0.06 * (ymax - ymin)
                mask = occ_t >= 0.5
                dots = self.ax_temp.scatter(
                    tt[mask], np.full(mask.sum(), rail_y),
                    s=12, marker="o", alpha=0.7, color=self.colors["occ_band"],
                    zorder=2.2
                )
                self._occ_temp_artists.append(dots)

            else:
                # default: bigger comfort-like band when occupied
                band = self.ax_temp.fill_between(
                    tt, band_lo, band_hi, where=(occ_t >= 0.5),
                    alpha=0.22, color=self.colors["occ_band"], edgecolor="none",
                    zorder=0.3, label="_nolegend_"
                )
                self._occ_temp_artists.append(band)

        # --- comfort band (draw above occ cue so it's always visible)
        if getattr(self, "_temp_band", None):
            try: self._temp_band.remove()
            except Exception: pass
            self._temp_band = None

        if n > 0:
            self._temp_band = self.ax_temp.fill_between(
                tt, lo, hi, alpha=0.18, facecolor=self.colors["comfort_band"],
                edgecolor="none", label="_nolegend_", zorder=0.4
            )

        # --- reward lines (right axis)
        if not self._styled_reward_lines:
            for ln in (self.l_reward_fin, self.l_reward_comf, self.l_reward_tot):
                ln.set_linestyle("--")
                ln.set_linewidth(1.2)
                ln.set_alpha(0.8)
            self._styled_reward_lines = True

        def _set_series(line, series):
            k = min(t.size, series.size)
            if k > 0:
                line.set_data(t[-k:], series[-k:])
            else:
                line.set_data([], [])

        _set_series(self.l_reward_fin,  r_fin)
        _set_series(self.l_reward_comf, r_com)
        _set_series(self.l_reward_tot,  r_tot)

        # --- autoscale
        self.ax_temp.relim();    self.ax_temp.autoscale_view(scalex=True,  scaley=True)
        ymin, ymax = self.ax_temp.get_ylim()
        self.ax_temp.set_ylim(min(ymin, 15.0), max(ymax, 30.0))

        self.ax_reward.relim();  self.ax_reward.autoscale_view(scalex=True, scaley=True)

        # --- legends
        self.ax_temp.legend(loc="upper left")
        self.ax_reward.legend(loc="upper right")

        # --- time axis formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        self.ax_temp.xaxis.set_major_locator(locator)
        self.ax_temp.xaxis.set_major_formatter(formatter)
        self.ax_reward.xaxis.set_major_locator(locator)
        self.ax_reward.xaxis.set_major_formatter(formatter)


    def _draw_weather(self):
        t = np.asarray(self.buf["t"], dtype=float)
        if t.size < 2:
            return

        Tout  = np.asarray(self.buf["Tout"],  dtype=float)
        solar = np.asarray(self.buf["solar"], dtype=float)

        n = min(t.size, Tout.size)
        tt, TT = t[-n:], Tout[-n:]
        self.l_Tout.set_data(tt, TT)

        ns = min(t.size, solar.size)
        t_s, s_s = t[-ns:], solar[-ns:]
        self.l_solar.set_data(t_s, s_s)

        if self._solar_fill:
            try: self._solar_fill.remove()
            except Exception: pass
            self._solar_fill = None
        if ns > 0:
            self._solar_fill = self.ax_solar.fill_between(t_s, 0.0, s_s, alpha=0.25,
                                                          color=self.colors["solar_area"], label="_nolegend_")

        if self._forecast:
            try:
                tf = mdates.date2num(np.asarray(self._forecast["ts"]))
            except Exception:
                tf = np.asarray(self._forecast.get("ts", []), dtype=float)
            Tout_f  = np.asarray(self._forecast.get("t_out_c", []), dtype=float)
            solar_f = np.asarray(self._forecast.get("solar_per_kwp", []), dtype=float) * 1000.0
            self.l_Tout_fc.set_data(tf, Tout_f)
            self.l_solar_fc.set_data(tf, solar_f)
        else:
            self.l_Tout_fc.set_data([], [])
            self.l_solar_fc.set_data([], [])

        self.ax_weather.relim(); self.ax_weather.autoscale_view(scalex=True, scaley=True)
        self.ax_solar.relim();   self.ax_solar.autoscale_view(scalex=True, scaley=True)

    def _compute_next_sample(self, step_or_state, metrics):
        """Return the single-sample values we would append for `update()`."""
        s = getattr(step_or_state, "state", step_or_state)
        def nz(x):
            import numpy as np
            return np.nan if x is None else float(x)

        # time
        import matplotlib.dates as mdates
        if metrics.get("timestamp") is not None:
            t_new = mdates.date2num(metrics["timestamp"])
        else:
            t_new = mdates.date2num(mdates.num2date(0)) + float(getattr(s, "t", 0.0)) / 86400.0

        # electricity pieces
        pv   = nz(metrics.get("pv_kw", 0.0))
        batt = nz(metrics.get("battery_kw", 0.0))
        hvac = -nz(metrics.get("hvac_kw", 0.0))   # chart convention: draws negative
        other= nz(metrics.get("other_kw", 0.0))
        total = pv + batt + hvac + other

        # actions / externals
        hvac_act = nz(metrics.get("hvac_act"))
        batt_act = nz(metrics.get("batt_act"))
        price    = nz(metrics.get("price"))
        occ      = nz(metrics.get("occupied"))
        soc      = float(getattr(s, "soc", metrics.get("soc", float("nan"))))

        # temps
        Tin = getattr(s, "T_inside", metrics.get("T_inside", float("nan")))
        Te  = getattr(s, "T_envelope", metrics.get("T_envelope", Tin))
        Tout= metrics.get("T_outside", getattr(s, "T_outside", float("nan")))
        solar = metrics.get("solar", float("nan"))

        # rewards
        import numpy as np
        r_tot = metrics.get("reward", np.nan)
        r_fin = metrics.get("reward_fin", -float(metrics.get("net_opex", 0.0)) if r_tot is not np.nan else np.nan)
        r_com = metrics.get("reward_comf",
                            (float(r_tot) + float(metrics.get("net_opex", 0.0))) if r_tot is not np.nan else np.nan)

        return {
            "t": t_new,
            "pv_kw": pv, "batt_kw": batt, "hvac_kw": hvac, "other_kw": other, "total_kw": total,
            "hvac_act": hvac_act, "batt_act": batt_act, "price": price, "occupied": occ, "soc": soc,
            "Tin": Tin, "Te": Te, "Tout": Tout, "solar": solar,
            "reward": nz(r_tot), "reward_fin": nz(r_fin), "reward_comf": nz(r_com),
            "_forecast": metrics.get("forecast", None),
        }

    def _append_sample(self, sample: dict):
        """Append one sample to buffers (like update does, but single step)."""
        B = self.buf
        for k in ("t","pv_kw","batt_kw","hvac_kw","other_kw","total_kw",
                  "hvac_act","batt_act","price","occupied","soc",
                  "Tin","Te","Tout","solar","reward_fin","reward_comf","reward"):
            if k in sample:
                B[k].append(sample[k])
        self._forecast = sample.get("_forecast", None)

    def update_smooth(self, step_or_state, metrics, frames=6, duration_ms=120, easing="linear"):
        """
        Interpolate between the last point and the new point.
        Creates `frames` interim samples over `duration_ms` using Tk .after().
        """
        # If no prior data, just do a normal update
        if len(self.buf["t"]) == 0:
            self.update(step_or_state, metrics)
            return

        import numpy as np
        # prev values (last)
        prev = {k: (self.buf[k][-1] if len(self.buf[k]) else np.nan)
                for k in self.buf.keys()}

        # target values (what the next update would append)
        target = self._compute_next_sample(step_or_state, metrics)
        t0, t1 = prev["t"], target["t"]
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            # fallback if time is weird/non-monotonic
            self.update(step_or_state, metrics)
            return

        # simple easing
        def ease(u):
            if easing == "quad":     # ease-in-out
                return 2*u*u if u < 0.5 else 1 - 2*(1-u)*(1-u)
            return u  # linear

        # build all interim frames (don't touch buffers yet)
        xs = np.linspace(0.0, 1.0, frames+1)[1:]  # skip 0, end at 1
        series_keys = ("pv_kw","batt_kw","hvac_kw","other_kw","total_kw",
                       "hvac_act","batt_act","price","occupied","soc",
                       "Tin","Te","Tout","solar","reward_fin","reward_comf","reward")

        # per-frame append function
        widget = self.canvas.get_tk_widget()
        per_frame_ms = max(1, duration_ms // max(1, frames))
        self._animating = True

        def do_frame(i=0):
            u = ease(xs[i])
            # interpolate one sample
            samp = {"t": (t0 + u*(t1 - t0)), "_forecast": target["_forecast"]}
            for k in series_keys:
                a = prev.get(k, np.nan); b = target.get(k, np.nan)
                if np.isfinite(a) and np.isfinite(b):
                    samp[k] = a + u*(b - a)
                else:
                    samp[k] = b  # jump if NaN

            # append and draw
            self._append_sample(samp)
            self._maybe_update_xwindow(samp["t"])
            self._draw_electricity()
            self._draw_actions()
            self._draw_temperature()
            self._draw_weather()
            self.canvas.draw_idle()

            # schedule next or finish on exact target (to avoid drift)
            if i+1 < len(xs):
                widget.after(per_frame_ms, lambda: do_frame(i+1))
            else:
                # ensure final equals target exactly once
                self._append_sample(target)
                self._maybe_update_xwindow(target["t"])
                self._draw_electricity()
                self._draw_actions()
                self._draw_temperature()
                self._draw_weather()
                self.canvas.draw_idle()
                self._animating = False
        do_frame(0)
