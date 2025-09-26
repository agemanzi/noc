# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image, ImageDraw

import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from ..engine.state import GameState, Action
from ..engine.simulation import SimulationEngine
from ..engine.recorder import GameRecorder
from ..engine.settings import GameSettings
from ..engine.datafeed import DataFeed
from ..engine.reward import RewardConfig
from .house_render import HouseRenderer, HouseRenderData
# top of file, with the other imports
import json  # >>> NEW
from datetime import timedelta  # >>> NEW

import datetime as dt


from .plots import Charts
TICKS = 20  # ms between steps when playing
RAND = random.Random(42)

class App:
    def __init__(self, root):
        
        data_dir = Path(__file__).resolve().parent.parent / "data"
        weather_csv = data_dir / "weekXX_prices_weather_seasons_FROM_2023_RELABELED_TO_2025.csv"
        weather_csv = data_dir / "_2ndweekXX_prices_weather_seasons_FROM_2023_RELABELED_TO_2025.csv"


        
        load_csv    = data_dir / "load_profile.csv"

        self.root = root
        self.engine = SimulationEngine(dt=900)
        self.rec = GameRecorder()
        self.settings = GameSettings()
        # Create feed *after* settings so we can rebase timestamps
        self.feed = DataFeed(weather_csv, load_csv)
        self.feed.set_anchor_date(self.settings.start_date)
        self.start_dt = dt.datetime.combine(self.settings.start_date, dt.time(0, 0))
        self.state = GameState(
            t=0, T_inside=22.0, T_outside=30.0, soc=0.5, kwh_used=0.0, T_envelope=22.0
        )

        # --- Game/session trackers ---------------------------------------------------
        self.session_score = 0.0                 # >>> NEW (accumulates reward over the run)
        self.day_var = tk.StringVar(value="Day 1/7  Score: +0.00 €")  # >>> NEW (HUD text)

        # Persisted scores (JSON) under repo/outputs/scores.json
        self.scores_path = Path(__file__).resolve().parents[2] / "outputs" / "scores.json"  # >>> NEW
        self.scores_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists  # >>> NEW

        # --- Layout frames --------------------------------------------------
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=0)
        root.rowconfigure(2, weight=0)

        self.left = ttk.Frame(root, padding=8)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.right = ttk.Frame(root, padding=8)
        self.right.grid(row=0, column=1, sticky="nsew")
        self.controls = ttk.Frame(root, padding=8)
        self.controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status = ttk.Label(root, anchor="w", text="Status: Ready.")
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew")

        # --- LEFT: House + Devices + Outputs ---------------------------------
        self._build_left()

        # --- RIGHT: Charts ----------------------------------------------------
        # self.charts = Charts(self.right)

        # --- Controls ---------------------------------------------------------
        # replay fields (BEFORE _build_controls)
        self.replay_rows = []
        self.replay_idx = 0
        self.replay_on = tk.BooleanVar(value=False)
        # repo root: .../noc  (two parents up from gui/)
        self.replay_path_default = Path(__file__).resolve().parents[2] / "outputs" / "replays" / "ghost_run.csv"

        self._build_controls()

        # Use settings for both rewards and plotting
        self.reward_cfg = self._mk_reward_cfg_from_settings()
        # Charts: set comfort band from settings using occupied tolerance initially
        init_tol = self.settings.comfort_tolerance_occupied_C
        lo = self.settings.comfort_target_C - init_tol
        hi = self.settings.comfort_target_C + init_tol
        self.charts = Charts(self.right, comfort=(lo, hi))        
        self.charts.reset_time_axes(clear_buffers=True)  # ensure clean state for initial anchor
        
        # play/pause
        self.playing = False
        self._tick_after_id = None

        self.root.bind("<F1>", lambda e: self.bat.set(-1))
        self.root.bind("<F2>", lambda e: self.bat.set(0))
        self.root.bind("<F3>", lambda e: self.bat.set(1))

        # Arrow keys: Left = cool, Right = heat
        self.root.bind("<Left>",  lambda e: self._hvac_nudge(-0.05))
        self.root.bind("<Right>", lambda e: self._hvac_nudge(+0.05))

        # Shift+Arrow for bigger steps
        self.root.bind("<Shift-Left>",  lambda e: self._hvac_nudge(-0.15))
        self.root.bind("<Shift-Right>", lambda e: self._hvac_nudge(+0.15))

        # Press-and-hold auto-repeat
        self.root.bind("<KeyPress-Left>",  lambda e: self._hvac_hold_start(-0.03))
        self.root.bind("<KeyPress-Right>", lambda e: self._hvac_hold_start(+0.03))
        self.root.bind("<KeyRelease-Left>",  self._hvac_hold_stop)
        self.root.bind("<KeyRelease-Right>", self._hvac_hold_stop)

        

    # --------- LEFT SIDE -----------------------------------------------------
    def _badge(self, parent, text, bg):
        f = ttk.Frame(parent, padding=(6,2))
        lbl = tk.Label(f, text=text, bg=bg, fg="white", padx=8, pady=2)
        lbl.pack()
        return f, lbl

    # def draw_house(self, occupied: bool, pv_on: bool, night: float):
    #     """
    #     Render the house image with overlays.

    #     Args:
    #         occupied: if True, windows glow
    #         pv_on:    if True, add a subtle roof/sun accent
    #         night:    0.0 (day) .. 1.0 (deep night) darkening filter
    #     """
    #     # 1) start from base
    #     img = self._img_house_base.copy()

    #     # 2) resize to canvas
    #     img = img.resize((self.house_canvas_w, self.house_canvas_h), Image.Resampling.LANCZOS)

    #     # 3) night filter (darken via multiply towards black)
    #     if night > 0:
    #         night = max(0.0, min(1.0, float(night)))
    #         black = Image.new("RGBA", img.size, (0, 0, 0, int(180 * night)))
    #         img = Image.alpha_composite(img, black)

    #     # 4) occupancy lights (simple yellow windows)
    #     if occupied:
    #         overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    #         d = ImageDraw.Draw(overlay)
    #         # rectangles approximating “windows” — adjust to your PNG
    #         win = [(20, 35, 40, 55), (50, 35, 70, 55), (80, 35, 100, 55)]
    #         for x0, y0, x1, y1 in win:
    #             d.rectangle([x0, y0, x1, y1], fill=(255, 220, 90, 180))
    #         img = Image.alpha_composite(img, overlay)

    #     # 5) PV accent (little sun + roof tint)
    #     if pv_on:
    #         overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    #         d = ImageDraw.Draw(overlay)
    #         # sun
    #         d.ellipse([img.size[0]-32, 8, img.size[0]-12, 28], fill=(255, 215, 0, 180))
    #         # roof strip (adjust coords to your house.png roof)
    #         d.rectangle([20, 20, img.size[0]-20, 30], fill=(34, 197, 94, 90))  # greenish
    #         img = Image.alpha_composite(img, overlay)

    #     # 6) push to canvas
    #     self._house_tk = ImageTk.PhotoImage(img)  # keep ref to avoid GC
    #     self.house_canvas.delete("all")
    #     self.house_canvas.create_image(0, 0, image=self._house_tk, anchor="nw")
        
    def _build_left(self):
        self.house_view = HouseRenderer(
            self.left,
            width=600, height=1000,
            image_path=os.path.join(os.path.dirname(__file__), "house.png"),
            title="House"
        )
        # Only SOC + temps; SOC at top-left
        self.house_view.set_config({
            "feature_flags": {"show_soc": True, "show_temps": True},
            "gauges": {"soc": {"corner": "top_left"}},
            # Optional: tweak outside thresholds if you want different bands
            # "temps": {"tout": {"cold_max_C": -5.0, "hot_min_C": 28.0}}
        })
        self.house_view.pack(fill="x", pady=4)


    # def _build_left(self):
    #     # HOUSE card
    #     house = ttk.Labelframe(self.left, text="House")
    #     house.pack(fill="x", pady=4)

    #     # canvas for the PNG + first neutral draw (no occupancy, day)
    #     self.house_canvas_w = 160
    #     self.house_canvas_h = 120
    #     self.house_canvas = tk.Canvas(house, width=self.house_canvas_w,
    #                                 height=self.house_canvas_h, highlightthickness=0)
    #     self.house_canvas.pack(pady=(6, 0))

    #     # preload image once (requires Pillow)
    #     import os
    #     from PIL import Image, ImageTk, ImageDraw
    #     self._img_house_base_path = os.path.join(os.path.dirname(__file__), "house.png")
    #     try:
    #         from PIL import Image
    #         self._img_house_base = Image.open(self._img_house_base_path).convert("RGBA")
    #     except Exception:
    #         self._img_house_base = Image.new("RGBA", (160, 120), (200, 200, 200, 255))
    #         ImageDraw.Draw(self._img_house_base).text((8,8), "house.png\nmissing", fill=(80,80,80,255))

    #     # create label variable AFTER declaring it
    #     self.occupancy_var = tk.StringVar(value="Empty")
    #     ttk.Label(house, textvariable=self.occupancy_var).pack(anchor="w", padx=6, pady=6)

    #     # first static render
    #     self._house_tk = None
    #     self.draw_house(occupied=False, pv_on=False, night=0.0)

    #     # DEVICES card
    #     devices = ttk.Labelframe(self.left, text="Devices")
    #     devices.pack(fill="x", pady=4)
    #     row = ttk.Frame(devices); row.pack(fill="x", pady=4)
    #     (b1, self.badge_hvac) = self._badge(row, "HVAC", "#4f46e5"); b1.pack(side="left", padx=4)
    #     (b2, self.badge_pv)   = self._badge(row, "PV OFF", "#6b7280"); b2.pack(side="left", padx=4)
    #     (b3, self.badge_bat)  = self._badge(row, "Batt 50%", "#059669"); b3.pack(side="left", padx=4)

    #     # OUTPUTS card
    #     outputs = ttk.Labelframe(self.left, text="Outputs")
    #     outputs.pack(fill="both", expand=True, pady=4)
    #     grid = ttk.Frame(outputs); grid.pack(fill="x", padx=6, pady=6)
    #     labels = {
    #         "hvac_heat": "HVAC Heat/Cool (kW)",
    #         "pv_gen": "PV Generation (kW)",
    #         "batt_soc": "Battery SOC",
    #         "reward_fin": "Financial score (€/step)",
    #         "reward_comf": "Comfort score (€/step)",
    #         "reward": "Total reward (€/step)",
    #     }
    #     self.out_vars = {k: tk.StringVar(value="–") for k in labels}
    #     r = 0
    #     for key, title in labels.items():
    #         ttk.Label(grid, text=title).grid(row=r, column=0, sticky="w")
    #         ttk.Label(grid, textvariable=self.out_vars[key]).grid(row=r, column=1, sticky="e")
    #         r += 1

    # --------- CONTROLS ------------------------------------------------------
    def _build_controls(self):
        self.hvac = tk.DoubleVar(value=0.0)
        self.bat  = tk.IntVar(value=0)          # will be driven by radios: -1 / 0 / +1

        # HVAC slider + arrow buttons
        hvac_frame = ttk.Frame(self.controls)
        hvac_frame.grid(row=0, column=0, sticky="ew", padx=6)

        btn_left  = ttk.Button(hvac_frame, text="◀", width=2)
        btn_right = ttk.Button(hvac_frame, text="▶", width=2)
        btn_left.grid(row=0, column=0, padx=(0,4))
        btn_right.grid(row=0, column=2, padx=(4,0))

        scale = ttk.Scale(hvac_frame, from_=-1, to=1, variable=self.hvac, orient="horizontal")
        scale.grid(row=0, column=1, sticky="ew")
        hvac_frame.columnconfigure(1, weight=1)

        # click for single nudge
        btn_left.configure(command=lambda: self._hvac_nudge(-0.05))
        btn_right.configure(command=lambda: self._hvac_nudge(+0.05))

        # press-and-hold repeat on buttons
        btn_left.bind("<ButtonPress-1>",  lambda e: self._hvac_hold_start(-0.03))
        btn_right.bind("<ButtonPress-1>", lambda e: self._hvac_hold_start(+0.03))
        btn_left.bind("<ButtonRelease-1>",  self._hvac_hold_stop)
        btn_right.bind("<ButtonRelease-1>", self._hvac_hold_stop)

        ttk.Label(self.controls, text="HVAC -1..1").grid(row=1, column=0)

        # Battery radio group (in a small frame)
        bat_frame = ttk.Frame(self.controls)
        bat_frame.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Radiobutton(bat_frame, text="Charge", value=-1, variable=self.bat).grid(row=0, column=0, padx=2)
        ttk.Radiobutton(bat_frame, text="Idle",       value= 0, variable=self.bat).grid(row=0, column=1, padx=2)
        ttk.Radiobutton(bat_frame, text="Discharge",     value= 1, variable=self.bat).grid(row=0, column=2, padx=2)
        ttk.Label(self.controls, text="Battery (-1/0/+1)").grid(row=1, column=1)

        # Buttons
        ttk.Button(self.controls, text="Step", command=self.step_once).grid(row=0, column=2, padx=6)
        self.play_btn = ttk.Button(self.controls, text="▶ Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=3, padx=6)
        ttk.Button(self.controls, text="Reset", command=self.reset).grid(row=0, column=4, padx=6)
        ttk.Button(self.controls, text="Export CSV", command=self.export).grid(row=0, column=5, padx=6)
        ttk.Button(self.controls, text="⚙ Settings", command=self.open_settings).grid(row=0, column=6, padx=6)
        ttk.Button(self.controls, text="Load Replay", command=self._load_replay_csv).grid(row=0, column=7, padx=6)
        ttk.Checkbutton(self.controls, text="Replay", variable=self.replay_on).grid(row=0, column=8, padx=6)

        # >>> NEW: Day/Score HUD on the right side of the controls
        hud = ttk.Label(self.controls, textvariable=self.day_var, anchor="e")
        hud.grid(row=1, column=8, sticky="e", padx=6)  # reuse right-most column

        # stretchy columns
        for c in range(0, 9):
            self.controls.columnconfigure(c, weight=1)


    # --------- LOOP ----------------------------------------------------------
    # --------- LOOP ----------------------------------------------------------
    def _load_replay_csv(self, path=None):
        import csv
        p = Path(path) if path else Path(self.replay_path_default)
        if not p.exists():
            self.status.config(text=f"Replay file not found: {p}")
            return
        rows = []
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for d in r:
                # make all numerics floats/ints where sensible; ignore missing keys
                def fget(k, default=0.0):
                    try:
                        return float(d.get(k, default))
                    except (TypeError, ValueError):
                        return float(default)
                def iget(k, default=0):
                    try:
                        return int(float(d.get(k, default)))
                    except (TypeError, ValueError):
                        return int(default)
                rows.append({
                    "T_outside": fget("T_outside"),
                    "price": fget("price"),
                    "pv_kw": fget("pv_kw"),
                    "base_load_kw": fget("base_load_kw"),
                    "hvac": fget("hvac"),
                    "battery_cmd": iget("battery_cmd"),
                    # optional extras if present
                    "ts": d.get("ts", None),
                })
        if not rows:
            self.status.config(text=f"Replay file is empty: {p}")
            return
        self.replay_rows = rows
        self.replay_idx = 0
        self.replay_on.set(True)
        # optional: align feed start if CSV has a timestamp column (we'll still override exogenous)
        self.status.config(text=f"Replay loaded: {p.name} ({len(rows)} steps)")

    def step_once(self):
        # If replaying, pull next row; else read from UI/Feed
        replay_row = None
        if self.replay_on.get() and self.replay_idx < len(self.replay_rows):
            replay_row = self.replay_rows[self.replay_idx]
            self.replay_idx += 1
            # 1) action from CSV
            action = Action(hvac=float(replay_row["hvac"]),
                            battery=int(replay_row["battery_cmd"]))
        else:
            # normal interactive action
            action = Action(hvac=self.hvac.get(), battery=int(self.bat.get()))
            if self.replay_on.get() and self.replay_idx >= len(self.replay_rows) and self.playing:
                # reached EOF during replay
                self.playing = False
                self.play_btn.config(text="▶ Play")
                self.status.config(text="Status: Replay finished.")
                return

        # 2) get data row for current sim time
        row = self.feed.by_time(self.state.t)
        occupied = bool(row.occupied_home)

        # pick tolerance by occupancy and refresh reward cfg + chart band
        tol = self.settings.tolerance_for(occupied)
        self.reward_cfg = self._mk_reward_cfg_from_settings(tol_C=tol)
        self.charts.comfort = (
            self.settings.comfort_target_C - tol,
            self.settings.comfort_target_C + tol
        )

        # 3) inputs (use CSV values in replay; otherwise from feed/settings)
        if replay_row is not None:
            price        = float(replay_row["price"])
            T_outside    = float(replay_row["T_outside"])
            pv_kw        = float(replay_row["pv_kw"])
            base_load_kw = float(replay_row["base_load_kw"])
            # still define pv_kwp_yield for charts; safe default
            pv_kwp_yield = float(getattr(row, "solar_gen_kw_per_kwp", 0.0))
        else:
            price        = float(row.price_eur_per_kwh)
            T_outside    = float(row.t_out_c)
            pv_kwp_yield = float(row.solar_gen_kw_per_kwp)
            pv_kw        = pv_kwp_yield * float(self.settings.pv_size_kw)  # PV always ON
            base_load_kw = float(row.base_load_kw)

        # Optional simple gains
        q_internal_kw = 0.7 if occupied else 0.3
        q_solar_kw    = 0.0

        # 4) step the engine
        try:
            step = self.engine.step(self.state, action, {
                "ts": row.ts,
                "T_outside": T_outside,
                "price": price,
                "pv_kw": pv_kw,                # generation (+)
                "base_load_kw": base_load_kw,  # consumption (+)
                "battery_cmd": int(action.battery),  # <-- was self.bat.get()
                "settings": self.settings,
                "occupied_home": int(occupied),
                "q_internal_kw": q_internal_kw,
                "q_solar_kw": q_solar_kw,
                "reward_cfg": self.reward_cfg,   # <-- important
            })
        except TypeError:
            # legacy engine signature fallback
            self.state.T_outside = T_outside
            step = self.engine.step(self.state, action)

        # 5) read metrics once
        m = step.metrics
        hvac_kw    = float(m.get("hvac_kw", 0.0))        # (+) electric draw
        battery_kw = float(m.get("battery_kw", 0.0))     # (+) discharge, (−) charge
        other_kw   = float(m.get("other_kw", 0.0))       # (−) load for chart convention
        pv_kw      = float(m.get("pv_kw", pv_kw))
        total_kw   = float(m.get("total_kw", pv_kw + battery_kw - hvac_kw + other_kw))
        elec_kwh   = float(m.get("electricity", 0.0))    # energy for this step

        # 6) reward components (compute FIRST)
        reward_tot  = float(m.get("reward", 0.0))
        net_opex    = float(m.get("net_opex", m.get("opex_cost", 0.0)))  # compatibility
        reward_fin  = float(m.get("reward_fin", -net_opex))               # financial = -net_opex
        reward_comf = float(m.get("reward_comf", reward_tot - reward_fin))# comfort = total - financial

        # HVAC telemetry (thermal output, electric draw, COP)
        hvac_heat_kw = float(m.get("q_hvac_kw", 0.0))   # +heat, −cool
        hvac_elec_kw = float(m.get("hvac_kw", 0.0))     # +draw
        hvac_cop     = m.get("hvac_cop", None)
        try:
            hvac_cop = float(hvac_cop) if hvac_cop is not None else None
        except Exception:
            hvac_cop = None

        # 7) occupancy text + house overlay
        # self.occupancy_var.set("Occupied" if occupied else "Away")
        # night = 1.0 - max(0.0, min(1.0, pv_kwp_yield))  # daytime → 0, night → 1
        # self.draw_house(occupied=occupied, pv_on=self.pv_on.get(), night=night)

        # 8) advance state before showing SOC
        self.state = step.state

        # >>> REPLACE day/score tracking with tick-based logic (15 min per tick)

        # 4A) accumulate session score
        self.session_score += reward_tot

        # tick math
        tick_seconds   = int(getattr(self.engine, "dt", 900))  # engine dt in seconds (default 900)
        steps_per_day  = int(round(24 * 3600 / tick_seconds))  # 96 for 15-min ticks
        tick_index     = int(round(self.state.t / tick_seconds))

        # 4B) compute day index from ticks (1..7)
        day_idx = (tick_index // steps_per_day) + 1
        if day_idx < 1:
            day_idx = 1
        elif day_idx > 7:
            day_idx = 7

        # update HUD
        self.day_var.set(f"Day {day_idx}/7  Score: {self.session_score:+.2f} €")

        # 4C) end game right after completing 7 full days
        if tick_index >= steps_per_day * 7:
            self._end_game()
            return


        # # 9) badges/labels (now safe to use reward_* and new SOC)
        # self.badge_hvac.config(text=f"HVAC {action.hvac:+.1f}")
        # self.badge_pv.config(text="PV ON" if self.pv_on.get() else "PV OFF",
        #                     bg=("#10b981" if self.pv_on.get() else "#6b7280"))
        # self.badge_bat.config(text=f"Batt {int(self.state.soc*100):d}%")

        # self.out_vars["hvac_heat"].set(f"{hvac_kw:0.2f}")  # electric draw (kW)
        # self.out_vars["pv_gen"].set(f"{pv_kw:0.2f}")
        # self.out_vars["batt_soc"].set(f"{self.state.soc:0.2f}")
        # self.out_vars["reward_fin"].set(f"{reward_fin:+.3f}")
        # self.out_vars["reward_comf"].set(f"{reward_comf:+.3f}")
        # self.out_vars["reward"].set(f"{reward_tot:+.3f}")

        # 10) optional forecast (build BEFORE charts.update)
        try:
            fw = self.feed.window_by_time(self.state.t, horizon_steps=48)
            forecast_data = {
                "ts": fw.ts,
                "t_out_c": fw.t_out_c,
                "solar_per_kwp": fw.solar_gen_kw_per_kwp,
                "price": fw.price_eur_per_kwh,
                "base_kw": fw.base_load_kw,
                "occupied": fw.occupied_home,
            }
        except Exception:
            forecast_data = None

        # 11) one charts.update call with everything
        self.charts.update(step, {
            "timestamp": row.ts,
            "pv_kw": pv_kw,
            "battery_kw": battery_kw,
            "hvac_kw": hvac_kw,
            "other_kw": other_kw,
            "price": price,
            "solar": pv_kwp_yield * 1000.0,
            "T_outside": T_outside,
            "hvac_act": float(action.hvac),
            "batt_act": float(action.battery),
            "occupied": int(occupied),

            # reward channels
            "reward_fin": reward_fin,
            "reward_comf": reward_comf,
            "reward":     reward_tot,

            "forecast": forecast_data,
        })

        self.house_view.update(HouseRenderData(
            timestamp=row.ts,
            # temps
            T_inside=float(self.state.T_inside) if hasattr(self.state, "T_inside") else None,
            T_outside=T_outside,

            # comfort band (for T_in color)
            comfort_target_C=self.settings.comfort_target_C,
            comfort_tolerance_C=self.settings.comfort_tolerance_C,

            # battery
            soc=float(self.state.soc),
            
            # HVAC telemetry
            hvac_elec_kw=hvac_elec_kw,
            hvac_heat_kw=hvac_heat_kw,
            hvac_cop=hvac_cop,
            hvac_nameplate_kw=float(self.settings.hvac_size_kw),
            
            cumulative_score=float(getattr(self.state, "cumulative_reward", 0.0)),
            comfort_score=reward_comf,        # ← NEW
            financial_score=reward_fin,       # ← NEW
        ))

        # 12) status line
        self.status.config(text=f"Status: {row.ts:%Y-%m-%d %H:%M}  kWh={self.state.kwh_used:.3f}")



    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._schedule_tick()

    def _schedule_tick(self):
        if not self.playing:
            return
        self.step_once()
        self._tick_after_id = self.root.after(TICKS, self._schedule_tick)  # ~10 fps

    def reset(self):
        if self._tick_after_id:
            self.root.after_cancel(self._tick_after_id)
            self._tick_after_id = None
        self.playing = False
        self.play_btn.config(text="▶ Play")
        self.state = GameState(t=0, T_inside=22.0, T_outside=30.0, soc=0.5, kwh_used=0.0, T_envelope=22.0)
        self.rec = GameRecorder()
        self.charts.reset_time_axes(clear_buffers=True)
        self.session_score = 0.0  # >>> NEW: reset session score
        self.day_var.set("Day 1/7  Score: +0.00 €")  # >>> NEW: reset day tracker
        self.status.config(text="Status: Reset.")

    def _end_game(self):  # >>> NEW
        """Stop the loop and open the save score dialog."""
        # Stop the tick loop and reset play button
        if self._tick_after_id:
            self.root.after_cancel(self._tick_after_id)
            self._tick_after_id = None
        self.playing = False
        self.play_btn.config(text="▶ Play")

        self.status.config(text="Game over! Week complete.")
        self._show_save_score_dialog(final=True)

    def _show_save_score_dialog(self, final=False):  # >>> NEW
        """Arcade-style name/initials prompt to save the score to JSON."""
        win = tk.Toplevel(self.root)
        win.title("🏁 Game Over — Save Score")
        win.grab_set()
        win.resizable(False, False)
        padx = {"padx": 10, "pady": 6}

        ttk.Label(win, text="GAME OVER", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, **padx)
        ttk.Label(win, text=f"Final Score: {self.session_score:+.2f} €", font=("Segoe UI", 12)).grid(row=1, column=0, columnspan=2, **padx)

        ttk.Label(win, text="Enter Name/Initials:").grid(row=2, column=0, sticky="e", **padx)
        name_var = tk.StringVar(value="AAA")
        name_entry = ttk.Entry(win, textvariable=name_var, width=24)
        name_entry.grid(row=2, column=1, sticky="w", **padx)
        name_entry.focus_set()

        # Show a tiny leaderboard preview (top 5)
        board = self._load_scores()
        top = sorted(board, key=lambda r: r.get("score", 0.0), reverse=True)[:5]
        if top:
            ttk.Label(win, text="Top Scores:", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, columnspan=2, **padx)
            text = "\n".join([f"{i+1:>2}. {r.get('name','---'):<10}  {r.get('score',0.0):>+8.2f} €"
                              for i, r in enumerate(top)])
            lbl = ttk.Label(win, text=text, justify="left")
            lbl.grid(row=4, column=0, columnspan=2, sticky="w", **padx)

        def do_save(close_after=True):
            name = name_var.get().strip() or "AAA"
            self._append_score(name=name, score=float(self.session_score))
            if close_after:
                win.destroy()

        def save_and_restart():
            do_save(close_after=True)
            self.reset()   # back to t=0 with same settings

        btns = ttk.Frame(win)
        btns.grid(row=5, column=0, columnspan=2, sticky="ew", **padx)
        ttk.Button(btns, text="Save", command=do_save).pack(side="left", padx=4)
        ttk.Button(btns, text="Save & Restart", command=save_and_restart).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=4)

    def _load_scores(self):  # >>> NEW
        """Return the list of score dicts from JSON, or [] if missing/invalid."""
        try:
            if self.scores_path.exists():
                with self.scores_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            pass
        return []

    def _append_score(self, name: str, score: float):  # >>> NEW
        """Append a score to the JSON file."""
        board = self._load_scores()
        entry = {
            "name": name,
            "score": round(float(score), 3),
            "date": dt.date.today().isoformat(),
        }
        board.append(entry)
        try:
            with self.scores_path.open("w", encoding="utf-8") as f:
                json.dump(board, f, indent=2, ensure_ascii=False)
            self.status.config(text=f"Score saved for {name}: {entry['score']:+.3f} € → {self.scores_path.name}")
        except Exception as e:
            self.status.config(text=f"Failed to save score: {e}")

    def export(self):
        path = self.rec.export_csv("run.csv")
        self.status.config(text=f"Status: Saved {path}")
        print(f"Saved {path}")

    # def _update_badges(self):
    #     self.badge_pv.config(text="PV ON" if self.pv_on.get() else "PV OFF",
    #                          bg=("#10b981" if self.pv_on.get() else "#6b7280"))
    def _update_badges(self):
        # Repaint the house/HUD with the new PV toggle state using current time/state
        row = self.feed.by_time(self.state.t)
        pv_kwp_yield = float(row.solar_gen_kw_per_kwp)
        price        = float(row.price_eur_per_kwh)
        T_outside    = float(row.t_out_c)
        occupied     = bool(row.occupied_home)
        base_load_kw = float(row.base_load_kw)
        pv_kw        = pv_kwp_yield * float(self.settings.pv_size_kw)  # PV always ON

        self.house_view.update(HouseRenderData(
            timestamp=row.ts,
            T_inside=float(getattr(self.state, "T_inside", 0.0)) if hasattr(self.state, "T_inside") else None,
            T_outside=T_outside,
            comfort_target_C=self.settings.comfort_target_C,
            comfort_tolerance_C=self.settings.comfort_tolerance_C,
            soc=float(self.state.soc) if hasattr(self.state, "soc") else None,
            
            # HVAC telemetry (no current data in _update_badges)
            hvac_elec_kw=0.0,
            hvac_heat_kw=0.0,
            hvac_cop=None,
            hvac_nameplate_kw=float(self.settings.hvac_size_kw),
            
            comfort_score=None,        # ← NEW (no reward data in _update_badges)
            financial_score=None,      # ← NEW (no reward data in _update_badges)
        ))

    def _mk_reward_cfg_from_settings(self, tol_C: float | None = None) -> RewardConfig:
        # €/deg²·step from €/deg²·hour
        cw = float(self.settings.comfort_anchor_eur_per_deg2_hour) * self.engine.dt_h
        tol = float(tol_C if tol_C is not None else self.settings.comfort_tolerance_occupied_C)

        cfg_kwargs = dict(
            comfort_target_C    = self.settings.comfort_target_C,
            comfort_tolerance_occupied_C = self.settings.comfort_tolerance_occupied_C,
            comfort_tolerance_unoccupied_C = self.settings.comfort_tolerance_unoccupied_C,
            comfort_weight      = cw,
            export_tariff_ratio = self.settings.export_tariff_ratio,
        )
        if hasattr(RewardConfig, "__dataclass_fields__") and \
           "comfort_inside_bonus" in RewardConfig.__dataclass_fields__:
            cfg_kwargs["comfort_inside_bonus"] = float(self.settings.comfort_inside_bonus_eur_per_step)
        return RewardConfig(**cfg_kwargs)
    def open_settings(self):
        # --- PRESETS ------------------------------------------------------------
        PRESETS = {
            "Small PV / Large Batt": {"pv_kw": 5.0, "batt_kwh": 15.0},
            "Large PV / Small Batt": {"pv_kw": 15.0, "batt_kwh": 5.0},
            "Mid / Mid":             {"pv_kw": 10.0, "batt_kwh": 10.0},
        }

        def apply_preset(name: str):
            p = PRESETS[name]
            pv_var.set(p["pv_kw"])
            batt_var.set(p["batt_kwh"])

        win = tk.Toplevel(self.root)
        win.title("Game Settings")
        win.grab_set()
        # make the entry column stretch
        win.columnconfigure(1, weight=1)

        pv_var   = tk.DoubleVar(value=self.settings.pv_size_kw)
        hvac_var = tk.DoubleVar(value=self.settings.hvac_size_kw)
        batt_var = tk.DoubleVar(value=self.settings.batt_size_kwh)
        date_var = tk.StringVar(value=self.settings.start_date.isoformat())

        # comfort/econ vars
        ctgt_var     = tk.DoubleVar(value=self.settings.comfort_target_C)
        ctol_occ_var = tk.DoubleVar(value=self.settings.comfort_tolerance_occupied_C)
        ctol_unocc_var = tk.DoubleVar(value=self.settings.comfort_tolerance_unoccupied_C)
        cwgt_var     = tk.DoubleVar(value=self.settings.comfort_weight)
        xrt_var      = tk.DoubleVar(value=self.settings.export_tariff_ratio)
        anchor_var   = tk.DoubleVar(value=self.settings.comfort_anchor_eur_per_deg2_hour)
        bonus_var    = tk.DoubleVar(value=self.settings.comfort_inside_bonus_eur_per_step)

        r = 0

        # ---- Presets row at the very top --------------------------------------
        ttk.Label(win, text="Presets (kW)").grid(row=r, column=0, sticky="w", padx=6, pady=(8, 4))
        btns = ttk.Frame(win)
        btns.grid(row=r, column=1, sticky="w", padx=6, pady=(8, 4))
        ttk.Button(btns, text="Small PV / Large Batt",
                command=lambda: apply_preset("Small PV / Large Batt")).pack(side="left", padx=2)
        ttk.Button(btns, text="Large PV / Small Batt",
                command=lambda: apply_preset("Large PV / Small Batt")).pack(side="left", padx=2)
        ttk.Button(btns, text="Mid / Mid",
                command=lambda: apply_preset("Mid / Mid")).pack(side="left", padx=2)
        r += 1

        # ---- System sizes (PV / HVAC / Battery) --------------------------------
        ttk.Label(win, text="PV size (kW)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=pv_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="HVAC size (kW)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=hvac_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="Battery size (kWh)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=batt_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        # ---- Start date --------------------------------------------------------
        ttk.Label(win, text="Start date (YYYY-MM-DD)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=date_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        # ---- Comfort & tariff --------------------------------------------------
        ttk.Label(win, text="Comfort target (°C)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=ctgt_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="Comfort tolerance occupied (°C)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=ctol_occ_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="Comfort tolerance unoccupied (°C)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=ctol_unocc_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="Comfort weight (€/step)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=cwgt_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="Export tariff ratio (0..1)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=xrt_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="Comfort price (€/deg²·hour)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(win, textvariable=anchor_var).grid(row=r, column=1, sticky="ew", padx=6, pady=4); r += 1

        ttk.Label(win, text="Comfort bonus (€/step, inside band)").grid(
            row=r, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Entry(win, textvariable=bonus_var).grid(
            row=r, column=1, sticky="ew", padx=6, pady=4
        ); r += 1

        # ---- Save button pinned to the bottom ---------------------------------
        def save_and_close():
            import datetime as dt
            self.settings.pv_size_kw    = pv_var.get()
            self.settings.hvac_size_kw  = hvac_var.get()
            self.settings.batt_size_kwh = batt_var.get()
            try:
                self.settings.start_date = dt.date.fromisoformat(date_var.get())
            except ValueError:
                pass

            # persist comfort/tariff
            self.settings.comfort_target_C    = ctgt_var.get()
            self.settings.comfort_tolerance_occupied_C = ctol_occ_var.get()
            self.settings.comfort_tolerance_unoccupied_C = ctol_unocc_var.get()
            self.settings.comfort_weight      = cwgt_var.get()
            self.settings.export_tariff_ratio = xrt_var.get()
            self.settings.comfort_anchor_eur_per_deg2_hour = anchor_var.get()
            self.settings.comfort_inside_bonus_eur_per_step = bonus_var.get()

            # refresh reward cfg + chart band (use occupied tolerance for initial display)
            self.reward_cfg = self._mk_reward_cfg_from_settings()
            init_tol = self.settings.comfort_tolerance_occupied_C
            lo = self.settings.comfort_target_C - init_tol
            hi = self.settings.comfort_target_C + init_tol
            self.charts.comfort = (lo, hi)

            # jump to new start date (t=0)
            self.feed.set_anchor_date(self.settings.start_date)
            self.state.t = 0
            self.charts.reset_time_axes(clear_buffers=True)
            self.status.config(text=f"Status: Start date → {self.settings.start_date.isoformat()} (t=0)")
            win.destroy()

        save_btn = ttk.Button(win, text="Save", command=save_and_close)
        save_btn.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 10))


    # --- HVAC auto-repeat helpers --------------------------------------------
    def _clamp(self, x, lo=-1.0, hi=1.0):  # reuse anywhere
        return max(lo, min(hi, x))

    def _hvac_set(self, val: float):
        self.hvac.set(self._clamp(val))

    def _hvac_nudge(self, delta: float):
        self._hvac_set(self.hvac.get() + delta)

    def _hvac_hold_start(self, delta: float, interval_ms: int = 80):
        # start repeating nudge while key/mouse is held
        self._hvac_hold_delta = delta
        if getattr(self, "_hvac_hold_after", None):
            self.root.after_cancel(self._hvac_hold_after)
        def tick():
            self._hvac_nudge(self._hvac_hold_delta)
            self._hvac_hold_after = self.root.after(interval_ms, tick)
        tick()

    def _hvac_hold_stop(self, *_):
        if getattr(self, "_hvac_hold_after", None):
            self.root.after_cancel(self._hvac_hold_after)
            self._hvac_hold_after = None


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Thermal Game — Dashboard")
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
