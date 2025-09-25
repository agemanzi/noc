# -*- coding: utf-8 -*-
"""
Minimal renderer for now:
• Shows only: SOC gauge (top-left) + two big temperature labels centered (T_out left, T_in right).
• Colors:
   - T_out: very cold / normal / very hot via thresholds.
   - T_in: comfort band vs target±tol; blue below, red above, green inside (with "!" when outside).
• Fully responsive & tweakable via CONFIG and set_config({...}).

Public API keeps HouseRenderData + HouseRenderer.update().
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk, ImageDraw  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore
    ImageDraw = None  # type: ignore


CONFIG: Dict[str, Any] = {
    "theme": {
        "canvas_bg": "#0b1220",
        "band_steps": 6,
        "band_base_rgb": (12, 20, 28),
        "band_step": 10,
        "label_caption": {"fill": "#9fb0c9", "size": 12, "weight": "normal"},
        "label_units": {"fill": "#c7d4ec", "size": 14, "weight": "normal"},
    },
    "temps": {
        # Positions are relative; labels auto-center on left/right halves.
        "value_scale": 0.18,   # fraction of min(w,h) for big value font
        "caption_gap_px": 10,  # gap between caption and value
        "tout": {
            "cold_max_C": 0.0,    # ≤ this => "very cold"
            "hot_min_C": 30.0,    # ≥ this => "very hot"
            "cold_color": "#3b82f6",   # blue
            "normal_color": "#e6eeff", # light
            "hot_color": "#ef4444",    # red
        },
        "tin": {
            "ok_color": "#22c55e",  # green in band
            "hot_color": "#ef4444", # above band
            "cold_color": "#3b82f6",# below band
            "warn_mark": "!",       # appended when outside band
        }
    },
    "gauges": {
        "soc": {
            "corner": "top_left",     # top_left | top_right | bottom_left | bottom_right
            "radius_frac": 0.16,
            "pad_frac": 0.02,
            "span_deg": 260,
            "start_deg": 140,
            "bg": "#5a6780",
            "fg": "#22c55e",
            "label": {"fill": "#e5ecff", "size": 10, "weight": "bold"},
        }
    },
    "feature_flags": {
        # Only these two are considered now:
        "show_soc": True,
        "show_temps": True,
        # The rest of the old renderer elements are off by design.
    },
}


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


@dataclass
class HouseRenderData:
    # Timing (optional HUD-like timestamp if you ever want to show it later)
    timestamp: Optional[object] = None

    # Temps
    T_inside: Optional[float] = None
    T_outside: Optional[float] = None

    # For comfort coloring of T_in:
    comfort_target_C: Optional[float] = None
    comfort_tolerance_C: Optional[float] = None

    # Battery SOC (0..1)
    soc: Optional[float] = None


class HouseRenderer(ttk.Frame):
    def __init__(self, parent, *, width: int = 320, height: int = 220,
                 image_path: Optional[str | Path] = None,
                 title: str = "House") -> None:
        super().__init__(parent, padding=6)
        self._width = int(width)
        self._height = int(height)
        self._img_path = Path(image_path) if image_path else None

        import copy
        self._config = copy.deepcopy(CONFIG)

        self.card = ttk.Labelframe(self, text=title)
        self.card.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.card, width=self._width, height=self._height,
                                highlightthickness=0,
                                bg=self._config["theme"]["canvas_bg"])
        self.canvas.pack(fill="both", expand=True)

        self._last_size = (self._width, self._height)
        self._data = HouseRenderData()
        self.canvas.bind("<Configure>", self._on_resize)
        self.draw(self._data)

    # ---------------- public API ----------------
    def update(self, data: HouseRenderData) -> None:
        self._data = data
        self.draw(data)

    def set_config(self, overrides: Dict[str, Any]) -> None:
        _deep_merge(self._config, overrides)
        self.canvas.configure(bg=self._config["theme"]["canvas_bg"])
        self.draw(self._data)

    # --------------- internals ------------------
    def _on_resize(self, event):
        size = (max(1, event.width), max(1, event.height))
        if size != self._last_size:
            self._last_size = size
            self.draw(self._data)

    def draw(self, d: HouseRenderData) -> None:
        w, h = self._last_size
        self.canvas.delete("all")

        self._bg(w, h)

        # temps in center: T_out on left half, T_in on right half
        if self._config["feature_flags"]["show_temps"]:
            self._draw_center_temps(w, h, d)

        # SOC gauge in corner
        if self._config["feature_flags"]["show_soc"] and d.soc is not None:
            self._draw_soc(w, h, float(d.soc))

    # --- primitives ---
    def _bg(self, w: int, h: int) -> None:
        t = self._config["theme"]
        steps = int(t["band_steps"])
        base_r, base_g, base_b = t["band_base_rgb"]
        step = int(t["band_step"])
        for i in range(steps):
            y0 = int(i * h/steps)
            y1 = int((i+1) * h/steps)
            r = base_r + i*step
            g = base_g + i*step + 8
            b = base_b + i*step + 16
            self.canvas.create_rectangle(
                0, y0, w, y1, width=0,
                fill=f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"
            )

    def _draw_center_temps(self, w: int, h: int, d: HouseRenderData) -> None:
        cfg = self._config["temps"]
        cap = self._config["theme"]["label_caption"]
        units = self._config["theme"]["label_units"]

        m = min(w, h)
        value_font_size = max(14, int(cfg["value_scale"] * m))
        gap = int(cfg["caption_gap_px"])

        # positions
        y = h // 2
        x_left = w // 4
        x_right = (3 * w) // 4

        # --- T_out (left) ---
        t_out_txt = "—"
        t_out_color = cfg["tout"]["normal_color"]
        if d.T_outside is not None:
            t_out = float(d.T_outside)
            t_out_txt = f"{t_out:.1f}°C"
            if t_out <= float(cfg["tout"]["cold_max_C"]):
                t_out_color = cfg["tout"]["cold_color"]
            elif t_out >= float(cfg["tout"]["hot_min_C"]):
                t_out_color = cfg["tout"]["hot_color"]
            else:
                t_out_color = cfg["tout"]["normal_color"]

        # Caption
        self.canvas.create_text(x_left, y - value_font_size - gap,
                                text="T_out", anchor="s",
                                fill=cap["fill"], font=("", cap["size"], cap["weight"]))
        # Value
        self.canvas.create_text(x_left, y, text=t_out_txt, anchor="c",
                                fill=t_out_color, font=("", value_font_size, "bold"))
        # Units (subtle)
        self.canvas.create_text(x_left, y + value_font_size + gap,
                                text="outside", anchor="n",
                                fill=units["fill"], font=("", units["size"], units["weight"]))

        # --- T_in (right) ---
        t_in_txt = "—"
        t_in_color = cfg["tin"]["ok_color"]
        if d.T_inside is not None:
            t_in = float(d.T_inside)
            t_in_txt = f"{t_in:.1f}°C"

            tgt = d.comfort_target_C
            tol = d.comfort_tolerance_C
            if tgt is not None and tol is not None:
                lo = float(tgt) - float(tol)
                hi = float(tgt) + float(tol)
                if t_in < lo:
                    t_in_txt += cfg["tin"]["warn_mark"]
                    t_in_color = cfg["tin"]["cold_color"]
                elif t_in > hi:
                    t_in_txt += cfg["tin"]["warn_mark"]
                    t_in_color = cfg["tin"]["hot_color"]
                else:
                    t_in_color = cfg["tin"]["ok_color"]
            else:
                # If no target/tol provided, just show neutral color
                t_in_color = "#e6eeff"

        self.canvas.create_text(x_right, y - value_font_size - gap,
                                text="T_in", anchor="s",
                                fill=cap["fill"], font=("", cap["size"], cap["weight"]))
        self.canvas.create_text(x_right, y, text=t_in_txt, anchor="c",
                                fill=t_in_color, font=("", value_font_size, "bold"))
        self.canvas.create_text(x_right, y + value_font_size + gap,
                                text="inside", anchor="n",
                                fill=units["fill"], font=("", units["size"], units["weight"]))

    def _draw_soc(self, w: int, h: int, soc: float) -> None:
        cfg = self._config["gauges"]["soc"]
        m = min(w, h)
        r = max(8, int(cfg["radius_frac"] * m))
        pad = int(cfg["pad_frac"] * m)

        corner = str(cfg.get("corner", "top_left"))
        if corner == "top_left":
            cx, cy = r + pad, r + pad
        elif corner == "top_right":
            cx, cy = w - r - pad, r + pad
        elif corner == "bottom_left":
            cx, cy = r + pad, h - r - pad
        else:
            cx, cy = w - r - pad, h - r - pad

        self._gauge(cx, cy, r, max(0.0, min(1.0, soc)),
                    label=f"SOC {int(soc*100):d}%")

    def _gauge(self, cx: int, cy: int, r: int, frac: float, label: str = "") -> None:
        cfg = self._config["gauges"]["soc"]
        start = int(cfg["start_deg"])
        span = int(cfg["span_deg"])
        extent = int(span * max(0.0, min(1.0, frac)))
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=start, extent=span,
                               style="arc", outline=cfg["bg"], width=10)
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=start, extent=extent,
                               style="arc", outline=cfg["fg"], width=10)
        if label:
            self.canvas.create_text(cx, cy+2, text=label,
                                    fill=cfg["label"]["fill"],
                                    font=("", cfg["label"]["size"], cfg["label"]["weight"]))


__all__ = ["HouseRenderer", "HouseRenderData", "CONFIG"]
