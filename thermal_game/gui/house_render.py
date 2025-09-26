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
    from PIL import Image, ImageTk, ImageDraw, ImageOps  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageOps = None  # type: ignore


CONFIG: Dict[str, Any] = {
    "theme": {
        "canvas_bg": "#0b1220",
        "band_steps": 6,
        "band_base_rgb": (12, 20, 28),
        "band_step": 10,
        "label_caption": {"fill": "#9fb0c9", "size": 12, "weight": "normal"},
        "label_units": {"fill": "#c7d4ec", "size": 14, "weight": "normal"},
    },
    # Center overlay image controls (keeps your gradient as the true background)
    "overlay_image": {
        "enabled": True,     # draw centered image if available
        "max_w_frac": 0.70,  # image width ≤ 70% of canvas width
        "max_h_frac": 0.70,  # image height ≤ 70% of canvas height
        "y_offset_px": 0,    # move image up/down if you like
    },
    "temps": {
        # Positions are relative; labels auto-center on left/right halves.
        "value_scale": 0.14,   # fraction of min(w,h) for big value font
        "caption_gap_px": 10,  # gap between caption and value
        "backdrop": {          # NEW: translucent pill behind the big value
            "enabled": True,
            "pad_px": 8,
            "radius_px": 12,
            "alpha": 160,      # 0..255 (higher = more opaque)
            "fill_rgb": (0, 0, 0)
        },
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
            "corner": "top_left",
            "radius_frac": 0.20,        # was 0.16 → a bit larger
            "pad_frac": 0.02,
            "span_deg": 260,
            "start_deg": 140,
            "bg": "#5a6780",
            # label color stays readable; size is now scaled dynamically (see code)
            "label": {"fill": "#e5ecff", "size": 10, "weight": "bold"},
            # NEW: thickness as a fraction of radius
            "thickness_frac": 0.14,     # thicker arc
            # NEW: dynamic label sizing
            "font_scale": 0.22,         # fraction of radius → font px
            # NEW: color thresholds
            "colors": {"low": "#ef4444", "mid": "#f59e0b", "high": "#22c55e"},
            "thresholds": {"low": 0.20, "mid": 0.60},  # ≤20% red, ≤60% orange, else green
        },
        "hvac": {  # NEW
            "corner": "top_right",         # mirror SOC default (SOC is top_left)
            "radius_frac": 0.20,
            "pad_frac": 0.02,
            "span_deg": 260,
            "start_deg": 140,
            "bg": "#5a6780",
            "thickness_frac": 0.14,
            "font_scale": 0.22,
            # colors by operating mode
            "colors": {
                "heat": "#f59e0b",   # amber for heating
                "cool": "#3b82f6",   # blue for cooling
                "idle": "#9ca3af"    # gray when off
            },
        }
    },
    # NEW: bottom-center cumulative score
    "score": {
        "enabled": True,
        "value_scale": 0.07,        # ↓ reduced from 0.10 (smaller font/pills)
        "caption_gap_px": 4,        # ↓ tighter spacing from 6
        "caption": {"fill": "#9fb0c9", "size": 11, "weight": "normal"},
        "pos_color": "#22c55e",     # green
        "neg_color": "#ef4444",     # red
        "zero_color": "#e6eeff",    # neutral light
        # backdrop pill behind the number (same style as temps)
        "backdrop": {
            "enabled": True,
            "pad_px": 3,            # ↓ less padding from 6
            "radius_px": 6,         # ↓ smaller pill from 10
            "alpha": 160,
            "fill_rgb": (0, 0, 0)
        },
        "y_frac": 0.88,             # vertical position as a fraction of height
        "prefix": "Σ",              # leading symbol
        "unit_suffix": "€",         # optional suffix; set "" to hide
        "decimals": 2,
        "zero_eps": 1e-7,
    },
    "clock": {
        "enabled": True,
        "value_scale": 0.075,   # fraction of min(w,h)
        "y_frac": 0.08,         # vertical position (0..1 of height)
        "text_color": "#e6eeff",
        "emoji": "⏰",           # change to "" to hide, or pick any ASCII like "[time]"
        "backdrop": { "enabled": True, "pad_px": 6, "radius_px": 10, "alpha": 160, "fill_rgb": (0, 0, 0) },
    },
    "feature_flags": {
        # Only these two are considered now:
        "show_soc": True,
        "show_temps": True,
        "show_hvac": True,        # NEW: enable HVAC gauge
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
    # Timing
    timestamp: Optional[object] = None
    # Temps
    T_inside: Optional[float] = None
    T_outside: Optional[float] = None
    # Comfort band
    comfort_target_C: Optional[float] = None
    comfort_tolerance_C: Optional[float] = None
    # Battery
    soc: Optional[float] = None
    # NEW: HVAC telemetry
    hvac_elec_kw: Optional[float] = None   # electrical input (+kW draw)
    hvac_heat_kw: Optional[float] = None   # thermal output (+kW heat, −kW cool)
    hvac_cop: Optional[float] = None       # optional; when available
    hvac_nameplate_kw: Optional[float] = None  # optional; for % utilization
    # NEW: reward components
    cumulative_score: Optional[float] = None
    comfort_score: Optional[float] = None  # ← NEW
    financial_score: Optional[float] = None  # ← NEW


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
        self._img_refs = []  # hold ImageTk.PhotoImage objects (backdrops)
        # background image cache (original + last resized PhotoImage)
        self._bg_img_orig = None    # PIL.Image | None
        self._bg_img_tk = None      # ImageTk.PhotoImage | None
        self._bg_img_tk_size = (0, 0)  # (w, h) of the last scaled image
        if image_path and Image is not None:
            try:
                p = Path(image_path)
                if p.is_file():
                    self._bg_img_orig = Image.open(p).convert("RGBA")
            except Exception:
                self._bg_img_orig = None
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
        self._img_refs.clear()

        self._bg(w, h)
        self._draw_center_image(w, h)   # centered, aspect-ratio preserved (no stretch)
        self._draw_clock(w, h, d)

        # temps in center: T_out on left half, T_in on right half
        if self._config["feature_flags"]["show_temps"]:
            self._draw_center_temps(w, h, d)

        # SOC gauge in corner
        if self._config["feature_flags"]["show_soc"] and d.soc is not None:
            self._draw_soc(w, h, float(d.soc))

        # HVAC gauge (mirrors SOC in the opposite corner)
        if self._config["feature_flags"].get("show_hvac", False):
            self._draw_hvac(w, h, d)

        # NEW: bottom-center cumulative score
        self._draw_score(w, h, d)

        # NEW: left/right reward breakdown
        self._draw_reward_scores(w, h, d)

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

    def _draw_center_image(self, w: int, h: int) -> None:
        """Draw the house image centered, preserving aspect ratio, never stretching.
        Scales down to fit within max_w_frac/max_h_frac of the canvas."""
        cfg = self._config.get("overlay_image", {})
        if not cfg or not cfg.get("enabled", True):
            return
        if self._bg_img_orig is None or ImageTk is None or ImageOps is None:
            return

        max_w = max(1, int(float(cfg.get("max_w_frac", 0.7)) * w))
        max_h = max(1, int(float(cfg.get("max_h_frac", 0.7)) * h))

        # Compute target box and resize with high-quality filter, preserving AR.
        # We only scale DOWN (contain). If the source is smaller than the box, keep original size.
        src = self._bg_img_orig
        target_w = min(max_w, src.width)
        target_h = min(max_h, src.height)
        box_w, box_h = max(1, target_w), max(1, target_h)

        # Use contain to avoid stretching; if image already smaller, this returns it unchanged.
        scaled = ImageOps.contain(src, (box_w, box_h), method=Image.LANCZOS)

        # Cache the PhotoImage by size to avoid rebuilding every frame.
        need_new = (self._bg_img_tk is None) or (self._bg_img_tk_size != (scaled.width, scaled.height))
        if need_new:
            self._bg_img_tk = ImageTk.PhotoImage(scaled)
            self._bg_img_tk_size = (scaled.width, scaled.height)

        cx = w // 2
        cy = h // 2 + int(cfg.get("y_offset_px", 0))
        # Draw behind text that follows (we already drew gradient; image sits above gradient, below labels).
        self.canvas.create_image(cx, cy, image=self._bg_img_tk, anchor="center")

    def _text_with_backdrop(self, x: int, y: int, text: str, *, fill: str, font, anchor="c"):
        # draw once to measure
        tmp = self.canvas.create_text(x, y, text=text, anchor=anchor, fill=fill, font=font)
        bbox = self.canvas.bbox(tmp)  # (x0, y0, x1, y1)
        self.canvas.delete(tmp)
        if not bbox:
            return

        x0, y0, x1, y1 = bbox
        cfg = self._config["temps"]["backdrop"]
        if cfg.get("enabled", True) and Image is not None and ImageTk is not None:
            pad  = int(cfg["pad_px"])
            rad  = int(cfg["radius_px"])
            alpha = int(cfg["alpha"])
            r, g, b = cfg["fill_rgb"]

            w = (x1 - x0) + pad * 2
            h = (y1 - y0) + pad * 2

            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([0, 0, w-1, h-1], rad, fill=(r, g, b, alpha))

            tkimg = ImageTk.PhotoImage(img)
            self._img_refs.append(tkimg)  # prevent GC

            # place backdrop aligned with the text's bbox
            # compute top-left of the padded rect based on anchor
            ax, ay = x0 - pad, y0 - pad
            self.canvas.create_image(ax, ay, image=tkimg, anchor="nw")

        # draw the final text on top
        self.canvas.create_text(x, y, text=text, anchor=anchor, fill=fill, font=font)

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
        self._text_with_backdrop(x_left, y, t_out_txt,
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
        self._text_with_backdrop(x_right, y, t_in_txt,
                                 fill=t_in_color, font=("", value_font_size, "bold"))
        self.canvas.create_text(x_right, y + value_font_size + gap,
                                text="inside", anchor="n",
                                fill=units["fill"], font=("", units["size"], units["weight"]))

    def _corner_xy(self, corner: str, r: int, pad: int, w: int, h: int) -> tuple[int, int]:
        if corner == "top_left":
            return r + pad, r + pad
        if corner == "top_right":
            return w - r - pad, r + pad
        if corner == "bottom_left":
            return r + pad, h - r - pad
        return w - r - pad, h - r - pad

    def _draw_soc(self, w: int, h: int, soc: float) -> None:
        cfg = self._config["gauges"]["soc"]
        m = min(w, h)
        r = max(8, int(cfg["radius_frac"] * m))
        pad = int(cfg["pad_frac"] * m)

        corner = str(cfg.get("corner", "top_left"))
        cx, cy = self._corner_xy(corner, r, pad, w, h)

        frac = max(0.0, min(1.0, float(soc)))
        color = self._soc_color(frac, cfg)
        self._gauge(cx, cy, r, frac, label=f"SOC {int(frac*100):d}%", color=color)

    def _gauge(self, cx: int, cy: int, r: int, frac: float, label: str = "", color: Optional[str] = None) -> None:
        cfg = self._config["gauges"]["soc"]
        start = int(cfg["start_deg"])
        span = int(cfg["span_deg"])
        extent = int(span * max(0.0, min(1.0, frac)))

        # thickness in px from radius
        thickness = max(6, int(float(cfg.get("thickness_frac", 0.12)) * r))

        # background arc
        self.canvas.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=start, extent=span, style="arc",
            outline=cfg.get("bg", "#5a6780"), width=thickness
        )

        # foreground arc (colored by SOC)
        fg = color or cfg.get("fg", "#22c55e")
        self.canvas.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=start, extent=extent, style="arc",
            outline=fg, width=thickness
        )

        if label:
            base = int(max(cfg["label"].get("size", 10), float(cfg.get("font_scale", 0.20)) * r))
            
            # Use the same translucent black pill background
            backdrop_cfg = self._config["temps"]["backdrop"]
            
            # Use the same helper used for temperature and score labels
            self._text_with_backdrop(
                cx, cy + 2, label,
                fill=color,  # use the SOC arc color
                font=("", base, cfg["label"]["weight"]),
                anchor="c"
            )

    def _soc_color(self, frac: float, cfg: dict) -> str:
        th = cfg.get("thresholds", {"low": 0.20, "mid": 0.60})
        cols = cfg.get("colors", {"low": "#ef4444", "mid": "#f59e0b", "high": "#22c55e"})
        if frac <= float(th.get("low", 0.20)):
            return cols.get("low", "#ef4444")
        if frac <= float(th.get("mid", 0.60)):
            return cols.get("mid", "#f59e0b")
        return cols.get("high", "#22c55e")

    def _draw_hvac(self, w: int, h: int, d: HouseRenderData) -> None:
        cfg = self._config["gauges"]["hvac"]
        m = min(w, h)
        r = max(8, int(cfg["radius_frac"] * m))
        pad = int(cfg["pad_frac"] * m)
        cx, cy = self._corner_xy(str(cfg.get("corner", "top_right")), r, pad, w, h)

        # read values with safe fallbacks
        q_kw = float(d.hvac_heat_kw) if d.hvac_heat_kw is not None else 0.0
        p_kw = d.hvac_elec_kw
        cop  = d.hvac_cop

        # mode & color
        if abs(q_kw) < 1e-6 and (p_kw is None or p_kw < 1e-6):
            mode = "idle"
        else:
            mode = "heat" if q_kw >= 0 else "cool"
        color = cfg["colors"].get(mode, "#9ca3af")

        # normalize arc fill: show utilization vs plausible size
        # Use nameplate if available, else default to 5.0 kW
        denom = max(1e-6, float(d.hvac_nameplate_kw)) if getattr(d, "hvac_nameplate_kw", None) else 5.0
        frac = max(0.0, min(1.0, abs(q_kw) / denom))

        # label text (two lines)
        sign = "+" if q_kw >= 0 else "−"
        top_line = f"{sign}{abs(q_kw):.1f} kW" if (d.hvac_heat_kw is not None) else "—"
        if p_kw is not None and cop is not None and cop == cop:  # cop==cop filters NaN
            bottom_line = f"{p_kw:.1f} kW • COP {cop:.1f}"
        elif p_kw is not None:
            bottom_line = f"{p_kw:.1f} kW"
        else:
            bottom_line = "—"

        # draw base/active arcs
        start = int(cfg["start_deg"]); span = int(cfg["span_deg"])
        thickness = max(6, int(float(cfg.get("thickness_frac", 0.12)) * r))
        # background ring
        self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r,
                               start=start, extent=span, style="arc",
                               outline=cfg.get("bg", "#5a6780"), width=thickness)
        # foreground
        extent = int(span * frac)
        self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r,
                               start=start, extent=extent, style="arc",
                               outline=color, width=thickness)

        # labels (centered), reuse pill backdrop
        base = int(max(10, float(cfg.get("font_scale", 0.20)) * r))
        small = max(8, int(base * 0.60))
        self._text_with_backdrop(cx, cy - int(base*0.15),
                                 top_line, fill=color, font=("", base, "bold"))
        self._text_with_backdrop(cx, cy + int(base*0.85),
                                 bottom_line, fill="#e6eeff", font=("", small, "normal"))

    def _draw_clock(self, w: int, h: int, d: HouseRenderData) -> None:
        cfg = self._config.get("clock", {})
        if not cfg or not cfg.get("enabled", True):
            return
        ts = getattr(d, "timestamp", None)
        if ts is None:
            return

        # Coerce to datetime safely
        try:
            import datetime as _dt
            if isinstance(ts, _dt.datetime):
                dtv = ts
            else:
                dtv = _dt.datetime.fromisoformat(str(ts))
        except Exception:
            return

        hhmm = dtv.strftime("%H:%M")
        emoji = str(cfg.get("emoji", "⏰"))
        display = f"{emoji} {hhmm}" if emoji else hhmm

        m = min(w, h)
        value_font_size = max(12, int(float(cfg.get("value_scale", 0.075)) * m))
        x = w // 2
        y = int(float(cfg.get("y_frac", 0.08)) * h)

        # fallback to temps backdrop if not present
        if "backdrop" not in cfg:
            cfg["backdrop"] = self._config["temps"]["backdrop"]

        self._text_with_backdrop(
            x, y, display,
            fill=str(cfg.get("text_color", "#e6eeff")),
            font=("", value_font_size, "bold"),
        )

    def _draw_score(self, w: int, h: int, d: HouseRenderData) -> None:
        cfg = self._config.get("score", {})
        if not cfg or not cfg.get("enabled", True):
            return
        if d.cumulative_score is None:
            return

        m = min(w, h)
        value_font_size = max(12, int(float(cfg.get("value_scale", 0.10)) * m))
        gap = int(cfg.get("caption_gap_px", 6))
        y = int(float(cfg.get("y_frac", 0.88)) * h)
        x = w // 2

        val = float(d.cumulative_score)
        eps = float(cfg.get("zero_eps", 1e-7))
        if val > eps:
            color = cfg.get("pos_color", "#22c55e")
        elif val < -eps:
            color = cfg.get("neg_color", "#ef4444")
        else:
            color = cfg.get("zero_color", "#e6eeff")

        # text like: "Σ +12.34€"
        decimals = int(cfg.get("decimals", 2))
        prefix = str(cfg.get("prefix", "Σ")).strip()
        unit = str(cfg.get("unit_suffix", "€"))
        signed = f"{val:+.{decimals}f}"
        display_text = f"{prefix} {signed}{unit}" if unit else f"{prefix} {signed}"

        # caption above the value
        cap = cfg.get("caption", {"fill": "#9fb0c9", "size": 11, "weight": "normal"})
        self.canvas.create_text(x, y - value_font_size - gap,
                                text="Cumulative score",
                                anchor="s",
                                fill=cap.get("fill", "#9fb0c9"),
                                font=("", int(cap.get("size", 11)), cap.get("weight", "normal")))

        # draw value with translucent black pill
        # allow score to have its own backdrop settings; fallback to temps backdrop if missing
        if "backdrop" not in cfg:
            cfg["backdrop"] = self._config["temps"]["backdrop"]
        self._text_with_backdrop(x, y, display_text,
                                 fill=color, font=("", value_font_size, "bold"))

    def _draw_reward_scores(self, w: int, h: int, d: HouseRenderData) -> None:
        cfg = self._config.get("score", {})
        if not cfg or not cfg.get("enabled", True):
            return

        m = min(w, h)
        value_font_size = max(10, int(float(cfg.get("value_scale", 0.07)) * m))
        gap = int(cfg.get("caption_gap_px", 6))
        y = int(float(cfg.get("y_frac", 0.88)) * h)

        # Left = Comfort | Right = Financial
        side_x = {
            "comfort": int(w * 0.18),    # left
            "financial": int(w * 0.82),  # right
        }

        # Define caption and values
        scores = {
            "comfort": {
                "value": d.comfort_score,
                "label": "Comfort",
                "color": cfg.get("pos_color", "#22c55e") if (d.comfort_score or 0) >= 0 else cfg.get("neg_color", "#ef4444"),
            },
            "financial": {
                "value": d.financial_score,
                "label": "Financial",
                "color": cfg.get("pos_color", "#22c55e") if (d.financial_score or 0) >= 0 else cfg.get("neg_color", "#ef4444"),
            }
        }

        for key, meta in scores.items():
            val = meta["value"]
            if val is None:
                continue
            x = side_x[key]
            signed = f"{val:+.3f}€"
            self.canvas.create_text(x, y - value_font_size - gap,
                                    text=meta["label"],
                                    anchor="s",
                                    fill=cfg["caption"]["fill"],
                                    font=("", cfg["caption"]["size"], cfg["caption"]["weight"]))
            self._text_with_backdrop(x, y, signed,
                                     fill=meta["color"],
                                     font=("", value_font_size, "bold"))


__all__ = ["HouseRenderer", "HouseRenderData", "CONFIG"]
