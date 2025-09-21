import tkinter as tk
from ..engine.state import GameState, Action
from ..engine.simulation import SimulationEngine
from ..engine.recorder import GameRecorder
from .plots import Charts

class App:
    def __init__(self, root):
        self.engine = SimulationEngine(dt=1.0)
        self.rec = GameRecorder()
        self.state = GameState(t=0, T_inside=22.0, T_outside=30.0, soc=0.5, kwh_used=0.0)

        self.charts = Charts(root)

        # simple controls
        self.hvac = tk.DoubleVar(value=0.0)
        self.bat  = tk.IntVar(value=0)
        tk.Scale(root, from_=-1, to=1, resolution=0.1, orient="horizontal",
                 label="HVAC (-1 cool .. 1 heat)", variable=self.hvac).pack(fill="x")
        tk.Scale(root, from_=-1, to=1, resolution=1, orient="horizontal",
                 label="Battery (-1,0,1)", variable=self.bat).pack(fill="x")

        tk.Button(root, text="Export CSV", command=self.export).pack()

        self.tick()

    def tick(self):
        action = Action(hvac=self.hvac.get(), battery=int(self.bat.get()))
        step = self.engine.step(self.state, action)
        self.rec.append(self.state, action, step)
        self.state = step.state
        self.charts.update(step)
        root.after(100, self.tick)  # ~10 fps

    def export(self):
        self.rec.export_csv("run.csv")
        print("Saved run.csv")

if __name__ == "__main__":
    root = tk.Tk(); root.title("Thermal Game")
    App(root); root.mainloop()
