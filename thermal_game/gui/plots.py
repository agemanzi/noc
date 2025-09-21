from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class Charts:
    def __init__(self, root):
        self.fig = Figure(figsize=(7,4), dpi=100)
        self.ax_T = self.fig.add_subplot(211)
        self.ax_E = self.fig.add_subplot(212)

        (self.l_Ti,) = self.ax_T.plot([], [], lw=2, label="T_inside")
        self.ax_T.legend(); self.ax_T.grid(True)

        (self.l_E,) = self.ax_E.plot([], [], lw=2, label="Electricity (kWh)")
        self.ax_E.legend(); self.ax_E.grid(True)

        self.t, self.Ti, self.E = [], [], []

        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update(self, step_result):
        s = step_result.state
        self.t.append(s.t); self.Ti.append(s.T_inside); self.E.append(s.kwh_used)
        self.l_Ti.set_data(self.t, self.Ti)
        self.l_E.set_data(self.t, self.E)
        for ax in (self.ax_T, self.ax_E):
            ax.relim(); ax.autoscale_view()
        self.canvas.draw_idle()
