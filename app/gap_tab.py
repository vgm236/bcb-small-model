"""'Output gap' tab of the desktop application: history (smoothed and filtered estimates, the indicators' cycles,
the BCB's published estimates) and the projection ahead with bands, with the numbers in a table."""
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog


def gap_chart(fig, gap, q0, horizon, title="Output gap (% of potential output)"):
    ax = fig.add_subplot(111)
    g = gap.copy()
    x = pd.PeriodIndex(g.index, freq="Q").to_timestamp(how="end")
    hist = g[g["source"] != "projection"]
    xh = pd.PeriodIndex(hist.index, freq="Q").to_timestamp(how="end")
    if "smoothed_sd" in g.columns:
        ax.fill_between(xh, hist["smoothed"] - 1.64 * hist["smoothed_sd"], hist["smoothed"] + 1.64 * hist["smoothed_sd"], color="C0", alpha=0.10, label="90% band (history)")
    proj = g[g["source"] == "projection"]
    xp = pd.PeriodIndex(proj.index, freq="Q").to_timestamp(how="end")
    if "p5" in g.columns and proj["p5"].notna().any():
        ax.fill_between(xp, proj["p5"], proj["p95"], color="C0", alpha=0.12, label="90% band (projection)")
        ax.fill_between(xp, proj["p25"], proj["p75"], color="C0", alpha=0.25, label="50% band (projection)")
    ax.plot(xh, hist["smoothed"], color="C0", lw=2.2, label="gap, smoothed estimate")
    ax.plot(xh, hist["filtered"], color="C0", lw=1.0, ls=":", label="gap, filtered (real time)")
    ax.plot(xp, proj["gap"], color="C0", lw=2.2, ls="--", label="gap, projection")
    for c, lab, col in [("fpib", "GDP cycle (HP)", "C7"), ("nuci_implied", "capacity utilisation, implied", "C2"), ("emp_implied", "unemployment, implied", "C1"), ("caged_implied", "formal jobs, implied", "C4")]:
        if c in g.columns and hist[c].notna().any():
            ax.plot(xh, hist[c], color=col, lw=0.9, alpha=0.8, label=lab)
    if "bcb" in g.columns and g["bcb"].notna().any():
        b = g["bcb"].dropna()
        ax.plot(pd.PeriodIndex(b.index, freq="Q").to_timestamp(how="end"), b.values, marker="s", color="C3", ms=6, ls="none", label="BCB published estimate")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(pd.Period(q0, freq="Q").to_timestamp(how="end"), color="grey", lw=0.7)
    if horizon in g.index:
        ax.plot(pd.Period(horizon, freq="Q").to_timestamp(how="end"), g.loc[horizon, "gap"], marker="D", color="C0", ms=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel("%"); ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)


class GapTab(ttk.Frame):
    def __init__(self, master, app, chart_cls, table_cls):
        super().__init__(master)
        self.app = app
        self.res = None
        top = ttk.Frame(self); top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Output gap: estimate from the model's filter (history) and projection ahead", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Run / refresh (uses the 'Next meeting' settings)", command=self.run).pack(side="right", padx=4)
        ttk.Button(top, text="Save gap table (CSV)", command=self.save).pack(side="right", padx=4)
        self.info_var = tk.StringVar(value="Run a projection on the 'Next meeting' tab (or press Run here) to see the output gap.")
        ttk.Label(self, textvariable=self.info_var, wraplength=1400, justify="left").pack(fill="x", padx=6)
        body = ttk.Frame(self, width=max(900, master.winfo_screenwidth() - 60), height=max(500, master.winfo_screenheight() - 230))
        body.pack(fill="both", expand=True, padx=6, pady=6)
        body.pack_propagate(False)           # the wide table scrolls instead of pushing the chart past the window
        self.chart = chart_cls(body, figsize=(12.0, 4.2)); self.chart.pack(fill="both", expand=True)
        self.table = table_cls(body, height=9); self.table.pack(fill="x", pady=4)
        # the chart is usually drawn while this tab is hidden; redraw at the final size when the tab is shown/resized
        self.chart.canvas.get_tk_widget().bind("<Configure>", lambda e: self.after(60, self._redraw))

    def _redraw(self):
        try:
            self.chart.fig.tight_layout()
            self.chart.canvas.draw()
        except Exception:  # noqa: BLE001
            pass

    def run(self):
        nt = self.app.tab_next
        if nt.res is not None and nt.res.get("gap") is not None:
            self.show(nt.res)
        else:
            nt.run(callback=lambda res: self.show(res))

    def show(self, res):
        g = res.get("gap")
        if g is None:
            return
        self.res = res
        q0 = res["q0"]
        horizon = res.get("horizon", "")
        fig = self.chart.clear()
        gap_chart(fig, g.loc["2012Q1":], q0, horizon, title="Output gap (%% of potential output), meeting of %s" % res.get("decision", ""))
        self.chart.draw()
        hist = g[g["source"] != "projection"]
        proj = g[g["source"] == "projection"]
        parts = []
        if len(hist):
            parts.append("Origin %s: smoothed gap %.2f%% (filtered %.2f%%)" % (q0, hist["smoothed"].iloc[-1], hist["filtered"].iloc[-1]))
        if "bcb" in g.columns and g["bcb"].notna().any():
            b = g["bcb"].dropna()
            parts.append("BCB's latest published estimate %.1f%% (%s)" % (b.iloc[-1], b.index[-1]))
        if horizon in proj.index:
            parts.append("projection %.2f%% at %s (relevant horizon)" % (proj.loc[horizon, "gap"], horizon))
        if len(proj):
            parts.append("end of horizon %.2f%% (%s)" % (proj["gap"].iloc[-1], proj.index[-1]))
        self.info_var.set("; ".join(parts) + ". Indicators: GDP HP cycle, capacity utilisation, unemployment and formal jobs divided by their estimated loadings; "
                          "the smoothed estimate uses the whole sample, the filtered one only past data; the projection follows the IS curve under the scenario's real-rate path.")
        cols = [c for c in ["smoothed", "filtered", "bcb", "fpib", "nuci_implied", "emp_implied", "caged_implied", "gap", "p25", "p75", "p5", "p95"] if c in g.columns]
        names = {"smoothed": "gap (smoothed)", "filtered": "gap (filtered)", "bcb": "BCB published", "fpib": "GDP cycle", "nuci_implied": "NUCI implied",
                 "emp_implied": "unemployment implied", "caged_implied": "Caged implied", "gap": "gap (projection)", "p25": "25%", "p75": "75%", "p5": "5%", "p95": "95%"}
        show = pd.concat([hist.tail(8), proj])[cols].rename(columns=names).round(2)
        self.table.set(show.T, first_width=150, width=66)

    def save(self):
        if self.res is None or self.res.get("gap") is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="output_gap.csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.res["gap"].to_csv(path)
