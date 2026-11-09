"""
Native desktop interface (Tkinter, no browser) for the Copom-style projections of the BCB small-scale model.

Tabs
* Next meeting      – reference scenario for the next scheduled Copom meeting: unadjusted output and output
                      adjusted by the model's error at the previous meeting, bands, assumptions;
* Scenario builder  – own paths for the Selic and the BRL/USD by quarter, administered inflation by year and the
                      12-month expectation, compared with the reference scenario;
* Administered      – the BCB's administered-price model: coefficients versus the BCB's, fit, item projections;
* Backtest          – model versus the Copom at every meeting since July 2025 (re-run with new minutes);
* Data and updates  – refresh the inputs, update the Copom history, data stamps.

Launch with "BCB model app.bat" (uses the project's virtual environment; no console window) or
`python app/desktop_app.py`. Long computations run in a worker thread; the window stays responsive.
"""
import sys
import os
import json
import queue
import threading
import traceback
import pathlib
import datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = ROOT / "app" / "desktop_app.log"

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import projection as pj          # noqa: E402
import copom_meetings as cm      # noqa: E402
import adjust                    # noqa: E402
import backtest as bt            # noqa: E402
import fetch                     # noqa: E402
import model_system as ms        # noqa: E402
import administered as adm       # noqa: E402
import adm_projection as admp    # noqa: E402
from config import OUTPUT, DATA_RAW  # noqa: E402
from help_text import HelpTab      # noqa: E402
from gap_tab import GapTab         # noqa: E402

ADM_SOURCES = {
    "Annual figures (Copom / Focus)": {"adm_source": "input"},
    "Model, BCB calibration, after 4 quarters": {"adm_source": "model", "adm_theta": "bcb", "adm_handover": 4},
    "Model, BCB calibration, from next quarter": {"adm_source": "model", "adm_theta": "bcb", "adm_handover": 0},
    "Model, re-estimated, after 4 quarters": {"adm_source": "model", "adm_theta": "estimated", "adm_handover": 4},
    "Model, re-estimated, from next quarter": {"adm_source": "model", "adm_theta": "estimated", "adm_handover": 0},
}
FLAGS = ["verde", "amarela", "vermelha p1", "vermelha p2"]
ITEM_NAMES = {"5104001": "gasolina", "2202003": "energia elétrica", "6203": "plano de saúde", "6101": "farmacêuticos", "5102004": "emplacamento",
              "2101004": "água e esgoto", "2201004": "gás de botijão", "5101001": "ônibus urbano", "band": "flag contribution",
              "adm": "administered (aggregate)", "ipca": "headline used"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return ("%%.%df" % nd) % v
    return str(v)


class Table(ttk.Frame):
    """A read-only table (Treeview) filled from a DataFrame."""

    def __init__(self, master, height=8, **kw):
        super().__init__(master, **kw)
        self.tree = ttk.Treeview(self, show="headings", height=height)
        self.grid_propagate(True)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew"); vsb.grid(row=0, column=1, sticky="ns"); hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1); self.columnconfigure(0, weight=1)

    def set(self, df, index=True, nd=2, width=90, first_width=120):
        self.tree.delete(*self.tree.get_children())
        if df is None or len(df) == 0:
            self.tree["columns"] = ["(empty)"]
            self.tree.heading("(empty)", text="")
            return
        cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
        self.tree["columns"] = cols
        for k, c in enumerate(cols):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=first_width if (k == 0 and index) else width, anchor="e" if k > 0 or not index else "w", stretch=False)
        for idx, row in df.iterrows():
            vals = ([str(idx)] if index else []) + [fmt(v, nd) for v in row.values]
            self.tree.insert("", "end", values=vals)


class Chart(ttk.Frame):
    def __init__(self, master, figsize=(8, 4.2), **kw):
        super().__init__(master, **kw)
        self.fig = Figure(figsize=figsize, dpi=96)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear(self):
        self.fig.clear()
        return self.fig

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw_idle()


def projection_chart(fig, res_table, bands, q0, horizon, errors=None, damping=1.0, compare=None, title=""):
    ax = fig.add_subplot(111)
    t = res_table
    x = pd.PeriodIndex(t.index, freq="Q").to_timestamp(how="end")
    proj = t[t["source"] == "projection"]
    if bands is not None and len(bands):
        xb = pd.PeriodIndex(bands.index, freq="Q").to_timestamp(how="end")
        ax.fill_between(xb, bands["p5"], bands["p95"], color="C0", alpha=0.12, label="90% band")
        ax.fill_between(xb, bands["p25"], bands["p75"], color="C0", alpha=0.25, label="50% band")
    ax.plot(x, t["pi_4q"], color="C0", lw=2.2, label="IPCA, 4 quarters (unadjusted)")
    if errors is not None:
        tt = t.copy(); tt.index = pd.PeriodIndex(tt.index, freq="Q")
        adj = adjust.adjust_path(tt, pd.Period(q0, freq="Q"), errors, damping=damping, horizon_q=horizon)
        ax.plot(x, adj.values, color="C0", lw=1.6, ls="--", label="IPCA, 4 quarters (adjusted)")
    ax.plot(x, t["piL_4q"], color="C1", lw=1.2, ls=":", label="free prices")
    ax.plot(x, t["piA_4q"], color="C2", lw=1.2, ls=":", label="administered")
    if compare is not None:
        xc = pd.PeriodIndex(compare.index, freq="Q").to_timestamp(how="end")
        ax.plot(xc, compare["pi_4q"], color="grey", lw=1.8, label="reference scenario")
    ax.axhline(3.0, color="grey", lw=0.7, ls="--"); ax.axhline(4.5, color="lightgrey", lw=0.6, ls=":"); ax.axhline(1.5, color="lightgrey", lw=0.6, ls=":")
    ax.axvline(pd.Period(q0, freq="Q").to_timestamp(how="end"), color="grey", lw=0.7)
    if horizon in t.index:
        ax.plot(pd.Period(horizon, freq="Q").to_timestamp(how="end"), t.loc[horizon, "pi_4q"], marker="D", color="C0", ms=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel("% accumulated in 4 quarters"); ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)


def copom_style(res, years, horizon):
    rows = {}
    for y in years:
        if y in res["years"]:
            v = res["years"][y]
            rows[str(y)] = {"IPCA": v["pi"], "IPCA livres": v["piL"], "IPCA administrados": v["piA"]}
    t = res["table"]
    if horizon in t.index:
        rows[horizon + " (4q)"] = {"IPCA": t.loc[horizon, "pi_4q"], "IPCA livres": t.loc[horizon, "piL_4q"], "IPCA administrados": t.loc[horizon, "piA_4q"]}
    return pd.DataFrame(rows).T


def adjusted_style(table_df, errors, year0, damping=1.0):
    adj = table_df.copy()
    for idx in adj.index:
        typ = "relevant horizon" if "4q" in idx else ("current year" if idx == str(year0) else "next year")
        adj.loc[idx, "IPCA"] = table_df.loc[idx, "IPCA"] - damping * adjust.error_for_type(errors, typ, "ipca")
        adj.loc[idx, "IPCA livres"] = table_df.loc[idx, "IPCA livres"] - damping * adjust.error_for_type(errors, typ, "livres")
    return adj


def slim(out):
    t = out["table"].copy(); t.index = t.index.astype(str)
    a0 = out["a0"]
    res = {"table": t, "years": out["years"], "q0": str(out["q0"]), "settings": out["settings"],
           "state": {"gap": float(a0[ms.iv["h"]]), "pie": float(100 * np.expm1(a0[ms.iv["pie"]] / 100)),
                     "neutral": float(100 * np.expm1((a0[ms.iv["rrIS"]] + out["vintage"]["df"]["rr_trend"].iloc[-1]) / 100)),
                     "ie": out["paths"]["ie_F"], "oni": float(out["vintage"]["df"]["oni"].iloc[-1])},
           "selic_q": {str(k): float(v) for k, v in out["vintage"]["selic_q"].items()},
           "ptax_q": {str(k): float(v) for k, v in out["vintage"]["ptax_q"].items()},
           "brent_F": float(out["paths"]["brent_monthly"].loc[out["paths"]["snapshot"]["F"].to_period("M")]),
           "steo": out["paths"]["steo_vintage"], "nowcast": out["vintage"]["monthly_prices"][out["vintage"]["monthly_prices"]["source"] != "IBGE"].round(3),
           "bands": None, "adm_items": None, "feedback": out.get("feedback_log", []), "gap": None, "decision": out["paths"]["snapshot"]["F"].strftime("%Y-%m-%d")}
    if out.get("gap") is not None:
        g = out["gap"].copy(); g.index = g.index.astype(str); res["gap"] = g
    if "bands" in out:
        b = out["bands"].copy(); b.index = b.index.astype(str); res["bands"] = b
    if out.get("adm_model") is not None:
        it = out["adm_model"]["quarterly"].copy(); it.index = it.index.astype(str); res["adm_items"] = it
    return res


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self, selftest=False):
        super().__init__()
        self.title("BCB small-scale model: Copom projections")
        self.geometry("1400x860")
        self.minsize(1000, 650)
        try:
            self.state("zoomed")          # fill the screen on Windows
        except tk.TclError:
            pass
        self.selftest = selftest
        self.queue = queue.Queue()
        self.model = None
        self.meetings = cm.all_meetings()
        self.busy = False
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=22)
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 8))
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        self.status_var = tk.StringVar(value="Ready.")
        bar = ttk.Frame(self); bar.pack(fill="x", padx=6, pady=4)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")
        self.settings = SettingsPanel(self)
        self.tab_next = NextMeetingTab(self.nb, self)
        self.tab_scen = ScenarioTab(self.nb, self)
        self.tab_adm = AdministeredTab(self.nb, self)
        self.tab_back = BacktestTab(self.nb, self)
        self.tab_data = DataTab(self.nb, self)
        self.tab_help = HelpTab(self.nb, self)
        self.tab_gap = GapTab(self.nb, self, Chart, Table)
        for tab, name in [(self.tab_next, "Next meeting"), (self.tab_gap, "Output gap"), (self.tab_scen, "Scenario builder"), (self.tab_adm, "Administered prices"),
                          (self.tab_back, "Backtest"), (self.tab_data, "Data and updates"), (self.tab_help, "How it works")]:
            self.nb.add(tab, text=name)
        self.after(150, self._poll)
        self.after(300, self.tab_next.load_defaults)
        if selftest:
            self.after(1500, self._selftest)

    # -- worker thread plumbing -------------------------------------------------
    def run_async(self, fn, on_done, label="Working..."):
        if self.busy:
            messagebox.showinfo("Busy", "A computation is already running; please wait for it to finish.")
            return
        self.busy = True
        self.status_var.set(label)
        self.progress.start(12)

        def worker():
            try:
                res = fn()
                self.queue.put((on_done, res, None))
            except Exception as e:  # noqa: BLE001
                self.queue.put((on_done, None, (e, traceback.format_exc())))
        threading.Thread(target=worker, daemon=True).start()

    def post_status(self, text):
        self.queue.put((None, text, None))

    def _poll(self):
        try:
            while True:
                on_done, res, err = self.queue.get_nowait()
                if on_done is None:
                    self.status_var.set(str(res))
                    continue
                self.busy = False
                self.progress.stop()
                if err is not None:
                    self.status_var.set("Error: %s" % err[0])
                    LOG.write_text(err[1], encoding="utf-8")
                    messagebox.showerror("Error", "%s\n\n(details in app/desktop_app.log)" % err[0])
                else:
                    self.status_var.set("Done.")
                    try:
                        on_done(res)
                    except Exception as e:  # noqa: BLE001
                        LOG.write_text(traceback.format_exc(), encoding="utf-8")
                        messagebox.showerror("Error", "%s\n\n(details in app/desktop_app.log)" % e)
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def get_model(self):
        if self.model is None:
            self.model = pj.load_model()
        return self.model

    def _wait_idle(self, fn, delay=300):
        if self.busy:
            self.after(delay, lambda: self._wait_idle(fn, delay))
        else:
            self.after(delay, fn)

    def _shot(self, name):
        """Screenshot of this window only (self-test), saved in output/figs. Candidate rectangles: the operating
        system's rectangle of the top-level frame, then Tk's geometry; the first one that lies inside the screen
        grab is used, so that nothing outside the window is captured."""
        try:
            from PIL import ImageGrab
            self.deiconify()
            try:
                self.state("zoomed")
            except tk.TclError:
                pass
            self.lift(); self.update_idletasks(); self.update()
            full = ImageGrab.grab()
            cands = []
            try:
                import ctypes
                from ctypes import wintypes
                hwnd = int(self.wm_frame(), 16)
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    cands.append(("win32", (rect.left, rect.top, rect.right, rect.bottom)))
            except Exception:  # noqa: BLE001
                pass
            x, y, w, h = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
            cands.append(("tk", (x, y, x + w, y + h)))
            chosen = None
            for how, (l, t, r, b) in cands:
                l2, t2, r2, b2 = max(0, l), max(0, t), min(full.width, r), min(full.height, b)
                if r2 - l2 > 200 and b2 - t2 > 200:
                    chosen = (how, (l2, t2, r2, b2)); break
            p = OUTPUT / "figs" / ("desktop_%s.png" % name)
            if chosen is None:
                print("screenshot skipped: no valid window rectangle", cands, "grab", full.size)
                return
            full.crop(chosen[1]).save(p)
            print("screenshot", p, chosen, "candidates", cands, "grab", full.size, "state", self.state())
        except Exception as e:  # noqa: BLE001
            print("screenshot failed:", e)

    def _selftest(self):
        """Exercise the tabs in sequence (projection, scenario, administered estimation, backtest view) and exit."""
        def step_next(res):
            print("selftest ok:", res["table"].loc[res["q0"]:, ["pi_4q", "piL_4q", "piA_4q"]].round(2).tail(8).to_string())
            if res.get("gap") is not None:
                print("gap:", res["gap"].loc[res["q0"]:, ["smoothed", "filtered", "gap"]].head(7).round(2).to_string())
            self.after(800, lambda: (self._shot("next_meeting"), step_gap()))

        def step_gap():
            self.nb.select(self.tab_gap)
            self.after(1200, lambda: (self._shot("gap"), step_scen()))

        def step_scen():
            self.nb.select(self.tab_scen)
            self.tab_scen.load_reference()
            self._wait_idle(step_scen_run, 500)

        def step_scen_run():
            for k, (lab, e1, e2, ref, e3) in enumerate(self.tab_scen.rows):
                if 1 <= k <= 4:
                    e3.insert(0, "0.5")
            self.tab_scen.pie_shock_var.set("0.5")
            self.tab_scen.run()
            self._wait_idle(lambda: (self._shot("scenario"), step_adm()), 800)

        def step_adm():
            self.nb.select(self.tab_adm)
            self.tab_adm.load()
            self._wait_idle(lambda: (self._shot("administered"), step_back()), 800)

        def step_back():
            self.nb.select(self.tab_back)
            self.after(1200, lambda: (self._shot("backtest"), step_help()))

        def step_help():
            self.nb.select(self.tab_help)
            self.after(600, lambda: (self._shot("help"), self.after(400, self.destroy)))
        self.tab_next.run(callback=step_next, draws=0)


class SettingsPanel:
    """Shared model settings (variables only; the widgets are built by the tabs that show them)."""

    def __init__(self, app):
        self.neutral = tk.DoubleVar(value=float(pj.ADOPTED["neutral_rate"] or 5.0))
        self.climate = tk.StringVar(value=pj.ADOPTED["climate"])
        self.gap_anchor = tk.BooleanVar(value=pj.ADOPTED["gap_anchor"] == "rpm")
        self.coefficients = tk.StringVar(value=pj.ADOPTED["coefficients"])
        self.adm_source = tk.StringVar(value=list(ADM_SOURCES)[0])
        self.flag = tk.StringVar(value="verde")
        self.draws = tk.IntVar(value=300)
        self.damping = tk.DoubleVar(value=1.0)

    def build(self, master):
        f = ttk.LabelFrame(master, text="Model settings")
        r = 0
        ttk.Label(f, text="Neutral real rate (% p.a.)").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(f, from_=2.0, to=9.0, increment=0.25, textvariable=self.neutral, width=8).grid(row=r, column=1, sticky="w", padx=4); r += 1
        ttk.Label(f, text="Climate after the current quarter").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(f, values=["realtime", "off", "realized"], textvariable=self.climate, width=12, state="readonly").grid(row=r, column=1, sticky="w", padx=4); r += 1
        ttk.Checkbutton(f, text="Anchor the output gap on the BCB's published estimates", variable=self.gap_anchor).grid(row=r, column=0, columnspan=2, sticky="w", padx=4); r += 1
        ttk.Label(f, text="Coefficients").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(f, values=["estimated", "bcb"], textvariable=self.coefficients, width=12, state="readonly").grid(row=r, column=1, sticky="w", padx=4); r += 1
        ttk.Label(f, text="Administered prices").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(f, values=list(ADM_SOURCES), textvariable=self.adm_source, width=34, state="readonly").grid(row=r, column=1, sticky="w", padx=4); r += 1
        ttk.Label(f, text="Tariff flag assumed for December").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(f, values=FLAGS, textvariable=self.flag, width=12, state="readonly").grid(row=r, column=1, sticky="w", padx=4); r += 1
        ttk.Label(f, text="Simulation draws for the bands").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(f, from_=0, to=2000, increment=100, textvariable=self.draws, width=8).grid(row=r, column=1, sticky="w", padx=4); r += 1
        ttk.Label(f, text="Weight of the previous-meeting error (adjusted output)").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.25, textvariable=self.damping, width=8).grid(row=r, column=1, sticky="w", padx=4); r += 1
        return f

    def kwargs(self):
        kw = dict(pj.ADOPTED)
        kw.update(neutral_rate=float(self.neutral.get()), climate=self.climate.get(), gap_anchor="rpm" if self.gap_anchor.get() else None,
                  coefficients=self.coefficients.get())
        kw.update(ADM_SOURCES[self.adm_source.get()])
        return kw


def meeting_defaults():
    n, d = cm.next_meeting()
    if n is None:
        n, d = cm.next_meeting(pd.Timestamp.today() - pd.Timedelta(days=60))
    F = pd.Timestamp(fetch.focus_selic_reunioes()["Data"].max())
    fx, win = pj.fx_start_convention(F)
    snap = pj.focus_snapshot(F)
    q0 = d.to_period("Q")
    adm_y = {y: round(snap["adm_annual"][y], 2) for y in [q0.year, q0.year + 1, q0.year + 2] if y in snap["adm_annual"]}
    return {"n": n, "decision": str(d.date()), "focus": str(F.date()), "fx_start": fx, "fx_window": (str(win.index[0].date()), str(win.index[-1].date()), float(win.mean())),
            "adm_years": adm_y, "snap": snap, "q0": str(q0), "horizon": cm.relevant_horizon(d)}


class NextMeetingTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.dft = None
        self.res = None
        left = ttk.Frame(self); left.pack(side="left", fill="y", padx=6, pady=6)
        right = ttk.Frame(self, width=max(700, master.winfo_screenwidth() - 760), height=max(600, master.winfo_screenheight() - 150))
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right.pack_propagate(False)          # the result widgets adapt to the panel instead of widening it past the screen
        app.settings.build(left).pack(fill="x", pady=(0, 6))
        info = ttk.LabelFrame(left, text="Meeting and assumptions"); info.pack(fill="x", pady=(0, 6))
        self.info_var = tk.StringVar(value="loading...")
        ttk.Label(info, textvariable=self.info_var, justify="left", wraplength=400).grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        ttk.Label(info, text="Exchange-rate start (BRL/USD)").grid(row=1, column=0, sticky="w", padx=4)
        self.fx_var = tk.DoubleVar(value=5.0)
        ttk.Entry(info, textvariable=self.fx_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(info, text="Administered prices (% per year, annual-figure source)").grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(6, 0))
        self.adm_vars = {}
        self.adm_frame = ttk.Frame(info); self.adm_frame.grid(row=3, column=0, columnspan=4, sticky="w", padx=4)
        ttk.Label(info, text="12-month inflation expectation (%)").grid(row=4, column=0, sticky="w", padx=4)
        self.pie_var = tk.StringVar(value="")
        ttk.Entry(info, textvariable=self.pie_var, width=8).grid(row=4, column=1, sticky="w")
        btns = ttk.Frame(left); btns.pack(fill="x")
        ttk.Button(btns, text="Run projection", command=self.run).pack(side="left", padx=2)
        ttk.Button(btns, text="Reload defaults", command=self.load_defaults).pack(side="left", padx=2)
        ttk.Button(btns, text="Save quarterly path (CSV)", command=self.save).pack(side="left", padx=2)
        self.focus_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.focus_var, justify="left", wraplength=400, style="Small.TLabel").pack(fill="x", pady=4)
        # right: tables and chart
        tabs = ttk.Frame(right); tabs.pack(fill="x")
        ttk.Label(tabs, text="Model, unadjusted", style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(tabs, text="Adjusted by the previous meeting's error", style="Title.TLabel").grid(row=0, column=1, sticky="w", padx=4)
        self.tab_un = Table(tabs, height=5); self.tab_un.grid(row=1, column=0, sticky="nsew", padx=4)
        self.tab_adj = Table(tabs, height=5); self.tab_adj.grid(row=1, column=1, sticky="nsew", padx=4)
        tabs.columnconfigure(0, weight=1, uniform="t"); tabs.columnconfigure(1, weight=1, uniform="t")
        self.err_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.err_var, justify="left", wraplength=740, style="Small.TLabel").pack(fill="x", padx=4)
        self.state_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.state_var, justify="left", wraplength=740).pack(fill="x", padx=4, pady=2)
        self.chart = Chart(right, figsize=(7.6, 3.9)); self.chart.pack(fill="both", expand=True)
        self.items_table = Table(right, height=6); self.items_table.pack(fill="x", padx=4, pady=4)

    def load_defaults(self):
        def done(d):
            self.dft = d
            self.info_var.set("Next meeting: %dth, %s. Focus survey: %s. Relevant horizon: %s (six quarters ahead).\nExchange-rate convention: PTAX average %s to %s = %.4f, rounded to %.2f."
                              % (d["n"], d["decision"], d["focus"], d["horizon"], d["fx_window"][0], d["fx_window"][1], d["fx_window"][2], d["fx_start"]))
            self.fx_var.set(d["fx_start"])
            for w in self.adm_frame.winfo_children():
                w.destroy()
            self.adm_vars = {}
            for k, (y, v) in enumerate(d["adm_years"].items()):
                ttk.Label(self.adm_frame, text=str(y)).grid(row=0, column=2 * k, padx=2)
                var = tk.DoubleVar(value=float(v)); self.adm_vars[y] = var
                ttk.Entry(self.adm_frame, textvariable=var, width=7).grid(row=0, column=2 * k + 1, padx=2)
            self.pie_var.set("%.2f" % d["snap"]["pie_12m"])
            s = d["snap"]
            self.focus_var.set("Focus Selic per meeting (%s): %s\nFocus end-year Selic %s; IPCA %s" % (
                s["selic_date"].date(), ", ".join("%s %.2f" % (k, v) for k, v in sorted(s["selic_meetings"].items(), key=lambda kv: (kv[0].split("/")[1], int(kv[0].split("/")[0][1:])))),
                s["selic_annual"], {k: round(v, 2) for k, v in s["ipca_annual"].items()}))
        self.app.run_async(meeting_defaults, done, "Loading the next meeting and the Focus survey...")

    def params(self, draws=None):
        d = self.dft
        kw = self.app.settings.kwargs()
        adm_years = {y: float(v.get()) for y, v in self.adm_vars.items()}
        flag = None
        if kw["adm_source"] == "model":
            y0 = int(d["decision"][:4])
            flag = {y: self.app.settings.flag.get() for y in range(y0, y0 + 4)}
        pie = None
        try:
            pv = float(self.pie_var.get())
            if abs(pv - d["snap"]["pie_12m"]) > 1e-6:
                pie = pv
        except ValueError:
            pass
        return dict(decision_date=d["decision"], focus_date=d["focus"], fx_start=float(self.fx_var.get()), adm_years=adm_years, adm_horizon=None, H=12,
                    n_draws=int(self.app.settings.draws.get()) if draws is None else draws, flag_december=flag, pie_override=pie, **kw)

    def run(self, callback=None, draws=None):
        if self.dft is None:
            messagebox.showinfo("Please wait", "The meeting data are still loading.")
            return
        p = self.params(draws)
        model = self.app.get_model

        def work():
            self.app.post_status("Running the model for the meeting of %s..." % p["decision_date"])
            out = pj.project(model=model(), **p)
            return slim(out)

        def done(res):
            self.res = res
            self.show(res)
            if callback:
                callback(res)
        self.app.run_async(work, done, "Running the model...")

    def show(self, res):
        d = self.dft
        res["horizon"] = d["horizon"]
        res["decision"] = d["decision"]
        year0 = int(res["q0"][:4])
        years = [year0, year0 + 1, year0 + 2]
        tab = copom_style(res, years, d["horizon"])
        self.tab_un.set(tab.round(2), first_width=96, width=74)
        if getattr(self.app, "tab_gap", None) is not None:
            self.app.tab_gap.show(res)
        errs, prev_n = adjust.previous_errors(d["n"], self.app.meetings)
        if errs is not None:
            self.tab_adj.set(adjusted_style(tab, errs, year0, float(self.app.settings.damping.get())).round(2), first_width=96, width=74)
            self.err_var.set("Adjustment uses the model's error at meeting %d (model minus Copom): %s" % (prev_n, {k: round(v["ipca"], 2) for k, v in errs.items()}))
        else:
            self.tab_adj.set(None)
            self.err_var.set("No stored backtest result for a previous meeting: run the backtest once to enable the adjusted output.")
        s = res["state"]
        self.state_var.set("State at the origin %s: output gap %.2f%%, neutral real rate %.2f%%, 12-month expectations %.2f%%, expected Selic 12 months %.2f%%, ONI %.2f. "
                           "Brent at the Focus date %.1f US$/bbl (STEO %s). Settings: %s" % (res["q0"], s["gap"], s["neutral"], s["pie"], s["ie"], s["oni"], res["brent_F"], res["steo"],
                                                                                           {k: v for k, v in res["settings"].items() if v is not None}))
        fig = self.chart.clear()
        projection_chart(fig, res["table"], res["bands"], res["q0"], d["horizon"], errs, float(self.app.settings.damping.get()),
                         title="Inflation projection, reference scenario (meeting %d)" % d["n"])
        self.chart.draw()
        if res["adm_items"] is not None:
            it = res["adm_items"]
            cols = [c for c in ITEM_NAMES if c in it.columns]
            self.items_table.set(it.loc[res["q0"]:, cols].rename(columns=ITEM_NAMES).round(2).T, nd=2, first_width=160, width=70)
        else:
            self.items_table.set(res["table"].loc[res["q0"]:, ["piL_sa", "piL", "piA", "pi", "pi_4q", "piL_4q", "piA_4q", "h", "pie", "selic_pct"]].round(2).T, first_width=90, width=70)

    def save(self):
        if self.res is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="projection_%s.csv" % self.dft["decision"], filetypes=[("CSV", "*.csv")])
        if path:
            self.res["table"].to_csv(path)
            self.app.status_var.set("Saved %s" % path)


class ScenarioTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.ref = None
        self.res = None
        left = ttk.Frame(self); left.pack(side="left", fill="y", padx=6, pady=6)
        right = ttk.Frame(self, width=max(700, master.winfo_screenwidth() - 700), height=max(600, master.winfo_screenheight() - 150))
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right.pack_propagate(False)
        ttk.Label(left, text="Own assumptions (edit the cells; blank keeps the reference value)", style="Title.TLabel").pack(anchor="w")
        grid = ttk.Frame(left); grid.pack(anchor="w", pady=4)
        ttk.Label(grid, text="quarter").grid(row=0, column=0, padx=3); ttk.Label(grid, text="Selic (% p.a.)").grid(row=0, column=1, padx=3)
        ttk.Label(grid, text="BRL/USD").grid(row=0, column=2, padx=3); ttk.Label(grid, text="exp. shock (pp)").grid(row=0, column=3, padx=3)
        ttk.Label(grid, text="reference Selic / BRL").grid(row=0, column=4, padx=3)
        self.rows = []
        for k in range(9):
            lab = ttk.Label(grid, text=""); lab.grid(row=k + 1, column=0, padx=3)
            e1 = ttk.Entry(grid, width=9); e1.grid(row=k + 1, column=1, padx=3, pady=1)
            e2 = ttk.Entry(grid, width=9); e2.grid(row=k + 1, column=2, padx=3, pady=1)
            e3 = ttk.Entry(grid, width=9); e3.grid(row=k + 1, column=3, padx=3, pady=1)
            ref = ttk.Label(grid, text=""); ref.grid(row=k + 1, column=4, padx=3)
            self.rows.append((lab, e1, e2, ref, e3))
        inf = ttk.Frame(left); inf.pack(anchor="w", pady=4)
        ttk.Label(inf, text="Administered prices (% per year)").grid(row=0, column=0, columnspan=6, sticky="w")
        self.adm_vars = {}
        self.adm_frame = ttk.Frame(inf); self.adm_frame.grid(row=1, column=0, columnspan=6, sticky="w")
        ttk.Label(inf, text="12-month inflation expectation at the origin (%)").grid(row=2, column=0, sticky="w", pady=2)
        self.pie_var = tk.StringVar(value=""); ttk.Entry(inf, textvariable=self.pie_var, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(inf, text="Exchange-rate start (BRL/USD)").grid(row=3, column=0, sticky="w", pady=2)
        self.fx_var = tk.StringVar(value=""); ttk.Entry(inf, textvariable=self.fx_var, width=8).grid(row=3, column=1, sticky="w")
        ttk.Label(inf, text="Shock to the Focus 12-month expectation at the origin (pp)").grid(row=4, column=0, sticky="w", pady=2)
        self.pie_shock_var = tk.StringVar(value="0"); ttk.Entry(inf, textvariable=self.pie_shock_var, width=8).grid(row=4, column=1, sticky="w")
        ttk.Label(left, text="Selic changes apply from the day after the meeting, exchange-rate levels from the Focus date; the last edited value is carried forward. "
                  "Expectation shocks are in percentage points relative to the Focus survey: the origin shock shifts the expectation the model starts from; "
                  "the quarterly shocks keep the model's expectations that far above or below their own path in those quarters (blank = 0, not carried forward).",
                  wraplength=430, style="Small.TLabel").pack(anchor="w", pady=2)
        btns = ttk.Frame(left); btns.pack(anchor="w", pady=4)
        ttk.Button(btns, text="Load reference scenario", command=self.load_reference).pack(side="left", padx=2)
        ttk.Button(btns, text="Run scenario", command=self.run).pack(side="left", padx=2)
        ttk.Button(btns, text="Save scenario path (CSV)", command=self.save).pack(side="left", padx=2)
        self.note_var = tk.StringVar(value="Model settings are those of the 'Next meeting' tab.")
        ttk.Label(left, textvariable=self.note_var, wraplength=420, style="Small.TLabel").pack(anchor="w")
        tabs = ttk.Frame(right); tabs.pack(fill="x")
        ttk.Label(tabs, text="Own scenario, unadjusted", style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(tabs, text="Reference scenario, unadjusted", style="Title.TLabel").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(tabs, text="Own scenario, adjusted by the previous meeting's error", style="Title.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", padx=4)
        self.t_own = Table(tabs, height=5); self.t_own.grid(row=1, column=0, sticky="nsew", padx=4)
        self.t_ref = Table(tabs, height=5); self.t_ref.grid(row=1, column=1, sticky="nsew", padx=4)
        self.t_adj = Table(tabs, height=5); self.t_adj.grid(row=3, column=0, sticky="nsew", padx=4)
        tabs.columnconfigure(0, weight=1, uniform="s"); tabs.columnconfigure(1, weight=1, uniform="s")
        self.state_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.state_var, wraplength=740).pack(fill="x", padx=4, pady=2)
        self.chart = Chart(right, figsize=(7.6, 3.4)); self.chart.pack(fill="both", expand=True)

    def load_reference(self):
        nt = self.app.tab_next
        if nt.dft is None:
            messagebox.showinfo("Please wait", "The meeting data are still loading (tab 'Next meeting').")
            return
        p = nt.params(draws=0)
        model = self.app.get_model

        def work():
            out = pj.project(model=model(), **p)
            return slim(out)

        def done(res):
            self.ref = res
            q0 = pd.Period(res["q0"], freq="Q")
            for k, (lab, e1, e2, ref, e3) in enumerate(self.rows):
                qq = str(q0 + k)
                lab.config(text=qq)
                e1.delete(0, "end"); e2.delete(0, "end"); e3.delete(0, "end")
                ref.config(text="%.2f / %.3f" % (res["selic_q"].get(qq, np.nan), res["ptax_q"].get(qq, np.nan)))
            for w in self.adm_frame.winfo_children():
                w.destroy()
            self.adm_vars = {}
            for k, (y, v) in enumerate(nt.dft["adm_years"].items()):
                ttk.Label(self.adm_frame, text=str(y)).grid(row=0, column=2 * k, padx=2)
                var = tk.DoubleVar(value=float(v)); self.adm_vars[y] = var
                ttk.Entry(self.adm_frame, textvariable=var, width=7).grid(row=0, column=2 * k + 1, padx=2)
            self.pie_var.set("%.2f" % nt.dft["snap"]["pie_12m"])
            self.fx_var.set("%.2f" % nt.dft["fx_start"])
            self.t_ref.set(copom_style(res, [q0.year, q0.year + 1, q0.year + 2], nt.dft["horizon"]).round(2), first_width=90, width=76)
            self.note_var.set("Reference scenario loaded for meeting %d (%s)." % (nt.dft["n"], nt.dft["decision"]))
        self.app.run_async(work, done, "Running the reference scenario...")

    def run(self):
        if self.ref is None:
            self.load_reference()
            return
        nt = self.app.tab_next
        p = nt.params()
        q0 = pd.Period(self.ref["q0"], freq="Q")
        selic_ov, fx_ov, pie_shocks = {}, {}, {}
        for k, (lab, e1, e2, ref, e3) in enumerate(self.rows):
            qq = str(q0 + k)
            v1, v2, v3 = e1.get().strip(), e2.get().strip(), e3.get().strip()
            if v1:
                selic_ov[qq] = float(v1.replace(",", "."))
            if v2:
                fx_ov[qq] = float(v2.replace(",", "."))
            if v3 and k > 0:
                pie_shocks[qq] = float(v3.replace(",", "."))
        for ov, growth in [(selic_ov, 1.0), (fx_ov, (1 + 1.0 / 100.0) ** 0.25)]:
            if ov:
                last_q = max(ov, key=lambda q: pd.Period(q, freq="Q"))
                for k in range(1, 13):
                    qq = pd.Period(last_q, freq="Q") + k
                    if str(qq) not in ov and qq <= q0 + 12:
                        ov[str(qq)] = ov[last_q] * (growth ** k)
        p["selic_override"] = selic_ov or None
        p["fx_override"] = fx_ov or None
        p["pie_shock_path"] = pie_shocks or None
        try:
            p["pie_shock_origin"] = float(self.pie_shock_var.get().replace(",", ".") or 0.0)
        except ValueError:
            p["pie_shock_origin"] = 0.0
        p["adm_years"] = {y: float(v.get()) for y, v in self.adm_vars.items()}
        try:
            p["fx_start"] = float(self.fx_var.get().replace(",", "."))
            pv = float(self.pie_var.get().replace(",", "."))
            p["pie_override"] = None if abs(pv - nt.dft["snap"]["pie_12m"]) < 1e-6 else pv
        except ValueError:
            pass
        model = self.app.get_model

        def work():
            out = pj.project(model=model(), **p)
            return slim(out)

        def done(res):
            self.res = res
            year0 = q0.year
            tab = copom_style(res, [year0, year0 + 1, year0 + 2], nt.dft["horizon"])
            self.t_own.set(tab.round(2), first_width=90, width=76)
            errs, prev_n = adjust.previous_errors(nt.dft["n"], self.app.meetings)
            if errs is not None:
                self.t_adj.set(adjusted_style(tab, errs, year0, float(self.app.settings.damping.get())).round(2), first_width=90, width=76)
            s = res["state"]
            pie_path = {k: round(100 * np.expm1(v / 100), 2) for k, v in res["table"].loc[res["q0"]:, "pie"].head(7).items()}
            self.state_var.set("State at the origin: output gap %.2f%%, neutral rate %.2f%%, 12-month expectations %.2f%%, expected Selic 12 months %.2f%%. Overrides: Selic %s; BRL %s; "
                               "expectation shocks: origin %s pp, path %s. Expectations path (%%): %s"
                               % (s["gap"], s["neutral"], s["pie"], s["ie"], {k: round(v, 2) for k, v in list(selic_ov.items())[:6]} or "none", {k: round(v, 3) for k, v in list(fx_ov.items())[:6]} or "none",
                                  p.get("pie_shock_origin", 0.0), pie_shocks or "none", pie_path))
            fig = self.chart.clear()
            projection_chart(fig, res["table"], res["bands"], res["q0"], nt.dft["horizon"], errs, float(self.app.settings.damping.get()),
                             compare=self.ref["table"], title="Own scenario versus the reference scenario")
            self.chart.draw()
        self.app.run_async(work, done, "Running the scenario...")

    def save(self):
        if self.res is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="scenario.csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.res["table"].to_csv(path)


class AdministeredTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        top = ttk.Frame(self); top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="The BCB's administered-price model (June-2025 box): coefficients versus OLS estimates on the same regressors", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Re-download the item data and re-estimate", command=lambda: self.load(refresh=True)).pack(side="right", padx=4)
        ttk.Button(top, text="Estimate", command=lambda: self.load(refresh=False)).pack(side="right", padx=4)
        self.info_var = tk.StringVar(value="Press 'Estimate' to build the data and estimate the equations.")
        ttk.Label(self, textvariable=self.info_var, wraplength=1300).pack(fill="x", padx=6)
        body = ttk.Frame(self); body.pack(fill="both", expand=True, padx=6, pady=6)
        self.table = Table(body, height=22); self.table.pack(side="left", fill="both", expand=True)
        self.chart = Chart(body, figsize=(6.2, 4.4)); self.chart.pack(side="left", fill="both", expand=True, padx=6)

    def load(self, refresh=False):
        def work():
            data = admp.data_cached(refresh=refresh)
            tab, theta = adm.estimate(data)
            agg = adm.aggregate_history(data)
            q = data["q"]
            origin = q.index[-1] - 8
            proj = pd.period_range(origin + 1, q.index[-1], freq="Q")
            exo = q.loc[proj, ["ipca", "x", "fx", "cpi_us", "band"]]
            fitB = adm.fit_in_sample(adm.BCB_THETA, data, start="2016Q1")
            dd = pd.DataFrame({"BCB calibration": adm.simulate(adm.BCB_THETA, data, origin, len(proj), exo).loc[proj, "adm"],
                               "estimated": adm.simulate(theta, data, origin, len(proj), exo).loc[proj, "adm"],
                               "actual": fitB.loc[proj, "adm_actual"]})
            return {"tab": tab, "corr": float(agg.corr().iloc[0, 1]), "last": str(q.index[-1]), "T0": data["T0"], "dyn": dd, "origin": str(origin)}

        def done(r):
            t = r["tab"][["item", "parameter", "bcb", "estimate", "se", "t_diff", "sample", "n", "r2"]].copy()
            t.columns = ["item", "parameter", "BCB", "estimate", "s.e.", "(est-BCB)/s.e.", "sample", "n", "R²"]
            self.table.set(t.round(3), index=False, nd=3, width=80)
            self.info_var.set("Data through %s; reference tariff for the flag conversion %s R$/MWh (2024 prices); the 24 items reproduce the official administered index with correlation %.4f. "
                              "%d of %d parameters differ from the BCB's by more than two standard errors." % (r["last"], r["T0"], r["corr"], int((r["tab"]["t_diff"].abs() > 2).sum()), int(r["tab"]["t_diff"].notna().sum())))
            fig = self.chart.clear()
            ax = fig.add_subplot(111)
            dd = r["dyn"]
            x = dd.index.to_timestamp()
            ax.plot(x, dd["actual"], color="k", lw=2, label="actual")
            ax.plot(x, dd["BCB calibration"], marker="o", lw=1.3, label="dynamic simulation, BCB calibration")
            ax.plot(x, dd["estimated"], marker="o", lw=1.3, label="dynamic simulation, estimated")
            ax.set_title("Administered inflation, dynamic simulation from %s with realized drivers (quarterly log-%%)" % r["origin"], fontsize=9)
            ax.legend(fontsize=8); ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            self.chart.draw()
        self.app.run_async(work, done, "Building the administered-price data and estimating...")


class BacktestTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        top = ttk.Frame(self); top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Model versus the Copom's published projections (model minus Copom; adjusted = minus the previous meeting's error)", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Update Copom history and re-run the backtest", command=self.rerun).pack(side="right", padx=4)
        ttk.Button(top, text="Reload stored results", command=self.load).pack(side="right", padx=4)
        self.info_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.info_var, wraplength=1300).pack(fill="x", padx=6)
        body = ttk.Frame(self); body.pack(fill="both", expand=True, padx=6, pady=6)
        self.table = Table(body, height=14); self.table.pack(fill="x")
        self.summary = Table(body, height=5); self.summary.pack(fill="x", pady=4)
        self.chart = Chart(body, figsize=(12, 4.2)); self.chart.pack(fill="both", expand=True)
        self.after(600, self.load)

    def load(self):
        p = OUTPUT / "08_backtest_table.csv"
        if not p.exists():
            self.info_var.set("No stored backtest results yet.")
            return
        tab = pd.read_csv(p)
        if "model_ipca" not in tab.columns:
            self.info_var.set("The stored table comes from an earlier version; re-run the backtest.")
            return
        show = tab[["meeting", "decision", "target", "type", "copom_ipca", "model_ipca", "adjusted_ipca", "diff_ipca", "diff_adjusted_ipca", "copom_livres", "model_livres", "adjusted_livres", "copom_adm", "real_ipca"]].round(2)
        show.columns = ["meeting", "decision", "target", "type", "Copom IPCA", "model", "adjusted", "diff", "diff adj.", "Copom free", "model free", "adjusted free", "Copom adm.", "realized"]
        self.table.set(show, index=False, width=78)
        if (OUTPUT / "08_backtest_summary_adjusted.csv").exists():
            s = pd.read_csv(OUTPUT / "08_backtest_summary_adjusted.csv", header=[0, 1], index_col=0)
            s.columns = [" / ".join(c) for c in s.columns]
            self.summary.set(s.round(2), width=110, first_width=120)
        settings = json.loads((OUTPUT / "08_backtest_settings.json").read_text()) if (OUTPUT / "08_backtest_settings.json").exists() else {}
        self.info_var.set("Meetings: %s. Settings of the stored run: %s" % (", ".join(str(m["n"]) for m in self.app.meetings), settings))
        fig_path = OUTPUT / "figs" / "08_backtest_panels.png"
        if fig_path.exists():
            import matplotlib.image as mpimg
            fig = self.chart.clear()
            ax = fig.add_subplot(111)
            ax.imshow(mpimg.imread(str(fig_path)))
            ax.axis("off")
            self.chart.draw()

    def rerun(self):
        s = self.app.settings.kwargs()
        s = {k: s[k] for k in ["neutral_rate", "climate", "climate_index", "coefficients", "gap_anchor", "adm_profile"]}
        draws = int(self.app.settings.draws.get())
        model = self.app.get_model
        post = self.app.post_status

        def work():
            post("Checking the BCB site for new minutes...")
            meetings = cm.update_history(verbose=False)
            res = bt.run_backtest(meetings, settings=s, n_draws=draws, model=model(), progress=lambda k, n, m: post("Backtest: meeting %d (%d of %d)" % (m, k, n)))
            bt.plot_backtest(res, meetings, OUTPUT / "figs" / "08_backtest_panels.png")
            return meetings

        def done(meetings):
            self.app.meetings = meetings
            self.load()
            messagebox.showinfo("Backtest", "Backtest done for meetings %s. The adjusted output now uses the latest meeting's error." % ", ".join(str(m["n"]) for m in meetings))
        self.app.run_async(work, done, "Running the backtest...")


class DataTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        top = ttk.Frame(self); top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Inputs are cached in data/raw; the Copom history in data/processed/copom_history.json", style="Title.TLabel").pack(side="left")
        btns = ttk.Frame(self); btns.pack(fill="x", padx=6)
        ttk.Button(btns, text="Refresh the market and macro inputs (Focus, Selic, PTAX, prices, activity, Brent, climate)", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(btns, text="Update the Copom history from the minutes", command=self.update_history).pack(side="left", padx=4)
        ttk.Button(btns, text="Show data stamps", command=self.stamps).pack(side="left", padx=4)
        self.info_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.info_var, wraplength=1300).pack(fill="x", padx=6, pady=4)
        self.table = Table(self, height=14); self.table.pack(fill="x", padx=6, pady=6)
        self.after(800, self.stamps)

    def stamps(self):
        rows = []
        for name, f in [("Focus Selic per meeting", "focus_selic_reunioes.csv"), ("Focus 12-month inflation", "focus_ipca_12m_S.csv"),
                        ("Selic target (daily)", "sgs_432_selic_meta_daily.csv"), ("PTAX (daily)", "sgs_1_ptax_venda_daily.csv"), ("IPCA", "sgs_433_ipca.csv"),
                        ("IPCA subitems (SIDRA)", "sidra_ipca_subitems_adm.csv"), ("ANEEL tariff flags", "aneel_bandeiras.csv"), ("US CPI (BLS)", "bls_cpi_u.csv"),
                        ("IC-Br", "sgs_27574_icbr.csv"), ("NUCI", "sgs_28561_nuci_fgv_sa.csv"), ("PNAD unemployment", "sgs_24369_desocup_pnadc.csv"),
                        ("Caged stock", "sgs_28763_caged_estoque.csv"), ("Brent (EIA)", "eia_brent_daily.csv")]:
            p = DATA_RAW / f
            if p.exists():
                try:
                    s = pd.read_csv(p)
                    col = "Data" if "Data" in s.columns else ("month" if "month" in s.columns else s.columns[0])
                    rows.append({"series": name, "last observation": str(pd.to_datetime(s[col].astype(str).str[:10], errors="coerce").max().date()),
                                 "downloaded": dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
                except Exception:  # noqa: BLE001
                    rows.append({"series": name, "last observation": "?", "downloaded": dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
        self.table.set(pd.DataFrame(rows), index=False, width=200)
        n, d = cm.next_meeting()
        self.info_var.set("Next scheduled meeting: %s (%s); relevant horizon %s. Minutes stored for meetings %d to %d." % (
            n, d.date() if d is not None else "?", cm.relevant_horizon(d) if d is not None else "?", self.app.meetings[0]["n"], self.app.meetings[-1]["n"]))

    def refresh(self):
        def done(_):
            self.stamps()
            messagebox.showinfo("Data", "Inputs refreshed. Re-run the projections to use them.")
        self.app.run_async(pj.refresh_inputs, done, "Downloading the inputs...")

    def update_history(self):
        def done(meetings):
            self.app.meetings = meetings
            self.stamps()
            messagebox.showinfo("Copom history", "Meetings known: %s" % ", ".join(str(m["n"]) for m in meetings))
        self.app.run_async(lambda: cm.update_history(verbose=False), done, "Checking the BCB site for new minutes...")


def main():
    selftest = "--selftest" in sys.argv
    try:
        app = App(selftest=selftest)
        app.mainloop()
    except Exception:  # noqa: BLE001
        LOG.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            messagebox.showerror("BCB model app", "The application could not start; details in app/desktop_app.log")
        except Exception:  # noqa: BLE001
            pass
        raise


if __name__ == "__main__":
    main()
