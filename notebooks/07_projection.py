# %% [markdown]
# # 07 – Conditional inflation projection for a Copom meeting (reference scenario)
#
# Runs the estimated model the way the BCB runs its reference scenario for a Copom meeting or a Monetary Policy
# Report: Selic path from the Focus survey (per-meeting medians, then end-year medians), exchange rate starting at
# the 10-business-day PTAX average rounded to R$0.05 and following PPP, Brent following the spot price and the
# futures-curve proxy for six months and +2% a year afterwards, administered prices as an outside input (the Copom's
# projection when available, otherwise the Focus medians), neutral real rate of 5.00%, climate term from the latest
# Niño 3.4 readings. The current quarter is nowcast with the Focus monthly medians. Output: the Copom-style table
# (calendar years and the four-quarter accumulation six quarters ahead), the quarterly path with uncertainty bands,
# and the assumptions used.
#
# Parameters are set in the next cell. Set REFRESH=True to re-download the inputs (Focus, Selic, PTAX, IPCA, ...).

# %%
import sys, pathlib, warnings, json
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fetch, projection as pj, model_system as ms
from config import OUTPUT

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

REFRESH = False                 # re-download the inputs before projecting
DECISION_DATE = "2026-11-04"    # decision day of the meeting being prepared (282nd meeting, 3-4 November 2026)
FOCUS_DATE = None               # Friday before the meeting week; None -> latest survey available
FX_START = None                 # R$/US$; None -> Copom convention from the PTAX (10 business days to FOCUS_DATE, rounded)
ADM_YEARS = None                # {year: %}; None -> Focus medians for "IPCA Administrados" (this and the next two years)
ADM_HORIZON = None              # (quarter, %) four-quarter administered projection at the relevant horizon, optional
CLIMATE = "realtime"            # 'realtime', 'off', 'realized', or a pd.Series of ONI anomalies by quarter
NEUTRAL_RATE = 5.0              # % p.a. (None -> HP trend of the ex-ante real rate)
N_DRAWS = 500                   # simulations for the uncertainty bands (0 = none)
H = 12                          # projection horizon in quarters

if REFRESH:
    pj.refresh_inputs()

# %% [markdown]
# ## Assumptions

# %%
if FOCUS_DATE is None:
    FOCUS_DATE = fetch.focus_selic_reunioes()["Data"].max()
F = pd.Timestamp(FOCUS_DATE)
D = pd.Timestamp(DECISION_DATE)
if FX_START is None:
    FX_START, win = pj.fx_start_convention(F)
    print("exchange-rate start R$%.2f/US$ (PTAX average %s..%s = %.4f)" % (FX_START, win.index[0].date(), win.index[-1].date(), win.mean()))
snap = pj.focus_snapshot(F)
q0 = D.to_period("Q")
if ADM_YEARS is None:
    ADM_YEARS = {y: snap["adm_annual"][y] for y in [q0.year, q0.year + 1, q0.year + 2] if y in snap["adm_annual"]}
    print("administered prices from the Focus survey (%s):" % snap["selic_date"].date(), ADM_YEARS)
print("Focus %s: Selic per meeting %s" % (snap["selic_date"].date(), snap["selic_meetings"]))
print("Focus end-year Selic %s; IPCA %s; GDP %s" % (snap["selic_annual"], snap["ipca_annual"], snap["gdp_annual"]))

# %% [markdown]
# ## Projection

# %%
SETTINGS = dict(pj.ADOPTED)                      # adopted configuration (gap anchor, coefficients, climate index, ...)
SETTINGS.update(neutral_rate=NEUTRAL_RATE, climate=CLIMATE)
print("settings:", SETTINGS)
out = pj.project(DECISION_DATE, FOCUS_DATE, FX_START, ADM_YEARS, ADM_HORIZON, H=H, n_draws=N_DRAWS, verbose=True, **SETTINGS)
p, v, t = out["paths"], out["vintage"], out["table"]
horizon = str(q0 + 6)
a0 = out["a0"]
print("\nState at the origin (%s): output gap %.2f%%, neutral real rate %.2f%%, 12-month inflation expectations %.2f%%, expected Selic 12m %.2f%%"
      % (q0, a0[ms.iv["h"]], 100 * np.expm1((a0[ms.iv["rrIS"]] + v["df"]["rr_trend"].iloc[-1]) / 100), p["pie_F"], p["ie_F"]))
print("Brent at the Focus date %.1f US$/bbl (STEO vintage %s); ONI nowcast %.2f" % (p["brent_monthly"].loc[F.to_period("M")], p["steo_vintage"], v["df"]["oni"].iloc[-1]))
print("Nowcast months (Focus headline, administered assumption):")
print(v["monthly_prices"][v["monthly_prices"]["source"] != "IBGE"].round(2).to_string())
table = pj.copom_table(out, years=[q0.year, q0.year + 1, q0.year + 2], horizon=horizon)
print("\nProjections in the Copom's format (IPCA accumulated over the calendar year / four quarters, %), UNADJUSTED model output:")
print(table.round(1).to_string())
# adjusted output: minus the model's error (model minus Copom) at the previous meeting, by type of target
import adjust, copom_meetings as cm
n_next, _ = cm.next_meeting(D - pd.Timedelta(days=1))
errs, prev_n = adjust.previous_errors(n_next if n_next else 10 ** 6, cm.all_meetings())
table_adj = None
if errs is not None:
    table_adj = table.copy()
    for idx in table_adj.index:
        typ = "relevant horizon" if "Q" in idx else ("current year" if idx == str(q0.year) else "next year")
        table_adj.loc[idx, "IPCA"] = table.loc[idx, "IPCA"] - adjust.error_for_type(errs, typ, "ipca")
        table_adj.loc[idx, "IPCA livres"] = table.loc[idx, "IPCA livres"] - adjust.error_for_type(errs, typ, "livres")
    print("\nADJUSTED output (minus the model's error at meeting %d: %s):" % (prev_n, {k: round(v["ipca"], 2) for k, v in errs.items()}))
    print(table_adj.round(1).to_string())
    t["pi_4q_adjusted"] = adjust.adjust_path(t, q0, errs, horizon_q=horizon)
else:
    print("\n(no stored backtest result for a previous meeting: run notebooks/08 to enable the adjusted output)")
print("\nQuarterly path:")
cols = ["piL_sa", "piL", "piA", "pi", "pi_4q", "piL_4q", "piA_4q", "h", "pie", "selic_pct", "de", "source"]
print(t[cols].round(2).to_string())
if "bands" in out:
    print("\nUncertainty bands for the four-quarter IPCA (%):"); print(out["bands"].round(2).to_string())

tag = D.strftime("%Y%m%d")
table.to_csv(OUTPUT / ("07_projection_%s_table.csv" % tag))
if table_adj is not None:
    table_adj.to_csv(OUTPUT / ("07_projection_%s_table_adjusted.csv" % tag))
t.to_csv(OUTPUT / ("07_projection_%s_quarterly.csv" % tag))
if "bands" in out:
    out["bands"].to_csv(OUTPUT / ("07_projection_%s_bands.csv" % tag))
assump = {"decision_date": str(D.date()), "focus_date": str(F.date()), "fx_start": FX_START, "adm_years": ADM_YEARS, "adm_horizon": ADM_HORIZON,
          "neutral_rate": NEUTRAL_RATE, "climate": CLIMATE if isinstance(CLIMATE, str) else "explicit path", "steo_vintage": p["steo_vintage"],
          "selic_path_quarterly": {str(k): round(float(x), 3) for k, x in v["selic_q"].loc[q0:].items()},
          "ptax_path_quarterly": {str(k): round(float(x), 4) for k, x in v["ptax_q"].loc[q0:].items()},
          "brent_monthly": {str(k): round(float(x), 2) for k, x in p["brent_monthly"].loc[F.to_period("M"):].items()},
          "adm_monthly_pct": {str(k): round(float(x), 3) for k, x in p["adm_monthly_pct"].items()},
          "oni_path": {str(k): round(float(x), 3) for k, x in p["oni_path"].items()},
          "focus_12m_expectation": p["pie_F"], "expected_selic_12m": p["ie_F"], "cutoffs": {k: str(x) for k, x in v["cut"].items() if k != "D"}}
(OUTPUT / ("07_projection_%s_assumptions.json" % tag)).write_text(json.dumps(assump, indent=1, ensure_ascii=False), encoding="utf-8")

# %% [markdown]
# ## Chart

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
x = t.index.to_timestamp()
proj = t[t["source"] == "projection"]
xp = proj.index.to_timestamp()
ax = axes[0]
if "bands" in out:
    b = out["bands"]
    ax.fill_between(xp, b["p5"], b["p95"], color="C0", alpha=0.12, label="90%")
    ax.fill_between(xp, b["p25"], b["p75"], color="C0", alpha=0.25, label="50%")
ax.plot(x, t["pi_4q"], color="C0", lw=2, label="IPCA, 4 quarters (unadjusted)")
if "pi_4q_adjusted" in t.columns:
    ax.plot(x, t["pi_4q_adjusted"], color="C0", lw=1.5, ls="--", label="IPCA, 4 quarters (adjusted by previous error)")
ax.plot(x, t["piL_4q"], color="C1", lw=1.2, ls=":", label="free prices")
ax.plot(x, t["piA_4q"], color="C2", lw=1.2, ls="--", label="administered (assumption)")
ax.axhline(3.0, color="grey", lw=0.7, ls="--"); ax.axhline(4.5, color="grey", lw=0.5, ls=":"); ax.axhline(1.5, color="grey", lw=0.5, ls=":")
ax.axvline(q0.to_timestamp(how="end"), color="grey", lw=0.6)
hq = pd.Period(horizon, freq="Q")
ax.plot(hq.to_timestamp(), t.loc[hq, "pi_4q"], marker="D", color="C0", ms=7)
ax.set_title("Inflation, accumulated in 4 quarters (%)"); ax.legend(fontsize=8)
ax = axes[1]
ax.plot(x, t["selic_pct"], color="C3", lw=2, label="Selic (Focus path)")
ax.plot(x, 100 * np.expm1(t["pie"] / 100), color="C4", lw=1.5, label="12-month inflation expectations")
ax.axvline(q0.to_timestamp(how="end"), color="grey", lw=0.6); ax.legend(fontsize=8); ax.set_title("Policy rate and expectations (% p.a.)")
ax = axes[2]
ax.plot(x, t["h"], color="C5", lw=2, label="output gap (%)")
ax.plot(x, 100 * np.expm1(4 * t["piL_sa"] / 100), color="C1", lw=1.2, label="free prices, SA annualised (%)")
ax.axhline(0, color="k", lw=0.5); ax.axvline(q0.to_timestamp(how="end"), color="grey", lw=0.6); ax.legend(fontsize=8)
ax.set_title("Output gap and quarterly free-price inflation")
for ax in axes:
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
fig.suptitle("Reference scenario for the Copom meeting of %s (Focus of %s, exchange rate from %.2f BRL/USD, neutral rate %s%%)" % (D.date(), F.date(), FX_START, NEUTRAL_RATE))
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / ("07_projection_%s.png" % tag), dpi=110)
# %% [markdown]
# ## Output gap: history (smoothed and filtered), indicator-implied gaps, BCB published estimates and projection

# %%
gap = out["gap"]
print(gap.loc[q0 - 6:, ["smoothed", "filtered", "bcb", "fpib", "nuci_implied", "emp_implied", "caged_implied", "gap"] + ([c for c in ["p5", "p95"] if c in gap.columns])].round(2).to_string())
gap.to_csv(OUTPUT / ("07_projection_%s_gap.csv" % tag))
import sys as _sys; _sys.path.insert(0, str(ROOT / "app"))
from gap_tab import gap_chart
fig = plt.figure(figsize=(13, 5))
gap_chart(fig, gap.loc["2012Q1":], str(q0), horizon, title="Output gap (%% of potential output), meeting of %s" % D.date())
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / ("07_gap_%s.png" % tag), dpi=110)
print("saved outputs with tag", tag)
