# %% [markdown]
# # 10 – The BCB's administered-price model: data, re-estimation, comparison with the BCB's calibration, projections
#
# The Monetary Policy Report of June 2025 (box "Atualização do modelo para projeção de médio prazo dos preços
# administrados", `refs/rpm202506b9p.pdf`) publishes 24 quarterly item equations, calibrated from the institutional
# rules of each item (fuels: Petrobras price in reais, ICMS, federal taxes, margins; electricity: Itaipu share and
# the tariff flag; health plans: the ANS rule with the IVDA; pharmaceuticals: the CMED rule; the other items indexed
# to past inflation with seasonal weights). `src/administered.py` implements the equations exactly as published
# (coefficients in `BCB_THETA`), builds the data from public sources (IBGE subitems, ANEEL flags, BLS CPI, EIA
# Brent, PTAX) and re-estimates every coefficient by OLS on the same regressors. Running this notebook downloads the
# data again (REFRESH), re-estimates, compares with the BCB's values, checks the fit and projects the quarters
# ahead inside the reference scenario of the next Copom meeting (with the feedback with free prices).

# %%
import sys, pathlib, warnings, json
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import administered as adm, adm_projection as admp, projection as pj, copom_meetings as cm, fetch, fetch_adm
from config import OUTPUT

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
REFRESH = (sys.argv[1] == "refresh") if len(sys.argv) > 1 else False

# %% [markdown]
# ## Data

# %%
data = adm.build_data(refresh=REFRESH)
q = data["q"]
print("items and drivers through %s; reference tariff for the flag conversion %s R$/MWh (2024 prices)" % (q.index[-1], data["T0"]))
agg = adm.aggregate_history(data)
d = agg["items_24"] - agg["official"]
print("24-item aggregate vs the official administered index (quarterly log-%%): corr %.4f, mean diff %.3f, sd %.3f" % (agg.corr().iloc[0, 1], d.mean(), d.std()))
print("ANS readjustments inferred from the health-plan item (log-%):", q.groupby(q.index.year)["R_plano"].first().dropna().round(2).to_dict())
print("CMED readjustments inferred from the pharmaceutical item (log-%):", {str(k): round(v, 2) for k, v in q["reaj_farm"].dropna().items()})
print("IVDA implied by the ANS rule:", adm.ivda_history(q).round(2).to_dict())
print("weights (last month, %):", data["weights_last"].round(2).to_dict())

# %% [markdown]
# ## Re-estimation and comparison with the BCB's calibration

# %%
tab, theta_hat = adm.estimate(data)
tab.to_csv(OUTPUT / "10_administered_estimates.csv", index=False)
print(tab.round(3).to_string())
sig = tab[(tab["t_diff"].abs() > 2)]
print("\nparameters whose estimate differs from the BCB's by more than two standard errors: %d of %d" % (len(sig), tab["t_diff"].notna().sum()))
print(sig[["item", "parameter", "bcb", "estimate", "se"]].round(3).to_string())

# %% [markdown]
# ## Fit: static (realized regressors) and dynamic (from eight quarters back, realized drivers)

# %%
fits = {}
for name, th in [("BCB calibration", adm.BCB_THETA), ("estimated", theta_hat)]:
    f = adm.fit_in_sample(th, data, start="2016Q1")
    e = (f["adm_fit"] - f["adm_actual"]).dropna()
    fits[name] = f
    print("%s: static fit of the aggregate, corr %.3f, RMSE %.2f (quarterly log-%%), 4-quarter RMSE %.2f" % (
        name, f[["adm_fit", "adm_actual"]].corr().iloc[0, 1], np.sqrt((e ** 2).mean()),
        np.sqrt(((f["adm_fit"].rolling(4).sum() - f["adm_actual"].rolling(4).sum()) ** 2).dropna().mean())))
origin = q.index[-1] - 8
proj = pd.period_range(origin + 1, q.index[-1], freq="Q")
exo = q.loc[proj, ["ipca", "x", "fx", "cpi_us", "band"]]
dyn = {}
for name, th in [("BCB calibration", adm.BCB_THETA), ("estimated", theta_hat)]:
    dyn[name] = adm.simulate(th, data, origin, len(proj), exo).loc[proj, "adm"]
actual = fits["BCB calibration"].loc[proj, "adm_actual"]
dd = pd.DataFrame(dict(dyn, actual=actual))
print("\ndynamic simulation from %s with realized drivers (quarterly log-%%):" % origin); print(dd.round(2).T.to_string())
print("4-quarter sums at the end: " + ", ".join("%s %.2f" % (k, v.tail(4).sum()) for k, v in dd.items()))
dd.to_csv(OUTPUT / "10_administered_dynamic_fit.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
f = fits["BCB calibration"]
axes[0].plot(f.index.to_timestamp(), f["adm_actual"].rolling(4).sum(), color="k", lw=2, label="actual")
for name in fits:
    axes[0].plot(f.index.to_timestamp(), fits[name]["adm_fit"].rolling(4).sum(), lw=1.3, label="static fit, " + name)
axes[0].set_title("Administered inflation, 4-quarter sums (log-%)"); axes[0].legend(fontsize=8)
axes[1].plot(dd.index.to_timestamp(), dd["actual"], color="k", lw=2, label="actual")
for name in dyn:
    axes[1].plot(dd.index.to_timestamp(), dd[name], lw=1.3, marker="o", label="dynamic, " + name)
axes[1].set_title("Dynamic simulation from %s (quarterly log-%%)" % origin); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "10_administered_fit.png", dpi=110)

# %% [markdown]
# ## Projection inside the reference scenario of the next Copom meeting (with the free-price feedback)

# %%
model = pj.load_model()
n_next, d_next = cm.next_meeting()
F = pd.Timestamp(fetch.focus_selic_reunioes()["Data"].max())
fx_start, _ = pj.fx_start_convention(F)
snap = pj.focus_snapshot(F)
q0 = d_next.to_period("Q")
adm_focus = {y: snap["adm_annual"][y] for y in [q0.year, q0.year + 1, q0.year + 2] if y in snap["adm_annual"]}
horizon = cm.relevant_horizon(d_next)
flag_now = str(data["flags"]["flag"].iloc[-1])
runs = {}
runs["input: Focus administered medians"] = pj.project(str(d_next.date()), str(F.date()), fx_start, adm_focus, None, H=12, model=model, **pj.ADOPTED)
for th in ["bcb", "estimated"]:
    runs["model: %s" % th] = pj.project(str(d_next.date()), str(F.date()), fx_start, None, None, H=12, model=model, adm_source="model", adm_theta=th,
                                        flag_december={q0.year: flag_now, q0.year + 1: flag_now, q0.year + 2: flag_now}, **pj.ADOPTED)
print("meeting %d (%s), Focus of %s, exchange rate %.2f, flag held at '%s'; relevant horizon %s" % (n_next, d_next.date(), F.date(), fx_start, flag_now, horizon))
tables = {}
for name, out in runs.items():
    tables[name] = pj.copom_table(out, years=[q0.year, q0.year + 1, q0.year + 2], horizon=horizon)
    print("==", name); print(tables[name].round(2).to_string())
    if out.get("adm_model") is not None:
        it = out["adm_model"]["quarterly"]
        it.to_csv(OUTPUT / ("10_administered_items_%s.csv" % name.split()[-1]))
        print(it.loc[q0:, ["5104001", "2202003", "6203", "6101", "2101004", "5101001", "band", "adm", "ipca"]].round(2).T.to_string())
comp = pd.concat({k: v for k, v in tables.items()}, axis=1)
comp.to_csv(OUTPUT / ("10_administered_projection_%s.csv" % d_next.strftime("%Y%m%d")))

# %% [markdown]
# ## Report

# %%
L = ["# Administered-price model: estimates versus the BCB's calibration, fit and projection", "",
     "Generated by `notebooks/10_administered.py` (data through %s)." % q.index[-1], "",
     "## Coefficients: BCB calibration (June-2025 box) versus OLS estimates on the same regressors", "",
     "| item | parameter | BCB | estimate | s.e. | (est−BCB)/s.e. | sample | n | R² |", "|---|---|---|---|---|---|---|---|---|"]
for _, r in tab.iterrows():
    L.append("| %s | %s | %.3f | %.3f | %.3f | %s | %s | %d | %.2f |" % (r["item"], r["parameter"], r["bcb"], r["estimate"], r["se"],
             ("%.1f" % r["t_diff"]) if pd.notna(r["t_diff"]) else "", r["sample"], r["n"], r["r2"]))
L += ["", "## Fit", "", "Dynamic simulation from %s with realized drivers (quarterly log-%%):" % origin, "", "```", dd.round(2).T.to_string(), "```", ""]
L += ["## Projection for the meeting of %s (reference scenario, flag '%s')" % (d_next.date(), flag_now), "", "```", comp.round(2).to_string(), "```", ""]
(ROOT / "docs" / "results_administered.md").write_text("\n".join(L), encoding="utf-8")
print("saved outputs to", OUTPUT)
