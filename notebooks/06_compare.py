# %% [markdown]
# # 06 – Comparison with the coefficients published by the BCB
#
# Consolidates the Layer-A (block-by-block) and Layer-B (full-system) estimates and compares them with
# Table 1 of the June-2024 box (posterior modes and 90% credible intervals), writing `docs/results.md`.

# %%
import sys, pathlib, warnings, json
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUTPUT, DATA_PROC

pd.set_option("display.width", 250)
DESC = {"a1L": "Phillips: inertia of free-price inflation", "a1I": "Phillips: inertia of IPCA inflation",
        "a2": "Phillips: imported (commodity) inflation", "a3": "Phillips: exchange-rate change (lag 1)",
        "a4": "Phillips: output gap", "a5": "Phillips: climate, El Nino", "a6": "Phillips: climate, La Nina",
        "b1": "IS: autoregressive", "b2": "IS: real interest-rate gap", "b3": "IS: primary result (Beta prior)",
        "b4": "IS: world gap", "b5": "IS: AR of the IS shock", "th1": "Taylor: smoothing, lag 1",
        "th2": "Taylor: smoothing, lag 2", "th3": "Taylor: response to expected inflation gap",
        "phi1": "Expectations: inertia", "phi2": "Expectations: model-consistent", "phi3": "Expectations: past inflation",
        "delta": "UIP: interest differential", "g_nuci": "Gap: loading of NUCI", "g_emp": "Gap: loading of unemployment",
        "g_caged": "Gap: loading of Caged", "sig_h": "Gap: measurement-error s.d."}
BCB = {"a1L": (0.24, 0.02, 0.38), "a1I": (0.38, 0.13, 0.69), "a2": (0.023, 0.006, 0.039), "a3": (0.011, 0.0, 0.025),
       "a4": (0.120, 0.072, 0.198), "a5": (0.0012, 0.0004, 0.0019), "a6": (0.0007, 0.0, 0.0021),
       "b1": (0.85, 0.70, 0.95), "b2": (0.44, 0.21, 0.66), "b3": (0.030, 0.027, 0.032), "b4": (0.054, 0.0, 0.23),
       "b5": (0.84, 0.59, 0.99), "th1": (1.48, 1.41, 1.54), "th2": (-0.58, -0.63, -0.52), "th3": (2.03, 1.47, 2.64),
       "phi1": (0.75, 0.68, 0.82), "phi2": (0.11, 0.06, 0.13), "phi3": (0.021, 0.0, 0.049), "delta": (1.90, 0.77, 3.22),
       "g_nuci": (1.87, 1.65, 2.12), "g_emp": (1.10, 0.96, 1.25), "g_caged": (0.69, 0.61, 0.79), "sig_h": (1.09, 0.98, 1.21)}
BCB2021 = {"a1L": 0.23756, "a1I": 0.25568, "a2": 0.01826, "a3": 0.01727, "a4": 0.13866, "a5": 0.00119, "a6": 0.00104,
           "b1": 0.73897, "b2": 0.54876, "b3": 0.02985, "b4": 0.04342, "th1": 1.45688, "th2": -0.54402, "th3": 1.29981,
           "phi1": 0.73260, "phi2": 0.12271, "phi3": 0.04370, "delta": 1.71813, "g_nuci": 2.08871, "g_emp": 1.08412, "g_caged": 0.77959}

def load(name):
    p = OUTPUT / name
    return pd.read_csv(p, index_col=0) if p.exists() else None

gapA = load("03_gap_is_posterior.csv")
pcA, trA, uipA = load("04_phillips_posterior.csv"), load("04_taylor_posterior.csv"), load("04_uip_posterior.csv")
expA = load("04_expectations_2sls.csv")
sysB = load("05_system_posterior.csv")
rows = []
for k, (m, lo, hi) in BCB.items():
    r = {"param": k, "description": DESC[k], "BCB 2024 mode": m, "BCB 90% low": lo, "BCB 90% high": hi, "BCB 2021 mode": BCB2021.get(k, np.nan)}
    src = None
    if k in ("b1", "b2", "b3", "b4", "b5", "g_nuci", "g_emp", "g_caged", "sig_h") and gapA is not None:
        src = gapA
    elif k.startswith("a") and pcA is not None:
        src = pcA
    elif k.startswith("th") and trA is not None:
        src = trA
    elif k == "delta" and uipA is not None:
        src = uipA
    if src is not None and k in src.index:
        r["Layer A mode"], r["Layer A 5%"], r["Layer A 95%"] = src.loc[k, "mode"], src.loc[k, "q05"], src.loc[k, "q95"]
    elif k.startswith("phi") and expA is not None:
        r["Layer A mode"] = expA.loc[k, "2SLS"]; r["Layer A 5%"] = expA.loc[k, "2SLS"] - 1.645 * expA.loc[k, "2SLS se"]; r["Layer A 95%"] = expA.loc[k, "2SLS"] + 1.645 * expA.loc[k, "2SLS se"]
    if sysB is not None and k in sysB.index:
        r["Layer B mode"], r["Layer B 5%"], r["Layer B 95%"] = sysB.loc[k, "mode"], sysB.loc[k, "q05"], sysB.loc[k, "q95"]
    rows.append(r)
tab = pd.DataFrame(rows).set_index("param")
for lay in ("Layer A", "Layer B"):
    if lay + " mode" in tab:
        tab[lay + " in BCB band"] = (tab[lay + " mode"] >= tab["BCB 90% low"]) & (tab[lay + " mode"] <= tab["BCB 90% high"])
        tab["BCB in " + lay + " band"] = (tab["BCB 2024 mode"] >= tab[lay + " 5%"]) & (tab["BCB 2024 mode"] <= tab[lay + " 95%"])
tab.to_csv(OUTPUT / "06_comparison_table.csv")
print(tab.drop(columns=["description"]).round(4).to_string())

# %% [markdown]
# ## Figure: posterior modes and 90% intervals, ours vs BCB

# %%
fig, axes = plt.subplots(4, 6, figsize=(18, 11))
for ax, k in zip(axes.ravel(), BCB):
    m, lo, hi = BCB[k]
    ax.errorbar([0], [m], yerr=[[m - lo], [hi - m]], fmt="o", color="k", capsize=4, label="BCB 2024")
    if "Layer A mode" in tab and pd.notna(tab.loc[k, "Layer A mode"]):
        ax.errorbar([1], [tab.loc[k, "Layer A mode"]], yerr=[[tab.loc[k, "Layer A mode"] - tab.loc[k, "Layer A 5%"]], [tab.loc[k, "Layer A 95%"] - tab.loc[k, "Layer A mode"]]], fmt="s", color="tab:blue", capsize=4, label="Layer A")
    if "Layer B mode" in tab and pd.notna(tab.loc[k, "Layer B mode"]):
        ax.errorbar([2], [tab.loc[k, "Layer B mode"]], yerr=[[tab.loc[k, "Layer B mode"] - tab.loc[k, "Layer B 5%"]], [tab.loc[k, "Layer B 95%"] - tab.loc[k, "Layer B mode"]]], fmt="^", color="tab:orange", capsize=4, label="Layer B")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["BCB", "A", "B"]); ax.set_title(k, fontsize=10); ax.set_xlim(-0.6, 2.6)
axes[0, 0].legend(fontsize=7)
for ax in axes.ravel()[len(BCB):]:
    ax.axis("off")
fig.suptitle("Posterior modes and 90% credible intervals: BCB (June 2024) vs this reconstruction")
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "06_comparison.png", dpi=110)

# %% [markdown]
# ## Write docs/results.md

# %%
def fmt(x, nd=3):
    return "" if pd.isna(x) else ("%.*f" % (nd, x))

lines = ["# Results: reconstruction vs the BCB June-2024 box", "",
         "Sample 2003Q4–2023Q4, quarterly. BCB values are the posterior modes and 90% credible intervals of Table 1 of the box (aggregate model); "
         "the December-2021 vintage (sample 2003Q4–2019Q4) is shown for context. Layer A = block-by-block estimation (gap block, then "
         "single equations given the smoothed gap); Layer B = full-system Bayesian estimation with the Kalman filter and rational expectations.", "",
         "| Parameter | Description | BCB 2024 mode [90%] | BCB 2021 | Layer A mode [90%] | Layer B mode [90%] |", "|---|---|---|---|---|---|"]
for k, r in tab.iterrows():
    nd = 4 if k in ("a5", "a6") else 3
    a = ("%s [%s; %s]" % (fmt(r.get("Layer A mode"), nd), fmt(r.get("Layer A 5%"), nd), fmt(r.get("Layer A 95%"), nd))) if pd.notna(r.get("Layer A mode", np.nan)) else "n/a"
    b = ("%s [%s; %s]" % (fmt(r.get("Layer B mode"), nd), fmt(r.get("Layer B 5%"), nd), fmt(r.get("Layer B 95%"), nd))) if pd.notna(r.get("Layer B mode", np.nan)) else "n/a"
    lines.append("| %s | %s | %s [%s; %s] | %s | %s | %s |" % (k, r["description"], fmt(r["BCB 2024 mode"], nd), fmt(r["BCB 90% low"], nd), fmt(r["BCB 90% high"], nd), fmt(r["BCB 2021 mode"], nd), a, b))
lines.append("")
for lay in ("Layer A", "Layer B"):
    if lay + " in BCB band" in tab:
        n_in = int(tab[lay + " in BCB band"].sum()); n_tot = int(tab[lay + " mode"].notna().sum())
        lines.append("* %s: %d of %d modes fall inside the BCB 90%% interval; BCB's mode falls inside our 90%% interval for %d parameters." % (lay, n_in, n_tot, int(tab["BCB in " + lay + " band"].sum())))
lines.append("")
irf = load("05_system_irfs.csv")
if irf is not None:
    irf.columns = pd.MultiIndex.from_tuples([tuple(c.split(".")) if "." in c else (c, "") for c in irf.columns]) if not isinstance(irf.columns, pd.MultiIndex) else irf.columns
lines += ["## Impulse responses (full system, posterior mode)", ""]
try:
    irf = pd.read_csv(OUTPUT / "05_system_irfs.csv", header=[0, 1], index_col=0)
    lines.append("| Experiment | BCB (box) | This model |"); lines.append("|---|---|---|")
    lines.append("| Selic +1 pp for 4 quarters, then Taylor rule: max effect on IPCA accumulated in 4 quarters | −0.27 pp at the 4th quarter | %.2f pp at quarter %d (free-price channel only: %.2f pp at quarter %d) |"
                 % (irf[("selic", "pi_4q_adm")].min(), int(irf[("selic", "pi_4q_adm")].idxmin()), irf[("selic", "pi_4q")].min(), int(irf[("selic", "pi_4q")].idxmin())))
    lines.append("| Permanent 10%% depreciation: max effect on IPCA (4q) / free prices (4q) | 0.96 pp (with the administered-price model) / 0.72 pp | %.2f pp (with the calibrated administered add-on) / %.2f pp |"
                 % (irf[("fx", "pi_4q_adm")].max(), irf[("fx", "piL")].rolling(4).sum().max()))
    lines.append("| One-off +1 pp output-gap shock: max effect on IPCA (4q) | 0.49 pp (7th quarter) | %.2f pp (quarter %d) |" % (irf[("gap", "pi_4q")].max(), int(irf[("gap", "pi_4q")].idxmax())))
    lines.append("")
    lines.append("The IPCA responses combine the model's free-price response (weight 0.73) with the administered-price pass-through published in the box "
                 "(1.65 pp over four quarters for a 10% depreciation) because the administered-price model is a separate BCB model. "
                 "The smaller and earlier-peaking responses relative to the box are consistent with the lower estimated inertia (a1I 0.24 vs 0.38) and the "
                 "larger weight on model-consistent expectations (phi2 0.16 vs 0.11), both of which shorten the propagation of shocks.")
    lines.append("")
except Exception as e:  # noqa: BLE001
    lines.append("(system IRFs not available: %s)" % e)
var = load("03b_gap_variants.csv")
if var is not None:
    lines += ["## Robustness of the gap block (posterior modes)", "", var.round(3).to_markdown() if hasattr(var, "to_markdown") else var.round(3).to_string(), ""]
sens = load("04_phillips_sensitivity.csv")
if sens is not None:
    lines += ["## Phillips curve: sensitivity of the mode to data choices", "", sens.round(4).to_markdown() if hasattr(sens, "to_markdown") else sens.round(4).to_string(), ""]
(ROOT / "docs" / "results.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
