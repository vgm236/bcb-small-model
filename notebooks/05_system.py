# %% [markdown]
# # 05 – Full-system Bayesian estimation (Layer B)
#
# The complete aggregate model (Phillips curve, IS curve with the unobserved gap and neutral rate, Taylor rule,
# UIP, expectations with model-consistent term, gap measurement block) is solved as a linear rational-
# expectations model (gensys) and estimated by Bayesian methods with the Kalman filter, as the BCB does.
# Priors: Table 1 of the June-2024 box (uniform supports; Beta for the fiscal coefficient). Calibrated items
# (not reported in the box): s.d. of the random-walk neutral-rate components, crisis-shock and high-variance
# multiples, the IPCA weight of free prices, and the AR(1) processes of the exogenous observed drivers.

# %%
import sys, pathlib, warnings, time, json
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model_system as ms
from bayes import Posterior
from config import DATA_PROC, OUTPUT, SAMPLE_START, SAMPLE_END

pd.set_option("display.width", 220)
df = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="Q")
PRESAMPLE = 4
data = ms.prepare_data(df, "2002Q4", SAMPLE_END)        # 4 presample quarters before 2003Q4
exo = ms.exog_processes(data, SAMPLE_START, SAMPLE_END)
calib = dict(ms.DEFAULT_CALIB)
calib["w"] = float(df.loc[SAMPLE_START:SAMPLE_END, "w_livres"].mean())
print("calibration:", calib)
print("exogenous processes:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in exo.items()}))

BCB = {"a1L": 0.24, "a1I": 0.38, "a2": 0.023, "a3": 0.011, "a4": 0.120, "a5": 0.0012, "a6": 0.0007,
       "b1": 0.85, "b2": 0.44, "b3": 0.030, "b4": 0.054, "b5": 0.84, "th1": 1.48, "th2": -0.58, "th3": 2.03,
       "phi1": 0.75, "phi2": 0.11, "phi3": 0.021, "delta": 1.90, "g_nuci": 1.87, "g_emp": 1.10, "g_caged": 0.69, "sig_h": 1.09}
BCB_BAND = {"a1L": (0.02, 0.38), "a1I": (0.13, 0.69), "a2": (0.006, 0.039), "a3": (0, 0.025), "a4": (0.072, 0.198),
            "a5": (0.0004, 0.0019), "a6": (0, 0.0021), "b1": (0.7, 0.95), "b2": (0.21, 0.66), "b3": (0.027, 0.032),
            "b4": (0, 0.23), "b5": (0.59, 0.99), "th1": (1.41, 1.54), "th2": (-0.63, -0.52), "th3": (1.47, 2.64),
            "phi1": (0.68, 0.82), "phi2": (0.06, 0.13), "phi3": (0, 0.049), "delta": (0.77, 3.22),
            "g_nuci": (1.65, 2.12), "g_emp": (0.96, 1.25), "g_caged": (0.61, 0.79), "sig_h": (0.98, 1.21)}

# %% [markdown]
# ## Posterior mode
# Starting values: Layer-A estimates where available, BCB modes otherwise.

# %%
start = dict(BCB)
start.update({"sd_piL": 0.5, "sd_pie": 0.3, "sd_i": 0.5, "sd_e": 5.0, "sd_h": 0.5, "sd_ie": 0.5})
try:
    la = pd.read_csv(OUTPUT / "03_gap_is_posterior.csv", index_col=0)["mode"]
    for k in ["b1", "b2", "b4", "b5", "g_nuci", "g_emp", "g_caged", "sig_h"]:
        start[k] = float(la[k])
    start["sd_h"] = float(la["sd_is"])
    pc = pd.read_csv(OUTPUT / "04_phillips_posterior.csv", index_col=0)["mode"]
    for k in ["a1L", "a1I", "a2", "a3", "a4", "a5", "a6"]:
        start[k] = float(pc[k])
    start["sd_piL"] = float(pc["sigma"])
    tr = pd.read_csv(OUTPUT / "04_taylor_posterior.csv", index_col=0)["mode"]
    for k in ["th1", "th2", "th3"]:
        start[k] = float(tr[k])
    start["sd_i"] = float(tr["sd_i"])
    up = pd.read_csv(OUTPUT / "04_uip_posterior.csv", index_col=0)["mode"]
    start["delta"] = float(up["delta"]); start["sd_e"] = float(up["sigma"])
except Exception as e:  # noqa: BLE001
    print("Layer-A results not found, starting from BCB modes:", e)
post = Posterior(ms.loglik_factory(data, calib, exo, presample=PRESAMPLE), ms.priors_system())
t0 = time.time()
print("log posterior at start: %.2f" % post.log_post(start))
mode = post.find_mode(start=start, n_restarts=1, verbose=True, maxiter=6000)
print("mode found in %.0fs (%d evaluations); log posterior %.2f" % (time.time() - t0, post.n_evals, post.mode_val))
mode_s = pd.Series(mode)
mode_s.to_csv(OUTPUT / "05_system_mode.csv")
print(mode_s.round(4))

# %% [markdown]
# ## MCMC (random-walk Metropolis-Hastings)

# %%
N_DRAWS, BURN = int(sys.argv[1]) if len(sys.argv) > 1 else 40000, 10000
t0 = time.time()
chain = post.rwmh(n_draws=N_DRAWS, burn=BURN, seed=1, verbose=True)
print("MCMC done in %.0fs" % (time.time() - t0))
summ = post.summary()
summ.to_csv(OUTPUT / "05_system_posterior.csv")
np.save(OUTPUT / "05_system_chain.npy", chain)
print(summ.round(4))

# %% [markdown]
# ## Comparison with the BCB June-2024 box

# %%
rows = []
for k, m in BCB.items():
    lo, hi = BCB_BAND[k]
    rows.append({"param": k, "BCB mode": m, "BCB 5%": lo, "BCB 95%": hi, "our mode": summ.loc[k, "mode"],
                 "our mean": summ.loc[k, "mean"], "our 5%": summ.loc[k, "q05"], "our 95%": summ.loc[k, "q95"]})
comp = pd.DataFrame(rows).set_index("param")
comp["mode inside BCB band"] = (comp["our mode"] >= comp["BCB 5%"]) & (comp["our mode"] <= comp["BCB 95%"])
comp["BCB mode inside our band"] = (comp["BCB mode"] >= comp["our 5%"]) & (comp["BCB mode"] <= comp["our 95%"])
comp.to_csv(OUTPUT / "05_system_comparison.csv")
print(comp.round(4))

# %% [markdown]
# ## Smoothed output gap and neutral rate, impulse responses

# %%
ss, eu = ms.state_space(mode, calib, exo, data)
f = ss.smooth(ms.observation_matrix(data))
sm = pd.DataFrame(f["asm"], index=data.index, columns=ms.VARS)
sm["rr_is"] = sm["rrIS"] + data["rr_trend"].values
sm["rr_taylor"] = sm["rrTAY"] + data["rr_trend"].values
sm[["h", "s", "rrIS", "rrTAY", "rr_is", "rr_taylor"]].to_csv(OUTPUT / "05_system_smoothed.csv")
x = data.index.to_timestamp()
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
axes[0].plot(x, sm["h"], lw=2, label="gap (system)")
axes[0].plot(x, data["fpib"], lw=1, ls="--", label="GDP HP cycle")
try:
    g3 = pd.read_csv(OUTPUT / "03_gap_smoothed.csv", index_col=0)["gap"]; g3.index = pd.PeriodIndex(g3.index, freq="Q")
    axes[0].plot(g3.index.to_timestamp(), g3, lw=1, label="gap (Layer A)")
except Exception:  # noqa: BLE001
    pass
axes[0].axhline(0, color="k", lw=0.5); axes[0].legend(); axes[0].set_title("Output gap, full-system estimate")
axes[1].plot(x, sm["rr_is"], lw=2, label="neutral rate (IS block)")
axes[1].plot(x, sm["rr_taylor"], lw=2, label="equilibrium rate (Taylor block)")
axes[1].plot(x, data["rr_trend"], lw=1, ls="--", label="HP trend of Focus real rate")
axes[1].legend(); axes[1].set_title("Neutral / equilibrium real rates (% p.a.)")
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "05_system_states.png", dpi=110)

# %% [markdown]
# ### Impulse responses, replicating the three experiments of the box
# * Selic +1 pp for four quarters, then the Taylor rule (box: IPCA accumulated in 4 quarters falls 0.27 pp at the 4th quarter);
# * permanent 10% depreciation (box: +0.96 pp on the IPCA including the administered-price model; +0.72 pp on free prices);
# * one-off +1 pp shock to the output gap (box: +0.49 pp).
#
# The small model has no administered-price equation (the BCB combines it with a separate model), so the IPCA responses
# below use the free-price response with weight w and, as an add-on, the administered-price pass-through published in the
# box (administered prices +1.65 pp over four quarters for a 10% depreciation, spread evenly) for the exchange-rate channel.

# %%
W = calib["w"]
ADM_PASS = 1.65 / 10.0   # pp of administered inflation (4q) per 1% depreciation, from the box


def summarise(irf, name):
    irf = irf.copy()
    irf["piL_4q"] = irf["piL"].rolling(4).sum()
    irf["pi_4q"] = irf["pi"].rolling(4).sum()
    # administered-price add-on: exchange-rate pass-through spread evenly over the 4 quarters following each change
    adm = ADM_PASS * sum(irf["de"].shift(k).fillna(0) / 4.0 for k in range(4))
    irf["pi_4q_adm"] = (W * irf["piL"] + (1 - W) * adm).rolling(4).sum()
    irf.to_csv(OUTPUT / ("05_irf_%s.csv" % name))
    return irf

irf_i = summarise(ms.conditional_path(mode, calib, exo, "i", [1.0] * 4, "e_i", horizon=16), "selic")
irf_e = summarise(ms.conditional_path(mode, calib, exo, "de", [10.0] + [0.0] * 16, "e_e", horizon=16), "fx")
irf_h = summarise(ms.conditional_path(mode, calib, exo, "h", [1.0], "e_crisis", horizon=16), "gap")
lines = [
    "Selic +1pp for 4 quarters: IPCA 4q (free-price channel only) min %.3f pp at q%d; with administered add-on %.3f pp at q%d; free prices 4q min %.3f; gap min %.2f  [BCB: IPCA -0.27 at q4]"
    % (irf_i["pi_4q"].min(), irf_i["pi_4q"].idxmin(), irf_i["pi_4q_adm"].min(), irf_i["pi_4q_adm"].idxmin(), irf_i["piL_4q"].min(), irf_i["h"].min()),
    "10%% depreciation: free prices 4q max %.3f pp at q%d; IPCA 4q (free only) %.3f; with administered add-on %.3f  [BCB: free 0.72, IPCA 0.96]"
    % (irf_e["piL_4q"].max(), irf_e["piL_4q"].idxmax(), irf_e["pi_4q"].max(), irf_e["pi_4q_adm"].max()),
    "+1pp gap shock: IPCA 4q max %.3f pp at q%d  [BCB: 0.49]" % (irf_h["pi_4q"].max(), irf_h["pi_4q"].idxmax()),
]
print("\n".join(lines))
(OUTPUT / "05_irf_summary.txt").write_text("\n".join(lines), encoding="utf-8")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(irf_i["i"], label="Selic"); axes[0].plot(irf_i["h"], label="gap"); axes[0].plot(irf_i["pi_4q_adm"], label="IPCA 4q (with adm. add-on)"); axes[0].plot(irf_i["piL_4q"], label="free prices 4q"); axes[0].axhline(0, color="k", lw=0.5); axes[0].legend(fontsize=8); axes[0].set_title("Selic +1pp for 4 quarters")
axes[1].plot(irf_e["de"].cumsum() / 10, label="exchange rate (cum. %, /10)"); axes[1].plot(irf_e["pi_4q_adm"], label="IPCA 4q (with adm. add-on)"); axes[1].plot(irf_e["piL_4q"], label="free prices 4q"); axes[1].axhline(0, color="k", lw=0.5); axes[1].legend(fontsize=8); axes[1].set_title("Permanent 10% depreciation")
axes[2].plot(irf_h["h"], label="gap"); axes[2].plot(irf_h["pi_4q"], label="IPCA 4q"); axes[2].axhline(0, color="k", lw=0.5); axes[2].legend(fontsize=8); axes[2].set_title("+1pp gap shock")
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "05_system_irfs.png", dpi=110)
pd.concat({"selic": irf_i[["i", "h", "piL", "pi", "pie", "de", "pi_4q", "pi_4q_adm"]], "fx": irf_e[["i", "h", "piL", "pi", "pie", "de", "pi_4q", "pi_4q_adm"]],
           "gap": irf_h[["i", "h", "piL", "pi", "pie", "de", "pi_4q", "pi_4q_adm"]]}, axis=1).to_csv(OUTPUT / "05_system_irfs.csv")
print("saved outputs to", OUTPUT)
