# %% [markdown]
# # 03 – Output gap and IS curve: multivariate unobserved-components model (Layer A)
#
# The BCB treats the output gap as an unobserved state identified by four activity indicators (GDP, NUCI,
# unemployment, Caged) and by the IS curve. Here the block is estimated on its own (without the nominal
# equations) with Bayesian methods and the priors of the June-2024 box:
#
# $$h_t=\beta_1 h_{t-1}-\beta_2\frac{\hat r_{t-1}}{4}-\beta_3\widetilde{rp}_t+\beta_4 h^*_t+s^h_t+\varepsilon^{h,crise}_t,\quad
# s^h_t=\beta_5 s^h_{t-1}+\varepsilon^h_t,\quad \hat r_t=i^e_t-\pi^e_t-\overline{rr}^{trend}_t-\widetilde{rr}_t$$
#
# $$fpib_t=h_t+\sigma^h\epsilon,\; fnuci_t/\gamma_{nuci}=h_t+\sigma^h\epsilon,\; femp_t/\gamma_{emp}=h_{t-1}+\sigma^h\epsilon,\; fcaged_t/\gamma_{caged}=h_{t-1}+\sigma^h\epsilon$$
#
# Calibrated: s.d. of the random-walk component of the neutral rate (0.15 pp/quarter) and the crisis-shock
# s.d. multiple (3× the IS shock s.d.) in 2008Q4–2009Q4 and 2020Q1–2020Q4.

# %%
import sys, pathlib, warnings, time
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gap_filter as gf
from config import DATA_PROC, OUTPUT, SAMPLE_START, SAMPLE_END

pd.set_option("display.width", 200)
df = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="Q")
data = gf.prepare(df, SAMPLE_START, SAMPLE_END)
print(data[gf.OBS + ["rhat_obs_l1", "rp_tilde", "h_star"]].describe().round(2))

# %% [markdown]
# ## Posterior mode and MCMC

# %%
N_DRAWS, BURN = 30000, 10000
t0 = time.time()
post = gf.estimate_gap(data, n_draws=N_DRAWS, burn=BURN, seed=0, verbose=True)
print("elapsed %.0fs, likelihood evaluations: %d" % (time.time() - t0, post.n_evals))
summ = post.summary()
print(summ.round(4))

# %% [markdown]
# ## Comparison with the BCB June-2024 box (Table 1, aggregate model)

# %%
BCB = {"b1": (0.85, 0.70, 0.95), "b2": (0.44, 0.21, 0.66), "b3": (0.030, 0.027, 0.032), "b4": (0.054, 0.0, 0.23),
       "b5": (0.84, 0.59, 0.99), "g_nuci": (1.87, 1.65, 2.12), "g_emp": (1.10, 0.96, 1.25), "g_caged": (0.69, 0.61, 0.79),
       "sig_h": (1.09, 0.98, 1.21)}
rows = []
for k, (m, lo, hi) in BCB.items():
    rows.append({"param": k, "BCB mode": m, "BCB 5%": lo, "BCB 95%": hi, "our mode": summ.loc[k, "mode"],
                 "our 5%": summ.loc[k, "q05"], "our 95%": summ.loc[k, "q95"]})
comp = pd.DataFrame(rows).set_index("param")
comp["inside BCB 90% band"] = (comp["our mode"] >= comp["BCB 5%"]) & (comp["our mode"] <= comp["BCB 95%"])
print(comp.round(3))
comp.to_csv(OUTPUT / "03_gap_is_comparison.csv")
summ.to_csv(OUTPUT / "03_gap_is_posterior.csv")
np.save(OUTPUT / "03_gap_is_chain.npy", post.chain)

# %% [markdown]
# ## Smoothed output gap

# %%
sm = gf.smoothed_gap(post, data)
sm.to_csv(OUTPUT / "03_gap_smoothed.csv")
x = data.index.to_timestamp()
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
ax = axes[0]
ax.fill_between(x, sm["gap"] - 1.64 * sm["gap_sd"], sm["gap"] + 1.64 * sm["gap_sd"], alpha=0.2, label="90% band")
ax.plot(x, sm["gap"], lw=2, label="smoothed gap h_t")
ax.plot(x, data["fpib"], lw=1, ls="--", label="GDP HP cycle")
ax.plot(x, data["fnuci"] / post.mode["g_nuci"], lw=0.8, label="NUCI / gamma")
ax.plot(x, (data["femp"] / post.mode["g_emp"]).shift(-1), lw=0.8, label="-u / gamma (led 1q)")
ax.plot(x, (data["fcaged"] / post.mode["g_caged"]).shift(-1), lw=0.8, label="Caged / gamma (led 1q)")
ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_title("Output gap (Layer A, IS + 4 indicators)")
ax = axes[1]
ax.plot(x, sm["rr_is"], lw=2, label="neutral real rate rr_IS = trend + RW component")
ax.plot(x, data["rr_trend"], lw=1, ls="--", label="HP trend of Focus real rate")
ax.plot(x, data["r_focus"], lw=0.8, label="Focus ex-ante real rate")
ax.legend(fontsize=8); ax.set_title("Neutral real interest rate implied by the IS block (% p.a.)")
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "03_gap_is.png", dpi=110)
print(sm.loc["2019Q1":"2023Q4"].round(2))
print("gap 2023Q4: %.2f  (BCB RI Dec-2023/Mar-2024 projections: around -0.5 to 0)" % sm.loc["2023Q4", "gap"])
