# %% [markdown]
# # 04 – Nominal block, equation by equation (Layer A)
#
# Using the smoothed output gap of `03_gap_is.py`, each equation of the June-2024 box is estimated on its own:
# Phillips curve (restricted least squares + Bayesian with the box's uniform priors and the high-variance
# treatment of 2020Q2–2022Q4), Taylor rule (Kalman-filter likelihood with a random-walk equilibrium-rate
# component, as in the box), UIP (least squares / Bayesian) and the expectations equation (2SLS, the
# model-consistent expectation being instrumented by date-t information). Results are compared with Table 1
# of the box.

# %%
import sys, pathlib, warnings, time
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import single_eq as se
from config import DATA_PROC, OUTPUT, SAMPLE_START, SAMPLE_END

pd.set_option("display.width", 220)
df = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="Q")
gap = pd.read_csv(OUTPUT / "03_gap_smoothed.csv", index_col=0)["gap"]
gap.index = pd.PeriodIndex(gap.index, freq="Q")
gap = gap.reindex(df.index)
N_DRAWS, BURN = 20000, 5000

BCB = {  # posterior modes and 90% bands, Table 1 (aggregate model), RI June 2024
    "a1L": (0.24, 0.02, 0.38), "a1I": (0.38, 0.13, 0.69), "a2": (0.023, 0.006, 0.039), "a3": (0.011, 0.0, 0.025),
    "a4": (0.120, 0.072, 0.198), "a5": (0.0012, 0.0004, 0.0019), "a6": (0.0007, 0.0, 0.0021),
    "th1": (1.48, 1.41, 1.54), "th2": (-0.58, -0.63, -0.52), "th3": (2.03, 1.47, 2.64),
    "phi1": (0.75, 0.68, 0.82), "phi2": (0.11, 0.06, 0.13), "phi3": (0.021, 0.0, 0.049),
    "delta": (1.90, 0.77, 3.22)}


def compare(summ, keys, label):
    rows = []
    for k in keys:
        m, lo, hi = BCB[k]
        rows.append({"param": k, "BCB mode": m, "BCB 5%": lo, "BCB 95%": hi, "our mode": summ.loc[k, "mode"],
                     "our mean": summ.loc[k, "mean"], "our 5%": summ.loc[k, "q05"], "our 95%": summ.loc[k, "q95"]})
    c = pd.DataFrame(rows).set_index("param")
    c["inside BCB band"] = (c["our mode"] >= c["BCB 5%"]) & (c["our mode"] <= c["BCB 95%"])
    print("\n=== %s ===" % label); print(c.round(4))
    return c

# %% [markdown]
# ## Phillips curve

# %%
t0 = time.time()
m_pc, wls_pc, idx_pc = se.estimate_phillips(df, gap, SAMPLE_START, SAMPLE_END, n_draws=N_DRAWS, burn=BURN)
print("Phillips sample:", idx_pc[0], "-", idx_pc[-1], "n =", len(idx_pc), " (%.0fs)" % (time.time() - t0))
print("Restricted WLS (HAC s.e.):"); print(wls_pc.round(4))
summ_pc = m_pc.post.summary()
print(summ_pc.round(4))
comp_pc = compare(summ_pc, ["a1L", "a1I", "a2", "a3", "a4", "a5", "a6"], "Phillips curve vs BCB")
print("implied weight on expectations 1-a1L-a1I: %.3f (BCB: %.3f)" % (1 - summ_pc.loc["a1L", "mode"] - summ_pc.loc["a1I", "mode"], 1 - 0.24 - 0.38))

# %% [markdown]
# ### Phillips curve: sensitivity (unweighted; raw expectations; exchange-rate lag 2; 2003Q4–2019Q4 sample as in the Dec-2021 box)

# %%
sens = {}
for label, kw, s0, s1 in [("no HV weighting", dict(hv_mult=1.0), SAMPLE_START, SAMPLE_END),
                          ("HV variance x3 (s.d. x1.73)", dict(hv_mult=3 ** 0.5), SAMPLE_START, SAMPLE_END),
                          ("HV s.d. x5", dict(hv_mult=5.0), SAMPLE_START, SAMPLE_END),
                          ("seasonal adjustment by quarter dummies", dict(sa_dummies=True), SAMPLE_START, SAMPLE_END),
                          ("raw (non-smoothed) Focus", dict(exp_col="pi_e_q_raw"), SAMPLE_START, SAMPLE_END),
                          ("exchange-rate lag 2 (2021 spec)", dict(de_lag=2), SAMPLE_START, SAMPLE_END),
                          ("sample 2003Q4-2019Q4 (2021 box)", dict(), SAMPLE_START, "2019Q4")]:
    if "exp_col" in kw and kw["exp_col"] not in df:
        df["pi_e_q_raw"] = 100 * np.log1p(df["pi_e_12m_raw"] / 100) / 4
    dfx = df
    if kw.pop("sa_dummies", False):
        from tsutil import seasonal_adjust
        dfx = df.copy()
        dfx["pi_livres_sa"] = seasonal_adjust(df["pi_livres"].dropna(), method="dummies")
        dfx["pi_ipca_sa"] = seasonal_adjust(df["pi_ipca"].dropna(), method="dummies")
        dfx["pi_ipca_sa_ma4_l1"] = dfx["pi_ipca_sa"].shift(1).rolling(4).mean()
    y, X, w, idx = se.phillips_design(dfx, gap, s0, s1, **kw)
    m = se.BayesLinear(y, X, list(se.PHILLIPS_BOUNDS), [se.PHILLIPS_BOUNDS[k] for k in se.PHILLIPS_BOUNDS], weights=w)
    m.fit(n_draws=0)
    sens[label] = {k: m.post.mode[k] for k in se.PHILLIPS_BOUNDS}
sens = pd.DataFrame(sens).T
sens.loc["BCB 2024 (2003Q4-2023Q4)"] = [BCB[k][0] for k in se.PHILLIPS_BOUNDS]
sens.loc["BCB 2021 (2003Q4-2019Q4)"] = [0.23756, 0.25568, 0.01826, 0.01727, 0.13866, 0.00119, 0.00104]
print(sens.round(4))
sens.to_csv(OUTPUT / "04_phillips_sensitivity.csv")

# %% [markdown]
# ## Taylor rule

# %%
t0 = time.time()
post_tr, d_tr = se.estimate_taylor(df, SAMPLE_START, SAMPLE_END, n_draws=N_DRAWS, burn=BURN, sd_rr=0.15, rw=True)
summ_tr = post_tr.summary(); print(summ_tr.round(4), "(%.0fs)" % (time.time() - t0))
comp_tr = compare(summ_tr, ["th1", "th2", "th3"], "Taylor rule vs BCB (with RW equilibrium-rate component)")
post_tr0, _ = se.estimate_taylor(df, SAMPLE_START, SAMPLE_END, n_draws=0, rw=False)
print("Taylor rule without the RW component (mode):", {k: round(v, 3) for k, v in post_tr0.mode.items()})

# %% [markdown]
# ## UIP

# %%
m_uip, wls_uip, idx_uip = se.estimate_uip(df, SAMPLE_START, SAMPLE_END, n_draws=N_DRAWS, burn=BURN)
print("UIP OLS (HAC):"); print(wls_uip.round(4))
summ_uip = m_uip.post.summary(); print(summ_uip.round(4))
comp_uip = compare(summ_uip, ["delta"], "UIP vs BCB")
m_uip_raw, wls_uip_raw, _ = se.estimate_uip(df, SAMPLE_START, SAMPLE_END, n_draws=0, idif_col="i_dif_raw")
print("UIP with raw (non-log) rates, mode delta = %.3f" % m_uip_raw.post.mode["delta"])

# %% [markdown]
# ## Expectations equation (2SLS)

# %%
exp_tab, idx_e = se.estimate_expectations_2sls(df, gap, SAMPLE_START, SAMPLE_END)
exp_tab["BCB mode"] = [BCB["phi1"][0], BCB["phi2"][0], BCB["phi3"][0]]
print("\n=== Expectations equation (sample %s-%s) ===" % (idx_e[0], idx_e[-1])); print(exp_tab.round(4))

# %%
out = pd.concat([comp_pc, comp_tr, comp_uip])
out.to_csv(OUTPUT / "04_single_eq_comparison.csv")
exp_tab.to_csv(OUTPUT / "04_expectations_2sls.csv")
for name, s in [("phillips", summ_pc), ("taylor", summ_tr), ("uip", summ_uip)]:
    s.to_csv(OUTPUT / ("04_%s_posterior.csv" % name))
print(out.round(4))
