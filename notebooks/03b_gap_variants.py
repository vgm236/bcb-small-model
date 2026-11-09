# %% [markdown]
# # 03b – Output-gap block: robustness of the posterior mode to data and calibration choices
#
# Variants (posterior modes only):
# 1. baseline (spliced unemployment, sd_rr = 0.15, crisis multiple 3);
# 2. unemployment observed only from 2012Q1 (PNAD Contínua, no PME splice);
# 3. no unemployment indicator at all;
# 4. neutral-rate random walk with s.d. 0.30;
# 5. crisis-shock multiple 5;
# 6. NUCI / unemployment demeaned over the whole sample instead of the pre-Covid period.

# %%
import sys, pathlib, warnings, time
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import gap_filter as gf
from config import DATA_PROC, OUTPUT, SAMPLE_START, SAMPLE_END

pd.set_option("display.width", 200)
df = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="Q")
base = gf.prepare(df, SAMPLE_START, SAMPLE_END)
BCB = {"b1": 0.85, "b2": 0.44, "b3": 0.030, "b4": 0.054, "b5": 0.84, "g_nuci": 1.87, "g_emp": 1.10, "g_caged": 0.69, "sig_h": 1.09}


def run(label, data, **prior_kw):
    t0 = time.time()
    post = gf.estimate_gap(data, n_draws=0, verbose=False, priors=gf.priors_gap(**prior_kw))
    sm = gf.smoothed_gap(post, data)
    m = {k: post.mode[k] for k in BCB}
    m["logpost"] = post.mode_val
    m["gap_2023Q4"] = sm.loc["2023Q4", "gap"]
    m["gap_2020Q2"] = sm.loc["2020Q2", "gap"]
    m["rr_is_2023Q4"] = sm.loc["2023Q4", "rr_is"]
    print("%-40s done in %.0fs" % (label, time.time() - t0))
    return m

from config import CRISIS_SD_MULT, SD_RR_RW
res = {}
res["1 baseline (crisis x%g, sd_rr %.2f)" % (CRISIS_SD_MULT, SD_RR_RW)] = run("baseline", base)
d2 = base.copy(); d2.loc[:"2011Q4", "femp"] = np.nan
res["2 PNADC only (u from 2012Q1)"] = run("PNADC only", d2)
d3 = base.copy(); d3["femp"] = np.nan
res["3 no unemployment"] = run("no unemployment", d3)
res["4 sd_rr = 0.30"] = run("sd_rr 0.30", base, sd_rr=0.30)
res["5 crisis multiple 3"] = run("crisis x3", base, sd_crisis_mult=3.0)
res["6 crisis multiple 5"] = run("crisis x5", base, sd_crisis_mult=5.0)
d6 = base.copy()
d6["fnuci"] = df["nuci"].loc[SAMPLE_START:SAMPLE_END] - df["nuci"].loc[SAMPLE_START:SAMPLE_END].mean()
d6["femp"] = -(df["u_sa"].loc[SAMPLE_START:SAMPLE_END] - df["u_sa"].loc[SAMPLE_START:SAMPLE_END].mean())
res["7 full-sample means"] = run("full-sample means", d6)
tab = pd.DataFrame(res).T
tab.loc["BCB June 2024 (mode)"] = [BCB[k] for k in BCB] + [np.nan] * 4
print(tab.round(3).to_string())
tab.to_csv(OUTPUT / "03b_gap_variants.csv")
