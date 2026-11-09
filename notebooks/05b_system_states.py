# %% [markdown]
# # 05b – Smoothed states of the full system (output gap, neutral and equilibrium rates)
#
# Uses the posterior mode saved by `05_system.py` (no re-estimation) and the Durbin–Koopman smoother.

# %%
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model_system as ms
from config import DATA_PROC, OUTPUT, SAMPLE_START, SAMPLE_END

df = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
df.index = pd.PeriodIndex(df.index, freq="Q")
data = ms.prepare_data(df, "2002Q4", SAMPLE_END)
exo = ms.exog_processes(data, SAMPLE_START, SAMPLE_END)
calib = dict(ms.DEFAULT_CALIB)
calib["w"] = float(df.loc[SAMPLE_START:SAMPLE_END, "w_livres"].mean())
mode = pd.read_csv(OUTPUT / "05_system_mode.csv", index_col=0).iloc[:, 0].to_dict()

ss, eu = ms.state_space(mode, calib, exo, data)
f = ss.smooth(ms.observation_matrix(data))
sm = pd.DataFrame(f["asm"], index=data.index, columns=ms.VARS)
sd = pd.DataFrame(np.sqrt(np.maximum(np.einsum("tii->ti", f["Psm"]), 0)), index=data.index, columns=ms.VARS)
sm["rr_is"] = sm["rrIS"] + data["rr_trend"].values
sm["rr_taylor"] = sm["rrTAY"] + data["rr_trend"].values
out = sm[["h", "s", "rrIS", "rrTAY", "rr_is", "rr_taylor"]].copy()
out["h_sd"] = sd["h"]
out.loc[SAMPLE_START:].to_csv(OUTPUT / "05_system_smoothed.csv")

g3 = pd.read_csv(OUTPUT / "03_gap_smoothed.csv", index_col=0)["gap"]
g3.index = pd.PeriodIndex(g3.index, freq="Q")
x = data.loc[SAMPLE_START:].index.to_timestamp()
s = sm.loc[SAMPLE_START:]
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
axes[0].fill_between(x, s["h"] - 1.64 * sd.loc[SAMPLE_START:, "h"], s["h"] + 1.64 * sd.loc[SAMPLE_START:, "h"], alpha=0.2)
axes[0].plot(x, s["h"], lw=2, label="gap (full system)")
axes[0].plot(x, g3.loc[SAMPLE_START:], lw=1, label="gap (Layer A, activity indicators only)")
axes[0].plot(x, data.loc[SAMPLE_START:, "fpib"], lw=1, ls="--", label="GDP HP cycle")
axes[0].axhline(0, color="k", lw=0.5); axes[0].legend(fontsize=8); axes[0].set_title("Output gap (%)")
axes[1].plot(x, s["rr_is"], lw=2, label="neutral rate, IS block")
axes[1].plot(x, s["rr_taylor"], lw=2, label="equilibrium rate, Taylor block")
axes[1].plot(x, data.loc[SAMPLE_START:, "rr_trend"], lw=1, ls="--", label="HP trend of Focus real rate")
axes[1].plot(x, data.loc[SAMPLE_START:, "r_focus"], lw=0.8, label="Focus ex-ante real rate")
axes[1].legend(fontsize=8); axes[1].set_title("Neutral / equilibrium real rates (% p.a.)")
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "05_system_states.png", dpi=110)
print(out.loc[["2008Q4", "2009Q1", "2014Q1", "2016Q4", "2019Q4", "2020Q2", "2021Q4", "2023Q2", "2023Q4"]].round(2))
print("corr(system gap, Layer-A gap) = %.3f" % np.corrcoef(s["h"], g3.loc[SAMPLE_START:])[0, 1])
