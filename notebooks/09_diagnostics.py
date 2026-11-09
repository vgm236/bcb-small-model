# %% [markdown]
# # 09 – Where do the differences from the Copom's projections come from?
#
# Targeted variants of the backtest, each switching one ingredient to what the Copom plausibly used:
# * **gap anchor**: the filtered output gap is anchored on the estimates published in the Monetary Policy Reports
#   available at the meeting (the gap the Copom conditions on; our real-time gap runs about 0.5 pp above it);
# * **BCB coefficients**: the posterior modes of the June-2024 box instead of our estimates;
# * **RONI**: NOAA's Relative ONI for the climate term (the BCB's index since 2026);
# * **perfect nowcast**: the published IPCA of the current quarter instead of the Focus medians (an upper bound on
#   what better short-term information could do);
# and their combinations. Output: output/09_diagnostics*.csv.

# %%
import sys, pathlib, warnings, time
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import projection as pj, copom_meetings as cm, model_system as ms
from config import OUTPUT

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
model = pj.load_model()
print(pj.rpm_gap_estimates().to_string())

VARIANTS = {
    "baseline": {},
    "gap anchor (RPM)": {"gap_anchor": "rpm"},
    "BCB coefficients": {"coefficients": "bcb"},
    "RONI": {"climate_index": "roni"},
    "perfect nowcast": {"nowcast": "realized"},
    "gap anchor + RONI": {"gap_anchor": "rpm", "climate_index": "roni"},
    "gap anchor + BCB coefficients": {"gap_anchor": "rpm", "coefficients": "bcb"},
    "gap anchor + RONI + BCB coefficients": {"gap_anchor": "rpm", "climate_index": "roni", "coefficients": "bcb"},
    "gap anchor + perfect nowcast": {"gap_anchor": "rpm", "nowcast": "realized"},
    "gap anchor + RONI + perfect nowcast": {"gap_anchor": "rpm", "climate_index": "roni", "nowcast": "realized"},
}


def targets(m, out, name):
    rows = []
    for y, (ipca, liv, adm) in m["years"].items():
        o = out["years"][y]
        rows.append({"variant": name, "meeting": m["n"], "target": str(y), "type": "current year" if y == int(m["decision"][:4]) else "next year",
                     "copom_ipca": ipca, "our_ipca": o["pi"], "copom_livres": liv, "our_livres": o["piL"]})
    hq, ipca, liv, adm = m["horizon"]
    t = out["table"]; hp = pd.Period(hq, freq="Q")
    rows.append({"variant": name, "meeting": m["n"], "target": hq, "type": "relevant horizon", "copom_ipca": ipca, "our_ipca": t.loc[hp, "pi_4q"],
                 "copom_livres": liv, "our_livres": t.loc[hp, "piL_4q"]})
    return rows

rows, origins = [], []
for name, kw in VARIANTS.items():
    t0 = time.time()
    for m in cm.MEETINGS:
        adm_years = {y: v[2] for y, v in m["years"].items()}
        out = pj.project(m["decision"], m["focus"], m["fx_start"], adm_years, (m["horizon"][0], m["horizon"][3]), H=12, model=model, **kw)
        rows += targets(m, out, name)
        origins.append({"variant": name, "meeting": m["n"], "q0": str(out["q0"]), "gap_q0": float(out["a0"][ms.iv["h"]]),
                        "rpm_gap_known": float(out["gap_obs"].iloc[-1]) if out["gap_obs"] is not None and len(out["gap_obs"]) else np.nan,
                        "pie_q0": float(out["a0"][ms.iv["pie"]]), "piL_q0_sa": float(out["a0"][ms.iv["piL"]])})
    print("variant '%s' done (%.0fs)" % (name, time.time() - t0))
df = pd.DataFrame(rows)
for c in ["ipca", "livres"]:
    df["diff_" + c] = df["our_" + c] - df["copom_" + c]
df.to_csv(OUTPUT / "09_diagnostics.csv", index=False)
og = pd.DataFrame(origins)
og.to_csv(OUTPUT / "09_diagnostics_origins.csv", index=False)
summ = df.groupby(["variant", "type"]).agg(mean_diff_ipca=("diff_ipca", "mean"), mae_ipca=("diff_ipca", lambda x: x.abs().mean()),
                                            mean_diff_livres=("diff_livres", "mean"), mae_livres=("diff_livres", lambda x: x.abs().mean())).round(2)
vall = df.groupby("variant").agg(mean_diff_ipca=("diff_ipca", "mean"), mae_ipca=("diff_ipca", lambda x: x.abs().mean()),
                                 mean_diff_livres=("diff_livres", "mean"), mae_livres=("diff_livres", lambda x: x.abs().mean()),
                                 max_abs_ipca=("diff_ipca", lambda x: x.abs().max())).round(2)
vall = vall.loc[list(VARIANTS)]
print(summ.to_string()); print(vall.to_string())
summ.to_csv(OUTPUT / "09_diagnostics_summary_by_type.csv"); vall.to_csv(OUTPUT / "09_diagnostics_summary.csv")
piv = df.pivot_table(index=["meeting", "target"], columns="variant", values="diff_ipca").round(2)[list(VARIANTS)]
print(piv.to_string())
piv.to_csv(OUTPUT / "09_diagnostics_by_meeting.csv")
print(og.pivot(index="meeting", columns="variant", values="gap_q0").round(2)[list(VARIANTS)].to_string())
print(og[og["variant"] == "gap anchor (RPM)"][["meeting", "q0", "gap_q0", "rpm_gap_known"]].round(2).to_string())
