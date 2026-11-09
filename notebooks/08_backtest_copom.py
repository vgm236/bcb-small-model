# %% [markdown]
# # 08 – Backtest: conditional projections at the last Copom meetings versus the Copom's own projections
#
# For each meeting from July 2025 (272nd) onwards (new meetings are added automatically from the minutes by
# `copom_meetings.update_history()`), the model is run with the information available at the meeting and the Copom's
# reference-scenario assumptions (see docs/projection.md), with the configuration adopted after the diagnostics of
# notebook 09 (`projection.ADOPTED`). Two outputs are produced for every meeting:
# * the **unadjusted** model projection;
# * the **adjusted** projection: unadjusted minus the model's error (model minus Copom) at the previous meeting,
#   by type of target (current year, next year, relevant horizon).
# Results are compared with Table 1 of the minutes; the variants of the first version of this notebook (climate
# off/realized, HP-trend neutral rate, seasonal administered profile) are kept at the end.

# %%
import sys, pathlib, warnings, time
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import projection as pj, copom_meetings as cm, backtest as bt
from config import OUTPUT

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
RUN_VARIANTS = (sys.argv[2] != "novariants") if len(sys.argv) > 2 else True
try:
    assert cm.check_against_minutes(verbose=False), "transcription of the minutes does not match"
    assert cm.check_parser(verbose=False), "the minutes parser does not reproduce the transcribed table"
except FileNotFoundError:
    print("minutes texts not found (run src/download_refs.py to fetch them); consistency checks skipped")
meetings = cm.update_history(verbose=True)          # adds meetings whose minutes were published since the last run
print("meetings:", [m["n"] for m in meetings], "| settings:", pj.ADOPTED)

# %% [markdown]
# ## Baseline run (adopted configuration): unadjusted and adjusted outputs

# %%
res = bt.run_backtest(meetings, n_draws=N_DRAWS, progress=lambda k, n, m: print("meeting %d done (%d/%d)" % (m, k, n)))
tab = res["table"]
print(tab[["meeting", "target", "type", "copom_ipca", "model_ipca", "adjusted_ipca", "diff_ipca", "diff_adjusted_ipca",
           "copom_livres", "model_livres", "adjusted_livres", "real_ipca"]].round(2).to_string())
print("\nSummary (model minus Copom), all targets:"); print(res["summary"].round(2).to_string())
print("\nSummary restricted to meetings with a previous meeting (where the adjustment applies):"); print(res["summary_adjusted"].round(2).to_string())
print("\nMAE by damping factor of the previous-meeting error (0 = unadjusted, 1 = full adjustment):")
print(res["damping"].to_string()); print(res["damping_by_type"].to_string())
print(res["assumptions"].round(2).to_string())
bt.plot_backtest(res, meetings, OUTPUT / "figs" / "08_backtest_panels.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (c, lab) in zip(axes, [("ipca", "IPCA"), ("livres", "IPCA free prices"), ("adm", "IPCA administered prices")]):
    for typ, mk in [("current year", "o"), ("next year", "s"), ("relevant horizon", "D")]:
        s = tab[tab["type"] == typ]
        ax.scatter(s["copom_" + c], s["model_" + c], marker=mk, label=typ + " (unadjusted)", s=40)
        if c != "adm":
            ax.scatter(s["copom_" + c], s["adjusted_" + c], marker=mk, facecolors="none", edgecolors="grey", s=40, label=typ + " (adjusted)")
    lo, hi = min(tab["copom_" + c].min(), tab["model_" + c].min()) - 0.3, max(tab["copom_" + c].max(), tab["model_" + c].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
    ax.set_xlabel("Copom projection (%)"); ax.set_ylabel("model projection (%)"); ax.set_title(lab)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "08_backtest_scatter.png", dpi=110)

# %% [markdown]
# ## Variants (unadjusted output; diagnostics of the ingredients)

# %%
if RUN_VARIANTS:
    variants = {"baseline": {}, "climate off": {"climate": "off"}, "climate realized (perfect foresight)": {"climate": "realized"},
                "neutral rate = HP trend": {"neutral_rate": None}, "administered: seasonal profile": {"adm_profile": "seasonal"}}
    vrows = []
    model = pj.load_model()
    for name, kw in variants.items():
        t0 = time.time()
        for m in meetings:
            adm_years = {y: v[2] for y, v in m["years"].items()}
            adm_h = (m["horizon"][0], m["horizon"][3]) if m.get("horizon") else None
            out = res["outputs"][m["n"]] if name == "baseline" else pj.project(m["decision"], m["focus"], m["fx_start"], adm_years, adm_h, H=12, model=model, **dict(pj.ADOPTED, **kw))
            import adjust
            for r in adjust.target_rows(m, out):
                r["variant"] = name; r["meeting"] = m["n"]; vrows.append(r)
        print("variant '%s' done (%.0fs)" % (name, time.time() - t0))
    vt = pd.DataFrame(vrows)
    for c in ["ipca", "livres"]:
        vt["diff_" + c] = vt["model_" + c] - vt["copom_" + c]
    vsum = vt.groupby(["variant", "type"]).agg(mean_diff_ipca=("diff_ipca", "mean"), mae_ipca=("diff_ipca", lambda x: x.abs().mean()),
                                                mean_diff_livres=("diff_livres", "mean"), mae_livres=("diff_livres", lambda x: x.abs().mean())).round(2)
    print(vsum.to_string())
    vt.to_csv(OUTPUT / "08_backtest_variants.csv", index=False); vsum.to_csv(OUTPUT / "08_backtest_variants_summary.csv")
    piv = vt[vt["type"] == "relevant horizon"].pivot(index="meeting", columns="variant", values="model_livres")
    piv["Copom"] = vt[vt["type"] == "relevant horizon"].groupby("meeting")["copom_livres"].first()
    piv.to_csv(OUTPUT / "08_backtest_horizon_by_variant.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(piv))
    for col in [c for c in piv.columns if c != "Copom"]:
        ax.plot(xs, piv[col], marker="o", lw=1 if col != "baseline" else 2, label=col)
    ax.plot(xs, piv["Copom"], marker="D", color="k", lw=2, label="Copom")
    ax.set_xticks(xs); ax.set_xticklabels(["%d\n%s" % (n, cm.meeting(n)["horizon"][0]) for n in piv.index], fontsize=8)
    ax.set_ylabel("free prices, 4 quarters to the relevant horizon (%)"); ax.legend(fontsize=8); ax.set_title("Free-price projections at the relevant horizon")
    fig.tight_layout(); fig.savefig(OUTPUT / "figs" / "08_backtest_horizon.png", dpi=110)
print("saved outputs to", OUTPUT)
