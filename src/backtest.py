"""
Backtest of the Copom-style projections at past meetings, with the unadjusted model output and the output adjusted
by the model's error at the previous meeting. Used by notebooks/08_backtest_copom.py and by the app.
"""
import json
import time

import numpy as np
import pandas as pd

import projection as pj
import copom_meetings as cm
import adjust
import model_system as ms
from config import OUTPUT, DATA_PROC


def realized():
    act = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
    act.index = pd.PeriodIndex(act.index, freq="Q")
    real4 = {c: 100 * np.expm1(act[c].rolling(4).sum() / 100) for c in ["pi_ipca", "pi_livres", "pi_monit"]}
    real_year = {}
    for y in sorted({p.year for p in act.index}):
        sel = act[[p.year == y for p in act.index]]
        if len(sel) == 4 and sel["pi_ipca"].notna().all():
            real_year[y] = {c: float(100 * np.expm1(sel[c].sum() / 100)) for c in ["pi_ipca", "pi_livres", "pi_monit"]}
    return real4, real_year


def run_backtest(meetings=None, settings=None, n_draws=300, model=None, progress=None, save=True, tag="08"):
    """Run the projections at the given meetings (default: all known) with `settings` (default: projection.ADOPTED),
    store each result in the cache (for the previous-meeting adjustment) and return a dict with the comparison
    table (unadjusted and adjusted), summaries, assumptions, quarterly paths and the damping evaluation."""
    meetings = meetings or cm.all_meetings()
    settings = dict(pj.ADOPTED, **(settings or {}))
    model = model or pj.load_model()
    real4, real_year = realized()
    rows, assume, quarterly, records, outs = [], [], [], [], {}
    t0 = time.time()
    for k, m in enumerate(meetings):
        adm_years = {y: v[2] for y, v in m["years"].items()}
        adm_h = (m["horizon"][0], m["horizon"][3]) if m.get("horizon") and m["horizon"][3] is not None else None
        out = pj.project(m["decision"], m["focus"], m["fx_start"], adm_years, adm_h, H=12, model=model, n_draws=n_draws, seed=m["n"], **settings)
        outs[m["n"]] = out
        rec = adjust.save_meeting_result(m, out, settings)
        records.append(rec)
        errs, prev_n = adjust.previous_errors(m["n"], meetings)
        for r in adjust.adjust_targets(adjust.target_rows(m, out), errs):
            y = int(r["target"]) if r["type"] != "relevant horizon" else None
            rows.append({"meeting": m["n"], "decision": m["decision"], "target": r["target"], "type": r["type"], "previous_meeting": prev_n,
                         "copom_ipca": r["copom_ipca"], "model_ipca": r["model_ipca"], "adjusted_ipca": r["adjusted_ipca"],
                         "copom_livres": r["copom_livres"], "model_livres": r["model_livres"], "adjusted_livres": r["adjusted_livres"],
                         "copom_adm": r["copom_adm"], "model_adm": r["model_adm"],
                         "real_ipca": real_year.get(y, {}).get("pi_ipca", np.nan) if y else real4["pi_ipca"].get(pd.Period(r["target"], freq="Q"), np.nan),
                         "real_livres": real_year.get(y, {}).get("pi_livres", np.nan) if y else real4["pi_livres"].get(pd.Period(r["target"], freq="Q"), np.nan)})
        p, v = out["paths"], out["vintage"]
        a0 = out["a0"]
        sq = v["selic_q"]
        assume.append({"meeting": m["n"], "decision": m["decision"], "focus": m["focus"], "fx_start": m["fx_start"],
                       "selic_now": float(out["table"].loc[out["q0"], "selic_pct"]),
                       "selic_path_q1_q6": ", ".join("%.2f" % x for x in sq.loc[out["q0"] + 1:out["q0"] + 6]),
                       "ie_F": p["ie_F"], "pie_F": p["pie_F"], "steo": p["steo_vintage"], "brent_F": float(p["brent_monthly"].loc[pd.Timestamp(m["focus"]).to_period("M")]),
                       "adm_years": json.dumps(p["adm_years"]), "nowcast_months": ", ".join(v["monthly_prices"].index[v["monthly_prices"]["source"] != "IBGE"].astype(str)),
                       "gap_q0": float(a0[ms.iv["h"]]), "rpm_gap": float(out["gap_obs"].iloc[-1]) if out.get("gap_obs") is not None and len(out["gap_obs"]) else np.nan,
                       "neutral_q0": float(100 * np.expm1((a0[ms.iv["rrIS"]] + v["df"]["rr_trend"].iloc[-1]) / 100)),
                       "oni_q0": float(v["df"]["oni"].iloc[-1]), "cut_gdp": str(v["cut"]["gdp"]), "cut_ipca": str(v["cut"]["ipca"])})
        tq = out["table"].copy(); tq.insert(0, "meeting", m["n"])
        tq["pi_4q_adjusted"] = adjust.adjust_path(out["table"], out["q0"], errs, horizon_q=m["horizon"][0] if m.get("horizon") else None)
        if "bands" in out:
            tq = tq.join(out["bands"])
        quarterly.append(tq)
        if progress:
            progress(k + 1, len(meetings), m["n"])
    tab = pd.DataFrame(rows)
    for c in ["ipca", "livres"]:
        tab["diff_" + c] = tab["model_" + c] - tab["copom_" + c]
        tab["diff_adjusted_" + c] = tab["adjusted_" + c] - tab["copom_" + c]

    def summ_of(frame, col):
        parts = []
        for typ, g in list(frame.groupby("type")) + [("all", frame)]:
            parts.append(pd.DataFrame({"n": [len(g)], "mean": [g[col].mean()], "mae": [g[col].abs().mean()], "max_abs": [g[col].abs().max()]}, index=[typ]))
        return pd.concat(parts)
    cols = [("unadjusted IPCA", "diff_ipca"), ("adjusted IPCA", "diff_adjusted_ipca"), ("unadjusted free", "diff_livres"), ("adjusted free", "diff_adjusted_livres")]
    summary = pd.concat({name: summ_of(tab, col) for name, col in cols}, axis=1)
    # the adjusted comparison only exists from the second meeting onwards
    has_prev = tab["previous_meeting"].notna()
    summary_adj = pd.concat({name: summ_of(tab[has_prev], col)[["mean", "mae"]] for name, col in cols}, axis=1)
    damp_by_type, damp_all, _ = adjust.evaluate_damping(records)
    res = {"table": tab, "summary": summary, "summary_adjusted": summary_adj, "assumptions": pd.DataFrame(assume),
           "quarterly": pd.concat(quarterly), "damping_by_type": damp_by_type, "damping": damp_all, "settings": settings,
           "outputs": outs, "elapsed": time.time() - t0}
    if save:
        tab.to_csv(OUTPUT / ("%s_backtest_table.csv" % tag), index=False)
        summary.to_csv(OUTPUT / ("%s_backtest_summary.csv" % tag))
        summary_adj.to_csv(OUTPUT / ("%s_backtest_summary_adjusted.csv" % tag))
        res["assumptions"].to_csv(OUTPUT / ("%s_backtest_assumptions.csv" % tag), index=False)
        res["quarterly"].to_csv(OUTPUT / ("%s_backtest_quarterly.csv" % tag))
        damp_by_type.to_csv(OUTPUT / ("%s_backtest_damping_by_type.csv" % tag)); damp_all.to_csv(OUTPUT / ("%s_backtest_damping.csv" % tag))
        known = tab[tab["real_ipca"].notna() & (tab["type"] != "relevant horizon")].copy()
        if len(known):
            known["copom_err"] = known["copom_ipca"] - known["real_ipca"]; known["model_err"] = known["model_ipca"] - known["real_ipca"]
            known["adjusted_err"] = known["adjusted_ipca"] - known["real_ipca"]
            known.to_csv(OUTPUT / ("%s_backtest_known_outcomes.csv" % tag), index=False)
        (OUTPUT / ("%s_backtest_settings.json" % tag)).write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return res


def plot_backtest(res, meetings=None, path=None):
    """Panel figure: unadjusted and adjusted four-quarter IPCA paths with bands, realized inflation and the Copom's
    published projections, one panel per meeting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meetings = meetings or cm.all_meetings()
    real4, _ = realized()
    n = len(meetings)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4 * nrow), sharey=True, squeeze=False)
    for ax, m in zip(axes.ravel(), meetings):
        out = res["outputs"][m["n"]]
        tq = res["quarterly"][res["quarterly"]["meeting"] == m["n"]]
        q0 = out["q0"]
        x = tq.index.to_timestamp()
        proj = tq[tq["source"] == "projection"]
        xp = proj.index.to_timestamp()
        if "p5" in tq.columns:
            ax.fill_between(xp, proj["p5"], proj["p95"], color="C0", alpha=0.12)
            ax.fill_between(xp, proj["p25"], proj["p75"], color="C0", alpha=0.25)
        ax.plot(x, tq["pi_4q"], color="C0", lw=2, label="model, unadjusted")
        ax.plot(xp, proj["pi_4q_adjusted"], color="C0", lw=1.5, ls="--", label="adjusted by previous error")
        r = real4["pi_ipca"].loc[q0 - 8:]
        ax.plot(r.index.to_timestamp(), r.values, color="k", lw=1.2, label="realized")
        for y, vals in m["years"].items():
            ax.plot(pd.Timestamp(year=y, month=10, day=1), vals[0], marker="s", color="C3", ms=7, ls="none", label="Copom" if y == min(m["years"]) else None)
        if m.get("horizon"):
            hq = pd.Period(m["horizon"][0], freq="Q")
            ax.plot(hq.to_timestamp(), m["horizon"][1], marker="D", color="C3", ms=7, ls="none")
        ax.axhline(3.0, color="grey", lw=0.6, ls="--")
        ax.axvline(q0.to_timestamp(how="end"), color="grey", lw=0.6)
        ax.set_title("%d (%s)" % (m["n"], m["decision"]), fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.set_xlim((q0 - 6).to_timestamp(), (q0 + 9).to_timestamp())
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Conditional projections at each Copom meeting (bands: 50% and 90%) versus the Copom's published projections (red)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig
