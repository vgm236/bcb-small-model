"""
Interface for the Copom-style projections of the BCB small-scale model (Streamlit).

Pages
* Next meeting      – reference scenario for the next scheduled Copom meeting (Focus survey, PTAX convention, oil,
                      administered prices), unadjusted and adjusted by the model's error at the previous meeting;
* Scenario builder  – own assumptions for the Selic path, the BRL/USD path, administered inflation and the
                      12-month inflation expectation, compared with the reference scenario;
* Backtest          – the model versus the Copom's published projections at every meeting since July 2025
                      (new meetings are added automatically from the minutes), unadjusted and adjusted;
* Data and updates  – refresh the inputs (Focus, Selic, PTAX, prices, Brent, ...) and the Copom history.

Launch with run_app.bat (uses the project's virtual environment) or `python -m streamlit run app/app.py`.
"""
import sys
import json
import pathlib
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import projection as pj          # noqa: E402
import copom_meetings as cm      # noqa: E402
import adjust                    # noqa: E402
import backtest as bt            # noqa: E402
import fetch                     # noqa: E402
import model_system as ms        # noqa: E402
from config import OUTPUT, DATA_RAW  # noqa: E402

st.set_page_config(page_title="BCB small model: Copom projections", layout="wide")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def data_stamp():
    """Changes whenever an input file or a cached backtest result changes (invalidates the projection cache)."""
    files = list(DATA_RAW.glob("*.csv")) + list(DATA_RAW.glob("*.txt")) + list((OUTPUT / "backtest_cache").glob("*.json")) \
        + [cm.HISTORY] + list(DATA_RAW.glob("eia/steo/*.xlsx"))
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


@st.cache_resource(show_spinner="Loading the estimated model...")
def load_model():
    return pj.load_model()


def latest_focus_date():
    return pd.Timestamp(fetch.focus_selic_reunioes()["Data"].max())


def slim(out):
    t = out["table"].copy()
    t.index = t.index.astype(str)
    res = {"table": t, "years": out["years"], "q0": str(out["q0"]), "settings": out["settings"],
           "state": {"gap": float(out["a0"][ms.iv["h"]]), "pie": float(100 * np.expm1(out["a0"][ms.iv["pie"]] / 100)),
                     "neutral": float(100 * np.expm1((out["a0"][ms.iv["rrIS"]] + out["vintage"]["df"]["rr_trend"].iloc[-1]) / 100)),
                     "ie": out["paths"]["ie_F"], "oni": float(out["vintage"]["df"]["oni"].iloc[-1])},
           "assumptions": {"fx_start": out["paths"]["fx_start"], "adm_years": out["paths"]["adm_years"], "steo": out["paths"]["steo_vintage"],
                           "brent_F": float(out["paths"]["brent_monthly"].loc[out["paths"]["snapshot"]["F"].to_period("M")]),
                           "selic_q": {str(k): float(v) for k, v in out["vintage"]["selic_q"].items()},
                           "ptax_q": {str(k): float(v) for k, v in out["vintage"]["ptax_q"].items()},
                           "oni_path": {str(k): float(v) for k, v in out["paths"]["oni_path"].items()},
                           "nowcast": out["vintage"]["monthly_prices"][out["vintage"]["monthly_prices"]["source"] != "IBGE"].round(3)}}
    if "bands" in out:
        b = out["bands"].copy(); b.index = b.index.astype(str); res["bands"] = b
    if out.get("gap") is not None:
        g = out["gap"].copy(); g.index = g.index.astype(str); res["gap"] = g
    if out.get("adm_model") is not None:
        am = out["adm_model"]
        items = am["quarterly"].copy(); items.index = items.index.astype(str)
        res["adm_items"] = items
        res["adm_estimates"] = am["estimates"]
        res["adm_surcharge"] = {str(k): float(v) for k, v in am["surcharge"].items()}
        res["feedback"] = out.get("feedback_log", [])
    return res


@st.cache_data(show_spinner="Running the model...")
def run_projection(params_json, stamp):
    p = json.loads(params_json)
    adm_src = p.get("adm_source", "input")
    flag = None
    if adm_src == "model" and p.get("flag_december"):
        y0 = int(p["decision"][:4])
        flag = {y: p["flag_december"] for y in range(y0, y0 + 4)}
    out = pj.project(p["decision"], p["focus"], p["fx_start"], {int(k): v for k, v in p["adm_years"].items()} if adm_src == "input" else None,
                     tuple(p["adm_horizon"]) if (p.get("adm_horizon") and adm_src == "input") else None, H=12, model=load_model(), n_draws=p["n_draws"],
                     neutral_rate=p["neutral_rate"], climate=p["climate"], climate_index=p["climate_index"], coefficients=p["coefficients"],
                     gap_anchor=p["gap_anchor"], adm_profile=p["adm_profile"], selic_override=p.get("selic_override"),
                     fx_override=p.get("fx_override"), pie_override=p.get("pie_override"),
                     adm_source=adm_src, adm_theta=p.get("adm_theta", "bcb"), flag_december=flag)
    return slim(out)


def copom_style(res, years, horizon):
    rows = {}
    for y in years:
        if y in res["years"]:
            v = res["years"][y]
            rows[str(y)] = {"IPCA": v["pi"], "IPCA livres": v["piL"], "IPCA administrados": v["piA"]}
    t = res["table"]
    if horizon in t.index:
        rows[horizon + " (4 quarters)"] = {"IPCA": t.loc[horizon, "pi_4q"], "IPCA livres": t.loc[horizon, "piL_4q"], "IPCA administrados": t.loc[horizon, "piA_4q"]}
    return pd.DataFrame(rows).T


def adjusted_style(table_df, errors, year0, damping=1.0):
    adj = table_df.copy()
    for idx in adj.index:
        if idx.startswith(str(year0)) and "quarters" not in idx:
            typ = "current year"
        elif "quarters" in idx:
            typ = "relevant horizon"
        else:
            typ = "next year"
        adj.loc[idx, "IPCA"] = table_df.loc[idx, "IPCA"] - damping * adjust.error_for_type(errors, typ, "ipca")
        adj.loc[idx, "IPCA livres"] = table_df.loc[idx, "IPCA livres"] - damping * adjust.error_for_type(errors, typ, "livres")
    return adj


def chart(res, title, errors=None, q0=None, horizon=None, compare=None, damping=1.0):
    import plotly.graph_objects as go
    t = res["table"]
    x = pd.PeriodIndex(t.index, freq="Q").to_timestamp(how="end")
    fig = go.Figure()
    if "bands" in res:
        b = res["bands"]
        xb = pd.PeriodIndex(b.index, freq="Q").to_timestamp(how="end")
        fig.add_trace(go.Scatter(x=list(xb) + list(xb[::-1]), y=list(b["p95"]) + list(b["p5"][::-1]), fill="toself", fillcolor="rgba(31,119,180,0.12)",
                                 line=dict(color="rgba(0,0,0,0)"), name="90% band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=list(xb) + list(xb[::-1]), y=list(b["p75"]) + list(b["p25"][::-1]), fill="toself", fillcolor="rgba(31,119,180,0.25)",
                                 line=dict(color="rgba(0,0,0,0)"), name="50% band", hoverinfo="skip"))
    hist = t[t["source"] != "projection"]
    fig.add_trace(go.Scatter(x=x, y=t["pi_4q"], mode="lines", name="IPCA, 4 quarters (unadjusted)", line=dict(color="#1f77b4", width=3)))
    if errors is not None and q0 is not None:
        tt = t.copy(); tt.index = pd.PeriodIndex(tt.index, freq="Q")
        adj = adjust.adjust_path(tt, pd.Period(q0, freq="Q"), errors, damping=damping, horizon_q=horizon)
        fig.add_trace(go.Scatter(x=x, y=adj.values, mode="lines", name="IPCA, 4 quarters (adjusted)", line=dict(color="#1f77b4", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=t["piL_4q"], mode="lines", name="free prices", line=dict(color="#ff7f0e", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=t["piA_4q"], mode="lines", name="administered (assumption)", line=dict(color="#2ca02c", width=1.5, dash="dot")))
    if compare is not None:
        tc = compare["table"]
        xc = pd.PeriodIndex(tc.index, freq="Q").to_timestamp(how="end")
        fig.add_trace(go.Scatter(x=xc, y=tc["pi_4q"], mode="lines", name="reference scenario (unadjusted)", line=dict(color="grey", width=2)))
    fig.add_hline(y=3.0, line=dict(color="grey", dash="dash", width=1))
    fig.add_hline(y=4.5, line=dict(color="lightgrey", dash="dot", width=1)); fig.add_hline(y=1.5, line=dict(color="lightgrey", dash="dot", width=1))
    if q0 is not None:
        fig.add_vline(x=pd.Period(q0, freq="Q").to_timestamp(how="end"), line=dict(color="grey", width=1))
    if horizon is not None and horizon in t.index:
        fig.add_trace(go.Scatter(x=[pd.Period(horizon, freq="Q").to_timestamp(how="end")], y=[t.loc[horizon, "pi_4q"]], mode="markers",
                                 marker=dict(symbol="diamond", size=12, color="#1f77b4"), name="relevant horizon"))
    fig.update_layout(title=title, yaxis_title="% (accumulated in 4 quarters)", height=460, legend=dict(orientation="h", y=-0.2), margin=dict(l=40, r=20, t=50, b=40))
    return fig


def sidebar_settings():
    st.sidebar.header("Model settings")
    s = dict(pj.ADOPTED)
    s["neutral_rate"] = st.sidebar.number_input("Neutral real rate (% p.a.)", value=float(s["neutral_rate"] or 5.0), step=0.25, format="%.2f")
    s["climate"] = st.sidebar.selectbox("Climate (El Niño/La Niña) after the current quarter", ["realtime", "off", "realized"],
                                        index=["realtime", "off", "realized"].index(s["climate"]),
                                        help="realtime: latest Niño 3.4 readings then AR(1) decay; off: no climate effects; realized: published seasons (hindsight)")
    s["climate_index"] = st.sidebar.selectbox("Climate index", ["oni", "roni"], index=["oni", "roni"].index(s["climate_index"]))
    s["gap_anchor"] = "rpm" if st.sidebar.checkbox("Anchor the output gap on the BCB's published estimates", value=s["gap_anchor"] == "rpm") else None
    s["coefficients"] = st.sidebar.selectbox("Coefficients", ["estimated", "bcb"], index=["estimated", "bcb"].index(s["coefficients"]),
                                             help="estimated: posterior mode of this project; bcb: modes of the June-2024 box")
    s["adm_profile"] = st.sidebar.selectbox("Monthly profile of administered prices", ["uniform", "seasonal"], index=["uniform", "seasonal"].index(s["adm_profile"]))
    st.sidebar.subheader("Administered prices")
    src = st.sidebar.selectbox("Source of the administered-price projection",
                               ["Annual figures (Copom projection / Focus medians)", "BCB administered model, BCB calibration", "BCB administered model, re-estimated parameters"],
                               index=0, help="The administered model (June-2025 box) projects the 24 administered items with the free-price feedback; the annual figures are an outside input")
    s["adm_source"] = "input" if src.startswith("Annual") else "model"
    s["adm_theta"] = "bcb" if "calibration" in src else "estimated"
    flag_opts = ["verde", "amarela", "vermelha p1", "vermelha p2"]
    s["flag_december"] = st.sidebar.selectbox("Tariff flag assumed for December (model source)", flag_opts, index=0,
                                              help="The Copom states its December flag assumption in every statement; the flag is held through the current quarter and the December assumption applies afterwards")
    s["n_draws"] = st.sidebar.slider("Simulation draws for the bands", 0, 1000, 300, step=100)
    s["damping"] = st.sidebar.slider("Weight of the previous-meeting error in the adjusted output", 0.0, 1.0, 1.0, step=0.25)
    return s


def previous_error_block(n_next, meetings):
    errs, prev_n = adjust.previous_errors(n_next, meetings)
    if errs is None:
        st.info("No stored backtest result for a previous meeting: run the backtest once (page Backtest) to enable the adjusted output.")
        return None, None
    rows = [{"type": k, "target": v["target"], "error IPCA (model minus Copom)": v["ipca"], "error free prices": v["livres"]} for k, v in errs.items()]
    st.caption("Adjustment uses the model's error at meeting %d:" % prev_n)
    st.dataframe(pd.DataFrame(rows).set_index("type").round(2), use_container_width=True)
    return errs, prev_n


def next_meeting_defaults():
    n, d = cm.next_meeting()
    if n is None:
        n, d = cm.next_meeting(pd.Timestamp.today() - pd.Timedelta(days=60))
    F = latest_focus_date()
    fx, win = pj.fx_start_convention(F)
    snap = pj.focus_snapshot(F)
    q0 = d.to_period("Q")
    adm = {y: round(snap["adm_annual"][y], 2) for y in [q0.year, q0.year + 1, q0.year + 2] if y in snap["adm_annual"]}
    return {"n": n, "decision": str(d.date()), "focus": str(F.date()), "fx_start": fx, "fx_window": (str(win.index[0].date()), str(win.index[-1].date()), float(win.mean())),
            "adm_years": adm, "snap": snap, "q0": str(q0), "horizon": cm.relevant_horizon(d)}


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Page", ["Next meeting", "Scenario builder", "Administered prices", "Backtest", "Data and updates"])
settings = sidebar_settings()
meetings = cm.all_meetings()
stamp = data_stamp()

if page == "Next meeting":
    st.title("Reference scenario for the next Copom meeting")
    dft = next_meeting_defaults()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next meeting", "%dth, %s" % (dft["n"], dft["decision"]))
    c2.metric("Focus survey used", dft["focus"])
    c3.metric("Exchange rate start (BRL/USD)", "%.2f" % dft["fx_start"], help="average PTAX %s to %s = %.4f, rounded to 0.05" % (dft["fx_window"][0], dft["fx_window"][1], dft["fx_window"][2]))
    c4.metric("Relevant horizon", dft["horizon"], help="six quarters after the quarter of the meeting")
    with st.expander("Assumptions (editable)", expanded=True):
        cols = st.columns(len(dft["adm_years"]) + 1)
        adm = {}
        for k, (y, v) in enumerate(dft["adm_years"].items()):
            adm[y] = cols[k].number_input("Administered prices %d (%%)" % y, value=float(v), step=0.1, format="%.2f", key="adm_%d" % y,
                                          help="default: Focus median 'IPCA Administrados' of %s" % dft["focus"])
        fx_start = cols[-1].number_input("Exchange rate start", value=float(dft["fx_start"]), step=0.05, format="%.2f")
        snap = dft["snap"]
        st.write("Focus Selic per meeting (%s): " % snap["selic_date"].date() + ", ".join("%s: %.2f" % (k, v) for k, v in sorted(snap["selic_meetings"].items(), key=lambda kv: (kv[0].split("/")[1], int(kv[0].split("/")[0][1:])))))
        st.write("Focus end-year Selic: %s; IPCA: %s; 12-month inflation expectation: %.2f%%" % (snap["selic_annual"], {k: round(v, 2) for k, v in snap["ipca_annual"].items()}, snap["pie_12m"]))
    if st.button("Run projection", type="primary"):
        params = {"decision": dft["decision"], "focus": dft["focus"], "fx_start": fx_start, "adm_years": {str(k): v for k, v in adm.items()}, "adm_horizon": None,
                  "n_draws": settings["n_draws"], "neutral_rate": settings["neutral_rate"], "climate": settings["climate"], "climate_index": settings["climate_index"],
                  "coefficients": settings["coefficients"], "gap_anchor": settings["gap_anchor"], "adm_profile": settings["adm_profile"],
                  "adm_source": settings["adm_source"], "adm_theta": settings["adm_theta"], "flag_december": settings["flag_december"]}
        st.session_state["next_params"] = params
    if "next_params" in st.session_state:
        params = st.session_state["next_params"]
        res = run_projection(json.dumps(params, sort_keys=True), stamp)
        q0 = res["q0"]; year0 = int(q0[:4])
        years = [year0, year0 + 1, year0 + 2]
        tab = copom_style(res, years, dft["horizon"])
        st.subheader("Projections in the Copom's format (IPCA accumulated in the calendar year / four quarters, %)")
        errs, prev_n = previous_error_block(dft["n"], meetings)
        cA, cB = st.columns(2)
        cA.markdown("**Model, unadjusted**"); cA.dataframe(tab.round(2), use_container_width=True)
        if errs is not None:
            adj = adjusted_style(tab, errs, year0, settings["damping"])
            cB.markdown("**Adjusted by the error at meeting %d** (weight %.2f)" % (prev_n, settings["damping"])); cB.dataframe(adj.round(2), use_container_width=True)
        s = res["state"]
        st.caption("State at the origin %s: output gap %.2f%%, neutral real rate %.2f%%, 12-month inflation expectations %.2f%%, expected Selic 12 months %.2f%%, ONI %.2f. "
                   "Brent at the Focus date %.1f US$/bbl (STEO %s)." % (q0, s["gap"], s["neutral"], s["pie"], s["ie"], s["oni"], res["assumptions"]["brent_F"], res["assumptions"]["steo"]))
        st.plotly_chart(chart(res, "Inflation projection, reference scenario (meeting %d)" % dft["n"], errs, q0, dft["horizon"], damping=settings["damping"]), use_container_width=True)
        if res.get("gap") is not None:
            with st.expander("Output gap: history and projection", expanded=True):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                sys.path.insert(0, str(ROOT / "app"))
                from gap_tab import gap_chart
                figg = plt.figure(figsize=(12, 4.6))
                gap_chart(figg, res["gap"].loc["2012Q1":], q0, dft["horizon"])
                figg.tight_layout(); st.pyplot(figg)
                gg = res["gap"]
                st.dataframe(pd.concat([gg[gg["source"] != "projection"].tail(8), gg[gg["source"] == "projection"]])[[c for c in ["smoothed", "filtered", "bcb", "fpib", "nuci_implied", "emp_implied", "caged_implied", "gap", "p5", "p25", "p75", "p95"] if c in gg.columns]].round(2).T, use_container_width=True)
        if "adm_items" in res:
            with st.expander("Administered-price model: items (quarterly log-%) and flag assumption"):
                it = res["adm_items"]
                show_cols = [c for c in ["5104001", "2202003", "6203", "6101", "5102004", "2101004", "2201004", "5101001", "band", "adm", "ipca"] if c in it.columns]
                names = {"5104001": "gasolina", "2202003": "energia elétrica", "6203": "plano de saúde", "6101": "farmacêuticos", "5102004": "emplacamento",
                         "2101004": "água e esgoto", "2201004": "gás de botijão", "5101001": "ônibus urbano", "band": "flag contribution", "adm": "administered (aggregate)", "ipca": "headline used"}
                st.dataframe(it.loc[q0:, show_cols].rename(columns=names).round(2).T, use_container_width=True)
                st.caption("Feedback iterations (max change of the administered path, pp): %s. Surcharge path (R$/MWh): %s" % (
                    ", ".join("%.3f" % c for c in res.get("feedback", [])), {k: v for k, v in list(res["adm_surcharge"].items())[-6:]}))
        with st.expander("Quarterly path and assumptions"):
            st.dataframe(res["table"][["piL_sa", "piL", "piA", "pi", "pi_4q", "piL_4q", "piA_4q", "h", "pie", "selic_pct", "de", "source"]].round(2), use_container_width=True)
            a = res["assumptions"]
            st.dataframe(pd.DataFrame({"Selic (quarterly average, %)": pd.Series(a["selic_q"]), "BRL/USD (quarterly average)": pd.Series(a["ptax_q"]),
                                       "ONI": pd.Series(a["oni_path"])}).loc[q0:].round(3), use_container_width=True)
            st.write("Nowcast months (Focus headline, administered assumption):"); st.dataframe(a["nowcast"], use_container_width=True)
            if "bands" in res:
                st.write("Bands (model shocks only):"); st.dataframe(res["bands"].round(2), use_container_width=True)
        st.download_button("Download quarterly path (CSV)", res["table"].to_csv().encode("utf-8"), "projection_%s.csv" % dft["decision"], "text/csv")

elif page == "Scenario builder":
    st.title("Scenario builder: own assumptions for the Selic, the BRL and inflation")
    dft = next_meeting_defaults()
    q0 = pd.Period(dft["q0"], freq="Q")
    st.caption("Origin: meeting %d (%s), Focus survey of %s. Edit the paths below; quarters not edited keep the reference-scenario values. "
               "The reference scenario is the Focus Selic path, the PPP exchange-rate path from %.2f and the Focus administered-price medians." % (dft["n"], dft["decision"], dft["focus"], dft["fx_start"]))
    ref_params = {"decision": dft["decision"], "focus": dft["focus"], "fx_start": dft["fx_start"], "adm_years": {str(k): v for k, v in dft["adm_years"].items()}, "adm_horizon": None,
                  "n_draws": 0, "neutral_rate": settings["neutral_rate"], "climate": settings["climate"], "climate_index": settings["climate_index"],
                  "coefficients": settings["coefficients"], "gap_anchor": settings["gap_anchor"], "adm_profile": settings["adm_profile"],
                  "adm_source": settings["adm_source"], "adm_theta": settings["adm_theta"], "flag_december": settings["flag_december"]}
    ref = run_projection(json.dumps(ref_params, sort_keys=True), stamp)
    quarters = [str(q0 + k) for k in range(0, 9)]
    base = pd.DataFrame({"quarter": quarters,
                         "Selic (% p.a., quarterly average)": [round(ref["assumptions"]["selic_q"].get(q, np.nan), 2) for q in quarters],
                         "BRL/USD (quarterly average)": [round(ref["assumptions"]["ptax_q"].get(q, np.nan), 3) for q in quarters]})
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Selic and exchange-rate paths** (reference values shown; edit the cells)")
        edited = st.data_editor(base, hide_index=True, use_container_width=True, num_rows="fixed", key="paths_editor")
    with c2:
        st.markdown("**Inflation assumptions**")
        adm = {}
        for y, v in dft["adm_years"].items():
            adm[y] = st.number_input("Administered prices %d (%%)" % y, value=float(v), step=0.1, format="%.2f", key="sb_adm_%d" % y)
        pie = st.number_input("12-month inflation expectation at the origin (%)", value=float(round(dft["snap"]["pie_12m"], 2)), step=0.1, format="%.2f",
                              help="Focus median 'IPCA 12 meses à frente (suavizada)'; the model's expectations equation takes over afterwards")
        fx_start = st.number_input("Exchange-rate start (BRL/USD)", value=float(dft["fx_start"]), step=0.05, format="%.2f")
        st.caption("Selic changes apply from the day after the meeting; exchange-rate levels apply from the Focus date; "
                   "quarters beyond the table follow the reference paths from the last edited value.")
    if st.button("Run scenario", type="primary"):
        selic_ov = {r["quarter"]: float(r["Selic (% p.a., quarterly average)"]) for _, r in edited.iterrows() if pd.notna(r["Selic (% p.a., quarterly average)"])}
        fx_ov = {r["quarter"]: float(r["BRL/USD (quarterly average)"]) for _, r in edited.iterrows() if pd.notna(r["BRL/USD (quarterly average)"])}
        # keep only edited values (so that untouched quarters follow the reference daily paths exactly)
        selic_ov = {q: v for q, v in selic_ov.items() if abs(v - round(ref["assumptions"]["selic_q"].get(q, v), 2)) > 1e-9}
        fx_ov = {q: v for q, v in fx_ov.items() if abs(v - round(ref["assumptions"]["ptax_q"].get(q, v), 3)) > 1e-9}
        # carry the last edited value forward beyond the table
        if selic_ov:
            last_q = max(selic_ov, key=lambda q: pd.Period(q, freq="Q"))
            for k in range(1, 13):
                qq = str(pd.Period(last_q, freq="Q") + k)
                if qq not in selic_ov and pd.Period(qq, freq="Q") <= q0 + 12:
                    selic_ov[qq] = selic_ov[last_q]
        if fx_ov:
            last_q = max(fx_ov, key=lambda q: pd.Period(q, freq="Q"))
            for k in range(1, 13):
                qq = str(pd.Period(last_q, freq="Q") + k)
                if qq not in fx_ov and pd.Period(qq, freq="Q") <= q0 + 12:
                    fx_ov[qq] = fx_ov[last_q] * (1 + (3.0 - 2.0) / 100.0) ** (k / 4.0)
        params = dict(ref_params, adm_years={str(k): v for k, v in adm.items()}, fx_start=fx_start, n_draws=settings["n_draws"],
                      selic_override=selic_ov or None, fx_override=fx_ov or None,
                      pie_override=None if abs(pie - dft["snap"]["pie_12m"]) < 1e-6 else pie)
        st.session_state["scenario_params"] = params
    if "scenario_params" in st.session_state:
        params = st.session_state["scenario_params"]
        res = run_projection(json.dumps(params, sort_keys=True), stamp)
        year0 = int(dft["q0"][:4])
        years = [year0, year0 + 1, year0 + 2]
        errs, prev_n = adjust.previous_errors(dft["n"], meetings)
        tab_s, tab_r = copom_style(res, years, dft["horizon"]), copom_style(ref, years, dft["horizon"])
        st.subheader("Projections (IPCA accumulated in the calendar year / four quarters, %)")
        cA, cB, cC = st.columns(3)
        cA.markdown("**Own scenario, unadjusted**"); cA.dataframe(tab_s.round(2), use_container_width=True)
        cB.markdown("**Reference scenario, unadjusted**"); cB.dataframe(tab_r.round(2), use_container_width=True)
        if errs is not None:
            cC.markdown("**Own scenario, adjusted by the error at meeting %d**" % prev_n); cC.dataframe(adjusted_style(tab_s, errs, year0, settings["damping"]).round(2), use_container_width=True)
        s = res["state"]
        st.caption("State at the origin: output gap %.2f%%, neutral rate %.2f%%, 12-month expectations %.2f%%, expected Selic 12 months %.2f%%." % (s["gap"], s["neutral"], s["pie"], s["ie"]))
        st.plotly_chart(chart(res, "Own scenario versus the reference scenario", errs, dft["q0"], dft["horizon"], compare=ref, damping=settings["damping"]), use_container_width=True)
        with st.expander("Quarterly paths"):
            a = res["assumptions"]
            st.dataframe(pd.DataFrame({"Selic (%)": pd.Series(a["selic_q"]), "BRL/USD": pd.Series(a["ptax_q"]), "IPCA 4q (own)": res["table"]["pi_4q"],
                                       "IPCA 4q (reference)": ref["table"]["pi_4q"], "free prices 4q (own)": res["table"]["piL_4q"], "gap (own)": res["table"]["h"]}).loc[dft["q0"]:].round(3), use_container_width=True)
        st.download_button("Download scenario path (CSV)", res["table"].to_csv().encode("utf-8"), "scenario_%s.csv" % dft["decision"], "text/csv")

elif page == "Administered prices":
    st.title("Administered prices: the BCB's medium-term model")
    st.write("Twenty-four quarterly item equations calibrated by the BCB (Monetary Policy Report, June 2025). The table compares the BCB's "
             "coefficients with OLS estimates on the same regressors (re-estimated at every run from the IBGE subitems, ANEEL flags, Brent, PTAX, US CPI).")
    if st.button("Re-download the item data and re-estimate"):
        import administered as adm_mod, adm_projection as admp_mod
        with st.spinner("Downloading and estimating..."):
            admp_mod.data_cached(refresh=True)
        st.cache_data.clear()
    import administered as adm_mod, adm_projection as admp_mod
    data_adm = admp_mod.data_cached()
    tab_est, theta_hat = adm_mod.estimate(data_adm)
    st.caption("Data through %s. Reference tariff for the flag conversion: %s R$/MWh (2024 prices)." % (data_adm["q"].index[-1], data_adm["T0"]))
    st.dataframe(tab_est[["item", "parameter", "bcb", "estimate", "se", "t_diff", "sample", "n", "r2"]].round(3), use_container_width=True, height=600)
    agg = adm_mod.aggregate_history(data_adm)
    st.caption("Aggregate of the 24 items versus the official administered index (quarterly): correlation %.4f." % agg.corr().iloc[0, 1])
    q_adm = data_adm["q"]
    origin = q_adm.index[-1] - 8
    proj_q = pd.period_range(origin + 1, q_adm.index[-1], freq="Q")
    exo = q_adm.loc[proj_q, ["ipca", "x", "fx", "cpi_us", "band"]]
    fitB = adm_mod.fit_in_sample(adm_mod.BCB_THETA, data_adm, start="2016Q1")
    dd = pd.DataFrame({"BCB calibration": adm_mod.simulate(adm_mod.BCB_THETA, data_adm, origin, len(proj_q), exo).loc[proj_q, "adm"],
                       "estimated": adm_mod.simulate(theta_hat, data_adm, origin, len(proj_q), exo).loc[proj_q, "adm"],
                       "actual": fitB.loc[proj_q, "adm_actual"]})
    dd.index = dd.index.astype(str)
    st.subheader("Dynamic simulation from %s with realized drivers (quarterly log-%%)" % origin)
    st.line_chart(dd)
    st.dataframe(dd.round(2).T, use_container_width=True)
    if (OUTPUT / "11_backtest_administered_summary.csv").exists():
        st.subheader("Backtest versus the Copom's administered projections (notebook 11)")
        st.dataframe(pd.read_csv(OUTPUT / "11_backtest_administered_summary.csv").round(2), use_container_width=True)
    if (OUTPUT / "figs" / "10_administered_fit.png").exists():
        st.image(str(OUTPUT / "figs" / "10_administered_fit.png"), use_container_width=True)

elif page == "Backtest":
    st.title("Backtest: model versus the Copom's published projections")
    st.caption("Meetings known: %s. New meetings are added when their minutes are published (page Data and updates, or the button below)." % ", ".join("%d (%s)" % (m["n"], m["decision"]) for m in meetings))
    tab_path = OUTPUT / "08_backtest_table.csv"
    if tab_path.exists():
        tab = pd.read_csv(tab_path)
        if "model_ipca" in tab.columns:
            st.subheader("Meeting by meeting (model minus Copom; adjusted = minus the previous meeting's error)")
            show = tab[["meeting", "decision", "target", "type", "copom_ipca", "model_ipca", "adjusted_ipca", "diff_ipca", "diff_adjusted_ipca",
                        "copom_livres", "model_livres", "adjusted_livres", "copom_adm", "real_ipca"]].round(2)
            st.dataframe(show, use_container_width=True, height=min(900, 40 + 36 * len(show)))
            if (OUTPUT / "08_backtest_summary_adjusted.csv").exists():
                st.subheader("Summary (meetings with a previous meeting; mean and mean absolute difference, pp)")
                st.dataframe(pd.read_csv(OUTPUT / "08_backtest_summary_adjusted.csv", header=[0, 1], index_col=0).round(2), use_container_width=True)
            if (OUTPUT / "08_backtest_damping.csv").exists():
                st.subheader("Mean absolute difference by weight of the previous-meeting error")
                st.dataframe(pd.read_csv(OUTPUT / "08_backtest_damping.csv", index_col=0).round(3), use_container_width=True)
        else:
            st.info("The stored backtest table comes from an earlier version; re-run the backtest below.")
        if (OUTPUT / "08_backtest_settings.json").exists():
            st.caption("Settings of the stored run: %s" % json.loads((OUTPUT / "08_backtest_settings.json").read_text()))
        for f in ["08_backtest_panels.png", "08_backtest_scatter.png"]:
            if (OUTPUT / "figs" / f).exists():
                st.image(str(OUTPUT / "figs" / f), use_container_width=True)
    else:
        st.info("No backtest results stored yet.")
    st.subheader("Re-run the backtest with the current settings")
    st.caption("Runs every known meeting (about 10 seconds each with 300 draws) and stores the results used for the adjusted output.")
    if st.button("Update Copom history and re-run the backtest"):
        meetings = cm.update_history(verbose=False)
        prog = st.progress(0.0, text="starting")
        s = {k: settings[k] for k in ["neutral_rate", "climate", "climate_index", "coefficients", "gap_anchor", "adm_profile"]}
        res = bt.run_backtest(meetings, settings=s, n_draws=settings["n_draws"], model=load_model(),
                              progress=lambda k, n, m: prog.progress(k / n, text="meeting %d (%d of %d)" % (m, k, n)))
        bt.plot_backtest(res, meetings, OUTPUT / "figs" / "08_backtest_panels.png")
        st.success("Backtest done in %.0f s. Reload the page to see the updated tables." % res["elapsed"])
        st.cache_data.clear()

else:
    st.title("Data and updates")
    st.write("Inputs are cached in `data/raw`; the Copom history (projections published in the minutes) in `data/processed/copom_history.json`.")
    stamps = []
    for name, f in [("Focus Selic per meeting", "focus_selic_reunioes.csv"), ("Focus 12-month inflation", "focus_ipca_12m_S.csv"),
                    ("Selic target (daily)", "sgs_432_selic_meta_daily.csv"), ("PTAX (daily)", "sgs_1_ptax_venda_daily.csv"), ("IPCA", "sgs_433_ipca.csv"),
                    ("IC-Br", "sgs_27574_icbr.csv"), ("NUCI", "sgs_28561_nuci_fgv_sa.csv"), ("PNAD unemployment", "sgs_24369_desocup_pnadc.csv"),
                    ("Caged stock", "sgs_28763_caged_estoque.csv"), ("Brent (EIA)", "eia_brent_daily.csv")]:
        p = DATA_RAW / f
        if p.exists():
            try:
                s = pd.read_csv(p)
                col = "Data" if "Data" in s.columns else s.columns[0]
                stamps.append({"series": name, "last observation": str(pd.to_datetime(s[col]).max().date()), "downloaded": dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
            except Exception:  # noqa: BLE001
                pass
    st.dataframe(pd.DataFrame(stamps), use_container_width=True, hide_index=True)
    n, d = cm.next_meeting()
    st.write("Next scheduled meeting: %s (%s); relevant horizon %s. Minutes stored for meetings %d to %d." % (n, d.date() if d is not None else "?", cm.relevant_horizon(d) if d is not None else "?", meetings[0]["n"], meetings[-1]["n"]))
    c1, c2 = st.columns(2)
    if c1.button("Update the Copom history from the minutes"):
        with st.spinner("Checking the BCB site for new minutes..."):
            ms_ = cm.update_history(verbose=False)
        st.success("Meetings known: %s" % ", ".join(str(m["n"]) for m in ms_))
        st.cache_data.clear()
    if c2.button("Refresh the market and macro inputs (Focus, Selic, PTAX, prices, activity, Brent, climate)"):
        with st.spinner("Downloading..."):
            pj.refresh_inputs()
        st.success("Inputs refreshed.")
        st.cache_data.clear()
    st.caption("The projection cache is cleared after each update; the next run uses the new data. Re-run the backtest after a new meeting so that the adjusted output uses its error.")
