"""
Real-time ("vintage") version of the quarterly model dataset for a given information date.

For a Copom meeting decided on date D with Focus reference date F (the Friday before the meeting week), the dataset
contains every quarter up to the current quarter q0 (the quarter of D), built only from information that was public
at D (see RELEASE for the publication-lag rules) and from the Focus survey of F. The current quarter is a
"nowcast quarter": the months of q0 whose IPCA was not yet published are filled with the Focus monthly medians
(headline) and the Copom's administered-price assumption (so that free prices follow by the IPCA identity); the
Selic, exchange-rate and commodity paths of the reference scenario complete the quarter.

Filtered objects (HP cycles, STL seasonal factors) are recomputed on the truncated sample; the level series are
extended by simple forecasts before the HP filter to limit the end-point bias (GDP: Focus growth medians; Caged and
partners' GDP: recent average growth; fiscal balance: constant). Data revisions cannot be undone: the current
vintage of each series is used, truncated at the real-time cut-off (the only revised inputs are GDP, Caged and
PNAD data; the IPCA, Focus, Selic, exchange-rate and IC-Br series are never revised).

The neutral real rate follows the Copom's conditioning value (5.00% in the 2025-2026 Monetary Policy Reports) from
2024Q1 onwards, i.e. after the model's estimation sample; the HP trend of the ex-ante real rate that the model uses
in-sample is dominated by the restrictive policy stance in 2024-2026 and rises to about 9%, a level that no
neutral-rate estimate supports. Set neutral_rate=None to keep the HP trend instead.
"""
import numpy as np
import pandas as pd

import fetch
from config import (SGS, SIDRA_PIB_SA, SIDRA_PIB_NOMINAL, IPEA_EMBI, DATA_PROC, HP_LAMBDA, PRE_COVID_END, PI_STAR_SS,
                    SAMPLE_END, CRISIS_2008, CRISIS_2020, PHILLIPS_HIGH_VAR)
from build_dataset import (INFLATION_TARGET, WORLD_WEIGHTS, FISCAL_ONE_OFFS, FISCAL_SEMI_ELASTICITY, _target_series,
                           _target_12m_ahead, _sidra_quarter)
from tsutil import to_quarterly_mean, to_quarterly_sum, quarterly_log_inflation, hp_cycle, seasonal_adjust, splice_by_differences


def LG(x):
    """Annual rate in % -> log form 100*log(1+x/100)."""
    return 100 * np.log1p(np.asarray(x, dtype=float) / 100.0)


def target(year):
    """Calendar-year inflation target (% p.a.); the last value set (3%, continuous target) is kept for later years."""
    return INFLATION_TARGET.get(year, INFLATION_TARGET[max(INFLATION_TARGET)])


N_EXT = 8   # quarters of extension before the HP filter

# Publication-lag rules: period -> first date on which the value is public (conservative approximations of the
# IBGE/FGV/BCB/Ministry of Labour calendars).
RELEASE = {
    "ipca": lambda m: (m + 1).to_timestamp() + pd.Timedelta(days=11),     # IPCA of month m: ~12th of m+1
    "icbr": lambda m: (m + 1).to_timestamp() + pd.Timedelta(days=9),      # IC-Br of month m: first days of m+1
    "nuci": lambda m: m.end_time.normalize(),                             # FGV NUCI: end of the reference month
    "pnadc": lambda m: (m + 2).to_timestamp(),                            # PNADC rolling quarter ending in m: ~end of m+1
    "caged": lambda m: (m + 2).to_timestamp(),                            # Caged of month m: ~end of m+1
    "fiscal": lambda m: (m + 2).to_timestamp(),                           # central-government result of month m: ~end of m+1
    "gdp": lambda q: q.end_time.normalize() + pd.Timedelta(days=65),      # IBGE quarterly accounts: ~2 months after the quarter
    "oecd": lambda q: q.end_time.normalize() + pd.Timedelta(days=60),     # partners' GDP
    "oni": lambda q: (q.asfreq("M", "end") + 1).to_timestamp() + pd.Timedelta(days=9),  # ONI season centred on the quarter: ~10th of the month after the quarter
}


def cutoffs(decision_date):
    D = pd.Timestamp(decision_date)

    def last(rule, freq):
        p = D.to_period(freq)
        while rule(p) > D:
            p -= 1
        return p
    c = {"D": D, "q0": D.to_period("Q")}
    for k, freq in [("ipca", "M"), ("icbr", "M"), ("nuci", "M"), ("pnadc", "M"), ("caged", "M"), ("fiscal", "M"),
                    ("gdp", "Q"), ("oecd", "Q"), ("oni", "Q")]:
        c[k] = last(RELEASE[k], freq)
    return c


def _monthly(s, cut_m=None):
    """SGS monthly series (Timestamp index) -> PeriodIndex 'M', truncated at cut_m."""
    s = s.copy()
    s.index = s.index.to_period("M")
    return s.loc[:cut_m] if cut_m is not None else s


def _extend_log(lx, n_ext, g):
    """Extend a log-level series (PeriodIndex Q) by n_ext quarters; g = growth per quarter (log-%), scalar or
    function(quarter)->growth."""
    lx = lx.copy()
    last = lx.index[-1]
    for k in range(1, n_ext + 1):
        q = last + k
        gq = g(q) if callable(g) else g
        lx.loc[q] = lx.loc[q - 1] + gq
    return lx


def implied_free_weight(ipca, liv, mon):
    """Implicit IPCA weight of free prices from the aggregation identity (monthly, robust rolling median)."""
    m = pd.DataFrame({"ipca": ipca, "liv": liv, "mon": mon}).dropna()
    d = m["liv"] - m["mon"]
    w = ((m["ipca"] - m["mon"]) / d).where(d.abs() > 0.4)
    return w.clip(0.55, 0.9).rolling(13, center=True, min_periods=4).median().interpolate().bfill().ffill()


def commodity_params(cut_m):
    """(weights of the IC-Br sub-indices, elasticity of the USD energy sub-index to Brent) estimated on data <= cut_m."""
    br = {k: _monthly(fetch.sgs(c, name="sgs_%d_%s" % (c, k)), cut_m) for k, c in
          [("icbr", 27574), ("icbr_agro", 27575), ("icbr_metal", 27576), ("icbr_energia", 27577)]}
    X = np.column_stack([np.log(br[k]).diff().dropna().values for k in ["icbr_agro", "icbr_metal", "icbr_energia"]])
    y = np.log(br["icbr"]).diff().dropna().values
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    ptax_m = _monthly(fetch.sgs(SGS["ptax_venda_mensal"], name="sgs_3698_ptax_venda_mensal"), cut_m)
    brent = fetch.eia_brent_daily()
    brent_m = brent.groupby(brent.index.to_period("M")).mean().loc[:cut_m]
    en_usd = np.log(br["icbr_energia"] / ptax_m.reindex(br["icbr_energia"].index)).diff()
    d = pd.DataFrame({"en": en_usd, "oil": np.log(brent_m).diff()}).dropna().loc["2005-01":]
    beta = float(np.linalg.lstsq(np.column_stack([np.ones(len(d)), d["oil"].values]), d["en"].values, rcond=None)[0][1])
    return {"w_agro": float(w[0]), "w_metal": float(w[1]), "w_energia": float(w[2]), "beta_oil": beta}


def world_levels():
    """Partners' real GDP levels (OECD QNA), China rebuilt from growth rates (same construction as build_dataset)."""
    oecd = fetch.oecd_qna_gdp(list(WORLD_WEIGHTS))
    sub = oecd[(oecd["ADJUSTMENT"] == "Y") & (oecd["PRICE_BASE"] == "LR") & (oecd["UNIT_MEASURE"] == "IX")
               & (oecd["TRANSFORMATION"] == "N") & (oecd["TABLE_IDENTIFIER"] == "T0102")]
    levels = {}
    for c, g in sub.groupby("REF_AREA"):
        levels[c] = g.set_index(pd.PeriodIndex(g["TIME_PERIOD"], freq="Q"))["OBS_VALUE"].astype(float).sort_index()
    chn = oecd[(oecd["REF_AREA"] == "CHN") & (oecd["ADJUSTMENT"] == "Y") & (oecd["UNIT_MEASURE"] == "PC")]
    g1s, gys = chn[chn["TRANSFORMATION"] == "G1"], chn[chn["TRANSFORMATION"] == "GY"]
    g1 = g1s.set_index(pd.PeriodIndex(g1s["TIME_PERIOD"], freq="Q"))["OBS_VALUE"].astype(float).sort_index()
    gy = gys.set_index(pd.PeriodIndex(gys["TIME_PERIOD"], freq="Q"))["OBS_VALUE"].astype(float).sort_index()
    if len(g1):
        idx = pd.period_range(gy.index.min(), max(g1.index.max(), gy.index.max()), freq="Q")
        lvl = pd.Series(np.nan, index=idx)
        lvl.loc[g1.index[0]] = 100.0
        for p in g1.index[1:]:
            lvl.loc[p] = lvl.loc[p - 1] * (1 + g1.loc[p] / 100)
        for p in reversed(idx[idx < g1.index[0]]):
            p4 = p + 4
            if p4 in gy.index and not np.isnan(lvl.loc[p4]):
                lvl.loc[p] = lvl.loc[p4] / (1 + gy.loc[p4] / 100)
        levels["CHN"] = lvl.dropna()
    return levels


def build_vintage(decision_date, focus_date, paths, neutral_rate=5.0, horizon=12, verbose=False, climate_index="oni",
                  nowcast="focus"):
    """Quarterly dataset through the current quarter q0, built from the information available at `decision_date`.

    `paths` (built by projection.py from the Focus survey of `focus_date` and the reference-scenario assumptions):
      selic_daily        daily Selic target (realized, then the Focus path), through the end of q0 + horizon
      ptax_daily         daily PTAX (realized through F, then the PPP path), through the end of q0 + horizon
      brent_monthly      monthly Brent (realized, then the futures-curve proxy), PeriodIndex 'M'
      adm_monthly_pct    administered-price inflation, monthly %, for the months after the IPCA cut-off
      focus_monthly_ipca Focus median of the monthly IPCA (%), PeriodIndex 'M', survey of F
      pie_F, ie_F        Focus 12-month inflation expectation and expected Selic over 12 months at F (% p.a.)
      oni_path           ONI anomaly for the quarters after the last published season (nowcast + projection)
      gdp_growth         Focus medians of GDP growth (% per year) by calendar year, survey of F
    Returns a dict with the quarterly frame `df` (columns as in data/processed/model_data.csv), the monthly price
    table, the quarterly seasonal factors, the paths of the climate terms over the projection horizon and
    auxiliary information.
    """
    cut = cutoffs(decision_date)
    q0, D, F = cut["q0"], cut["D"], pd.Timestamp(focus_date)
    m_end = q0.asfreq("M", "end")
    q_end = q0 + horizon
    lg = LG

    # ------------------------------------------------------------------ prices
    ipca = _monthly(fetch.sgs(SGS["ipca"], name="sgs_433_ipca"))
    liv = _monthly(fetch.sgs(SGS["ipca_livres"], name="sgs_11428_ipca_livres"))
    mon = _monthly(fetch.sgs(SGS["ipca_monit"], name="sgs_4449_ipca_monit"))
    w_m = implied_free_weight(ipca.loc[:cut["ipca"]], liv.loc[:cut["ipca"]], mon.loc[:cut["ipca"]])
    w_last = float(w_m.iloc[-1])
    mp = pd.DataFrame({"ipca": ipca, "liv": liv, "mon": mon}).loc[:cut["ipca"]]
    mp["source"] = "IBGE"
    for m in pd.period_range(cut["ipca"] + 1, m_end, freq="M"):
        if nowcast == "realized" and m in ipca.index and m in liv.index and m in mon.index:
            mp.loc[m] = [float(ipca[m]), float(liv[m]), float(mon[m]), "realized (published after the meeting)"]
            continue
        h = float(paths["focus_monthly_ipca"][m])
        a = float(paths["adm_monthly_pct"][m])
        mp.loc[m] = [h, (h - (1 - w_last) * a) / w_last, a, "nowcast (Focus headline, Copom administered)"]
    mp[["ipca", "liv", "mon"]] = mp[["ipca", "liv", "mon"]].astype(float)
    ts = lambda s: pd.Series(s.values, index=s.index.to_timestamp())  # noqa: E731
    q = pd.DataFrame({"pi_ipca": quarterly_log_inflation(ts(mp["ipca"])), "pi_livres": quarterly_log_inflation(ts(mp["liv"])),
                      "pi_monit": quarterly_log_inflation(ts(mp["mon"]))}).loc["1996Q1":q0]
    seas = {}
    for c in ["pi_ipca", "pi_livres", "pi_monit"]:
        q[c + "_sa"] = seasonal_adjust(q[c], method="stl")
        seas[c] = (q[c] - q[c + "_sa"])
    q["w_livres"] = to_quarterly_mean(ts(w_m)).reindex(q.index).ffill()

    # ------------------------------------------------------------------ Focus expectations
    f12 = fetch.focus_inflacao_12m("S")
    f12 = f12[(f12["baseCalculo"] == 0) & (f12["Data"] <= F)].set_index("Data")["Mediana"].astype(float).sort_index()
    pe = to_quarterly_mean(f12)
    pe.loc[q0] = float(paths.get("pie_F", f12.iloc[-1]))  # current quarter: the value at the Focus date (with any override or shock)
    ie_d = pd.read_csv(DATA_PROC / "focus_expected_selic_12m_daily.csv", index_col=0, parse_dates=True)["i_e_12m_meetings"].dropna()
    base = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
    base.index = pd.PeriodIndex(base.index, freq="Q")
    ie = base["i_e_12m"].copy()                           # spliced series (monthly-based before 2004Q4)
    ie_q = to_quarterly_mean(ie_d[ie_d.index <= F])
    ie.loc[ie_q.index.intersection(ie.index)] = ie_q.loc[ie_q.index.intersection(ie.index)]
    ie = ie.loc[:q0]
    ie.loc[q0] = float(paths["ie_F"])
    pe = pe.loc[:q0]

    # ------------------------------------------------------------------ policy rate and exchange rate
    selic_q = to_quarterly_mean(paths["selic_daily"]).loc[:q_end]
    ptax_q = to_quarterly_mean(paths["ptax_daily"]).loc[:q_end]

    # ------------------------------------------------------------------ activity
    pib = fetch.sidra(SIDRA_PIB_SA, "sidra_1621_pib_sa")
    pib["q"] = pd.PeriodIndex([_sidra_quarter(x) for x in pib["D3N"]], freq="Q")
    pib_sa = pib.set_index("q")["V"].astype(float).sort_index().loc[:cut["gdp"]]
    lp = 100 * np.log(pib_sa)
    gg = paths["gdp_growth"]
    g_last = gg[max(gg)]
    lp_ext = _extend_log(lp, N_EXT, lambda qq: 100 * np.log1p(gg.get(qq.year, g_last) / 100.0) / 4)
    fpib_all, trend_all = hp_cycle(lp_ext, HP_LAMBDA)
    fpib = fpib_all.loc[:cut["gdp"]]
    nuci = to_quarterly_mean(ts(_monthly(fetch.sgs(SGS["nuci_fgv_sa"], name="sgs_28561_nuci_fgv_sa"), cut["nuci"])))
    fnuci = nuci - nuci.loc["2003Q4":PRE_COVID_END].mean()
    pn = fetch.sgs(SGS["desocup_pnadc"], name="sgs_24369_desocup_pnadc")
    pn = pn[pn.index.to_period("M") <= cut["pnadc"]]
    pn_q = pn[pn.index.month.isin([3, 6, 9, 12])]
    pn_q = pn_q.groupby(pn_q.index.to_period("Q")).last()
    pme = to_quarterly_mean(fetch.sgs(SGS["desocup_pme"], name="sgs_10777_desocup_pme"))
    u_nsa = splice_by_differences(pn_q, pme)
    u_sa = seasonal_adjust(u_nsa.dropna(), method="stl")
    femp = -(u_sa - u_sa.loc["2003Q4":PRE_COVID_END].mean())
    caged = to_quarterly_mean(ts(_monthly(fetch.sgs(SGS["caged_estoque"], name="sgs_28763_caged_estoque"), cut["caged"])))
    lc = 100 * np.log(caged)
    lc_ext = _extend_log(lc, N_EXT, float(lc.diff().tail(4).mean()))
    fcaged = hp_cycle(lc_ext, HP_LAMBDA)[0].loc[:lc.index[-1]]

    # ------------------------------------------------------------------ fiscal
    prim = -fetch.sgs(SGS["prim_gov_central_rs"], name="sgs_4639_prim_gov_central")
    prim = prim.copy()
    for ym, amt in FISCAL_ONE_OFFS.items():
        d = pd.Timestamp(ym + "-01")
        if d in prim.index:
            prim.loc[d] = prim.loc[d] - amt
    prim = prim[prim.index.to_period("M") <= cut["fiscal"]]
    pib_nom_m = fetch.sgs(SGS["pib_nominal_mensal"], name="sgs_4380_pib_nominal_mensal")
    pib_nom_m = pib_nom_m[pib_nom_m.index.to_period("M") <= cut["fiscal"]]
    prim_q, pib_q = to_quarterly_sum(prim), to_quarterly_sum(pib_nom_m)
    try:
        ib = fetch.sidra(SIDRA_PIB_NOMINAL, "sidra_1846_pib_nominal")
        ib["q"] = pd.PeriodIndex([_sidra_quarter(x) for x in ib["D3N"]], freq="Q")
        pib_q = ib.set_index("q")["V"].astype(float).sort_index().loc[:cut["gdp"]].combine_first(pib_q)
    except Exception:  # noqa: BLE001
        pass
    fis = pd.DataFrame({"prim": prim_q, "pib": pib_q}).dropna()
    pb = 100 * fis["prim"].rolling(4).sum() / fis["pib"].rolling(4).sum()
    cab = (pb - FISCAL_SEMI_ELASTICITY * fpib.reindex(pb.index).rolling(4).mean()).dropna()
    rp_tilde = hp_cycle(_extend_log(cab, N_EXT, 0.0), HP_LAMBDA)[0].loc[:cab.index[-1]]

    # ------------------------------------------------------------------ world gap
    cyc = {}
    for c, s in world_levels().items():
        s = s.loc[:cut["oecd"]]
        if len(s) < 40:
            continue
        ls = 100 * np.log(s)
        ls_ext = _extend_log(ls, N_EXT, float(ls.diff().tail(8).mean()))
        cyc[c] = hp_cycle(ls_ext, HP_LAMBDA)[0].loc[:s.index[-1]]
    cyc_df = pd.DataFrame(cyc)
    w = pd.Series(WORLD_WEIGHTS)
    w = w[[c for c in w.index if c in cyc_df.columns]]
    avail = cyc_df[w.index].notna()
    wsum = (avail * w).sum(axis=1)
    h_star = ((cyc_df[w.index].fillna(0) * w).sum(axis=1) / wsum).where(wsum > 0.7 * w.sum())

    # ------------------------------------------------------------------ external
    ff = fetch.oecd_fed_funds()
    i_star = to_quarterly_mean(ff[ff.index <= D])
    embi = fetch.ipeadata(IPEA_EMBI)
    embi_q = to_quarterly_mean(embi[embi.index <= D])

    # ------------------------------------------------------------------ commodities (IC-Br), with the nowcast months
    cp = commodity_params(cut["icbr"])
    br = {k: _monthly(fetch.sgs(c, name="sgs_%d_%s" % (c, k)), cut["icbr"]) for k, c in
          [("icbr", 27574), ("icbr_agro", 27575), ("icbr_metal", 27576), ("icbr_energia", 27577)]}
    ptax_hist = _monthly(fetch.sgs(SGS["ptax_venda_mensal"], name="sgs_3698_ptax_venda_mensal"))
    pd_path = paths["ptax_daily"]
    ptax_path_m = pd_path.groupby(pd_path.index.to_period("M")).mean()
    ptax_m = ptax_hist.combine_first(ptax_path_m)
    mF = F.to_period("M")                       # from the month of the Focus date the path (realized days + PPP) applies
    ptax_m.loc[ptax_path_m.index[ptax_path_m.index >= mF]] = ptax_path_m.loc[ptax_path_m.index >= mF]
    brent_m = paths["brent_monthly"]
    last_m = br["icbr"].index[-1]
    com = pd.DataFrame({k: v for k, v in br.items()})
    for m in pd.period_range(last_m + 1, q_end.asfreq("M", "end"), freq="M"):
        fx_ratio = ptax_m[m] / ptax_m[last_m]
        oil_ratio = (brent_m[m] / brent_m[last_m]) ** cp["beta_oil"]
        com.loc[m, "icbr_agro"] = com.loc[last_m, "icbr_agro"] * fx_ratio
        com.loc[m, "icbr_metal"] = com.loc[last_m, "icbr_metal"] * fx_ratio
        com.loc[m, "icbr_energia"] = com.loc[last_m, "icbr_energia"] * fx_ratio * oil_ratio
        com.loc[m, "icbr"] = com.loc[last_m, "icbr"] * fx_ratio * oil_ratio ** cp["w_energia"]
    com_q = pd.DataFrame({k: to_quarterly_mean(ts(com[k])) for k in com.columns})

    # ------------------------------------------------------------------ climate
    oni_raw = fetch.noaa_roni() if climate_index == "roni" else fetch.noaa_oni()
    season_to_q = {"JFM": 1, "AMJ": 2, "JAS": 3, "OND": 4}
    o = oni_raw[oni_raw["SEAS"].isin(season_to_q)].copy()
    o["q"] = pd.PeriodIndex([pd.Period(year=int(y), quarter=season_to_q[s], freq="Q") for y, s in zip(o["YR"], o["SEAS"])], freq="Q")
    oni = o.set_index("q")["ANOM"].astype(float).sort_index().loc[:cut["oni"]]
    oni_full = pd.concat([oni, paths["oni_path"].loc[oni.index[-1] + 1:q_end]])
    clima2 = (10.0 * oni_full) ** 2
    el = (oni_full > 0).astype(float) * clima2
    la = (oni_full < 0).astype(float) * clima2
    clim_el = el.rolling(3).mean() - el.shift(3).rolling(3).mean()
    clim_la = la.rolling(3).mean() - la.shift(3).rolling(3).mean()

    # ------------------------------------------------------------------ assemble
    idx = pd.period_range("1999Q1", q0, freq="Q")
    df = pd.DataFrame(index=idx)
    df = df.join(q).join(pe.rename("pi_e_12m")).join(ie.rename("i_e_12m")).join(selic_q.rename("selic"))
    df["lpib_sa"] = lp
    df["fpib"] = fpib
    df["nuci"], df["fnuci"] = nuci, fnuci
    df["u_nsa"], df["u_sa"], df["femp"] = u_nsa, u_sa, femp
    df["lcaged"], df["fcaged"] = lc, fcaged
    df["rp_tilde"] = rp_tilde
    df["h_star"] = h_star
    df["i_star"], df["embi_bps"], df["ptax"] = i_star, embi_q, ptax_q
    for c in com_q.columns:
        df[c] = com_q[c]
    df["oni"] = oni_full
    df["clima2"], df["clim_el"], df["clim_la"] = clima2, clim_el, clim_la
    df["meta"] = pd.Series([target(p.year) for p in idx], index=idx)
    df["meta_12m"] = pd.Series([(1 - p.end_time.month / 12.0) * target(p.year) + p.end_time.month / 12.0 * target(p.year + 1) for p in idx], index=idx)
    df["i"], df["i_e"], df["pi_e"], df["pi_meta"] = lg(df["selic"]), lg(df["i_e_12m"]), lg(df["pi_e_12m"]), lg(df["meta_12m"])
    df["i_star_l"] = lg(df["i_star"])
    df["pi_e_q"] = df["pi_e"] / 4
    df["r_focus_raw"] = df["i_e_12m"] - df["pi_e_12m"]
    df["r_focus"] = df["i_e"] - df["pi_e"]
    if neutral_rate is None:
        rf = df["r_focus"].dropna()
        df["rr_trend"] = hp_cycle(_extend_log(rf, N_EXT, 0.0), HP_LAMBDA)[1].loc[:rf.index[-1]]
    else:
        est_end = pd.Period(SAMPLE_END, freq="Q")
        df["rr_trend"] = base["rr_trend"].reindex(idx)
        df.loc[df.index > est_end, "rr_trend"] = float(lg(neutral_rate))
    df["de"] = 100 * np.log(df["ptax"]).diff()
    df["de_hat"] = df["de"] - (df["meta"] - PI_STAR_SS) / 4
    for c in ["icbr", "icbr_agro", "icbr_metal", "icbr_energia"]:
        df["pi_" + c] = 100 * np.log(df[c]).diff()
        df["pi_" + c + "_hat"] = df["pi_" + c] - df["meta"] / 4
    df["cds_proxy"] = df["embi_bps"] / 100.0
    df["i_dif_raw"] = df["selic"] - (df["i_star"] + df["cds_proxy"])
    df["i_dif"] = df["i"] - lg(df["i_star"] + df["cds_proxy"])
    df["de_ppc"] = (df["meta"] - PI_STAR_SS) / 4
    per = lambda a, b: ((df.index >= pd.Period(a, "Q")) & (df.index <= pd.Period(b, "Q"))).astype(float)  # noqa: E731
    df["d_crise2008"], df["d_crise2020"], df["d_phillips_hv"] = per(*CRISIS_2008), per(*CRISIS_2020), per(*PHILLIPS_HIGH_VAR)
    df["pi_ipca_sa_ma4_l1"] = df["pi_ipca_sa"].shift(1).rolling(4).mean()
    df["pi_ipca_sum4_l1"] = df["pi_ipca"].shift(1).rolling(4).sum()

    # paths of the exogenous drivers over the projection quarters (for the conditional projection)
    proj_idx = pd.period_range(q0 + 1, q_end, freq="Q")
    meta_proj = pd.Series([target(p.year) for p in proj_idx], index=proj_idx)
    exog_paths = pd.DataFrame({
        "i": lg(selic_q.reindex(proj_idx)),
        "de": 100 * np.log(ptax_q).diff().reindex(proj_idx),
        "pistar_usd": (100 * np.log(com_q["icbr"]).diff() - meta_proj / 4 - 100 * np.log(ptax_q).diff()).reindex(proj_idx),
        "el": clim_el.reindex(proj_idx), "la": clim_la.reindex(proj_idx),
        "meta": lg(meta_proj), "meta_year": meta_proj,
    })
    if verbose:
        print("cut-offs:", {k: str(v) for k, v in cut.items() if k != "D"})
        print("nowcast months:", list(mp.index[mp["source"] != "IBGE"].astype(str)))
    return {"df": df, "monthly_prices": mp, "seasonal": pd.DataFrame(seas), "w_last": w_last, "cut": cut, "q0": q0,
            "exog_paths": exog_paths, "commodity_params": cp, "ptax_q": ptax_q, "selic_q": selic_q, "com_q": com_q}
