"""
Construction of the quarterly dataset of the BCB aggregate small-scale model (June-2024 specification).

Every series below is built from the raw downloads in data/raw (see fetch.py). Conventions:
* quarterly PeriodIndex; inflation rates in log-percent per quarter (100*dlog); interest rates in % p.a.;
* "deviation from target" terms use the calendar-year inflation target converted to a quarterly rate (target/4);
* the 12-month-ahead target used in the Taylor rule / expectations equation is the month-weighted average of the
  current-year and next-year targets.

Output: data/processed/model_data.csv (+ model_data_dictionary.csv) and intermediate monthly files.
"""
import json
import numpy as np
import pandas as pd

import fetch
from config import (SGS, SIDRA_PIB_SA, SIDRA_PIB_NOMINAL, IPEA_EMBI, DATA_PROC, DATA_RAW,
                    PI_STAR_SS, HP_LAMBDA, PRE_COVID_END, CRISIS_2008, CRISIS_2020, PHILLIPS_HIGH_VAR)
from tsutil import (to_quarterly_mean, to_quarterly_last, to_quarterly_sum, quarterly_log_inflation,
                    hp_cycle, seasonal_adjust, splice_by_differences)

# ---------------------------------------------------------------------------
# Calibrated inputs with documented sources
# ---------------------------------------------------------------------------
# Inflation targets (% p.a.) set by the CMN; 2003 and 2004 use the "adjusted targets" (8.5% and 5.5%) that the BCB
# adopted in its January-2003 open letter, as in BCB practice for that period. From 2025 the target is continuous (3%).
INFLATION_TARGET = {1999: 8.0, 2000: 6.0, 2001: 4.0, 2002: 3.5, 2003: 8.5, 2004: 5.5, 2005: 4.5, 2006: 4.5,
                    2007: 4.5, 2008: 4.5, 2009: 4.5, 2010: 4.5, 2011: 4.5, 2012: 4.5, 2013: 4.5, 2014: 4.5,
                    2015: 4.5, 2016: 4.5, 2017: 4.5, 2018: 4.5, 2019: 4.25, 2020: 4.0, 2021: 3.75, 2022: 3.5,
                    2023: 3.25, 2024: 3.0, 2025: 3.0, 2026: 3.0, 2027: 3.0, 2028: 3.0}

# Export-share weights for the "relevant world GDP" (approximate 2010-2023 average shares of Brazilian exports,
# MDIC/Comex Stat, renormalised over the countries with OECD quarterly GDP). Updated weights could be pulled from
# Comex Stat; the world gap carries a small coefficient (beta4 ~ 0.05) so the approximation is innocuous.
WORLD_WEIGHTS = {"CHN": 0.28, "USA": 0.13, "ARG": 0.06, "NLD": 0.05, "JPN": 0.025, "CHL": 0.025, "MEX": 0.025,
                 "DEU": 0.025, "ESP": 0.02, "IND": 0.02, "KOR": 0.02, "CAN": 0.015, "ITA": 0.015, "GBR": 0.015,
                 "FRA": 0.01, "BEL": 0.01}

# One-off primary-revenue/expenditure items of the central government removed before building the fiscal variable
# (BRL million, month, sign = effect on the primary surplus). Sources: BCB fiscal press notes / Treasury reports.
FISCAL_ONE_OFFS = {
    "2010-09": +74_808,   # Petrobras capitalisation: onerous assignment of oil rights (cessão onerosa) revenue
    "2016-10": +46_815,   # repatriation (RERCT) revenue, first round
    "2016-11": +1_000,    # repatriation, residual
    "2017-08": +1_600,    # repatriation, second round
    "2019-12": +69_960,   # cessão onerosa surplus auction (gross revenue; part transferred to states/municipalities)
    "2023-12": -92_400,   # payment of accumulated court-ordered debts (precatórios) after the 2023 constitutional change
}
# Semi-elasticity of the central-government primary balance (% GDP) to the output gap (pp): revenue share ~0.18-0.20
# of GDP with an elasticity of ~1.1-1.2 to the cycle (SPE "Resultado Fiscal Estrutural" methodology) -> ~0.22.
FISCAL_SEMI_ELASTICITY = 0.22


def _target_series(idx):
    """Calendar-year target (% p.a.) on a quarterly PeriodIndex."""
    return pd.Series([INFLATION_TARGET[p.year] for p in idx], index=idx)


def _target_12m_ahead(idx):
    """Target relevant for the 12 months after quarter t: month-weighted average of this and next year's targets."""
    out = []
    for p in idx:
        m_end = p.end_time.month  # 3, 6, 9, 12
        w_next = m_end / 12.0     # share of the next 12 months falling in the next calendar year
        out.append((1 - w_next) * INFLATION_TARGET[p.year] + w_next * INFLATION_TARGET[p.year + 1])
    return pd.Series(out, index=idx)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def build_prices(sa_method="stl"):
    ipca = fetch.sgs(SGS["ipca"], name="sgs_433_ipca")
    liv = fetch.sgs(SGS["ipca_livres"], name="sgs_11428_ipca_livres")
    mon = fetch.sgs(SGS["ipca_monit"], name="sgs_4449_ipca_monit")
    q = pd.DataFrame({
        "pi_ipca": quarterly_log_inflation(ipca),
        "pi_livres": quarterly_log_inflation(liv),
        "pi_monit": quarterly_log_inflation(mon),
    })
    q = q.loc["1996Q1":]
    for c in ["pi_ipca", "pi_livres", "pi_monit"]:
        q[c + "_sa"] = seasonal_adjust(q[c], method=sa_method)
    # implicit weight of free prices in the IPCA from the identity pi = w*pi_L + (1-w)*pi_A (monthly, robust)
    m = pd.DataFrame({"ipca": ipca, "liv": liv, "mon": mon}).dropna()
    d = m["liv"] - m["mon"]
    w = ((m["ipca"] - m["mon"]) / d).where(d.abs() > 0.4)
    w = w.clip(0.55, 0.9).rolling(13, center=True, min_periods=4).median().interpolate().bfill().ffill()
    q["w_livres"] = to_quarterly_mean(w).reindex(q.index).ffill()
    return q


# ---------------------------------------------------------------------------
# Focus expectations
# ---------------------------------------------------------------------------
def build_focus_inflation(suavizada="S"):
    f = fetch.focus_inflacao_12m(suavizada)
    f = f[f["baseCalculo"] == 0]
    s = f.set_index("Data")["Mediana"].astype(float).sort_index()
    return to_quarterly_mean(s).rename("pi_e_12m")


def _meeting_dates():
    """Map Focus meeting labels 'Rk/yyyy' to actual Copom dates (regular meetings only); future meetings get the
    usual calendar month of the k-th meeting of the year."""
    cop = fetch.copom_meetings()
    cop = cop[cop["ReuniaoExtraordinaria"] != True].copy()  # noqa: E712
    cop["year"] = cop["DataReuniaoCopom"].dt.year
    cop["k"] = cop.groupby("year")["DataReuniaoCopom"].rank(method="first").astype(int)
    known = {"R%d/%d" % (k, y): d for k, y, d in zip(cop["k"], cop["year"], cop["DataReuniaoCopom"])}
    typical_month = {1: 1, 2: 3, 3: 5, 4: 6, 5: 7, 6: 9, 7: 11, 8: 12}  # 8 meetings a year since 2006

    def lookup(label):
        if label in known:
            return known[label]
        k, y = label.split("/")
        k, y = int(k[1:]), int(y)
        if y <= 2005:  # monthly meetings until 2005
            return pd.Timestamp(year=y, month=min(k, 12), day=18)
        return pd.Timestamp(year=y, month=typical_month.get(k, 12), day=20)
    return lookup


def expected_selic_12m_from_meetings(selic_daily):
    """Expected average Selic over the 12 months after each survey date, from Focus per-meeting expectations.

    The expected path is a step function: the current target until the first future meeting (decision effective the
    next business day), then the expected target after each subsequent meeting.
    """
    f = fetch.focus_selic_reunioes()
    f = f[f["baseCalculo"] == 0].copy()
    lookup = _meeting_dates()
    f["mdate"] = f["Reuniao"].map(lookup)
    f = f.dropna(subset=["mdate"])
    out = {}
    sel = selic_daily.sort_index()
    for d, g in f.groupby("Data"):
        g = g[g["mdate"] >= d].sort_values("mdate")
        if g.empty:
            continue
        cur = sel.asof(d)
        horizon = pd.date_range(d + pd.Timedelta(days=1), d + pd.Timedelta(days=365), freq="D")
        path = pd.Series(cur, index=horizon)
        for md, val in zip(g["mdate"], g["Mediana"].astype(float)):
            path.loc[path.index > md] = val
        out[d] = path.mean()
    s = pd.Series(out).sort_index()
    s.index = pd.to_datetime(s.index)
    return s.rename("i_e_12m")


def expected_selic_12m_from_monthly(selic_daily):
    """Expected average Selic over the next 12 months from Focus end-of-month Selic expectations (2001-2005).

    For survey date d, the expected end-of-month rates for the 12 months starting at the survey month are averaged;
    missing horizons are filled with the year-end (annual) expectation when available.
    """
    f = fetch.focus_mensais("Selic")
    f = f[f["baseCalculo"] == 0].copy()
    f["ref"] = pd.to_datetime(f["DataReferencia"], format="%m/%Y")
    a = fetch.focus_anuais("Selic")
    a = a[a["baseCalculo"] == 0].copy()
    a["ref"] = pd.to_datetime(a["DataReferencia"].astype(str) + "-12-01")
    out = {}
    for d, g in f.groupby("Data"):
        months = pd.period_range(pd.Timestamp(d).to_period("M"), periods=12, freq="M").to_timestamp()
        exp = g.set_index("ref")["Mediana"].astype(float)
        ann = a[a["Data"] == d].set_index("ref")["Mediana"].astype(float)
        path = pd.Series(np.nan, index=months)
        for m in months:
            if m in exp.index:
                path.loc[m] = exp.loc[m]
            elif m in ann.index:
                path.loc[m] = ann.loc[m]
        if path.notna().sum() < 6:
            continue
        path = path.interpolate().ffill().bfill()
        out[d] = path.mean()
    s = pd.Series(out).sort_index()
    s.index = pd.to_datetime(s.index)
    return s.rename("i_e_12m_monthly")


def build_focus_selic():
    selic_daily = fetch.sgs(SGS["selic_meta_daily"], name="sgs_432_selic_meta_daily", daily=True)
    a = expected_selic_12m_from_meetings(selic_daily)
    b = expected_selic_12m_from_monthly(selic_daily)
    qa, qb = to_quarterly_mean(a), to_quarterly_mean(b)
    # splice: meetings-based series where available (from 2004Q4), monthly-based before
    overlap = qa.index.intersection(qb.index)
    diff = float((qa.loc[overlap] - qb.loc[overlap]).mean()) if len(overlap) else 0.0
    q = qa.reindex(qa.index.union(qb.index))
    q.loc[q.isna()] = (qb + diff).reindex(q.index).loc[q.isna()]
    daily = pd.DataFrame({"i_e_12m_meetings": a, "i_e_12m_monthly": b})
    daily.to_csv(DATA_PROC / "focus_expected_selic_12m_daily.csv")
    return q.rename("i_e_12m"), diff


# ---------------------------------------------------------------------------
# Activity indicators for the output-gap block
# ---------------------------------------------------------------------------
def build_activity(lamb=HP_LAMBDA, pre_covid_end=PRE_COVID_END, start="2003Q4"):
    # GDP: IBGE chained volume index, seasonally adjusted (SIDRA 1621)
    pib = fetch.sidra(SIDRA_PIB_SA, "sidra_1621_pib_sa")
    pib["q"] = pd.PeriodIndex([_sidra_quarter(x) for x in pib["D3N"]], freq="Q")
    pib_sa = pib.set_index("q")["V"].astype(float).sort_index()
    lpib = 100 * np.log(pib_sa)
    fpib, trend = hp_cycle(lpib, lamb)
    # NUCI (FGV, SA, monthly) -> quarterly mean -> deviation from pre-Covid mean
    nuci = to_quarterly_mean(fetch.sgs(SGS["nuci_fgv_sa"], name="sgs_28561_nuci_fgv_sa"))
    fnuci = nuci - nuci.loc[start:pre_covid_end].mean()
    # Unemployment: PNADC (monthly rolling quarter; value at the quarter's last month = calendar quarter) spliced
    # backwards with PME changes; seasonally adjusted; sign flipped so that + = tighter labour market
    pnadc = fetch.sgs(SGS["desocup_pnadc"], name="sgs_24369_desocup_pnadc")
    pnadc_q = pnadc[pnadc.index.month.isin([3, 6, 9, 12])]
    pnadc_q = pnadc_q.groupby(pnadc_q.index.to_period("Q")).last()
    pme = to_quarterly_mean(fetch.sgs(SGS["desocup_pme"], name="sgs_10777_desocup_pme"))
    u_nsa = splice_by_differences(pnadc_q, pme)
    u_sa = seasonal_adjust(u_nsa.dropna(), method="stl")
    femp = -(u_sa - u_sa.loc[start:pre_covid_end].mean())
    # Formal employment stock (Caged), quarterly mean of monthly stock -> HP cycle of 100*log
    caged = to_quarterly_mean(fetch.sgs(SGS["caged_estoque"], name="sgs_28763_caged_estoque"))
    fcaged, _ = hp_cycle(100 * np.log(caged), lamb)
    act = pd.DataFrame({"lpib_sa": lpib, "fpib": fpib, "pib_trend": trend, "nuci": nuci, "fnuci": fnuci,
                        "u_nsa": u_nsa, "u_sa": u_sa, "femp": femp, "lcaged": 100 * np.log(caged), "fcaged": fcaged})
    return act


def _sidra_quarter(label):
    # e.g. '1º trimestre 1996'
    k = int(label[0])
    y = int(label[-4:])
    return pd.Period(year=y, quarter=k, freq="Q")


# ---------------------------------------------------------------------------
# Fiscal variable
# ---------------------------------------------------------------------------
def build_fiscal(gap_for_adjustment, lamb=HP_LAMBDA, semi_elasticity=FISCAL_SEMI_ELASTICITY):
    """Central-government primary surplus, 4-quarter sum in % of GDP, net of one-offs, cyclically adjusted,
    expressed as deviation from its HP trend (the model's rp-tilde)."""
    prim = -fetch.sgs(SGS["prim_gov_central_rs"], name="sgs_4639_prim_gov_central")  # + = surplus
    prim = prim.copy()
    for ym, amt in FISCAL_ONE_OFFS.items():
        d = pd.Timestamp(ym + "-01")
        if d in prim.index:
            prim.loc[d] = prim.loc[d] - amt
    pib_nom_m = fetch.sgs(SGS["pib_nominal_mensal"], name="sgs_4380_pib_nominal_mensal")
    prim_q = to_quarterly_sum(prim)
    pib_q = to_quarterly_sum(pib_nom_m)
    # IBGE quarterly nominal GDP where available (SIDRA 1846), BCB monthly estimate otherwise
    try:
        ib = fetch.sidra(SIDRA_PIB_NOMINAL, "sidra_1846_pib_nominal")
        ib["q"] = pd.PeriodIndex([_sidra_quarter(x) for x in ib["D3N"]], freq="Q")
        pib_ibge = ib.set_index("q")["V"].astype(float).sort_index()
        pib_q = pib_ibge.combine_first(pib_q)
    except Exception:  # noqa: BLE001
        pass
    df = pd.DataFrame({"prim": prim_q, "pib": pib_q}).dropna()
    prim4 = df["prim"].rolling(4).sum()
    pib4 = df["pib"].rolling(4).sum()
    pb = 100 * prim4 / pib4                       # primary surplus, 4-quarter flow, % GDP
    gap4 = gap_for_adjustment.reindex(pb.index).rolling(4).mean()
    cab = pb - semi_elasticity * gap4             # cyclically adjusted primary balance (% GDP)
    cab = cab.dropna()
    cyc, trend = hp_cycle(cab, lamb)
    out = pd.DataFrame({"prim_surplus_pct_gdp_4q": pb, "cab_pct_gdp": cab, "cab_trend": trend, "rp_tilde": cyc})
    return out


# ---------------------------------------------------------------------------
# External block
# ---------------------------------------------------------------------------
def build_world_gap(lamb=HP_LAMBDA):
    oecd = fetch.oecd_qna_gdp(list(WORLD_WEIGHTS))
    sub = oecd[(oecd["ADJUSTMENT"] == "Y") & (oecd["PRICE_BASE"] == "LR") & (oecd["UNIT_MEASURE"] == "IX")
               & (oecd["TRANSFORMATION"] == "N") & (oecd["TABLE_IDENTIFIER"] == "T0102")]
    levels = {}
    for c, g in sub.groupby("REF_AREA"):
        s = g.set_index(pd.PeriodIndex(g["TIME_PERIOD"], freq="Q"))["OBS_VALUE"].astype(float).sort_index()
        levels[c] = s
    # China: seasonally adjusted index built from q/q growth (from 2011) and y/y growth (backwards)
    chn = oecd[(oecd["REF_AREA"] == "CHN") & (oecd["ADJUSTMENT"] == "Y") & (oecd["UNIT_MEASURE"] == "PC")]
    g1 = chn[chn["TRANSFORMATION"] == "G1"].set_index(pd.PeriodIndex(chn[chn["TRANSFORMATION"] == "G1"]["TIME_PERIOD"], freq="Q"))["OBS_VALUE"].astype(float).sort_index()
    gy = chn[chn["TRANSFORMATION"] == "GY"].set_index(pd.PeriodIndex(chn[chn["TRANSFORMATION"] == "GY"]["TIME_PERIOD"], freq="Q"))["OBS_VALUE"].astype(float).sort_index()
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
    cycles = {}
    for c, s in levels.items():
        cyc, _ = hp_cycle(100 * np.log(s), lamb)
        cycles[c] = cyc
    cyc_df = pd.DataFrame(cycles)
    w = pd.Series(WORLD_WEIGHTS)
    w = w[[c for c in w.index if c in cyc_df.columns]]
    avail = cyc_df[w.index].notna()
    wsum = (avail * w).sum(axis=1)
    hstar = (cyc_df[w.index].fillna(0) * w).sum(axis=1) / wsum
    hstar = hstar.where(wsum > 0.7 * w.sum())  # require most of the weight to be present
    cyc_df["h_star"] = hstar
    return cyc_df


def build_external():
    ff = to_quarterly_mean(fetch.oecd_fed_funds()).rename("i_star")
    embi = to_quarterly_mean(fetch.ipeadata(IPEA_EMBI)).rename("embi_bps")
    ptax = to_quarterly_mean(fetch.sgs(SGS["ptax_venda_mensal"], name="sgs_3698_ptax_venda_mensal")).rename("ptax")
    return pd.DataFrame({"i_star": ff, "embi_bps": embi, "ptax": ptax})


def build_commodities():
    cols = {"icbr": 27574, "icbr_agro": 27575, "icbr_metal": 27576, "icbr_energia": 27577}
    df = pd.DataFrame({k: to_quarterly_mean(fetch.sgs(v, name="sgs_%d_%s" % (v, k))) for k, v in cols.items()})
    return df


def build_climate():
    oni = fetch.noaa_oni()
    season_to_q = {"JFM": 1, "AMJ": 2, "JAS": 3, "OND": 4}
    o = oni[oni["SEAS"].isin(season_to_q)].copy()
    o["q"] = pd.PeriodIndex([pd.Period(year=int(y), quarter=season_to_q[s], freq="Q") for y, s in zip(o["YR"], o["SEAS"])], freq="Q")
    s = o.set_index("q")["ANOM"].astype(float).sort_index()
    # Scale: the box's prior support for the climate coefficients, U[0, 0.01], and the published modes (0.0012 and
    # 0.0007) are only compatible with sizeable climate contributions (the 2017 box reports up to ~1.5 pp of
    # 4-quarter free-price inflation in the 2015-16 El Nino) if Clima^2 is of order 10^2, i.e. if the ONI anomaly is
    # measured in tenths of a degree Celsius. Unrestricted least squares on the degree-Celsius scale indeed returns
    # coefficients ~100 times the BCB values, so the anomaly is expressed in tenths of a degree here (ONI_10 = 10*ONI).
    clima2 = (10.0 * s) ** 2
    d_el = (s > 0).astype(float)
    d_la = (s < 0).astype(float)
    el = d_el * clima2
    la = d_la * clima2
    # Phillips-curve terms: mean of t..t-2 minus mean of t-3..t-5 (transitory supply shock with reversal)
    el_term = el.rolling(3).mean() - el.shift(3).rolling(3).mean()
    la_term = la.rolling(3).mean() - la.shift(3).rolling(3).mean()
    return pd.DataFrame({"oni": s, "clima2": clima2, "d_el": d_el, "d_la": d_la, "clim_el": el_term, "clim_la": la_term})


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def assemble(sa_method="stl", save=True):
    prices = build_prices(sa_method)
    pe = build_focus_inflation("S")
    pe_n = build_focus_inflation("N").rename("pi_e_12m_raw")
    ie, splice_diff = build_focus_selic()
    act = build_activity()
    fis = build_fiscal(act["fpib"])
    world = build_world_gap()
    ext = build_external()
    com = build_commodities()
    clim = build_climate()
    selic_daily = fetch.sgs(SGS["selic_meta_daily"], name="sgs_432_selic_meta_daily", daily=True)
    selic_q = to_quarterly_mean(selic_daily).rename("selic")

    idx = pd.period_range("1999Q1", max(prices.index.max(), act.index.max()), freq="Q")
    df = pd.DataFrame(index=idx)
    df = df.join(prices).join(pe).join(pe_n).join(ie).join(selic_q)
    df = df.join(act).join(fis).join(world[["h_star"]]).join(ext).join(com).join(clim)

    # targets
    df["meta"] = _target_series(idx)
    df["meta_12m"] = _target_12m_ahead(idx)
    # Annual rates in log form, 100*log(1+x/100): the box states that variables are "generally in logarithm";
    # with log rates the ex-ante real rate i_e - pi_e is the (log) Fisher real rate. Raw versions are kept.
    lg = lambda x: 100 * np.log1p(x / 100.0)  # noqa: E731
    df["i"] = lg(df["selic"])                  # Selic target, log % p.a.
    df["i_e"] = lg(df["i_e_12m"])              # expected Selic over the next 12 months, log % p.a.
    df["pi_e"] = lg(df["pi_e_12m"])            # expected inflation next 12 months, log % p.a.
    df["pi_meta"] = lg(df["meta_12m"])         # 12-month-ahead target, log % p.a.
    df["i_star_l"] = lg(df["i_star"])
    # expectations in quarterly log-rate for the Phillips curve
    df["pi_e_q"] = df["pi_e"] / 4
    # ex-ante real interest rate (Focus), % p.a., and its HP trend (rr_trend) over the available sample
    df["r_focus_raw"] = df["i_e_12m"] - df["pi_e_12m"]
    df["r_focus"] = df["i_e"] - df["pi_e"]
    cyc, tr = hp_cycle(df["r_focus"].dropna(), HP_LAMBDA)
    df["rr_trend"] = tr
    # exchange rate: quarterly log change and deviation from PPP-implied drift
    df["de"] = 100 * np.log(df["ptax"]).diff()
    df["de_hat"] = df["de"] - (df["meta"] - PI_STAR_SS) / 4
    # imported commodity inflation (IC-Br in BRL) in deviation from the quarterly target
    for c in ["icbr", "icbr_agro", "icbr_metal", "icbr_energia"]:
        df["pi_" + c] = 100 * np.log(df[c]).diff()
        df["pi_" + c + "_hat"] = df["pi_" + c] - df["meta"] / 4
    # UIP inputs
    df["cds_proxy"] = df["embi_bps"] / 100.0           # EMBI+ in % p.a. (CDS 5y is not public; EMBI+ is the proxy)
    df["i_dif_raw"] = df["selic"] - (df["i_star"] + df["cds_proxy"])
    df["i_dif"] = df["i"] - lg(df["i_star"] + df["cds_proxy"])   # log-rate version used in the model
    df["de_ppc"] = (df["meta"] - PI_STAR_SS) / 4
    # dummies
    df["d_crise2008"] = ((df.index >= pd.Period(CRISIS_2008[0], "Q")) & (df.index <= pd.Period(CRISIS_2008[1], "Q"))).astype(float)
    df["d_crise2020"] = ((df.index >= pd.Period(CRISIS_2020[0], "Q")) & (df.index <= pd.Period(CRISIS_2020[1], "Q"))).astype(float)
    df["d_phillips_hv"] = ((df.index >= pd.Period(PHILLIPS_HIGH_VAR[0], "Q")) & (df.index <= pd.Period(PHILLIPS_HIGH_VAR[1], "Q"))).astype(float)
    # 4-quarter moving average of SA IPCA inflation (lagged in the Phillips curve) and 4-quarter sum (expectations eq.)
    df["pi_ipca_sa_ma4_l1"] = df["pi_ipca_sa"].shift(1).rolling(4).mean()
    df["pi_ipca_sum4_l1"] = df["pi_ipca"].shift(1).rolling(4).sum()

    if save:
        df.to_csv(DATA_PROC / "model_data.csv")
        world.to_csv(DATA_PROC / "world_gdp_cycles.csv")
        meta = {"splice_diff_focus_selic_meetings_minus_monthly": splice_diff, "sa_method": sa_method,
                "fiscal_semi_elasticity": FISCAL_SEMI_ELASTICITY, "fiscal_one_offs": FISCAL_ONE_OFFS,
                "world_weights": WORLD_WEIGHTS, "inflation_targets": INFLATION_TARGET}
        (DATA_PROC / "model_data_build_info.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return df


DATA_DICTIONARY = {
    "pi_ipca": "IPCA quarterly inflation, log-% (100*sum of monthly log changes)",
    "pi_livres": "IPCA free prices, quarterly inflation, log-%",
    "pi_monit": "IPCA administered prices, quarterly inflation, log-%",
    "pi_ipca_sa / pi_livres_sa / pi_monit_sa": "seasonally adjusted versions (STL, robust, period 4)",
    "w_livres": "implicit weight of free prices in the IPCA (from the aggregation identity, rolling median)",
    "pi_e_12m": "Focus median expectation of IPCA over the next 12 months, smoothed (% p.a.), quarterly mean of daily data",
    "pi_e_12m_raw": "same, non-smoothed",
    "pi_e_q": "pi_e_12m converted to a quarterly log rate: 100*log(1+pi_e/100)/4",
    "i_e_12m": "Focus-implied expected average Selic over the next 12 months (% p.a.), from per-meeting expectations (monthly expectations before Nov/2004)",
    "selic": "Selic target, quarterly average of daily values (% p.a.)",
    "r_focus": "ex-ante real rate i_e_12m - pi_e_12m (% p.a.)",
    "rr_trend": "HP(1600) trend of r_focus (the model's rr_trend)",
    "fpib": "HP(1600) cycle of 100*log(GDP SA, IBGE chained index)",
    "fnuci": "FGV capacity utilisation (SA) minus its 2003Q4-2019Q4 mean (pp)",
    "femp": "-(unemployment rate SA - 2003Q4-2019Q4 mean); PNADC spliced backwards with PME changes",
    "fcaged": "HP(1600) cycle of 100*log(stock of formal jobs, Caged)",
    "rp_tilde": "cyclically-adjusted central-government primary surplus (4-quarter, % GDP, net of one-offs) minus HP trend",
    "h_star": "trade-weighted HP cycle of partners' real GDP (OECD QNA), %",
    "i_star": "Fed funds effective rate (OECD MEI), quarterly mean, % p.a.",
    "embi_bps / cds_proxy": "EMBI+ Brazil (bps; /100 in % p.a.) as the risk-premium proxy",
    "ptax / de / de_hat": "BRL/USD quarterly mean; 100*dlog; deviation from PPP drift (meta-2)/4",
    "pi_icbr_hat (and components)": "100*dlog(IC-Br) minus meta/4",
    "oni / clima2 / clim_el / clim_la": "ONI anomaly (quarter season); squared; Phillips-curve climate terms (3-quarter mean minus previous 3-quarter mean, split by sign)",
    "meta / meta_12m": "calendar-year inflation target; month-weighted target for the next 12 months",
    "d_crise2008 / d_crise2020 / d_phillips_hv": "high-variance shock periods used in estimation",
}


if __name__ == "__main__":
    d = assemble()
    print(d.tail())
