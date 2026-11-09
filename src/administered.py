"""
The BCB's medium-term model of administered prices (Monetary Policy Report, June 2025, box "Atualização do modelo
para projeção de médio prazo dos preços administrados"): 24 quarterly item equations, calibrated by the BCB from the
institutional rules of each item, that are combined with the small-scale models through the IPCA feedback.

Units: quarterly log-% (100 x sum of monthly log changes), as in the rest of the project; the box writes the
equations in percentage variations, which is equivalent to first order. Every equation is written as
pi_item_t = theta' z_t, where z_t is a vector of regressors built from the drivers (headline IPCA, Brent in dollars,
exchange rate, US CPI, the tariff flag, the annual health-plan and pharmaceutical readjustments) and theta is the
BCB's calibration (`BCB_THETA`) or an OLS estimate on the same regressors (`estimate`). This makes the comparison
with the BCB's coefficients direct and lets the projections use either set.

Items (IBGE subitem code, weight in the IPCA of May 2025 as in the box):
 gasolina 5104001 (5.24), plano de saúde 6203 (4.06), energia elétrica residencial 2202003 (3.78), produtos
 farmacêuticos 6101 (3.46), emplacamento e licença 5102004 (2.69), taxa de água e esgoto 2101004 (1.84), gás de
 botijão 2201004 (1.25), ônibus urbano 5101001 (1.12), jogos de azar 7201063 (0.44), ônibus intermunicipal 5101006
 (0.41), óleo diesel 5104003 (0.25), plano de telefonia fixa 9101002 (0.22), táxi 5101002 (0.20), gás encanado
 2201005 (0.15), ônibus interestadual 5101007 (0.11), multa 5102006 (0.088), pedágio 5102015 (0.087), gás veicular
 5104005 (0.070), metrô 5101011 (0.066), correio 9101001 (0.065), integração transporte público 5101053 (0.052),
 conselho de classe 7101090 (0.048), trem 5101004 (0.038), cartório 7101034 (0.023).
"""
import numpy as np
import pandas as pd

import fetch
import fetch_adm
from config import SGS, DATA_PROC, OUTPUT
from tsutil import to_quarterly_mean, quarterly_log_inflation

# ---------------------------------------------------------------------------
# Items and the BCB's calibration
# ---------------------------------------------------------------------------
AA = 0.086 * 0.6      # alpha_Itaipu * alpha_energia (electricity)

ITEMS = {
    "5104001": {"name": "Gasolina", "type": "fuel", "w": 5.24, "lags": 3,
                "theta": [0.48 * 0.4, 0.48 * 0.4, 0.48 * 0.2, 0.48 * 0.0, 0.24, 0.17],
                "labels": ["oil_t", "oil_t-1", "oil_t-2", "oil_t-3", "icms", "margin"], "margin": "gasolina"},
    "5104003": {"name": "Óleo diesel", "type": "fuel", "w": 0.25, "lags": 3,
                "theta": [0.593 * 0.3, 0.593 * 0.4, 0.593 * 0.3, 0.593 * 0.0, 0.173, 0.184],
                "labels": ["oil_t", "oil_t-1", "oil_t-2", "oil_t-3", "icms", "margin"], "margin": "diesel"},
    "2201004": {"name": "Gás de botijão", "type": "fuel", "w": 1.25, "lags": 5,
                "theta": [0.32 * 0.0, 0.32 * 0.2, 0.32 * 0.2, 0.32 * 0.2, 0.32 * 0.2, 0.32 * 0.2, 0.17, 0.51],
                "labels": ["oil_t", "oil_t-1", "oil_t-2", "oil_t-3", "oil_t-4", "oil_t-5", "icms", "margin"], "margin": "botijao"},
    "2201005": {"name": "Gás encanado", "type": "gas_encanado", "w": 0.15, "theta": [0.55, 0.25, 0.05], "labels": ["ipca4/4", "botijao_t", "oil_sum4"]},
    "5104005": {"name": "Gás veicular", "type": "gas_veicular", "w": 0.070, "theta": [0.5, 0.1, 0.2, 0.1, 0.05, 0.05],
                "labels": ["ipca4", "gasolina_t-1", "gasolina_t-2", "oil_t-1", "oil_t-2", "oil_t-3"]},
    "2202003": {"name": "Energia elétrica residencial", "type": "energia", "w": 3.78,
                "theta": [(1 - AA) * 0.08, (1 - AA) * 0.40, (1 - AA) * 0.42, (1 - AA) * 0.10, AA * 0.10, AA * 0.35, AA * 0.42, AA * 0.13],
                "labels": ["q1*ipca4", "q2*ipca4", "q3*ipca4", "q4*ipca4", "q1*itaipu", "q2*itaipu", "q3*itaipu", "q4*itaipu"]},
    "6101": {"name": "Produtos farmacêuticos", "type": "farmacia", "w": 3.46, "theta": [0.1, 0.8, 0.05, 0.05],
             "labels": ["q1*reaj_t-3", "q2*reaj_t", "q3*reaj_t-1", "q4*reaj_t-2"], "fatorY": {"cambio": 0.15, "energia": 0.07}},
    "6203": {"name": "Plano de saúde", "type": "plano", "w": 4.06, "theta": [1.0, 1.0], "labels": ["q1*R_{A-1}/4", "q2-4*R_A/4"], "ivda_a": 0.6},
    "5102004": {"name": "Emplacamento e licença", "type": "emplacamento", "w": 2.69, "theta": [1.0, 1.0], "labels": ["q1*ipca4/4", "q2-4*pi_t-1"]},
    "7101090": {"name": "Conselho de classe", "type": "emplacamento", "w": 0.048, "theta": [1.0, 1.0], "labels": ["q1*ipca4/4", "q2-4*pi_t-1"]},
    "2101004": {"name": "Taxa de água e esgoto", "type": "seasonal", "w": 1.84, "theta": [0.19, 0.50, 0.09, 0.22], "labels": ["q1", "q2", "q3", "q4"]},
    "5101001": {"name": "Ônibus urbano", "type": "seasonal", "w": 1.12, "theta": [0.75, 0.20, 0.05, 0.00], "labels": ["q1", "q2", "q3", "q4"]},
    "5101006": {"name": "Ônibus intermunicipal", "type": "seasonal", "w": 0.41, "theta": [0.55, 0.14, 0.18, 0.13], "labels": ["q1", "q2", "q3", "q4"]},
    "5101002": {"name": "Táxi", "type": "seasonal", "w": 0.20, "theta": [0.48, 0.31, 0.04, 0.17], "labels": ["q1", "q2", "q3", "q4"]},
    "5101007": {"name": "Ônibus interestadual", "type": "seasonal", "w": 0.11, "theta": [0.17, 0.02, 0.06, 0.75], "labels": ["q1", "q2", "q3", "q4"]},
    "5102015": {"name": "Pedágio", "type": "seasonal", "w": 0.087, "theta": [0.22, 0.04, 0.42, 0.32], "labels": ["q1", "q2", "q3", "q4"]},
    "5101011": {"name": "Metrô", "type": "seasonal", "w": 0.066, "theta": [0.38, 0.58, 0.04, 0.00], "labels": ["q1", "q2", "q3", "q4"]},
    "9101001": {"name": "Correio", "type": "seasonal", "w": 0.065, "theta": [0.17, 0.83, 0.00, 0.00], "labels": ["q1", "q2", "q3", "q4"]},
    "7201063": {"name": "Jogos de azar", "type": "nonseasonal", "w": 0.44, "theta": [1.0], "labels": ["ipca4/4"]},
    "9101002": {"name": "Plano de telefonia fixa", "type": "nonseasonal", "w": 0.22, "theta": [1.0], "labels": ["ipca4/4"]},
    "5102006": {"name": "Multa", "type": "nonseasonal", "w": 0.088, "theta": [1.0], "labels": ["ipca4/4"]},
    "5101053": {"name": "Integração transporte público", "type": "q1", "w": 0.052, "theta": [1.0], "labels": ["q1*ipca4"]},
    "5101004": {"name": "Trem", "type": "q1", "w": 0.038, "theta": [1.0], "labels": ["q1*ipca4"]},
    "7101034": {"name": "Cartório", "type": "q1", "w": 0.023, "theta": [1.0], "labels": ["q1*ipca4"]},
}
ORDER = ["5104001", "5104003", "2201004", "2201005", "5104005", "2202003", "6101", "6203", "5102004", "7101090", "2101004", "5101001",
         "5101006", "5101002", "5101007", "5102015", "5101011", "9101001", "7201063", "9101002", "5102006", "5101053", "5101004", "7101034"]
BCB_THETA = {k: np.array(v["theta"], dtype=float) for k, v in ITEMS.items()}
C_REP_FARM = 0.8          # share of the CMED readjustment passed through in the second quarter (used to infer it from data)


def lg(x):
    """Percent change -> log-%; keeps pandas objects (and their index) as they are."""
    return 100 * np.log1p(x / 100.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def monthly_items():
    """Monthly variation (%) and weight (%) of the 24 items and the general index (SIDRA), PeriodIndex 'M'."""
    df = fetch_adm.sidra_subitems()
    var = df[df["variable"] == "var"].pivot(index="month", columns="code", values="value")
    w = df[df["variable"] == "weight"].pivot(index="month", columns="code", values="value")
    return var, w


def flag_contribution(var_elec, flags, T0_grid=np.arange(250, 1600, 10)):
    """Monthly contribution of the tariff flag to the electricity item (log-%), from the ANEEL surcharge S_m
    (R$/MWh): b_m = 100*[log(1+S_m/T_m) - log(1+S_{m-1}/T_{m-1})], with the reference tariff T_m = T0 * I_m / I_ref
    (I: cumulative index of the item). T0 is the value on the grid that minimises the residual variance of the item's
    monthly inflation net of the flag effect (a one-parameter calibration, re-done at each run)."""
    s = flags["surcharge"].reindex(var_elec.index).fillna(0.0)
    I = (1 + var_elec / 100).cumprod()
    I = I / I.loc["2024-01":"2024-12"].mean()
    best = None
    for T0 in T0_grid:
        e = 100 * np.log1p(s / (T0 * I))
        b = e.diff().fillna(0.0)
        resid = lg(var_elec) - b
        v = float(resid.loc["2015-01":].var())
        if best is None or v < best[0]:
            best = (v, T0, b, e)
    return best[2], best[1]


def build_data(refresh=False):
    """Quarterly dataset of the administered model: item inflations (log-%), weights, drivers and the inferred
    annual readjustments. Returns dict(q=DataFrame, m=DataFrame(monthly), weights_last=Series, T0=float)."""
    if refresh:
        fetch_adm.sidra_subitems(refresh=True); fetch_adm.aneel_flags(refresh=True); fetch_adm.bls_cpi(refresh=True)
        fetch.eia_brent_daily(refresh=True)
    var, w = monthly_items()
    var = var.loc["2012-01":]
    flags = fetch_adm.aneel_flags()
    band_m, T0 = flag_contribution(var["2202003"], flags)
    m = pd.DataFrame({c: lg(var[c]) for c in var.columns})          # monthly log-%
    m["band"] = band_m.reindex(m.index).fillna(0.0)       # no flags before 2015
    m["energia_exband"] = m["2202003"] - m["band"]
    ts = lambda s: pd.Series(s.values, index=s.index.to_timestamp())  # noqa: E731
    qsum = lambda c: ts(m[c]).groupby(ts(m[c]).index.to_period("Q")).agg(["sum", "count"])  # noqa: E731
    q = pd.DataFrame(index=pd.period_range(m.index[0].asfreq("Q"), m.index[-1].asfreq("Q"), freq="Q"))
    for c in m.columns:
        g = qsum(c)
        q[c] = g["sum"].where(g["count"] == 3)
    # headline IPCA (log-%), Brent, exchange rate, US CPI
    ipca = fetch.sgs(SGS["ipca"], name="sgs_433_ipca")
    q["ipca"] = quarterly_log_inflation(ipca).reindex(q.index)
    brent = fetch.eia_brent_daily()
    q["oil_usd"] = (100 * np.log(to_quarterly_mean(brent))).diff().reindex(q.index)
    ptax = fetch.sgs(SGS["ptax_venda_mensal"], name="sgs_3698_ptax_venda_mensal")
    q["fx"] = (100 * np.log(to_quarterly_mean(ptax))).diff().reindex(q.index)
    cpi = fetch_adm.bls_cpi()
    cpi_q = cpi.groupby(cpi.index.asfreq("Q")).mean()
    q["cpi_us"] = (100 * np.log(cpi_q)).diff().reindex(q.index)
    q["x"] = q["oil_usd"] + q["fx"]                                   # oil in reais
    q["quarter"] = q.index.quarter
    # annual readjustments inferred from the data
    q["reaj_farm"] = np.nan           # CMED readjustment (log-%), placed in Q2 of each year
    q["R_plano"] = np.nan             # ANS readjustment (log-%) of the year, placed in every quarter of the year
    for y in sorted({p.year for p in q.index}):
        q2 = pd.Period(year=y, quarter=2, freq="Q")
        if q2 in q.index and pd.notna(q.loc[q2, "6101"]):
            q.loc[q2, "reaj_farm"] = q.loc[q2, "6101"] / C_REP_FARM
        mon = m["6203"][[(p.year == y) and (p.month >= 6) for p in m.index]].dropna()
        if len(mon):
            R = 12 * float(mon.iloc[-1])      # latest month from June: the monthly rate IBGE applies is R/12
            q.loc[[p for p in q.index if p.year == y], "R_plano"] = R
    # share of health plans in the IPCA (for the ex-plano IPCA in the ANS rule)
    w_plano = (w["6203"] / 100.0).reindex(m.index).ffill()
    wq = w_plano.groupby(w_plano.index.asfreq("Q")).mean().reindex(q.index)
    q["ipca_explano"] = (q["ipca"] - wq * q["6203"]) / (1 - wq)
    weights_last = (w.iloc[-1].drop("geral")).astype(float)
    return {"q": q, "m": m, "var_m": var, "w_m": w, "weights_last": weights_last, "T0": T0, "flags": flags}


def ivda_history(q, a=0.6):
    """IVDA_A inferred from the ANS readjustment: R_A = 0.8 IVDA_A + 0.2 IPCA^{ex plano}_{A-1} (annual, log-%)."""
    years = sorted({p.year for p in q.index})
    out = {}
    for y in years:
        R = q.loc[[p for p in q.index if p.year == y], "R_plano"].dropna()
        prev = q.loc[[p for p in q.index if p.year == y - 1], "ipca_explano"]
        if len(R) and len(prev) == 4 and prev.notna().all():
            out[y] = (float(R.iloc[0]) - 0.2 * float(prev.sum())) / 0.8
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Regressors of each item (quarter t of frame `f`, which holds history and projections)
# ---------------------------------------------------------------------------
def _sum_lags(f, col, t, lags):
    return float(sum(f.at[t - i, col] for i in lags))


def regressors(code, f, t):
    it = ITEMS[code]
    typ = it["type"]
    qtr = t.quarter
    d = [1.0 if qtr == j else 0.0 for j in (1, 2, 3, 4)]
    ipca4_l1 = _sum_lags(f, "ipca", t, range(1, 5))        # IPCA accumulated over the previous four quarters
    if typ == "fuel":
        L = it["lags"]
        oil = [f.at[t - i, "x"] for i in range(L + 1)]
        icms = d[0] * ipca4_l1
        ipca4_0 = _sum_lags(f, "ipca", t, range(0, 4))
        if it["margin"] == "gasolina":
            margin = 0.25 * ((2.0 / 3.0) * ipca4_l1 + (1.0 / 3.0) * ipca4_0)
        elif it["margin"] == "diesel":
            margin = 0.25 * ipca4_l1
        else:
            margin = d[2] * ((2.0 / 3.0) * ipca4_l1 + (1.0 / 3.0) * ipca4_0)
        return np.array(oil + [icms, margin])
    if typ == "gas_encanado":
        return np.array([ipca4_l1 / 4.0, f.at[t, "2201004"], _sum_lags(f, "x", t, range(1, 5))])
    if typ == "gas_veicular":
        return np.array([ipca4_l1, f.at[t - 1, "5104001"], f.at[t - 2, "5104001"], f.at[t - 1, "x"], f.at[t - 2, "x"], f.at[t - 3, "x"]])
    if typ == "energia":
        ic4 = _sum_lags(f, "itaipu", t, range(1, 5)) + _sum_lags(f, "fx", t, range(1, 5))
        return np.array([dj * ipca4_l1 for dj in d] + [dj * ic4 for dj in d])
    if typ == "farmacia":
        # the CMED readjustment of the year is known in the second quarter; the dummies select the relevant lag
        lag = {1: 3, 2: 0, 3: 1, 4: 2}[qtr]
        r = f.at[t - lag, "reaj_farm"]
        return np.array([r if qtr == 1 else 0.0, r if qtr == 2 else 0.0, r if qtr == 3 else 0.0, r if qtr == 4 else 0.0])
    if typ == "plano":
        prev = pd.Period(year=t.year - 1, quarter=4, freq="Q")
        return np.array([f.at[prev, "R_plano"] / 4.0 if qtr == 1 else 0.0, 0.0 if qtr == 1 else f.at[t, "R_plano"] / 4.0])
    if typ == "emplacamento":
        return np.array([ipca4_l1 / 4.0 if qtr == 1 else 0.0, 0.0 if qtr == 1 else f.at[t - 1, code]])
    if typ == "seasonal":
        return np.array([dj * ipca4_l1 for dj in d])
    if typ == "nonseasonal":
        return np.array([ipca4_l1 / 4.0])
    if typ == "q1":
        return np.array([d[0] * ipca4_l1])
    raise ValueError(typ)


def _prepare_frame(q):
    f = q.copy()
    f["itaipu"] = 0.0
    for t in f.index:
        if t.quarter == 1 and all((t - i) in f.index for i in range(1, 5)):
            f.at[t, "itaipu"] = _sum_lags(f, "cpi_us", t, range(1, 5))
    # readjustment series as lag-able columns: reaj_farm is placed in Q2 (the equations look it up with lags);
    # R_plano is a yearly constant
    return f


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _hac_ols(y, X, lags=4):
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        wgt = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    r2 = 1 - e.var() / y.var() if y.var() > 0 else np.nan
    return beta, se, r2, len(y)


def estimate(data, start="2012Q2", start_seasonal="2020Q1", end=None):
    """OLS estimates of theta for every item on the model's own regressors. Items indexed to past inflation with
    seasonal weights use the sample from `start_seasonal` (the BCB calibrates them 'with a sample from 2020');
    the other items use the sample from `start`. Returns (table, theta_hat dict)."""
    q = data["q"]
    f = _prepare_frame(q)
    end = pd.Period(end, freq="Q") if end else f.index[-1]
    rows, theta_hat = [], {}
    for code in ORDER:
        it = ITEMS[code]
        s0 = pd.Period(start_seasonal if it["type"] in ("seasonal", "nonseasonal", "q1", "emplacamento") else start, freq="Q")
        ys, Xs = [], []
        ycol = "energia_exband" if it["type"] == "energia" else code
        for t in f.index:
            if t < s0 or t > end or (t - 5) not in f.index:
                continue
            try:
                z = regressors(code, f, t)
            except KeyError:
                continue
            y = f.at[t, ycol]
            if np.isfinite(y) and np.all(np.isfinite(z)):
                ys.append(y); Xs.append(z)
        if len(ys) < 8:
            theta_hat[code] = BCB_THETA[code]
            continue
        y, X = np.array(ys), np.array(Xs)
        keep = X.std(axis=0) > 1e-12          # regressors that vary in the sample (all-zero dummies cannot be estimated)
        beta = np.full(X.shape[1], np.nan); se = np.full(X.shape[1], np.nan)
        b, s, r2, n = _hac_ols(y, X[:, keep])
        beta[keep], se[keep] = b, s
        th = BCB_THETA[code].copy()
        th[keep] = b
        theta_hat[code] = th
        for k, lab in enumerate(it["labels"]):
            rows.append({"code": code, "item": it["name"], "parameter": lab, "bcb": BCB_THETA[code][k], "estimate": beta[k], "se": se[k],
                         "t_diff": (beta[k] - BCB_THETA[code][k]) / se[k] if np.isfinite(se[k]) and se[k] > 0 else np.nan,
                         "sample": "%s-%s" % (s0, end), "n": n, "r2": r2})
    return pd.DataFrame(rows), theta_hat


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(theta, data, origin, horizon, exo, weights=None, overrides=None):
    """Dynamic simulation of the 24 equations from the quarter after `origin` for `horizon` quarters.

    exo: DataFrame indexed by the projection quarters with columns ipca (headline, log-%), x (oil in reais, log-%),
    fx (exchange rate, log-%), cpi_us (log-%), band (flag contribution to the electricity item, log-% per quarter;
    NaN -> the repeating rule of the box) and optionally R_plano / reaj_farm (annual readjustments, log-%, for years
    not yet observed; NaN -> the rules of the box: ANS rule with the IVDA recursion, CMED rule with Fator Y).
    Returns a DataFrame with the item paths, the flag contribution, the aggregate administered inflation ('adm')
    and the headline used ('ipca')."""
    q = data["q"]
    f = _prepare_frame(q.loc[:origin])
    weights = weights if weights is not None else data["weights_last"]
    proj = pd.period_range(origin + 1, origin + horizon, freq="Q")
    f = f.reindex(f.index.union(proj))
    for c in ["ipca", "x", "fx", "cpi_us"]:
        f.loc[proj, c] = exo[c].reindex(proj).values
    f.loc[proj, "oil_usd"] = f.loc[proj, "x"] - f.loc[proj, "fx"]
    f.loc[proj, "quarter"] = [p.quarter for p in proj]
    ivda = ivda_history(q.loc[:origin], ITEMS["6203"]["ivda_a"])
    a_ivda = ITEMS["6203"]["ivda_a"]
    cY = ITEMS["6101"]["fatorY"]
    overrides = overrides or {}
    for t in proj:
        # drivers that depend on the year
        if t.quarter == 1:
            f.at[t, "itaipu"] = _sum_lags(f, "cpi_us", t, range(1, 5))
        else:
            f.at[t, "itaipu"] = 0.0
        y = t.year
        if np.isnan(f.at[t, "R_plano"]):
            if "R_plano" in overrides and y in overrides["R_plano"]:
                f.at[t, "R_plano"] = overrides["R_plano"][y]
            elif t.quarter == 1 and pd.notna(f.at[t - 1, "R_plano"]):
                f.at[t, "R_plano"] = f.at[t - 1, "R_plano"]       # Q1 uses last year's readjustment anyway
            else:
                prev_year = [p for p in f.index if p.year == y - 1]
                ipca_prev = float(f.loc[prev_year, "ipca"].sum()) if len(prev_year) == 4 else float(f.loc[prev_year, "ipca"].sum())
                ipca_explano_prev = float(f.loc[prev_year, "ipca_explano"].sum()) if f.loc[prev_year, "ipca_explano"].notna().all() else ipca_prev
                if (y - 1) in ivda.index:
                    iv_prev = float(ivda.loc[y - 1])
                elif len(ivda):
                    iv_prev = float(ivda.iloc[-1])
                else:
                    iv_prev = ipca_prev
                iv = a_ivda * iv_prev + (1 - a_ivda) * ipca_prev
                ivda.loc[y] = iv
                R = 0.8 * iv + 0.2 * ipca_explano_prev
                f.loc[[p for p in f.index if p.year == y], "R_plano"] = f.loc[[p for p in f.index if p.year == y], "R_plano"].fillna(R)
        if t.quarter == 2 and np.isnan(f.at[t, "reaj_farm"]):
            if "reaj_farm" in overrides and y in overrides["reaj_farm"]:
                f.at[t, "reaj_farm"] = overrides["reaj_farm"][y]
            else:
                ipca4 = _sum_lags(f, "ipca", t, range(1, 5))
                fatorY = cY["cambio"] * (_sum_lags(f, "fx", t, range(1, 5)) + _sum_lags(f, "cpi_us", t, range(1, 5)) - ipca4) \
                    + cY["energia"] * (_sum_lags(f, "energia_exband", t, range(1, 5)) - ipca4)
                f.at[t, "reaj_farm"] = ipca4 + fatorY
        # items in dependency order
        for code in ORDER:
            z = regressors(code, f, t)
            val = float(np.dot(theta[code], z))
            if ITEMS[code]["type"] == "energia":
                f.at[t, "energia_exband"] = val
                b = exo["band"].get(t, np.nan) if "band" in exo.columns else np.nan
                if not np.isfinite(b):
                    b = -(f.at[t - 1, "band"] + f.at[t - 2, "band"] + f.at[t - 3, "band"])
                f.at[t, "band"] = b
                val = val + b
            f.at[t, code] = val
    wsum = float(weights[ORDER].sum())
    f["adm"] = sum(f[c] * float(weights[c]) for c in ORDER) / wsum
    return f.loc[proj[0] - 8:, ORDER + ["band", "energia_exband", "adm", "ipca", "x", "fx", "R_plano", "reaj_farm"]]


def aggregate_history(data):
    """Aggregate of the 24 items with their monthly weights (log-%, quarterly), to compare with the official
    administered-price index (SGS 4449)."""
    m, w = data["m"], data["w_m"]
    wa = w[ORDER].reindex(m.index).ffill()
    agg_m = (m[ORDER] * wa).sum(axis=1) / wa.sum(axis=1)
    ts = pd.Series(agg_m.values, index=agg_m.index.to_timestamp())
    g = ts.groupby(ts.index.to_period("Q")).agg(["sum", "count"])
    off = quarterly_log_inflation(fetch.sgs(SGS["ipca_monit"], name="sgs_4449_ipca_monit"))
    return pd.DataFrame({"items_24": g["sum"].where(g["count"] == 3), "official": off}).dropna()


def fit_in_sample(theta, data, start="2014Q1", end=None):
    """One-step (static) fit of every equation with the realized regressors, and the implied aggregate."""
    q = data["q"]
    f = _prepare_frame(q)
    end = pd.Period(end, freq="Q") if end else f.index[-1]
    idx = [t for t in f.index if pd.Period(start, freq="Q") <= t <= end]
    fit = pd.DataFrame(index=idx, columns=ORDER, dtype=float)
    for t in idx:
        for code in ORDER:
            try:
                z = regressors(code, f, t)
            except KeyError:
                continue
            if np.all(np.isfinite(z)):
                v = float(np.dot(theta[code], z))
                if ITEMS[code]["type"] == "energia":
                    v += f.at[t, "band"] if np.isfinite(f.at[t, "band"]) else 0.0
                fit.at[t, code] = v
    w = data["weights_last"]
    actual = q.loc[idx, ORDER]
    fit["adm_fit"] = sum(fit[c] * float(w[c]) for c in ORDER) / float(w[ORDER].sum())
    fit["adm_actual"] = sum(actual[c] * float(w[c]) for c in ORDER) / float(w[ORDER].sum())
    return fit
