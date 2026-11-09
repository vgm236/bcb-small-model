"""
Projection of administered prices with the BCB's medium-term model (administered.py) inside the reference
scenario: real-time truncation of the item data, scenario paths for the drivers (oil in reais, exchange rate,
US CPI, tariff flag), the nowcast of the current quarter, the dynamic simulation over the horizon, and the
feedback loop with the small model (the headline IPCA that indexes the administered items is the one produced by
the combination of free and administered prices).
"""
import numpy as np
import pandas as pd

import fetch_adm
import administered as adm

_CACHE = {}


def data_cached(refresh=False):
    key = "data"
    if refresh or key not in _CACHE:
        _CACHE[key] = adm.build_data(refresh=refresh)
    return _CACHE[key]


def truncate(data, cut_m, decision_date):
    """Real-time version of the administered dataset: items through the IPCA cut-off month, flags through the
    month of the decision (flags are announced before the month starts)."""
    D = pd.Timestamp(decision_date)
    q = data["q"].copy()
    last_q = cut_m.asfreq("Q") if cut_m.month in (3, 6, 9, 12) else cut_m.asfreq("Q") - 1
    q = q.loc[:last_q]
    m = data["m"].loc[:cut_m].copy()
    w_m = data["w_m"].loc[:cut_m]
    flags = data["flags"].loc[:D.to_period("M")]
    # readjustments inferred only from months up to the cut-off (build_data used the full sample)
    for y in sorted({p.year for p in q.index}):
        mon = m["6203"][[(p.year == y) and (p.month >= 6) for p in m.index]].dropna()
        rows = [p for p in q.index if p.year == y]
        q.loc[rows, "R_plano"] = 12 * float(mon.iloc[-1]) if len(mon) else np.nan
        q2 = pd.Period(year=y, quarter=2, freq="Q")
        if q2 in q.index:
            q.loc[q2, "reaj_farm"] = q.loc[q2, "6101"] / adm.C_REP_FARM if pd.notna(q.loc[q2, "6101"]) else np.nan
    return {"q": q, "m": m, "var_m": data["var_m"].loc[:cut_m], "w_m": w_m, "weights_last": w_m.iloc[-1].drop("geral").astype(float),
            "T0": data["T0"], "flags": flags}


def flag_path_monthly(data, start_m, end_m, flag_december=None, hold="last"):
    """Surcharge (R$/MWh) by month for the projection months: the last flag in force is held through the current
    quarter; from the next quarter the December assumption of each year applies when given (e.g.
    {2026: 'amarela'}), otherwise the last flag is held."""
    flags = data["flags"]
    vals = fetch_adm.current_flag_values()
    last_flag = str(flags["flag"].iloc[-1]) if len(flags) else "verde"
    last_s = float(flags["surcharge"].iloc[-1]) if len(flags) else 0.0
    out = {}
    q_now = start_m.asfreq("Q")
    for mth in pd.period_range(start_m, end_m, freq="M"):
        if mth in flags.index:
            out[mth] = float(flags.loc[mth, "surcharge"])
        elif mth.asfreq("Q") == q_now or not flag_december:
            out[mth] = last_s
        else:
            name = flag_december.get(mth.year, flag_december.get(str(mth.year), last_flag))
            out[mth] = float(vals.get(str(name).lower(), last_s))
    return pd.Series(out)


def band_quarterly(data, surcharge_m):
    """Quarterly flag contribution (log-%) over a monthly surcharge path, using the calibrated reference tariff
    (held at its last observed level)."""
    m = data["m"]
    var = data["var_m"]["2202003"]
    I = (1 + var / 100).cumprod()
    I = I / I.loc["2024-01":"2024-12"].mean()
    I_last = float(I.iloc[-1])
    T = data["T0"] * I_last
    flags = data["flags"]
    s_prev = float(flags["surcharge"].iloc[-1]) if len(flags) else 0.0
    # contribution of the last observed month relative to its predecessor is already in the data; start after it
    rows = {}
    prev = s_prev
    for mth, s in surcharge_m.items():
        if mth <= flags.index.max():
            prev = s
            continue
        rows[mth] = 100 * (np.log1p(s / T) - np.log1p(prev / T))
        prev = s
    b = pd.Series(rows)
    if not len(b):
        return pd.Series(dtype=float)
    return b.groupby(b.index.asfreq("Q")).sum()


def run(decision_date, cut_ipca_m, paths, H, theta="bcb", flag_december=None, headline=None, cpi_us_rate=None,
        data=None, overrides=None, estimate_start="2012Q2"):
    """Administered-price projection for the quarters from the current one (q0) to q0+H.

    paths: the reference-scenario paths (projection.build_assumptions): brent_monthly, ptax_daily.
    headline: Series of the headline IPCA (log-%, quarterly) over q0..q0+H used for indexation (from the small
    model's iteration); None -> the Focus nowcast for q0 and 3%/year afterwards (first pass).
    Returns dict(quarterly=DataFrame of items and aggregate, adm=Series log-%, nowcast_monthly_pct=Series for the
    months of q0 after the cut-off, theta=dict, estimates=DataFrame or None, origin=Period).
    """
    D = pd.Timestamp(decision_date)
    q0 = D.to_period("Q")
    full = data or data_cached()
    dv = truncate(full, cut_ipca_m, D)
    q = dv["q"]
    origin = q.index[-1]                      # last complete quarter in the real-time data (q0-1 or q0-2)
    n = (q0 + H - origin).n
    proj = pd.period_range(origin + 1, origin + n, freq="Q")
    # drivers over the projection quarters
    brent = paths["brent_monthly"]
    bq = brent.groupby(brent.index.asfreq("Q")).mean()
    oil = (100 * np.log(bq)).diff()
    ptax = paths["ptax_daily"]
    pq = ptax.groupby(ptax.index.to_period("Q")).mean()
    fx = (100 * np.log(pq)).diff()
    cpi_q = q["cpi_us"].dropna()
    cpi_rate = cpi_us_rate if cpi_us_rate is not None else float(cpi_q.tail(8).mean())
    surcharge = flag_path_monthly(dv, cut_ipca_m + 1, (origin + n).asfreq("M", "end"), flag_december)
    band_q = band_quarterly(dv, surcharge)
    exo = pd.DataFrame(index=proj)
    exo["oil_usd"] = oil.reindex(proj).values
    exo["fx"] = fx.reindex(proj).values
    exo["x"] = exo["oil_usd"] + exo["fx"]
    exo["cpi_us"] = cpi_rate
    exo["band"] = band_q.reindex(proj).values
    exo.loc[exo.index < q0, "band"] = exo.loc[exo.index < q0, "band"].fillna(0.0)
    if headline is None:
        exo["ipca"] = 100 * np.log1p(3.0 / 100.0) / 4
        if "focus_monthly_ipca" in paths:
            fm = paths["focus_monthly_ipca"]
            for qq in proj:
                if qq <= q0:
                    months = pd.period_range(qq.asfreq("M", "start"), qq.asfreq("M", "end"), freq="M")
                    vals = []
                    for mm in months:
                        if mm in dv["m"].index:
                            vals.append(float(adm.lg(dv["var_m"].loc[mm, "geral"])))
                        elif mm in fm:
                            vals.append(float(adm.lg(fm[mm])))
                    if len(vals) == 3:
                        exo.loc[qq, "ipca"] = sum(vals)
    else:
        exo["ipca"] = pd.Series(headline).reindex(proj).values
        exo["ipca"] = exo["ipca"].fillna(100 * np.log1p(3.0 / 100.0) / 4)
    estimates = None
    if isinstance(theta, dict):
        th = theta
    elif theta == "bcb":
        th = adm.BCB_THETA
    else:
        estimates, th = adm.estimate(dv, start=estimate_start)
    sim = adm.simulate(th, dv, origin, n, exo, weights=dv["weights_last"], overrides=overrides)
    adm_q = sim.loc[proj, "adm"]
    # monthly nowcast (%) for the months of q0 (and of any incomplete earlier quarter) after the cut-off: the
    # model's quarterly rate spread evenly
    now = {}
    for qq in proj:
        if qq > q0:
            break
        for mm in pd.period_range(qq.asfreq("M", "start"), qq.asfreq("M", "end"), freq="M"):
            if mm > cut_ipca_m:
                now[mm] = 100 * np.expm1(adm_q[qq] / 3.0 / 100.0)
    return {"quarterly": sim, "adm": adm_q, "nowcast_monthly_pct": pd.Series(now), "theta": th, "estimates": estimates,
            "origin": origin, "exo": exo, "surcharge": surcharge, "data": dv}


def monthly_to_quarterly_log(nowcast_pct, realized_log_m, quarters):
    """Quarterly log-% of administered inflation combining realized months and the nowcast months."""
    out = {}
    for qq in quarters:
        months = pd.period_range(qq.asfreq("M", "start"), qq.asfreq("M", "end"), freq="M")
        tot = 0.0
        for mm in months:
            if mm in realized_log_m.index and pd.notna(realized_log_m.loc[mm]):
                tot += float(realized_log_m.loc[mm])
            elif mm in nowcast_pct.index:
                tot += float(100 * np.log1p(nowcast_pct.loc[mm] / 100.0))
        out[qq] = tot
    return pd.Series(out)
