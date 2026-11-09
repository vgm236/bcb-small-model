"""
Conditional inflation projections with the estimated model, in the format of the Copom's reference scenario.

Pipeline for an information date (a Copom meeting or "today"):
 1. assumptions of the reference scenario (Focus survey of the reference Friday F, exchange rate from the 10-day
    PTAX average and PPP, Brent from the spot price and the futures-curve proxy, administered prices from the Copom
    or from the Focus survey) -> daily/monthly paths;
 2. real-time dataset through the current quarter (vintage.py), with the current quarter nowcast;
 3. Kalman filter of the full system (model_system.py) up to the current quarter -> state at the forecast origin;
 4. conditional forecast: the Selic, the exchange-rate change, administered inflation, USD commodity inflation and
    the climate terms follow their paths, imposed through *anticipated* shocks (news) so that the model-consistent
    expectations are consistent with the published Selic path and administered-price assumption; the other drivers
    (world gap, fiscal impulse, external rate) follow their AR(1) processes from the filtered state;
 5. free-price inflation (seasonally adjusted, log-% per quarter) is converted back to calendar inflation with the
    seasonal factors, combined with the administered-price path through the IPCA identity, and accumulated over
    calendar years and four quarters, as in the Copom's Table 1.
"""
import numpy as np
import pandas as pd

import fetch
import model_system as ms
import vintage
from re_solver import gensys
from config import DATA_PROC, OUTPUT, SAMPLE_START, SAMPLE_END, PI_STAR_SS
from build_dataset import INFLATION_TARGET, _meeting_dates
from copom_meetings import STEO_VINTAGES
from vintage import LG

H_DEFAULT = 12
COND = {"i": "e_i", "de": "e_e", "piA": "e_piA", "pistar_usd": "e_pistar_usd", "el": "e_el", "la": "e_la"}
# Configuration adopted for the Copom-style projections after the diagnostics of notebook 09 (see docs/projection.md):
# the settings closest to the Copom's numbers over the backtest meetings. Overridden by the app's sidebar.
ADOPTED = {"neutral_rate": 5.0, "climate": "realtime", "climate_index": "oni", "coefficients": "estimated",
           "gap_anchor": "rpm", "adm_profile": "uniform", "nowcast": "focus"}
# Administered-price model (docs/administered.md, section 5): when the model is used, the BCB's calibration with the
# annual figures kept for four quarters (the specialists' horizon) is the configuration closest to the Copom.
ADOPTED_ADM_MODEL = {"adm_source": "model", "adm_theta": "bcb", "adm_handover": 4}
# Copom meetings not yet in the BCB history API (decision dates, from the published calendars); other future
# meetings use the usual month of the year.
FUTURE_MEETINGS = {"R7/2026": "2026-11-04", "R8/2026": "2026-12-09",
                   "R1/2027": "2027-01-27", "R2/2027": "2027-03-17", "R3/2027": "2027-04-28", "R4/2027": "2027-06-16",
                   "R5/2027": "2027-08-04", "R6/2027": "2027-09-22", "R7/2027": "2027-10-27", "R8/2027": "2027-12-08"}


# ---------------------------------------------------------------------------
# Focus survey at a reference date
# ---------------------------------------------------------------------------
def _at(df, F, col="Mediana"):
    """Rows of the latest survey date <= F (full respondent base)."""
    d = df[(df["baseCalculo"] == 0) & (df["Data"] <= F)]
    return d[d["Data"] == d["Data"].max()]


def focus_snapshot(F):
    F = pd.Timestamp(F)
    out = {"F": F}
    s = _at(fetch.focus_selic_reunioes(), F)
    out["selic_meetings"] = s.set_index("Reuniao")["Mediana"].astype(float).to_dict()
    out["selic_date"] = s["Data"].max()
    a = _at(fetch.focus_anuais("Selic"), F)
    out["selic_annual"] = a.set_index("DataReferencia")["Mediana"].astype(float).to_dict()
    for ind, key in [("IPCA", "ipca_annual"), ("IPCA Administrados", "adm_annual"), ("IPCA Livres", "livres_annual"),
                     ("PIB Total", "gdp_annual"), ("Câmbio", "fx_annual")]:
        a = _at(fetch.focus_anuais(ind), F)
        out[key] = {int(k): float(v) for k, v in a.set_index("DataReferencia")["Mediana"].items()}
    m = _at(fetch.focus_monthly("IPCA"), F)
    out["ipca_monthly"] = {pd.Period(r, freq="M"): float(v) for r, v in zip(pd.to_datetime(m["DataReferencia"], format="%m/%Y"), m["Mediana"])}
    f12 = _at(fetch.focus_inflacao_12m("S"), F)
    out["pie_12m"] = float(f12["Mediana"].iloc[0])
    return out


def selic_path_daily(snap, decision_date, end, selic_daily):
    """Daily Selic target: realized through the decision date (the new rate applies from the next day), then the
    Focus per-meeting medians, then the Focus end-year medians (linear interpolation), then constant."""
    D = pd.Timestamp(decision_date)
    sel = selic_daily.sort_index()
    cal = pd.date_range(sel.index.min(), end, freq="D")
    path = sel.reindex(cal).ffill()
    cur = float(path.loc[D])
    lookup = _meeting_dates()
    meet = sorted((pd.Timestamp(FUTURE_MEETINGS[k]) if k in FUTURE_MEETINGS else lookup(k), v) for k, v in snap["selic_meetings"].items())
    path.loc[path.index > D] = cur
    last_date, last_val = D, cur
    decided = sel.index.max()            # decisions up to this date are in the realized series
    for md, val in meet:
        if md < D or (md == D and decided > D):
            continue
        path.loc[path.index > md] = val
        last_date, last_val = md, val
    # end-year medians beyond the last meeting with a median
    anchors = [(pd.Timestamp(year=int(y), month=12, day=31), v) for y, v in sorted(snap["selic_annual"].items())
               if pd.Timestamp(year=int(y), month=12, day=31) > last_date + pd.Timedelta(days=120)]
    prev_d, prev_v = last_date, last_val
    for ad, av in anchors:
        seg = path.index[(path.index > prev_d) & (path.index <= ad)]
        if len(seg):
            # step changes at the (assumed) meeting dates are not known; use a linear glide between anchors
            path.loc[seg] = prev_v + (av - prev_v) * ((seg - prev_d).days / (ad - prev_d).days)
        prev_d, prev_v = ad, av
    path.loc[path.index > prev_d] = prev_v
    return path


def expected_selic_12m(path, F):
    F = pd.Timestamp(F)
    return float(path.loc[F + pd.Timedelta(days=1):F + pd.Timedelta(days=365)].mean())


def fx_start_convention(F, ptax_daily=None):
    """Copom convention: average PTAX of the ten business days ending on the Friday before the meeting week (F),
    rounded to R$0.05."""
    ptax_daily = ptax_daily if ptax_daily is not None else fetch.sgs(1, start="2023-01-01", name="sgs_1_ptax_venda_daily", daily=True)
    win = ptax_daily[ptax_daily.index <= pd.Timestamp(F)].tail(10)
    return round(float(win.mean()) * 20) / 20, win


def fx_path_daily(F, start_value, end, ptax_daily):
    """Daily PTAX: realized through F, then the start value (10-day average, rounded) growing with the PPP drift
    (target minus foreign inflation of 2%)."""
    F = pd.Timestamp(F)
    hist = ptax_daily[ptax_daily.index <= F]
    fut = pd.bdate_range(F + pd.Timedelta(days=1), end)
    drift = np.array([(vintage.target(d.year) - PI_STAR_SS) / 100.0 / 365.0 for d in fut])
    days = np.array([(d - F).days for d in fut], dtype=float)
    vals = start_value * np.exp(np.cumsum(drift * np.diff(np.concatenate([[0.0], days]))))
    return pd.concat([hist, pd.Series(vals, index=fut)])


def steo_vintage_for(F):
    """Latest STEO published (forecast date + 6 days) on or before F. Vintages after the list in copom_meetings
    (the month of F and the month before) are tried as downloads."""
    F = pd.Timestamp(F)
    cands = list(STEO_VINTAGES)
    for k in (1, 0):
        v = (F - pd.DateOffset(months=k)).strftime("%b%y").lower()
        if v not in cands:
            cands.append(v)
    best = None
    for v in cands:
        try:
            fd, _ = fetch.eia_steo_brent(v)
        except Exception:  # noqa: BLE001  (vintage not published yet)
            continue
        if fd + pd.Timedelta(days=6) <= F and (best is None or fd > best[1]):
            best = (v, fd)
    if best is None:
        raise ValueError("no STEO vintage before %s" % F.date())
    return best[0]


def refresh_inputs():
    """Re-download the inputs that change between meetings (Focus survey, Selic, PTAX, prices, activity, Brent)."""
    for ind in ["Selic", "IPCA", "IPCA Administrados", "IPCA Livres", "PIB Total", "Câmbio"]:
        fetch.focus_anuais(ind, refresh=True)
    fetch.focus_selic_reunioes(refresh=True)
    fetch.focus_inflacao_12m("S", refresh=True)
    fetch.focus_monthly("IPCA", refresh=True)
    fetch.sgs(432, name="sgs_432_selic_meta_daily", daily=True, refresh=True)
    fetch.sgs(1, start="2023-01-01", name="sgs_1_ptax_venda_daily", daily=True, refresh=True)
    for code, name in [(433, "sgs_433_ipca"), (11428, "sgs_11428_ipca_livres"), (4449, "sgs_4449_ipca_monit"),
                       (3698, "sgs_3698_ptax_venda_mensal"), (27574, "sgs_27574_icbr"), (27575, "sgs_27575_icbr_agro"),
                       (27576, "sgs_27576_icbr_metal"), (27577, "sgs_27577_icbr_energia"), (28561, "sgs_28561_nuci_fgv_sa"),
                       (24369, "sgs_24369_desocup_pnadc"), (28763, "sgs_28763_caged_estoque"), (4639, "sgs_4639_prim_gov_central"),
                       (4380, "sgs_4380_pib_nominal_mensal")]:
        fetch.sgs(code, name=name, refresh=True)
    fetch.eia_brent_daily(refresh=True)
    fetch.noaa_oni(refresh=True)
    fetch.noaa_nino34_weekly(refresh=True)
    fetch.copom_meetings(refresh=True)


def brent_path_monthly(F, end_month, months_curve=6, growth_after=2.0):
    """Monthly Brent: realized monthly averages, the partial month of F, then the futures-curve proxy (the shape of
    the latest EIA STEO path applied to the spot level at F) for `months_curve` months, then +growth_after % a year."""
    F = pd.Timestamp(F)
    spot = fetch.eia_brent_daily()
    spot = spot[spot.index <= F]
    m = spot.groupby(spot.index.to_period("M")).mean()
    mF = F.to_period("M")
    vint = steo_vintage_for(F)
    _, steo = fetch.eia_steo_brent(vint)
    level = float(spot.iloc[-5:].mean())            # spot level at F (5-day average)
    out = m.copy()
    out.loc[mF] = level
    for k in range(1, months_curve + 1):
        mm = mF + k
        out.loc[mm] = level * float(steo.get(mm, steo.iloc[-1]) / steo.get(mF, steo.iloc[-1]))
    g = (1 + growth_after / 100.0) ** (1 / 12.0)
    for mm in pd.period_range(mF + months_curve + 1, end_month, freq="M"):
        out.loc[mm] = out.loc[mm - 1] * g
    return out.sort_index(), vint


def administered_seasonal(cut_m):
    """Monthly seasonal factors (log-% points) of administered-price inflation: robust STL on the monthly series
    through cut_m; the factors of the last twelve months are used for the projection."""
    from statsmodels.tsa.seasonal import STL
    mon = vintage._monthly(fetch.sgs(4449, name="sgs_4449_ipca_monit"), cut_m).loc["2004-01":]
    lm = 100 * np.log1p(mon / 100.0)
    res = STL(lm.values, period=12, robust=True).fit()
    seas = pd.Series(res.seasonal, index=lm.index)
    return {p.month: float(v) for p, v in seas.iloc[-12:].items()}


def administered_path_monthly(spec_years, horizon, cut_m, end_month, profile="uniform", seasonal_m=None):
    """Monthly administered-price inflation for the months after the IPCA cut-off.

    The path is consistent with the calendar-year projections `spec_years` {year: %} (the months already published
    are taken into account) and, for the year after the last one given, with the four-quarter projection
    `horizon` = (quarter, %) over the months of its window; later months keep the last annual rate. Within each block
    the monthly profile is flat (profile='uniform', the default: the Copom publishes only annual figures, and the
    monthly seasonality of administered prices is dominated by one-off tariff events) or follows the STL seasonal
    factors of administered inflation (profile='seasonal'). Returns (% simple, log-%)."""
    mon = vintage._monthly(fetch.sgs(4449, name="sgs_4449_ipca_monit"), cut_m)
    lmon = 100 * np.log1p(mon / 100.0)
    if profile == "uniform":
        seasonal_m = {k: 0.0 for k in range(1, 13)}
    else:
        seasonal_m = seasonal_m or administered_seasonal(cut_m)
    years = sorted(spec_years)
    last_rate = 100 * np.log1p(spec_years[years[-1]] / 100.0)   # log-% per year
    out = {}

    def value(mm):          # log-% of a month: published, already filled, or None
        if mm in out:
            return out[mm]
        if mm in lmon.index:
            return float(lmon[mm])
        return None

    def fill(months, total_log):
        s = np.array([seasonal_m[mm.month] for mm in months])
        base = (total_log - s.sum()) / len(months)
        for mm, sv in zip(months, s):
            out[mm] = base + sv

    m = cut_m + 1
    while m <= end_month:
        y = m.year
        months = list(pd.period_range(m, pd.Period(year=y, month=12, freq="M"), freq="M"))
        hq = pd.Period(horizon[0], freq="Q") if horizon is not None else None
        if y in spec_years:
            realized = sum(value(mm) for mm in pd.period_range(pd.Period(year=y, month=1, freq="M"), m - 1, freq="M")) if m.month > 1 else 0.0
            fill(months, 100 * np.log1p(spec_years[y] / 100.0) - realized)
        elif hq is not None and hq.year == y:
            win = list(pd.period_range((hq - 3).asfreq("M", "start"), hq.asfreq("M", "end"), freq="M"))
            known = sum(value(mm) for mm in win if mm < m)
            remaining = [mm for mm in win if mm >= m]
            fill(remaining, 100 * np.log1p(horizon[1] / 100.0) - known)
            rest = [mm for mm in months if mm > remaining[-1]]
            if rest:
                fill(rest, last_rate * len(rest) / 12.0)
        else:
            fill(months, last_rate * len(months) / 12.0)
        m = pd.Period(year=y + 1, month=1, freq="M")
    s = pd.Series(out).sort_index()
    return 100 * np.expm1(s / 100.0), s


def oni_ar1():
    """AR(1) coefficient of the quarterly ONI (seasonal values, 1950 onwards)."""
    oni = fetch.noaa_oni()
    o = oni[oni["SEAS"].isin(["JFM", "AMJ", "JAS", "OND"])]["ANOM"].astype(float).values
    return float(np.corrcoef(o[1:], o[:-1])[0, 1])


def oni_path(cut_q, end_q, decision_date, rho=None, mode="realtime", index="oni"):
    """ONI (or RONI) anomaly for the quarters after the last season published at the decision date.

    mode='realtime': the quarters up to the current one use the monthly Niño 3.4 anomalies (OISST) published by
    the decision date (quarterly means of the available months); later quarters decay with an AR(1) towards zero.
    mode='realized': the seasons published by now are used wherever available (perfect foresight of the climate
    variable, for sensitivity analysis), then the AR(1) decay.
    index='roni': NOAA's Relative ONI (the BCB's choice since 2026); the monthly nowcast is shifted by the average
    ONI-RONI difference of the last four published seasons."""
    rho = rho if rho is not None else oni_ar1()
    oni = fetch.noaa_roni() if index == "roni" else fetch.noaa_oni()
    season_to_q = {"JFM": 1, "AMJ": 2, "JAS": 3, "OND": 4}
    q_to_season = {v: k for k, v in season_to_q.items()}
    o = oni[oni["SEAS"].isin(season_to_q)].copy()
    o["q"] = pd.PeriodIndex([pd.Period(year=int(y), quarter=season_to_q[s], freq="Q") for y, s in zip(o["YR"], o["SEAS"])], freq="Q")
    oq = o.set_index("q")["ANOM"].astype(float).sort_index()
    D = pd.Timestamp(decision_date)
    mo = fetch.noaa_nino34_weekly()                          # monthly OISST anomalies
    mo = mo[[(p + 1).to_timestamp() + pd.Timedelta(days=4) <= D for p in mo.index]]   # public ~5th of the next month
    if index == "roni":
        on = fetch.noaa_oni()
        on = on[on["SEAS"].isin(season_to_q)].copy()
        on["q"] = pd.PeriodIndex([pd.Period(year=int(y), quarter=season_to_q[s], freq="Q") for y, s in zip(on["YR"], on["SEAS"])], freq="Q")
        onq = on.set_index("q")["ANOM"].astype(float).sort_index()
        common = oq.index.intersection(onq.index)
        common = common[common <= cut_q][-4:]
        mo = mo - float((onq.loc[common] - oq.loc[common]).mean())
    out = {}
    prev = float(oq.loc[cut_q])
    for q in pd.period_range(cut_q + 1, end_q, freq="Q"):
        avail = mo[[p.asfreq("Q") == q for p in mo.index]]
        if mode == "realized" and q in oq.index:
            prev = float(oq.loc[q])
        elif mode == "realtime" and q <= D.to_period("Q") and len(avail):
            prev = float(avail.mean())
        else:
            prev = rho * prev
        out[q] = prev
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Conditional forecasts with anticipated shocks (news)
# ---------------------------------------------------------------------------
def augment_news(G0, G1, C, Psi, Pi, shock_cols, K):
    """Add anticipated components to the shocks in `shock_cols`: e_t = u_t + z_{1,t-1}, z_{k,t} = z_{k+1,t-1} + nu_{k,t}
    (k = 1..K-1), z_{K,t} = nu_{K,t}. Returns the augmented matrices and the map (shock col, k) -> nu column."""
    n, ns, ne = G0.shape[0], Psi.shape[1], Pi.shape[1]
    S = len(shock_cols)
    na = n + S * K
    nn = ns + S * K
    G0a, G1a = np.zeros((na, na)), np.zeros((na, na))
    G0a[:n, :n], G1a[:n, :n] = G0, G1
    Ca = np.concatenate([C, np.zeros(S * K)])
    Psia = np.zeros((na, nn)); Psia[:n, :ns] = Psi
    Pia = np.zeros((na, ne)); Pia[:n, :] = Pi
    cols = {}
    for j, s in enumerate(shock_cols):
        cols[(s, 0)] = s
        for k in range(1, K + 1):
            z = n + j * K + (k - 1)          # state z_{s,k}
            nu = ns + j * K + (k - 1)        # news nu_{s,k}
            cols[(s, k)] = nu
            G0a[z, z] = 1.0
            if k < K:
                G1a[z, z + 1] = 1.0           # z_{k,t} = z_{k+1,t-1} + nu_{k,t}
            Psia[z, nu] = 1.0
        z1 = n + j * K
        G1a[:n, z1] += Psi[:, s]             # e_{s,t} includes z_{s,1,t-1}
    return G0a, G1a, Ca, Psia, Pia, cols


class Projector:
    """Solved model with news shocks for the conditioned variables."""

    def __init__(self, p, calib, exo, H=H_DEFAULT, cond=None):
        self.p, self.calib, self.exo, self.H = p, calib, exo, H
        self.cond = cond or COND
        G0, G1, C, Psi, Pi = ms.build_system(p, calib, exo)
        self.n0 = G0.shape[0]
        shock_cols = [ms.ish[s] for s in self.cond.values()]
        G0a, G1a, Ca, Psia, Pia, self.cols = augment_news(G0, G1, C, Psi, Pi, shock_cols, H)
        self.T, self.c, self.impact, self.eu = gensys(G0a, G1a, Ca, Psia, Pia)
        if self.eu != [1, 1]:
            raise RuntimeError("augmented system: existence/uniqueness %s" % self.eu)
        # plain system for the stochastic simulations
        self.T0, self.c0, self.impact0, _ = ms.solve(p, calib, exo)
        self.sd0 = self._shock_sd()

    def _shock_sd(self):
        p, calib, exo = self.p, self.calib, self.exo
        base = {"e_piL": p["sd_piL"], "e_pie": p["sd_pie"], "e_i": p["sd_i"], "e_e": p["sd_e"], "e_h": p["sd_h"],
                "e_crisis": 0.0, "e_rrIS": calib["sd_rrIS"], "e_rrTAY": calib["sd_rrTAY"], "e_ie": p["sd_ie"]}
        for v in ms.EXOG:
            base["e_" + v] = exo[v]["sd"]
        return np.array([base[s] for s in ms.SHOCKS])

    def deterministic(self, a0, paths):
        """Central conditional path. `paths`: DataFrame (H rows) with columns among the conditioned variables (model
        units); NaN entries leave that horizon unconditioned (the corresponding news is set to zero)."""
        H, n0 = self.H, self.n0
        na = self.T.shape[0]
        a = np.concatenate([np.asarray(a0, float), np.zeros(na - n0)])
        # baseline (no news) and multipliers M_j = T^{j-1} impact
        base = np.zeros((H, na))
        M = np.zeros((H, na, self.impact.shape[1]))
        y = a.copy()
        Mj = self.impact.copy()
        for j in range(H):
            y = self.T @ y + self.c
            base[j] = y
            M[j] = Mj
            Mj = self.T @ Mj
        rows, colsel, target = [], [], []
        for v, s in self.cond.items():
            if v not in paths.columns:
                continue
            sc = ms.ish[s]
            for j in range(H):
                val = paths[v].iloc[j]
                if np.isfinite(val):
                    rows.append((ms.iv[v], j)); colsel.append(self.cols[(sc, j)]); target.append(val - base[j, ms.iv[v]])
        R = np.array([[M[j][iv, c] for c in colsel] for (iv, j) in rows])
        nu = np.linalg.solve(R, np.array(target)) if len(rows) else np.zeros(0)
        nu_full = np.zeros(self.impact.shape[1])
        nu_full[colsel] = nu
        path = base + np.einsum("jnk,k->jn", M, nu_full)
        out = pd.DataFrame(path[:, :n0], columns=ms.VARS)
        return out, nu_full

    def simulate(self, a0, P0, paths, n_draws=500, seed=0):
        """Stochastic simulations around the conditional path: shocks other than the conditioned ones are drawn,
        the initial state is drawn from the filtered distribution, and the conditioning paths are re-imposed
        period by period (unanticipated). Returns the array (n_draws, H, n_vars) of simulated deviations from
        their mean (to be added to the central path)."""
        rng = np.random.default_rng(seed)
        H, n0 = self.H, self.n0
        T, c, imp = self.T0, self.c0, self.impact0
        cond_v = [v for v in self.cond if v in paths.columns]
        cond_s = [ms.ish[self.cond[v]] for v in cond_v]
        cond_i = [ms.iv[v] for v in cond_v]
        sd = self.sd0.copy()
        sd[cond_s] = 0.0
        w, V = np.linalg.eigh(0.5 * (P0 + P0.T))
        L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
        A = imp[np.ix_(cond_i, cond_s)]
        sims = np.zeros((n_draws, H, n0))
        for d in range(n_draws):
            a = np.asarray(a0, float) + L @ rng.standard_normal(n0)
            for j in range(H):
                a = T @ a + c + imp @ (sd * rng.standard_normal(len(sd)))
                tgt = paths[cond_v].iloc[j].values
                ok = np.isfinite(tgt)
                if ok.any():
                    delta = np.linalg.solve(A[np.ix_(ok, ok)], tgt[ok] - a[np.array(cond_i)[ok]])
                    a = a + imp[:, np.array(cond_s)[ok]] @ delta
                sims[d, j] = a
        return sims - sims.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_model():
    df = pd.read_csv(DATA_PROC / "model_data.csv", index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="Q")
    mode = pd.read_csv(OUTPUT / "05_system_mode.csv", index_col=0).iloc[:, 0].to_dict()
    data_est = ms.prepare_data(df, "2002Q4", SAMPLE_END)
    exo = ms.exog_processes(data_est, SAMPLE_START, SAMPLE_END)
    calib = dict(ms.DEFAULT_CALIB)
    calib["w"] = float(df.loc[SAMPLE_START:SAMPLE_END, "w_livres"].mean())
    return mode, calib, exo


def build_assumptions(decision_date, focus_date, fx_start, adm_years, adm_horizon=None, H=H_DEFAULT, oil_months=6,
                      oil_growth=2.0, climate="realtime", adm_profile="uniform", climate_index="oni", nowcast="focus",
                      selic_override=None, fx_override=None, pie_override=None, pie_shock_origin=0.0, verbose=False):
    """Reference-scenario paths for the vintage builder.

    adm_years: {year: administered inflation %}; adm_horizon: (quarter, %) four-quarter administered projection at
    the relevant horizon (optional). climate: 'realtime' (latest published Niño 3.4 readings, then AR(1) decay),
    'realized' (published ONI seasons, perfect foresight), 'off' (no climate effects after the current quarter) or a
    pd.Series of ONI anomalies by quarter (an explicit outlook, e.g. NOAA's official ENSO outlook).
    Own assumptions (scenario builder): selic_override {quarter: Selic % p.a.} replaces the Focus path in those
    quarters (days after the decision date); fx_override {quarter: BRL/USD level} replaces the PPP path (days after
    the Focus date); pie_override (% p.a.) replaces the Focus 12-month inflation expectation at the origin."""
    D, F = pd.Timestamp(decision_date), pd.Timestamp(focus_date)
    q0 = D.to_period("Q")
    end_q = q0 + H
    end_day = end_q.end_time.normalize()
    cut = vintage.cutoffs(D)
    snap = focus_snapshot(F)
    if pie_override is not None:
        snap["pie_12m"] = float(pie_override)
    if pie_shock_origin:
        snap["pie_12m"] = float(snap["pie_12m"]) + float(pie_shock_origin)   # shock to the Focus expectation at the origin (pp)
    selic_daily = fetch.sgs(432, name="sgs_432_selic_meta_daily", daily=True)
    selic_path = selic_path_daily(snap, D, end_day, selic_daily)
    if selic_override:
        qs = selic_path.index.to_period("Q")
        for q, v in selic_override.items():
            mask = (selic_path.index > D) & (qs == pd.Period(q, freq="Q"))
            selic_path.loc[mask] = float(v)
    ie_F = expected_selic_12m(selic_path, F)
    ptax_daily = fetch.sgs(1, start="2023-01-01", name="sgs_1_ptax_venda_daily", daily=True)
    fx_path = fx_path_daily(F, fx_start, end_day, ptax_daily)
    if fx_override:
        qs = fx_path.index.to_period("Q")
        for q, v in fx_override.items():
            mask = (fx_path.index > F) & (qs == pd.Period(q, freq="Q"))
            fx_path.loc[mask] = float(v)
    brent, steo_v = brent_path_monthly(F, end_q.asfreq("M", "end"), oil_months, oil_growth)
    adm_cut = cut["ipca"]
    if nowcast == "realized":      # months of the current quarter published since the meeting count as known
        pub = vintage._monthly(fetch.sgs(4449, name="sgs_4449_ipca_monit")).index.max()
        adm_cut = min(pub, q0.asfreq("M", "end"))
    adm_pct, adm_log = administered_path_monthly(adm_years, adm_horizon, adm_cut, end_q.asfreq("M", "end"), profile=adm_profile)
    if isinstance(climate, pd.Series):
        oni = oni_path(cut["oni"], end_q, D, mode="realtime", index=climate_index)
        ov = climate.copy(); ov.index = pd.PeriodIndex(ov.index, freq="Q")
        oni.loc[oni.index.intersection(ov.index)] = ov.loc[oni.index.intersection(ov.index)]
    else:
        oni = oni_path(cut["oni"], end_q, D, mode="realized" if climate == "realized" else "realtime", index=climate_index)
    gdp_growth = snap["gdp_annual"]
    paths = {"selic_daily": selic_path, "ptax_daily": fx_path, "brent_monthly": brent, "adm_monthly_pct": adm_pct,
             "adm_monthly_log": adm_log, "focus_monthly_ipca": snap["ipca_monthly"], "pie_F": snap["pie_12m"], "ie_F": ie_F,
             "oni_path": oni, "gdp_growth": gdp_growth, "steo_vintage": steo_v, "snapshot": snap, "fx_start": fx_start,
             "adm_years": adm_years, "adm_horizon": adm_horizon, "adm_profile": adm_profile,
             "climate": climate if isinstance(climate, str) else "explicit path", "climate_index": climate_index, "nowcast": nowcast,
             "selic_override": selic_override, "fx_override": fx_override, "pie_override": pie_override}
    if verbose:
        print("Focus %s: Selic per meeting %s; end-year %s" % (snap["selic_date"].date(), snap["selic_meetings"], snap["selic_annual"]))
        print("expected Selic 12m at F: %.2f; 12m inflation expectation: %.2f; STEO vintage %s; Brent at F %.1f" % (ie_F, snap["pie_12m"], steo_v, brent.loc[F.to_period('M')]))
    return paths


def filter_origin(mode, calib, exo, df_v, start="2002Q4", gap_obs=None, gap_sd=0.1):
    """Kalman filter through the last quarter of the vintage. `gap_obs` (Series by quarter) adds the BCB's published
    output-gap estimates as an extra observation of the gap state (standard deviation gap_sd), so that the
    projection starts from the gap the Copom conditions on."""
    data = ms.prepare_data(df_v, start, str(df_v.index[-1]))
    ss, eu = ms.state_space(mode, calib, exo, data)
    if ss is None:
        raise RuntimeError("no determinate solution: %s" % eu)
    y = ms.observation_matrix(data)
    if gap_obs is not None and len(gap_obs):
        extra = np.full((len(data), 1), np.nan)
        for q, v in gap_obs.items():
            if q in data.index:
                extra[data.index.get_loc(q), 0] = v
        y = np.hstack([y, extra])
        Z = np.vstack([ss.Z, np.zeros((1, ss.Z.shape[1]))]); Z[-1, ms.iv["h"]] = 1.0
        H = np.zeros((ss.H.shape[0] + 1, ss.H.shape[1] + 1)); H[:-1, :-1] = ss.H; H[-1, -1] = gap_sd ** 2
        ss.Z, ss.H, ss.d = Z, H, np.concatenate([ss.d, [0.0]])
    f = ss.smooth(y)                       # filter (att, Ptt) and Durbin-Koopman smoother (asm, Psm) through the origin
    return data, f["att"][-1], f["Ptt"][-1], f


# Posterior modes of the BCB (June-2024 box, Table 1) for the "BCB coefficients" variant
BCB_MODES = {"a1L": 0.24, "a1I": 0.38, "a2": 0.023, "a3": 0.011, "a4": 0.120, "a5": 0.0012, "a6": 0.0007,
             "b1": 0.85, "b2": 0.44, "b3": 0.030, "b4": 0.054, "b5": 0.84, "th1": 1.48, "th2": -0.58, "th3": 2.03,
             "phi1": 0.75, "phi2": 0.11, "phi3": 0.021, "delta": 1.90, "g_nuci": 1.87, "g_emp": 1.10, "g_caged": 0.69, "sig_h": 1.09}

# Publication dates of the Monetary Policy Reports (approximate: the Thursday after the Copom meeting of the month)
RPM_PUBLISHED = {"rpm202503": "2025-03-27", "rpm202506": "2025-06-26", "rpm202509": "2025-09-25", "rpm202512": "2025-12-18",
                 "rpm202603": "2026-03-26", "rpm202606": "2026-06-25", "rpm202609": "2026-09-24"}
_PT_Q = {"primeiro": 1, "primeiros": 1, "segundo": 2, "terceiro": 3, "quarto": 4}


# Fallback for rpm_gap_estimates() when the report texts (refs/txt/rpm*.txt, extracted by src/download_refs.py) are
# absent: the estimates read from the Monetary Policy Reports of 2025-2026 (report, quarter, gap in %).
RPM_GAP_FALLBACK = [("rpm202503", "2024Q4", 0.8), ("rpm202503", "2025Q1", 0.6), ("rpm202506", "2025Q1", 0.9), ("rpm202506", "2025Q2", 0.5),
                    ("rpm202509", "2025Q2", 0.7), ("rpm202509", "2025Q3", 0.5), ("rpm202512", "2025Q3", 0.5), ("rpm202512", "2025Q4", 0.2),
                    ("rpm202603", "2025Q4", 0.4), ("rpm202603", "2026Q1", 0.1), ("rpm202606", "2026Q1", 0.5), ("rpm202606", "2026Q2", 0.4),
                    ("rpm202609", "2026Q2", 0.5), ("rpm202609", "2026Q3", 0.4)]


def rpm_gap_estimates():
    """Output-gap estimates published in the Monetary Policy Reports ('O hiato do produto estimado para ... é de x% e
    y%'), parsed from refs/txt/rpm*.txt. Returns a DataFrame with columns report, published, quarter, gap."""
    import re
    from config import REFS
    rows = []
    for f in sorted((REFS / "txt").glob("rpm*.txt")):
        t = re.sub(r"\s+", " ", f.read_text(encoding="utf-8", errors="ignore"))
        m = re.search(r"O hiato do produto estimado para (.*?) é de (-?\d,\d)% e (-?\d,\d)%", t)
        if not m:
            continue
        txt, v1, v2 = m.group(1), float(m.group(2).replace(",", ".")), float(m.group(3).replace(",", "."))
        years = [int(y) for y in re.findall(r"20\d\d", txt)]
        words = [w for w in re.findall(r"(primeiros|primeiro|segundo|terceiro|quarto)", txt)]
        if "dois primeiros trimestres" in txt:
            qs = [pd.Period(year=years[0], quarter=1, freq="Q"), pd.Period(year=years[0], quarter=2, freq="Q")]
        elif len(words) >= 2:
            y1 = years[0]
            y2 = years[1] if len(years) > 1 else years[0]
            q1, q2 = _PT_Q[words[0]], _PT_Q[words[1]]
            if len(years) == 1 and q2 < q1:
                y2 = y1 + 1
            qs = [pd.Period(year=y1, quarter=q1, freq="Q"), pd.Period(year=y2, quarter=q2, freq="Q")]
        else:
            continue
        key = f.stem[:9]
        pub = pd.Timestamp(RPM_PUBLISHED.get(key, pd.Period(key[3:7] + "-" + key[7:9], freq="M").to_timestamp(how="end")))
        for q, v in zip(qs, (v1, v2)):
            rows.append({"report": key, "published": pub, "quarter": q, "gap": v})
    if not rows:
        for key, q, v in RPM_GAP_FALLBACK:
            pub = pd.Timestamp(RPM_PUBLISHED.get(key, pd.Period(key[3:7] + "-" + key[7:9], freq="M").to_timestamp(how="end")))
            rows.append({"report": key, "published": pub, "quarter": pd.Period(q, freq="Q"), "gap": v})
    return pd.DataFrame(rows)


def rpm_gap_at(decision_date):
    """Gap estimates known at the decision date: for each quarter, the value from the latest report published before D."""
    g = rpm_gap_estimates()
    g = g[g["published"] <= pd.Timestamp(decision_date)].sort_values("published")
    return g.groupby("quarter")["gap"].last()


def project(decision_date, focus_date, fx_start, adm_years, adm_horizon=None, H=H_DEFAULT, neutral_rate=5.0,
            n_draws=0, seed=0, model=None, verbose=False, climate="realtime", adm_profile="uniform",
            climate_index="oni", nowcast="focus", coefficients="estimated", gap_anchor=None,
            selic_override=None, fx_override=None, pie_override=None,
            adm_source="input", adm_theta="bcb", flag_december=None, adm_overrides=None, feedback_iter=4, adm_handover=0,
            pie_shock_origin=0.0, pie_shock_path=None):
    """Full conditional projection at an information date. Returns a dict with the quarterly results table, the
    calendar-year and four-quarter summaries, the paths used and diagnostics.

    coefficients: 'estimated' (posterior mode of notebook 05) or 'bcb' (modes of the June-2024 box, Table 1).
    gap_anchor: None, or 'rpm' to anchor the filtered output gap on the estimates published in the Monetary Policy
    Reports available at the decision date (the gap the Copom conditions its projections on).
    climate_index: 'oni' or 'roni'. nowcast: 'focus' (real time) or 'realized' (published data, for diagnostics).
    adm_source: 'input' (administered prices from the annual figures adm_years/adm_horizon: the Copom's projections
    or the Focus medians) or 'model' (the BCB's administered-price model, administered.py, with the free-price
    feedback: adm_theta 'bcb' (the box's calibration) or 'estimated' (OLS on the same regressors, re-estimated at
    each run); flag_december {year: flag} is the tariff-flag assumption; adm_overrides {'R_plano': {year: %},
    'reaj_farm': {year: %}} fix the health-plan / pharmaceutical readjustments of years not yet observed).
    The current quarter is always nowcast with the Focus headline medians and the annual administered figures
    (the BCB uses specialists' short-term projections there); adm_handover (quarters after the current one that
    keep the annual-figure path before the model takes over, as the BCB's specialists cover 4-5 quarters) blends
    the two sources.
    Shocks to inflation expectations, relative to the Focus survey: pie_shock_origin (pp) shifts the 12-month
    expectation observed at the origin (the model's expectations equation then drives it); pie_shock_path
    {quarter: pp} keeps the model's expectations that many points above (below) their baseline path in those
    projection quarters, imposed with anticipated expectation shocks so that the model-consistent terms and the
    Phillips curve respond to them."""
    mode, calib, exo = model or load_model()
    if coefficients == "bcb":
        mode = dict(mode); mode.update(BCB_MODES)
    if adm_source == "model" and not adm_years:        # provisional annual figures (Focus medians) for the first pass
        snap0 = focus_snapshot(focus_date)
        y0 = pd.Timestamp(decision_date).year
        adm_years = {y: snap0["adm_annual"][y] for y in [y0, y0 + 1, y0 + 2] if y in snap0["adm_annual"]}
        adm_horizon = None
    paths = build_assumptions(decision_date, focus_date, fx_start, adm_years, adm_horizon, H, climate=climate,
                              adm_profile=adm_profile, climate_index=climate_index, nowcast=nowcast,
                              selic_override=selic_override, fx_override=fx_override, pie_override=pie_override,
                              pie_shock_origin=pie_shock_origin, verbose=verbose)
    adm_run = None
    if adm_source == "model":
        import adm_projection as admp
        cut = vintage.cutoffs(decision_date)
        ov = {k: {int(y): 100 * np.log1p(v / 100.0) for y, v in d.items()} for k, d in (adm_overrides or {}).items()}
        adm_run = admp.run(decision_date, cut["ipca"], paths, H, theta=adm_theta, flag_december=flag_december, overrides=ov)
    vint = vintage.build_vintage(decision_date, focus_date, paths, neutral_rate=neutral_rate, horizon=H, verbose=verbose,
                                 climate_index=climate_index, nowcast=nowcast)
    df_v, q0 = vint["df"], vint["q0"]
    gap_obs = rpm_gap_at(decision_date) if gap_anchor == "rpm" else None
    data, a0, P0, filt = filter_origin(mode, calib, exo, df_v, gap_obs=gap_obs)
    # conditioning paths in model units
    ex = vint["exog_paths"]
    seasA = vint["seasonal"]["pi_monit"]
    seasL = vint["seasonal"]["pi_livres"]
    w = vint["w_last"]

    def seas_last(s, qtr):
        hist = s[[p.quarter == qtr for p in s.index]].dropna()
        return float(hist.iloc[-1])
    seasL_path = pd.Series({q: seas_last(seasL, q.quarter) for q in ex.index})
    adm_log = paths["adm_monthly_log"]
    piA_input = pd.Series({q: float(adm_log[[m for m in adm_log.index if m.asfreq("Q") == q]].sum()) for q in ex.index})
    pinned = [q for q in ex.index if (q - q0).n <= adm_handover] if adm_source == "model" else list(ex.index)

    def blend(model_path):
        p = model_path.reindex(ex.index).copy()
        p.loc[pinned] = piA_input.loc[pinned]
        return p
    if adm_source == "model":
        piA_nsa = blend(adm_run["adm"])
        piA_sa = pd.Series({q: piA_nsa[q] - (0.0 if q in pinned and paths.get("adm_profile", "uniform") == "uniform" else seas_last(seasA, q.quarter)) for q in ex.index})
    else:
        piA_nsa = piA_input.copy()
        if paths.get("adm_profile", "uniform") == "uniform":
            piA_sa = piA_nsa.copy()                 # a flat monthly path carries no seasonality
        else:
            piA_sa = pd.Series({q: piA_nsa[q] - seas_last(seasA, q.quarter) for q in ex.index})
    cond = pd.DataFrame({"i": ex["i"], "de": ex["de"], "piA": piA_sa, "pistar_usd": ex["pistar_usd"], "el": ex["el"], "la": ex["la"]})
    if isinstance(climate, str) and climate == "off":
        cond["el"] = 0.0
        cond["la"] = 0.0
    pr = Projector(mode, calib, exo, H=H)
    shock_path = None
    if pie_shock_path:
        shock_path = pd.Series({pd.Period(k, freq="Q"): float(v) for k, v in pie_shock_path.items()}).reindex(ex.index).fillna(0.0)
        pr_pie = Projector(mode, calib, exo, H=H, cond=dict(COND, pie="e_pie"))

    def solve_paths(cond_df):
        """Central path; with an expectations shock path, the model's expectations are conditioned on their
        baseline path plus the shock (anticipated expectation shocks)."""
        base_path, base_nu = pr.deterministic(a0, cond_df)
        if shock_path is None or not (shock_path != 0).any():
            return base_path, base_nu, pr
        c2 = cond_df.copy()
        c2["pie"] = base_path["pie"].values + shock_path.values
        p2, nu2 = pr_pie.deterministic(a0, c2)
        return p2, nu2, pr_pie
    feedback_log = []
    for it in range(feedback_iter if adm_source == "model" else 1):
        path, nu, pr_used = solve_paths(cond)
        path.index = ex.index
        if adm_source != "model":
            break
        # feedback: the headline IPCA implied by free and administered prices indexes the administered items
        headline = w * (path["piL"] + seasL_path) + (1 - w) * piA_nsa
        hist_head = df_v["pi_ipca"].loc[adm_run["origin"] + 1:q0]
        headline_full = pd.concat([hist_head, headline])
        adm_run = admp.run(decision_date, cut["ipca"], paths, H, theta=adm_run["theta"], flag_december=flag_december,
                           headline=headline_full, overrides=ov)
        new_nsa = blend(adm_run["adm"])
        change = float((new_nsa - piA_nsa).abs().max())
        feedback_log.append(change)
        piA_nsa = new_nsa
        piA_sa = pd.Series({q: piA_nsa[q] - (0.0 if q in pinned and paths.get("adm_profile", "uniform") == "uniform" else seas_last(seasA, q.quarter)) for q in ex.index})
        cond["piA"] = piA_sa
        if change < 0.005:
            path, nu, pr_used = solve_paths(cond)
            path.index = ex.index
            break
    if shock_path is not None and (shock_path != 0).any():
        cond = cond.copy(); cond["pie"] = path["pie"].values   # for the bands, expectations stay on the shocked path
    # calendar inflation
    res = pd.DataFrame(index=ex.index)
    res["piL_sa"] = path["piL"]
    res["piL"] = path["piL"] + seasL_path
    res["piA"] = piA_nsa
    res["pi"] = w * res["piL"] + (1 - w) * res["piA"]
    res["pi_sa_model"] = path["pi"]
    res["h"] = path["h"]; res["pie"] = path["pie"]; res["i"] = path["i"]; res["de"] = path["de"]; res["ie"] = path["ie"]
    res["selic_pct"] = 100 * np.expm1(res["i"] / 100)
    res["source"] = "projection"
    hist = pd.DataFrame({"piL": df_v["pi_livres"], "piA": df_v["pi_monit"], "pi": df_v["pi_ipca"], "piL_sa": df_v["pi_livres_sa"],
                         "h": pd.Series(filt["att"][:, ms.iv["h"]], index=data.index).reindex(df_v.index),
                         "pie": df_v["pi_e"], "i": df_v["i"], "de": df_v["de"], "ie": df_v["i_e"]})
    hist["selic_pct"] = df_v["selic"]
    hist["source"] = "history"
    hist.loc[q0, "source"] = "nowcast"
    full = pd.concat([hist.loc[q0 - 8:], res], sort=False)
    for c in ["pi", "piL", "piA"]:
        full[c + "_4q"] = 100 * np.expm1(full[c].rolling(4).sum() / 100)
    years = {}
    for y in sorted({p.year for p in full.index if p.year >= q0.year}):
        sel = full[[p.year == y for p in full.index]]
        if len(sel) == 4:
            years[y] = {c: float(100 * np.expm1(sel[c].sum() / 100)) for c in ["pi", "piL", "piA"]}
    out = {"table": full, "years": years, "paths": paths, "vintage": vint, "a0": a0, "P0": P0, "news": nu, "q0": q0,
           "data": data, "cond": cond, "filter": filt, "model": (mode, calib, exo), "gap_obs": gap_obs,
           "settings": {"neutral_rate": neutral_rate, "climate": climate if isinstance(climate, str) else "explicit path",
                        "climate_index": climate_index, "nowcast": nowcast, "coefficients": coefficients, "gap_anchor": gap_anchor,
                        "adm_profile": adm_profile, "adm_source": adm_source, "adm_theta": adm_theta if adm_source == "model" else None,
                        "flag_december": flag_december, "adm_handover": adm_handover if adm_source == "model" else None,
                        "pie_shock_origin": pie_shock_origin, "pie_shock_path": pie_shock_path},
           "adm_model": adm_run, "feedback_log": feedback_log}
    # output gap: history (smoothed and filtered estimates, indicator-implied gaps, BCB published estimates) and projection
    ih = ms.iv["h"]
    gap = pd.DataFrame(index=pd.period_range(pd.Period("2003Q1", freq="Q"), ex.index[-1], freq="Q"))
    gap["smoothed"] = pd.Series(filt["asm"][:, ih], index=data.index).reindex(gap.index)
    gap["smoothed_sd"] = pd.Series(np.sqrt(np.maximum(filt["Psm"][:, ih, ih], 0)), index=data.index).reindex(gap.index)
    gap["filtered"] = pd.Series(filt["att"][:, ih], index=data.index).reindex(gap.index)
    gap["fpib"] = df_v["fpib"].reindex(gap.index)
    gap["nuci_implied"] = (df_v["fnuci"] / mode["g_nuci"]).reindex(gap.index)
    gap["emp_implied"] = (df_v["femp"] / mode["g_emp"]).shift(-1).reindex(gap.index)      # femp_t loads on h_{t-1}
    gap["caged_implied"] = (df_v["fcaged"] / mode["g_caged"]).shift(-1).reindex(gap.index)
    try:
        bcb_gap = rpm_gap_at(decision_date)
        gap["bcb"] = bcb_gap.reindex(gap.index)
    except Exception:  # noqa: BLE001
        gap["bcb"] = np.nan
    gap["gap"] = path["h"].reindex(gap.index)
    gap["source"] = "history"
    gap.loc[ex.index, "source"] = "projection"
    gap.loc[q0, "source"] = "nowcast"
    out["gap"] = gap
    if n_draws:
        # uncertainty around the central path: shocks other than the conditioned ones and the state uncertainty at
        # the origin; the administered path is kept fixed, so the band applies to free prices and, through the
        # identity, to the headline IPCA (four-quarter accumulation)
        sims = pr_used.simulate(a0, P0, cond, n_draws=n_draws, seed=seed)
        dev = sims[:, :, ms.iv["piL"]]                       # deviations of free-price inflation (log-% per quarter)
        pre = full.loc[:q0, "piL"].values[-3:]                # last three quarters before the projection (history/nowcast)
        piA_4q = full.loc[res.index, "piA"].values
        piA_pre = full.loc[:q0, "piA"].values[-3:]
        acc = np.zeros((sims.shape[0], H))
        for d in range(sims.shape[0]):
            sL = pd.Series(np.concatenate([pre, res["piL"].values + dev[d]])).rolling(4).sum().values[3:]
            sA = pd.Series(np.concatenate([piA_pre, piA_4q])).rolling(4).sum().values[3:]
            acc[d] = 100 * np.expm1((w * sL + (1 - w) * sA) / 100)
        out["bands"] = pd.DataFrame(np.percentile(acc, [5, 25, 50, 75, 95], axis=0).T, index=res.index,
                                    columns=["p5", "p25", "p50", "p75", "p95"])
        gh = res["h"].values[None, :] + sims[:, :, ms.iv["h"]]
        gb = pd.DataFrame(np.percentile(gh, [5, 25, 50, 75, 95], axis=0).T, index=res.index, columns=["p5", "p25", "p50", "p75", "p95"])
        for c in gb.columns:
            out["gap"].loc[gb.index, c] = gb[c].values
    return out


def copom_table(out, years=None, horizon=None):
    """Numbers in the format of the Copom's Table 1 (IPCA, free, administered; calendar years and the four-quarter
    accumulation at the relevant horizon)."""
    rows = {}
    for y, v in out["years"].items():
        if years is None or y in years:
            rows[str(y)] = {"IPCA": v["pi"], "IPCA livres": v["piL"], "IPCA administrados": v["piA"]}
    if horizon is not None:
        hq = pd.Period(horizon, freq="Q")
        t = out["table"]
        if hq in t.index:
            rows[str(hq)] = {"IPCA": t.loc[hq, "pi_4q"], "IPCA livres": t.loc[hq, "piL_4q"], "IPCA administrados": t.loc[hq, "piA_4q"]}
    return pd.DataFrame(rows).T
