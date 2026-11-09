"""
Layer B: the full aggregate small-scale model (BCB, RI June 2024) as a linear rational-expectations system,
solved with gensys and cast in state-space form for Bayesian estimation with the Kalman filter.

Equations (quarterly; inflation in log-% per quarter unless noted; interest rates in log-% p.a.):

 (1) piL_t = a1L piL_{t-1} + a1I (pi_{t-1}+..+pi_{t-4})/4 + (1-a1L-a1I) pie_t/4 + a2 pistar_t + a3 dehat_{t-1}
             + a4 h_t + a5 el_t + a6 la_t + e_piL
 (id) pi_t = w piL_t + (1-w) piA_t                                        (piA exogenous AR(1), observed)
 (2) h_t = b1 h_{t-1} - (b2/4) rhat_{t-1} - b3 rp_t + b4 hstar_t + s_t + e_crisis
     rhat_t = ie_t - pie_t - rrtrend_t - rrIS_t ;  s_t = b5 s_{t-1} + e_h ;  rrIS_t = rrIS_{t-1} + e_rrIS
 (2.2) ie_t = (0.5 i_t + E i_{t+1} + E i_{t+2} + E i_{t+3} + 0.5 E i_{t+4})/4 + e_ie     (Focus expected Selic)
 (3) i_t = th1 i_{t-1} + th2 i_{t-2} + (1-th1-th2)[rrtrend_t + rrTAY_t + meta_t + th3 (pie_t - meta_t)] + e_i
     rrTAY_t = rrTAY_{t-1} + e_rrTAY
 (4) de_t = (meta_t - pistar_ss)/4 - delta (idif_t - idif_{t-1}) + e_e ;  idif_t = i_t - f_t ;  dehat_t = de_t - (meta_t - 2)/4
 (5) pie_t = phi1 pie_{t-1} + phi2 (E pi_{t+1}+..+E pi_{t+4}) + phi3 (pi_{t-1}+..+pi_{t-4}) + (1-phi1-phi2-phi3) meta_t + e_pie
 (6-9) fpib_t = h_t + sig e ; fnuci_t = gn (h_t + sig e) ; femp_t = ge (h_{t-1} + sig e) ; fcaged_t = gc (h_{t-1} + sig e)

Exogenous observed processes (AR(1) or random walk, parameters pre-estimated by OLS): piA, pistar, hstar, rp,
el, la, f (= log(1 + fed funds + EMBI)), meta (RW), rrtrend (RW).
"""
import numpy as np
import pandas as pd

from re_solver import gensys
from ssm import StateSpace
from bayes import Prior
from config import CRISIS_SD_MULT, HV_SD_MULT, SD_RR_RW

PI_STAR_SS = 2.0

ENDO = ["piL", "pi", "pie", "i", "de", "dehat", "h", "s", "rrIS", "rrTAY", "ie", "idif", "pistar",
        "xpi1", "xpi2", "xpi3", "xpi4", "xi1", "xi2", "xi3", "xi4",
        "piL_l1", "pi_l1", "pi_l2", "pi_l3", "pi_l4", "i_l1", "i_l2", "h_l1", "dehat_l1", "idif_l1", "pie_l1"]
# pistar (IC-Br inflation in BRL, deviation from target) = pistar_usd (exogenous, in USD) + exchange-rate change:
# the box's footnote 4 states that the imported inflation terms "incorporam a variação cambial".
EXOG = ["piA", "pistar_usd", "hstar", "rp", "el", "la", "f", "meta", "rrtrend"]
VARS = ENDO + EXOG
SHOCKS = ["e_piL", "e_pie", "e_i", "e_e", "e_h", "e_crisis", "e_rrIS", "e_rrTAY", "e_ie"] + ["e_" + x for x in EXOG]
ETAS = ["eta_pi%d" % k for k in range(1, 5)] + ["eta_i%d" % k for k in range(1, 5)]
OBS_ENDO = ["piL", "pie", "i", "de", "ie"]
OBS_GAP = ["fpib", "fnuci", "femp", "fcaged"]
OBS_EXOG = EXOG
DATA_COLS = {"piL": "pi_livres_sa", "pie": "pi_e", "i": "i", "de": "de", "ie": "i_e", "piA": "pi_monit_sa",
             "pistar_usd": "pistar_usd", "hstar": "h_star", "rp": "rp_tilde", "el": "clim_el", "la": "clim_la",
             "f": "f_ext", "meta": "pi_meta", "rrtrend": "rr_trend"}

iv = {v: k for k, v in enumerate(VARS)}
ish = {v: k for k, v in enumerate(SHOCKS)}
iet = {v: k for k, v in enumerate(ETAS)}


def priors_system():
    p = {
        "a1L": Prior("uniform", 0, 1), "a1I": Prior("uniform", 0, 1), "a2": Prior("uniform", 0, 1),
        "a3": Prior("uniform", 0, 1), "a4": Prior("uniform", 0, 1), "a5": Prior("uniform", 0, 0.01), "a6": Prior("uniform", 0, 0.01),
        "b1": Prior("uniform", 0, 1), "b2": Prior("uniform", 0, 1), "b3": Prior("beta", mean=0.03, sd=0.002),
        "b4": Prior("uniform", 0, 1), "b5": Prior("uniform", 0, 1),
        "th1": Prior("uniform", 0, 2), "th2": Prior("uniform", -1, 1), "th3": Prior("uniform", 0, 4),
        "phi1": Prior("uniform", 0, 1), "phi2": Prior("uniform", 0, 1), "phi3": Prior("uniform", 0, 1),
        "delta": Prior("uniform", 0, 10),
        "g_nuci": Prior("uniform", 0, 3), "g_emp": Prior("uniform", 0, 3), "g_caged": Prior("uniform", 0, 3),
        "sig_h": Prior("uniform", 0, 2),
        "sd_piL": Prior("invgamma", mean=0.5, sd=1.0), "sd_pie": Prior("invgamma", mean=0.3, sd=1.0),
        "sd_i": Prior("invgamma", mean=0.5, sd=1.0), "sd_e": Prior("invgamma", mean=5.0, sd=5.0),
        "sd_h": Prior("invgamma", mean=0.5, sd=1.0), "sd_ie": Prior("invgamma", mean=0.5, sd=1.0),
    }
    return p


DEFAULT_CALIB = {"sd_rrIS": SD_RR_RW, "sd_rrTAY": SD_RR_RW, "crisis_mult": CRISIS_SD_MULT, "hv_mult": HV_SD_MULT, "w": 0.73}


def exog_processes(df, start, end):
    """OLS AR(1) parameters (rho, const, sd) for the exogenous observed processes; meta and rrtrend as random walks."""
    out = {}
    d = df.loc[start:end]
    for v in EXOG:
        x = d[DATA_COLS[v]].dropna()
        if v in ("meta", "rrtrend"):
            out[v] = {"rho": 1.0, "c": 0.0, "sd": max(float(x.diff().std()), 1e-3)}
            continue
        X = np.column_stack([np.ones(len(x) - 1), x.values[:-1]])
        b = np.linalg.lstsq(X, x.values[1:], rcond=None)[0]
        e = x.values[1:] - X @ b
        rho = float(np.clip(b[1], -0.98, 0.98))
        out[v] = {"rho": rho, "c": float(b[0]), "sd": float(e.std())}
    return out


def build_system(p, calib, exo):
    """Return gensys matrices for parameters p (dict), calibration dict and exogenous-process dict."""
    n, ns, ne = len(VARS), len(SHOCKS), len(ETAS)
    G0, G1, C, Psi, Pi = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n), np.zeros((n, ns)), np.zeros((n, ne))
    row = [0]

    def eq(lhs, rhs=None, const=0.0, shocks=None, etas=None):
        r = row[0]
        for v, c in (lhs or {}).items():
            G0[r, iv[v]] += c
        for v, c in (rhs or {}).items():
            G1[r, iv[v]] += c
        C[r] = const
        for s, c in (shocks or {}).items():
            Psi[r, ish[s]] += c
        for s, c in (etas or {}).items():
            Pi[r, iet[s]] += c
        row[0] += 1

    w = calib["w"]
    # (1) Phillips curve
    eq({"piL": 1.0, "piL_l1": -p["a1L"], "pi_l1": -p["a1I"] / 4, "pi_l2": -p["a1I"] / 4, "pi_l3": -p["a1I"] / 4, "pi_l4": -p["a1I"] / 4,
        "pie": -(1 - p["a1L"] - p["a1I"]) / 4, "pistar": -p["a2"], "dehat_l1": -p["a3"], "h": -p["a4"], "el": -p["a5"], "la": -p["a6"]},
       shocks={"e_piL": 1.0})
    # identity IPCA
    eq({"pi": 1.0, "piL": -w, "piA": -(1 - w)})
    # (2) IS curve: h = b1 h_l1 - (b2/4)(ie_l1 - pie_l1 - rrtrend_l1 - rrIS_l1) - b3 rp + b4 hstar + s + e_crisis
    #     with s_t = b5 s_{t-1} + e_h  ->  write h_t using s_t as a variable
    eq({"h": 1.0, "rp": p["b3"], "hstar": -p["b4"], "s": -1.0},
       rhs={"h": p["b1"], "ie": -p["b2"] / 4, "pie": p["b2"] / 4, "rrtrend": p["b2"] / 4, "rrIS": p["b2"] / 4},
       shocks={"e_crisis": 1.0})
    eq({"s": 1.0}, rhs={"s": p["b5"]}, shocks={"e_h": 1.0})
    eq({"rrIS": 1.0}, rhs={"rrIS": 1.0}, shocks={"e_rrIS": 1.0})
    # (2.2) Focus expected Selic over the next 4 quarters (trapezoid of expected quarterly rates)
    eq({"ie": 1.0, "i": -0.5 / 4, "xi1": -1.0 / 4, "xi2": -1.0 / 4, "xi3": -1.0 / 4, "xi4": -0.5 / 4}, shocks={"e_ie": 1.0})
    # (3) Taylor rule
    k = 1 - p["th1"] - p["th2"]
    eq({"i": 1.0, "i_l1": -p["th1"], "i_l2": -p["th2"], "rrtrend": -k, "rrTAY": -k, "meta": -k * (1 - p["th3"]), "pie": -k * p["th3"]},
       shocks={"e_i": 1.0})
    eq({"rrTAY": 1.0}, rhs={"rrTAY": 1.0}, shocks={"e_rrTAY": 1.0})
    # (4) UIP: de = (meta - pistar_ss)/4 - delta (idif - idif_l1) + e_e ; idif = i - f ; dehat = de - (meta-2)/4
    eq({"de": 1.0, "meta": -0.25, "idif": p["delta"], "idif_l1": -p["delta"]}, const=-PI_STAR_SS / 4, shocks={"e_e": 1.0})
    eq({"idif": 1.0, "i": -1.0, "f": 1.0})
    eq({"dehat": 1.0, "de": -1.0, "meta": 0.25}, const=-PI_STAR_SS / 4)
    # imported commodity inflation in BRL = USD commodity inflation + exchange-rate change
    eq({"pistar": 1.0, "pistar_usd": -1.0, "de": -1.0})
    # (5) expectations
    kphi = 1 - p["phi1"] - p["phi2"] - p["phi3"]
    eq({"pie": 1.0, "pie_l1": -p["phi1"], "xpi1": -p["phi2"], "xpi2": -p["phi2"], "xpi3": -p["phi2"], "xpi4": -p["phi2"],
        "pi_l1": -p["phi3"], "pi_l2": -p["phi3"], "pi_l3": -p["phi3"], "pi_l4": -p["phi3"], "meta": -kphi}, shocks={"e_pie": 1.0})
    # expectation auxiliaries: pi_t = xpi1_{t-1} + eta ; xpi1_t = xpi2_{t-1} + eta ; ...
    eq({"pi": 1.0}, rhs={"xpi1": 1.0}, etas={"eta_pi1": 1.0})
    eq({"xpi1": 1.0}, rhs={"xpi2": 1.0}, etas={"eta_pi2": 1.0})
    eq({"xpi2": 1.0}, rhs={"xpi3": 1.0}, etas={"eta_pi3": 1.0})
    eq({"xpi3": 1.0}, rhs={"xpi4": 1.0}, etas={"eta_pi4": 1.0})
    eq({"i": 1.0}, rhs={"xi1": 1.0}, etas={"eta_i1": 1.0})
    eq({"xi1": 1.0}, rhs={"xi2": 1.0}, etas={"eta_i2": 1.0})
    eq({"xi2": 1.0}, rhs={"xi3": 1.0}, etas={"eta_i3": 1.0})
    eq({"xi3": 1.0}, rhs={"xi4": 1.0}, etas={"eta_i4": 1.0})
    # lag definitions
    for cur, lag in [("piL", "piL_l1"), ("pi", "pi_l1"), ("pi_l1", "pi_l2"), ("pi_l2", "pi_l3"), ("pi_l3", "pi_l4"),
                     ("i", "i_l1"), ("i_l1", "i_l2"), ("h", "h_l1"), ("dehat", "dehat_l1"), ("idif", "idif_l1"), ("pie", "pie_l1")]:
        eq({lag: 1.0}, rhs={cur: 1.0})
    # exogenous processes
    for v in EXOG:
        e = exo[v]
        eq({v: 1.0}, rhs={v: e["rho"]}, const=e["c"], shocks={"e_" + v: 1.0})
    assert row[0] == n, (row[0], n)
    return G0, G1, C, Psi, Pi


def solve(p, calib, exo):
    G0, G1, C, Psi, Pi = build_system(p, calib, exo)
    T, c, impact, eu = gensys(G0, G1, C, Psi, Pi)
    return T, c, impact, eu


def shock_sd_path(p, calib, exo, data):
    """Time-varying shock standard deviations (n_t x n_shocks)."""
    n = len(data)
    sd = np.zeros((n, len(SHOCKS)))
    base = {"e_piL": p["sd_piL"], "e_pie": p["sd_pie"], "e_i": p["sd_i"], "e_e": p["sd_e"], "e_h": p["sd_h"],
            "e_crisis": 1e-6, "e_rrIS": calib["sd_rrIS"], "e_rrTAY": calib["sd_rrTAY"], "e_ie": p["sd_ie"]}
    for v in EXOG:
        base["e_" + v] = exo[v]["sd"]
    for s, v in base.items():
        sd[:, ish[s]] = v
    crisis = (data["d_crise2008"].values + data["d_crise2020"].values) > 0
    sd[crisis, ish["e_crisis"]] = calib["crisis_mult"] * p["sd_h"]
    hv = data["d_phillips_hv"].values > 0
    sd[hv, ish["e_piL"]] = calib["hv_mult"] * p["sd_piL"]
    return sd


def measurement(p):
    """Z (n_obs x n_vars), d, H for observables [OBS_ENDO + OBS_GAP + OBS_EXOG]."""
    obs = OBS_ENDO + OBS_GAP + OBS_EXOG
    Z = np.zeros((len(obs), len(VARS)))
    H = np.zeros((len(obs), len(obs)))
    gn, ge, gc, sig = p["g_nuci"], p["g_emp"], p["g_caged"], p["sig_h"]
    for k, o in enumerate(obs):
        if o in VARS:
            Z[k, iv[o]] = 1.0
        elif o == "fpib":
            Z[k, iv["h"]] = 1.0; H[k, k] = sig ** 2
        elif o == "fnuci":
            Z[k, iv["h"]] = gn; H[k, k] = (gn * sig) ** 2
        elif o == "femp":
            Z[k, iv["h_l1"]] = ge; H[k, k] = (ge * sig) ** 2
        elif o == "fcaged":
            Z[k, iv["h_l1"]] = gc; H[k, k] = (gc * sig) ** 2
    return obs, Z, np.zeros(len(obs)), H


def observation_matrix(data):
    """Observed data matrix (n_t x n_obs) in the order of measurement()."""
    obs = OBS_ENDO + OBS_GAP + OBS_EXOG
    cols = [DATA_COLS.get(o, o) for o in obs]
    return data[cols].values.astype(float)


def state_space(p, calib, exo, data, presample=4):
    T, c, impact, eu = solve(p, calib, exo)
    if eu != [1, 1]:
        return None, eu
    sd = shock_sd_path(p, calib, exo, data)
    n = len(data)
    Q = np.zeros((n, len(SHOCKS), len(SHOCKS)))
    for t in range(n):
        Q[t] = np.diag(sd[t] ** 2)
    obs, Z, d, H = measurement(p)
    # initial state: observed exogenous at their first values, others zero; large variance
    a1 = np.zeros(len(VARS))
    y0 = data.iloc[0]
    for v in EXOG:
        a1[iv[v]] = y0[DATA_COLS[v]] if np.isfinite(y0[DATA_COLS[v]]) else 0.0
    for v in ["piL", "pie", "i", "ie"]:
        a1[iv[v]] = y0[DATA_COLS[v]] if np.isfinite(y0[DATA_COLS[v]]) else 0.0
    a1[iv["pi"]] = y0["pi_ipca_sa"]
    for v in ["piL_l1", "pi_l1", "pi_l2", "pi_l3", "pi_l4", "i_l1", "i_l2", "pie_l1"]:
        base = v.split("_l")[0]
        a1[iv[v]] = a1[iv[base]]
    a1[iv["idif"]] = a1[iv["i"]] - a1[iv["f"]]
    a1[iv["idif_l1"]] = a1[iv["idif"]]
    a1[iv["de"]] = y0["de"] if np.isfinite(y0["de"]) else 0.0
    a1[iv["pistar"]] = a1[iv["pistar_usd"]] + a1[iv["de"]]
    for v in ["xi1", "xi2", "xi3", "xi4"]:
        a1[iv[v]] = a1[iv["i"]]
    for v in ["xpi1", "xpi2", "xpi3", "xpi4"]:
        a1[iv[v]] = a1[iv["pi"]]
    P1 = np.eye(len(VARS)) * 4.0
    ss = StateSpace(T, c, impact, Q, Z, d, H, a1, P1)
    return ss, eu


def loglik_factory(data, calib, exo, presample=4):
    y = observation_matrix(data)

    def ll(p):
        try:
            ss, eu = state_space(p, calib, exo, data)
            if ss is None:
                return -np.inf
            return float(ss.loglik(y, skip=presample))
        except Exception:  # noqa: BLE001
            return -np.inf
    return ll


def prepare_data(df, start="2002Q4", end="2023Q4"):
    d = df.copy()
    d["f_ext"] = 100 * np.log1p((d["i_star"] + d["cds_proxy"]) / 100.0)
    d["pistar_usd"] = d["pi_icbr_hat"] - d["de"]      # IC-Br inflation in USD terms (deviation from target)
    return d.loc[start:end]


def impulse_responses(p, calib, exo, shock, horizon=16, size=1.0):
    T, c, impact, eu = solve(p, calib, exo)
    n = T.shape[0]
    y = np.zeros((horizon + 1, n))
    y[0] = impact[:, ish[shock]] * size
    for t in range(1, horizon + 1):
        y[t] = T @ y[t - 1]
    return pd.DataFrame(y, columns=VARS)


def conditional_path(p, calib, exo, var, path, shock, horizon=16, extra=None):
    """Deviation-from-baseline simulation in which variable `var` is forced to follow `path` (list of values,
    one per period from t=0) by choosing the shock `shock` period by period; afterwards the model runs freely.
    `extra` optionally adds one-off shocks {t: {shock_name: value}}."""
    T, c, impact, eu = solve(p, calib, exo)
    n = T.shape[0]
    y = np.zeros(n)
    rows = []
    for t in range(horizon + 1):
        y = T @ y if t > 0 else np.zeros(n)
        for s, val in (extra or {}).get(t, {}).items():
            y = y + impact[:, ish[s]] * val
        if t < len(path):
            need = path[t] - y[iv[var]]
            y = y + impact[:, ish[shock]] * (need / impact[iv[var], ish[shock]])
        rows.append(y.copy())
    return pd.DataFrame(rows, columns=VARS)
