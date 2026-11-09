"""
Layer A: multivariate unobserved-components model for the output gap (IS curve + four activity indicators),
estimated by Bayesian methods with the BCB June-2024 priors. This reproduces the output-gap block of the
small-scale model (equations 2, 2.1-2.4 and 6-9 of the box) without the nominal block, and delivers
(i) IS-curve coefficients and (ii) a smoothed gap used in the single-equation estimation of the other curves.

State vector:  a_t = [h_t, h_{t-1}, s_t, rrtilde_t]
    h_t        = b1 h_{t-1} - (b2/4) rhat_{t-1} - b3 rp_t + b4 hstar_t + s_t + e_crisis_t
    rhat_{t-1} = r_focus_{t-1} - rr_trend_{t-1} - rrtilde_{t-1}
    s_t        = b5 s_{t-1} + e_h_t
    rrtilde_t  = rrtilde_{t-1} + e_rr_t                      (s.d. calibrated)
Observations:  fpib_t = h_t + sig*e1;   fnuci_t = g_n (h_t + sig*e2);   femp_t = g_e (h_{t-1} + sig*e3);
               fcaged_t = g_c (h_{t-1} + sig*e4)          [BCB writes (f_x/gamma_x) = h + sig*e]
"""
import numpy as np
import pandas as pd

from ssm import StateSpace
from bayes import Prior, Posterior
from config import CRISIS_SD_MULT, SD_RR_RW

OBS = ["fpib", "fnuci", "femp", "fcaged"]


def priors_gap(sd_rr=SD_RR_RW, sd_crisis_mult=CRISIS_SD_MULT):
    return {
        "b1": Prior("uniform", 0, 1, name="IS autoregressive"),
        "b2": Prior("uniform", 0, 1, name="real interest rate gap"),
        "b3": Prior("beta", mean=0.03, sd=0.002, name="primary result"),
        "b4": Prior("uniform", 0, 1, name="world gap"),
        "b5": Prior("uniform", 0, 1, name="AR of IS shock"),
        "g_nuci": Prior("uniform", 0, 3, name="loading NUCI"),
        "g_emp": Prior("uniform", 0, 3, name="loading unemployment"),
        "g_caged": Prior("uniform", 0, 3, name="loading Caged"),
        "sig_h": Prior("uniform", 0, 2, name="measurement error s.d."),
        "sd_is": Prior("invgamma", mean=0.5, sd=1.0, name="IS shock s.d."),
        "sd_rr": Prior("fixed", mean=sd_rr, name="neutral-rate RW s.d. (calibrated)"),
        "crisis_mult": Prior("fixed", mean=sd_crisis_mult, name="crisis shock s.d. multiple (calibrated)"),
    }


def prepare(df, start="2003Q4", end="2023Q4"):
    """Slice the model dataset and pre-compute the lagged exogenous regressors used by the gap model."""
    d = df.copy()
    d["rhat_obs_l1"] = (d["r_focus"] - d["rr_trend"]).shift(1)   # observed part of the real-rate gap, lagged
    return d.loc[start:end]


def build_ss(p, data):
    """StateSpace for the gap model given parameter dict p and a DataFrame `data` (quarterly, from `prepare`)
    containing rhat_obs_l1, rp_tilde, h_star, d_crise2008, d_crise2020 and the observables."""
    n = len(data)
    b1, b2, b3, b4, b5 = p["b1"], p["b2"], p["b3"], p["b4"], p["b5"]
    T = np.array([[b1, 0.0, b5, b2 / 4.0],
                  [1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, b5, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    rhat_obs = data["rhat_obs_l1"].values  # observed part of rhat_{t-1}
    c = np.zeros((n, 4))
    c[:, 0] = -(b2 / 4.0) * rhat_obs - b3 * data["rp_tilde"].values + b4 * data["h_star"].values
    c = np.nan_to_num(c)
    # shocks: e_h (IS), e_crisis (2008/2020 periods), e_rr
    R = np.array([[1.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    Q = np.zeros((n, 3, 3))
    crisis = (data["d_crise2008"].values + data["d_crise2020"].values) > 0
    Q[:, 0, 0] = p["sd_is"] ** 2
    Q[:, 1, 1] = np.where(crisis, (p["crisis_mult"] * p["sd_is"]) ** 2, 1e-12)
    Q[:, 2, 2] = p["sd_rr"] ** 2
    gn, ge, gc, sig = p["g_nuci"], p["g_emp"], p["g_caged"], p["sig_h"]
    Z = np.array([[1.0, 0.0, 0.0, 0.0],
                  [gn, 0.0, 0.0, 0.0],
                  [0.0, ge, 0.0, 0.0],
                  [0.0, gc, 0.0, 0.0]])
    d = np.zeros(4)
    H = np.diag([sig ** 2, (gn * sig) ** 2, (ge * sig) ** 2, (gc * sig) ** 2])
    a1 = np.zeros(4)
    P1 = np.diag([4.0, 4.0, 1.0, 1.0])
    return StateSpace(T, c, R, Q, Z, d, H, a1, P1)


def loglik_factory(data):
    y = data[OBS].values

    def ll(p):
        try:
            return build_ss(p, data).loglik(y)
        except Exception:  # noqa: BLE001
            return -np.inf
    return ll


def estimate_gap(data, n_draws=20000, burn=5000, seed=0, verbose=True, priors=None):
    priors = priors or priors_gap()
    post = Posterior(loglik_factory(data), priors)
    start = {"b1": 0.85, "b2": 0.44, "b3": 0.03, "b4": 0.05, "b5": 0.5, "g_nuci": 1.8, "g_emp": 1.1,
             "g_caged": 0.7, "sig_h": 1.0, "sd_is": 0.4}
    mode = post.find_mode(start=start, verbose=verbose)
    if verbose:
        print("posterior mode:", {k: round(v, 4) for k, v in mode.items()})
    if n_draws:
        post.rwmh(n_draws=n_draws, burn=burn, seed=seed, verbose=verbose)
    return post


def smoothed_gap(post, data, params=None):
    p = params or post.mode
    ss = build_ss(p, data)
    f = ss.smooth(data[OBS].values)
    gap = pd.Series(f["asm"][:, 0], index=data.index, name="gap")
    sd = pd.Series(np.sqrt(f["Psm"][:, 0, 0]), index=data.index, name="gap_sd")
    rr = pd.Series(f["asm"][:, 3], index=data.index, name="rr_tilde_is")
    return pd.DataFrame({"gap": gap, "gap_sd": sd, "rr_tilde_is": rr, "rr_is": rr + data["rr_trend"]})
