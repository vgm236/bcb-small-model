"""
Layer A: single-equation estimation of the nominal block of the BCB aggregate small-scale model, given an
output-gap series (from gap_filter). Each equation is estimated

  * by (weighted) least squares with HAC standard errors, imposing the long-run restrictions of the box, and
  * by Bayesian methods with the box's priors (uniform supports), posterior mode + random-walk MH,

so that the comparison with the published posterior modes / 90% credible intervals is like for like.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from bayes import Prior, Posterior
from ssm import StateSpace

# ---------------------------------------------------------------------------
# generic Bayesian linear regression y = X b + e, e ~ N(0, s^2 / w_t)
# ---------------------------------------------------------------------------
class BayesLinear:
    def __init__(self, y, X, names, bounds, weights=None, sigma_prior=("invgamma", 1.0, 2.0)):
        self.y, self.X, self.names = np.asarray(y, float), np.asarray(X, float), list(names)
        self.w = np.ones(len(self.y)) if weights is None else np.asarray(weights, float)
        priors = {n: Prior("uniform", lo, hi, name=n) for n, (lo, hi) in zip(self.names, bounds)}
        priors["sigma"] = Prior(sigma_prior[0], mean=sigma_prior[1], sd=sigma_prior[2], name="sigma")
        self.post = Posterior(self._loglik, priors)

    def _loglik(self, p):
        b = np.array([p[n] for n in self.names])
        s2 = p["sigma"] ** 2
        e = self.y - self.X @ b
        return float(np.sum(-0.5 * (np.log(2 * np.pi * s2 / self.w) + self.w * e ** 2 / s2)))

    def wls(self, hac_lags=4):
        """(Weighted) least squares with HAC (Newey-West) covariance, for reference."""
        res = sm.WLS(self.y, self.X, weights=self.w).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        out = pd.DataFrame({"coef": res.params, "se_hac": res.bse}, index=self.names)
        out["t"] = out["coef"] / out["se_hac"]
        self.wls_res = res
        return out

    def fit(self, n_draws=20000, burn=5000, seed=0, verbose=False, start=None):
        if start is None:
            res = sm.WLS(self.y, self.X, weights=self.w).fit()
            start = {n: float(np.clip(v, lo + 1e-4, hi - 1e-4)) for n, v, (lo, hi) in
                     zip(self.names, res.params, [(self.post.priors[n].lo, self.post.priors[n].hi) for n in self.names])}
            start["sigma"] = float(np.sqrt(res.scale))
        self.post.find_mode(start=start, verbose=verbose, n_restarts=2)
        if n_draws:
            self.post.rwmh(n_draws=n_draws, burn=burn, seed=seed, verbose=verbose)
        return self.post


# ---------------------------------------------------------------------------
# Phillips curve (equation 1 of the June-2024 box), restricted form
# ---------------------------------------------------------------------------
PHILLIPS_BOUNDS = {"a1L": (0, 1), "a1I": (0, 1), "a2": (0, 1), "a3": (0, 1), "a4": (0, 1), "a5": (0, 0.01), "a6": (0, 0.01)}


def phillips_design(df, gap, start, end, exp_col="pi_e_q", comm_col="pi_icbr_hat", de_lag=1, hv_mult=None):
    from config import HV_SD_MULT
    hv_mult = HV_SD_MULT if hv_mult is None else hv_mult
    """Returns y, X (restricted form), weights and the index.

    pi_L - pi_e = a1L (pi_L_{t-1} - pi_e) + a1I (ma4 - pi_e) + a2 pihat* + a3 dehat_{t-lag} + a4 h + a5 clim_el + a6 clim_la
    """
    d = df.copy()
    d["gap"] = gap
    d["pe"] = d[exp_col]
    d["y"] = d["pi_livres_sa"] - d["pe"]
    d["x_a1L"] = d["pi_livres_sa"].shift(1) - d["pe"]
    d["x_a1I"] = d["pi_ipca_sa_ma4_l1"] - d["pe"]
    d["x_a2"] = d[comm_col]
    d["x_a3"] = d["de_hat"].shift(de_lag)
    d["x_a4"] = d["gap"]
    d["x_a5"] = d["clim_el"]
    d["x_a6"] = d["clim_la"]
    d = d.loc[start:end]
    cols = ["x_a1L", "x_a1I", "x_a2", "x_a3", "x_a4", "x_a5", "x_a6"]
    dd = d[["y"] + cols + ["d_phillips_hv"]].dropna()
    w = np.where(dd["d_phillips_hv"] > 0, 1.0 / hv_mult ** 2, 1.0)
    return dd["y"].values, dd[cols].values, w, dd.index


def estimate_phillips(df, gap, start, end, n_draws=20000, burn=5000, verbose=False, **kw):
    y, X, w, idx = phillips_design(df, gap, start, end, **kw)
    names = list(PHILLIPS_BOUNDS)
    m = BayesLinear(y, X, names, [PHILLIPS_BOUNDS[n] for n in names], weights=w, sigma_prior=("invgamma", 0.5, 1.0))
    wls = m.wls()
    m.fit(n_draws=n_draws, burn=burn, verbose=verbose)
    return m, wls, idx


# ---------------------------------------------------------------------------
# UIP (equation 4): de - de_ppc = -delta (idif - idif_{-1}) + e
# ---------------------------------------------------------------------------
def estimate_uip(df, start, end, n_draws=20000, burn=5000, verbose=False, idif_col="i_dif"):
    d = df.copy()
    d["y"] = d["de"] - d["de_ppc"]
    d["x"] = -(d[idif_col] - d[idif_col].shift(1))
    dd = d.loc[start:end, ["y", "x"]].dropna()
    m = BayesLinear(dd["y"].values, dd[["x"]].values, ["delta"], [(0, 10)], sigma_prior=("invgamma", 5.0, 5.0))
    wls = m.wls()
    m.fit(n_draws=n_draws, burn=burn, verbose=verbose)
    return m, wls, dd.index


# ---------------------------------------------------------------------------
# Taylor rule (equation 3) with a random-walk component in the equilibrium rate (Kalman filter)
# ---------------------------------------------------------------------------
TAYLOR_PRIORS = {"th1": Prior("uniform", 0, 2), "th2": Prior("uniform", -1, 1), "th3": Prior("uniform", 0, 4),
                 "sd_i": Prior("invgamma", mean=0.7, sd=1.0)}


def taylor_loglik_factory(df, start, end, sd_rr=0.15, rw=True):
    d = df.copy()
    d["i_l1"], d["i_l2"] = d["i"].shift(1), d["i"].shift(2)
    d = d.loc[start:end, ["i", "i_l1", "i_l2", "rr_trend", "pi_meta", "pi_e"]].dropna()
    i, i1, i2 = d["i"].values, d["i_l1"].values, d["i_l2"].values
    rr, pm, pe = d["rr_trend"].values, d["pi_meta"].values, d["pi_e"].values
    n = len(d)

    def ll(p):
        th1, th2, th3 = p["th1"], p["th2"], p["th3"]
        k = 1 - th1 - th2
        y = i - th1 * i1 - th2 * i2 - k * (rr + pm + th3 * (pe - pm))   # = k * rrtilde_t + e_t
        if not rw:
            return float(np.sum(-0.5 * (np.log(2 * np.pi * p["sd_i"] ** 2) + y ** 2 / p["sd_i"] ** 2)))
        ss = StateSpace(T=np.array([[1.0]]), c=np.zeros(1), R=np.array([[1.0]]), Q=np.array([[sd_rr ** 2]]),
                        Z=np.array([[k]]), d=np.zeros(1), H=np.array([[p["sd_i"] ** 2]]), a1=np.zeros(1), P1=np.array([[4.0]]))
        return ss.loglik(y.reshape(-1, 1))
    return ll, d


def estimate_taylor(df, start, end, n_draws=20000, burn=5000, verbose=False, sd_rr=0.15, rw=True):
    ll, d = taylor_loglik_factory(df, start, end, sd_rr=sd_rr, rw=rw)
    post = Posterior(ll, dict(TAYLOR_PRIORS))
    post.find_mode(start={"th1": 1.4, "th2": -0.5, "th3": 1.5, "sd_i": 0.5}, verbose=verbose)
    if n_draws:
        post.rwmh(n_draws=n_draws, burn=burn, verbose=verbose)
    return post, d


# ---------------------------------------------------------------------------
# Expectations (equation 5) by 2SLS: future 4-quarter inflation instrumented with date-t information
# ---------------------------------------------------------------------------
def expectations_design(df, gap, start, end):
    d = df.copy()
    d["gap"] = gap
    d["y"] = d["pi_e"] - d["pi_meta"]
    d["x_phi1"] = d["pi_e"].shift(1) - d["pi_meta"]
    fut = sum(d["pi_ipca"].shift(-k) for k in range(1, 5))    # realised inflation over t+1..t+4 (annual log %)
    d["x_phi2"] = fut - d["pi_meta"]
    d["x_phi3"] = d["pi_ipca_sum4_l1"] - d["pi_meta"]
    inst = ["x_phi1", "x_phi3", "gap", "i", "pi_icbr_hat", "de_hat", "pi_livres_sa"]
    cols = ["y", "x_phi1", "x_phi2", "x_phi3"] + [c for c in inst if c not in ("x_phi1", "x_phi3")]
    dd = d.loc[start:end, cols].dropna()
    return dd, inst


def estimate_expectations_2sls(df, gap, start, end):
    from statsmodels.sandbox.regression.gmm import IV2SLS
    dd, inst = expectations_design(df, gap, start, end)
    y = dd["y"].values
    X = dd[["x_phi1", "x_phi2", "x_phi3"]].values
    Z = dd[inst].values
    res = IV2SLS(y, X, Z).fit()
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    out = pd.DataFrame({"2SLS": res.params, "2SLS se": res.bse, "OLS": ols.params, "OLS se_hac": ols.bse},
                       index=["phi1", "phi2", "phi3"])
    return out, dd.index
