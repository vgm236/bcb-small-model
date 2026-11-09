"""
Bayesian estimation toolkit (written from scratch): prior distributions, parameter transformations,
posterior-mode optimisation, Laplace approximation and an adaptive random-walk Metropolis-Hastings sampler.

Priors follow the BCB June-2024 box: mostly uniform on a support ("priors pouco informativas, limitando apenas
o suporte"), a Beta prior (mean, s.d.) for the fiscal coefficient, and inverse-gamma / uniform priors for
standard deviations that the box does not report.
"""
import numpy as np
from scipy import optimize, stats, special


class Prior:
    """A univariate prior with log-density and a bijection to the real line for optimisation."""

    def __init__(self, kind, a=None, b=None, mean=None, sd=None, name=""):
        self.kind, self.name = kind, name
        if kind == "uniform":
            self.lo, self.hi = float(a), float(b)
        elif kind == "beta":
            # Beta on [0,1] parameterised by mean and s.d. (as in the BCB table: Beta([0.03; 0.002]))
            m, s = float(mean), float(sd)
            k = m * (1 - m) / s ** 2 - 1
            self.alpha, self.beta = m * k, (1 - m) * k
            self.lo, self.hi = 0.0, 1.0
        elif kind == "normal":
            self.mean, self.sd = float(mean), float(sd)
            self.lo, self.hi = -np.inf, np.inf
        elif kind == "invgamma":
            # inverse gamma for standard deviations, parameterised by mean and s.d. (shape > 2)
            m, s = float(mean), float(sd)
            a_ = 2 + (m / s) ** 2
            b_ = m * (a_ - 1)
            self.alpha, self.beta = a_, b_
            self.lo, self.hi = 0.0, np.inf
        elif kind == "fixed":
            self.value = float(mean)
            self.lo, self.hi = self.value, self.value
        else:
            raise ValueError(kind)

    def logpdf(self, x):
        if self.kind == "uniform":
            return 0.0 if (self.lo <= x <= self.hi) else -np.inf
        if self.kind == "beta":
            if not (0 < x < 1):
                return -np.inf
            return stats.beta.logpdf(x, self.alpha, self.beta)
        if self.kind == "normal":
            return stats.norm.logpdf(x, self.mean, self.sd)
        if self.kind == "invgamma":
            if x <= 0:
                return -np.inf
            return stats.invgamma.logpdf(x, self.alpha, scale=self.beta)
        if self.kind == "fixed":
            return 0.0
        raise ValueError

    # bijections: unconstrained z <-> constrained x
    def to_unc(self, x):
        if self.kind in ("uniform", "beta"):
            p = (x - self.lo) / (self.hi - self.lo)
            p = min(max(p, 1e-10), 1 - 1e-10)
            return np.log(p / (1 - p))
        if self.kind == "invgamma":
            return np.log(max(x, 1e-12))
        return float(x)

    def to_con(self, z):
        if self.kind in ("uniform", "beta"):
            return self.lo + (self.hi - self.lo) * special.expit(z)
        if self.kind == "invgamma":
            return float(np.exp(z))
        return float(z)

    def log_jacobian(self, z):
        """log |dx/dz| for the change of variables when optimising / sampling in z."""
        if self.kind in ("uniform", "beta"):
            p = special.expit(z)
            return np.log(self.hi - self.lo) + np.log(p) + np.log(1 - p)
        if self.kind == "invgamma":
            return float(z)
        return 0.0

    def mode_guess(self):
        if self.kind == "uniform":
            return 0.5 * (self.lo + self.hi)
        if self.kind == "beta":
            return self.alpha / (self.alpha + self.beta)
        if self.kind == "normal":
            return self.mean
        if self.kind == "invgamma":
            return self.beta / (self.alpha + 1)
        return self.value


class Posterior:
    """Combines a log-likelihood function of a parameter dict with a dict of priors."""

    def __init__(self, loglik_fn, priors, fixed=None):
        self.loglik_fn = loglik_fn
        self.priors = {k: v for k, v in priors.items() if v.kind != "fixed"}
        self.fixed = dict(fixed or {})
        self.fixed.update({k: v.value for k, v in priors.items() if v.kind == "fixed"})
        self.names = list(self.priors)
        self.n_evals = 0

    def params_from_unc(self, z):
        p = dict(self.fixed)
        for name, zi in zip(self.names, z):
            p[name] = self.priors[name].to_con(zi)
        return p

    def unc_from_params(self, p):
        return np.array([self.priors[n].to_unc(p[n]) for n in self.names])

    def log_prior(self, p):
        return sum(self.priors[n].logpdf(p[n]) for n in self.names)

    def log_post(self, p):
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        self.n_evals += 1
        ll = self.loglik_fn(p)
        if not np.isfinite(ll):
            return -np.inf
        return ll + lp

    def log_post_unc(self, z):
        p = self.params_from_unc(z)
        lp = self.log_post(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + sum(self.priors[n].log_jacobian(zi) for n, zi in zip(self.names, z))

    # ---------------------------------------------------------------- mode
    def find_mode(self, start=None, n_restarts=3, seed=0, maxiter=4000, verbose=False):
        rng = np.random.default_rng(seed)
        best_z, best_val = None, -np.inf
        if start is None:
            start = {n: self.priors[n].mode_guess() for n in self.names}
        z0 = self.unc_from_params(start)
        starts = [z0] + [z0 + rng.normal(scale=0.5, size=len(z0)) for _ in range(n_restarts - 1)]
        f0 = self.log_post_unc(z0)
        if np.isfinite(f0):          # the starting point itself is a candidate
            best_z, best_val = z0.copy(), f0
        for k, z in enumerate(starts):
            f = lambda zz: -self.log_post_unc(zz)  # noqa: E731
            if not np.isfinite(f(z)):
                continue
            r = optimize.minimize(f, z, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8, "adaptive": True})
            r = optimize.minimize(f, r.x, method="BFGS", options={"maxiter": 500, "gtol": 1e-5})
            if verbose:
                print("  start %d: log posterior %.3f" % (k, -r.fun))
            if -r.fun > best_val:
                best_val, best_z = -r.fun, r.x
        self.mode_z, self.mode_val = best_z, best_val
        self.mode = self.params_from_unc(best_z)
        return self.mode

    def hessian_unc(self, z=None, eps=1e-4):
        """Numerical Hessian of the log posterior in the unconstrained space (for the proposal covariance)."""
        z = self.mode_z if z is None else z
        n = len(z)
        H = np.zeros((n, n))
        f0 = self.log_post_unc(z)
        for i in range(n):
            for j in range(i, n):
                zpp = z.copy(); zpp[i] += eps; zpp[j] += eps
                zpm = z.copy(); zpm[i] += eps; zpm[j] -= eps
                zmp = z.copy(); zmp[i] -= eps; zmp[j] += eps
                zmm = z.copy(); zmm[i] -= eps; zmm[j] -= eps
                H[i, j] = H[j, i] = (self.log_post_unc(zpp) - self.log_post_unc(zpm) - self.log_post_unc(zmp) + self.log_post_unc(zmm)) / (4 * eps ** 2)
        return H

    def laplace_cov(self):
        H = self.hessian_unc()
        try:
            cov = np.linalg.inv(-H)
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 1e-6, None)
            return V @ np.diag(w) @ V.T
        except np.linalg.LinAlgError:
            return np.eye(len(self.mode_z)) * 0.01

    # ---------------------------------------------------------------- MCMC
    def rwmh(self, n_draws=20000, burn=5000, scale=None, cov=None, seed=1, adapt_until=None, target=0.25, verbose=True):
        """Adaptive random-walk Metropolis-Hastings in the unconstrained space.

        The proposal is N(0, scale^2 * cov); during the adaptation phase `scale` is tuned towards the target
        acceptance rate and then frozen so that the chain is a valid Metropolis sampler.
        """
        rng = np.random.default_rng(seed)
        z = self.mode_z.copy()
        n = len(z)
        cov = self.laplace_cov() if cov is None else cov
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
        scale = (2.38 / np.sqrt(n)) if scale is None else scale
        adapt_until = burn if adapt_until is None else adapt_until
        lp = self.log_post_unc(z)
        draws = np.zeros((n_draws, n)); lps = np.zeros(n_draws)
        acc = 0
        for i in range(n_draws):
            prop = z + scale * (L @ rng.normal(size=n))
            lpp = self.log_post_unc(prop)
            if np.log(rng.uniform()) < lpp - lp:
                z, lp = prop, lpp
                acc += 1
            draws[i], lps[i] = z, lp
            if i < adapt_until and (i + 1) % 200 == 0:
                rate = acc / (i + 1)
                scale *= np.exp(0.5 * (rate - target))
            if verbose and (i + 1) % max(1, n_draws // 10) == 0:
                print("  draw %d/%d  acceptance %.3f  scale %.3f  logpost %.2f" % (i + 1, n_draws, acc / (i + 1), scale, lp))
        keep = draws[burn:]
        cons = np.array([[self.priors[nm].to_con(zz) for nm, zz in zip(self.names, row)] for row in keep])
        self.chain, self.chain_lp = cons, lps[burn:]
        return cons

    def summary(self, chain=None, q=(0.05, 0.95)):
        import pandas as pd
        chain = self.chain if chain is None else chain
        rows = []
        for j, nm in enumerate(self.names):
            x = chain[:, j]
            rows.append({"param": nm, "mode": self.mode[nm], "mean": x.mean(), "median": np.median(x),
                         "sd": x.std(), "q%02d" % int(q[0] * 100): np.quantile(x, q[0]), "q%02d" % int(q[1] * 100): np.quantile(x, q[1])})
        return pd.DataFrame(rows).set_index("param")
