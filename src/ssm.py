"""
Linear Gaussian state-space model: Kalman filter (log-likelihood) and fixed-interval (RTS) smoother.

    state:        a_t = T_t a_{t-1} + c_t + R_t eta_t,   eta_t ~ N(0, Q_t)
    observation:  y_t = Z_t a_t + d_t + eps_t,          eps_t ~ N(0, H_t)

System matrices may be constant (2-D arrays) or time varying (3-D arrays with the time index first).
Missing observations (NaN) are handled by dropping the corresponding rows at each date.
Written from scratch in numpy so that every step of the estimation is transparent.
"""
import numpy as np


def _at(M, t):
    """Time-t slice of a system matrix (3-D array) or vector (2-D array); constants are returned as is."""
    return M[t] if M.ndim == 3 else M


def _vat(v, t):
    return v[t] if v.ndim == 2 else v


class StateSpace:
    def __init__(self, T, c, R, Q, Z, d, H, a1, P1):
        self.T, self.c, self.R, self.Q, self.Z, self.d, self.H = [np.asarray(x, dtype=float) for x in (T, c, R, Q, Z, d, H)]
        self.a1, self.P1 = np.asarray(a1, dtype=float), np.asarray(P1, dtype=float)

    def filter(self, y, store=True, skip=0):
        """Kalman filter. y: (n_obs_periods, n_y) array with NaNs for missing values.

        Returns dict with loglik, filtered means/variances, predicted means/variances and innovations.
        The timing convention is that y_t is observed after the transition from t-1 to t (a1, P1 describe a_0).
        `skip` excludes the first `skip` periods from the log-likelihood (presample used for initialisation).
        """
        y = np.asarray(y, dtype=float)
        n, ny = y.shape
        m = self.a1.shape[0]
        a_prev, P_prev = self.a1.copy(), self.P1.copy()
        ll = 0.0
        if store:
            att = np.zeros((n, m)); Ptt = np.zeros((n, m, m)); apred = np.zeros((n, m)); Ppred = np.zeros((n, m, m))
            vlist = []
        for t in range(n):
            T, c, R, Q, Z, d, H = (_at(self.T, t), _vat(self.c, t), _at(self.R, t), _at(self.Q, t),
                                   _at(self.Z, t), _vat(self.d, t), _at(self.H, t))
            a = T @ a_prev + c
            P = T @ P_prev @ T.T + R @ Q @ R.T
            P = 0.5 * (P + P.T)
            obs = ~np.isnan(y[t])
            if store:
                apred[t], Ppred[t] = a, P
            if obs.any():
                Zt, dt, Ht = Z[obs], d[obs], H[np.ix_(obs, obs)]
                v = y[t, obs] - Zt @ a - dt
                F = Zt @ P @ Zt.T + Ht
                F = 0.5 * (F + F.T)
                try:
                    L = np.linalg.cholesky(F)
                except np.linalg.LinAlgError:
                    return {"loglik": -np.inf}
                Finv_v = np.linalg.solve(L.T, np.linalg.solve(L, v))
                logdet = 2 * np.log(np.diag(L)).sum()
                if t >= skip:
                    ll += -0.5 * (obs.sum() * np.log(2 * np.pi) + logdet + v @ Finv_v)
                K = P @ Zt.T
                a = a + K @ Finv_v
                KFi = np.linalg.solve(L.T, np.linalg.solve(L, K.T)).T  # P Z' F^{-1}
                P = P - KFi @ K.T
                P = 0.5 * (P + P.T)
                if store:
                    vlist.append((t, v, F))
            if store:
                att[t], Ptt[t] = a, P
            a_prev, P_prev = a, P
        out = {"loglik": ll}
        if store:
            out.update(att=att, Ptt=Ptt, apred=apred, Ppred=Ppred, innovations=vlist)
        return out

    def loglik(self, y, skip=0):
        return self.filter(y, store=False, skip=skip)["loglik"]

    def smooth(self, y):
        """Fixed-interval state smoother (Durbin & Koopman, 2012, section 4.4), which does not invert the state
        covariance and therefore works with singular P (identities / lags in the state vector).

        Returns the filter output augmented with smoothed means `asm` (n, m) and variances `Psm` (n, m, m).
        """
        y = np.asarray(y, dtype=float)
        f = self.filter(y, store=True)
        if not np.isfinite(f["loglik"]):
            raise ValueError("filter failed")
        apred, Ppred = f["apred"], f["Ppred"]
        n, m = apred.shape
        inn = {t: (v, F) for (t, v, F) in f["innovations"]}
        r = np.zeros(m)
        N = np.zeros((m, m))
        asm = np.zeros((n, m)); Psm = np.zeros((n, m, m))
        for t in range(n - 1, -1, -1):
            Tn = _at(self.T, t + 1) if t + 1 < n else _at(self.T, t)  # transition from t to t+1
            obs = ~np.isnan(y[t])
            if obs.any() and t in inn:
                v, F = inn[t]
                Zt = _at(self.Z, t)[obs]
                Finv = np.linalg.inv(F)
                K = Tn @ Ppred[t] @ Zt.T @ Finv            # Kalman gain in DK notation (K_t = T P Z' F^-1)
                L = Tn - K @ Zt
                r = Zt.T @ Finv @ v + L.T @ r
                N = Zt.T @ Finv @ Zt + L.T @ N @ L
            else:
                r = Tn.T @ r
                N = Tn.T @ N @ Tn
            asm[t] = apred[t] + Ppred[t] @ r
            Psm[t] = Ppred[t] - Ppred[t] @ N @ Ppred[t]
            Psm[t] = 0.5 * (Psm[t] + Psm[t].T)
        f.update(asm=asm, Psm=Psm)
        return f


def unconditional_variance(T, RQR, max_iter=5000, tol=1e-10):
    """Solve P = T P T' + RQR for a stationary T by doubling; falls back to iteration."""
    m = T.shape[0]
    try:
        # vec(P) = (I - T kron T)^{-1} vec(RQR)
        I = np.eye(m * m)
        vecP = np.linalg.solve(I - np.kron(T, T), RQR.reshape(-1, order="F"))
        P = vecP.reshape(m, m, order="F")
        P = 0.5 * (P + P.T)
        if np.all(np.isfinite(P)):
            return P
    except np.linalg.LinAlgError:
        pass
    P = RQR.copy()
    for _ in range(max_iter):
        Pn = T @ P @ T.T + RQR
        if np.max(np.abs(Pn - P)) < tol:
            return Pn
        P = Pn
    return P
