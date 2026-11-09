"""
Linear rational-expectations solver: a numpy/scipy port of Christopher Sims' `gensys` (Sims, 2002,
"Solving Linear Rational Expectations Models", Computational Economics).

System:   G0 y_t = G1 y_{t-1} + C + Psi z_t + Pi eta_t,   with E_{t-1} eta_t = 0 (expectational errors)
Solution: y_t = G1s y_{t-1} + Cs + impact z_t

`eu` = [existence, uniqueness] flags (1 = yes). The QZ ordering puts stable generalized eigenvalues first.
"""
import numpy as np
from scipy.linalg import ordqz, svd


def gensys(g0, g1, c, psi, pi, div=1.0 + 1e-8, realsmall=1e-7):
    g0, g1 = np.asarray(g0, float), np.asarray(g1, float)
    c = np.asarray(c, float).reshape(-1, 1)
    psi, pi = np.asarray(psi, float), np.asarray(pi, float)
    n = g0.shape[0]
    eu = [0, 0]

    def stable(alpha, beta):
        return np.abs(beta) <= div * np.abs(alpha)

    a, b, alpha, beta, Q, Z = ordqz(g0, g1, sort=stable, output="complex")
    # scipy: g0 = Q a Z^H, g1 = Q b Z^H.  Sims' q = Q^H, Sims' z = Z.
    q = Q.conj().T
    z = Z
    nstab = int(np.sum(stable(alpha, beta)))
    nunstab = n - nstab
    q1, q2 = q[:nstab, :], q[nstab:, :]

    def _svd_trim(M):
        if M.size == 0:
            return np.zeros((M.shape[0], 0)), np.zeros((0, 0)), np.zeros((M.shape[1], 0))
        u, s, vh = svd(M, full_matrices=False)
        keep = s > realsmall
        return u[:, keep], np.diag(s[keep]), vh.conj().T[:, keep]

    etawt = q2 @ pi
    ueta, deta, veta = _svd_trim(etawt)
    # existence
    if nunstab == 0:
        eu[0] = 1
    else:
        zwt = q2 @ psi
        uz, dz, vz = _svd_trim(zwt)
        if uz.shape[1] == 0:
            eu[0] = 1
        else:
            resid = uz - ueta @ (ueta.conj().T @ uz)
            eu[0] = 1 if np.linalg.norm(resid) < realsmall * n else 0
    # uniqueness
    etawt1 = q1 @ pi
    ueta1, deta1, veta1 = _svd_trim(etawt1)
    if veta1.shape[1] == 0:
        eu[1] = 1
    else:
        loose = veta1 - veta @ (veta.conj().T @ veta1)
        ul, dl, vl = svd(loose, full_matrices=False)
        eu[1] = 1 if (dl.size == 0 or np.max(dl) < realsmall * n) else 0

    if deta.size:
        tmat_right = -(ueta @ np.linalg.inv(deta) @ veta.conj().T @ veta1 @ deta1 @ ueta1.conj().T).conj().T
    else:
        tmat_right = np.zeros((nstab, nunstab), dtype=complex)
    tmat = np.hstack([np.eye(nstab, dtype=complex), tmat_right])
    G0 = np.vstack([tmat @ a, np.hstack([np.zeros((nunstab, nstab)), np.eye(nunstab)])])
    G1 = np.vstack([tmat @ b, np.zeros((nunstab, n))])
    G0I = np.linalg.inv(G0)
    G1 = G0I @ G1
    usix = slice(nstab, n)
    if nunstab > 0:
        cu = np.linalg.solve(a[usix, usix] - b[usix, usix], q2 @ c)
    else:
        cu = np.zeros((0, 1))
    C = G0I @ np.vstack([tmat @ q @ c, cu])
    impact = G0I @ np.vstack([tmat @ q @ psi, np.zeros((nunstab, psi.shape[1]))])
    G1s = np.real(z @ G1 @ z.conj().T)
    Cs = np.real(z @ C).ravel()
    imp = np.real(z @ impact)
    return G1s, Cs, imp, eu


def irf(G1, impact, shock_index, horizon=20, size=1.0):
    """Impulse response of y to a one-off shock z_{shock_index} = size at t=0."""
    n = G1.shape[0]
    y = np.zeros((horizon + 1, n))
    y[0] = impact[:, shock_index] * size
    for t in range(1, horizon + 1):
        y[t] = G1 @ y[t - 1]
    return y


if __name__ == "__main__":
    # self-test 1: purely backward-looking AR(1): y_t = 0.5 y_{t-1} + z_t
    G1s, Cs, imp, eu = gensys(np.array([[1.0]]), np.array([[0.5]]), np.zeros(1), np.array([[1.0]]), np.zeros((1, 0)))
    assert eu == [1, 1] and abs(G1s[0, 0] - 0.5) < 1e-10 and abs(imp[0, 0] - 1) < 1e-10
    # self-test 2: forward-looking x_t = 0.8 E_t x_{t+1} + z_t  ->  x_t = z_t (unique, no persistence)
    # variables: y = [x_t, xe_t = E_t x_{t+1}];  eq1: x_t - 0.8 xe_t = z_t;  eq2: x_t = xe_{t-1} + eta_t
    g0 = np.array([[1.0, -0.8], [1.0, 0.0]]); g1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    G1s, Cs, imp, eu = gensys(g0, g1, np.zeros(2), np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]]))
    assert eu == [1, 1], eu
    assert np.allclose(G1s, 0, atol=1e-8) and abs(imp[0, 0] - 1) < 1e-8 and abs(imp[1, 0]) < 1e-8
    # self-test 3: hybrid x_t = 0.5 x_{t-1} + 0.4 E_t x_{t+1} + z_t ; stable root of 0.4 L^2 - L + 0.5 = 0 -> ~0.6910
    g0 = np.array([[1.0, -0.4], [1.0, 0.0]]); g1 = np.array([[0.5, 0.0], [0.0, 1.0]])
    G1s, Cs, imp, eu = gensys(g0, g1, np.zeros(2), np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]]))
    root = (1 - np.sqrt(1 - 4 * 0.4 * 0.5)) / (2 * 0.4)
    assert eu == [1, 1] and abs(G1s[0, 0] - root) < 1e-8, (G1s, root)
    print("gensys self-tests passed; hybrid root =", root)
