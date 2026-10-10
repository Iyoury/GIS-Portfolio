"""Independent second solution for borehole-heat-flow.

Differences from solution.py (method, not formatting):
  * TVD: numerical Gauss-Legendre integration of the vertical component of
    the unit tangent along the minimum-curvature arc (solution.py uses the
    closed-form ratio-factor formula).
  * R and S: adaptive quadrature (scipy.integrate.quad) of the piecewise
    conductivity profile (solution.py sums exact layer contributions).
  * Paleoclimate: Duhamel convolution of the surface history with the
    half-space impulse response z/(2 sqrt(pi kappa tau^3)) exp(-z^2/4 kappa tau),
    integrated numerically (solution.py uses differences of erfc).
  * Fit: QR-based least squares with covariance from the inverse of the
    triangular factor (solution.py solves the normal equations).
  * Layered paleoclimate perturbation (step 5): one global linear system per
    Laplace node for the scaled layer coefficients and the hyperbolic contour of
    Weideman and Trefethen (2007) (solution.py carries the admittance with a
    tanh recursion and inverts on the fixed Talbot contour).
"""
import numpy as np
from scipy.integrate import quad

YEAR = 365.25 * 86400.0
_X, _W = np.polynomial.legendre.leggauss(48)


def _t(i, a):
    return np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])


def true_vertical_depth(survey_md, inc_deg, azi_deg, md_query):
    md = np.asarray(survey_md, float)
    inc = np.radians(np.asarray(inc_deg, float))
    azi = np.radians(np.asarray(azi_deg, float))
    q_in = np.asarray(md_query, float)
    q = np.atleast_1d(q_in).ravel()
    if md.ndim != 1 or md.size < 2 or not (md.shape == inc.shape == azi.shape):
        raise ValueError("bad survey")
    if not np.all(np.isfinite(np.concatenate([md, inc, azi]))):
        raise ValueError("bad survey")
    if md[0] != 0 or np.any(np.diff(md) <= 0) or np.any(inc < 0) or np.any(inc >= np.pi):
        raise ValueError("bad survey")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or np.any(q > md[-1]):
        raise ValueError("query outside survey")

    def seg_tvd(i, b):
        a = md[i]
        if b <= a:
            return 0.0
        t1, t2 = _t(inc[i], azi[i]), _t(inc[i + 1], azi[i + 1])
        beta = np.arctan2(np.linalg.norm(np.cross(t1, t2)), t1 @ t2)
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _X + 0.5 * (b + a)
        f = (s - a) / L
        # slerp written with sinc so that it stays exact for any dogleg, including 0
        sc = lambda x: np.sinc(x / np.pi)
        tz = ((1 - f) * sc((1 - f) * beta) * t1[2] + f * sc(f * beta) * t2[2]) / sc(beta)
        return 0.5 * (b - a) * np.sum(_W * tz)

    for i in range(md.size - 1):
        if _t(inc[i], azi[i]) @ _t(inc[i + 1], azi[i + 1]) <= np.cos(np.pi - 1e-6):
            raise ValueError("180 degree dogleg")
    full = np.concatenate(([0.0], np.cumsum([seg_tvd(i, md[i + 1]) for i in range(md.size - 1)])))
    out = []
    for m in q:
        i = min(np.searchsorted(md, m, side="right") - 1, md.size - 2)
        out.append(full[i] + seg_tvd(i, m))
    out = np.array(out)
    return float(out[0]) if q_in.ndim == 0 else out.reshape(q_in.shape)


def layer_integrals(layer_top_tvd, layer_k, z):
    tops = np.asarray(layer_top_tvd, float)
    k = np.asarray(layer_k, float)
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    if tops.ndim != 1 or tops.shape != k.shape or not np.all(np.isfinite(tops)) or tops[0] != 0 or np.any(np.diff(tops) <= 0):
        raise ValueError("bad layers")
    if np.any(~np.isfinite(k)) or np.any(k <= 0) or np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("bad input")
    from scipy.integrate import quad  # local import: harness may strip top-level imports
    kf = lambda x: k[np.searchsorted(tops, x, side="right") - 1]
    R, S = [], []
    for zi in zz:
        pts = [t for t in tops if 0 < t < zi] or None
        R.append(quad(lambda x: 1 / kf(x), 0, zi, points=pts, limit=200, epsabs=1e-13)[0] if zi > 0 else 0.0)
        S.append(quad(lambda x: x / kf(x), 0, zi, points=pts, limit=200, epsabs=1e-13)[0] if zi > 0 else 0.0)
    R, S = np.array(R), np.array(S)
    if z_in.ndim == 0:
        return float(R[0]), float(S[0])
    return R.reshape(z_in.shape), S.reshape(z_in.shape)


def paleoclimate_perturbation(z, t_years, dT, kappa):
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    t = np.asarray(t_years, float)
    d = np.asarray(dT, float)
    if t.ndim != 1 or t.shape != d.shape or t.size < 1 or not np.all(np.isfinite(np.concatenate([t, d]))) or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("bad history")
    if not np.isfinite(kappa) or kappa <= 0 or np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("bad input")
    from scipy.integrate import quad  # local import: harness may strip top-level imports
    edges = np.concatenate(([0.0], t * YEAR))
    out = np.zeros(zz.size)
    for j, zj in enumerate(zz):
        if zj == 0:
            out[j] = d[0]
            continue
        g = lambda tau: zj / (2 * np.sqrt(np.pi * kappa * tau**3)) * np.exp(-zj**2 / (4 * kappa * tau))
        tot = 0.0
        for i in range(t.size):
            lo, hi = edges[i], edges[i + 1]
            # integrate in log-time for accuracy over many decades
            f = lambda u: g(np.exp(u)) * np.exp(u)
            lo_u = np.log(max(lo, 1e-3 * YEAR))
            tot += d[i] * quad(f, lo_u, np.log(hi), limit=400, epsabs=1e-14, epsrel=1e-12)[0]
        out[j] = tot
    return float(out[0]) if z_in.ndim == 0 else out.reshape(z_in.shape)


def fit_heat_flow(T, R, S, P, A):
    T, R, S, P = (np.asarray(v, float).ravel() for v in (T, R, S, P))
    n = T.size
    if not (R.size == S.size == P.size == n) or n < 4:
        raise ValueError("bad sizes")
    if not np.all(np.isfinite(np.concatenate([T, R, S, P]))) or not np.isfinite(A) or A < 0:
        raise ValueError("bad input")
    G = np.column_stack([np.ones(n), R, P])
    lengths = np.sqrt(np.sum(G * G, axis=0))
    if np.any(lengths == 0.0):
        raise ValueError("rank-deficient design")
    if np.linalg.svd(G / lengths, compute_uv=False)[-1] < 1e-8:
        raise ValueError("rank-deficient design")
    Q, Rm = np.linalg.qr(G)
    m = np.linalg.solve(Rm, Q.T @ (T + A * S))
    r = T + A * S - G @ m
    Rinv = np.linalg.inv(Rm)
    C = (r @ r) / (n - 3) * (Rinv @ Rinv.T)
    return {"T0": float(m[0]), "q0": float(m[1]), "amplitude": float(m[2]),
            "sigma_T0": float(np.sqrt(C[0, 0])), "sigma_q0": float(np.sqrt(C[1, 1])),
            "sigma_amplitude": float(np.sqrt(C[2, 2])), "rms_residual": float(np.sqrt(np.mean(r**2)))}


def surface_heat_flow(survey_md, survey_inc, survey_azi, log_md, log_temp,
                      layer_top_md, layer_k, heat_production, kappa,
                      hist_t_years, hist_dT_shape, z_min):
    log_md, log_temp = np.asarray(log_md, float), np.asarray(log_temp, float)
    if log_md.ndim != 1 or log_md.shape != log_temp.shape:
        raise ValueError("bad log")
    z = true_vertical_depth(survey_md, survey_inc, survey_azi, log_md)
    tops = true_vertical_depth(survey_md, survey_inc, survey_azi, np.asarray(layer_top_md, float))
    sel = z >= z_min
    R, S = layer_integrals(tops, layer_k, z[sel])
    P = paleoclimate_perturbation(z[sel], hist_t_years, hist_dT_shape, kappa)
    out = fit_heat_flow(log_temp[sel], R, S, P, heat_production)
    out["n_used"] = int(sel.sum())
    return out


# ---------------------------------------------------------------- step 5 (second solution)
def layered_paleoclimate_perturbation(z, layer_tops, layer_k, rho_c, t_years, dT):
    """Present-day departure (K) at depth z in a layered column (second solution)."""
    # Other methods: for each Laplace variable the layer coefficients come from one global linear system
    # (interface continuity of T and k dT/dz, surface value, decay in the bottom half-space), with the
    # exponentials scaled to each layer; the inversion uses the hyperbolic contour of Weideman and
    # Trefethen (2007) with the trapezoid rule instead of the fixed Talbot contour.
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    tops = np.asarray(layer_tops, float)
    kk = np.asarray(layer_k, float)
    tt = np.asarray(t_years, float)
    dd = np.asarray(dT, float)
    if tops.ndim != 1 or kk.ndim != 1 or tops.shape != kk.shape or tops.size < 1:
        raise ValueError("layer arrays must be 1-D with equal length")
    if not np.all(np.isfinite(tops)) or tops[0] != 0.0 or np.any(np.diff(tops) <= 0):
        raise ValueError("layer tops must start at 0 and increase strictly")
    if not np.all(np.isfinite(kk)) or np.any(kk <= 0):
        raise ValueError("conductivities must be finite and positive")
    if not (np.isfinite(rho_c) and rho_c > 0):
        raise ValueError("volumetric heat capacity must be finite and positive")
    if tt.ndim != 1 or tt.shape != dd.shape or tt.size < 1:
        raise ValueError("history arrays must be 1-D with equal length")
    if not (np.all(np.isfinite(tt)) and np.all(np.isfinite(dd))):
        raise ValueError("history values must be finite")
    if tt[0] <= 0 or np.any(np.diff(tt) <= 0):
        raise ValueError("history times must be positive and increasing")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("depths must be >= 0")
    nl = tops.size
    h = np.diff(tops)
    kap = kk / float(rho_c)
    lay = np.searchsorted(tops, zz, side="right") - 1

    def transform(s):
        # F(z, s) for all depths: layer i (top t_i, thickness h_i) holds a_i e^{-q d} + b_i e^{-q (h_i - d)},
        # d = z - t_i; the bottom layer holds a e^{-q d} only
        q = np.sqrt(s / kap)
        nun = 2 * nl - 1
        A = np.zeros((nun, nun), complex)
        rhs = np.zeros(nun, complex)
        ia = lambda i: 2 * i
        ib = lambda i: 2 * i + 1
        row = 0
        A[row, ia(0)] = 1.0
        if nl > 1:
            A[row, ib(0)] = np.exp(-q[0] * h[0])
        rhs[row] = 1.0
        row += 1
        for i in range(nl - 1):
            e = np.exp(-q[i] * h[i])
            en = np.exp(-q[i + 1] * h[i + 1]) if i + 1 < nl - 1 else 0.0
            A[row, ia(i)], A[row, ib(i)] = e, 1.0
            A[row, ia(i + 1)] = -1.0
            if i + 1 < nl - 1:
                A[row, ib(i + 1)] = -en
            row += 1
            kq, kqn = kk[i] * q[i], kk[i + 1] * q[i + 1]
            A[row, ia(i)], A[row, ib(i)] = -kq * e, kq
            A[row, ia(i + 1)] = kqn
            if i + 1 < nl - 1:
                A[row, ib(i + 1)] = -kqn * en
            row += 1
        c = np.linalg.solve(A, rhs)
        out = np.empty(zz.size, complex)
        for j in range(zz.size):
            i = lay[j]
            dd_ = zz[j] - tops[i]
            v = c[ia(i)] * np.exp(-q[i] * dd_)
            if i < nl - 1:
                v += c[ib(i)] * np.exp(-q[i] * (h[i] - dd_))
            out[j] = v
        return out

    def step(tau, N=40):
        # f(tau) = (1 / 2 pi i) int e^{s tau} F(s)/s ds on s(u) = mu (1 + sin(i u - alpha)), trapezoid in u
        alpha, hh, mu = 1.1721, 1.0818 / N, 4.4921 * N / tau
        acc = np.zeros(zz.size)
        for kidx in range(0, N + 1):
            u = kidx * hh
            sv = mu * (1.0 + np.sin(1j * u - alpha))
            ds = 1j * mu * np.cos(1j * u - alpha)
            term = np.exp(sv * tau) * transform(sv) / sv * ds
            w = 1.0 if kidx == 0 else 2.0            # conjugate symmetry of the integrand
            acc += w * (hh / (2.0 * np.pi) * term / 1j).real
        return acc

    ts = tt * 365.25 * 86400.0
    edges = np.concatenate(([0.0], ts))
    res = np.zeros(zz.size)
    pos = zz > 0
    U = {tau: step(tau) for tau in edges[1:]}
    for i in range(tt.size):
        u_new = U[edges[i]] if edges[i] > 0 else 0.0
        res += dd[i] * (U[edges[i + 1]] - u_new)
    res[~pos] = dd[0]
    return float(res[0]) if z_in.ndim == 0 else res.reshape(z_in.shape)
