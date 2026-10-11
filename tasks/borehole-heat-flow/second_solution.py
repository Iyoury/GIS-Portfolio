"""Independent second solution for borehole-heat-flow.

Differences from solution.py (method, not formatting):
  * TVD: circular-arc geometry. On each segment the path is a circle of radius
    L / beta in the plane of the first tangent t1 and the unit vector
    w = (t2 - t1 cos beta) / sin beta, so the displacement after an arc length s
    is (L / beta) [sin(phi) t1 + (1 - cos(phi)) w], phi = beta s / L, written
    with sinc functions (exact for beta = 0) (solution.py uses the ratio-factor
    formula 0.5 dMD (t1 + t2) tan(beta/2) / (beta/2) and the great-circle
    tangent of the partial arc).
  * R: superposition of the conductivity contrasts at the layer tops,
    1/k(z') = sum_i c_i H(z' - t_i) with c_0 = 1/k_0 and c_i = 1/k_i - 1/k_{i-1},
    so R(z) = sum_i c_i (z - t_i)_+ (a sum of ramps; solution.py clips every
    depth into every layer and sums (b - a)/k layer by layer).
  * S: integration by parts, S(z) = z R(z) - int_0^z R(z') dz', with the
    integral of the continuous, piecewise-linear R by the trapezoid rule on the
    layer tops and z, which is exact (solution.py sums (b - a)(b + a)/(2k)
    layer by layer). Both are exact for these elementary integrals, so they
    give the same values as the reference up to rounding; the independent
    evidence for step 2 is the hand-computed closed forms and the quadrature
    of the step 2 tests.
  * Paleoclimate: Duhamel convolution of the surface history with the
    half-space impulse response z/(2 sqrt(pi kappa tau^3)) exp(-z^2/4 kappa tau),
    integrated over each history interval from tau = 0 by adaptive quadrature
    in the similarity variable eta = z/(2 sqrt(kappa tau)) (solution.py uses
    differences of erfc).
  * Fit: QR-based least squares with covariance from the inverse of the
    triangular factor (solution.py solves the normal equations).
  * Layered paleoclimate perturbation (step 5): one global linear system per
    Laplace node for the scaled layer coefficients and the hyperbolic contour of
    Weideman and Trefethen (2007) (solution.py carries the admittance with a
    tanh recursion and inverts on the fixed Talbot contour).
  * Joint inversion (step 6): Levenberg-Marquardt on all four parameters
    (T0, q0, g, t_end) started from every minimum of a coarse profile scan,
    with dP/dt_end from the inverse transform of F itself (the impulse
    response) on the hyperbolic contour of its own layered solver, the
    covariance from the QR factor of the Jacobian, the threshold from the
    Student t quantile (F(1, nu) = t(nu)^2) and the interval ends by
    Illinois regula falsi (solution.py: variable projection over t_end with
    a 121-point scan, bounded Brent and Gauss-Newton polish, a fourth-order
    finite difference for dP/dt_end, scipy.stats.f and Brent's root finder).
"""
import math
import numpy as np
from scipy.integrate import quad
from scipy import stats

YEAR = 365.25 * 86400.0


def _t(i, a):
    return np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])


def _sc(x):
    return float(np.sinc(x / np.pi))          # sin(x) / x, exactly 1 at x = 0


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
    segs = []
    for i in range(md.size - 1):
        t1, t2 = _t(inc[i], azi[i]), _t(inc[i + 1], azi[i + 1])
        if t1 @ t2 <= np.cos(np.pi - 1e-6):
            raise ValueError("180 degree dogleg")
        beta = math.atan2(float(np.linalg.norm(np.cross(t1, t2))), float(t1 @ t2))
        # circle of radius L / beta in the plane of t1 and w = (t2 - t1 cos beta) / sin beta:
        # r(s) = (L / beta) [sin(phi) t1 + (1 - cos(phi)) w], phi = beta s / L, written with sinc
        segs.append((md[i + 1] - md[i], beta, t1[2], (t2[2] - t1[2] * math.cos(beta)) / _sc(beta)))

    def rise(i, s):
        L, beta, t1z, wz = segs[i]
        phi = beta * s / L
        return s * (_sc(phi) * t1z + 0.5 * (s / L) * _sc(0.5 * phi) ** 2 * wz)

    full = np.concatenate(([0.0], np.cumsum([rise(i, segs[i][0]) for i in range(len(segs))])))
    out = np.empty(q.size)
    for j, m in enumerate(q):
        i = min(int(np.searchsorted(md, m, side="right")) - 1, md.size - 2)
        out[j] = full[i] + rise(i, m - md[i])
    return float(out[0]) if q_in.ndim == 0 else out.reshape(q_in.shape)


def layer_integrals(layer_top_tvd, layer_k, z):
    tops = np.asarray(layer_top_tvd, float)
    k = np.asarray(layer_k, float)
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    if tops.ndim != 1 or tops.shape != k.shape or tops.size < 1 or not np.all(np.isfinite(tops)) \
            or tops[0] != 0 or np.any(np.diff(tops) <= 0):
        raise ValueError("bad layers")
    if np.any(~np.isfinite(k)) or np.any(k <= 0) or np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("bad input")
    # R: the resistivity profile 1/k(z') is a superposition of steps c_i H(z' - t_i) at the layer tops
    # (c_0 = 1/k_0, c_i = 1/k_i - 1/k_{i-1}), so R(z) = sum_i c_i (z - t_i)_+ (sum of ramps; no per-layer
    # partition and no layer thickness)
    c = np.diff(np.concatenate(([0.0], 1.0 / k)))

    def ramp_sum(x):
        return np.maximum(x[:, None] - tops[None, :], 0.0) @ c

    R = ramp_sum(zz)
    # S by integration by parts: S(z) = int_0^z z' dR(z') = z R(z) - int_0^z R(z') dz'. R is continuous and
    # linear between the tops, so the trapezoid rule on the nodes (tops above z, then z) is exact for the
    # integral of R; no z'/k integrand is formed
    R_top = ramp_sum(tops)
    I_top = np.concatenate(([0.0], np.cumsum(np.diff(tops) * 0.5 * (R_top[:-1] + R_top[1:]))))
    j = np.searchsorted(tops, zz, side="right") - 1
    I = I_top[j] + (zz - tops[j]) * 0.5 * (R_top[j] + R)
    S = zz * R - I
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
    # Duhamel: theta(z) = sum_i dT_i int_{edges[i]}^{edges[i+1]} g(z, tau) dtau with the half-space impulse
    # response g = z / (2 sqrt(pi kappa tau^3)) exp(-z^2 / (4 kappa tau)); with eta = z / (2 sqrt(kappa tau)),
    # g dtau = (2 / sqrt(pi)) exp(-eta^2) deta, and tau in [0, edges[1]] maps to eta in [eta(edges[1]), inf)
    kern = lambda e: 2.0 / math.sqrt(math.pi) * math.exp(-e * e)
    ETA_MAX = 40.0                               # exp(-1600): nothing beyond
    out = np.zeros(zz.size)
    for j, zj in enumerate(zz):
        if zj == 0:
            out[j] = d[0]
            continue
        tot = 0.0
        for i in range(t.size):
            a = zj / (2.0 * math.sqrt(kappa * edges[i + 1]))
            b = ETA_MAX if edges[i] == 0.0 else min(ETA_MAX, zj / (2.0 * math.sqrt(kappa * edges[i])))
            if a < b:
                tot += d[i] * quad(kern, a, b, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
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


# ---------------------------------------------------------------- step 6 (second solution)
def _ss_response_and_rate(z, tau, tops, k, rc, N=40):
    """Step response U(z, tau) and its rate dU/dtau (per second) at depths z > 0 of the layered column.

    For every node of the hyperbolic contour s(u) = mu (1 + sin(i u - alpha)) (Weideman and Trefethen 2007)
    the layer coefficients come from one global linear system (continuity of F and k F' at the tops, F = 1
    at the surface, decay in the bottom half-space); U is the inverse transform of F / s and the rate the
    inverse transform of F itself (U(z > 0, 0) = 0)."""
    nl = tops.size
    h = np.diff(tops)
    kap = k / rc
    lay = np.searchsorted(tops, z, side="right") - 1
    alpha, hh, mu = 1.1721, 1.0818 / N, 4.4921 * N / tau
    accU = np.zeros(z.size)
    accR = np.zeros(z.size)
    nun = 2 * nl - 1
    for kidx in range(0, N + 1):
        u = kidx * hh
        s = mu * (1.0 + np.sin(1j * u - alpha))
        ds = 1j * mu * np.cos(1j * u - alpha)
        q = np.sqrt(s / kap)
        A = np.zeros((nun, nun), complex)
        rhs = np.zeros(nun, complex)
        A[0, 0] = 1.0
        if nl > 1:
            A[0, 1] = np.exp(-q[0] * h[0])
        rhs[0] = 1.0
        row = 1
        for i in range(nl - 1):
            e = np.exp(-q[i] * h[i])
            en = np.exp(-q[i + 1] * h[i + 1]) if i + 1 < nl - 1 else 0.0
            A[row, 2 * i], A[row, 2 * i + 1], A[row, 2 * i + 2] = e, 1.0, -1.0
            if i + 1 < nl - 1:
                A[row, 2 * i + 3] = -en
            kq, kqn = k[i] * q[i], k[i + 1] * q[i + 1]
            A[row + 1, 2 * i], A[row + 1, 2 * i + 1], A[row + 1, 2 * i + 2] = -kq * e, kq, kqn
            if i + 1 < nl - 1:
                A[row + 1, 2 * i + 3] = -kqn * en
            row += 2
        c = np.linalg.solve(A, rhs)
        F = np.empty(z.size, complex)
        for i in np.unique(lay):
            sel = lay == i
            d = z[sel] - tops[i]
            v = c[2 * i] * np.exp(-q[i] * d)
            if i < nl - 1:
                v = v + c[2 * i + 1] * np.exp(-q[i] * (h[i] - d))
            F[sel] = v
        w = (1.0 if kidx == 0 else 2.0) * hh / (2.0 * np.pi)
        base = np.exp(s * tau) * ds / 1j
        accU += w * (base * F / s).real
        accR += w * (base * F).real
    return accU, accR


def deglaciation_heat_flow_inversion(survey_md, survey_inc, survey_azi, log_md, log_temp,
                                     layer_top_md, layer_k, heat_production, rho_c,
                                     hist_t_years, hist_dT_shape, t_end_bounds, z_min):
    """Second solution: Levenberg-Marquardt on (T0, q0, g, t_end) from the minima of a coarse profile scan,
    analytic dP/dt_end from the impulse response of the layered column, covariance from QR, threshold
    from the Student t quantile, interval ends by regula falsi (Illinois)."""
    log_md = np.asarray(log_md, float)
    log_temp = np.asarray(log_temp, float)
    if log_md.ndim != 1 or log_temp.shape != log_md.shape or not np.all(np.isfinite(log_temp)):
        raise ValueError("bad log")
    sh = np.asarray(hist_dT_shape, float)
    tk = np.asarray(hist_t_years, float)
    if sh.ndim != 1 or tk.ndim != 1 or sh.size < 2 or tk.size + 1 != sh.size:
        raise ValueError("bad history")
    if not np.all(np.isfinite(np.concatenate([sh, tk]))) or tk[0] <= 0 or np.any(np.diff(tk) <= 0):
        raise ValueError("bad history")
    if not sh[0] != sh[1]:
        raise ValueError("t_end not identifiable")
    bnd = np.asarray(t_end_bounds, float).ravel()
    if np.asarray(t_end_bounds).shape != (2,) or not np.all(np.isfinite(bnd)):
        raise ValueError("bad bounds")
    lo, hi = float(bnd[0]), float(bnd[1])
    if not (0 < lo < hi < tk[0]):
        raise ValueError("bad bounds")
    if not (np.isfinite(heat_production) and heat_production >= 0):
        raise ValueError("bad heat production")
    zr = true_vertical_depth(survey_md, survey_inc, survey_azi, log_md)
    tops = np.asarray(true_vertical_depth(survey_md, survey_inc, survey_azi, np.asarray(layer_top_md, float)), float)
    use = zr >= z_min
    n = int(use.sum())
    if n < 5:
        raise ValueError("fewer than 5 readings")
    z = zr[use]
    R, S = layer_integrals(tops, layer_k, z)
    # validates the column and rho_c through the layered step
    layered_paleoclimate_perturbation(z[:1], tops, layer_k, rho_c, tk, sh[1:])
    kk = np.asarray(layer_k, float)
    rc = float(rho_c)
    y = log_temp[use] + heat_production * S
    pos = z > 0

    def resp(t_years):
        U = np.ones(n)
        Rt = np.zeros(n)
        if np.any(pos):
            u, r = _ss_response_and_rate(z[pos], t_years * YEAR, tops, kk, rc)
            U[pos], Rt[pos] = u, r * YEAR          # rate per year
        return U, Rt

    # fixed part of P: s1 U(t_1) + sum_{i>=2} s_i (U(t_i) - U(t_{i-1})); P(t) = P_fix + (s0 - s1) U(t)
    Uk = [resp(t)[0] for t in tk]
    P_fix = sh[1] * Uk[0]
    for i in range(2, sh.size):
        P_fix = P_fix + sh[i] * (Uk[i - 1] - Uk[i - 2])
    dsh = sh[0] - sh[1]
    memo = {}

    def model_parts(t):
        if t not in memo:
            U, Rt = resp(t)
            memo[t] = (P_fix + dsh * U, dsh * Rt)
        return memo[t]

    def lin_fit(t):
        P = model_parts(t)[0]
        G = np.column_stack([np.ones(n), R, P])
        Q, Rm = np.linalg.qr(G)
        m = np.linalg.solve(Rm, Q.T @ y)
        r = y - G @ m
        return m, float(r @ r)

    # coarse scan, then Levenberg-Marquardt on the four parameters from every scan minimum
    ts = np.exp(np.linspace(np.log(lo), np.log(hi), 90))
    vals = np.array([lin_fit(t)[1] for t in ts])
    starts = [i for i in range(1, ts.size - 1) if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]]
    best = None
    for i in starts:
        m = lin_fit(ts[i])[0]
        th = np.array([m[0], m[1], m[2], ts[i]])
        P, dP = model_parts(th[3])
        res = y - (th[0] + th[1] * R + th[2] * P)
        cost = float(res @ res)
        lam = 1e-3
        for _ in range(200):
            J = np.column_stack([np.ones(n), R, P, th[2] * dP])
            scale = np.sqrt(np.sum(J * J, axis=0))
            Js = J / scale
            A = Js.T @ Js
            gvec = Js.T @ res
            accepted = False
            while lam < 1e12:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A)), gvec) / scale
                new = th + step
                new[3] = min(max(new[3], lo), hi)
                Pn, dPn = model_parts(float(new[3]))
                rn = y - (new[0] + new[1] * R + new[2] * Pn)
                cn = float(rn @ rn)
                if cn <= cost:
                    accepted = True
                    break
                lam *= 10.0
            if not accepted:
                break
            dt_rel = abs(new[3] - th[3]) / th[3]
            th, P, dP, res, cost = new, Pn, dPn, rn, cn
            lam = max(lam / 10.0, 1e-12)
            if dt_rel < 1e-13 and np.all(np.abs(step[:3]) <= 1e-13 * (np.abs(th[:3]) + 1e-12)):
                break
        if best is None or cost < best[1]:
            best = (th.copy(), cost)
    if best is None:
        raise ValueError("minimum of the profile at an end point")
    th, cmin = best
    if not (lo < th[3] < hi) or cmin >= min(vals[0], vals[-1]):
        raise ValueError("minimum of the profile at an end point")
    t_hat = float(th[3])
    # resolvability at the estimate (criterion of step 4)
    P, dP = model_parts(t_hat)
    G = np.column_stack([np.ones(n), R, P])
    ln = np.sqrt(np.sum(G * G, axis=0))
    if np.any(ln == 0) or np.linalg.svd(G / ln, compute_uv=False)[-1] < 1e-8:
        raise ValueError("not resolvable")
    J = np.column_stack([np.ones(n), R, P, th[2] * dP])
    Qj, Rj = np.linalg.qr(J)
    Ri = np.linalg.inv(Rj)
    s2 = cmin / (n - 4)
    sig = np.sqrt(np.diag(s2 * (Ri @ Ri.T)))
    tq = float(stats.t.ppf(0.975, n - 4))
    level = cmin * (1.0 + tq * tq / (n - 4))

    def g(t):
        return lin_fit(t)[1] - level

    def edge(direction):
        bound = lo if direction < 0 else hi
        a, fa = t_hat, g(t_hat)
        step = 0.1 * sig[3]
        while True:
            b = a + direction * step
            if direction * (b - bound) >= 0:
                b = bound
            fb = g(b)
            if fb >= 0:
                break
            if b == bound:
                raise ValueError("interval not contained in the bounds")
            a, fa = b, fb
            step = min(step * 1.3, 0.4 * sig[3])
        for _ in range(300):                    # Illinois regula falsi
            c = b - fb * (b - a) / (fb - fa)
            fc = g(c)
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) <= 1e-12 * t_hat or fc == 0:
                break
        return b

    t_low, t_high = edge(-1), edge(+1)
    return {"T0": float(th[0]), "q0": float(th[1]), "amplitude": float(th[2]), "t_end": t_hat,
            "sigma_T0": float(sig[0]), "sigma_q0": float(sig[1]), "sigma_amplitude": float(sig[2]),
            "sigma_t_end": float(sig[3]), "t_end_low": float(t_low), "t_end_high": float(t_high),
            "rms_residual": float(np.sqrt(cmin / n)), "n_used": n}
