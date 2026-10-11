"""Whole-task mutant: Measured depth along the inclined holes used as depth (no trajectory correction) for the readings and the layer tops.

Complete solution with this single error in regional_deglaciation_inversion.
"""
import numpy as np
import math
from scipy import stats
from scipy.optimize import brentq, minimize_scalar


# ------------------------------------------------------------ step 1: true vertical depth
def _unit(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])


def _sinc(x):
    return np.sinc(x / np.pi)          # sin(x) / x, exact at x = 0


def _ratio_factor(beta):
    h = 0.5 * beta                     # tan(h) / h, no straight-line shortcut
    if h < 1e-4:
        return 1.0 + h * h / 3.0 + 2.0 * h ** 4 / 15.0   # series, truncation < 1e-24
    return np.tan(h) / h


def true_vertical_depth(survey_md, inc_deg, azi_deg, md_query):
    """TVD (m) at md_query by the minimum-curvature method."""
    md = np.asarray(survey_md, float)
    inc = np.radians(np.asarray(inc_deg, float))
    azi = np.radians(np.asarray(azi_deg, float))
    q_in = np.asarray(md_query, float)
    q = np.atleast_1d(q_in).ravel()
    if not (md.shape == inc.shape == azi.shape) or md.ndim != 1 or md.size < 2:
        raise ValueError("survey arrays must be 1-D, equal length, >= 2 stations")
    if not (np.all(np.isfinite(md)) and np.all(np.isfinite(inc)) and np.all(np.isfinite(azi))):
        raise ValueError("survey values must be finite")
    if md[0] != 0.0 or np.any(np.diff(md) <= 0):
        raise ValueError("survey must start at MD 0 and increase strictly")
    if np.any(inc < 0) or np.any(inc >= np.pi):
        raise ValueError("inclination must be in [0, 180) degrees")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or np.any(q > md[-1]):
        raise ValueError("query depths must lie within the survey")
    # cumulative TVD at stations
    tvd_st = np.zeros(md.size)
    units = [_unit(inc[i], azi[i]) for i in range(md.size)]
    betas = np.zeros(md.size - 1)
    for i in range(md.size - 1):
        # atan2 keeps full precision for tiny doglegs (arccos of a dot product does not)
        betas[i] = np.arctan2(np.linalg.norm(np.cross(units[i], units[i + 1])), units[i] @ units[i + 1])
        if betas[i] >= np.pi - 1e-6:
            raise ValueError("dogleg of 180 degrees between stations: path undefined")
        dmd = md[i + 1] - md[i]
        tvd_st[i + 1] = tvd_st[i] + 0.5 * dmd * (units[i][2] + units[i + 1][2]) * _ratio_factor(betas[i])
    out = np.empty(q.size)
    for j, m in enumerate(q):
        i = min(np.searchsorted(md, m, side="right") - 1, md.size - 2)
        dmd_seg = md[i + 1] - md[i]
        f = (m - md[i]) / dmd_seg
        b = betas[i]
        t1, t2 = units[i], units[i + 1]
        # tangent on the arc; the sinc form is exact for every dogleg, including 0
        tq = ((1 - f) * _sinc((1 - f) * b) * t1 + f * _sinc(f * b) * t2) / _sinc(b)
        dm = m - md[i]
        out[j] = tvd_st[i] + 0.5 * dm * (t1[2] + tq[2]) * _ratio_factor(f * b)
    return float(out[0]) if q_in.ndim == 0 else out.reshape(q_in.shape)


# ------------------------------------------------------------ step 2: layer integrals R and S
def layer_integrals(layer_top_tvd, layer_k, z):
    """Thermal resistance R(z) = int_0^z dz'/k and S(z) = int_0^z z'/k dz'."""
    tops = np.asarray(layer_top_tvd, float)
    k = np.asarray(layer_k, float)
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    if tops.ndim != 1 or tops.shape != k.shape or tops.size < 1:
        raise ValueError("layer arrays must be 1-D with equal length")
    if not np.all(np.isfinite(tops)):
        raise ValueError("layer tops must be finite")
    if tops[0] != 0.0 or np.any(np.diff(tops) <= 0):
        raise ValueError("layer tops must start at 0 and increase strictly")
    if np.any(~np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("conductivities must be positive")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("depths must be >= 0")
    bottoms = np.append(tops[1:], np.inf)
    R = np.zeros(zz.size)
    S = np.zeros(zz.size)
    for top, bot, kk in zip(tops, bottoms, k):
        lo = np.clip(zz, top, bot) - top
        a = top
        b = top + lo
        R += (b - a) / kk
        S += (b - a) * (b + a) / (2 * kk)        # factored: exactly 0 in the layers below z
    if z_in.ndim == 0:
        return float(R[0]), float(S[0])
    return R.reshape(z_in.shape), S.reshape(z_in.shape)


# ------------------------------------------------------------ step 3: half-space paleoclimate perturbation
def _erfc_array(x):
    """Element-wise complementary error function (numpy + math only)."""
    f = np.frompyfunc(math.erfc, 1, 1)
    return np.asarray(f(np.asarray(x, float)), dtype=float)


SECONDS_PER_YEAR = 365.25 * 86400.0


def paleoclimate_perturbation(z, t_years, dT, kappa):
    """Present-day temperature perturbation (K) at depth z (m) from a
    piecewise-constant surface-temperature history."""
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    t = np.asarray(t_years, float)
    d = np.asarray(dT, float)
    if t.ndim != 1 or t.shape != d.shape or t.size < 1:
        raise ValueError("history arrays must be 1-D with equal length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(d))):
        raise ValueError("history values must be finite")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("history times must be positive and increasing")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("diffusivity must be positive")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("depths must be >= 0")
    ts = t * SECONDS_PER_YEAR
    edges = np.concatenate(([0.0], ts))
    out = np.zeros(zz.size)
    for i in range(t.size):
        t_new, t_old = edges[i], edges[i + 1]
        e_old = _erfc_array(zz / (2 * np.sqrt(kappa * t_old)))
        if t_new == 0.0:
            e_new = np.zeros_like(zz)       # limit z / sqrt(kappa * 0) -> inf
        else:
            e_new = _erfc_array(zz / (2 * np.sqrt(kappa * t_new)))
        out += d[i] * (e_old - e_new)
    return float(out[0]) if z_in.ndim == 0 else out.reshape(z_in.shape)


# ------------------------------------------------------------ step 4: joint least-squares estimate
def fit_heat_flow(T, R, S, P, A):
    """Least-squares T = T0 + q0 R - A S + g P  ->  (T0, q0, g) + errors."""
    T, R, S, P = (np.asarray(v, float).ravel() for v in (T, R, S, P))
    n = T.size
    if not (R.size == S.size == P.size == n):
        raise ValueError("arrays must have equal length")
    if n < 4:
        raise ValueError("need at least 4 temperatures")
    if not np.all(np.isfinite(np.concatenate([T, R, S, P]))) or not np.isfinite(A) or A < 0:
        raise ValueError("invalid input")
    y = T + A * S
    G = np.column_stack([np.ones(n), R, P])
    # stated criterion: a zero column, or smallest singular value < 1e-8 after
    # scaling each column to unit Euclidean length
    norms = np.linalg.norm(G, axis=0)
    if np.any(norms == 0.0) or np.linalg.svd(G / norms, compute_uv=False)[-1] < 1e-8:
        raise ValueError("T0, q0 and amplitude are not separately resolvable (rank-deficient design)")
    GtG = G.T @ G
    m = np.linalg.solve(GtG, G.T @ y)
    res = y - G @ m
    s2 = float(res @ res) / (n - 3)
    C = s2 * np.linalg.inv(GtG)
    return {"T0": float(m[0]), "q0": float(m[1]), "amplitude": float(m[2]),
            "sigma_T0": float(np.sqrt(C[0, 0])), "sigma_q0": float(np.sqrt(C[1, 1])),
            "sigma_amplitude": float(np.sqrt(C[2, 2])),
            "rms_residual": float(np.sqrt(np.mean(res**2)))}


# ------------------------------------------------------------ step 5: layered paleoclimate perturbation
def _log_surface_response(z, lay, s, tops, k, rho_c):
    """log F(z, s): Laplace transform (in time) of the departure at depth z for a unit departure applied
    at the surface, F(0, s) = 1, in the layered column (z: depths, lay: their layer indices, s: complex)."""
    nl = tops.size
    kap = k / rho_c
    q = np.sqrt(s[None, :] / kap[:, None])                  # (layers, nodes), Re q > 0
    kq = k[:, None] * q
    h = np.diff(tops)
    # admittance Y = k F' / F at the top of each layer, from the bottom up (continuity of F and k F'):
    # the bottom half-space keeps only the decaying solution, Y = -k q
    Y = np.empty((nl, s.size), complex)
    Y[-1] = -kq[-1]
    for i in range(nl - 2, -1, -1):
        th = np.tanh(q[i] * h[i])
        Y[i] = (Y[i + 1] - kq[i] * th) / (1.0 - Y[i + 1] * th / kq[i])
    # F at the layer tops: F_{i+1} / F_i = 1 / (cosh(q h) - r sinh(q h)) with r = Y_{i+1} / (k_i q_i),
    # written as 2 e^{-q h} / ((1 + e^{-2qh}) - r (1 - e^{-2qh})): no cancellation, no overflow
    lFtop = np.zeros((nl, s.size), complex)
    for i in range(nl - 1):
        e2 = np.exp(-2.0 * q[i] * h[i])
        r = Y[i + 1] / kq[i]
        lFtop[i + 1] = lFtop[i] - q[i] * h[i] + np.log(2.0 / ((1.0 + e2) - r * (1.0 - e2)))
    out = np.empty((z.size, s.size), complex)
    for i in np.unique(lay):
        sel = np.where(lay == i)[0]
        d = (z[sel] - tops[i])[:, None]
        if i == nl - 1:
            out[sel] = lFtop[i][None, :] - q[i][None, :] * d
        else:
            # inside the layer: F(d) / F_top = (cosh(q(h-d)) - r sinh(q(h-d))) / (cosh(qh) - r sinh(qh))
            r = (Y[i + 1] / kq[i])[None, :]
            e2a = np.exp(-2.0 * q[i][None, :] * (h[i] - d))
            e2b = np.exp(-2.0 * q[i] * h[i])[None, :]
            num = (1.0 + e2a) - r * (1.0 - e2a)
            den = (1.0 + e2b) - r * (1.0 - e2b)
            out[sel] = lFtop[i][None, :] - q[i][None, :] * d + np.log(num / den)
    return out


def _step_response(z, lay, tau, tops, k, rho_c, M=32):
    """Departure now at depth z after the surface was raised by 1 K a time tau (s) ago: inverse Laplace
    transform of F / s at tau on the fixed Talbot contour (Abate and Valko 2004)."""
    r = 2.0 * M / (5.0 * tau)
    th = np.arange(1, M) * np.pi / M
    cot = np.cos(th) / np.sin(th)
    sig = r * th * (cot + 1j)
    beta = th + (th * cot - 1.0) * cot
    s = np.concatenate(([r + 0j], sig))
    terms = np.exp(_log_surface_response(z, lay, s, tops, k, rho_c) - np.log(s)[None, :] + s[None, :] * tau)
    val = 0.5 * terms[:, 0].real + np.sum((terms[:, 1:] * (1.0 + 1j * beta)[None, :]).real, axis=1)
    return r / M * val


def layered_paleoclimate_perturbation(z, layer_tops, layer_k, rho_c, t_years, dT):
    """Present-day departure (K) from the steady state at depth z in a layered column, for a
    piecewise-constant surface-temperature history."""
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    tops = np.asarray(layer_tops, float)
    k = np.asarray(layer_k, float)
    t = np.asarray(t_years, float)
    d = np.asarray(dT, float)
    if tops.ndim != 1 or k.ndim != 1 or tops.shape != k.shape or tops.size < 1:
        raise ValueError("layer arrays must be 1-D with equal length")
    if not np.all(np.isfinite(tops)) or tops[0] != 0.0 or np.any(np.diff(tops) <= 0):
        raise ValueError("layer tops must start at 0 and increase strictly")
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("conductivities must be finite and positive")
    if not (np.isfinite(rho_c) and rho_c > 0):
        raise ValueError("volumetric heat capacity must be finite and positive")
    if t.ndim != 1 or t.shape != d.shape or t.size < 1:
        raise ValueError("history arrays must be 1-D with equal length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(d))):
        raise ValueError("history values must be finite")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("history times must be positive and increasing")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("depths must be >= 0")
    # The departure obeys rho_c dT/dt = d/dz (k dT/dz) with T and k dT/dz continuous at the interfaces
    # and the surface history as boundary value; the erfc half-space formula holds only for a uniform
    # column. By superposition it is a sum of step responses U(z, tau) (surface raised by 1 K a time tau
    # ago), each obtained from its Laplace transform F(z, s) / s, which the layers give in closed form.
    lay = np.searchsorted(tops, zz, side="right") - 1
    ts = t * SECONDS_PER_YEAR
    edges = np.concatenate(([0.0], ts))
    inside = zz > 0
    out = np.zeros(zz.size)
    if np.any(inside):
        zi, li = zz[inside], lay[inside]
        U = {}
        for tau in edges[1:]:
            U[tau] = _step_response(zi, li, tau, tops, k, float(rho_c))
        acc = np.zeros(zi.size)
        for i in range(t.size):
            u_old = U[edges[i + 1]]
            u_new = U[edges[i]] if edges[i] > 0 else 0.0     # U(z > 0, 0) = 0
            acc += d[i] * (u_old - u_new)
        out[inside] = acc
    out[~inside] = d[0]                                       # at the surface: the present departure
    dT_z = float(out[0]) if z_in.ndim == 0 else out.reshape(z_in.shape)
    return dT_z


# ------------------------------------------------------------ step 6: joint inversion with the end of the glaciation
def _dg_profile_fit(y, R, P):
    """Ordinary least squares of y on [1, R, P]: (coefficients, residual vector)."""
    G = np.column_stack([np.ones(y.size), R, P])
    m, *_ = np.linalg.lstsq(G, y, rcond=None)
    return m, y - G @ m


def deglaciation_heat_flow_inversion(survey_md, survey_inc, survey_azi, log_md, log_temp,
                                     layer_top_md, layer_k, heat_production, rho_c,
                                     hist_t_years, hist_dT_shape, t_end_bounds, z_min):
    """Steady heat flow, surface temperature, glacial amplitude and end of the glaciation from one log."""
    log_md = np.asarray(log_md, float)
    log_temp = np.asarray(log_temp, float)
    if log_md.ndim != 1 or log_md.shape != log_temp.shape:
        raise ValueError("log_md and log_temp must be 1-D arrays of equal length")
    if not np.all(np.isfinite(log_temp)):
        raise ValueError("temperatures must be finite")
    shape = np.asarray(hist_dT_shape, float)
    t_known = np.asarray(hist_t_years, float)
    if shape.ndim != 1 or shape.size < 2 or t_known.ndim != 1 or t_known.size != shape.size - 1:
        raise ValueError("hist_dT_shape needs m >= 2 entries and hist_t_years m - 1 entries (1-D)")
    if not (np.all(np.isfinite(shape)) and np.all(np.isfinite(t_known))):
        raise ValueError("history values must be finite")
    if t_known[0] <= 0 or np.any(np.diff(t_known) <= 0):
        raise ValueError("hist_t_years must be positive and strictly increasing")
    if shape[0] == shape[1]:
        raise ValueError("t_end has no influence when the first two shape values are equal")
    bounds = np.asarray(t_end_bounds, float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError("t_end_bounds must be two finite numbers")
    t_lo, t_hi = float(bounds[0]), float(bounds[1])
    if not (0.0 < t_lo < t_hi < t_known[0]):
        raise ValueError("t_end_bounds must satisfy 0 < t_lo < t_hi < hist_t_years[0]")
    A = float(heat_production)
    if not np.isfinite(A) or A < 0:
        raise ValueError("heat production must be finite and >= 0")
    z_all = true_vertical_depth(survey_md, survey_inc, survey_azi, log_md)
    tops = true_vertical_depth(survey_md, survey_inc, survey_azi, np.asarray(layer_top_md, float))
    keep = z_all >= z_min
    n = int(np.count_nonzero(keep))
    if n < 5:
        raise ValueError("fewer than 5 readings with TVD >= z_min")
    z, T = z_all[keep], log_temp[keep]
    R, S = layer_integrals(tops, layer_k, z)
    R, S = np.asarray(R, float), np.asarray(S, float)
    y = T + A * S
    ds = float(shape[0] - shape[1])
    # P(z; t) = P_fix(z) + (s0 - s1) U(z, t): P_fix is the history with interval 0 merged into interval 1
    P_fix = np.asarray(layered_paleoclimate_perturbation(
        z, tops, layer_k, rho_c, np.concatenate(([t_lo], t_known)), np.concatenate(([shape[1]], shape[1:]))), float)
    cache = {}

    def U(t):
        t = float(t)
        if t not in cache:
            cache[t] = np.asarray(layered_paleoclimate_perturbation(z, tops, layer_k, rho_c, [t], [1.0]), float)
        return cache[t]

    def P(t):
        return P_fix + ds * U(t)

    def dPdt(t):
        h = 1e-3 * t
        u = [U(t + j * h) for j in (-2, -1, 1, 2)]
        return ds * (u[0] - 8.0 * u[1] + 8.0 * u[2] - u[3]) / (12.0 * h)

    def rss(t):
        _, r = _dg_profile_fit(y, R, P(t))
        return float(r @ r)

    # global minimum of the profile on the closed interval: every interior grid minimum is refined over its two
    # neighbouring cells, and an end value not above its neighbour over the end cell, where the minimum can still
    # be interior; the end point is the minimum only if no refined value is below it
    grid = np.geomspace(t_lo, t_hi, 121)
    vals = np.array([rss(t) for t in grid])
    m_g = grid.size - 1
    cells = [(i - 1, i + 1, i) for i in range(1, m_g) if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]]
    if vals[0] <= vals[1]:
        cells.append((0, 1, 0))
    if vals[m_g] <= vals[m_g - 1]:
        cells.append((m_g - 1, m_g, m_g))
    best_t, best_v = None, np.inf
    for i, j, c in cells:
        res = minimize_scalar(rss, bounds=(grid[i], grid[j]), method="bounded",
                              options={"xatol": 1e-10 * grid[c], "maxiter": 300})
        if res.fun < best_v:
            best_t, best_v = float(res.x), float(res.fun)
    if best_t is None or min(vals[0], vals[-1]) <= best_v:
        raise ValueError("the minimum of the profile RSS over [t_lo, t_hi] lies at an end point")
    t_hat = best_t
    # Gauss-Newton polish of the stationarity condition (variable projection), kept inside the bracket
    for _ in range(3):
        m, r = _dg_profile_fit(y, R, P(t_hat))
        G = np.column_stack([np.ones(n), R, P(t_hat)])
        jt = m[2] * dPdt(t_hat)
        jt_perp = jt - G @ np.linalg.lstsq(G, jt, rcond=None)[0]
        step = float(jt_perp @ r) / float(jt_perp @ jt_perp)
        if not np.isfinite(step) or abs(step) > 1e-3 * t_hat or not (t_lo < t_hat + step < t_hi):
            break
        t_hat = t_hat + step
        if abs(step) < 1e-12 * t_hat:
            break
    if not (t_lo < t_hat < t_hi):
        raise ValueError("the minimum of the profile RSS over [t_lo, t_hi] lies at an end point")
    P_hat = P(t_hat)
    G = np.column_stack([np.ones(n), R, P_hat])
    norms = np.linalg.norm(G, axis=0)
    if np.any(norms == 0.0) or np.linalg.svd(G / norms, compute_uv=False)[-1] < 1e-8:
        raise ValueError("T0, q0 and g are not separately resolvable")
    m, r = _dg_profile_fit(y, R, P_hat)
    rss_min = float(r @ r)
    T0, q0, g = (float(v) for v in m)
    # Wald covariance with the Jacobian [1, R, P, g dP/dt_end] (t_end in years), s^2 = RSS_min / (n - 4)
    J = np.column_stack([np.ones(n), R, P_hat, g * dPdt(t_hat)])
    s2 = rss_min / (n - 4)
    C = s2 * np.linalg.inv(J.T @ J)
    sig = np.sqrt(np.diag(C))
    # profile-likelihood 95 % interval: nearest crossings of the F threshold on each side of t_hat
    thr = rss_min * (1.0 + stats.f.ppf(0.95, 1, n - 4) / (n - 4))

    def crossing(direction):
        bound = t_lo if direction < 0 else t_hi
        step = 0.05 * sig[3]
        a = t_hat
        while True:
            b = a + direction * step
            if (b - bound) * direction >= 0:
                b = bound
            if rss(b) >= thr:
                lo_, hi_ = min(a, b), max(a, b)
                return brentq(lambda t: rss(t) - thr, lo_, hi_, xtol=1e-11 * t_hat, rtol=1e-15, maxiter=300)
            if b == bound:
                raise ValueError("the profile-likelihood interval is not contained in [t_lo, t_hi]")
            a = b
            step = min(1.25 * step, 0.5 * sig[3])

    t_low = float(crossing(-1))
    t_high = float(crossing(+1))
    return {"T0": T0, "q0": q0, "amplitude": g, "t_end": float(t_hat),
            "sigma_T0": float(sig[0]), "sigma_q0": float(sig[1]), "sigma_amplitude": float(sig[2]),
            "sigma_t_end": float(sig[3]), "t_end_low": t_low, "t_end_high": t_high,
            "rms_residual": float(np.sqrt(rss_min / n)), "n_used": n}


# ------------------------------------------------------------ step 7: regional maximum-likelihood inversion
_RG_KEYS = ("survey_md", "survey_inc", "survey_azi", "log_md", "log_temp", "layer_top_md", "layer_k",
            "heat_production", "rho_c", "z_min")


def _rg_best_g(co):
    """Global maximum over g of -sum_h (n_h / 2) log(a_h g^2 - 2 b_h g + c_h): (g, value without constants).

    Q_h = a_h (g - g_h)^2 + m_h with g_h = b_h / a_h and m_h = c_h - b_h^2 / a_h > 0. Every stationary point lies
    between the smallest and the largest g_h (the derivative is positive below and negative above them). In the
    scaled variable x = (g - mid) / half these points are the real roots in [-1, 1] of the polynomial
    sum_h n_h (x - x_h) / v_h prod_{j != h} (1 + (x - x_j)^2 / v_j), v_h = m_h / (a_h half^2); each root is polished
    by Newton steps on the stationarity condition. When the holes' widths sqrt(m_h / a_h) differ by many orders of
    magnitude the polynomial roots can be inaccurate, so the sign changes of the derivative on a grid refined around
    every hole's optimum (offsets of 1e-3 to 1e3 widths) are added, located with Brent's root finder; the best
    candidate is returned."""
    gh = np.array([b / a for a, b, c, n in co])
    lo, hi = float(gh.min()), float(gh.max())

    def value(g):
        return -sum(0.5 * n * np.log((a * g * g - 2.0 * b * g + c) / n) for a, b, c, n in co)

    if hi - lo <= 1e-14 * (abs(lo) + abs(hi)):
        return lo, float(value(lo))
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    Pn = np.polynomial.polynomial
    xh = (gh - mid) / half
    vh = np.array([(c - b * b / a) / (a * half * half) for a, b, c, n in co])
    num = np.zeros(1)
    for i, (a, b, c, n) in enumerate(co):
        term = np.array([-xh[i], 1.0]) * (n / vh[i])
        for j in range(len(co)):
            if j != i:
                term = Pn.polymul(term, [1.0 + xh[j] ** 2 / vh[j], -2.0 * xh[j] / vh[j], 1.0 / vh[j]])
        num = Pn.polyadd(num, term)

    def grad(g):
        return sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)

    def hess(g):
        return sum(n * (a * (a * g * g - 2.0 * b * g + c) - 2.0 * (a * g - b) ** 2) / (a * g * g - 2.0 * b * g + c) ** 2
                   for a, b, c, n in co)

    cand = [lo, hi]
    for z in Pn.polyroots(num):
        if abs(z.imag) <= 1e-3 * (1.0 + abs(z.real)) and abs(z.real) <= 1.0 + 1e-3:
            g = min(max(mid + half * float(z.real), lo), hi)
            for _ in range(8):           # Newton polish on the stationarity condition
                hg = hess(g)
                if hg == 0.0:
                    break
                gn = g - grad(g) / hg
                if not (lo <= gn <= hi) or gn == g:
                    break
                g = gn
            cand.append(g)
    wh = np.sqrt(np.maximum(vh, 0.0)) * half
    offs = np.concatenate((-np.geomspace(1e3, 1e-3, 61), [0.0], np.geomspace(1e-3, 1e3, 61)))
    pts = np.unique(np.clip(np.concatenate([np.linspace(lo, hi, 201)] + [g0 + w0 * offs for g0, w0 in zip(gh, wh)]),
                            lo, hi))
    ca = np.array([[a, b, c, n] for a, b, c, n in co], float)
    dv = np.sum(ca[:, 3:4] * (ca[:, 0:1] * pts - ca[:, 1:2])
                / (ca[:, 0:1] * pts * pts - 2.0 * ca[:, 1:2] * pts + ca[:, 2:3]), axis=0)
    for i in np.nonzero((dv[:-1] < 0.0) & (dv[1:] >= 0.0))[0]:
        cand.append(brentq(grad, pts[i], pts[i + 1], xtol=1e-15 * max(1.0, abs(pts[i])), rtol=1e-15, maxiter=300)
                    if dv[i + 1] > 0.0 else float(pts[i + 1]))
    vals = [value(g) for g in cand]
    k = int(np.argmax(vals))
    return cand[k], float(vals[k])


def regional_deglaciation_inversion(holes, hist_t_years, hist_dT_shape, t_end_bounds):
    """Maximum-likelihood regional estimate of the glacial amplitude g and the end of the glaciation t_end."""
    if not isinstance(holes, (list, tuple)) or len(holes) < 2:
        raise ValueError("holes must be a list of at least 2 boreholes")
    for hd in holes:
        if not isinstance(hd, dict) or set(hd) != set(_RG_KEYS):
            raise ValueError("every hole must be a dict with exactly the keys %s" % (_RG_KEYS,))
    shape = np.asarray(hist_dT_shape, float)
    t_known = np.asarray(hist_t_years, float)
    if shape.ndim != 1 or shape.size < 2 or t_known.ndim != 1 or t_known.size != shape.size - 1:
        raise ValueError("hist_dT_shape needs m >= 2 entries and hist_t_years m - 1 entries (1-D)")
    if not (np.all(np.isfinite(shape)) and np.all(np.isfinite(t_known))):
        raise ValueError("history values must be finite")
    if t_known[0] <= 0 or np.any(np.diff(t_known) <= 0):
        raise ValueError("hist_t_years must be positive and strictly increasing")
    if shape[0] == shape[1]:
        raise ValueError("t_end has no influence when the first two shape values are equal")
    bounds = np.asarray(t_end_bounds, float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError("t_end_bounds must be two finite numbers")
    t_lo, t_hi = float(bounds[0]), float(bounds[1])
    if not (0.0 < t_lo < t_hi < t_known[0]):
        raise ValueError("t_end_bounds must satisfy 0 < t_lo < t_hi < hist_t_years[0]")
    ds = float(shape[0] - shape[1])
    data = []
    for hd in holes:
        log_md = np.asarray(hd["log_md"], float)
        log_temp = np.asarray(hd["log_temp"], float)
        if log_md.ndim != 1 or log_md.shape != log_temp.shape:
            raise ValueError("log_md and log_temp must be 1-D arrays of equal length")
        if not np.all(np.isfinite(log_temp)):
            raise ValueError("temperatures must be finite")
        A = float(hd["heat_production"])
        if not np.isfinite(A) or A < 0:
            raise ValueError("heat production must be finite and >= 0")
        true_vertical_depth(hd["survey_md"], hd["survey_inc"], hd["survey_azi"], log_md)   # (validation only)
        z_all = np.array(log_md, float)                                   # MUTANT: MD used as depth
        tops = np.array(hd["layer_top_md"], float)                        # MUTANT: MD used as depth
        keep = np.asarray(z_all >= hd["z_min"])
        n = int(np.count_nonzero(keep))
        if n < 6:
            raise ValueError("fewer than 6 readings with TVD >= z_min in a hole")
        z, T = np.asarray(z_all, float)[keep], log_temp[keep]
        R, S = layer_integrals(tops, hd["layer_k"], z)
        R, S = np.asarray(R, float), np.asarray(S, float)
        G2 = np.column_stack([np.ones(n), R])
        nrm = np.linalg.norm(G2, axis=0)
        if np.any(nrm == 0.0) or np.linalg.svd(G2 / nrm, compute_uv=False)[-1] < 1e-8:
            raise ValueError("T0 and q0 of a hole are not separately resolvable")
        Q2, _ = np.linalg.qr(G2)
        y = T + A * S
        yp = y - Q2 @ (Q2.T @ y)
        P_fix = np.asarray(layered_paleoclimate_perturbation(
            z, tops, hd["layer_k"], hd["rho_c"], np.concatenate(([t_lo], t_known)),
            np.concatenate(([shape[1]], shape[1:]))), float)
        data.append({"n": n, "z": z, "tops": tops, "k": hd["layer_k"], "rc": hd["rho_c"], "R": R, "y": y,
                     "Q2": Q2, "yp": yp, "c": float(yp @ yp), "P_fix": P_fix, "cache": {}})

    def U(d, t):
        t = float(t)
        if t not in d["cache"]:
            d["cache"][t] = np.asarray(layered_paleoclimate_perturbation(d["z"], d["tops"], d["k"], d["rc"], [t], [1.0]),
                                       float)
        return d["cache"][t]

    def P(d, t):
        return d["P_fix"] + ds * U(d, t)

    def dPdt(d, t):
        h = 1e-3 * t
        u = [U(d, t + j * h) for j in (-2, -1, 1, 2)]
        return ds * (u[0] - 8.0 * u[1] + 8.0 * u[2] - u[3]) / (12.0 * h)

    const = -0.5 * sum(d["n"] for d in data) * (np.log(2.0 * np.pi) + 1.0)

    def profile(t):
        co = []
        for d in data:
            p = P(d, t)
            p = p - d["Q2"] @ (d["Q2"].T @ p)
            co.append((float(p @ p), float(p @ d["yp"]), d["c"], d["n"]))
        g, v = _rg_best_g(co)
        return v + const, g

    def lp(t):
        return profile(t)[0]

    # global maximum of the profile log-likelihood on the closed interval: every interior grid maximum is refined
    # over its two neighbouring cells, and an end value not below its neighbour over the end cell, where the
    # maximum can still be interior; the end point is the maximum only if no refined value exceeds it
    grid = np.geomspace(t_lo, t_hi, 121)
    vals = np.array([lp(t) for t in grid])
    m_g = grid.size - 1
    cells = [(i - 1, i + 1, i) for i in range(1, m_g) if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]]
    if vals[0] >= vals[1]:
        cells.append((0, 1, 0))
    if vals[m_g] >= vals[m_g - 1]:
        cells.append((m_g - 1, m_g, m_g))
    best_t, best_v = None, -np.inf
    for i, j, c in cells:
        res = minimize_scalar(lambda t: -lp(t), bounds=(grid[i], grid[j]), method="bounded",
                              options={"xatol": 1e-10 * grid[c], "maxiter": 300})
        if -res.fun > best_v:
            best_t, best_v = float(res.x), float(-res.fun)
    if best_t is None or max(vals[0], vals[-1]) >= best_v or not (t_lo < best_t < t_hi):
        raise ValueError("the maximum of the profile likelihood over [t_lo, t_hi] lies at an end point")
    t_hat = best_t
    l_max, g = profile(t_hat)
    # per-hole estimates, ML noise levels and the expected Fisher information at the estimate
    H = len(data)
    npar = 2 * H + 2
    info = np.zeros((npar, npar))
    out_holes = []
    for h, d in enumerate(data):
        Ph = P(d, t_hat)
        G2 = np.column_stack([np.ones(d["n"]), d["R"]])
        m, *_ = np.linalg.lstsq(G2, d["y"] - g * Ph, rcond=None)
        r = d["y"] - g * Ph - G2 @ m
        s2 = float(r @ r) / d["n"]
        J = np.zeros((d["n"], npar))
        J[:, 2 * h] = 1.0
        J[:, 2 * h + 1] = d["R"]
        J[:, -2] = Ph
        J[:, -1] = g * dPdt(d, t_hat)
        info += J.T @ J / s2
        out_holes.append({"T0": float(m[0]), "q0": float(m[1]), "noise_sd": float(np.sqrt(s2)), "n_used": d["n"]})
    sc = 1.0 / np.sqrt(np.diag(info))
    cov = np.linalg.inv(info * np.outer(sc, sc)) * np.outer(sc, sc)
    sig = np.sqrt(np.diag(cov))
    for h, oh in enumerate(out_holes):
        oh["sigma_T0"] = float(sig[2 * h])
        oh["sigma_q0"] = float(sig[2 * h + 1])
    # 95 % likelihood-ratio interval: nearest crossings of l_max - chi2_0.95(1) / 2 on each side of t_hat
    thr = l_max - 0.5 * stats.chi2.ppf(0.95, 1)

    def crossing(direction):
        bound = t_lo if direction < 0 else t_hi
        step = 0.05 * sig[-1]
        a = t_hat
        while True:
            b = a + direction * step
            if (b - bound) * direction >= 0:
                b = bound
            if lp(b) <= thr:
                lo_, hi_ = min(a, b), max(a, b)
                return brentq(lambda t: lp(t) - thr, lo_, hi_, xtol=1e-11 * t_hat, rtol=1e-15, maxiter=300)
            if b == bound:
                raise ValueError("the likelihood-ratio interval is not contained in [t_lo, t_hi]")
            a = b
            step = min(1.25 * step, 0.5 * sig[-1])

    t_low = float(crossing(-1))
    t_high = float(crossing(+1))
    return {"g": float(g), "t_end": float(t_hat), "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
            "t_end_low": t_low, "t_end_high": t_high, "log_likelihood": float(l_max), "holes": out_holes}
