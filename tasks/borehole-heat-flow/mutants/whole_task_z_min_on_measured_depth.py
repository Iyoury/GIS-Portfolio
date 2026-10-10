"""Whole-task mutant: Shallow-reading cut applied to measured depth instead of true vertical depth.

Complete solution with this single error in surface_heat_flow.
"""
import numpy as np
import math


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
        S += (b**2 - a**2) / (2 * kk)
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


# ------------------------------------------------------------ step 6: full workflow
def surface_heat_flow(survey_md, survey_inc, survey_azi, log_md, log_temp,
                      layer_top_md, layer_k, heat_production, kappa,
                      hist_t_years, hist_dT_shape, z_min):
    """Paleoclimate-corrected surface heat flow from an inclined borehole."""
    log_md = np.asarray(log_md, float)
    log_temp = np.asarray(log_temp, float)
    if log_md.shape != log_temp.shape or log_md.ndim != 1:
        raise ValueError("log arrays must be 1-D with equal length")
    z = true_vertical_depth(survey_md, survey_inc, survey_azi, log_md)
    tops = true_vertical_depth(survey_md, survey_inc, survey_azi, np.asarray(layer_top_md, float))
    keep = log_md >= z_min                               # MUTANT: cut on MD
    z, T = z[keep], log_temp[keep]
    R, S = layer_integrals(tops, layer_k, z)
    P = paleoclimate_perturbation(z, hist_t_years, hist_dT_shape, kappa)
    fit = fit_heat_flow(T, R, S, P, heat_production)
    fit["n_used"] = int(z.size)
    return fit
