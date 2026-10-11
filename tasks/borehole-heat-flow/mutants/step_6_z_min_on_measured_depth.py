"""Mutant (step 6): Shallow-reading cut applied to measured depth instead of true vertical depth."""
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar


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
    keep = log_md >= z_min                     # MUTANT: cut applied to MD
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
