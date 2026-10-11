"""Mutant (step 7): Conditional (Cochrane-Orcutt) likelihood: the first reading of each hole is conditioned on and dropped, so the likelihood has n_h - 1 innovations and no log(1 - phi^2) term, instead of the exact joint density of all readings."""
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

_RG_KEYS = ("survey_md", "survey_inc", "survey_azi", "log_md", "log_temp", "layer_top_md", "layer_k",
            "heat_production", "rho_c", "z_min")
_RG_PHI = np.linspace(-0.995, 0.995, 399)


def _rg_white(v, phi):
    """Prais-Winsten transform along axis 0 (readings by increasing MD)."""
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0] = np.sqrt(1.0 - phi * phi) * v[0]
    out[1:] = v[1:] - phi * v[:-1]
    return out


class _RgHole:
    """Gram polynomials in phi of the whitened [q1, q2, y_c, P_perp] of one hole at one t_end."""

    def __init__(self, d, P):
        Q2 = d["Q2"]
        pp = P - Q2 @ (Q2.T @ P)
        yp = d["yp"]
        self.g0 = float(pp @ yp) / float(pp @ pp)
        V = np.column_stack([Q2, yp - self.g0 * pp, pp])
        self.C0 = V[1:].T @ V[1:]                      # MUTANT: conditional likelihood (first reading dropped)
        self.C1 = V[1:].T @ V[:-1] + V[:-1].T @ V[1:]
        self.C2 = V[:-1].T @ V[:-1]
        self.n = d["n"] - 1
        # coefficient arrays on the phi grid
        self.a, self.b, self.c = self.abc(_RG_PHI)
        self.lp1 = 0.0 * _RG_PHI                       # MUTANT: no determinant term
        self.flat = tuple([float(C[i, j]) for i in range(4) for j in range(4)] for C in (self.C0, self.C1, self.C2))

    def abc(self, phi):
        """(a, b, c) of Q(phi, g) = c - 2 b (g - g0) + a (g - g0)^2 for an array of phi."""
        phi = np.asarray(phi, float)[..., None, None]
        G = self.C0 - phi * self.C1 + phi * phi * self.C2
        A11, A12, A22 = G[..., 0, 0], G[..., 0, 1], G[..., 1, 1]
        det = A11 * A22 - A12 * A12
        By, Bp = G[..., 0:2, 2], G[..., 0:2, 3]

        def quad(u, v):
            return (u[..., 0] * (A22 * v[..., 0] - A12 * v[..., 1]) + u[..., 1] * (A11 * v[..., 1] - A12 * v[..., 0])) / det

        c = G[..., 2, 2] - quad(By, By)
        b = G[..., 2, 3] - quad(By, Bp)
        a = G[..., 3, 3] - quad(Bp, Bp)
        return a, b, c

    def abc1(self, phi):
        """Scalar version of abc in plain floats."""
        p2 = phi * phi
        c0, c1, c2 = self.flat
        G = [c0[k] - phi * c1[k] + p2 * c2[k] for k in range(16)]
        A11, A12, A22 = G[0], G[1], G[5]
        det = A11 * A22 - A12 * A12
        y1, y2, p1, q2 = G[2], G[6], G[3], G[7]
        cyy = G[10] - (y1 * (A22 * y1 - A12 * y2) + y2 * (A11 * y2 - A12 * y1)) / det
        cyp = G[11] - (y1 * (A22 * p1 - A12 * q2) + y2 * (A11 * q2 - A12 * p1)) / det
        cpp = G[15] - (p1 * (A22 * p1 - A12 * q2) + q2 * (A11 * q2 - A12 * p1)) / det
        return cpp, cyp, cyy

    def value(self, phi, g):
        """-(n/2) log Q(phi, g) + (1/2) log(1 - phi^2) for a scalar phi."""
        a, b, c = self.abc1(phi)
        x = g - self.g0
        q = c - 2.0 * b * x + a * x * x
        if not (q > 0.0) or not (abs(phi) < 1.0):
            return -np.inf
        return -0.5 * self.n * np.log(q)               # MUTANT: conditional likelihood

    def grid_values(self, g):
        """Values on the phi grid for an array of g: shape (len(phi grid), len(g))."""
        x = np.atleast_1d(g)[None, :] - self.g0
        q = self.c[:, None] - 2.0 * self.b[:, None] * x + self.a[:, None] * x * x
        with np.errstate(divide="ignore", invalid="ignore"):
            v = -0.5 * self.n * np.log(q) + self.lp1[:, None]
        return np.where(q > 0, v, -np.inf)

    def best_phi(self, g):
        """Maximum over phi of value(phi, g): best grid point refined by bounded Brent over its neighbouring cells,
        every grid maximum within 1 of the best one refined."""
        v = self.grid_values(g)[:, 0]
        m = v.size
        best = (-np.inf, 0.0)
        top = float(np.max(v))
        for i in range(m):
            if v[i] < top - 1.0:
                continue
            if (i > 0 and v[i] < v[i - 1]) or (i < m - 1 and v[i] < v[i + 1]):
                continue
            lo, hi = _RG_PHI[max(i - 1, 0)], _RG_PHI[min(i + 1, m - 1)]
            if i == 0:
                lo = -0.99999
            if i == m - 1:
                hi = 0.99999
            res = minimize_scalar(lambda p: -self.value(p, g), bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-11, "maxiter": 200})
            if -res.fun > best[0]:
                best = (float(-res.fun), float(res.x))
        return best


def _rg_profile_at(hs):
    """Maximum over g and every phi_h of sum_h value_h(phi_h, g) (all other parameters concentrated):
    (value, g, [phi_h])."""
    own = []
    glo, ghi = np.inf, -np.inf
    for hh in hs:
        with np.errstate(divide="ignore", invalid="ignore"):
            gopt = hh.g0 + hh.b / hh.a
            vopt = -0.5 * hh.n * np.log(hh.c - hh.b * hh.b / hh.a) + hh.lp1
        k = int(np.argmax(vopt))
        width = np.sqrt(max((hh.c[k] - hh.b[k] ** 2 / hh.a[k]) / hh.a[k], 0.0))
        own.append((float(gopt[k]), float(width)))
        glo, ghi = min(glo, float(np.min(gopt))), max(ghi, float(np.max(gopt)))
    span = ghi - glo
    glo, ghi = glo - 0.05 * span - 1e-12, ghi + 0.05 * span + 1e-12
    offs = np.concatenate((-np.geomspace(1e3, 1e-3, 61), [0.0], np.geomspace(1e-3, 1e3, 61)))
    pts = np.unique(np.clip(np.concatenate([np.linspace(glo, ghi, 201)] + [g0 + w0 * offs for g0, w0 in own]), glo, ghi))
    tot = np.zeros(pts.size)
    for hh in hs:
        tot += np.max(hh.grid_values(pts), axis=0)

    def total(g):
        return sum(hh.best_phi(g)[0] for hh in hs)

    cand = []
    m = pts.size
    for i in range(m):
        if (i > 0 and tot[i] < tot[i - 1]) or (i < m - 1 and tot[i] < tot[i + 1]):
            continue
        lo, hi = pts[max(i - 1, 0)], pts[min(i + 1, m - 1)]
        res = minimize_scalar(lambda g: -total(g), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-13 * max(1.0, abs(pts[i])), "maxiter": 300})
        cand.append((float(-res.fun), float(res.x)))
    val, g = max(cand)
    phis = [hh.best_phi(g)[1] for hh in hs]
    return val, g, phis


def regional_deglaciation_inversion(holes, hist_t_years, hist_dT_shape, t_end_bounds):
    """Maximum-likelihood regional estimate of the glacial amplitude g and the end of the glaciation t_end with
    stationary AR(1) noise in every hole."""
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
        z_all = true_vertical_depth(hd["survey_md"], hd["survey_inc"], hd["survey_azi"], log_md)
        tops = true_vertical_depth(hd["survey_md"], hd["survey_inc"], hd["survey_azi"],
                                   np.asarray(hd["layer_top_md"], float))
        used = np.nonzero(np.asarray(z_all) >= hd["z_min"])[0]
        n = int(used.size)
        if n < 8:
            raise ValueError("fewer than 8 readings with TVD >= z_min in a hole")
        used = used[np.argsort(log_md[used], kind="stable")]          # the AR(1) order: increasing MD
        z, T = np.asarray(z_all, float)[used], log_temp[used]
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
                     "Q2": Q2, "yp": yp, "P_fix": P_fix, "cache": {}})

    def U(d, t):
        t = float(t)
        if t not in d["cache"]:
            d["cache"][t] = np.asarray(layered_paleoclimate_perturbation(d["z"], d["tops"], d["k"], d["rc"], [t], [1.0]),
                                       float)
        return d["cache"][t]

    def P(d, t):
        return d["P_fix"] + ds * U(d, t)

    def dPdt(d, t):
        h = 1e-2 * t
        u = [U(d, t + j * h) for j in (-2, -1, 1, 2)]
        return ds * (u[0] - 8.0 * u[1] + 8.0 * u[2] - u[3]) / (12.0 * h)

    const = -0.5 * sum((d["n"] - 1) * (np.log(2.0 * np.pi / (d["n"] - 1)) + 1.0) for d in data)   # MUTANT
    memo = {}

    def profile(t):
        t = float(t)
        if t not in memo:
            v, g, phis = _rg_profile_at([_RgHole(d, P(d, t)) for d in data])
            memo[t] = (v + const, g, phis)
        return memo[t]

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
    _, g, phis = profile(t_hat)
    # per-hole estimates (generalized least squares at the estimates), noise parameters, exact log-likelihood and the
    # expected Fisher information of the regression parameters at the estimate
    H = len(data)
    npar = 2 * H + 2
    info = np.zeros((npar, npar))
    out_holes = []
    l_max = 0.0
    for h, d in enumerate(data):
        ph = phis[h]
        Ph = P(d, t_hat)
        G2 = np.column_stack([np.ones(d["n"]), d["R"]])
        m, *_ = np.linalg.lstsq(_rg_white(G2, ph)[1:], _rg_white(d["y"] - g * Ph, ph)[1:], rcond=None)   # MUTANT
        u = _rg_white(d["y"] - g * Ph - G2 @ m, ph)[1:]
        s2 = float(u @ u) / (d["n"] - 1)
        l_max += -0.5 * (d["n"] - 1) * (np.log(2.0 * np.pi * s2) + 1.0)
        J = np.zeros((d["n"], npar))
        J[:, 2 * h] = 1.0
        J[:, 2 * h + 1] = d["R"]
        J[:, -2] = Ph
        J[:, -1] = g * dPdt(d, t_hat)
        Jw = _rg_white(J, ph)[1:]                      # MUTANT: conditional information
        info += Jw.T @ Jw / s2
        out_holes.append({"T0": float(m[0]), "q0": float(m[1]), "phi": float(ph), "innovation_sd": float(np.sqrt(s2)),
                          "n_used": d["n"]})
    sc = 1.0 / np.sqrt(np.diag(info))
    cov = np.linalg.inv(info * np.outer(sc, sc)) * np.outer(sc, sc)
    sig = np.sqrt(np.diag(cov))
    for h, oh in enumerate(out_holes):
        oh["sigma_T0"] = float(sig[2 * h])
        oh["sigma_q0"] = float(sig[2 * h + 1])
    # 95 % likelihood-ratio interval: nearest crossings of l_p(t_hat) - chi2_0.95(1) / 2 on each side of t_hat
    thr = lp(t_hat) - 0.5 * stats.chi2.ppf(0.95, 1)

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
