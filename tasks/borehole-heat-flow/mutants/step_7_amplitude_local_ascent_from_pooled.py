"""Mutant (step 7): Amplitude at fixed t_end from a local fixed-point (iteratively reweighted) ascent started at the pooled least-squares amplitude instead of the global maximum over g: when the likelihood in g is bimodal the ascent can stop at the secondary maximum."""
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

_RG_KEYS = ("survey_md", "survey_inc", "survey_azi", "log_md", "log_temp", "layer_top_md", "layer_k",
            "heat_production", "rho_c", "z_min")


def _rg_best_g(co):
    """Local maximum over g of -sum_h (n_h / 2) log(a_h g^2 - 2 b_h g + c_h) reached by fixed-point (iteratively
    reweighted) ascent from the pooled least-squares amplitude sum_h b_h / sum_h a_h: (g, value without constants)."""
    g = sum(b for a, b, c, n in co) / sum(a for a, b, c, n in co)
    for _ in range(20000):
        w = [n / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co]
        gn = sum(wi * b for wi, (a, b, c, n) in zip(w, co)) / sum(wi * a for wi, (a, b, c, n) in zip(w, co))
        done = abs(gn - g) <= 1e-15 * max(1.0, abs(g))
        g = gn
        if done:
            break
    return g, float(-sum(0.5 * n * np.log((a * g * g - 2.0 * b * g + c) / n) for a, b, c, n in co))


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
        z_all = true_vertical_depth(hd["survey_md"], hd["survey_inc"], hd["survey_azi"], log_md)
        tops = true_vertical_depth(hd["survey_md"], hd["survey_inc"], hd["survey_azi"],
                                   np.asarray(hd["layer_top_md"], float))
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
