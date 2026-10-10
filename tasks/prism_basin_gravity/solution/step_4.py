import numpy as np
import math
from scipy.optimize import least_squares

G = 6.6743e-11          # m^3 kg^-1 s^-2 (CODATA 2018)

_GX16, _GW16 = np.polynomial.legendre.leggauss(16)
_GX20, _GW20 = np.polynomial.legendre.leggauss(20)


def _check_bounds(bounds, rho):
    b = np.asarray(bounds, float)
    if b.shape != (6,) or not np.all(np.isfinite(b)):
        raise ValueError("bounds must be six finite numbers x1, x2, y1, y2, z1, z2")
    if not (b[0] < b[1] and b[2] < b[3] and b[4] < b[5]):
        raise ValueError("need x1 < x2, y1 < y2, z1 < z2")
    rho = float(rho)
    if not math.isfinite(rho):
        raise ValueError("rho must be finite")
    return b, rho


def _check_points(points):
    p = np.asarray(points, float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1 or not np.all(np.isfinite(p)):
        raise ValueError("points must be finite, of shape (3,) or (n, 3)")
    return p, single


def _lamina(X, Y, Z, wx, wy):
    # solid angle of the horizontal rectangle at depth Z > 0 below the station, i.e. the vertical attraction
    # / (G rho dz) of a lamina; X = (x1 - x_s, x2 - x_s), Y = (y1 - y_s, y2 - y_s) are (2, n) corner offsets,
    # wx, wy the widths x2 - x1, y2 - y1 (n,), Z (m, n) or (m, 1). Near the rectangle: the corner sum of
    # atan2(x y, Z r) (+ at (x2, y2)). At a horizontal distance of at least the larger width that sum cancels
    # (terms of order 1, result of order w^2 Z / d^3: 1e-9 relative lost 1000 widths away), so there the two
    # triangles of the rectangle are taken by the Van Oosterom-Strackee formula
    # tan(Omega / 2) = a.(b x c) / (|a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|), whose triple product is
    # Z wx wy and whose dot products are all positive that far: no cancellation at any distance.
    X0, X1, Y0, Y1 = X[0][None, :], X[1][None, :], Y[0][None, :], Y[1][None, :]
    Z2 = Z * Z
    tot = 0.0
    for i, x in enumerate((X0, X1)):
        for j, y in enumerate((Y0, Y1)):
            r = np.sqrt(x * x + y * y + Z2)
            tot = tot + (1.0 if (i + j) % 2 == 0 else -1.0) * np.arctan2(x * y, Z * r)
    far = np.hypot(np.maximum(np.maximum(X0, -X1), 0.0), np.maximum(np.maximum(Y0, -Y1), 0.0)) >= np.maximum(wx, wy)
    if not np.any(far):
        return tot
    ra = np.sqrt(X0 * X0 + Y0 * Y0 + Z2)
    rb = np.sqrt(X1 * X1 + Y0 * Y0 + Z2)
    rc = np.sqrt(X1 * X1 + Y1 * Y1 + Z2)
    rd = np.sqrt(X0 * X0 + Y1 * Y1 + Z2)
    num = Z * (wx * wy)
    vo = 2.0 * (np.arctan2(num, ra * rb * rc + (X0 * X1 + Y0 * Y0 + Z2) * rc + (X0 * X1 + Y0 * Y1 + Z2) * rb
                           + (X1 * X1 + Y0 * Y1 + Z2) * ra)
                + np.arctan2(num, ra * rc * rd + (X0 * X1 + Y0 * Y1 + Z2) * rd + (X0 * X0 + Y0 * Y1 + Z2) * rc
                             + (X0 * X1 + Y1 * Y1 + Z2) * ra))
    return np.where(far, vo, tot)


def _basin_forward(xc, yc, xe, ye, h, drho0, lam):
    # vertical attraction at the cell centres (z = 0) of all columns, and its Jacobian
    # dg_i / dh_j = G drho0 exp(-lam h_j) L_ij(h_j), L_ij the solid angle of the bottom of column j seen from
    # station i. Depth panels 0, s, 2s, 4s, ..., h_j with s a sixty-fourth of the smallest cell width (the
    # closest an edge comes to a station is half a width).
    n = xc.size
    wmin = min(np.min(np.diff(xe)), np.min(np.diff(ye)))
    X1 = (xe[:-1][:, None] * np.ones(len(ye) - 1)[None, :]).ravel()
    X2 = (xe[1:][:, None] * np.ones(len(ye) - 1)[None, :]).ravel()
    Y1 = (np.ones(len(xe) - 1)[:, None] * ye[:-1][None, :]).ravel()
    Y2 = (np.ones(len(xe) - 1)[:, None] * ye[1:][None, :]).ravel()
    dX = np.stack([X1[None, :] - xc[:, None], X2[None, :] - xc[:, None]])      # (2, stations, cells)
    dY = np.stack([Y1[None, :] - yc[:, None], Y2[None, :] - yc[:, None]])
    g = np.zeros(n)
    J = np.zeros((n, n))
    for j in range(n):
        hj = h[j]
        wx, wy = np.full(n, X2[j] - X1[j]), np.full(n, Y2[j] - Y1[j])
        e = [0.0]
        s = wmin / 64.0
        while s < hj:
            e.append(s)
            s *= 2.0
        e.append(hj)
        e = np.array(e)
        a, b = e[:-1], e[1:]
        z = ((0.5 * (b - a))[:, None] * _GX20[None, :] + (0.5 * (a + b))[:, None]).ravel()
        w = ((0.5 * (b - a))[:, None] * _GW20[None, :]).ravel()
        L = _lamina(dX[:, :, j], dY[:, :, j], z[:, None], wx, wy)                 # (nodes, stations)
        g += G * drho0 * (w * np.exp(-lam * z)) @ L
        J[:, j] = G * drho0 * math.exp(-lam * hj) * _lamina(dX[:, :, j], dY[:, :, j], np.array([[hj]]), wx, wy)[0]
    return g, J


def invert_basin_depths(x_edges, y_edges, gz_obs, drho0, lam):
    """Depths of a basin of columns (density contrast drho0 exp(-lam z)) from its gravity anomaly.

    Inputs:
      x_edges, y_edges: 1-D float arrays (nx + 1 and ny + 1 entries), strictly increasing, nx ny <= 100.
      gz_obs: float array (nx, ny), the vertical attraction (m s^-2) of all columns at the cell centres on
        the surface z = 0; it is that of a basin whose every depth satisfies 0 < depth <= 3 w_min (w_min
        the smallest cell width, in x or y), depth <= 5e4 m and lam depth <= 2.
      drho0: float, nonzero density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      depths: numpy array (nx, ny) in m, each with a relative error below 1e-8.

    Raises:
      ValueError for invalid edges, nx ny > 100, gz_obs of the wrong shape or non-finite, drho0 zero or
      non-finite, lam outside [0, 1e-2], an observation without the sign of drho0, or (lam > 0) an
      observation of at least 2 pi G |drho0| / lam.
    """
    xe = np.asarray(x_edges, float)
    ye = np.asarray(y_edges, float)
    go = np.asarray(gz_obs, float)
    if xe.ndim != 1 or ye.ndim != 1 or xe.size < 2 or ye.size < 2:
        raise ValueError("cell edges must be 1-D with at least two entries")
    if not (np.all(np.isfinite(xe)) and np.all(np.isfinite(ye))) or np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0):
        raise ValueError("cell edges must be finite and strictly increasing")
    nx, ny = xe.size - 1, ye.size - 1
    if nx * ny > 100:
        raise ValueError("at most 100 cells")
    if go.shape != (nx, ny) or not np.all(np.isfinite(go)):
        raise ValueError("gz_obs must be finite with shape (nx, ny)")
    drho0, lam = float(drho0), float(lam)
    if not (math.isfinite(drho0) and drho0 != 0.0):
        raise ValueError("drho0 must be finite and nonzero")
    if not (math.isfinite(lam) and 0.0 <= lam <= 1e-2):
        raise ValueError("need 0 <= lam <= 1e-2 per m")
    d = go.ravel() / (2.0 * math.pi * G * drho0)        # slab-equivalent thickness (> 0 for a basin)
    if np.any(d <= 0):
        raise ValueError("every observation must have the sign of drho0")
    if lam > 0 and np.any(d * lam >= 1.0):
        raise ValueError("an observation exceeds the attraction of an infinite slab")
    xc = (0.5 * (xe[:-1] + xe[1:])[:, None] * np.ones(ny)[None, :]).ravel()
    yc = (np.ones(nx)[:, None] * 0.5 * (ye[:-1] + ye[1:])[None, :]).ravel()
    gobs = go.ravel()
    gmax = float(np.max(np.abs(gobs)))
    wmin = min(float(np.min(np.diff(xe))), float(np.min(np.diff(ye))))
    # 1. global phase: trust-region reflective least squares (scipy) on the depths, bounded by 1.5 times
    # the largest depth of the stated domain (3 w_min, 5e4 m, 2 / lam). Plain Newton with step halving can
    # overshoot a column far below its depth, where its attraction hardly changes with depth, and stall
    # there (rough basins, shallow columns beside deep ones); the trust region does not.
    hcap = 1.5 * min(5e4, 3.0 * wmin, 2.0 / lam if lam > 0 else math.inf)
    h0 = -np.log1p(-d * lam) / lam if lam > 0 else d.copy()       # infinite-slab thickness under each station
    h0 = np.minimum(h0, hcap * (1.0 - 1e-9))
    cache = {}

    def model(hh):
        key = hh.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = _basin_forward(xc, yc, xe, ye, hh, drho0, lam)
        return cache[key]

    sol = least_squares(lambda hh: (model(hh)[0] - gobs) / gmax, h0, jac=lambda hh: model(hh)[1] / gmax,
                        bounds=(0.0, hcap), method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1000)
    # 2. Newton's method from there (quadratic convergence), the step halved until the depths stay in
    # (0, 5e4] and the residual decreases, until the relative update is below 1e-13 or the residual has
    # reached its rounding level
    h = np.minimum(np.maximum(sol.x, 1e-6 * wmin), 5e4)
    g, Jh = model(h)
    res = g - gobs
    for it in range(50):
        if np.max(np.abs(res)) <= 1e-15 * gmax:
            break
        try:
            step = np.linalg.solve(Jh, res)
        except np.linalg.LinAlgError:
            raise ValueError("the depth iteration met a singular Jacobian")
        rnorm = np.linalg.norm(res)
        t = 1.0
        stalled = False
        while True:
            hn = h - t * step
            if np.all(hn > 0) and np.all(hn <= 5e4):
                gn, Jn = model(hn)
                rn = gn - gobs
                if np.linalg.norm(rn) < rnorm:
                    break
                if np.max(np.abs(hn - h) / h) < 1e-13:
                    stalled = True            # no decrease along the Newton direction: rounding level
                    break
            t *= 0.5
            if t < 1e-12:
                raise ValueError("no depths in (0, 5e4] m reproduce the observations")
        if stalled:
            if np.max(np.abs(res)) > 1e-12 * gmax:
                raise ValueError("the depth iteration stalled")
            break
        dh = np.max(np.abs(hn - h) / hn)
        h, g, Jh, res = hn, gn, Jn, rn
        if dh < 1e-13:
            break
    else:
        raise ValueError("the depth iteration did not converge")
    depths = h.reshape(nx, ny)
    return depths
