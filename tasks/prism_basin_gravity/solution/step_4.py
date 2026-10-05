import numpy as np
import math

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


def _lamina(X, Y, Z):
    # sum over the four corners of atan(x y / (Z r)) (+ at (x2, y2)), the vertical attraction / (G rho dz)
    # of a horizontal lamina at depth Z below the station; X, Y are (2, n) corner offsets, Z (m, n) > 0
    tot = 0.0
    for i in range(2):
        for j in range(2):
            x, y = X[i][None, :], Y[j][None, :]
            r = np.sqrt(x * x + y * y + Z * Z)
            s = 1.0 if (i + j) % 2 == 0 else -1.0
            tot = tot + s * np.arctan2(x * y, Z * r)
    return tot


def _basin_forward(xc, yc, xe, ye, h, drho0, lam, jac=False):
    # vertical attraction at the cell centres (z = 0) of all columns; panels in depth refined toward the
    # top on the scale of a quarter of the smallest cell width (the closest edge to any station)
    n = xc.size
    wmin = min(np.min(np.diff(xe)), np.min(np.diff(ye)))
    X1 = (xe[:-1][:, None] * np.ones(len(ye) - 1)[None, :]).ravel()
    X2 = (xe[1:][:, None] * np.ones(len(ye) - 1)[None, :]).ravel()
    Y1 = (np.ones(len(xe) - 1)[:, None] * ye[:-1][None, :]).ravel()
    Y2 = (np.ones(len(xe) - 1)[:, None] * ye[1:][None, :]).ravel()
    dX = np.stack([X1[None, :] - xc[:, None], X2[None, :] - xc[:, None]])      # (2, stations, cells)
    dY = np.stack([Y1[None, :] - yc[:, None], Y2[None, :] - yc[:, None]])
    g = np.zeros(n)
    J = np.zeros((n, n)) if jac else None
    for j in range(n):
        hj = h[j]
        # breakpoints 0, s, 2s, 4s, ... up to hj with s = wmin / 64
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
        L = _lamina(dX[:, :, j], dY[:, :, j], z[:, None])                      # (nodes, stations)
        g += G * drho0 * (w * np.exp(-lam * z)) @ L
        if jac:
            Lh = _lamina(dX[:, :, j], dY[:, :, j], np.array([[hj]]))[0]
            J[:, j] = G * drho0 * math.exp(-lam * hj) * Lh
    return g, J


def invert_basin_depths(x_edges, y_edges, gz_obs, drho0, lam):
    """Depths of a basin of columns (density contrast drho0 exp(-lam z)) from its gravity anomaly.

    Inputs:
      x_edges, y_edges: 1-D float arrays (nx + 1 and ny + 1 entries), strictly increasing, nx ny <= 100.
      gz_obs: float array (nx, ny), the vertical attraction (m s^-2) of all columns at the cell centres on
        the surface z = 0; it is that of a basin with every depth in (0, 5e4] m.
      drho0: float, nonzero density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      depths: numpy array (nx, ny) in m, each with a relative error below 1e-8. Each call within 30 s.

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
    # start: infinite-slab thickness under each station
    h = -np.log1p(-d * lam) / lam if lam > 0 else d.copy()
    h = np.minimum(h, 5e4)
    gobs = go.ravel()
    g, J = _basin_forward(xc, yc, xe, ye, h, drho0, lam, jac=True)
    res = g - gobs
    for it in range(100):
        step = np.linalg.solve(J, res)
        t = 1.0
        while True:
            hn = h - t * step
            if np.all(hn > 0) and np.all(hn <= 5e4):
                gn, Jn = _basin_forward(xc, yc, xe, ye, hn, drho0, lam, jac=True)
                rn = gn - gobs
                if np.linalg.norm(rn) <= np.linalg.norm(res) or t < 1e-6:
                    break
            t *= 0.5
            if t < 1e-12:
                raise ValueError("no depths in (0, 5e4] m reproduce the observations")
        h, g, J, res = hn, gn, Jn, rn
        if np.max(np.abs(t * step) / h) < 1e-13:
            break
    else:
        raise ValueError("the depth iteration did not converge")
    depths = h.reshape(nx, ny)
    return depths
