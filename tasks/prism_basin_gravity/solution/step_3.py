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


def _column_nodes(z0, h):
    # depths z in [0, h] of a column top at 0, station at height z0 >= 0 above the top: geometric panels
    # from the top downward (ratio 2, down to 1e-18 h) so that the lamina term, which varies on the scale of
    # the station's horizontal distance to the column edges when z0 is small, is resolved for any distance
    edges = [0.0] + [h * 2.0 ** (-k) for k in range(60, -1, -1)]
    e = np.array(edges)
    a, b = e[:-1], e[1:]
    z = (0.5 * (b - a))[:, None] * _GX20[None, :] + (0.5 * (a + b))[:, None]
    w = (0.5 * (b - a))[:, None] * _GW20[None, :]
    return z.ravel(), w.ravel()


def column_gz(stations, x1, x2, y1, y2, depth, drho0, lam):
    """Vertical attraction of a column with density contrast drho0 exp(-lam z), 0 <= z <= depth.

    Inputs:
      stations: float array of shape (3,) or (n, 3), stations (x, y, z) in m with z <= 0.
      x1, x2, y1, y2: floats, the column x1 <= x <= x2, y1 <= y <= y2 (m), x1 < x2, y1 < y2.
      depth: float, 0 < depth <= 5e4 (m).
      drho0: float, density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      gz: z component of the attraction (m s^-2, positive down for drho0 > 0); a Python float for one
        station, a numpy array (n,) otherwise; relative error below 1e-10 at every station. Each call
        within 10 s for up to 1000 stations.

    Raises:
      ValueError for stations of the wrong shape, non-finite or with z > 0, a non-finite parameter,
      x1 >= x2 or y1 >= y2, depth outside (0, 5e4] or lam outside [0, 1e-2].
    """
    st, single = _check_points(stations)
    vals = np.array([x1, x2, y1, y2, depth, drho0, lam], float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("column parameters must be finite")
    if not (x1 < x2 and y1 < y2):
        raise ValueError("need x1 < x2 and y1 < y2")
    if not 0.0 < depth <= 5e4:
        raise ValueError("need 0 < depth <= 5e4 m")
    if not 0.0 <= lam <= 1e-2:
        raise ValueError("need 0 <= lam <= 1e-2 per m")
    if np.any(st[:, 2] > 0.0):
        raise ValueError("stations must be at or above the surface z = 0")
    # g_z = G drho0 int_0^h e^{-lam z} L(z - z_s) dz with L the lamina term
    z, w = _column_nodes(None, float(depth))
    out = np.empty(st.shape[0])
    for i, q in enumerate(st):
        X = np.array([[x1 - q[0]], [x2 - q[0]]])
        Y = np.array([[y1 - q[1]], [y2 - q[1]]])
        Z = (z - q[2])[:, None]
        L = _lamina(X, Y, Z)[:, 0]
        out[i] = G * drho0 * float(np.sum(w * np.exp(-lam * z) * L))
    gz = float(out[0]) if single else out
    return gz
