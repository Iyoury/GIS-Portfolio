import numpy as np
import math
import warnings
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import root

# Second solution: the closed forms evaluated point by point with compensated summation of the eight
# corner terms (math.fsum) and a 24^3-point Gauss-Legendre volume rule beyond three diagonals; the columns
# by adaptive QUADPACK integration over depth with geometric breakpoints; the basin depths by
# scipy.optimize.root (hybrid Powell) with the analytic Jacobian.

warnings.simplefilter("ignore", IntegrationWarning)
G = 6.6743e-11
_X24, _W24 = np.polynomial.legendre.leggauss(24)


def _bounds(bounds, rho):
    b = [float(v) for v in np.asarray(bounds, float).ravel()]
    if len(b) != 6 or not all(math.isfinite(v) for v in b) or not (b[0] < b[1] and b[2] < b[3] and b[4] < b[5]):
        raise ValueError("invalid bounds")
    if not math.isfinite(float(rho)):
        raise ValueError("rho must be finite")
    return b, float(rho)


def _pts(points):
    p = np.asarray(points, float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1 or not np.all(np.isfinite(p)):
        raise ValueError("points must be finite, of shape (3,) or (n, 3)")
    return p, single


def _lnr(a, b, c):
    r = math.sqrt(a * a + b * b + c * c)
    if a >= 0:
        return math.log(a + r) if a + r > 0 else None
    s = b * b + c * c
    return (math.log(s) if s > 0 else 0.0) - math.log(r - a)


def _mul(c, v):
    return 0.0 if c == 0.0 else c * v


def _at(n, d):
    if d == 0.0:
        return 0.0 if n == 0.0 else math.copysign(math.pi / 2, n)
    return math.atan(n / d)


def _corner_terms(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    lx, ly, lz = _lnr(x, y, z), _lnr(y, z, x), _lnr(z, x, y)
    lx = 0.0 if lx is None else lx
    ly = 0.0 if ly is None else ly
    lz = 0.0 if lz is None else lz
    ax, ay, az = _at(y * z, x * r), _at(z * x, y * r), _at(x * y, z * r)
    U = [_mul(x * y, lz), _mul(y * z, lx), _mul(z * x, ly), -0.5 * _mul(x * x, ax), -0.5 * _mul(y * y, ay), -0.5 * _mul(z * z, az)]
    Fx = [_mul(y, lz), _mul(z, ly), -_mul(x, ax)]
    Fy = [_mul(z, lx), _mul(x, lz), -_mul(y, ay)]
    Fz = [_mul(x, ly), _mul(y, lx), -_mul(z, az)]
    T = [[-ax], [-ay], [-az], [lz], [ly], [lx]]
    return U, Fx, Fy, Fz, T


def _closed(q, b):
    sums = [[] for _ in range(4)]
    tsum = [[] for _ in range(6)]
    for i, X in enumerate((b[0] - q[0], b[1] - q[0])):
        for j, Y in enumerate((b[2] - q[1], b[3] - q[1])):
            for k, Z in enumerate((b[4] - q[2], b[5] - q[2])):
                s = 1.0 if (i + j + k) % 2 == 1 else -1.0
                U, Fx, Fy, Fz, T = _corner_terms(X, Y, Z)
                for lst, terms in zip(sums, (U, Fx, Fy, Fz)):
                    lst.extend(s * t for t in terms)
                for lst, terms in zip(tsum, T):
                    lst.extend(s * t for t in terms)
    return [math.fsum(v) for v in sums], [math.fsum(v) for v in tsum]


def _gl(q, b):
    xs = 0.5 * (b[1] - b[0]) * _X24 + 0.5 * (b[0] + b[1])
    ys = 0.5 * (b[3] - b[2]) * _X24 + 0.5 * (b[2] + b[3])
    zs = 0.5 * (b[5] - b[4]) * _X24 + 0.5 * (b[4] + b[5])
    W = 0.125 * (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4]) * _W24[:, None, None] * _W24[None, :, None] * _W24[None, None, :]
    dx = xs[:, None, None] - q[0]
    dy = ys[None, :, None] - q[1]
    dz = zs[None, None, :] - q[2]
    r2 = dx * dx + dy * dy + dz * dz
    r = np.sqrt(r2)
    U = np.sum(W / r)
    g = [np.sum(W * d / (r2 * r)) for d in (dx, dy, dz)]
    ds = (dx, dy, dz)
    T = [[np.sum(W * (3 * ds[a] * ds[c] - (a == c) * r2) / (r2 * r2 * r)) for c in range(3)] for a in range(3)]
    return U, np.array(g), np.array(T)


def _isfar(q, b):
    c = (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
    D = math.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
    return math.dist(q, c) >= 3.0 * D


def prism_gravity(points, bounds, rho):
    p, single = _pts(points)
    b, rho = _bounds(bounds, rho)
    U = np.empty(len(p))
    g = np.empty((len(p), 3))
    for n, q in enumerate(p):
        if _isfar(q, b):
            u, gg, _ = _gl(q, b)
            U[n], g[n] = G * rho * u, G * rho * gg
        else:
            s, _ = _closed(q, b)
            U[n] = G * rho * s[0]
            g[n] = -G * rho * np.array(s[1:])
    return (float(U[0]), g[0]) if single else (U, g)


def prism_gradients(points, bounds, rho):
    p, single = _pts(points)
    b, rho = _bounds(bounds, rho)
    out = np.empty((len(p), 3, 3))
    for n, q in enumerate(p):
        inside = b[0] <= q[0] <= b[1] and b[2] <= q[1] <= b[3] and b[4] <= q[2] <= b[5]
        strict = b[0] < q[0] < b[1] and b[2] < q[1] < b[3] and b[4] < q[2] < b[5]
        if inside and not strict:
            raise ValueError("the gradients are not defined on the surface of the prism")
        if _isfar(q, b):
            out[n] = G * rho * _gl(q, b)[2]
        else:
            _, t = _closed(q, b)
            xx, yy, zz, xy, xz, yz = t
            out[n] = G * rho * np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return out[0] if single else out


def _lam_term(x1, x2, y1, y2, Z):
    tot = 0.0
    for i, x in enumerate((x1, x2)):
        for j, y in enumerate((y1, y2)):
            r = math.sqrt(x * x + y * y + Z * Z)
            tot += (1.0 if (i + j) % 2 == 0 else -1.0) * math.atan2(x * y, Z * r)
    return tot


def _col(q, x1, x2, y1, y2, h, drho0, lam):
    X1, X2, Y1, Y2 = x1 - q[0], x2 - q[0], y1 - q[1], y2 - q[1]
    f = lambda z: math.exp(-lam * z) * _lam_term(X1, X2, Y1, Y2, z - q[2])
    br = [0.0] + [h * 2.0 ** (-k) for k in range(50, -1, -1)]
    return G * drho0 * math.fsum(quad(f, a, c, epsabs=0.0, epsrel=1e-13, limit=200)[0] for a, c in zip(br[:-1], br[1:]))


def column_gz(stations, x1, x2, y1, y2, depth, drho0, lam):
    st, single = _pts(stations)
    vals = [x1, x2, y1, y2, depth, drho0, lam]
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError("column parameters must be finite")
    if not (x1 < x2 and y1 < y2) or not 0.0 < depth <= 5e4 or not 0.0 <= lam <= 1e-2:
        raise ValueError("invalid column")
    if np.any(st[:, 2] > 0.0):
        raise ValueError("stations must be at or above the surface")
    out = np.array([_col(q, x1, x2, y1, y2, depth, drho0, lam) for q in st])
    return float(out[0]) if single else out


def invert_basin_depths(x_edges, y_edges, gz_obs, drho0, lam):
    xe, ye, go = np.asarray(x_edges, float), np.asarray(y_edges, float), np.asarray(gz_obs, float)
    if xe.ndim != 1 or ye.ndim != 1 or xe.size < 2 or ye.size < 2:
        raise ValueError("cell edges must be 1-D with at least two entries")
    if not (np.all(np.isfinite(xe)) and np.all(np.isfinite(ye))) or np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0):
        raise ValueError("cell edges must be finite and strictly increasing")
    nx, ny = xe.size - 1, ye.size - 1
    if nx * ny > 100 or go.shape != (nx, ny) or not np.all(np.isfinite(go)):
        raise ValueError("invalid grid or observations")
    drho0, lam = float(drho0), float(lam)
    if not (math.isfinite(drho0) and drho0 != 0.0) or not (math.isfinite(lam) and 0.0 <= lam <= 1e-2):
        raise ValueError("invalid drho0 or lam")
    d = go.ravel() / (2.0 * math.pi * G * drho0)
    if np.any(d <= 0) or (lam > 0 and np.any(d * lam >= 1.0)):
        raise ValueError("observations outside the attainable range")
    xc = np.repeat(0.5 * (xe[:-1] + xe[1:]), ny)
    yc = np.tile(0.5 * (ye[:-1] + ye[1:]), nx)
    X1, X2 = np.repeat(xe[:-1], ny), np.repeat(xe[1:], ny)
    Y1, Y2 = np.tile(ye[:-1], nx), np.tile(ye[1:], nx)
    x16, w16 = np.polynomial.legendre.leggauss(16)

    def lam_terms(z, j):
        # (nodes, stations) corner sums of atan(x y / (z r)) for column j
        tot = 0.0
        for xa, sx in ((X1[j] - xc, -1.0), (X2[j] - xc, 1.0)):
            for ya, sy in ((Y1[j] - yc, -1.0), (Y2[j] - yc, 1.0)):
                x, y = xa[None, :], ya[None, :]
                r = np.sqrt(x * x + y * y + z[:, None] ** 2)
                tot = tot + sx * sy * np.arctan2(x * y, z[:, None] * r)
        return tot

    def forward(h):
        out = np.zeros(xc.size)
        for j in range(xc.size):
            br = np.array([0.0] + [h[j] * 2.0 ** (-k) for k in range(24, -1, -1)])
            a, b = br[:-1], br[1:]
            z = ((0.5 * (b - a))[:, None] * x16[None, :] + (0.5 * (a + b))[:, None]).ravel()
            w = ((0.5 * (b - a))[:, None] * w16[None, :]).ravel()
            out += (w * np.exp(-lam * z)) @ lam_terms(z, j)
        return G * drho0 * out

    def jac(h):
        return np.stack([G * drho0 * math.exp(-lam * h[j]) * lam_terms(np.array([h[j]]), j)[0] for j in range(xc.size)], axis=1)

    h = -np.log1p(-d * lam) / lam if lam > 0 else d.copy()
    gobs = go.ravel()
    for _ in range(60):
        step = np.linalg.lstsq(jac(h), forward(h) - gobs, rcond=None)[0]
        t = 1.0
        while np.any(h - t * step <= 0):
            t *= 0.5
        h = h - t * step
        if np.max(np.abs(t * step) / h) < 1e-14:
            break
    return h.reshape(nx, ny)
