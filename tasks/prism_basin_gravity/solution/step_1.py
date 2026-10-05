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


def _ln_plus_r(a, b, c):
    # ln(a + r), r = sqrt(a^2 + b^2 + c^2); for a < 0 as ln((b^2 + c^2) / (r - a)) (no cancellation);
    # -inf where a + r = 0 (only multiplied by a zero coefficient there)
    r = np.sqrt(a * a + b * b + c * c)
    s = b * b + c * c
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.log(a + r)
        # s = 0 (P on the line of an edge, outside it): ln s is dropped; it appears at the two corners of that
        # edge with opposite signs and cancels exactly in every corner sum
        neg = np.where(s == 0.0, 0.0, np.log(np.where(s == 0.0, 1.0, s))) - np.log(r - a)
    return np.where(a >= 0, pos, neg)


def _times(coef, val):
    # coef * val with 0 * (+-inf) = 0 (the limits of the closed forms at faces, edges and corners)
    with np.errstate(invalid="ignore"):
        return np.where(coef == 0.0, 0.0, coef * val)


def _atan_ratio(num, den):
    # atan(num / den); den = 0 -> sign(num) pi / 2 (num = 0 -> 0): these values are only used where the
    # corner sums are continuous or are multiplied by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den == 0.0, np.sign(num) * (0.5 * np.pi), np.arctan(num / np.where(den == 0.0, 1.0, den)))


def _corners(p, b, fn):
    xs = (b[0] - p[:, 0], b[1] - p[:, 0])
    ys = (b[2] - p[:, 1], b[3] - p[:, 1])
    zs = (b[4] - p[:, 2], b[5] - p[:, 2])
    acc = None
    for i in range(2):
        for j in range(2):
            for k in range(2):
                v = fn(xs[i], ys[j], zs[k])
                v = v if (i + j + k) % 2 == 1 else [-t for t in v]     # + at (x2, y2, z2)
                acc = v if acc is None else [a + t for a, t in zip(acc, v)]
    return acc


def _uvw(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    lx, ly, lz = _ln_plus_r(x, y, z), _ln_plus_r(y, z, x), _ln_plus_r(z, x, y)
    ax, ay, az = _atan_ratio(y * z, x * r), _atan_ratio(z * x, y * r), _atan_ratio(x * y, z * r)
    U = (_times(x * y, lz) + _times(y * z, lx) + _times(z * x, ly)
         - 0.5 * _times(x * x, ax) - 0.5 * _times(y * y, ay) - 0.5 * _times(z * z, az))
    Fx = _times(y, lz) + _times(z, ly) - _times(x, ax)
    Fy = _times(z, lx) + _times(x, lz) - _times(y, ay)
    Fz = _times(x, ly) + _times(y, lx) - _times(z, az)
    return [U, Fx, Fy, Fz]


def _tens(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    return [-_atan_ratio(y * z, x * r), -_atan_ratio(z * x, y * r), -_atan_ratio(x * y, z * r),
            _ln_plus_r(z, x, y), _ln_plus_r(y, z, x), _ln_plus_r(x, y, z)]


def _far(p, b):
    # points at least two diagonals from the centre: the corner sums lose about (distance / size)^2 of their
    # digits there (their terms grow like distance^2 while U and g fall like 1/distance and 1/distance^2),
    # so the volume integral is taken by a 16^3-point Gauss-Legendre rule, exact to rounding that far out
    c = 0.5 * np.array([b[0] + b[1], b[2] + b[3], b[4] + b[5]])
    diag = math.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
    return np.linalg.norm(p - c[None, :], axis=1) >= 2.0 * diag


def _cubature(p, b):
    xs = 0.5 * (b[1] - b[0]) * _GX16 + 0.5 * (b[0] + b[1])
    ys = 0.5 * (b[3] - b[2]) * _GX16 + 0.5 * (b[2] + b[3])
    zs = 0.5 * (b[5] - b[4]) * _GX16 + 0.5 * (b[4] + b[5])
    w = (0.125 * (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])
         * _GW16[:, None, None] * _GW16[None, :, None] * _GW16[None, None, :]).ravel()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    out = []
    for q in p:
        dx, dy, dz = X - q[0], Y - q[1], Z - q[2]
        r2 = dx * dx + dy * dy + dz * dz
        r = np.sqrt(r2)
        i1, i3, i5 = w / r, w / (r2 * r), w / (r2 * r2 * r)
        U = i1.sum()
        g = np.array([(i3 * dx).sum(), (i3 * dy).sum(), (i3 * dz).sum()])
        T = np.array([[(i5 * (3 * a * c - (k == l) * r2)).sum() for l, c in enumerate((dx, dy, dz))]
                      for k, a in enumerate((dx, dy, dz))])
        out.append((U, g, T))
    return out


def prism_gravity(points, bounds, rho):
    """Potential and attraction of a right rectangular prism of uniform density.

    Inputs:
      points: float array of shape (3,) or (n, 3), observation points (x, y, z) in m, z positive down.
      bounds: (x1, x2, y1, y2, z1, z2), the prism x1 <= x <= x2, y1 <= y <= y2, z1 <= z <= z2 (m).
      rho: float, density (kg m^-3).

    Output:
      (U, g): U = G rho int dV / |Q - P| (m^2 s^-2) and g = grad U (m s^-2, toward the mass for rho > 0,
        z component positive down). Python float and array (3,) for one point, arrays (n,) and (n, 3)
        otherwise. U with a relative error below 1e-10; each component of g within
        1e-10 |g| + 1e-14 G |rho| L (L the longest edge), at every point, inside, on the surface or
        outside at any distance. Each call within 10 s for up to 1000 points.

    Raises:
      ValueError for invalid bounds (not six finite numbers with x1 < x2, y1 < y2, z1 < z2), a
      non-finite rho, or points of the wrong shape or with a non-finite value.
    """
    p, single = _check_points(points)
    b, rho = _check_bounds(bounds, rho)
    far = _far(p, b)
    U = np.empty(p.shape[0])
    g = np.empty((p.shape[0], 3))
    near = ~far
    if np.any(near):
        U_, Fx, Fy, Fz = _corners(p[near], b, _uvw)
        U[near] = G * rho * U_
        g[near] = -G * rho * np.stack([Fx, Fy, Fz], axis=1)
    if np.any(far):
        res = _cubature(p[far], b)
        U[far] = G * rho * np.array([r[0] for r in res])
        g[far] = G * rho * np.array([r[1] for r in res])
    result = (float(U[0]), g[0]) if single else (U, g)
    return result
