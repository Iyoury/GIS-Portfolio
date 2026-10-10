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


_KSPLIT = 200.0     # largest diag^3 / volume of a prism (or piece) whose corner sums are used directly


def _shape_factor(e):
    # diag^3 / volume of a box with edges e: 5.2 for a cube, about aspect^2 for a rod and 2.8 aspect for a
    # plate. Within two diagonals the corner sums lose about 1e-16 times this factor (relative to U and g):
    # 4e-12 for 200, 1e-10 at 1e4 (a 1 x 1 x 100 rod), 1e-8 at 1e6 (a 1 x 1 x 1000 rod)
    return float(np.sum(e * e)) ** 1.5 / float(np.prod(e))


def _split_counts(b):
    # numbers of equal pieces along x, y, z so that every piece has a shape factor <= _KSPLIT
    e = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    n = np.ones(3, int)
    while _shape_factor(e / n) > _KSPLIT:
        n[int(np.argmax(e / n))] += 1
    return n


def _gl_orders(e, diag, R):
    # Gauss-Legendre points per axis for a box with edges e and diagonal diag, seen from at least R >= 2 diag
    # from its centre: along axis k the integrand is analytic inside the Bernstein ellipse of parameter
    # rho_k >= q + sqrt(q^2 - 1), q = (2 R - diag) / e_k, so n_k points leave an error ~ rho_k^(-2 n_k) <= 1e-24
    q = (2.0 * R - diag) / e
    rho = q + np.sqrt(q * q - 1.0)
    return np.clip(np.ceil(12.0 / np.log10(rho)), 2, 16).astype(int)


def _box_cubature(p, lo, hi, n, want_T):
    # U and g (or T) of a unit-density box lo <= Q <= hi by an n[0] x n[1] x n[2] Gauss-Legendre rule,
    # vectorised over the points p (m, 3)
    xs, ws = [], []
    for k in range(3):
        t, w = np.polynomial.legendre.leggauss(int(n[k]))
        xs.append(0.5 * (hi[k] - lo[k]) * t + 0.5 * (hi[k] + lo[k]))
        ws.append(0.5 * (hi[k] - lo[k]) * w)
    X, Y, Z = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
    W = (ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]).ravel()
    dx = X.ravel()[None, :] - p[:, 0:1]
    dy = Y.ravel()[None, :] - p[:, 1:2]
    dz = Z.ravel()[None, :] - p[:, 2:3]
    r2 = dx * dx + dy * dy + dz * dz
    r = np.sqrt(r2)
    if not want_T:
        i1 = W / r
        i3 = i1 / r2
        return i1.sum(axis=1), np.stack([(i3 * dx).sum(1), (i3 * dy).sum(1), (i3 * dz).sum(1)], axis=1)
    i5 = W / (r2 * r2 * r)
    d = (dx, dy, dz)
    T = np.empty((p.shape[0], 3, 3))
    for a in range(3):
        for c in range(a, 3):
            v = (i5 * (3.0 * d[a] * d[c] - (r2 if a == c else 0.0))).sum(1)
            T[:, a, c] = v
            T[:, c, a] = v
    return T


def _split_sums(p, b, want_T):
    # points within two diagonals of an elongated prism (shape factor > _KSPLIT): the prism is cut into
    # equal pieces of shape factor <= _KSPLIT (superposition is exact); corner sums for the pieces within two
    # of their diagonals of a point, Gauss-Legendre for the others (rules chosen per band of distance).
    # A point closer than a quarter piece to an internal plane of the default cuts uses, on that axis, the
    # cuts shifted by half a piece, so that no point lies on an internal face, edge or corner (where the
    # gradients of the pieces jump or diverge although their sum does not).
    # Returns the unscaled sums (U, g) or T of unit density.
    n = _split_counts(b)
    lo = np.array([b[0], b[2], b[4]])
    hi = np.array([b[1], b[3], b[5]])
    e = (hi - lo) / n
    m = p.shape[0]
    shift = np.zeros((m, 3), bool)
    for k in range(3):
        if n[k] > 1:
            f = (p[:, k] - lo[k]) / e[k]
            shift[:, k] = np.abs(f - np.clip(np.round(f), 1, n[k] - 1)) < 0.25
    if want_T:
        T = np.zeros((m, 3, 3))
    else:
        U = np.zeros(m)
        g = np.zeros((m, 3))
    codes = shift[:, 0] * 1 + shift[:, 1] * 2 + shift[:, 2] * 4
    for code in np.unique(codes):
        sel = np.nonzero(codes == code)[0]
        ps = p[sel]
        cuts = []
        for k in range(3):
            if n[k] == 1:
                cuts.append(np.array([lo[k], hi[k]]))
            elif (code >> k) & 1:
                cuts.append(np.concatenate([[lo[k]], lo[k] + e[k] * (np.arange(n[k]) + 0.5), [hi[k]]]))
            else:
                cuts.append(np.concatenate([[lo[k]], lo[k] + e[k] * np.arange(1, n[k]), [hi[k]]]))
        for i in range(len(cuts[0]) - 1):
            for j in range(len(cuts[1]) - 1):
                for k in range(len(cuts[2]) - 1):
                    plo = np.array([cuts[0][i], cuts[1][j], cuts[2][k]])
                    phi = np.array([cuts[0][i + 1], cuts[1][j + 1], cuts[2][k + 1]])
                    pe = phi - plo
                    diag = math.sqrt(float(np.sum(pe * pe)))
                    dist = np.linalg.norm(ps - 0.5 * (plo + phi)[None, :], axis=1)
                    near = dist < 2.0 * diag
                    if np.any(near):
                        bb = np.array([plo[0], phi[0], plo[1], phi[1], plo[2], phi[2]])
                        idx = sel[near]
                        if want_T:
                            xx, yy, zz, xy, xz, yz = _corners(ps[near], bb, _tens)
                            T[idx] += np.stack([np.stack([xx, xy, xz], 1), np.stack([xy, yy, yz], 1),
                                                np.stack([xz, yz, zz], 1)], 1)
                        else:
                            U_, Fx, Fy, Fz = _corners(ps[near], bb, _uvw)
                            U[idx] += U_
                            g[idx] -= np.stack([Fx, Fy, Fz], axis=1)
                    fidx = np.nonzero(~near)[0]
                    if fidx.size:
                        band = np.floor(np.log2(dist[fidx] / diag)).astype(int)    # [2, 4), [4, 8), ... diagonals
                        for bnd in np.unique(band):
                            ii = fidx[band == bnd]
                            nn = _gl_orders(pe, diag, diag * 2.0 ** bnd)
                            if want_T:
                                T[sel[ii]] += _box_cubature(ps[ii], plo, phi, nn, True)
                            else:
                                u_, g_ = _box_cubature(ps[ii], plo, phi, nn, False)
                                U[sel[ii]] += u_
                                g[sel[ii]] += g_
    return T if want_T else (U, g)


def _elongated(b):
    return _shape_factor(np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])) > _KSPLIT


def prism_gradients(points, bounds, rho):
    """Gravity-gradient tensor T_ij = d g_i / d x_j of a right rectangular prism of uniform density.

    Inputs:
      points: float array of shape (3,) or (n, 3), up to 1000 points, observation points (m), not on the
        prism surface.
      bounds: (x1, x2, y1, y2, z1, z2) (m), the longest edge at most 1000 times the shortest.
      rho: float, density (kg m^-3).

    Output:
      T: numpy array of shape (3, 3) (one point) or (n, 3, 3), in s^-2; each component within
        1e-9 ||T|| (Frobenius norm of the exact tensor), inside or outside at any distance up to a million
        times the diagonal of the prism.

    Raises:
      ValueError as in prism_gravity, and for a point on the surface of the prism.
    """
    p, single = _check_points(points)
    b, rho = _check_bounds(bounds, rho)
    inside_closed = ((p[:, 0] >= b[0]) & (p[:, 0] <= b[1]) & (p[:, 1] >= b[2]) & (p[:, 1] <= b[3])
                     & (p[:, 2] >= b[4]) & (p[:, 2] <= b[5]))
    strictly = ((p[:, 0] > b[0]) & (p[:, 0] < b[1]) & (p[:, 1] > b[2]) & (p[:, 1] < b[3])
                & (p[:, 2] > b[4]) & (p[:, 2] < b[5]))
    if np.any(inside_closed & ~strictly):
        raise ValueError("the gradients are not defined on the surface of the prism")
    far = _far(p, b)
    T = np.empty((p.shape[0], 3, 3))
    near = ~far
    if np.any(near) and _elongated(b):
        T[near] = G * rho * _split_sums(p[near], b, True)
    elif np.any(near):
        xx, yy, zz, xy, xz, yz = _corners(p[near], b, _tens)
        T[near] = G * rho * np.stack([np.stack([xx, xy, xz], 1), np.stack([xy, yy, yz], 1),
                                      np.stack([xz, yz, zz], 1)], 1)
    if np.any(far):
        T[far] = G * rho * np.array([r[2] for r in _cubature(p[far], b)])
    T = T[0] if single else T
    return T
