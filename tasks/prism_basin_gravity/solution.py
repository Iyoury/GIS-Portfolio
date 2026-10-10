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


def prism_gravity(points, bounds, rho):
    """Potential and attraction of a right rectangular prism of uniform density.

    Inputs:
      points: float array of shape (3,) or (n, 3), up to 1000 points, observation points (x, y, z) in m,
        z positive down.
      bounds: (x1, x2, y1, y2, z1, z2), the prism x1 <= x <= x2, y1 <= y <= y2, z1 <= z <= z2 (m), its
        longest edge at most 1000 times its shortest.
      rho: float, density (kg m^-3).

    Output:
      (U, g): U = G rho int dV / |Q - P| (m^2 s^-2) and g = grad U (m s^-2, toward the mass for rho > 0,
        z component positive down). Python float and array (3,) for one point, arrays (n,) and (n, 3)
        otherwise. U with a relative error below 1e-10; each component of g within
        1e-10 |g| + 1e-14 G |rho| L (L the longest edge), at every point, inside, on the surface or
        outside at any distance up to a million times the diagonal of the prism.

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
    if np.any(near) and _elongated(b):
        U_, g_ = _split_sums(p[near], b, False)
        U[near] = G * rho * U_
        g[near] = G * rho * g_
    elif np.any(near):
        U_, Fx, Fy, Fz = _corners(p[near], b, _uvw)
        U[near] = G * rho * U_
        g[near] = -G * rho * np.stack([Fx, Fy, Fz], axis=1)
    if np.any(far):
        res = _cubature(p[far], b)
        U[far] = G * rho * np.array([r[0] for r in res])
        g[far] = G * rho * np.array([r[1] for r in res])
    result = (float(U[0]), g[0]) if single else (U, g)
    return result


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
      stations: float array of shape (3,) or (n, 3), up to 1000 stations (x, y, z) in m with z <= 0.
      x1, x2, y1, y2: floats, the column x1 <= x <= x2, y1 <= y <= y2 (m), x1 < x2, y1 < y2.
      depth: float, 0 < depth <= 5e4 (m).
      drho0: float, density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      gz: z component of the attraction (m s^-2, positive down for drho0 > 0); a Python float for one
        station, a numpy array (n,) otherwise; relative error below 1e-10 at every station up to 1e7 m
        from the column.

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
        L = _lamina(X, Y, Z, np.array([x2 - x1]), np.array([y2 - y1]))[:, 0]
        out[i] = G * drho0 * float(np.sum(w * np.exp(-lam * z) * L))
    gz = float(out[0]) if single else out
    return gz


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
