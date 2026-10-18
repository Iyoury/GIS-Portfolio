import numpy as np
import math
from scipy.integrate import quad

# Second solution, by methods different from the reference:
# - prism (steps 1 and 2): beyond three diagonals of the prism, the multipole (Cartesian Taylor) expansion of
#   1/|R - s| about its centre with the exact moments of the box; closer, a reduction to one dimension: the
#   prism is a stack of rectangular laminae along one axis, whose integrals of 1/r, of the attraction and of
#   the gradients have two-dimensional closed forms (a lamina far from the point by a 10 x 10
#   Gauss-Legendre rule), integrated over the axis with Gauss-Legendre panels that shrink geometrically
#   toward the plane of the point; T_ww from Poisson's (or Laplace's) equation;
# - column (step 3): adaptive QUADPACK integration over depth with geometric breakpoints of the lamina
#   corner sum, the lamina taken by a 24 x 24 Gauss-Legendre rule beyond two lamina diagonals;
# - basin (step 4): a tanh-sinh (double-exponential) rule over the depth of each column for the forward
#   model, and Levenberg-Marquardt iterations (Marquardt scaling, Nielsen's damping update, steps from the
#   augmented least-squares system) on the depths from the infinite-slab thicknesses, run to convergence,
#   the iterates projected on the box of admissible depths (or, if that stalls, reflected at its bounds).

G = 6.6743e-11
_X24, _W24 = np.polynomial.legendre.leggauss(24)


def _bounds(bounds, rho):
    b = [float(v) for v in np.asarray(bounds, float).ravel()]
    if len(b) != 6 or not all(math.isfinite(v) for v in b) or not (b[0] < b[1] and b[2] < b[3] and b[4] < b[5]):
        raise ValueError("invalid bounds")
    if not math.isfinite(float(rho)) or float(rho) == 0.0:
        raise ValueError("rho must be finite and nonzero")
    return b, float(rho)


def _pts(points):
    p = np.asarray(points, float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1 or not np.all(np.isfinite(p)):
        raise ValueError("points must be finite, of shape (3,) or (n, 3)")
    return p, single


# ---------------------------------------------------------------------------------------------------------
# far field: multipole (Cartesian Taylor) expansion of 1/|R - s| about the centre of the box, with the
# exact moments of the box, int s^k dV = prod_i 2 h_i^(k_i + 1) / (k_i + 1) for even k_i (0 for odd k_i)

def _taylor_coeffs(R, nmax):
    # a_k = (1/k!) d^k (1/|R|) for |k| <= nmax, by the recurrence
    # |k| |R|^2 a_k + (2|k| - 1) sum_i R_i a_{k - e_i} + (|k| - 1) sum_i a_{k - 2 e_i} = 0
    r2 = np.sum(R * R, axis=1)
    a = {(0, 0, 0): 1.0 / np.sqrt(r2)}
    for n in range(1, nmax + 1):
        for k1 in range(n, -1, -1):
            for k2 in range(n - k1, -1, -1):
                k = (k1, k2, n - k1 - k2)
                s1 = 0.0
                s2 = 0.0
                for i in range(3):
                    if k[i] >= 1:
                        km = list(k)
                        km[i] -= 1
                        s1 = s1 + R[:, i] * a[tuple(km)]
                        if k[i] >= 2:
                            km[i] -= 1
                            s2 = s2 + a[tuple(km)]
                a[k] = -((2 * n - 1) * s1 + (n - 1) * s2) / (n * r2)
    return a


def _multipole(p, b, want_T):
    c = np.array([0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])])
    h = 0.5 * np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    hd = float(np.sqrt(np.sum(h * h)))          # half-diagonal: every source point is within hd of c
    R = (p - c[None, :]) / hd                   # in units of the half-diagonal
    hs = h / hd
    rmin = float(np.min(np.sqrt(np.sum(R * R, axis=1))))
    # the order-n terms are bounded by (1 / |R|)^n relative to the monopole
    N = int(math.ceil(19.0 / math.log10(rmin)))
    N += N % 2
    a = _taylor_coeffs(R, N + 2)
    mom = {}
    for n in range(0, N + 1, 2):
        for k1 in range(0, n + 1, 2):
            for k2 in range(0, n - k1 + 1, 2):
                k = (k1, k2, n - k1 - k2)
                mom[k] = np.prod([2.0 * hs[i] ** (k[i] + 1) / (k[i] + 1) for i in range(3)])
    m = p.shape[0]
    if want_T:
        T = np.zeros((m, 3, 3))
        for k, M in mom.items():
            for i in range(3):
                for j in range(i, 3):
                    kk = list(k)
                    kk[i] += 1
                    kk[j] += 1
                    f = (k[i] + 1) * (k[j] + 1 + (i == j))
                    T[:, i, j] += M * f * a[tuple(kk)]
        for i in range(3):
            for j in range(i):
                T[:, i, j] = T[:, j, i]
        return T
    U = np.zeros(m)
    g = np.zeros((m, 3))
    for k, M in mom.items():
        U += M * a[k]
        for i in range(3):
            kk = list(k)
            kk[i] += 1
            g[:, i] += M * (k[i] + 1) * a[tuple(kk)]
    # g = grad_P U = grad_R U; U = int dV / |R - s|
    return U * hd * hd, g * hd


# ---------------------------------------------------------------------------------------------------------
# near field: reduction to one dimension. Along one axis w the prism is a stack of rectangular laminae;
# the integrals of 1/r, of the attraction and of the gradients over a lamina have closed forms (corner sums
# in two dimensions), and the stack is integrated numerically over w with Gauss-Legendre panels that shrink
# geometrically toward the plane of the observation point, where the lamina integrals vary on the scale of
# the point's distance to the lamina edges (and jump inside the prism). A lamina whose centre is more than
# four lamina diagonals from the point is integrated by a 10 x 10 Gauss-Legendre rule instead, since its
# corner sums would be differences of much larger terms.

_GX12, _GW12 = np.polynomial.legendre.leggauss(12)
_GX10, _GW10 = np.polynomial.legendre.leggauss(10)
_KGRID = np.arange(-2, 61)


def _lnpr(y, x, Z, r):
    # ln(y + r), r = sqrt(x^2 + y^2 + Z^2), Z != 0; for y < 0 as ln(x^2 + Z^2) - ln(r - y)
    s = x * x + Z * Z
    return np.where(y >= 0, np.log(np.abs(y) + r), np.log(s) - np.log(r + np.abs(y)))


def _inv_ypr(y, x, Z, r):
    # 1 / (y + r), with (r - y) / (x^2 + Z^2) for y < 0
    s = x * x + Z * Z
    return np.where(y >= 0, 1.0 / (np.abs(y) + r), (r + np.abs(y)) / s)


def _lamina_closed(xs, ys, Z, want_T):
    # xs, ys: lists of the two corner offsets (n, 1) along u and v; Z (n, m) offsets along w (never 0)
    out = None
    for i in range(2):
        for j in range(2):
            x, y = xs[i], ys[j]
            sg = 1.0 if (i + j) % 2 == 0 else -1.0
            r = np.sqrt(x * x + y * y + Z * Z)
            if want_T:
                iy = _inv_ypr(y, x, Z, r)
                ix = _inv_ypr(x, y, Z, r)
                v = [x * iy / r, y * ix / r, 1.0 / r, Z * iy / r, Z * ix / r]
            else:
                ly = _lnpr(y, x, Z, r)
                lx = _lnpr(x, y, Z, r)
                at = np.arctan(x * y / (Z * r))
                v = [x * ly + y * lx - Z * at, -ly, -lx, at]
            v = [sg * t for t in v]
            out = v if out is None else [o + t for o, t in zip(out, v)]
    return out


def _lamina_gl(xc, yc, hx, hy, Z, want_T):
    # the same lamina integrals by a 10 x 10 Gauss-Legendre rule; xc, yc, hx, hy, Z flat arrays (K,)
    x = xc[:, None, None] + hx * _GX10[None, :, None]
    y = yc[:, None, None] + hy * _GX10[None, None, :]
    w = hx * hy * _GW10[None, :, None] * _GW10[None, None, :]
    Zb = Z[:, None, None]
    r2 = x * x + y * y + Zb * Zb
    r = np.sqrt(r2)
    i3 = w / (r2 * r)
    if want_T:
        i5 = w / (r2 * r2 * r)
        return [np.sum(i5 * (3 * x * x - r2), axis=(1, 2)), np.sum(i5 * (3 * y * y - r2), axis=(1, 2)),
                np.sum(3 * i5 * x * y, axis=(1, 2)), np.sum(3 * i5 * x * Zb, axis=(1, 2)),
                np.sum(3 * i5 * y * Zb, axis=(1, 2))]
    return [np.sum(w / r, axis=(1, 2)), np.sum(i3 * x, axis=(1, 2)), np.sum(i3 * y, axis=(1, 2)),
            np.sum(i3 * Zb, axis=(1, 2))]


def _near(p, b, want_T):
    e = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    asp = [max(e[(k + 1) % 3], e[(k + 2) % 3]) / min(e[(k + 1) % 3], e[(k + 2) % 3]) for k in range(3)]
    ax = int(np.argmin(asp))
    iu, iv = [k for k in range(3) if k != ax]
    perm = [iu, iv, ax]
    lo = np.array([b[0], b[2], b[4]])[perm]
    hi = np.array([b[1], b[3], b[5]])[perm]
    q = p[:, perm]
    n = q.shape[0]
    L = hi[2] - lo[2]
    # panel breakpoints in the offset Z = w - w_P: 0 and +-L 2^-k, clipped to [w1 - w_P, w2 - w_P]
    za = (lo[2] - q[:, 2])[:, None]
    zb = (hi[2] - q[:, 2])[:, None]
    off = L * 2.0 ** (-_KGRID.astype(float))
    br = np.concatenate([za, zb, np.zeros((n, 1)), -off[None, :] * np.ones((n, 1)), off[None, :] * np.ones((n, 1))], 1)
    br = np.sort(np.clip(br, za, zb), axis=1)
    a0, a1 = br[:, :-1], br[:, 1:]
    Z = (0.5 * (a1 - a0))[:, :, None] * _GX12[None, None, :] + (0.5 * (a0 + a1))[:, :, None]
    W = (0.5 * (a1 - a0))[:, :, None] * _GW12[None, None, :] * np.ones_like(Z)
    Z = Z.reshape(n, -1)
    W = W.reshape(n, -1)
    keep = W > 0
    Z = np.where(keep, Z, 1.0)                  # zero-length panels: dummy offsets, zero weight
    xs = [(lo[0] - q[:, 0])[:, None], (hi[0] - q[:, 0])[:, None]]
    ys = [(lo[1] - q[:, 1])[:, None], (hi[1] - q[:, 1])[:, None]]
    vals = _lamina_closed(xs, ys, Z, want_T)
    # laminae far from the point: Gauss-Legendre over the lamina
    xc = 0.5 * (xs[0] + xs[1])
    yc = 0.5 * (ys[0] + ys[1])
    hx, hy = 0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1])
    ldiag = 2.0 * math.hypot(hx, hy)
    far = (xc * xc + yc * yc + Z * Z >= (4.0 * ldiag) ** 2) & keep
    if np.any(far):
        ii, jj = np.nonzero(far)
        for s in range(0, ii.size, 20000):
            a, c = ii[s:s + 20000], jj[s:s + 20000]
            gl = _lamina_gl(xc[a, 0], yc[a, 0], hx, hy, Z[a, c], want_T)
            for v, t in zip(vals, gl):
                v[a, c] = t
    tot = [np.sum(W * v, axis=1) for v in vals]
    # back to the original axes
    inv = np.argsort(perm)
    if want_T:
        Tuu, Tvv, Tuv, Tuw, Tvw = tot
        inside = np.all((q > lo[None, :]) & (q < hi[None, :]), axis=1)
        Tww = -(Tuu + Tvv) - 4.0 * math.pi * inside
        Tp = np.stack([np.stack([Tuu, Tuv, Tuw], 1), np.stack([Tuv, Tvv, Tvw], 1), np.stack([Tuw, Tvw, Tww], 1)], 1)
        return Tp[:, inv][:, :, inv]
    U, gu, gv, gw = tot
    gp = np.stack([gu, gv, gw], 1)
    return U, gp[:, inv]


def _split_far(p, b):
    c = np.array([0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])])
    D = math.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
    return np.sqrt(np.sum((p - c[None, :]) ** 2, axis=1)) >= 3.0 * D


def prism_gravity(points, bounds, rho):
    p, single = _pts(points)
    b, rho = _bounds(bounds, rho)
    U = np.empty(len(p))
    g = np.empty((len(p), 3))
    far = _split_far(p, b)
    if np.any(far):
        U[far], g[far] = _multipole(p[far], b, False)
    if np.any(~far):
        U[~far], g[~far] = _near(p[~far], b, False)
    U, g = G * rho * U, G * rho * g
    return (float(U[0]), g[0]) if single else (U, g)


def prism_gradients(points, bounds, rho):
    p, single = _pts(points)
    b, rho = _bounds(bounds, rho)
    lo = np.array([b[0], b[2], b[4]])
    hi = np.array([b[1], b[3], b[5]])
    closed = np.all((p >= lo) & (p <= hi), axis=1)
    strict = np.all((p > lo) & (p < hi), axis=1)
    if np.any(closed & ~strict):
        raise ValueError("the gradients are not defined on the surface of the prism")
    T = np.empty((len(p), 3, 3))
    far = _split_far(p, b)
    if np.any(far):
        T[far] = _multipole(p[far], b, True)
    if np.any(~far):
        T[~far] = _near(p[~far], b, True)
    T = G * rho * T
    return T[0] if single else T


def _lam_term(x1, x2, y1, y2, Z):
    tot = 0.0
    for i, x in enumerate((x1, x2)):
        for j, y in enumerate((y1, y2)):
            r = math.sqrt(x * x + y * y + Z * Z)
            tot += (1.0 if (i + j) % 2 == 0 else -1.0) * math.atan2(x * y, Z * r)
    return tot


def _col(q, x1, x2, y1, y2, h, drho0, lam):
    X1, X2, Y1, Y2 = x1 - q[0], x2 - q[0], y1 - q[1], y2 - q[1]
    dxh, dyh = max(X1, -X2, 0.0), max(Y1, -Y2, 0.0)
    if math.hypot(dxh, dyh) >= 2.0 * math.hypot(x2 - x1, y2 - y1):
        # far from the column the corner sum of the lamina term cancels; integrate Z / r^3 over the lamina
        # with a 24 x 24 Gauss-Legendre rule instead
        xs = (0.5 * (x2 - x1) * _X24 + 0.5 * (x1 + x2) - q[0])[:, None]
        ys = (0.5 * (y2 - y1) * _X24 + 0.5 * (y1 + y2) - q[1])[None, :]
        W = 0.25 * (x2 - x1) * (y2 - y1) * _W24[:, None] * _W24[None, :]
        R2 = xs * xs + ys * ys

        def f(z):
            Z = z - q[2]
            return math.exp(-lam * z) * float(np.sum(W * Z / (R2 + Z * Z) ** 1.5))
    else:
        def f(z):
            return math.exp(-lam * z) * _lam_term(X1, X2, Y1, Y2, z - q[2])
    br = [0.0] + [h * 2.0 ** (-k) for k in range(50, -1, -1)]
    # full_output=1: QUADPACK reports roundoff-limited panels in its output instead of warning
    return G * drho0 * math.fsum(quad(f, a, c, epsabs=0.0, epsrel=1e-13, limit=200, full_output=1)[0]
                                 for a, c in zip(br[:-1], br[1:]))


def column_gz(stations, x1, x2, y1, y2, depth, drho0, lam):
    st, single = _pts(stations)
    vals = [x1, x2, y1, y2, depth, drho0, lam]
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError("column parameters must be finite")
    if float(drho0) == 0.0:
        raise ValueError("drho0 must be nonzero")
    if not (x1 < x2 and y1 < y2) or not 0.0 < depth <= 5e4 or not 0.0 <= lam <= 1e-2:
        raise ValueError("invalid column")
    if np.any(st[:, 2] > 0.0):
        raise ValueError("stations must be at or above the surface")
    out = np.array([_col(q, x1, x2, y1, y2, depth, drho0, lam) for q in st])
    return float(out[0]) if single else out


def _de_rule(K, tau):
    # tanh-sinh (double-exponential) rule on [0, 1]: x = (1 + tanh(pi/2 sinh t)) / 2, t = k tau, |k| <= K
    t = tau * np.arange(-K, K + 1)
    s = 0.5 * math.pi * np.sinh(t)
    x = 1.0 / (1.0 + np.exp(-2.0 * s))
    w = 0.25 * math.pi * tau * np.cosh(t) / np.cosh(s) ** 2
    return x, w


_DE_X, _DE_W = _de_rule(48, 1.0 / 12.0)


class _Basin:
    # forward model of the basin: stations at the cell centres, columns in the order of the flattened grid

    def __init__(self, xe, ye, drho0, lam):
        nx, ny = xe.size - 1, ye.size - 1
        self.xc = np.repeat(0.5 * (xe[:-1] + xe[1:]), ny)
        self.yc = np.tile(0.5 * (ye[:-1] + ye[1:]), nx)
        self.X1, self.X2 = np.repeat(xe[:-1], ny), np.repeat(xe[1:], ny)
        self.Y1, self.Y2 = np.tile(ye[:-1], nx), np.tile(ye[1:], nx)
        self.drho0, self.lam, self.n = drho0, lam, nx * ny

    def omega(self, z, j):
        # solid angle of the lamina of column j at the depths z (nodes,) seen from every station,
        # corner sum of atan(x y / (z r)): array (nodes, stations)
        tot = 0.0
        zz = z[:, None]
        for xa, sx in ((self.X1[j] - self.xc, -1.0), (self.X2[j] - self.xc, 1.0)):
            for ya, sy in ((self.Y1[j] - self.yc, -1.0), (self.Y2[j] - self.yc, 1.0)):
                x, y = xa[None, :], ya[None, :]
                r = np.sqrt(x * x + y * y + zz * zz)
                tot = tot + sx * sy * np.arctan(x * y / (zz * r))
        return tot

    def forward(self, h):
        # g_i = G drho0 sum_j int_0^h_j exp(-lam z) Omega_ij(z) dz, each integral by the tanh-sinh rule
        out = np.zeros(self.n)
        for j in range(self.n):
            z = h[j] * _DE_X
            out += (h[j] * _DE_W * np.exp(-self.lam * z)) @ self.omega(z, j)
        return G * self.drho0 * out

    def jac(self, h):
        # dg_i / dh_j = G drho0 exp(-lam h_j) Omega_ij(h_j)
        J = np.empty((self.n, self.n))
        for j in range(self.n):
            J[:, j] = G * self.drho0 * math.exp(-self.lam * h[j]) * self.omega(np.array([h[j]]), j)[0]
        return J


def _reflect(x, lo, hi):
    # fold x back into [lo, hi] by reflection at the bounds
    span = hi - lo
    y = np.mod(x - lo, 2.0 * span)
    return lo + np.where(y > span, 2.0 * span - y, y)


def _levenberg_marquardt(B, gobs, h, lo, hi, reflect):
    # Levenberg-Marquardt on the depths (Marquardt scaling, Nielsen's damping update), each step from the
    # least squares of [J; sqrt(mu diag(J^T J))] p = [-r; 0] and brought back into [lo, hi] by projection or
    # by reflection at the bounds; run until the step is below the rounding of the depths (as mu -> 0 the
    # steps become Gauss-Newton steps, which converge quadratically for data without noise).
    # Returns the depths and the largest residual relative to max |g_obs|.
    sc = float(np.max(np.abs(gobs)))
    r = (B.forward(h) - gobs) / sc
    J = B.jac(h) / sc
    f = r @ r
    dA = np.sum(J * J, axis=0)
    mu, nu = 1e-3, 2.0
    n = h.size
    for it in range(500):
        A = np.vstack([J, np.diag(np.sqrt(mu * dA))])
        p = np.linalg.lstsq(A, np.concatenate([-r, np.zeros(n)]), rcond=None)[0]
        hn = _reflect(h + p, lo, hi) if reflect else np.clip(h + p, lo, hi)
        pn = hn - h
        small = np.max(np.abs(pn) / h) < 1e-14
        rn = (B.forward(hn) - gobs) / sc
        fn = rn @ rn
        lin = r + J @ pn
        pred = f - lin @ lin
        ratio = (f - fn) / pred if pred > 0 else -1.0
        if fn < f:
            h, r, f = hn, rn, fn
            J = B.jac(h) / sc
            dA = np.sum(J * J, axis=0)
            if ratio > 0:
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
            nu = 2.0
        else:
            mu *= nu
            nu *= 2.0
        if small:
            break
    return h, float(np.max(np.abs(r)))


def invert_basin_depths(x_edges, y_edges, gz_obs, drho0, lam):
    """Depths of a basin of columns (density contrast drho0 exp(-lam z)) from its gravity anomaly.

    Inputs:
      x_edges, y_edges: 1-D float arrays of nx + 1 and ny + 1 entries (nx, ny >= 1), strictly increasing,
        nx ny <= 100.
      gz_obs: float array (nx, ny), the vertical attraction (m s^-2) of all columns at the cell centres on
        the surface z = 0; it is that of a basin whose every depth satisfies 1e-3 w_min <= depth <= 3 w_min
        (w_min the smallest cell width, in x or y), depth <= 5e4 m and lam depth <= 2.
      drho0: float, nonzero density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      depths: numpy array (nx, ny) in m (also when nx or ny is 1), each with a relative error below 1e-8.

    Raises:
      ValueError for an edge array that is not one-dimensional or has fewer than two entries, edges that
      are not strictly increasing, nx ny > 100, gz_obs of the wrong shape,
      drho0 = 0, lam outside [0, 1e-2], an observation that is zero or of the sign opposite to drho0,
      or (lam > 0) an observation of at least 2 pi G |drho0| / lam.
    """
    xe, ye, go =np.asarray(x_edges, float), np.asarray(y_edges, float), np.asarray(gz_obs, float)
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
    B = _Basin(xe, ye, drho0, lam)
    gobs = go.ravel()
    wmin = min(float(np.min(np.diff(xe))), float(np.min(np.diff(ye))))
    # admissible depths for the iteration: [1e-6 w_min, 1.5 times the largest depth of the stated domain],
    # a box that contains the stated domain [1e-3 w_min, min(3 w_min, 5e4, 2 / lam)] with margins
    lo = 1e-6 * wmin
    hi = 1.5 * min(5e4, 3.0 * wmin, 2.0 / lam if lam > 0 else math.inf)
    # start: the infinite-slab thickness under each station
    h0 = np.clip(-np.log1p(-lam * d) / lam if lam > 0 else d.copy(), lo, hi)
    # iterates projected on the bounds; if that stalls away from the data (a column held at a bound),
    # again with the steps reflected at the bounds
    for reflect in (False, True):
        h, res = _levenberg_marquardt(B, gobs, h0.copy(), lo, hi, reflect)
        if res <= 1e-10:
            return h.reshape(nx, ny)
    raise ValueError("no depths in the admissible range reproduce the observations")
