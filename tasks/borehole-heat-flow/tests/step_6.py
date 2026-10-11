# Tests for step 6: deglaciation_heat_flow_inversion (joint inversion of one borehole log for q0, T0, the glacial
# amplitude g and the end of the glaciation t_end, with the glacial signal in the layered column).
# No expected value comes from the reference solution. Each case builds its borehole forward with numpy and math
# only (TVD by 64-point Gauss-Legendre integration of the great-circle tangent, R and S by exact layer sums) and
# constructs the temperatures so that the exact least-squares solution is known: on the used readings
# y = T + A S = T0 + q0 R + g P(t_end) + r with r orthogonal to the Jacobian columns [1, R, P, g dP/dt_end] at
# the generating values (seeded, scaled to the stated rms), so the generating values are a stationary point
# and RSS_min = |r|^2. P and dP/dt_end come from an independent oracle, the branch-cut (real-axis) integral of
# the Laplace solution of the layered column with Gauss-Legendre quadrature (checked against the 90-digit
# mpmath values of tests/step_5.py to 9e-16 K and against erfc to 7e-16), or from erfc for a uniform column.
# The Wald standard errors use the oracle Jacobian, the F quantile is computed from the Student t density, and
# the interval ends are the oracle profile's nearest threshold crossings (march and Illinois bisection).
# Accuracy as stated: T0 2e-3 K; q0, g, t_end relative 5e-4; standard errors 5e-3; interval ends 5e-4;
# rms_residual 1e-3; n_used exact.

# --- test case 0 ---
# inclined hole in a seven-layer column: every output against the exact solution
import math
import numpy as np

_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_BX, _BW = np.polynomial.legendre.leggauss(24)

def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

def _g_tvd(md, inc_deg, azi_deg, m):
    # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total

def _g_RS(z, tops, k):
    # exact layer sums of int dz/k and int z dz/k
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b - top) * (b + top) / (2 * kk)
    return R, S

def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi

def _bc_U(z, tau_years, tops, k, rc, deriv=False):
    """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
    year, from the branch-cut (real-axis) form of the Laplace inversion:
    U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
    F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
    (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
    complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
    z = np.atleast_1d(np.asarray(z, float))
    tops = np.asarray(tops, float)
    k = np.asarray(k, float)
    tau = float(tau_years) * _YEAR
    nl = tops.size
    h = np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    width = min(0.25, 2.0 * math.pi / rate / 1.5)
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * _BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * _BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa = np.zeros((nl, w.size)); Pa = np.zeros((nl, w.size))
    Fb = np.zeros((nl, w.size)); Pb = np.zeros((nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
        c = np.cos(om[j] * h[j])
        skw = h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j]
        kws = k[j] * om[j] * np.sin(om[j] * h[j])
        Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
        Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
        sel = np.where(lay == j)[0]
        d = (z[sel] - tops[j])[:, None]
        if j == nl - 1:
            imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * om[j][None, :] * d)).imag
        else:
            fbz = Fb[j][None, :] * np.cos(om[j][None, :] * d) + Pb[j][None, :] * d * np.sinc(om[j][None, :] * d / np.pi) / k[j]
            imf[sel] = cc.imag[None, :] * fbz
    g = np.exp(-w * w)
    if deriv:
        out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _YEAR
        out[z <= 0] = 0.0
        return out
    out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 1.0
    return out

def _f95(nu):
    """0.95 quantile of the Fisher distribution F(1, nu): square of the two-sided 95 % Student t quantile, from
    the t density integrated by Gauss-Legendre quadrature and bisection."""
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)

    def central(x):
        e = np.linspace(0.0, x, 9)
        tot = 0.0
        for i in range(8):
            u = 0.5 * (e[i + 1] - e[i]) * _GL_X + 0.5 * (e[i] + e[i + 1])
            tot += (e[i + 1] - e[i]) * float(np.sum(_GL_W * c * (1.0 + u * u / nu) ** (-(nu + 1) / 2.0)))
        return tot

    lo, hi = 0.0, 1.0
    while central(hi) < 0.95:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    return (0.5 * (lo + hi)) ** 2

class _Hole:
    """Synthetic borehole whose least-squares solution is known exactly. On the used readings the data are
    y = T + A S = T0 + q0 R + g P(t_end) + r, with r orthogonal to the four columns of the Jacobian
    [1, R, P, g dP/dt_end] at the generating values, so those values are a stationary point of the residual
    sum of squares with RSS_min = |r|^2 exactly; P and dP/dt_end come from the oracle above (or from erfc for a
    uniform column). Readings shallower than z_min get disturbed values that follow no model."""

    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
                 log_md=None, log_step=None, log_top=None, uniform_k=None):
        self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
        tops_md, k = (np.asarray(v, float) for v in layers)
        if log_md is None:
            log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
        log_md = np.asarray(log_md, float)
        tops = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in tops_md])
        z = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in log_md])
        self.tops, self.k, self.rc, self.uniform_k = tops, k, rc, uniform_k
        keep = z >= z_min
        self.zk, self.n = z[keep], int(keep.sum())
        RS = np.array([_g_RS(v, list(tops), list(k)) for v in self.zk])
        self.R, self.S = RS[:, 0], RS[:, 1]
        sh = np.asarray(shape, float)
        tk = np.asarray(t_known, float)
        Uk = [self.U(t) for t in tk]
        Pf = sh[1] * Uk[0]
        for i in range(2, sh.size):
            Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
        self.Pfix, self.ds = Pf, sh[0] - sh[1]
        self.star = np.array([T0, q0, g, t_end], float)
        Pst = self.P(t_end)
        J = np.column_stack([np.ones(self.n), self.R, Pst, g * self.ds * self.U(t_end, True)])
        Q, _ = np.linalg.qr(J)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(self.n)
        r = w - Q @ (Q.T @ w)
        r = r - Q @ (Q.T @ r)
        r *= noise / math.sqrt(np.mean(r * r))
        self.y = T0 + q0 * self.R + g * Pst + r
        T = np.empty(z.size)
        T[keep] = self.y - A * self.S
        T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
        self.rss_min = float(r @ r)
        self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
        self.thr = self.rss_min * (1.0 + _f95(self.n - 4) / (self.n - 4))
        self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
        if self.uniform_k is None:
            return _bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
        kap, tau = self.uniform_k / self.rc, t * _YEAR      # uniform column: erfc and its time derivative
        if deriv:
            return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                             for v in self.zk]) * _YEAR
        return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

    def P(self, t):
        return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
        G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
        m, *_ = np.linalg.lstsq(G, self.y, rcond=None)
        res = self.y - G @ m
        return float(res @ res)

    def crossing(self, direction):
        """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
        in small steps, then bisection with Illinois steps."""
        ts = self.star[3]
        a, step = ts, 0.05 * self.sig[3]
        while True:
            b = a + direction * step
            if self.rss(b) >= self.thr:
                break
            a, step = b, min(step * 1.2, 0.3 * self.sig[3])
        fa, fb = self.rss(a) - self.thr, self.rss(b) - self.thr
        for _ in range(200):
            c = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < c < max(a, b)):
                c = 0.5 * (a + b)
            fc = self.rss(c) - self.thr
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) < 1e-12 * abs(b):
                break
        return b

def _check(r, h, label=""):
    """Every output against the exact values, with the stated accuracies."""
    ex = {"T0": h.star[0], "q0": h.star[1], "amplitude": h.star[2], "t_end": h.star[3],
          "sigma_T0": h.sig[0], "sigma_q0": h.sig[1], "sigma_amplitude": h.sig[2], "sigma_t_end": h.sig[3],
          "t_end_low": h.crossing(-1), "t_end_high": h.crossing(+1), "rms_residual": math.sqrt(h.rss_min / h.n)}
    assert int(round(float(r["n_used"]))) == h.n, (label, "n_used", r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (label, "T0", r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
                     ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
                     ("rms_residual", 1e-3)):
        assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
    return ex

# An inclined NQ hole (1800 m MD, 28 to 40 degrees) with 357 readings every 5 m, a seven-layer Abitibi-type
# column, the glacial history (0, -1, +0.25) with the glaciation starting 100 ka and the Eemian before it, the
# end of the glaciation at 11.5 ka and 5 mK of logging noise; 326 readings have TVD >= 150 m.
h = _Hole(_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
          0.9e-6, 2.4e6, 4.1, 0.037, 7.8, 11500.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5], 150.0, 0.005, 7,
          log_step=5.0, log_top=15.0)
r = deglaciation_heat_flow_inversion(*h.args, (3000.0, 60000.0), 150.0)
keys = ["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end", "t_end_low", "t_end_high", "rms_residual"]
assert isinstance(r, dict) and set(keys + ["n_used"]) <= set(r)
for key in keys:
    assert type(r[key]) is float, key           # plain Python floats as stated
assert type(r["n_used"]) is int
_check(r, h, "inclined hole")

# --- test case 1 ---
# short hole with strong conductivity contrasts: few readings, asymmetric profile interval
import math
import numpy as np

_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_BX, _BW = np.polynomial.legendre.leggauss(24)

def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

def _g_tvd(md, inc_deg, azi_deg, m):
    # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total

def _g_RS(z, tops, k):
    # exact layer sums of int dz/k and int z dz/k
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b - top) * (b + top) / (2 * kk)
    return R, S

def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi

def _bc_U(z, tau_years, tops, k, rc, deriv=False):
    """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
    year, from the branch-cut (real-axis) form of the Laplace inversion:
    U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
    F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
    (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
    complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
    z = np.atleast_1d(np.asarray(z, float))
    tops = np.asarray(tops, float)
    k = np.asarray(k, float)
    tau = float(tau_years) * _YEAR
    nl = tops.size
    h = np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    width = min(0.25, 2.0 * math.pi / rate / 1.5)
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * _BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * _BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa = np.zeros((nl, w.size)); Pa = np.zeros((nl, w.size))
    Fb = np.zeros((nl, w.size)); Pb = np.zeros((nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
        c = np.cos(om[j] * h[j])
        skw = h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j]
        kws = k[j] * om[j] * np.sin(om[j] * h[j])
        Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
        Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
        sel = np.where(lay == j)[0]
        d = (z[sel] - tops[j])[:, None]
        if j == nl - 1:
            imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * om[j][None, :] * d)).imag
        else:
            fbz = Fb[j][None, :] * np.cos(om[j][None, :] * d) + Pb[j][None, :] * d * np.sinc(om[j][None, :] * d / np.pi) / k[j]
            imf[sel] = cc.imag[None, :] * fbz
    g = np.exp(-w * w)
    if deriv:
        out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _YEAR
        out[z <= 0] = 0.0
        return out
    out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 1.0
    return out

def _f95(nu):
    """0.95 quantile of the Fisher distribution F(1, nu): square of the two-sided 95 % Student t quantile, from
    the t density integrated by Gauss-Legendre quadrature and bisection."""
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)

    def central(x):
        e = np.linspace(0.0, x, 9)
        tot = 0.0
        for i in range(8):
            u = 0.5 * (e[i + 1] - e[i]) * _GL_X + 0.5 * (e[i] + e[i + 1])
            tot += (e[i + 1] - e[i]) * float(np.sum(_GL_W * c * (1.0 + u * u / nu) ** (-(nu + 1) / 2.0)))
        return tot

    lo, hi = 0.0, 1.0
    while central(hi) < 0.95:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    return (0.5 * (lo + hi)) ** 2

class _Hole:
    """Synthetic borehole whose least-squares solution is known exactly. On the used readings the data are
    y = T + A S = T0 + q0 R + g P(t_end) + r, with r orthogonal to the four columns of the Jacobian
    [1, R, P, g dP/dt_end] at the generating values, so those values are a stationary point of the residual
    sum of squares with RSS_min = |r|^2 exactly; P and dP/dt_end come from the oracle above (or from erfc for a
    uniform column). Readings shallower than z_min get disturbed values that follow no model."""

    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
                 log_md=None, log_step=None, log_top=None, uniform_k=None):
        self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
        tops_md, k = (np.asarray(v, float) for v in layers)
        if log_md is None:
            log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
        log_md = np.asarray(log_md, float)
        tops = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in tops_md])
        z = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in log_md])
        self.tops, self.k, self.rc, self.uniform_k = tops, k, rc, uniform_k
        keep = z >= z_min
        self.zk, self.n = z[keep], int(keep.sum())
        RS = np.array([_g_RS(v, list(tops), list(k)) for v in self.zk])
        self.R, self.S = RS[:, 0], RS[:, 1]
        sh = np.asarray(shape, float)
        tk = np.asarray(t_known, float)
        Uk = [self.U(t) for t in tk]
        Pf = sh[1] * Uk[0]
        for i in range(2, sh.size):
            Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
        self.Pfix, self.ds = Pf, sh[0] - sh[1]
        self.star = np.array([T0, q0, g, t_end], float)
        Pst = self.P(t_end)
        J = np.column_stack([np.ones(self.n), self.R, Pst, g * self.ds * self.U(t_end, True)])
        Q, _ = np.linalg.qr(J)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(self.n)
        r = w - Q @ (Q.T @ w)
        r = r - Q @ (Q.T @ r)
        r *= noise / math.sqrt(np.mean(r * r))
        self.y = T0 + q0 * self.R + g * Pst + r
        T = np.empty(z.size)
        T[keep] = self.y - A * self.S
        T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
        self.rss_min = float(r @ r)
        self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
        self.thr = self.rss_min * (1.0 + _f95(self.n - 4) / (self.n - 4))
        self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
        if self.uniform_k is None:
            return _bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
        kap, tau = self.uniform_k / self.rc, t * _YEAR      # uniform column: erfc and its time derivative
        if deriv:
            return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                             for v in self.zk]) * _YEAR
        return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

    def P(self, t):
        return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
        G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
        m, *_ = np.linalg.lstsq(G, self.y, rcond=None)
        res = self.y - G @ m
        return float(res @ res)

    def crossing(self, direction):
        """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
        in small steps, then bisection with Illinois steps."""
        ts = self.star[3]
        a, step = ts, 0.05 * self.sig[3]
        while True:
            b = a + direction * step
            if self.rss(b) >= self.thr:
                break
            a, step = b, min(step * 1.2, 0.3 * self.sig[3])
        fa, fb = self.rss(a) - self.thr, self.rss(b) - self.thr
        for _ in range(200):
            c = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < c < max(a, b)):
                c = 0.5 * (a + b)
            fc = self.rss(c) - self.thr
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) < 1e-12 * abs(b):
                break
        return b

def _check(r, h, label=""):
    """Every output against the exact values, with the stated accuracies."""
    ex = {"T0": h.star[0], "q0": h.star[1], "amplitude": h.star[2], "t_end": h.star[3],
          "sigma_T0": h.sig[0], "sigma_q0": h.sig[1], "sigma_amplitude": h.sig[2], "sigma_t_end": h.sig[3],
          "t_end_low": h.crossing(-1), "t_end_high": h.crossing(+1), "rms_residual": math.sqrt(h.rss_min / h.n)}
    assert int(round(float(r["n_used"]))) == h.n, (label, "n_used", r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (label, "T0", r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
                     ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
                     ("rms_residual", 1e-3)):
        assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
    return ex

# A 1000 m hole with readings every 25 m (38 used), five layers alternating 1.9-5.2 W m^-1 K^-1, a four-interval
# history (post-glacial +0.4, glaciation -1, earlier stadial -0.35, interglacial +0.15); with n_used - 4 = 34
# the F quantile (4.13) differs from the chi-square value (3.84) and the profile interval is asymmetric.
h = _Hole((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
          ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 2.8, 0.048, 6.0, 9000.0,
          [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], 40.0, 0.004, 11, log_step=25.0, log_top=10.0)
r = deglaciation_heat_flow_inversion(*h.args, (1500.0, 29900.0), 40.0)
ex = _check(r, h, "short hole")
assert ex["t_end_high"] - ex["t_end"] > 1.15 * (ex["t_end"] - ex["t_end_low"])      # oracle premise: asymmetric

# --- test case 2 ---
# uniform column on a vertical hole: closed-form erfc oracle, reading at z_min kept
import math
import numpy as np

_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_BX, _BW = np.polynomial.legendre.leggauss(24)

def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

def _g_tvd(md, inc_deg, azi_deg, m):
    # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total

def _g_RS(z, tops, k):
    # exact layer sums of int dz/k and int z dz/k
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b - top) * (b + top) / (2 * kk)
    return R, S

def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi

def _bc_U(z, tau_years, tops, k, rc, deriv=False):
    """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
    year, from the branch-cut (real-axis) form of the Laplace inversion:
    U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
    F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
    (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
    complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
    z = np.atleast_1d(np.asarray(z, float))
    tops = np.asarray(tops, float)
    k = np.asarray(k, float)
    tau = float(tau_years) * _YEAR
    nl = tops.size
    h = np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    width = min(0.25, 2.0 * math.pi / rate / 1.5)
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * _BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * _BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa = np.zeros((nl, w.size)); Pa = np.zeros((nl, w.size))
    Fb = np.zeros((nl, w.size)); Pb = np.zeros((nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
        c = np.cos(om[j] * h[j])
        skw = h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j]
        kws = k[j] * om[j] * np.sin(om[j] * h[j])
        Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
        Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
        sel = np.where(lay == j)[0]
        d = (z[sel] - tops[j])[:, None]
        if j == nl - 1:
            imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * om[j][None, :] * d)).imag
        else:
            fbz = Fb[j][None, :] * np.cos(om[j][None, :] * d) + Pb[j][None, :] * d * np.sinc(om[j][None, :] * d / np.pi) / k[j]
            imf[sel] = cc.imag[None, :] * fbz
    g = np.exp(-w * w)
    if deriv:
        out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _YEAR
        out[z <= 0] = 0.0
        return out
    out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 1.0
    return out

def _f95(nu):
    """0.95 quantile of the Fisher distribution F(1, nu): square of the two-sided 95 % Student t quantile, from
    the t density integrated by Gauss-Legendre quadrature and bisection."""
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)

    def central(x):
        e = np.linspace(0.0, x, 9)
        tot = 0.0
        for i in range(8):
            u = 0.5 * (e[i + 1] - e[i]) * _GL_X + 0.5 * (e[i] + e[i + 1])
            tot += (e[i + 1] - e[i]) * float(np.sum(_GL_W * c * (1.0 + u * u / nu) ** (-(nu + 1) / 2.0)))
        return tot

    lo, hi = 0.0, 1.0
    while central(hi) < 0.95:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    return (0.5 * (lo + hi)) ** 2

class _Hole:
    """Synthetic borehole whose least-squares solution is known exactly. On the used readings the data are
    y = T + A S = T0 + q0 R + g P(t_end) + r, with r orthogonal to the four columns of the Jacobian
    [1, R, P, g dP/dt_end] at the generating values, so those values are a stationary point of the residual
    sum of squares with RSS_min = |r|^2 exactly; P and dP/dt_end come from the oracle above (or from erfc for a
    uniform column). Readings shallower than z_min get disturbed values that follow no model."""

    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
                 log_md=None, log_step=None, log_top=None, uniform_k=None):
        self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
        tops_md, k = (np.asarray(v, float) for v in layers)
        if log_md is None:
            log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
        log_md = np.asarray(log_md, float)
        tops = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in tops_md])
        z = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in log_md])
        self.tops, self.k, self.rc, self.uniform_k = tops, k, rc, uniform_k
        keep = z >= z_min
        self.zk, self.n = z[keep], int(keep.sum())
        RS = np.array([_g_RS(v, list(tops), list(k)) for v in self.zk])
        self.R, self.S = RS[:, 0], RS[:, 1]
        sh = np.asarray(shape, float)
        tk = np.asarray(t_known, float)
        Uk = [self.U(t) for t in tk]
        Pf = sh[1] * Uk[0]
        for i in range(2, sh.size):
            Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
        self.Pfix, self.ds = Pf, sh[0] - sh[1]
        self.star = np.array([T0, q0, g, t_end], float)
        Pst = self.P(t_end)
        J = np.column_stack([np.ones(self.n), self.R, Pst, g * self.ds * self.U(t_end, True)])
        Q, _ = np.linalg.qr(J)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(self.n)
        r = w - Q @ (Q.T @ w)
        r = r - Q @ (Q.T @ r)
        r *= noise / math.sqrt(np.mean(r * r))
        self.y = T0 + q0 * self.R + g * Pst + r
        T = np.empty(z.size)
        T[keep] = self.y - A * self.S
        T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
        self.rss_min = float(r @ r)
        self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
        self.thr = self.rss_min * (1.0 + _f95(self.n - 4) / (self.n - 4))
        self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
        if self.uniform_k is None:
            return _bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
        kap, tau = self.uniform_k / self.rc, t * _YEAR      # uniform column: erfc and its time derivative
        if deriv:
            return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                             for v in self.zk]) * _YEAR
        return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

    def P(self, t):
        return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
        G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
        m, *_ = np.linalg.lstsq(G, self.y, rcond=None)
        res = self.y - G @ m
        return float(res @ res)

    def crossing(self, direction):
        """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
        in small steps, then bisection with Illinois steps."""
        ts = self.star[3]
        a, step = ts, 0.05 * self.sig[3]
        while True:
            b = a + direction * step
            if self.rss(b) >= self.thr:
                break
            a, step = b, min(step * 1.2, 0.3 * self.sig[3])
        fa, fb = self.rss(a) - self.thr, self.rss(b) - self.thr
        for _ in range(200):
            c = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < c < max(a, b)):
                c = 0.5 * (a + b)
            fc = self.rss(c) - self.thr
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) < 1e-12 * abs(b):
                break
        return b

def _check(r, h, label=""):
    """Every output against the exact values, with the stated accuracies."""
    ex = {"T0": h.star[0], "q0": h.star[1], "amplitude": h.star[2], "t_end": h.star[3],
          "sigma_T0": h.sig[0], "sigma_q0": h.sig[1], "sigma_amplitude": h.sig[2], "sigma_t_end": h.sig[3],
          "t_end_low": h.crossing(-1), "t_end_high": h.crossing(+1), "rms_residual": math.sqrt(h.rss_min / h.n)}
    assert int(round(float(r["n_used"]))) == h.n, (label, "n_used", r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (label, "T0", r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
                     ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
                     ("rms_residual", 1e-3)):
        assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
    return ex

# Equal conductivities: the layered perturbation is the erfc half-space, and the oracle uses math.erfc and its
# closed-form time derivative z / (2 sqrt(pi kappa) t^1.5) exp(-z^2 / (4 kappa t)). The collar reading (MD 0)
# has TVD 0 on every trajectory, so with z_min = 0 it lies exactly at the cut and is used: n_used = 121.
h = _Hole((np.array([0.0, 600.0, 1200.0]), np.zeros(3), np.zeros(3)), ([0, 400, 900], [3.1, 3.1, 3.1]),
          0.7e-6, 2.2e6, 5.5, 0.044, 6.2, 10500.0, [0.0, -1.0, 0.3], [9.0e4, 1.15e5], 0.0, 0.006, 99,
          log_step=10.0, log_top=0.0, uniform_k=3.1)
r = deglaciation_heat_flow_inversion(*h.args, (2000.0, 80000.0), 0.0)
assert h.n == 121
_check(r, h, "uniform column")

# --- test case 3 ---
# short glaciation: two minima of the profile, nearest threshold crossings
import math
import numpy as np

_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_BX, _BW = np.polynomial.legendre.leggauss(24)

def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

def _g_tvd(md, inc_deg, azi_deg, m):
    # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total

def _g_RS(z, tops, k):
    # exact layer sums of int dz/k and int z dz/k
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b - top) * (b + top) / (2 * kk)
    return R, S

def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi

def _bc_U(z, tau_years, tops, k, rc, deriv=False):
    """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
    year, from the branch-cut (real-axis) form of the Laplace inversion:
    U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
    F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
    (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
    complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
    z = np.atleast_1d(np.asarray(z, float))
    tops = np.asarray(tops, float)
    k = np.asarray(k, float)
    tau = float(tau_years) * _YEAR
    nl = tops.size
    h = np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    width = min(0.25, 2.0 * math.pi / rate / 1.5)
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * _BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * _BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa = np.zeros((nl, w.size)); Pa = np.zeros((nl, w.size))
    Fb = np.zeros((nl, w.size)); Pb = np.zeros((nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
        c = np.cos(om[j] * h[j])
        skw = h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j]
        kws = k[j] * om[j] * np.sin(om[j] * h[j])
        Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
        Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
        sel = np.where(lay == j)[0]
        d = (z[sel] - tops[j])[:, None]
        if j == nl - 1:
            imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * om[j][None, :] * d)).imag
        else:
            fbz = Fb[j][None, :] * np.cos(om[j][None, :] * d) + Pb[j][None, :] * d * np.sinc(om[j][None, :] * d / np.pi) / k[j]
            imf[sel] = cc.imag[None, :] * fbz
    g = np.exp(-w * w)
    if deriv:
        out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _YEAR
        out[z <= 0] = 0.0
        return out
    out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 1.0
    return out

def _f95(nu):
    """0.95 quantile of the Fisher distribution F(1, nu): square of the two-sided 95 % Student t quantile, from
    the t density integrated by Gauss-Legendre quadrature and bisection."""
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)

    def central(x):
        e = np.linspace(0.0, x, 9)
        tot = 0.0
        for i in range(8):
            u = 0.5 * (e[i + 1] - e[i]) * _GL_X + 0.5 * (e[i] + e[i + 1])
            tot += (e[i + 1] - e[i]) * float(np.sum(_GL_W * c * (1.0 + u * u / nu) ** (-(nu + 1) / 2.0)))
        return tot

    lo, hi = 0.0, 1.0
    while central(hi) < 0.95:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    return (0.5 * (lo + hi)) ** 2

class _Hole:
    """Synthetic borehole whose least-squares solution is known exactly. On the used readings the data are
    y = T + A S = T0 + q0 R + g P(t_end) + r, with r orthogonal to the four columns of the Jacobian
    [1, R, P, g dP/dt_end] at the generating values, so those values are a stationary point of the residual
    sum of squares with RSS_min = |r|^2 exactly; P and dP/dt_end come from the oracle above (or from erfc for a
    uniform column). Readings shallower than z_min get disturbed values that follow no model."""

    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
                 log_md=None, log_step=None, log_top=None, uniform_k=None):
        self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
        tops_md, k = (np.asarray(v, float) for v in layers)
        if log_md is None:
            log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
        log_md = np.asarray(log_md, float)
        tops = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in tops_md])
        z = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in log_md])
        self.tops, self.k, self.rc, self.uniform_k = tops, k, rc, uniform_k
        keep = z >= z_min
        self.zk, self.n = z[keep], int(keep.sum())
        RS = np.array([_g_RS(v, list(tops), list(k)) for v in self.zk])
        self.R, self.S = RS[:, 0], RS[:, 1]
        sh = np.asarray(shape, float)
        tk = np.asarray(t_known, float)
        Uk = [self.U(t) for t in tk]
        Pf = sh[1] * Uk[0]
        for i in range(2, sh.size):
            Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
        self.Pfix, self.ds = Pf, sh[0] - sh[1]
        self.star = np.array([T0, q0, g, t_end], float)
        Pst = self.P(t_end)
        J = np.column_stack([np.ones(self.n), self.R, Pst, g * self.ds * self.U(t_end, True)])
        Q, _ = np.linalg.qr(J)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(self.n)
        r = w - Q @ (Q.T @ w)
        r = r - Q @ (Q.T @ r)
        r *= noise / math.sqrt(np.mean(r * r))
        self.y = T0 + q0 * self.R + g * Pst + r
        T = np.empty(z.size)
        T[keep] = self.y - A * self.S
        T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
        self.rss_min = float(r @ r)
        self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
        self.thr = self.rss_min * (1.0 + _f95(self.n - 4) / (self.n - 4))
        self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
        if self.uniform_k is None:
            return _bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
        kap, tau = self.uniform_k / self.rc, t * _YEAR      # uniform column: erfc and its time derivative
        if deriv:
            return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                             for v in self.zk]) * _YEAR
        return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

    def P(self, t):
        return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
        G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
        m, *_ = np.linalg.lstsq(G, self.y, rcond=None)
        res = self.y - G @ m
        return float(res @ res)

    def crossing(self, direction):
        """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
        in small steps, then bisection with Illinois steps."""
        ts = self.star[3]
        a, step = ts, 0.05 * self.sig[3]
        while True:
            b = a + direction * step
            if self.rss(b) >= self.thr:
                break
            a, step = b, min(step * 1.2, 0.3 * self.sig[3])
        fa, fb = self.rss(a) - self.thr, self.rss(b) - self.thr
        for _ in range(200):
            c = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < c < max(a, b)):
                c = 0.5 * (a + b)
            fc = self.rss(c) - self.thr
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) < 1e-12 * abs(b):
                break
        return b

def _check(r, h, label=""):
    """Every output against the exact values, with the stated accuracies."""
    ex = {"T0": h.star[0], "q0": h.star[1], "amplitude": h.star[2], "t_end": h.star[3],
          "sigma_T0": h.sig[0], "sigma_q0": h.sig[1], "sigma_amplitude": h.sig[2], "sigma_t_end": h.sig[3],
          "t_end_low": h.crossing(-1), "t_end_high": h.crossing(+1), "rms_residual": math.sqrt(h.rss_min / h.n)}
    assert int(round(float(r["n_used"]))) == h.n, (label, "n_used", r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (label, "T0", r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
                     ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
                     ("rms_residual", 1e-3)):
        assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
    return ex

# The glaciation lasts from t_end to 30 ka only. The timing-amplitude trade-off gives the profile RSS(t) a
# second, higher local minimum near 23.97 ka (0.54 Delta above RSS_min, Delta = threshold - RSS_min) that dips
# below the 95 % threshold, separated from the global minimum at 11.5 ka by a ridge; the upper bound 23960 years
# lies inside that second basin (RSS(t_hi) < threshold, the profile decreasing towards t_hi). The estimate is the
# global minimum and the interval is bounded by the crossings nearest to it (about 10.9 and 12.1 ka).
h = _Hole(_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
          0.9e-6, 2.4e6, 4.1, 0.037, 7.8, 11500.0, [0.0, -1.0, 0.25], [3.0e4, 1.2e5], 150.0, 0.005, 21,
          log_step=5.0, log_top=15.0)
r = deglaciation_heat_flow_inversion(*h.args, (3000.0, 23960.0), 150.0)
ex = _check(r, h, "short glaciation")
assert ex["t_end_high"] < 13000.0
# bounds 9000-29900 years contain both basins (the second minimum is interior, both end values are far above
# the threshold); a local search over the interval started away from 11.5 ka descends into the second basin,
# but the global minimizer, its standard errors and the nearest crossings are unchanged
r = deglaciation_heat_flow_inversion(*h.args, (9000.0, 29900.0), 150.0)
_check(r, h, "short glaciation, both basins inside the bounds")

# --- test case 4 ---
# minimum at an end point of [t_lo, t_hi] raises; a secondary minimum beyond t_hi does not
import math
import numpy as np

_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_BX, _BW = np.polynomial.legendre.leggauss(24)

def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

def _g_tvd(md, inc_deg, azi_deg, m):
    # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total

def _g_RS(z, tops, k):
    # exact layer sums of int dz/k and int z dz/k
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b - top) * (b + top) / (2 * kk)
    return R, S

def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi

def _bc_U(z, tau_years, tops, k, rc, deriv=False):
    """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
    year, from the branch-cut (real-axis) form of the Laplace inversion:
    U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
    F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
    (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
    complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
    z = np.atleast_1d(np.asarray(z, float))
    tops = np.asarray(tops, float)
    k = np.asarray(k, float)
    tau = float(tau_years) * _YEAR
    nl = tops.size
    h = np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    width = min(0.25, 2.0 * math.pi / rate / 1.5)
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * _BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * _BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa = np.zeros((nl, w.size)); Pa = np.zeros((nl, w.size))
    Fb = np.zeros((nl, w.size)); Pb = np.zeros((nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
        c = np.cos(om[j] * h[j])
        skw = h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j]
        kws = k[j] * om[j] * np.sin(om[j] * h[j])
        Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
        Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
        sel = np.where(lay == j)[0]
        d = (z[sel] - tops[j])[:, None]
        if j == nl - 1:
            imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * om[j][None, :] * d)).imag
        else:
            fbz = Fb[j][None, :] * np.cos(om[j][None, :] * d) + Pb[j][None, :] * d * np.sinc(om[j][None, :] * d / np.pi) / k[j]
            imf[sel] = cc.imag[None, :] * fbz
    g = np.exp(-w * w)
    if deriv:
        out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _YEAR
        out[z <= 0] = 0.0
        return out
    out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 1.0
    return out

def _f95(nu):
    """0.95 quantile of the Fisher distribution F(1, nu): square of the two-sided 95 % Student t quantile, from
    the t density integrated by Gauss-Legendre quadrature and bisection."""
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)

    def central(x):
        e = np.linspace(0.0, x, 9)
        tot = 0.0
        for i in range(8):
            u = 0.5 * (e[i + 1] - e[i]) * _GL_X + 0.5 * (e[i] + e[i + 1])
            tot += (e[i + 1] - e[i]) * float(np.sum(_GL_W * c * (1.0 + u * u / nu) ** (-(nu + 1) / 2.0)))
        return tot

    lo, hi = 0.0, 1.0
    while central(hi) < 0.95:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    return (0.5 * (lo + hi)) ** 2

class _Hole:
    """Synthetic borehole whose least-squares solution is known exactly. On the used readings the data are
    y = T + A S = T0 + q0 R + g P(t_end) + r, with r orthogonal to the four columns of the Jacobian
    [1, R, P, g dP/dt_end] at the generating values, so those values are a stationary point of the residual
    sum of squares with RSS_min = |r|^2 exactly; P and dP/dt_end come from the oracle above (or from erfc for a
    uniform column). Readings shallower than z_min get disturbed values that follow no model."""

    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
                 log_md=None, log_step=None, log_top=None, uniform_k=None):
        self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
        tops_md, k = (np.asarray(v, float) for v in layers)
        if log_md is None:
            log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
        log_md = np.asarray(log_md, float)
        tops = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in tops_md])
        z = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in log_md])
        self.tops, self.k, self.rc, self.uniform_k = tops, k, rc, uniform_k
        keep = z >= z_min
        self.zk, self.n = z[keep], int(keep.sum())
        RS = np.array([_g_RS(v, list(tops), list(k)) for v in self.zk])
        self.R, self.S = RS[:, 0], RS[:, 1]
        sh = np.asarray(shape, float)
        tk = np.asarray(t_known, float)
        Uk = [self.U(t) for t in tk]
        Pf = sh[1] * Uk[0]
        for i in range(2, sh.size):
            Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
        self.Pfix, self.ds = Pf, sh[0] - sh[1]
        self.star = np.array([T0, q0, g, t_end], float)
        Pst = self.P(t_end)
        J = np.column_stack([np.ones(self.n), self.R, Pst, g * self.ds * self.U(t_end, True)])
        Q, _ = np.linalg.qr(J)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(self.n)
        r = w - Q @ (Q.T @ w)
        r = r - Q @ (Q.T @ r)
        r *= noise / math.sqrt(np.mean(r * r))
        self.y = T0 + q0 * self.R + g * Pst + r
        T = np.empty(z.size)
        T[keep] = self.y - A * self.S
        T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
        self.rss_min = float(r @ r)
        self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
        self.thr = self.rss_min * (1.0 + _f95(self.n - 4) / (self.n - 4))
        self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
        if self.uniform_k is None:
            return _bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
        kap, tau = self.uniform_k / self.rc, t * _YEAR      # uniform column: erfc and its time derivative
        if deriv:
            return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                             for v in self.zk]) * _YEAR
        return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

    def P(self, t):
        return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
        G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
        m, *_ = np.linalg.lstsq(G, self.y, rcond=None)
        res = self.y - G @ m
        return float(res @ res)

    def crossing(self, direction):
        """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
        in small steps, then bisection with Illinois steps."""
        ts = self.star[3]
        a, step = ts, 0.05 * self.sig[3]
        while True:
            b = a + direction * step
            if self.rss(b) >= self.thr:
                break
            a, step = b, min(step * 1.2, 0.3 * self.sig[3])
        fa, fb = self.rss(a) - self.thr, self.rss(b) - self.thr
        for _ in range(200):
            c = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < c < max(a, b)):
                c = 0.5 * (a + b)
            fc = self.rss(c) - self.thr
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) < 1e-12 * abs(b):
                break
        return b

def _check(r, h, label=""):
    """Every output against the exact values, with the stated accuracies."""
    ex = {"T0": h.star[0], "q0": h.star[1], "amplitude": h.star[2], "t_end": h.star[3],
          "sigma_T0": h.sig[0], "sigma_q0": h.sig[1], "sigma_amplitude": h.sig[2], "sigma_t_end": h.sig[3],
          "t_end_low": h.crossing(-1), "t_end_high": h.crossing(+1), "rms_residual": math.sqrt(h.rss_min / h.n)}
    assert int(round(float(r["n_used"]))) == h.n, (label, "n_used", r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (label, "T0", r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
                     ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
                     ("rms_residual", 1e-3)):
        assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
    return ex

h = _Hole(_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
          0.9e-6, 2.4e6, 4.1, 0.037, 7.8, 11500.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5], 150.0, 0.005, 7,
          log_step=5.0, log_top=15.0)
# the generating t_end = 11500 years lies outside these bounds: the profile decreases towards t_hi = 9000
# (first pair) or increases from t_lo = 14000 (second pair), so its minimum over the closed interval is an end point
for bounds in ((3000.0, 9000.0), (14000.0, 60000.0)):
    try:
        deglaciation_heat_flow_inversion(*h.args, bounds, 150.0)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError expected for bounds %r" % (bounds,))
# wide bounds: the profile rises to a ridge near 88.5 ka and then decreases towards t_hi = 92000 (it dips again
# near 96 ka, where the design [1, R, P] becomes nearly degenerate, beyond t_hi), but the global minimum over the
# interval is interior and unchanged
r = deglaciation_heat_flow_inversion(*h.args, (2500.0, 92000.0), 150.0)
_check(r, h, "wide bounds")

# --- test case 5 ---
# profile interval not contained in [t_lo, t_hi] raises
import math
import numpy as np

_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_BX, _BW = np.polynomial.legendre.leggauss(24)

def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

def _g_tvd(md, inc_deg, azi_deg, m):
    # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total

def _g_RS(z, tops, k):
    # exact layer sums of int dz/k and int z dz/k
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b - top) * (b + top) / (2 * kk)
    return R, S

def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi

def _bc_U(z, tau_years, tops, k, rc, deriv=False):
    """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
    year, from the branch-cut (real-axis) form of the Laplace inversion:
    U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
    F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
    (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
    complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
    z = np.atleast_1d(np.asarray(z, float))
    tops = np.asarray(tops, float)
    k = np.asarray(k, float)
    tau = float(tau_years) * _YEAR
    nl = tops.size
    h = np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    width = min(0.25, 2.0 * math.pi / rate / 1.5)
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * _BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * _BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa = np.zeros((nl, w.size)); Pa = np.zeros((nl, w.size))
    Fb = np.zeros((nl, w.size)); Pb = np.zeros((nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
        c = np.cos(om[j] * h[j])
        skw = h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j]
        kws = k[j] * om[j] * np.sin(om[j] * h[j])
        Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
        Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
        sel = np.where(lay == j)[0]
        d = (z[sel] - tops[j])[:, None]
        if j == nl - 1:
            imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * om[j][None, :] * d)).imag
        else:
            fbz = Fb[j][None, :] * np.cos(om[j][None, :] * d) + Pb[j][None, :] * d * np.sinc(om[j][None, :] * d / np.pi) / k[j]
            imf[sel] = cc.imag[None, :] * fbz
    g = np.exp(-w * w)
    if deriv:
        out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _YEAR
        out[z <= 0] = 0.0
        return out
    out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 1.0
    return out

def _f95(nu):
    """0.95 quantile of the Fisher distribution F(1, nu): square of the two-sided 95 % Student t quantile, from
    the t density integrated by Gauss-Legendre quadrature and bisection."""
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)

    def central(x):
        e = np.linspace(0.0, x, 9)
        tot = 0.0
        for i in range(8):
            u = 0.5 * (e[i + 1] - e[i]) * _GL_X + 0.5 * (e[i] + e[i + 1])
            tot += (e[i + 1] - e[i]) * float(np.sum(_GL_W * c * (1.0 + u * u / nu) ** (-(nu + 1) / 2.0)))
        return tot

    lo, hi = 0.0, 1.0
    while central(hi) < 0.95:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    return (0.5 * (lo + hi)) ** 2

class _Hole:
    """Synthetic borehole whose least-squares solution is known exactly. On the used readings the data are
    y = T + A S = T0 + q0 R + g P(t_end) + r, with r orthogonal to the four columns of the Jacobian
    [1, R, P, g dP/dt_end] at the generating values, so those values are a stationary point of the residual
    sum of squares with RSS_min = |r|^2 exactly; P and dP/dt_end come from the oracle above (or from erfc for a
    uniform column). Readings shallower than z_min get disturbed values that follow no model."""

    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
                 log_md=None, log_step=None, log_top=None, uniform_k=None):
        self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
        tops_md, k = (np.asarray(v, float) for v in layers)
        if log_md is None:
            log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
        log_md = np.asarray(log_md, float)
        tops = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in tops_md])
        z = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in log_md])
        self.tops, self.k, self.rc, self.uniform_k = tops, k, rc, uniform_k
        keep = z >= z_min
        self.zk, self.n = z[keep], int(keep.sum())
        RS = np.array([_g_RS(v, list(tops), list(k)) for v in self.zk])
        self.R, self.S = RS[:, 0], RS[:, 1]
        sh = np.asarray(shape, float)
        tk = np.asarray(t_known, float)
        Uk = [self.U(t) for t in tk]
        Pf = sh[1] * Uk[0]
        for i in range(2, sh.size):
            Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
        self.Pfix, self.ds = Pf, sh[0] - sh[1]
        self.star = np.array([T0, q0, g, t_end], float)
        Pst = self.P(t_end)
        J = np.column_stack([np.ones(self.n), self.R, Pst, g * self.ds * self.U(t_end, True)])
        Q, _ = np.linalg.qr(J)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(self.n)
        r = w - Q @ (Q.T @ w)
        r = r - Q @ (Q.T @ r)
        r *= noise / math.sqrt(np.mean(r * r))
        self.y = T0 + q0 * self.R + g * Pst + r
        T = np.empty(z.size)
        T[keep] = self.y - A * self.S
        T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
        self.rss_min = float(r @ r)
        self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
        self.thr = self.rss_min * (1.0 + _f95(self.n - 4) / (self.n - 4))
        self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
        if self.uniform_k is None:
            return _bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
        kap, tau = self.uniform_k / self.rc, t * _YEAR      # uniform column: erfc and its time derivative
        if deriv:
            return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                             for v in self.zk]) * _YEAR
        return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

    def P(self, t):
        return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
        G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
        m, *_ = np.linalg.lstsq(G, self.y, rcond=None)
        res = self.y - G @ m
        return float(res @ res)

    def crossing(self, direction):
        """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
        in small steps, then bisection with Illinois steps."""
        ts = self.star[3]
        a, step = ts, 0.05 * self.sig[3]
        while True:
            b = a + direction * step
            if self.rss(b) >= self.thr:
                break
            a, step = b, min(step * 1.2, 0.3 * self.sig[3])
        fa, fb = self.rss(a) - self.thr, self.rss(b) - self.thr
        for _ in range(200):
            c = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < c < max(a, b)):
                c = 0.5 * (a + b)
            fc = self.rss(c) - self.thr
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) < 1e-12 * abs(b):
                break
        return b

def _check(r, h, label=""):
    """Every output against the exact values, with the stated accuracies."""
    ex = {"T0": h.star[0], "q0": h.star[1], "amplitude": h.star[2], "t_end": h.star[3],
          "sigma_T0": h.sig[0], "sigma_q0": h.sig[1], "sigma_amplitude": h.sig[2], "sigma_t_end": h.sig[3],
          "t_end_low": h.crossing(-1), "t_end_high": h.crossing(+1), "rms_residual": math.sqrt(h.rss_min / h.n)}
    assert int(round(float(r["n_used"]))) == h.n, (label, "n_used", r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (label, "T0", r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
                     ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
                     ("rms_residual", 1e-3)):
        assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
    return ex

h = _Hole((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
          ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 2.8, 0.048, 6.0, 9000.0,
          [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], 40.0, 0.004, 11, log_step=25.0, log_top=10.0)
# the nearest crossings of this hole lie near 7.73 ka and 10.73 ka; with t_lo = 8500 (minimum still interior)
# the lower crossing is outside the bounds, with t_hi = 10000 the upper one
for bounds in ((8500.0, 29900.0), (1500.0, 10000.0)):
    try:
        deglaciation_heat_flow_inversion(*h.args, bounds, 40.0)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError expected for bounds %r" % (bounds,))
# with t_lo = 7000 both crossings are inside
r = deglaciation_heat_flow_inversion(*h.args, (7000.0, 29900.0), 40.0)
_check(r, h, "bounds 7000-29900")

# --- test case 6 ---
# five readings is the smallest valid inversion; invalid inputs raise
import math
import numpy as np

_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_BX, _BW = np.polynomial.legendre.leggauss(24)

def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

def _g_tvd(md, inc_deg, azi_deg, m):
    # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total

def _g_RS(z, tops, k):
    # exact layer sums of int dz/k and int z dz/k
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b - top) * (b + top) / (2 * kk)
    return R, S

def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi

def _bc_U(z, tau_years, tops, k, rc, deriv=False):
    """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
    year, from the branch-cut (real-axis) form of the Laplace inversion:
    U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
    F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
    (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
    complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
    z = np.atleast_1d(np.asarray(z, float))
    tops = np.asarray(tops, float)
    k = np.asarray(k, float)
    tau = float(tau_years) * _YEAR
    nl = tops.size
    h = np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    width = min(0.25, 2.0 * math.pi / rate / 1.5)
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * _BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * _BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa = np.zeros((nl, w.size)); Pa = np.zeros((nl, w.size))
    Fb = np.zeros((nl, w.size)); Pb = np.zeros((nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
        c = np.cos(om[j] * h[j])
        skw = h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j]
        kws = k[j] * om[j] * np.sin(om[j] * h[j])
        Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
        Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
        sel = np.where(lay == j)[0]
        d = (z[sel] - tops[j])[:, None]
        if j == nl - 1:
            imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * om[j][None, :] * d)).imag
        else:
            fbz = Fb[j][None, :] * np.cos(om[j][None, :] * d) + Pb[j][None, :] * d * np.sinc(om[j][None, :] * d / np.pi) / k[j]
            imf[sel] = cc.imag[None, :] * fbz
    g = np.exp(-w * w)
    if deriv:
        out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _YEAR
        out[z <= 0] = 0.0
        return out
    out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 1.0
    return out

def _f95(nu):
    """0.95 quantile of the Fisher distribution F(1, nu): square of the two-sided 95 % Student t quantile, from
    the t density integrated by Gauss-Legendre quadrature and bisection."""
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)

    def central(x):
        e = np.linspace(0.0, x, 9)
        tot = 0.0
        for i in range(8):
            u = 0.5 * (e[i + 1] - e[i]) * _GL_X + 0.5 * (e[i] + e[i + 1])
            tot += (e[i + 1] - e[i]) * float(np.sum(_GL_W * c * (1.0 + u * u / nu) ** (-(nu + 1) / 2.0)))
        return tot

    lo, hi = 0.0, 1.0
    while central(hi) < 0.95:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if central(mid) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    return (0.5 * (lo + hi)) ** 2

class _Hole:
    """Synthetic borehole whose least-squares solution is known exactly. On the used readings the data are
    y = T + A S = T0 + q0 R + g P(t_end) + r, with r orthogonal to the four columns of the Jacobian
    [1, R, P, g dP/dt_end] at the generating values, so those values are a stationary point of the residual
    sum of squares with RSS_min = |r|^2 exactly; P and dP/dt_end come from the oracle above (or from erfc for a
    uniform column). Readings shallower than z_min get disturbed values that follow no model."""

    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
                 log_md=None, log_step=None, log_top=None, uniform_k=None):
        self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
        tops_md, k = (np.asarray(v, float) for v in layers)
        if log_md is None:
            log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
        log_md = np.asarray(log_md, float)
        tops = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in tops_md])
        z = np.array([_g_tvd(self.md, self.inc, self.azi, m) for m in log_md])
        self.tops, self.k, self.rc, self.uniform_k = tops, k, rc, uniform_k
        keep = z >= z_min
        self.zk, self.n = z[keep], int(keep.sum())
        RS = np.array([_g_RS(v, list(tops), list(k)) for v in self.zk])
        self.R, self.S = RS[:, 0], RS[:, 1]
        sh = np.asarray(shape, float)
        tk = np.asarray(t_known, float)
        Uk = [self.U(t) for t in tk]
        Pf = sh[1] * Uk[0]
        for i in range(2, sh.size):
            Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
        self.Pfix, self.ds = Pf, sh[0] - sh[1]
        self.star = np.array([T0, q0, g, t_end], float)
        Pst = self.P(t_end)
        J = np.column_stack([np.ones(self.n), self.R, Pst, g * self.ds * self.U(t_end, True)])
        Q, _ = np.linalg.qr(J)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(self.n)
        r = w - Q @ (Q.T @ w)
        r = r - Q @ (Q.T @ r)
        r *= noise / math.sqrt(np.mean(r * r))
        self.y = T0 + q0 * self.R + g * Pst + r
        T = np.empty(z.size)
        T[keep] = self.y - A * self.S
        T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
        self.rss_min = float(r @ r)
        self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
        self.thr = self.rss_min * (1.0 + _f95(self.n - 4) / (self.n - 4))
        self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
        if self.uniform_k is None:
            return _bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
        kap, tau = self.uniform_k / self.rc, t * _YEAR      # uniform column: erfc and its time derivative
        if deriv:
            return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                             for v in self.zk]) * _YEAR
        return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

    def P(self, t):
        return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
        G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
        m, *_ = np.linalg.lstsq(G, self.y, rcond=None)
        res = self.y - G @ m
        return float(res @ res)

    def crossing(self, direction):
        """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
        in small steps, then bisection with Illinois steps."""
        ts = self.star[3]
        a, step = ts, 0.05 * self.sig[3]
        while True:
            b = a + direction * step
            if self.rss(b) >= self.thr:
                break
            a, step = b, min(step * 1.2, 0.3 * self.sig[3])
        fa, fb = self.rss(a) - self.thr, self.rss(b) - self.thr
        for _ in range(200):
            c = b - fb * (b - a) / (fb - fa)
            if not (min(a, b) < c < max(a, b)):
                c = 0.5 * (a + b)
            fc = self.rss(c) - self.thr
            if fc * fb < 0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb = c, fc
            if abs(b - a) < 1e-12 * abs(b):
                break
        return b

def _check(r, h, label=""):
    """Every output against the exact values, with the stated accuracies."""
    ex = {"T0": h.star[0], "q0": h.star[1], "amplitude": h.star[2], "t_end": h.star[3],
          "sigma_T0": h.sig[0], "sigma_q0": h.sig[1], "sigma_amplitude": h.sig[2], "sigma_t_end": h.sig[3],
          "t_end_low": h.crossing(-1), "t_end_high": h.crossing(+1), "rms_residual": math.sqrt(h.rss_min / h.n)}
    assert int(round(float(r["n_used"]))) == h.n, (label, "n_used", r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (label, "T0", r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
                     ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
                     ("rms_residual", 1e-3)):
        assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
    return ex

# vertical hole, seven readings; z_min = 200 m keeps exactly five (n_used - 4 = 1 degree of freedom,
# F(1, 1) = 161.4) with a 0.2 mK residual, z_min = 400 m keeps four (ValueError)
h = _Hole((np.array([0.0, 1300.0]), np.zeros(2), np.zeros(2)), ([0, 350, 700], [2.6, 3.8, 3.0]),
          0.8e-6, 2.3e6, 6.0, 0.045, 5.5, 10000.0, [0.0, -1.0, 0.25], [9.0e4, 1.2e5], 200.0, 0.0002, 5,
          log_md=np.array([60.0, 180.0, 330.0, 520.0, 760.0, 1000.0, 1280.0]))
B = (3000.0, 60000.0)
r = deglaciation_heat_flow_inversion(*h.args, B, 200.0)
_check(r, h, "five readings")
ARGS = ["survey_md", "survey_inc", "survey_azi", "log_md", "log_temp", "layer_top_md", "layer_k",
        "heat_production", "rho_c", "hist_t_years", "hist_dT_shape", "t_end_bounds", "z_min"]
ix = {a: j for j, a in enumerate(ARGS)}
def _bad(name):
    d = [a.copy() if isinstance(a, np.ndarray) else a for a in h.args] + [B, 200.0]
    if name == "four_readings":
        d[ix["z_min"]] = 400.0
    elif name == "log_len":
        d[ix["log_temp"]] = d[ix["log_temp"]][:-1]
    elif name == "log_two_d":
        d[ix["log_md"]] = d[ix["log_md"]][None, :]
        d[ix["log_temp"]] = d[ix["log_temp"]][None, :]
    elif name == "temp_nan":
        d[ix["log_temp"]][4] = float("nan")
    elif name == "one_interval":
        d[ix["hist_dT_shape"]] = np.array([-1.0])
        d[ix["hist_t_years"]] = np.array([], float)
    elif name == "times_count":
        d[ix["hist_t_years"]] = np.array([9.0e4, 1.2e5, 1.5e5])
    elif name == "times_not_increasing":
        d[ix["hist_t_years"]] = np.array([9.0e4, 9.0e4])
    elif name == "times_nan":
        d[ix["hist_t_years"]] = np.array([9.0e4, float("nan")])
    elif name == "shape_nan":
        d[ix["hist_dT_shape"]] = np.array([0.0, -1.0, float("nan")])
    elif name == "times_nonpositive":
        d[ix["hist_t_years"]] = np.array([-9.0e4, 1.2e5])
    elif name == "equal_first_shapes":
        d[ix["hist_dT_shape"]] = np.array([-1.0, -1.0, 0.25])
    elif name == "bounds_lo_zero":
        d[ix["t_end_bounds"]] = (0.0, 60000.0)
    elif name == "bounds_reversed":
        d[ix["t_end_bounds"]] = (60000.0, 3000.0)
    elif name == "bounds_beyond_glaciation":
        d[ix["t_end_bounds"]] = (3000.0, 95000.0)
    elif name == "bounds_at_glaciation_start":
        d[ix["t_end_bounds"]] = (3000.0, 90000.0)        # t_hi == hist_t_years[0]: the upper limit is strict
    elif name == "bounds_equal":
        d[ix["t_end_bounds"]] = (5000.0, 5000.0)
    elif name == "shape_two_d":
        d[ix["hist_dT_shape"]] = np.array([[0.0, -1.0, 0.25]])
    elif name == "times_two_d":
        d[ix["hist_t_years"]] = np.array([[9.0e4, 1.2e5]])
    elif name == "bounds_nan":
        d[ix["t_end_bounds"]] = (float("nan"), 60000.0)
    elif name == "bounds_three":
        d[ix["t_end_bounds"]] = (3000.0, 30000.0, 60000.0)
    elif name == "A_negative":
        d[ix["heat_production"]] = -1e-6
    elif name == "A_nan":
        d[ix["heat_production"]] = float("nan")
    elif name == "rho_c_zero":
        d[ix["rho_c"]] = 0.0
    elif name == "k_zero":
        d[ix["layer_k"]] = np.array([2.6, 0.0, 3.0])
    elif name == "k_count":
        d[ix["layer_k"]] = np.array([2.6, 3.8])
    elif name == "top_beyond_hole":
        d[ix["layer_top_md"]] = np.array([0.0, 350.0, 1400.0])
    elif name == "log_beyond_hole":
        d[ix["log_md"]] = np.array([60.0, 180.0, 330.0, 520.0, 760.0, 1000.0, 1320.0])
    elif name == "survey_start":
        d[ix["survey_md"]] = np.array([5.0, 1300.0])
    elif name == "one_depth":
        d[ix["log_md"]] = np.full(7, 760.0)          # every reading at one depth: T0, q0, g not resolvable
    return d
for name in ["four_readings", "log_len", "log_two_d", "temp_nan", "one_interval", "times_count",
             "times_not_increasing", "times_nan", "shape_nan", "times_nonpositive", "equal_first_shapes",
             "bounds_lo_zero", "bounds_reversed", "bounds_beyond_glaciation", "bounds_at_glaciation_start",
             "bounds_equal", "shape_two_d", "times_two_d", "bounds_nan", "bounds_three",
             "A_negative", "A_nan", "rho_c_zero", "k_zero", "k_count", "top_beyond_hole", "log_beyond_hole",
             "survey_start", "one_depth"]:
    try:
        deglaciation_heat_flow_inversion(*_bad(name))
    except ValueError:
        pass
    else:
        raise AssertionError("deglaciation_heat_flow_inversion must raise ValueError (%s)" % name)
