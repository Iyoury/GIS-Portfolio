# Whole-task tests: deglaciation_heat_flow_inversion with every earlier step taken from the candidate solution.
# Boreholes and exact targets are built as in tests/step_6.py (orthogonal-residual construction, branch-cut
# oracle for the layered column); the accuracy is the stated one, which allows for the stated accuracies of
# steps 1, 2 and 5.

# --- test case 0 ---
# deep hole: the whole chain against the exact solution
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

# A 2400 m hole (8 to 15 degrees), 220 readings below 200 m TVD, five layers, a four-interval history with a
# glaciation from 60 ka to t_end = 9.8 ka and 10 mK noise; t_end is resolved to about 70 years.
h = _Hole(_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
          1.8e-6, 2.5e6, 3.6, 0.052, 5.0, 9800.0, [0.2, -1.0, -0.45, 0.3], [6.0e4, 9.5e4, 1.3e5], 200.0, 0.010, 4242,
          log_step=10.0, log_top=30.0)
r = deglaciation_heat_flow_inversion(*h.args, (2000.0, 55000.0), 200.0)
_check(r, h, "deep hole")

# --- test case 1 ---
# short hole: the whole chain against the exact solution; a constant offset moves T0 only
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
          [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], 40.0, 0.004, 12, log_step=25.0, log_top=10.0)
r = deglaciation_heat_flow_inversion(*h.args, (1500.0, 29900.0), 40.0)
_check(r, h, "short hole")
# adding 0.75 K to every reading changes only T0 (by 0.75 K): the least-squares problem is otherwise identical;
# two computed outputs are compared with twice the single-output accuracy
d = list(h.args)
d[4] = d[4] + 0.75
r2 = deglaciation_heat_flow_inversion(*d, (1500.0, 29900.0), 40.0)
assert abs((r2["T0"] - r["T0"]) - 0.75) < 4e-3
for key, tol in (("q0", 1e-3), ("amplitude", 1e-3), ("t_end", 1e-3), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2),
                 ("sigma_amplitude", 1e-2), ("sigma_t_end", 1e-2), ("t_end_low", 1e-3), ("t_end_high", 1e-3),
                 ("rms_residual", 2e-3)):
    assert abs(r2[key] / r[key] - 1.0) < tol, key
assert r2["n_used"] == r["n_used"]

# --- test case 2 ---
# short glaciation: the whole chain with a second profile minimum inside the bounds
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

# glaciation from t_end = 12.2 ka to 30 ka; the profile has a second local minimum near 23.8 ka below the 95 %
# threshold, and t_hi = 23800 years lies inside that basin; the interval is bounded by the nearest crossings
h = _Hole(_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
          0.9e-6, 2.4e6, 3.4, 0.041, 6.9, 12200.0, [0.0, -1.0, 0.25], [3.0e4, 1.2e5], 150.0, 0.005, 33,
          log_step=5.0, log_top=15.0)
r = deglaciation_heat_flow_inversion(*h.args, (3500.0, 23800.0), 150.0)
ex = _check(r, h, "short glaciation")
assert ex["t_end_high"] < 14000.0
# bounds 9000-29900 years contain both basins (second minimum 0.39 Delta above RSS_min, interior); the global
# minimizer at 12.2 ka, its standard errors and the nearest crossings are unchanged
r = deglaciation_heat_flow_inversion(*h.args, (9000.0, 29900.0), 150.0)
_check(r, h, "short glaciation, both basins inside the bounds")
