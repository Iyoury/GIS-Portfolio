# Tests for step 7: regional_deglaciation_inversion (regional maximum-likelihood inversion of several borehole logs
# that share one glacial history: amplitude g and end of the glaciation t_end common to all holes, a steady
# geotherm T0_h, q0_h and an unknown noise level sigma_h for each hole).
# No expected value comes from the reference solution. Each case builds its boreholes forward with numpy and math
# only (TVD by 64-point Gauss-Legendre integration of the great-circle tangent, R and S by exact layer sums) and
# constructs the temperatures so that the maximum-likelihood solution is known exactly: on the used readings of
# hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h, with r_h = b_h + c_h, b_h orthogonal to the four
# Jacobian columns [1, R_h, P_h, g dP_h/dt_end] (seeded), c_h in the span of the projections of P_h and
# g dP_h/dt_end off [1, R_h], and one hole's c_h solved in closed form so that the weighted score
# sum_h (n_h / |r_h|^2) [P_h, g dP_h/dt_end]^T r_h vanishes while the unweighted (pooled least-squares) one does not.
# With sigma_h^2 = |r_h|^2 / n_h every score of the log-likelihood is zero at the generating values; dense scans of
# the oracle profile (source.md) show that they are the global maximum over the bounds of every case. P and
# dP/dt_end come from an independent oracle, the branch-cut (real-axis) integral of the Laplace solution of each
# layered column with Gauss-Legendre quadrature, or from erfc for a uniform column. The standard errors come from
# the expected Fisher information of the oracle Jacobians at sigma_h, the chi-square quantile from math.erf, and
# the interval ends are the oracle profile's nearest crossings (the maximum over g at fixed t_end bracketed between
# the holes' own optima). Accuracy as stated: g and t_end relative 2e-3; q0 relative 5e-4; T0 2e-3 K; standard
# errors relative 1e-2; interval ends relative 5e-4; noise_sd relative 3e-3; log_likelihood 0.1; n_used exact.

# --- test case 0 ---
# three holes of different quality: every output against the exact maximum-likelihood solution
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    def _t7_incl7(step):
        # inclined NQ hole, 1800 m MD (28 to 40 degrees), seven-layer Abitibi-type column, z_min 150 m
        return _t7_Bore(_t7_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
                     0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

    def _t7_mid4():
        # 1200 m hole (5 to 12 degrees), four layers, higher heat production, z_min 100 m
        return _t7_Bore(_t7_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                     log_step=8.0, log_top=20.0)

    def _t7_short3():
        # 900 m hole (15 to 22 degrees), three layers, z_min 80 m
        return _t7_Bore(_t7_g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
                     log_step=6.0, log_top=10.0)

    # Three holes with 4, 12 and 25 mK of logging noise (163, 137 and 136 readings used), each with its own column, heat
    # production, rho_c, z_min, T0 and q0; glacial history (0, -1, +0.25) with the glaciation from t_end to 100 ka.
    # Pooled least squares (one common noise variance) would give t_end = 11.40 ka and g = 7.72 instead of 11.5 ka, 7.8.
    reg = _t7_Region([_t7_incl7(10.0), _t7_mid4(), _t7_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0, [0.0, -1.0, 0.25],
                  [1.0e5, 1.2e5], [0.004, 0.012, 0.025], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
    r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0))

    keys = ["g", "t_end", "sigma_g", "sigma_t_end", "t_end_low", "t_end_high", "log_likelihood"]
    assert isinstance(r, dict) and set(keys + ["holes"]) <= set(r)
    for key in keys:
        assert type(r[key]) is float, key           # plain Python floats as stated
    assert isinstance(r["holes"], list) and len(r["holes"]) == 3
    for rh in r["holes"]:
        assert isinstance(rh, dict)
        for key in ("T0", "q0", "sigma_T0", "sigma_q0", "noise_sd"):
            assert type(rh[key]) is float, key
        assert type(rh["n_used"]) is int
    _t7_check(r, reg, "three holes")


_crown_case()

# --- test case 1 ---
# a short precise log and a long noisy one: maximum-likelihood weighting of the two holes
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    def _t7_contrast5():
        # 1000 m hole (45 to 62 degrees from the collar), five strongly contrasted layers, readings every 40 m, z_min 40 m
        return _t7_Bore((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
                     ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 40.0, log_step=40.0, log_top=10.0)

    def _t7_deep5():
        # 2400 m hole (8 to 15 degrees), five layers, readings every 10 m, z_min 200 m
        return _t7_Bore(_t7_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                     1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0)

    # A 1000 m hole logged every 40 m with 2 mK noise (24 readings used) and a 2400 m hole logged every 10 m with 20 mK
    # (220 used); four-interval history (post-glacial +0.4, glaciation -1, stadial -0.35, interglacial +0.15). The
    # per-hole variances weight the precise short log heavily: pooled least squares would give t_end = 9.17 ka
    # (4.4 standard errors from the estimate), and the variance RSS_h / (n_h - 2) of the short hole changes its weight by 9 %.
    reg = _t7_Region([_t7_contrast5(), _t7_deep5()], [2.8, 3.6], [0.048, 0.052], 6.0, 9000.0, [0.4, -1.0, -0.35, 0.15],
                  [3.0e4, 7.5e4, 1.3e5], [0.002, 0.02], [(1.0, 1.0), None], 11, 1)
    r = regional_deglaciation_inversion(reg.holes, [3.0e4, 7.5e4, 1.3e5], [0.4, -1.0, -0.35, 0.15], (1500.0, 29900.0))
    _t7_check(r, reg, "short precise and long noisy hole")


_crown_case()

# --- test case 2 ---
# uniform columns: closed-form erfc oracle, the collar reading at z_min = 0 kept, wide asymmetric interval
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    # Two vertical holes whose layers all have one conductivity (3.1 and 2.4 W m^-1 K^-1) and different rho_c: the
    # layered perturbation is the erfc half-space of each hole's own diffusivity, and the oracle uses math.erfc and its
    # closed-form time derivative. The collar reading (MD 0, TVD 0) of the first hole lies exactly at z_min = 0 and is
    # used (121 readings); the second hole has 57 readings below 50 m. The log-likelihood profile is asymmetric.
    _t7_bA = _t7_Bore((np.array([0.0, 600.0, 1200.0]), np.zeros(3), np.zeros(3)), ([0, 400, 900], [3.1, 3.1, 3.1]), 0.7e-6, 2.2e6, 0.0,
               log_step=10.0, log_top=0.0, uniform_k=3.1)
    _t7_bB = _t7_Bore((np.array([0.0, 900.0]), np.zeros(2), np.zeros(2)), ([0, 300], [2.4, 2.4]), 1.2e-6, 2.6e6, 50.0,
               log_step=15.0, log_top=5.0, uniform_k=2.4)
    reg = _t7_Region([_t7_bA, _t7_bB], [5.5, 2.9], [0.044, 0.061], 6.2, 10500.0, [0.0, -1.0, 0.3], [9.0e4, 1.15e5], [0.006, 0.015],
                  [None, (-2.0, 1.5)], 99, 0)
    assert _t7_bA.n == 121
    r = regional_deglaciation_inversion(reg.holes, [9.0e4, 1.15e5], [0.0, -1.0, 0.3], (2000.0, 80000.0))
    lo_x, hi_x = _t7_check(r, reg, "uniform columns")
    assert hi_x - reg.t_end > 1.2 * (reg.t_end - lo_x)          # oracle premise: asymmetric interval


_crown_case()

# --- test case 3 ---
# short glaciation: a second maximum of the profile likelihood inside the bounds
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    def _t7_incl7(step):
        # inclined NQ hole, 1800 m MD (28 to 40 degrees), seven-layer Abitibi-type column, z_min 150 m
        return _t7_Bore(_t7_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
                     0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

    def _t7_mid4():
        # 1200 m hole (5 to 12 degrees), four layers, higher heat production, z_min 100 m
        return _t7_Bore(_t7_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                     log_step=8.0, log_top=20.0)

    def _t7_short3():
        # 900 m hole (15 to 22 degrees), three layers, z_min 80 m
        return _t7_Bore(_t7_g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
                     log_step=6.0, log_top=10.0)

    # The glaciation lasts from t_end to 24 ka only. The amplitude-timing trade-off gives the profile log-likelihood a
    # second local maximum near 17.98 ka, 9.96 below l_max, separated from the global maximum at 11.5 ka by a valley
    # near 15.7 ka (20.5 below l_max); both lie inside the bounds 9000-23900 years. A local search over the bounds
    # without a global scan ends in the second basin.
    reg = _t7_Region([_t7_incl7(5.0), _t7_mid4(), _t7_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0, [0.0, -1.0, 0.25],
                  [2.4e4, 1.2e5], [0.005, 0.010, 0.020], [None, (1.5, -1.0), (-1.0, 1.5)], 21, 0)
    assert reg.lp(17976.0) - reg.lp(15700.0) > 5.0 and reg.lp(17976.0) < reg.thr      # oracle premise: second basin
    r = regional_deglaciation_inversion(reg.holes, [2.4e4, 1.2e5], [0.0, -1.0, 0.25], (9000.0, 23900.0))
    _t7_check(r, reg, "short glaciation")


_crown_case()

# --- test case 4 ---
# maximum at an end point of [t_lo, t_hi] raises; narrow bounds around the estimate are valid; a maximum just
# inside either bound (precise logs) is returned
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    def _t7_incl7(step):
        # inclined NQ hole, 1800 m MD (28 to 40 degrees), seven-layer Abitibi-type column, z_min 150 m
        return _t7_Bore(_t7_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
                     0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

    def _t7_mid4():
        # 1200 m hole (5 to 12 degrees), four layers, higher heat production, z_min 100 m
        return _t7_Bore(_t7_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                     log_step=8.0, log_top=20.0)

    def _t7_short3():
        # 900 m hole (15 to 22 degrees), three layers, z_min 80 m
        return _t7_Bore(_t7_g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
                     log_step=6.0, log_top=10.0)

    reg = _t7_Region([_t7_incl7(10.0), _t7_mid4(), _t7_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0, [0.0, -1.0, 0.25],
                  [1.0e5, 1.2e5], [0.004, 0.012, 0.025], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
    # the 95 % interval of this region is 11147.9-11871.7 years: bounds 11000-12000 contain it
    r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (11000.0, 12000.0))
    _t7_check(r, reg, "bounds 11000-12000")
    # the profile has a single maximum at 11500 years: it increases towards t_hi = 11000 in the first pair of bounds
    # and decreases from t_lo = 12000 in the second, so the maximum over the closed interval is an end point
    for bounds in ((3000.0, 11000.0), (12000.0, 60000.0)):
        try:
            regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], bounds)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected for bounds %r" % (bounds,))

    # near-boundary valid controls of the end-point rule: precise logs (0.4, 1.2 and 2.5 mK) in the same three holes give
    # a profile with a single maximum at 11500 years (dense scans over both bound pairs, source.md) and the interval
    # 11463.8-11536.2 years. With t_lo = 11445 (l_p(t_lo) = l_max - 4.39) or t_hi = 11555 (l_p(t_hi) = l_max - 4.40) the
    # maximum over the closed interval is still interior, 55 years from the bound, and both crossings lie inside the
    # bounds: no ValueError
    regp = _t7_Region(reg.bores, [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5],
                   [0.0004, 0.0012, 0.0025], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
    for bounds in ((11445.0, 60000.0), (3000.0, 11555.0)):
        r = regional_deglaciation_inversion(regp.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], bounds)
        _t7_check(r, regp, "precise logs, bounds %r" % (bounds,))


_crown_case()

# --- test case 5 ---
# likelihood-ratio interval not contained in [t_lo, t_hi] raises
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    def _t7_contrast5():
        # 1000 m hole (45 to 62 degrees from the collar), five strongly contrasted layers, readings every 40 m, z_min 40 m
        return _t7_Bore((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
                     ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 40.0, log_step=40.0, log_top=10.0)

    def _t7_deep5():
        # 2400 m hole (8 to 15 degrees), five layers, readings every 10 m, z_min 200 m
        return _t7_Bore(_t7_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                     1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0)

    reg = _t7_Region([_t7_contrast5(), _t7_deep5()], [2.8, 3.6], [0.048, 0.052], 6.0, 9000.0, [0.4, -1.0, -0.35, 0.15],
                  [3.0e4, 7.5e4, 1.3e5], [0.002, 0.02], [(1.0, 1.0), None], 11, 1)
    _t7_T_K, _t7_SHAPE = [3.0e4, 7.5e4, 1.3e5], [0.4, -1.0, -0.35, 0.15]
    # the nearest crossings lie near 8925 and 9080 years: bounds 8900-9100 contain both
    r = regional_deglaciation_inversion(reg.holes, _t7_T_K, _t7_SHAPE, (8900.0, 9100.0))
    _t7_check(r, reg, "bounds 8900-9100")
    # with t_lo = 8950 (maximum still interior) the lower crossing is outside the bounds, with t_hi = 9050 the upper one
    for bounds in ((8950.0, 29900.0), (1500.0, 9050.0)):
        try:
            regional_deglaciation_inversion(reg.holes, _t7_T_K, _t7_SHAPE, bounds)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected for bounds %r" % (bounds,))


_crown_case()

# --- test case 6 ---
# six readings in a hole is the smallest valid hole; invalid inputs raise
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    def _t7_mid4():
        return _t7_Bore(_t7_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                     log_step=8.0, log_top=20.0)

    # a vertical hole with eight readings, of which z_min = 200 m keeps exactly six (z_min = 400 m keeps five), and the
    # four-layer hole of 137 readings
    _t7_bA = _t7_Bore((np.array([0.0, 1300.0]), np.zeros(2), np.zeros(2)), ([0, 350, 700], [2.6, 3.8, 3.0]), 0.8e-6, 2.3e6, 200.0,
               log_md=np.array([60.0, 180.0, 330.0, 520.0, 760.0, 1000.0, 1150.0, 1280.0]))
    reg = _t7_Region([_t7_bA, _t7_mid4()], [6.0, 3.2], [0.045, 0.048], 7.0, 10000.0, [0.0, -1.0, 0.25], [9.0e4, 1.2e5], [0.004, 0.008],
                  [(1.0, -1.0), None], 5, 1)
    assert _t7_bA.n == 6
    _t7_T_K, _t7_SHAPE, _t7_B = [9.0e4, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0)
    r = regional_deglaciation_inversion(reg.holes, _t7_T_K, _t7_SHAPE, _t7_B)
    _t7_check(r, reg, "six readings")

    def _t7_bad(name):
        holes = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in h.items()} for h in reg.holes]
        t_k, shape, bounds = np.array(_t7_T_K), np.array(_t7_SHAPE), _t7_B
        a, b = holes
        if name == "one_hole":
            holes = [a]
        elif name == "holes_not_a_list":
            holes = a
        elif name == "hole_not_a_dict":
            holes = [a, list(b.values())]
        elif name == "missing_key":
            del b["rho_c"]
        elif name == "extra_key":
            b["noise_sd"] = 0.01
        elif name == "five_readings":
            a["z_min"] = 400.0
        elif name == "log_len":
            b["log_temp"] = b["log_temp"][:-1]
        elif name == "log_two_d":
            a["log_md"], a["log_temp"] = a["log_md"][None, :], a["log_temp"][None, :]
        elif name == "temp_nan":
            b["log_temp"][40] = float("nan")
        elif name == "A_negative":
            a["heat_production"] = -1e-6
        elif name == "A_nan":
            b["heat_production"] = float("nan")
        elif name == "rho_c_zero":
            b["rho_c"] = 0.0
        elif name == "k_zero":
            a["layer_k"] = np.array([2.6, 0.0, 3.0])
        elif name == "k_count":
            a["layer_k"] = np.array([2.6, 3.8])
        elif name == "top_beyond_hole":
            a["layer_top_md"] = np.array([0.0, 350.0, 1400.0])
        elif name == "log_beyond_hole":
            a["log_md"] = np.array([60.0, 180.0, 330.0, 520.0, 760.0, 1000.0, 1150.0, 1320.0])
        elif name == "survey_start":
            a["survey_md"] = np.array([5.0, 1300.0])
        elif name == "one_depth":
            a["log_md"] = np.full(8, 760.0)          # every reading at one depth: T0 and q0 of the hole not resolvable
        elif name == "one_interval":
            shape, t_k = np.array([-1.0]), np.array([], float)
        elif name == "times_count":
            t_k = np.array([9.0e4, 1.2e5, 1.5e5])
        elif name == "times_not_increasing":
            t_k = np.array([9.0e4, 9.0e4])
        elif name == "times_nan":
            t_k = np.array([9.0e4, float("nan")])
        elif name == "shape_nan":
            shape = np.array([0.0, -1.0, float("nan")])
        elif name == "times_nonpositive":
            t_k = np.array([-9.0e4, 1.2e5])
        elif name == "equal_first_shapes":
            shape = np.array([-1.0, -1.0, 0.25])
        elif name == "shape_two_d":
            shape = np.array([[0.0, -1.0, 0.25]])
        elif name == "times_two_d":
            t_k = np.array([[9.0e4, 1.2e5]])
        elif name == "bounds_lo_zero":
            bounds = (0.0, 60000.0)
        elif name == "bounds_reversed":
            bounds = (60000.0, 3000.0)
        elif name == "bounds_beyond_glaciation":
            bounds = (3000.0, 95000.0)
        elif name == "bounds_at_glaciation_start":
            bounds = (3000.0, 90000.0)               # t_hi == hist_t_years[0]: the upper limit is strict
        elif name == "bounds_equal":
            bounds = (5000.0, 5000.0)
        elif name == "bounds_nan":
            bounds = (float("nan"), 60000.0)
        elif name == "bounds_three":
            bounds = (3000.0, 30000.0, 60000.0)
        return holes, t_k, shape, bounds

    for name in ["one_hole", "holes_not_a_list", "hole_not_a_dict", "missing_key", "extra_key", "five_readings", "log_len",
                 "log_two_d", "temp_nan", "A_negative", "A_nan", "rho_c_zero", "k_zero", "k_count", "top_beyond_hole",
                 "log_beyond_hole", "survey_start", "one_depth", "one_interval", "times_count", "times_not_increasing",
                 "times_nan", "shape_nan", "times_nonpositive", "equal_first_shapes", "shape_two_d", "times_two_d",
                 "bounds_lo_zero", "bounds_reversed", "bounds_beyond_glaciation", "bounds_at_glaciation_start",
                 "bounds_equal", "bounds_nan", "bounds_three"]:
        try:
            regional_deglaciation_inversion(*_t7_bad(name))
        except ValueError:
            pass
        else:
            raise AssertionError("regional_deglaciation_inversion must raise ValueError (%s)" % name)


_crown_case()

# --- test case 7 ---
# a disturbed hole: the log-likelihood is bimodal in g at the estimate, and its global maximum is the estimate
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t7_YEAR = 365.25 * 86400.0
    _t7_GL_X, _t7_GL_W = np.polynomial.legendre.leggauss(64)
    _t7_BX, _t7_BW = np.polynomial.legendre.leggauss(24)

    def _t7_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _t7_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _t7_g_tangent(inc[i], azi[i]), _t7_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _t7_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_t7_GL_W * tz)
        return total

    def _t7_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _t7_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _t7_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _t7_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _t7_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _t7_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _t7_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _t7_chi2_95():
        """0.95 quantile of chi-square with 1 degree of freedom: the square of the 0.975 quantile of the standard
        normal distribution, by bisection on math.erf."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erf(mid / math.sqrt(2.0)) < 0.95:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) ** 2

    class _t7_Bore:
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, R and S by
        exact layer sums, and the oracle step response of its own column (branch-cut form, or erfc for a uniform
        column)."""

        def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
            self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
            self.tops_md, self.k = (np.asarray(v, float) for v in layers)
            if log_md is None:
                log_md = np.arange(log_top, self.md[-1] + 1e-9, log_step)
            self.log_md = np.asarray(log_md, float)
            self.A, self.rc, self.z_min, self.uniform_k = A, rc, z_min, uniform_k
            self.tops = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_t7_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            self.zk, self.n = self.z[self.keep], int(self.keep.sum())
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.Q2, _ = np.linalg.qr(np.column_stack([np.ones(self.n), self.R]))

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _t7_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _t7_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _t7_YEAR
            return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])

        def history(self, shape, t_known):
            sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
            Uk = [self.U(t) for t in tk]
            Pf = sh[1] * Uk[0]
            for i in range(2, sh.size):
                Pf = Pf + sh[i] * (Uk[i - 1] - Uk[i - 2])
            self.Pfix, self.ds = Pf, sh[0] - sh[1]

        def P(self, t):
            return self.Pfix + self.ds * self.U(t)

        def dP(self, t):
            return self.ds * self.U(t, True)

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution is known exactly.
        On the used readings of hole h, y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h with
        r_h = b_h + c_h: b_h is orthogonal to the four columns [1, R_h, P_h, g dP_h/dt_end] (seeded, rms noise[h]),
        c_h lies in the span of the projections p_h, d_h of P_h and g dP_h/dt_end off [1, R_h] (coefficients
        kappa[h] in units of noise[h] for every hole but hole `solve`), and that hole's c_h is solved in closed form
        so that sum_h (n_h / |r_h|^2) [p_h, d_h]^T r_h = 0. With sigma_h^2 = |r_h|^2 / n_h every score of the
        log-likelihood vanishes at the generating values, while the unweighted sum sum_h [p_h, d_h]^T r_h does not."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, kappa, seed, solve):
            self.bores, self.g, self.t_end = bores, g, t_end
            self.shape, self.t_known = np.asarray(shape, float), np.asarray(t_known, float)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B, Pp, Dd, Ps = [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                J = np.column_stack([np.ones(bo.n), bo.R, P0, D0])
                Q, _ = np.linalg.qr(J)
                w = rng.standard_normal(bo.n)
                b = w - Q @ (Q.T @ w)
                b = b - Q @ (Q.T @ b)
                b *= noise[h] / math.sqrt(np.mean(b * b))
                Q2 = bo.Q2
                p = P0 - Q2 @ (Q2.T @ P0)
                d = D0 - Q2 @ (Q2.T @ D0)
                B.append(b); Pp.append(p); Dd.append(d); Ps.append(P0)
            C = [None] * H
            v = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                c = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                C[h] = c
                rr = float(B[h] @ B[h] + c @ c)
                v += bores[h].n / rr * np.array([Pp[h] @ c, Dd[h] @ c])
            X = np.column_stack([Pp[solve], Dd[solve]])
            G = X.T @ X
            Gi_v = np.linalg.solve(G, v)
            s = float(v @ Gi_v)
            nH, bb = bores[solve].n, float(B[solve] @ B[solve])
            lam = (nH - math.sqrt(nH * nH - 4.0 * s * bb)) / (2.0 * s)
            C[solve] = X @ (-lam * Gi_v)
            self.r = [B[h] + C[h] for h in range(H)]
            self.holes = []
            self.y = []
            for h, bo in enumerate(bores):
                y = T0[h] + q0[h] * bo.R + g * Ps[h] + self.r[h]
                self.y.append(y)
                T = np.empty(bo.z.size)
                T[bo.keep] = y - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [math.sqrt(float(self.r[h] @ self.r[h]) / bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end) at the generating values
            npar = 2 * H + 2
            I = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = g * bo.dP(t_end)
                I += Jf.T @ Jf / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(I))
            sig = np.sqrt(np.diag(np.linalg.inv(I * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1],
                                    "noise_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            self.yp = [y - bo.Q2 @ (bo.Q2.T @ y) for y, bo in zip(self.y, bores)]

        def coeffs(self, t):
            out = []
            for bo, y in zip(self.bores, self.yp):
                p = bo.P(t)
                p = p - bo.Q2 @ (bo.Q2.T @ p)
                out.append((float(p @ p), float(p @ y), float(y @ y), bo.n))
            return out

        def lp(self, t, with_g=False):
            """Profile log-likelihood at t_end = t: the maximum over g of sum_h -(n_h/2)(log(2 pi Q_h(g)/n_h) + 1),
            Q_h(g) = a_h g^2 - 2 b_h g + c_h the residual sum of squares of hole h after its T0_h and q0_h. Every
            stationary point lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            co = self.coeffs(t)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                dv = [df(x) for x in grid]
                for i in range(grid.size - 1):
                    if dv[i] > 0 >= dv[i + 1]:
                        x0, x1 = grid[i], grid[i + 1]
                        for _ in range(100):
                            xm = 0.5 * (x0 + x1)
                            if df(xm) > 0:
                                x0 = xm
                            else:
                                x1 = xm
                        cand.append(0.5 * (x0 + x1))
            vals = [f(x) for x in cand]
            i = int(np.argmax(vals))
            return (vals[i], cand[i]) if with_g else vals[i]

        def crossing(self, direction):
            """Crossing of the profile with the threshold nearest to the generating t_end on one side: march outward
            in small steps, then bisection with Illinois steps."""
            ts, st = self.t_end, self.star["sigma_t_end"]
            a, step = ts, 0.05 * st
            while True:
                b = a + direction * step
                if self.lp(b) <= self.thr:
                    break
                a, step = b, min(step * 1.2, 0.3 * st)
            fa, fb = self.lp(a) - self.thr, self.lp(b) - self.thr
            for _ in range(200):
                c = b - fb * (b - a) / (fb - fa)
                if not (min(a, b) < c < max(a, b)):
                    c = 0.5 * (a + b)
                fc = self.lp(c) - self.thr
                if fc * fb < 0:
                    a, fa = b, fb
                else:
                    fa *= 0.5
                b, fb = c, fc
                if abs(b - a) < 1e-12 * abs(b):
                    break
            return b

    def _t7_check(r, reg, label=""):
        """Every output against the exact values, with the stated accuracies."""
        ex = reg.star
        lo_x, hi_x = reg.crossing(-1), reg.crossing(+1)
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["t_end_low"]) / lo_x - 1.0) < 5e-4, (label, "t_end_low", r["t_end_low"], lo_x)
        assert abs(float(r["t_end_high"]) / hi_x - 1.0) < 5e-4, (label, "t_end_high", r["t_end_high"], hi_x)
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("noise_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        return lo_x, hi_x

    def _t7_incl7(step):
        # inclined NQ hole, 1800 m MD (28 to 40 degrees), seven-layer Abitibi-type column, z_min 150 m
        return _t7_Bore(_t7_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
                     0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

    def _t7_mid4():
        # 1200 m hole (5 to 12 degrees), four layers, higher heat production, z_min 100 m
        return _t7_Bore(_t7_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                     log_step=8.0, log_top=20.0)

    def _t7_deep5():
        # 2400 m hole (8 to 15 degrees), five layers, readings every 10 m, z_min 200 m
        return _t7_Bore(_t7_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                     1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0)

    # Three holes: a precise log (0.3 mK, 163 readings used), an ordinary one (8 mK, 137 readings) and a deep hole (220
    # readings) whose log carries a smooth disturbance along the glacial signature, so that alone it would give g = 7.553 K
    # (the other two 7.801 and 7.820) and its maximum-likelihood noise level is 18.2 mK. At t_end = 11500 years the
    # log-likelihood as a function of g (all other parameters maximized) has two maxima: the global one at the generating
    # g = 7.8 and a second one at g = 7.559, 137.7 lower; the pooled least-squares amplitude 7.588 lies on the side of the
    # second one (the minimum between them is at 7.687), and following that maximum gives g = 7.49 and t_end = 11090 years.
    # Dense scans of the oracle profile over the bounds (source.md) find a single maximum, at the generating values.
    reg = _t7_Region([_t7_incl7(10.0), _t7_mid4(), _t7_deep5()], [4.1, 3.2, 3.6], [0.037, 0.048, 0.052], 7.8, 11500.0,
                     [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.0003, 0.008, 0.003], [None, (0.5, -0.5), (-89.0, 0.0)], 21, 0)
    r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0))
    _t7_check(r, reg, "disturbed hole, bimodal amplitude likelihood")


_crown_case()
