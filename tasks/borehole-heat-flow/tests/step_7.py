# Tests for step 7: regional_deglaciation_inversion (regional maximum-likelihood inversion of several borehole logs
# that share one glacial history: amplitude g and end of the glaciation t_end common to all holes, a steady geotherm
# T0_h, q0_h and stationary AR(1) noise with lag-one correlation phi_h and innovation standard deviation s_h for each
# hole, along its used readings ordered by increasing MD).
# No expected value comes from the reference solution. Each case builds its boreholes forward with numpy and math only
# (TVD by 64-point Gauss-Legendre integration of the great-circle tangent, R and S by exact layer sums) and constructs
# the temperatures so that the exact maximum-likelihood solution is known: on the used readings of hole h (by MD),
# y_h = T + A S = T0_h + q0_h R_h + g P_h(t_end) + r_h, and the residual is built in whitened space, u_h = L(phi_h) r_h
# with L the Prais-Winsten transform (first row sqrt(1 - phi^2), then r_i - phi r_(i-1)): u_h = b_h + c_h, b_h
# orthogonal to the four whitened Jacobian columns L[1, R_h, P_h, g dP_h/dt_end] with |b_h|^2 = n_h noise_h^2, c_h in
# the span of the projections of L P_h and L g dP_h/dt_end off L[1, R_h] (one hole's c_h solved in closed form so that
# the weighted shared score sum_h (n_h / |u_h|^2) [., .]^T u_h vanishes), and b_h turned on its sphere until the score
# of phi_h vanishes. With s_h^2 = |u_h|^2 / n_h every score of the exact log-likelihood (first reading included) is zero
# at the generating values; dense scans of the oracle profile (source.md) show that they are the global maximum over the
# bounds. P and dP/dt_end come from an independent oracle, the branch-cut (real-axis) integral of the Laplace solution
# of each layered column, or from erfc for a uniform column. The standard errors come from the expected Fisher
# information of the oracle Jacobians with the AR(1) covariance at phi_h, s_h; the chi-square quantile from math.erf;
# the interval ends are checked against the oracle profile log-likelihood (maximum over g by sign changes of its
# derivative and bisection, over the phi_h by damped Newton steps): a threshold crossing lies within a relative 5e-4 of
# each returned end and the profile stays above the threshold between the ends and t_end. Accuracy as stated: g and
# t_end relative 2e-3; q0 relative 5e-4; T0 2e-3 K; standard errors relative 1e-2; interval ends relative 5e-4; phi
# 2e-3 (absolute); innovation_sd relative 3e-3; log_likelihood 0.1; n_used exact.

# --- test case 0 ---
# three holes with different noise levels and lag-one correlations: every output against the exact maximum-likelihood
# solution
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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

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

    # Three holes with innovation standard deviations of 4, 12 and 25 mK and lag-one correlations 0.35, 0.7 and 0.85 (163,
    # 137 and 136 readings used), each with its own column, heat production, rho_c, z_min, T0 and q0; glacial history
    # (0, -1, +0.25) with the glaciation from t_end to 100 ka.
    reg = _t7_Region([_t7_incl7(10.0), _t7_mid4(), _t7_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0,
                     [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.004, 0.012, 0.025], [0.35, 0.7, 0.85], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
    r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0))

    keys = ["g", "t_end", "sigma_g", "sigma_t_end", "t_end_low", "t_end_high", "log_likelihood"]
    assert isinstance(r, dict) and set(keys + ["holes"]) <= set(r)
    for key in keys:
        assert type(r[key]) is float, key           # plain Python floats as stated
    assert isinstance(r["holes"], list) and len(r["holes"]) == 3
    for rh in r["holes"]:
        assert isinstance(rh, dict)
        for key in ("T0", "q0", "sigma_T0", "sigma_q0", "phi", "innovation_sd"):
            assert type(rh[key]) is float, key
        assert type(rh["n_used"]) is int
    _t7_check(r, reg, "three holes")


_crown_case()

# --- test case 1 ---
# a short precise log and a long noisy one, a negative lag-one correlation, and a log listed as two runs
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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

    def _t7_contrast5():
        # 1000 m hole (45 to 62 degrees from the collar), five strongly contrasted layers, readings every 40 m, z_min 40 m
        return _t7_Bore((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
                        ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 40.0, log_step=40.0, log_top=10.0)

    def _t7_deep5(log_md=None):
        # 2400 m hole (8 to 15 degrees), five layers, readings every 10 m from MD 30 m, z_min 200 m
        return _t7_Bore(_t7_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                        1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0, log_md=log_md)

    # A 1000 m hole logged every 40 m (24 readings used, innovation sd 2.4 mK, lag-one correlation -0.3) and a 2400 m hole
    # logged every 10 m (220 used, 20 mK, 0.6); four-interval history (post-glacial +0.4, glaciation -1, stadial -0.35,
    # interglacial +0.15). The log of the deep hole is listed as two runs, the even readings going down and the odd ones
    # coming up: the AR(1) sequence is the order of increasing MD, not the order of the arrays.
    md = np.arange(30.0, 2400.0 + 1e-9, 10.0)
    reg = _t7_Region([_t7_contrast5(), _t7_deep5(np.concatenate((md[::2], md[1::2][::-1])))], [2.8, 3.6], [0.048, 0.052], 6.0, 9000.0,
                     [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], [0.002, 0.02], [-0.3, 0.6], [None, (0.3, 0.3)], 11, 0)
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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

    # Two vertical holes whose layers all have one conductivity (3.1 and 2.4 W m^-1 K^-1) and different rho_c: the layered
    # perturbation is the erfc half-space of each hole's own diffusivity, and the oracle uses math.erfc and its closed-form
    # time derivative. The collar reading (MD 0, TVD 0) of the first hole lies exactly at z_min = 0 and is used (121
    # readings, lag-one correlation 0.5); the second hole has 57 readings below 50 m (0.25).
    bA = _t7_Bore((np.array([0.0, 600.0, 1200.0]), np.zeros(3), np.zeros(3)), ([0, 400, 900], [3.1, 3.1, 3.1]), 0.7e-6, 2.2e6, 0.0,
                  log_step=10.0, log_top=0.0, uniform_k=3.1)
    bB = _t7_Bore((np.array([0.0, 900.0]), np.zeros(2), np.zeros(2)), ([0, 300], [2.4, 2.4]), 1.2e-6, 2.6e6, 50.0,
                  log_step=15.0, log_top=5.0, uniform_k=2.4)
    reg = _t7_Region([bA, bB], [5.5, 2.9], [0.044, 0.061], 6.2, 10500.0, [0.0, -1.0, 0.3], [9.0e4, 1.15e5], [0.006, 0.015], [0.5, 0.25],
                     [None, (-2.0, 1.5)], 99, 0)
    assert bA.n == 121
    r = regional_deglaciation_inversion(reg.holes, [9.0e4, 1.15e5], [0.0, -1.0, 0.3], (2000.0, 80000.0))
    _t7_check(r, reg, "uniform columns")
    assert r["t_end_high"] - reg.t_end > 1.5 * (reg.t_end - r["t_end_low"])      # the interval is asymmetric


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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

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

    # The glaciation lasts from t_end to 24 ka only. The amplitude-timing trade-off gives the profile log-likelihood a second
    # local maximum near 17.6 ka, about 8.3 below l_max, behind a valley near 15.7 ka (about 11.6 below l_max); both lie
    # inside the bounds 9000-23900 years, and a local search over the bounds without a global scan ends in the second basin.
    reg = _t7_Region([_t7_incl7(5.0), _t7_mid4(), _t7_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0, [0.0, -1.0, 0.25],
                     [2.4e4, 1.2e5], [0.0025, 0.005, 0.010], [0.6, 0.45, 0.75], [None, (1.5, -1.0), (-1.0, 1.5)], 21, 0)
    r = regional_deglaciation_inversion(reg.holes, [2.4e4, 1.2e5], [0.0, -1.0, 0.25], (9000.0, 23900.0))
    _t7_check(r, reg, "short glaciation")
    assert reg.lp(17640.0) - reg.lp(15730.0) > 2.0 and reg.lp(17640.0) < reg.thr      # oracle premise: second basin


_crown_case()

# --- test case 4 ---
# maximum at an end point of [t_lo, t_hi] raises; bounds around the estimate are valid; a maximum just inside either
# bound (precise logs) is returned
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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

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

    reg = _t7_Region([_t7_incl7(10.0), _t7_mid4(), _t7_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0,
                     [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.004, 0.012, 0.025], [0.35, 0.7, 0.85], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
    # the 95 % interval of this region is about 10866-12178 years: bounds 10700-12400 contain it
    r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (10700.0, 12400.0))
    _t7_check(r, reg, "bounds 10700-12400")
    # the profile has a single maximum at 11500 years: it increases towards t_hi = 10700 in the first pair of bounds and
    # decreases from t_lo = 12400 in the second, so the maximum over the closed interval is an end point
    for bounds in ((3000.0, 10700.0), (12400.0, 60000.0)):
        try:
            regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], bounds)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected for bounds %r" % (bounds,))
    # near-boundary valid controls of the end-point rule: precise logs (innovation sd 1 mK, lag-one correlations 0.1, 0.2
    # and 0.3; 326, 137 and 136 readings) give a profile with a single maximum at 11500 years (dense scans over both pairs
    # of bounds, source.md) and an interval of about 11445-11559 years. With t_lo = 11423 or t_hi = 11577 the maximum over the
    # closed interval is still interior, 77 years from the bound, the profile at the bound lies about 4.4 below l_max and
    # both crossings lie inside the bounds: no ValueError
    regp = _t7_Region([_t7_incl7(5.0), _t7_mid4(), _t7_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0,
                      [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.001, 0.001, 0.001], [0.1, 0.2, 0.3], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
    for bounds in ((11423.0, 99000.0), (1700.0, 11577.0)):
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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

    def _t7_contrast5():
        # 1000 m hole (45 to 62 degrees from the collar), five strongly contrasted layers, readings every 40 m, z_min 40 m
        return _t7_Bore((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
                        ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 40.0, log_step=40.0, log_top=10.0)

    def _t7_deep5(log_md=None):
        # 2400 m hole (8 to 15 degrees), five layers, readings every 10 m from MD 30 m, z_min 200 m
        return _t7_Bore(_t7_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                        1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0, log_md=log_md)

    md = np.arange(30.0, 2400.0 + 1e-9, 10.0)
    reg = _t7_Region([_t7_contrast5(), _t7_deep5(np.concatenate((md[::2], md[1::2][::-1])))], [2.8, 3.6], [0.048, 0.052], 6.0, 9000.0,
                     [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], [0.002, 0.02], [-0.3, 0.6], [None, (0.3, 0.3)], 11, 0)
    _t7_T_K, _t7_SHAPE = [3.0e4, 7.5e4, 1.3e5], [0.4, -1.0, -0.35, 0.15]
    # the nearest crossings lie near 8862 and 9135 years: bounds 8800-9200 contain both
    r = regional_deglaciation_inversion(reg.holes, _t7_T_K, _t7_SHAPE, (8800.0, 9200.0))
    _t7_check(r, reg, "bounds 8800-9200")
    # with t_lo = 8900 (maximum still interior) the lower crossing is outside the bounds, with t_hi = 9100 the upper one
    for bounds in ((8900.0, 29900.0), (1500.0, 9100.0)):
        try:
            regional_deglaciation_inversion(reg.holes, _t7_T_K, _t7_SHAPE, bounds)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected for bounds %r" % (bounds,))


_crown_case()

# --- test case 6 ---
# eight readings in a hole is the smallest valid hole; invalid inputs raise
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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

    def _t7_mid4():
        # 1200 m hole (5 to 12 degrees), four layers, higher heat production, z_min 100 m
        return _t7_Bore(_t7_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                        log_step=8.0, log_top=20.0)

    # a vertical hole with ten readings, of which z_min = 200 m keeps exactly eight (z_min = 300 m keeps seven), and the
    # four-layer hole of 137 readings
    bA = _t7_Bore((np.array([0.0, 1300.0]), np.zeros(2), np.zeros(2)), ([0, 350, 700], [2.6, 3.8, 3.0]), 0.8e-6, 2.3e6, 200.0,
                  log_md=np.array([60.0, 150.0, 240.0, 330.0, 470.0, 610.0, 760.0, 900.0, 1180.0, 1280.0]))
    reg = _t7_Region([bA, _t7_mid4()], [6.0, 3.2], [0.045, 0.048], 7.0, 10000.0, [0.0, -1.0, 0.25], [9.0e4, 1.2e5], [0.004, 0.008],
                     [0.0, 0.5], [None, (0.3, -0.3)], 5, 0)
    assert bA.n == 8
    _t7_T_K, _t7_SHAPE, _t7_B = [9.0e4, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0)
    r = regional_deglaciation_inversion(reg.holes, _t7_T_K, _t7_SHAPE, _t7_B)
    _t7_check(r, reg, "eight readings")

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
        elif name == "seven_readings":
            a["z_min"] = 300.0
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
            a["log_md"] = np.array([60.0, 150.0, 240.0, 330.0, 470.0, 610.0, 760.0, 900.0, 1180.0, 1320.0])
        elif name == "survey_start":
            a["survey_md"] = np.array([5.0, 1300.0])
        elif name == "one_depth":
            a["log_md"] = np.full(10, 760.0)        # every reading at one depth: T0 and q0 of the hole not resolvable
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

    for name in ["one_hole", "holes_not_a_list", "hole_not_a_dict", "missing_key", "extra_key", "seven_readings", "log_len",
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
# a strongly correlated log (lag-one correlation 0.9): the exact likelihood with the first reading
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
        """One borehole built forward: TVD of the readings and the layer tops, the inclusive z_min cut, the used
        readings in the order of increasing MD (the order of the AR(1) noise), R and S by exact layer sums, and the
        oracle step response of its own column (branch-cut form, or erfc for a uniform column)."""

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
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_t7_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

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
            t = float(t)
            if t not in self.cache:
                self.cache[t] = self.Pfix + self.ds * self.U(t)
            return self.cache[t]

        def dP(self, t):
            return self.ds * self.U(t, True)

    def _t7_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _t7_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _t7_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _t7_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _t7_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _t7_phi_score_grad(u, phi):
        """Gradient in u of _t7_phi_score (a quadratic form in u)."""
        n = u.size
        r = _t7_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _t7_unwhite_T(su, phi) + 2.0 * phi * r[0] * _t7_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _t7_Region:
        """Boreholes sharing one glacial history, with data whose maximum-likelihood solution under stationary AR(1)
        noise is known exactly. In hole h (used readings by increasing MD) the residual r_h = y_h - T0_h - q0_h R_h -
        g P_h(t_end), y = T + A S, is built in whitened space, u_h = L(phi_h) r_h (L the Prais-Winsten transform):
        u_h = b_h + c_h with b_h orthogonal to the four whitened columns L[1, R_h, P_h, g dP_h/dt_end], |b_h|^2 =
        n_h noise_h^2, and c_h in the span of the projections p_h, d_h of L P_h and L g dP_h/dt_end off L[1, R_h]
        (coefficients kappa[h] in units of noise_h for every hole but hole `solve`, whose c_h is solved in closed form
        so that sum_h (n_h / |u_h|^2) [p_h, d_h]^T u_h = 0). b_h = rho_h (cos a b0 + sin a v) with b0, v orthonormal
        (seeded) is rotated in its plane until the score of phi_h vanishes (|u_h| unchanged). With s_h^2 = |u_h|^2 / n_h
        every score of the exact log-likelihood vanishes at the generating values."""

        def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
            self.bores, self.g, self.t_end, self.phi = bores, g, t_end, list(phi)
            H = len(bores)
            rng = np.random.default_rng(seed)
            B0, Q4s, Pp, Dd, Ps, Ds, Xw = [], [], [], [], [], [], []
            for h, bo in enumerate(bores):
                bo.history(shape, t_known)
                P0, D0 = bo.P(t_end), g * bo.dP(t_end)
                xw = _t7_white(bo.X, phi[h])
                pw, dw = _t7_white(P0, phi[h]), _t7_white(D0, phi[h])
                Q4, _ = np.linalg.qr(np.column_stack([xw, pw, dw]))
                w = rng.standard_normal(bo.n)
                for _ in range(2):
                    w = w - Q4 @ (Q4.T @ w)
                Q2, _ = np.linalg.qr(xw)
                B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Xw.append(xw)
                Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw)); Ps.append(P0); Ds.append(D0)
            rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
            C = [None] * H
            vsum = np.zeros(2)
            for h in range(H):
                if h == solve:
                    continue
                e1 = Pp[h] / np.linalg.norm(Pp[h])
                e2 = Dd[h] - (Dd[h] @ e1) * e1
                e2 /= np.linalg.norm(e2)
                C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2)
                vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
            X = np.column_stack([Pp[solve], Dd[solve]])
            Gi_v = np.linalg.solve(X.T @ X, vsum)
            s = float(vsum @ Gi_v)
            nS, bb = bores[solve].n, rho[solve] ** 2
            lam = (nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0
            C[solve] = X @ (-lam * Gi_v)
            self.u, self.r, self.y, self.angle = [], [], [], []
            for h, bo in enumerate(bores):
                # b_h = rho_h b with |b| = 1 orthogonal to the four whitened columns: starting from the seeded b0, move b on
                # the great circle through b and the (projected) gradient of the phi score until the score vanishes (the
                # root nearest to b on that circle; if the circle has none, b moves to its point of smallest |score|)
                b = B0[h]
                for it in range(30):
                    w = _t7_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _t7_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] == 0.0 or fv[i] * fv[i + 1] < 0:
                            a0, a1, f0 = grid[i], grid[i + 1], fv[i]
                            for _ in range(80):
                                am = 0.5 * (a0 + a1)
                                fm = f(am)
                                if (fm < 0) == (f0 < 0):
                                    a0, f0 = am, fm
                                else:
                                    a1 = am
                            roots.append(0.5 * (a0 + a1))
                    a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
                    b = math.cos(a) * b + math.sin(a) * v
                    b /= np.linalg.norm(b)
                    if roots:
                        break
                self.angle.append(float(b @ B0[h]))
                u = rho[h] * b + C[h]
                r = _t7_unwhite(u, phi[h])
                self.u.append(u); self.r.append(r)
                self.y.append(T0[h] + q0[h] * bo.R + g * Ps[h] + r)
            self.holes = []
            for h, bo in enumerate(bores):
                T = np.empty(bo.z.size)
                T[bo.idx] = self.y[h] - bo.A * bo.S
                nk = int((~bo.keep).sum())
                T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(nk)
                self.holes.append({"survey_md": bo.md.copy(), "survey_inc": bo.inc.copy(), "survey_azi": bo.azi.copy(),
                                   "log_md": bo.log_md.copy(), "log_temp": T, "layer_top_md": bo.tops_md.copy(),
                                   "layer_k": bo.k.copy(), "heat_production": bo.A, "rho_c": bo.rc, "z_min": bo.z_min})
            self.sd = [float(np.linalg.norm(self.u[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
            # expected Fisher information of (T0_1, q0_1, ..., T0_H, q0_H, g, t_end): sum_h J_h^T V_h^-1 J_h / s_h^2,
            # V_h^-1 = L_h^T L_h, with the oracle Jacobians at the generating values
            npar = 2 * H + 2
            info = np.zeros((npar, npar))
            for h, bo in enumerate(bores):
                Jf = np.zeros((bo.n, npar))
                Jf[:, 2 * h] = 1.0
                Jf[:, 2 * h + 1] = bo.R
                Jf[:, -2] = Ps[h]
                Jf[:, -1] = Ds[h]
                Jw = _t7_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _t7_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _t7_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _t7_white(self.yc[h], ph[h]), _t7_white(bo.P(t), ph[h])
                yw = yw - Q @ (Q.T @ yw)
                pw = pw - Q @ (Q.T @ pw)
                out.append((float(pw @ pw), float(pw @ yw), float(yw @ yw), bo.n))
            return out

        def Fphi(self, t, ph, with_g=False):
            """Log-likelihood maximized over T0_h, q0_h, s_h and g at t_end = t with the lag-one correlations ph: every
            stationary point in g lies between the smallest and the largest b_h / a_h; they are bracketed by the sign
            changes of the derivative on a 4001-point grid and located by bisection."""
            if not all(abs(p) < 1.0 for p in ph):
                return (-math.inf, None) if with_g else -math.inf
            co = self.coeffs(t, ph)
            f = lambda g: sum(-0.5 * n * (math.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0) for a, b, c, n in co)
            df = lambda g: -sum(n * (a * g - b) / (a * g * g - 2.0 * b * g + c) for a, b, c, n in co)
            gs = [b / a for a, b, c, n in co]
            lo, hi = min(gs), max(gs)
            cand = [lo, hi]
            if hi > lo:
                grid = np.linspace(lo, hi, 4001)
                A = np.array([[a, b, c, n] for a, b, c, n in co])
                dv = -np.sum(A[:, 3:4] * (A[:, 0:1] * grid - A[:, 1:2]) / (A[:, 0:1] * grid * grid - 2.0 * A[:, 1:2] * grid + A[:, 2:3]), axis=0)
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
            k = int(np.argmax(vals))
            v = vals[k] + sum(0.5 * math.log(1.0 - p * p) for p in ph)
            return (v, cand[k] + 0.0) if with_g else v

        def lp(self, t, x0=None):
            """Profile log-likelihood at t_end = t: Fphi maximized over the lag-one correlations by a damped Newton
            ascent (central-difference derivatives, Levenberg damping), started from the correlations found at the
            nearest t already evaluated (the generating values first)."""
            t = float(t)
            if t in self.sol:
                return self.Fphi(t, self.sol[t])
            if x0 is None:
                x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
            x = np.array(x0, float)
            m, hs = x.size, 1e-4
            fx = self.Fphi(t, x)
            for _ in range(100):
                grad = np.zeros(m)
                hess = np.zeros((m, m))
                for i in range(m):
                    e = np.zeros(m); e[i] = hs
                    fp, fm = self.Fphi(t, x + e), self.Fphi(t, x - e)
                    grad[i] = (fp - fm) / (2 * hs)
                    hess[i, i] = (fp - 2 * fx + fm) / hs ** 2
                    for j in range(i):
                        ej = np.zeros(m); ej[j] = hs
                        hess[i, j] = hess[j, i] = (self.Fphi(t, x + e + ej) - self.Fphi(t, x + e - ej) - self.Fphi(t, x - e + ej)
                                                   + self.Fphi(t, x - e - ej)) / (4 * hs * hs)
                if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                    break
                mu = 0.0
                moved = False
                while mu < 1e12:
                    Hd = hess - mu * np.diag(np.abs(np.diag(hess)) + 1.0)
                    if np.max(np.linalg.eigvalsh(Hd)) < 0:
                        step = -np.linalg.solve(Hd, grad)
                        step *= min(1.0, 0.2 / max(1e-300, float(np.max(np.abs(step)))))
                        xn = x + step
                        fn = self.Fphi(t, xn)
                        if fn >= fx:
                            moved = True
                            break
                    mu = 1e-3 if mu == 0.0 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _t7_check(r, reg, label="", n_inner=5):
        """Every output against the exact values with the stated accuracies; the interval ends against the oracle
        profile: a crossing of the threshold l_max - chi2_0.95(1)/2 lies within a relative 5e-4 of each returned end
        (the profile is above the threshold at the end moved 5e-4 towards t_end and below it at the end moved 5e-4
        away), and the profile stays above the threshold at interior points between the ends and t_end."""
        ex = reg.star
        assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"]), label
        for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
            assert abs(float(r[key]) / ex[key] - 1.0) < tol, (label, key, r[key], ex[key])
        assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (label, "log_likelihood", r["log_likelihood"], ex["log_likelihood"])
        for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
            assert int(round(float(rh["n_used"]))) == eh["n_used"], (label, h, "n_used", rh["n_used"], eh["n_used"])
            assert abs(float(rh["T0"]) - eh["T0"]) < 2e-3, (label, h, "T0", rh["T0"], eh["T0"])
            assert abs(float(rh["phi"]) - eh["phi"]) < 2e-3, (label, h, "phi", rh["phi"], eh["phi"])
            for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
                assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (label, h, key, rh[key], eh[key])
        lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
        assert lo < ts < hi, (label, lo, hi)
        assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), (label, "t_end_low", lo)
        assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), (label, "t_end_high", hi)
        for k in range(1, n_inner + 1):
            for end in (lo, hi):
                tk = ts + (end - ts) * k / (n_inner + 1.0)
                assert reg.lp(tk) > reg.thr, (label, "profile below the threshold inside the interval", tk)

    def _t7_short3():
        # 900 m hole (15 to 22 degrees), three layers, z_min 80 m
        return _t7_Bore(_t7_g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
                        log_step=6.0, log_top=10.0)

    # A 1200 m hole logged every 30 m (37 readings used below 100 m TVD) whose noise has a lag-one correlation of 0.9
    # (innovation sd 3 mK, marginal sd 6.9 mK), and the 900 m hole with 10 mK and 0.4. With so strong a correlation the
    # first reading carries a large part of the information on the level of the log: its density, with the variance
    # s^2 / (1 - phi^2), is part of the exact likelihood, and the estimates differ measurably from those of a likelihood
    # conditional on it.
    bS = _t7_Bore(_t7_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                  log_step=30.0, log_top=20.0)
    reg = _t7_Region([bS, _t7_short3()], [3.2, 5.0], [0.048, 0.031], 7.5, 11000.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.003, 0.010],
                     [0.9, 0.4], [None, (0.3, -0.3)], 31, 0)
    assert bS.n == 37
    r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0))
    _t7_check(r, reg, "strongly correlated log")


_crown_case()
