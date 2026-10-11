# Whole-task tests: regional_deglaciation_inversion with every earlier step taken from the candidate solution.
# Regions and exact maximum-likelihood targets are built as in tests/step_7.py (whitened-space construction with the
# AR(1) noise of every hole, branch-cut oracle for the layered columns); the accuracy is the stated one, which allows
# for the stated accuracies of steps 1, 2 and 5.

# --- test case 0 ---
# deep region: the whole chain against the exact maximum-likelihood solution
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _tg_YEAR = 365.25 * 86400.0
    _tg_GL_X, _tg_GL_W = np.polynomial.legendre.leggauss(64)
    _tg_BX, _tg_BW = np.polynomial.legendre.leggauss(24)

    def _tg_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _tg_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _tg_g_tangent(inc[i], azi[i]), _tg_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _tg_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_tg_GL_W * tz)
        return total

    def _tg_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _tg_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _tg_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _tg_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _tg_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _tg_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _tg_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _tg_chi2_95():
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

    class _tg_Bore:
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
            self.tops = np.array([_tg_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_tg_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_tg_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _tg_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _tg_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _tg_YEAR
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

    def _tg_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _tg_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _tg_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _tg_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _tg_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _tg_phi_score_grad(u, phi):
        """Gradient in u of _tg_phi_score (a quadratic form in u)."""
        n = u.size
        r = _tg_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _tg_unwhite_T(su, phi) + 2.0 * phi * r[0] * _tg_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _tg_Region:
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
                xw = _tg_white(bo.X, phi[h])
                pw, dw = _tg_white(P0, phi[h]), _tg_white(D0, phi[h])
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
                    w = _tg_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _tg_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] * fv[i + 1] <= 0:
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
                r = _tg_unwhite(u, phi[h])
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
                Jw = _tg_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _tg_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _tg_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _tg_white(self.yc[h], ph[h]), _tg_white(bo.P(t), ph[h])
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
                    mu = 1e-3 if mu < 1e-300 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _tg_check(r, reg, label="", n_inner=5):
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

    def _tg_deep5(log_md=None):
        # 2400 m hole (8 to 15 degrees), five layers, readings every 10 m from MD 30 m, z_min 200 m
        return _tg_Bore(_tg_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                        1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0, log_md=log_md)

    def _tg_steep5():
        # 1400 m hole collared at 45 degrees and steepening to 62, five layers, z_min 60 m
        return _tg_Bore(_tg_g_survey(1400, 20, 45, 62, 10, 35, 2.0, 8), ([0, 150, 420, 760, 1000], [2.2, 4.6, 2.9, 3.7, 2.5]),
                        0.6e-6, 2.35e6, 60.0, log_step=12.0, log_top=12.0)

    def _tg_short3():
        # 900 m hole (15 to 22 degrees), three layers, z_min 80 m
        return _tg_Bore(_tg_g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
                        log_step=6.0, log_top=10.0)

    # A 2400 m hole with innovation sd 10 mK and lag-one correlation 0.8 (220 readings below 200 m TVD), a steep 1400 m hole
    # with 3.4 mK and 0.3 (109 readings) and a 900 m hole with 18 mK and 0.55 (136 readings); four-interval history with
    # the glaciation from 60 ka to t_end = 9.8 ka.
    reg = _tg_Region([_tg_deep5(), _tg_steep5(), _tg_short3()], [3.6, 2.4, 5.0], [0.052, 0.041, 0.031], 5.0, 9800.0,
                     [0.2, -1.0, -0.45, 0.3], [6.0e4, 9.5e4, 1.3e5], [0.010, 0.003, 0.018], [0.8, 0.3, 0.55], [(0.15, 0.1), None, (-0.15, 0.1)], 4242, 1)
    r = regional_deglaciation_inversion(reg.holes, [6.0e4, 9.5e4, 1.3e5], [0.2, -1.0, -0.45, 0.3], (2000.0, 55000.0))
    _tg_check(r, reg, "deep region")


_crown_case()

# --- test case 1 ---
# two holes: exact solution; a constant offset of one log moves only its T0; reordering the holes permutes them;
# listing a log in another order changes nothing
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _tg_YEAR = 365.25 * 86400.0
    _tg_GL_X, _tg_GL_W = np.polynomial.legendre.leggauss(64)
    _tg_BX, _tg_BW = np.polynomial.legendre.leggauss(24)

    def _tg_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _tg_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _tg_g_tangent(inc[i], azi[i]), _tg_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _tg_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_tg_GL_W * tz)
        return total

    def _tg_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _tg_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _tg_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _tg_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _tg_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _tg_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _tg_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _tg_chi2_95():
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

    class _tg_Bore:
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
            self.tops = np.array([_tg_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_tg_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_tg_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _tg_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _tg_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _tg_YEAR
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

    def _tg_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _tg_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _tg_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _tg_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _tg_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _tg_phi_score_grad(u, phi):
        """Gradient in u of _tg_phi_score (a quadratic form in u)."""
        n = u.size
        r = _tg_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _tg_unwhite_T(su, phi) + 2.0 * phi * r[0] * _tg_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _tg_Region:
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
                xw = _tg_white(bo.X, phi[h])
                pw, dw = _tg_white(P0, phi[h]), _tg_white(D0, phi[h])
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
                    w = _tg_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _tg_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] * fv[i + 1] <= 0:
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
                r = _tg_unwhite(u, phi[h])
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
                Jw = _tg_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _tg_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _tg_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _tg_white(self.yc[h], ph[h]), _tg_white(bo.P(t), ph[h])
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
                    mu = 1e-3 if mu < 1e-300 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _tg_check(r, reg, label="", n_inner=5):
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

    def _tg_contrast5():
        # 1000 m hole (45 to 62 degrees from the collar), five strongly contrasted layers, readings every 40 m, z_min 40 m
        return _tg_Bore((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
                        ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 40.0, log_step=40.0, log_top=10.0)

    def _tg_deep5(log_md=None):
        # 2400 m hole (8 to 15 degrees), five layers, readings every 10 m from MD 30 m, z_min 200 m
        return _tg_Bore(_tg_g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                        1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0, log_md=log_md)

    reg = _tg_Region([_tg_contrast5(), _tg_deep5()], [2.8, 3.6], [0.048, 0.052], 6.0, 9000.0, [0.4, -1.0, -0.35, 0.15],
                     [3.0e4, 7.5e4, 1.3e5], [0.002, 0.02], [0.4, 0.65], [None, (-0.3, 0.3)], 12, 0)
    _tg_T_K, _tg_SHAPE, _tg_B = [3.0e4, 7.5e4, 1.3e5], [0.4, -1.0, -0.35, 0.15], (1500.0, 29900.0)
    r = regional_deglaciation_inversion(reg.holes, _tg_T_K, _tg_SHAPE, _tg_B)
    _tg_check(r, reg, "two holes")
    # adding 0.75 K to every reading of the second hole changes only its T0 (by 0.75 K); listing the holes in reverse order
    # reverses the per-hole results and leaves the shared ones unchanged; listing the readings of the second hole in a
    # random order changes nothing (the AR(1) sequence is the order of increasing MD). Two computed outputs are compared
    # with twice the single-output accuracy.
    holes2 = [dict(h) for h in reg.holes]
    holes2[1]["log_temp"] = holes2[1]["log_temp"] + 0.75
    perm = np.random.default_rng(3).permutation(reg.holes[1]["log_md"].size)
    holes4 = [dict(h) for h in reg.holes]
    holes4[1]["log_md"], holes4[1]["log_temp"] = reg.holes[1]["log_md"][perm], reg.holes[1]["log_temp"][perm]
    r2 = regional_deglaciation_inversion(holes2, _tg_T_K, _tg_SHAPE, _tg_B)
    r3 = regional_deglaciation_inversion(list(reversed(reg.holes)), _tg_T_K, _tg_SHAPE, _tg_B)
    r4 = regional_deglaciation_inversion(holes4, _tg_T_K, _tg_SHAPE, _tg_B)
    for rr, order, shift in ((r2, (0, 1), (0.0, 0.75)), (r3, (1, 0), (0.0, 0.0)), (r4, (0, 1), (0.0, 0.0))):
        for key, tol in (("g", 4e-3), ("t_end", 4e-3), ("sigma_g", 2e-2), ("sigma_t_end", 2e-2), ("t_end_low", 1e-3),
                         ("t_end_high", 1e-3)):
            assert abs(float(rr[key]) / float(r[key]) - 1.0) < tol, (key, rr[key], r[key])
        assert abs(float(rr["log_likelihood"]) - float(r["log_likelihood"])) < 0.2
        for i, j in enumerate(order):
            a, b = rr["holes"][i], r["holes"][j]
            assert int(round(float(a["n_used"]))) == int(round(float(b["n_used"])))
            assert abs(float(a["T0"]) - float(b["T0"]) - shift[j]) < 4e-3, (i, a["T0"], b["T0"])
            assert abs(float(a["phi"]) - float(b["phi"])) < 4e-3, (i, a["phi"], b["phi"])
            for key, tol in (("q0", 1e-3), ("sigma_T0", 2e-2), ("sigma_q0", 2e-2), ("innovation_sd", 6e-3)):
                assert abs(float(a[key]) / float(b[key]) - 1.0) < tol, (i, key, a[key], b[key])


_crown_case()

# --- test case 2 ---
# short glaciation: the whole chain; the statistic falls below the quantile again around a second profile maximum,
# outside the interval
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _tg_YEAR = 365.25 * 86400.0
    _tg_GL_X, _tg_GL_W = np.polynomial.legendre.leggauss(64)
    _tg_BX, _tg_BW = np.polynomial.legendre.leggauss(24)

    def _tg_g_tangent(inc, azi):
        return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])

    def _tg_g_tvd(md, inc_deg, azi_deg, m):
        # TVD by 64-point Gauss-Legendre integration of the down component of the great-circle tangent
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2 = _tg_g_tangent(inc[i], azi[i]), _tg_g_tangent(inc[i + 1], azi[i + 1])
            beta = np.arccos(np.clip(t1 @ t2, -1, 1))
            L = md[i + 1] - md[i]
            s = 0.5 * (b - a) * _tg_GL_X + 0.5 * (b + a)
            f = (s - a) / L
            if beta < 1e-12:
                tz = np.full_like(s, t1[2])
            else:
                tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
            total += 0.5 * (b - a) * np.sum(_tg_GL_W * tz)
        return total

    def _tg_g_RS(z, tops, k):
        # exact layer sums of int dz/k and int z dz/k
        R = S = 0.0
        bots = list(tops[1:]) + [math.inf]
        for top, bot, kk in zip(tops, bots, k):
            b = min(max(z, top), bot)
            R += (b - top) / kk
            S += (b - top) * (b + top) / (2 * kk)
        return R, S

    def _tg_g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
        rng = np.random.default_rng(seed)
        md = np.arange(0, total + 1e-9, step)
        f = md / total
        inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
        azi = azi0 + (azi1 - azi0) * f**1.3
        return md, inc, azi

    def _tg_bc_U(z, tau_years, tops, k, rc, deriv=False):
        """Step response U(z, tau) of the layered column (surface raised by 1 K a time tau ago), or dU/dtau per
        year, from the branch-cut (real-axis) form of the Laplace inversion:
        U = 1 + (2/pi) int_0^inf exp(-w^2) Im F+(z, w^2/tau) / w dw, dU/dtau = -(2/(pi tau)) int exp(-w^2) w Im F+ dw,
        F+ the transform on the upper side of the cut s = u e^{i pi}, where q_j = i w / sqrt(kappa_j tau): real
        (cos, sin) transfer matrices from the surface, the decay condition of the bottom half-space fixing the
        complex surface flux; composite Gauss-Legendre quadrature in w on [0, 7]."""
        z = np.atleast_1d(np.asarray(z, float))
        tops = np.asarray(tops, float)
        k = np.asarray(k, float)
        tau = float(tau_years) * _tg_YEAR
        nl = tops.size
        h = np.diff(tops)
        slow = 1.0 / np.sqrt(k / rc * tau)
        rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
        width = min(0.25, 2.0 * math.pi / rate / 1.5)
        edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / width)) + 1)
        a, b = edges[:-1, None], edges[1:, None]
        w = (0.5 * (b - a) * _tg_BX[None, :] + 0.5 * (a + b)).ravel()
        wt = (0.5 * (b - a) * _tg_BW[None, :]).ravel()
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
            out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * _tg_YEAR
            out[z <= 0] = 0.0
            return out
        out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
        out[z <= 0] = 1.0
        return out

    def _tg_chi2_95():
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

    class _tg_Bore:
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
            self.tops = np.array([_tg_g_tvd(self.md, self.inc, self.azi, m) for m in self.tops_md])
            self.z = np.array([_tg_g_tvd(self.md, self.inc, self.azi, m) for m in self.log_md])
            self.keep = self.z >= z_min
            used = np.nonzero(self.keep)[0]
            self.idx = used[np.argsort(self.log_md[used], kind="stable")]      # used readings by increasing MD
            self.zk, self.n = self.z[self.idx], int(self.idx.size)
            self.cache = {}
            RS = np.array([_tg_g_RS(v, list(self.tops), list(self.k)) for v in self.zk])
            self.R, self.S = RS[:, 0], RS[:, 1]
            self.X = np.column_stack([np.ones(self.n), self.R])

        def U(self, t, deriv=False):
            if self.uniform_k is None:
                return _tg_bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)
            kap, tau = self.uniform_k / self.rc, t * _tg_YEAR      # uniform column: erfc and its time derivative
            if deriv:
                return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau))
                                 for v in self.zk]) * _tg_YEAR
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

    def _tg_white(v, phi):
        """Prais-Winsten transform of the readings (axis 0, by increasing MD) for the stationary AR(1) model: first row
        sqrt(1 - phi^2) v_1, then v_i - phi v_(i-1). For AR(1) errors the transformed errors are independent with
        variance s^2."""
        v = np.asarray(v, float)
        out = np.empty_like(v)
        out[0] = math.sqrt(1.0 - phi * phi) * v[0]
        out[1:] = v[1:] - phi * v[:-1]
        return out

    def _tg_unwhite(u, phi):
        """Inverse transform: r_1 = u_1 / sqrt(1 - phi^2), r_i = u_i + phi r_(i-1)."""
        r = np.empty(u.size)
        r[0] = u[0] / math.sqrt(1.0 - phi * phi)
        for i in range(1, u.size):
            r[i] = u[i] + phi * r[i - 1]
        return r

    def _tg_phi_score(u, phi):
        """n times the derivative in phi of -(n/2) log |L(phi) r|^2 + (1/2) log(1 - phi^2), with r = L(phi*)^-1 u fixed,
        at phi = phi* (multiplied by |u|^2 / n): n (sum_(i>=2) u_i r_(i-1) + phi r_1^2) - |u|^2 phi / (1 - phi^2)."""
        r = _tg_unwhite(u, phi)
        return u.size * (float(u[1:] @ r[:-1]) + phi * r[0] ** 2) - float(u @ u) * phi / (1.0 - phi * phi)

    def _tg_unwhite_T(z, phi):
        """Solve L(phi)^T x = z (the transpose of the Prais-Winsten transform), by back substitution."""
        x = np.empty(z.size)
        x[-1] = z[-1]
        for i in range(z.size - 2, 0, -1):
            x[i] = z[i] + phi * x[i + 1]
        x[0] = (z[0] + phi * x[1]) / math.sqrt(1.0 - phi * phi) if z.size > 1 else z[0] / math.sqrt(1.0 - phi * phi)
        return x

    def _tg_phi_score_grad(u, phi):
        """Gradient in u of _tg_phi_score (a quadratic form in u)."""
        n = u.size
        r = _tg_unwhite(u, phi)
        su = np.zeros(n)
        su[:-1] = u[1:]
        sr = np.zeros(n)
        sr[1:] = r[:-1]
        e1 = np.zeros(n)
        e1[0] = 1.0
        return n * (sr + _tg_unwhite_T(su, phi) + 2.0 * phi * r[0] * _tg_unwhite_T(e1, phi)) - 2.0 * phi / (1.0 - phi * phi) * u

    class _tg_Region:
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
                xw = _tg_white(bo.X, phi[h])
                pw, dw = _tg_white(P0, phi[h]), _tg_white(D0, phi[h])
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
                    w = _tg_phi_score_grad(rho[h] * b + C[h], phi[h])
                    for _ in range(2):
                        w = w - Q4s[h] @ (Q4s[h].T @ w)
                        w = w - (b @ w) * b
                    v = w / np.linalg.norm(w)
                    f = lambda a: _tg_phi_score(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h], phi[h])
                    grid = np.linspace(-math.pi, math.pi, 1441)
                    fv = [f(a) for a in grid]
                    roots = []
                    for i in range(grid.size - 1):
                        if fv[i] * fv[i + 1] <= 0:
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
                r = _tg_unwhite(u, phi[h])
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
                Jw = _tg_white(Jf, phi[h])
                info += Jw.T @ Jw / self.sd[h] ** 2
            sc = 1.0 / np.sqrt(np.diag(info))
            sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
            self.lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * self.sd[h] ** 2) + 1.0)
                            + 0.5 * math.log(1.0 - phi[h] ** 2) for h, bo in enumerate(bores))
            self.star = {"g": g, "t_end": t_end, "sigma_g": float(sig[-2]), "sigma_t_end": float(sig[-1]),
                         "log_likelihood": self.lmax,
                         "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": float(sig[2 * h]), "sigma_q0": float(sig[2 * h + 1]),
                                    "phi": phi[h], "innovation_sd": self.sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
            self.thr = self.lmax - 0.5 * _tg_chi2_95()
            # the generating geotherm removed from y (the profile is invariant to a shift in the span of [1, R])
            self.yc = [self.y[h] - T0[h] - q0[h] * bo.R for h, bo in enumerate(bores)]
            self.sol = {float(t_end): np.array(phi, float)}

        def coeffs(self, t, ph):
            """(a_h, b_h, c_h, n_h) of every hole: Q_h(g) = a_h g^2 - 2 b_h g + c_h is the generalized least-squares
            residual of hole h on [1, R_h] at t_end = t with lag-one correlation ph[h] (Prais-Winsten transform and QR)."""
            out = []
            for h, bo in enumerate(self.bores):
                xw = _tg_white(bo.X, ph[h])
                Q, _ = np.linalg.qr(xw)
                yw, pw = _tg_white(self.yc[h], ph[h]), _tg_white(bo.P(t), ph[h])
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
                    mu = 1e-3 if mu < 1e-300 else mu * 10.0
                if not moved:
                    break
                done = float(np.max(np.abs(xn - x))) < 1e-9
                x, fx = xn, fn
                if done:
                    break
            self.sol[t] = x
            return fx

    def _tg_check(r, reg, label="", n_inner=5):
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

    def _tg_incl7(step):
        # inclined NQ hole, 1800 m MD (28 to 40 degrees), seven-layer Abitibi-type column, z_min 150 m
        return _tg_Bore(_tg_g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
                        0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

    def _tg_mid4():
        # 1200 m hole (5 to 12 degrees), four layers, higher heat production, z_min 100 m
        return _tg_Bore(_tg_g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
                        log_step=8.0, log_top=20.0)

    def _tg_short3():
        # 900 m hole (15 to 22 degrees), three layers, z_min 80 m
        return _tg_Bore(_tg_g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
                        log_step=6.0, log_top=10.0)

    # glaciation from t_end = 11.07 ka to 24 ka; innovation sd 5, 9 and 15 mK, lag-one correlations 0.5, 0.7 and 0.3. The
    # profile log-likelihood has a second local maximum near 18.2 ka, only about 1.25 below l_max, behind a valley near
    # 15.5 ka (about 6.1 below): the likelihood-ratio statistic falls below the 0.95 quantile again in a region around 18 ka,
    # which is not part of the interval (the nearest crossing above t_end lies near 12.5 ka); the bounds contain both maxima
    # and a local search over the bounds without a global scan ends in the second basin.
    reg = _tg_Region([_tg_incl7(5.0), _tg_mid4(), _tg_short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.4, 11070.0, [0.0, -1.0, 0.25],
                     [2.4e4, 1.2e5], [0.005, 0.009, 0.015], [0.5, 0.7, 0.3], [None, (1.0, -1.0), (-1.0, 1.0)], 77, 0)
    r = regional_deglaciation_inversion(reg.holes, [2.4e4, 1.2e5], [0.0, -1.0, 0.25], (9000.0, 23900.0))
    _tg_check(r, reg, "short glaciation")
    # oracle premise: a second region below the quantile around 18.2 ka, separated from the interval by the valley
    assert reg.lp(18160.0) > reg.thr and reg.lp(15530.0) < reg.thr - 3.0
    assert float(r["t_end_high"]) < 15530.0, r["t_end_high"]      # the nearest crossing, not the far side of the second region


_crown_case()
