# --- test case 0 ---
# three holes with innovation sd 4, 12, 25 mK and phi 0.35, 0.7, 0.85: every output (and its type) against the exact
# maximum-likelihood solution. Targets: whitened-space data construction (every score of the exact log-likelihood vanishes
# at the generating values), branch-cut oracle for P and dP/dt_end, expected information of the oracle Jacobians,
# chi-square quantile from math.erf; interval ends bracketed by crossings of the oracle profile within a relative 5e-4.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    return md, inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1), azi0 + (azi1 - azi0) * f**1.3

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  def incl7(step):
    return Bore(g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
        0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

  def mid4():
    return Bore(g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
        log_step=8.0, log_top=20.0)

  def short3():
    return Bore(g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
        log_step=6.0, log_top=10.0)
  reg = Region([incl7(10.0), mid4(), short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0,
      [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.004, 0.012, 0.025], [0.35, 0.7, 0.85], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
  r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0))
  keys = ["g", "t_end", "sigma_g", "sigma_t_end", "t_end_low", "t_end_high", "log_likelihood"]
  assert isinstance(r, dict) and set(keys + ["holes"]) <= set(r)
  for key in keys:
    assert type(r[key]) is float, key
  assert isinstance(r["holes"], list) and len(r["holes"]) == 3
  for rh in r["holes"]:
    assert isinstance(rh, dict)
    for key in ("T0", "q0", "sigma_T0", "sigma_q0", "phi", "innovation_sd"):
      assert type(rh[key]) is float, key
    assert type(rh["n_used"]) is int
  check(r, reg)


_crown_case()

# --- test case 1 ---
# a short precise log (24 readings, phi = -0.3) and a long noisy one listed as two logging runs (down, then up): the AR(1)
# sequence is the order of increasing MD. Targets: exact maximum-likelihood construction and oracle as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    return md, inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1), azi0 + (azi1 - azi0) * f**1.3

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  def contrast5():
    return Bore((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
        ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 40.0, log_step=40.0, log_top=10.0)

  def deep5(log_md=None):
    return Bore(g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
        1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0, log_md=log_md)
  md = np.arange(30.0, 2400.0 + 1e-9, 10.0)
  reg = Region([contrast5(), deep5(np.concatenate((md[::2], md[1::2][::-1])))], [2.8, 3.6], [0.048, 0.052], 6.0, 9000.0,
      [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], [0.002, 0.02], [-0.3, 0.6], [None, (0.3, 0.3)], 11, 0)
  r = regional_deglaciation_inversion(reg.holes, [3.0e4, 7.5e4, 1.3e5], [0.4, -1.0, -0.35, 0.15], (1500.0, 29900.0))
  check(r, reg)


_crown_case()

# --- test case 2 ---
# uniform columns (closed-form erfc oracle), the collar reading at z_min = 0 kept (121 readings), wide asymmetric interval.
# Targets: exact maximum-likelihood construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  class UBore(Bore):  # uniform column: erfc and its closed-form time derivative
    def U(self, t, deriv=False):
      kap, tau = self.uk / self.rc, t * YEAR
      if deriv:
        return np.array([v / (2.0 * math.sqrt(math.pi * kap) * tau ** 1.5) * math.exp(-v * v / (4 * kap * tau)) for v in self.zk]) * YEAR
      return np.array([math.erfc(v / (2.0 * math.sqrt(kap * tau))) for v in self.zk])
  bA = UBore((np.array([0.0, 600.0, 1200.0]), np.zeros(3), np.zeros(3)), ([0, 400, 900], [3.1, 3.1, 3.1]), 0.7e-6, 2.2e6, 0.0,
      log_step=10.0, log_top=0.0, uniform_k=3.1)
  bB = UBore((np.array([0.0, 900.0]), np.zeros(2), np.zeros(2)), ([0, 300], [2.4, 2.4]), 1.2e-6, 2.6e6, 50.0,
      log_step=15.0, log_top=5.0, uniform_k=2.4)
  reg = Region([bA, bB], [5.5, 2.9], [0.044, 0.061], 6.2, 10500.0, [0.0, -1.0, 0.3], [9.0e4, 1.15e5], [0.006, 0.015], [0.5, 0.25],
      [None, (-2.0, 1.5)], 99, 0)
  assert bA.n == 121
  r = regional_deglaciation_inversion(reg.holes, [9.0e4, 1.15e5], [0.0, -1.0, 0.3], (2000.0, 80000.0))
  check(r, reg)
  assert r["t_end_high"] - reg.t_end > 1.5 * (reg.t_end - r["t_end_low"])


_crown_case()

# --- test case 3 ---
# short glaciation (11.1-24 ka): second profile maximum near 17.9 ka (about 10 below l_max) behind a valley near 15.6 ka,
# both inside the bounds. Targets: exact maximum-likelihood construction and oracle profile as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    return md, inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1), azi0 + (azi1 - azi0) * f**1.3

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  def incl7(step):
    return Bore(g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
        0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

  def mid4():
    return Bore(g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
        log_step=8.0, log_top=20.0)

  def short3():
    return Bore(g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
        log_step=6.0, log_top=10.0)
  reg = Region([incl7(5.0), mid4(), short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11100.0, [0.0, -1.0, 0.25],
      [2.4e4, 1.2e5], [0.0025, 0.005, 0.010], [0.6, 0.45, 0.75], [None, (1.5, -1.0), (-1.0, 1.5)], 21, 0)
  r = regional_deglaciation_inversion(reg.holes, [2.4e4, 1.2e5], [0.0, -1.0, 0.25], (9000.0, 23900.0))
  check(r, reg)
  assert reg.lp(17910.0) - reg.lp(15620.0) > 5.0 and reg.lp(17910.0) < reg.thr


_crown_case()

# --- test case 4 ---
# maximum at an end point of [t_lo, t_hi] raises; bounds 10700-12400 around the estimate are valid; precise logs (1 mK)
# whose maximum lies 77 years inside either bound are returned. Targets: exact construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    return md, inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1), azi0 + (azi1 - azi0) * f**1.3

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  def incl7(step):
    return Bore(g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
        0.9e-6, 2.4e6, 150.0, log_step=step, log_top=15.0)

  def mid4():
    return Bore(g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
        log_step=8.0, log_top=20.0)

  def short3():
    return Bore(g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
        log_step=6.0, log_top=10.0)
  reg = Region([incl7(10.0), mid4(), short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0,
      [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.004, 0.012, 0.025], [0.35, 0.7, 0.85], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
  r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (10700.0, 12400.0))
  check(r, reg)
  for bounds in ((3000.0, 10700.0), (12400.0, 60000.0)):
    try:
      regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], bounds)
    except ValueError:
      pass
    else:
      raise AssertionError("ValueError expected for bounds %r" % (bounds,))
  regp = Region([incl7(5.0), mid4(), short3()], [4.1, 3.2, 5.0], [0.037, 0.048, 0.031], 7.8, 11500.0,
      [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.001, 0.001, 0.001], [0.1, 0.2, 0.3], [None, (2.0, -1.5), (-1.5, 2.0)], 5, 0)
  for bounds in ((11423.0, 99000.0), (1700.0, 11577.0)):
    r = regional_deglaciation_inversion(regp.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], bounds)
    check(r, regp)


_crown_case()

# --- test case 5 ---
# likelihood-ratio interval not contained in [t_lo, t_hi] raises (crossings near 8862 and 9135 years); bounds 8800-9200
# are valid. Targets: exact maximum-likelihood construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    return md, inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1), azi0 + (azi1 - azi0) * f**1.3

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  def contrast5():
    return Bore((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
        ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 40.0, log_step=40.0, log_top=10.0)

  def deep5(log_md=None):
    return Bore(g_survey(2400, 50, 8, 15, 90, 110, 1.0, 3), ([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
        1.8e-6, 2.5e6, 200.0, log_step=10.0, log_top=30.0, log_md=log_md)
  md = np.arange(30.0, 2400.0 + 1e-9, 10.0)
  reg = Region([contrast5(), deep5(np.concatenate((md[::2], md[1::2][::-1])))], [2.8, 3.6], [0.048, 0.052], 6.0, 9000.0,
      [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], [0.002, 0.02], [-0.3, 0.6], [None, (0.3, 0.3)], 11, 0)
  T_K, SHAPE = [3.0e4, 7.5e4, 1.3e5], [0.4, -1.0, -0.35, 0.15]
  r = regional_deglaciation_inversion(reg.holes, T_K, SHAPE, (8800.0, 9200.0))
  check(r, reg)
  for bounds in ((8900.0, 29900.0), (1500.0, 9100.0)):
    try:
      regional_deglaciation_inversion(reg.holes, T_K, SHAPE, bounds)
    except ValueError:
      pass
    else:
      raise AssertionError("ValueError expected for bounds %r" % (bounds,))


_crown_case()

# --- test case 6 ---
# a hole with exactly eight used readings is valid; seven, and 28 other invalid inputs, raise ValueError.
# Targets: exact maximum-likelihood construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    return md, inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1), azi0 + (azi1 - azi0) * f**1.3

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  def mid4():
    return Bore(g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
        log_step=8.0, log_top=20.0)
  bA = Bore((np.array([0.0, 1300.0]), np.zeros(2), np.zeros(2)), ([0, 350, 700], [2.6, 3.8, 3.0]), 0.8e-6, 2.3e6, 200.0,
      log_md=np.array([60.0, 150.0, 240.0, 330.0, 470.0, 610.0, 760.0, 900.0, 1180.0, 1280.0]))
  reg = Region([bA, mid4()], [6.0, 3.2], [0.045, 0.048], 7.0, 10000.0, [0.0, -1.0, 0.25], [9.0e4, 1.2e5], [0.004, 0.008],
      [0.0, 0.5], [None, (0.3, -0.3)], 5, 0)
  assert bA.n == 8
  T_K, SHAPE, B = [9.0e4, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0)
  r = regional_deglaciation_inversion(reg.holes, T_K, SHAPE, B)
  check(r, reg)

  def bad(name):
    holes = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in h.items()} for h in reg.holes]
    t_k, shape, bounds = np.array(T_K), np.array(SHAPE), B
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
    elif name == "A_negative":
      a["heat_production"] = -1e-6
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
      a["log_md"] = np.full(10, 760.0)
    elif name == "one_interval":
      shape, t_k = np.array([-1.0]), np.array([], float)
    elif name == "times_count":
      t_k = np.array([9.0e4, 1.2e5, 1.5e5])
    elif name == "times_not_increasing":
      t_k = np.array([9.0e4, 9.0e4])
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
      bounds = (3000.0, 90000.0)
    elif name == "bounds_equal":
      bounds = (5000.0, 5000.0)
    elif name == "bounds_three":
      bounds = (3000.0, 30000.0, 60000.0)
    return holes, t_k, shape, bounds
  for name in ["one_hole", "holes_not_a_list", "hole_not_a_dict", "missing_key", "extra_key", "seven_readings", "log_len",
      "log_two_d", "A_negative", "rho_c_zero", "k_zero", "k_count", "top_beyond_hole",
      "log_beyond_hole", "survey_start", "one_depth", "one_interval", "times_count", "times_not_increasing",
      "times_nonpositive", "equal_first_shapes", "shape_two_d", "times_two_d",
      "bounds_lo_zero", "bounds_reversed", "bounds_beyond_glaciation", "bounds_at_glaciation_start",
      "bounds_equal", "bounds_three"]:
    try:
      regional_deglaciation_inversion(*bad(name))
    except ValueError:
      pass
    else:
      raise AssertionError("regional_deglaciation_inversion must raise ValueError (%s)" % name)


_crown_case()

# --- test case 7 ---
# strongly correlated log (phi = 0.9, 37 readings, s = 3 mK): the exact likelihood with the first reading (the conditional
# likelihood differs measurably). Targets: exact maximum-likelihood construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.optimize import brentq, minimize, minimize_scalar
  YEAR = 365.25 * 86400.0
  GL_X, GL_W = np.polynomial.legendre.leggauss(64)
  BX, BW = np.polynomial.legendre.leggauss(24)

  def g_tvd(md, inc_deg, azi_deg, m):  # Gauss-Legendre integral of the down component of the arc tangent
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tan = lambda i: np.array([np.sin(inc[i]) * np.cos(azi[i]), np.sin(inc[i]) * np.sin(azi[i]), np.cos(inc[i])])
    total = 0.0
    for i in range(len(md) - 1):
      a, b = md[i], min(md[i + 1], m)
      if b <= a:
        break
      t1, t2 = tan(i), tan(i + 1)
      beta = np.arccos(np.clip(t1 @ t2, -1, 1))
      s = 0.5 * (b - a) * GL_X + 0.5 * (b + a)
      f = (s - a) / (md[i + 1] - md[i])
      tz = np.full_like(s, t1[2]) if beta < 1e-12 else (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
      total += 0.5 * (b - a) * np.sum(GL_W * tz)
    return total

  def g_RS(z, tops, k):  # exact layer sums
    R = S = 0.0
    for top, bot, kk in zip(tops, list(tops[1:]) + [max(z, tops[-1])], k):
      b = min(max(z, top), bot)
      R += (b - top) / kk
      S += (b - top) * (b + top) / (2 * kk)
    return R, S

  def g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    return md, inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1), azi0 + (azi1 - azi0) * f**1.3

  def bc_U(z, tau_years, tops, k, rc, deriv=False):  # layered step response (or d/dtau per year), branch-cut integral
    z, tops, k = (np.atleast_1d(np.asarray(v, float)) for v in (z, tops, k))
    tau, nl, h = float(tau_years) * YEAR, tops.size, np.diff(tops)
    slow = 1.0 / np.sqrt(k / rc * tau)
    rate = 2.0 * float(np.sum(h * slow[:-1])) + float(np.max(z)) * float(np.max(slow)) + 1.0
    edges = np.linspace(0.0, 7.0, int(math.ceil(7.0 / min(0.25, 2.0 * math.pi / rate / 1.5))) + 1)
    a, b = edges[:-1, None], edges[1:, None]
    w = (0.5 * (b - a) * BX[None, :] + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * BW[None, :]).ravel()
    om = slow[:, None] * w[None, :]
    Fa, Pa, Fb, Pb = np.zeros((4, nl, w.size))
    Fa[0], Pb[0] = 1.0, 1.0
    for j in range(nl - 1):
      c, skw, kws = np.cos(om[j] * h[j]), h[j] * np.sinc(om[j] * h[j] / np.pi) / k[j], k[j] * om[j] * np.sin(om[j] * h[j])
      Fa[j + 1], Pa[j + 1] = Fa[j] * c + Pa[j] * skw, -Fa[j] * kws + Pa[j] * c
      Fb[j + 1], Pb[j + 1] = Fb[j] * c + Pb[j] * skw, -Fb[j] * kws + Pb[j] * c
    kwn = k[-1] * om[-1]
    cc = -(Pa[-1] + 1j * kwn * Fa[-1]) / (Pb[-1] + 1j * kwn * Fb[-1])
    lay = np.searchsorted(tops, z, side="right") - 1
    imf = np.empty((z.size, w.size))
    for j in np.unique(lay):
      sel = np.where(lay == j)[0]
      d, oj = (z[sel] - tops[j])[:, None], om[j][None, :]
      if j == nl - 1:
        imf[sel] = ((Fa[j] + cc * Fb[j])[None, :] * np.exp(-1j * oj * d)).imag
      else:
        imf[sel] = cc.imag[None, :] * (Fb[j][None, :] * np.cos(oj * d) + Pb[j][None, :] * d * np.sinc(oj * d / np.pi) / k[j])
    g = np.exp(-w * w)
    if deriv:
      out = -(2.0 / (math.pi * tau)) * (imf * (wt * g * w)[None, :]).sum(axis=1) * YEAR
    else:
      out = 1.0 + (2.0 / math.pi) * (imf * (wt * g / w)[None, :]).sum(axis=1)
    out[z <= 0] = 0.0 if deriv else 1.0
    return out

  class Bore:  # used readings (TVD >= z_min) in the order of increasing MD
    def __init__(self, survey, layers, A, rc, z_min, log_step=None, log_top=None, log_md=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      self.tops_md, self.k = (np.asarray(v, float) for v in layers)
      self.log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.A, self.rc, self.z_min, self.uk, self.cache = A, rc, z_min, uniform_k, {}
      self.tops, self.z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (self.tops_md, self.log_md))
      self.keep = self.z >= z_min
      used = np.nonzero(self.keep)[0]
      self.idx = used[np.argsort(self.log_md[used], kind="stable")]
      self.zk, self.n = self.z[self.idx], int(self.idx.size)
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(self.k)) for v in self.zk]).T
      self.X = np.column_stack([np.ones(self.n), self.R])

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      if float(t) not in self.cache:
        self.cache[float(t)] = self.Pfix + self.ds * self.U(t)
      return self.cache[float(t)]

  def white(v, phi):  # Prais-Winsten transform L(phi) along axis 0
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0], out[1:] = math.sqrt(1.0 - phi * phi) * v[0], v[1:] - phi * v[:-1]
    return out

  class Region:  # data whose exact maximum-likelihood solution is the generating one
    def __init__(self, bores, T0, q0, g, t_end, shape, t_known, noise, phi, kappa, seed, solve):
      self.bores, self.t_end, H = bores, t_end, len(bores)
      rng = np.random.default_rng(seed)
      B0, Q4s, Pp, Dd, Ps, Ds = [], [], [], [], [], []
      sh = np.asarray(shape, float)
      for h, bo in enumerate(bores):
        Uk = [bo.U(t) for t in t_known]
        bo.Pfix, bo.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
        P0, D0 = bo.P(t_end), g * bo.ds * bo.U(t_end, True)
        xw, pw, dw = white(bo.X, phi[h]), white(P0, phi[h]), white(D0, phi[h])
        Q4 = np.linalg.qr(np.column_stack([xw, pw, dw]))[0]
        w = rng.standard_normal(bo.n)
        for _ in range(2):
          w = w - Q4 @ (Q4.T @ w)
        Q2 = np.linalg.qr(xw)[0]
        B0.append(w / np.linalg.norm(w)); Q4s.append(Q4); Ps.append(P0); Ds.append(D0)
        Pp.append(pw - Q2 @ (Q2.T @ pw)); Dd.append(dw - Q2 @ (Q2.T @ dw))
      rho = [noise[h] * math.sqrt(bo.n) for h, bo in enumerate(bores)]
      C, vsum = [None] * H, np.zeros(2)
      for h in range(H):
        if h != solve:
          e1 = Pp[h] / np.linalg.norm(Pp[h])
          e2 = Dd[h] - (Dd[h] @ e1) * e1
          C[h] = noise[h] * (kappa[h][0] * e1 + kappa[h][1] * e2 / np.linalg.norm(e2))
          vsum += bores[h].n / (rho[h] ** 2 + float(C[h] @ C[h])) * np.array([Pp[h] @ C[h], Dd[h] @ C[h]])
      X = np.column_stack([Pp[solve], Dd[solve]])
      Gi_v = np.linalg.solve(X.T @ X, vsum)
      s, nS, bb = float(vsum @ Gi_v), bores[solve].n, rho[solve] ** 2
      C[solve] = X @ (-((nS - math.sqrt(nS * nS - 4.0 * s * bb)) / (2.0 * s) if s > 0 else 0.0) * Gi_v)
      self.holes, us, self.yc = [], [], []
      for h, bo in enumerate(bores):
        n, p, b = bo.n, phi[h], B0[h]
        Li = np.tril(p ** np.subtract.outer(np.arange(n), np.arange(n)).clip(0).astype(float))
        Li[:, 0] /= math.sqrt(1.0 - p * p)
        M = n * np.eye(n, k=-1) @ Li - p / (1.0 - p * p) * np.eye(n)
        M[0, 0] += n * p / (1.0 - p * p)
        M = M + M.T  # phi score = u^T M u / 2
        for it in range(30):
          w = M @ (rho[h] * b + C[h])
          for _ in range(2):
            w = w - Q4s[h] @ (Q4s[h].T @ w)
            w = w - (b @ w) * b
          v = w / np.linalg.norm(w)
          f = lambda a: (lambda u: 0.5 * float(u @ M @ u))(rho[h] * (math.cos(a) * b + math.sin(a) * v) + C[h])
          grid = np.linspace(-math.pi, math.pi, 1441)
          fv = np.array([f(a) for a in grid])
          roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in np.nonzero(fv[:-1] * fv[1:] <= 0)[0]]
          a = min(roots, key=abs) if roots else float(grid[int(np.argmin(np.abs(fv)))])
          b = math.cos(a) * b + math.sin(a) * v
          b /= np.linalg.norm(b)
          if roots:
            break
        us.append(rho[h] * b + C[h])
        self.yc.append(g * Ps[h] + Li @ us[h])
        T = np.empty(bo.z.size)
        T[bo.idx] = T0[h] + q0[h] * bo.R + self.yc[h] - bo.A * bo.S
        T[~bo.keep] = T0[h] + 1.5 + 0.02 * bo.z[~bo.keep] + 0.3 * rng.standard_normal(int((~bo.keep).sum()))
        self.holes.append(dict(survey_md=bo.md.copy(), survey_inc=bo.inc.copy(), survey_azi=bo.azi.copy(), log_md=bo.log_md.copy(),
            log_temp=T, layer_top_md=bo.tops_md.copy(), layer_k=bo.k.copy(), heat_production=bo.A,
            rho_c=bo.rc, z_min=bo.z_min))
      sd = [float(np.linalg.norm(us[h])) / math.sqrt(bo.n) for h, bo in enumerate(bores)]
      info = np.zeros((2 * H + 2, 2 * H + 2))
      for h, bo in enumerate(bores):
        Jf = np.zeros((bo.n, 2 * H + 2))
        Jf[:, 2 * h], Jf[:, 2 * h + 1], Jf[:, -2], Jf[:, -1] = 1.0, bo.R, Ps[h], Ds[h]
        Jw = white(Jf, phi[h])
        info += Jw.T @ Jw / sd[h] ** 2
      sc = 1.0 / np.sqrt(np.diag(info))
      sig = np.sqrt(np.diag(np.linalg.inv(info * np.outer(sc, sc)))) * sc
      lmax = sum(-0.5 * bo.n * (math.log(2.0 * math.pi * sd[h] ** 2) + 1.0) + 0.5 * math.log(1.0 - phi[h] ** 2)
          for h, bo in enumerate(bores))
      self.star = {"g": g, "t_end": t_end, "sigma_g": sig[-2], "sigma_t_end": sig[-1], "log_likelihood": lmax,
          "holes": [{"T0": T0[h], "q0": q0[h], "sigma_T0": sig[2 * h], "sigma_q0": sig[2 * h + 1], "phi": phi[h],
          "innovation_sd": sd[h], "n_used": bo.n} for h, bo in enumerate(bores)]}
      self.thr = lmax - 0.5 * brentq(lambda x: math.erf(math.sqrt(x / 2.0)) - 0.95, 1.0, 10.0, xtol=1e-14)
      self.sol = {float(t_end): np.array(phi, float)}

    def Fphi(self, t, ph):  # maximum over T0_h, q0_h, s_h and g
      if max(abs(p) for p in ph) >= 1.0:
        return -1e300
      co = []
      for h, bo in enumerate(self.bores):
        Q = np.linalg.qr(white(bo.X, ph[h]))[0]
        yw, pw = white(self.yc[h], ph[h]), white(bo.P(t), ph[h])
        yw, pw = yw - Q @ (Q.T @ yw), pw - Q @ (Q.T @ pw)
        co.append((pw @ pw, pw @ yw, yw @ yw, bo.n))
      a, b, c, n = (np.array(v)[:, None] for v in zip(*co))
      f = lambda g: np.sum(-0.5 * n * (np.log(2.0 * math.pi * (a * g * g - 2.0 * b * g + c) / n) + 1.0), axis=0)
      G = np.linspace(np.min(b / a), np.max(b / a), 4001)
      i = int(np.argmax(f(G)))
      if G[-1] > G[0]:
        G = [minimize_scalar(lambda x: -f(x)[0], bounds=(G[max(i - 1, 0)], G[min(i + 1, 4000)]), method="bounded",
            options={"xatol": 1e-13 * (1.0 + abs(G[i]))}).x, G[i]]
      return max(f(x)[0] for x in G) + sum(0.5 * math.log(1.0 - p * p) for p in ph)

    def lp(self, t):  # profile log-likelihood
      t = float(t)
      if t not in self.sol:
        x0 = self.sol[min(self.sol, key=lambda s: abs(math.log(s / t)))]
        self.sol[t] = minimize(lambda x: -self.Fphi(t, x), x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000}).x
      return self.Fphi(t, self.sol[t])

  def check(r, reg):
    ex = reg.star
    assert isinstance(r, dict) and len(r["holes"]) == len(ex["holes"])
    for key, tol in (("g", 2e-3), ("t_end", 2e-3), ("sigma_g", 1e-2), ("sigma_t_end", 1e-2)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    assert abs(float(r["log_likelihood"]) - ex["log_likelihood"]) < 0.1, (r["log_likelihood"], ex["log_likelihood"])
    for h, (rh, eh) in enumerate(zip(r["holes"], ex["holes"])):
      assert int(round(float(rh["n_used"]))) == eh["n_used"], (h, rh["n_used"])
      for key, tol in (("T0", 2e-3), ("phi", 2e-3)):
        assert abs(float(rh[key]) - eh[key]) < tol, (h, key, rh[key], eh[key])
      for key, tol in (("q0", 5e-4), ("sigma_T0", 1e-2), ("sigma_q0", 1e-2), ("innovation_sd", 3e-3)):
        assert abs(float(rh[key]) / eh[key] - 1.0) < tol, (h, key, rh[key], eh[key])
    lo, hi, ts = float(r["t_end_low"]), float(r["t_end_high"]), reg.t_end
    assert lo < ts < hi, (lo, hi)
    assert reg.lp(lo * (1 + 5e-4)) > reg.thr > reg.lp(lo * (1 - 5e-4)), lo
    assert reg.lp(hi * (1 - 5e-4)) > reg.thr > reg.lp(hi * (1 + 5e-4)), hi
    for k in range(1, 6):
      for end in (lo, hi):
        assert reg.lp(ts + (end - ts) * k / 6.0) > reg.thr, (end, k)

  def short3():
    return Bore(g_survey(900, 20, 15, 22, 200, 215, 0.5, 4), ([0, 90, 400], [3.8, 2.4, 3.1]), 0.5e-6, 2.5e6, 80.0,
        log_step=6.0, log_top=10.0)
  bS = Bore(g_survey(1200, 30, 5, 12, 80, 95, 1.0, 3), ([0, 260, 600, 950], [2.6, 3.4, 2.2, 3.0]), 1.6e-6, 2.2e6, 100.0,
      log_step=30.0, log_top=20.0)
  reg = Region([bS, short3()], [3.2, 5.0], [0.048, 0.031], 7.5, 11000.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5], [0.003, 0.010],
      [0.9, 0.4], [None, (0.3, -0.3)], 31, 0)
  assert bS.n == 37
  r = regional_deglaciation_inversion(reg.holes, [1.0e5, 1.2e5], [0.0, -1.0, 0.25], (3000.0, 60000.0))
  check(r, reg)


_crown_case()
