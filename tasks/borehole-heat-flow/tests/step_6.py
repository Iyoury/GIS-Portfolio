# --- test case 0 ---
# inclined hole in a seven-layer column: every output (and its type) against the exact solution.
# Targets: data built so that the generating T0, q0, g, t_end are the exact least-squares solution (residual orthogonal
# to the Jacobian [1, R, P, g dP/dt_end]); P from the branch-cut oracle; Wald errors from the oracle Jacobian; F quantile
# from the t density; interval ends = the oracle profile's nearest threshold crossings.
def _crown_case():
  import math
  import numpy as np
  from scipy.integrate import quad
  from scipy.optimize import brentq
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
    for top, bot, kk in zip(tops, list(tops[1:]) + [math.inf], k):
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

  def f95(nu):  # F(1, nu) 0.95 quantile: squared Student t quantile from the t density
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    cdf = lambda x: 2.0 * c * quad(lambda u: (1.0 + u * u / nu) ** (-(nu + 1) / 2.0), 0.0, x, epsabs=1e-15, epsrel=1e-14)[0]
    return brentq(lambda x: cdf(x) - 0.95, 0.0, 1e3, xtol=1e-14) ** 2

  class Hole:  # y = T + A S = model + r, r orthogonal to the Jacobian [1, R, P, g dP/dt_end]
    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
        log_md=None, log_step=None, log_top=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      tops_md, k = (np.asarray(v, float) for v in layers)
      log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.tops, z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (tops_md, log_md))
      self.k, self.rc, self.uk, keep = k, rc, uniform_k, z >= z_min
      self.zk, self.n = z[keep], int(keep.sum())
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(k)) for v in self.zk]).T
      sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
      Uk = [self.U(t) for t in tk]
      self.Pfix, self.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
      self.star = np.array([T0, q0, g, t_end], float)
      J = np.column_stack([np.ones(self.n), self.R, self.P(t_end), g * self.ds * self.U(t_end, True)])
      Q = np.linalg.qr(J)[0]
      rng = np.random.default_rng(seed)
      r = rng.standard_normal(self.n)
      for _ in range(2):
        r = r - Q @ (Q.T @ r)
      r *= noise / math.sqrt(np.mean(r * r))
      self.y = T0 + q0 * self.R + g * J[:, 2] + r
      T = np.empty(z.size)
      T[keep] = self.y - A * self.S
      T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
      self.rss_min = float(r @ r)
      self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
      self.thr = self.rss_min * (1.0 + f95(self.n - 4) / (self.n - 4))
      self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
      G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
      res = self.y - G @ np.linalg.lstsq(G, self.y, rcond=None)[0]
      return float(res @ res)

    def crossing(self, sgn):  # crossing nearest to t_end
      a, step, s3 = self.star[3], 0.05 * self.sig[3], self.sig[3]
      while self.rss(a + sgn * step) < self.thr:
        a, step = a + sgn * step, min(step * 1.2, 0.3 * s3)
      return brentq(lambda t: self.rss(t) - self.thr, *sorted((a, a + sgn * step)), xtol=1e-12 * a, rtol=1e-14)

  def check(r, h):
    ex = dict(zip(["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end"], [*h.star, *h.sig]))
    ex.update(t_end_low=h.crossing(-1), t_end_high=h.crossing(+1), rms_residual=math.sqrt(h.rss_min / h.n))
    assert int(round(float(r["n_used"]))) == h.n, (r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
        ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
        ("rms_residual", 1e-3)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    return ex
  h = Hole(g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
      0.9e-6, 2.4e6, 4.1, 0.037, 7.8, 11500.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5], 150.0, 0.005, 7,
      log_step=5.0, log_top=15.0)
  r = deglaciation_heat_flow_inversion(*h.args, (3000.0, 60000.0), 150.0)
  keys = ["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end", "t_end_low", "t_end_high", "rms_residual"]
  assert isinstance(r, dict) and set(keys + ["n_used"]) <= set(r)
  for key in keys:
    assert type(r[key]) is float, key
  assert type(r["n_used"]) is int
  check(r, h)


_crown_case()

# --- test case 1 ---
# short hole with strong conductivity contrasts (38 readings): F(1, 34) threshold, asymmetric profile interval.
# Targets: exact least-squares construction and branch-cut oracle as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.integrate import quad
  from scipy.optimize import brentq
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
    for top, bot, kk in zip(tops, list(tops[1:]) + [math.inf], k):
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

  def f95(nu):  # F(1, nu) 0.95 quantile: squared Student t quantile from the t density
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    cdf = lambda x: 2.0 * c * quad(lambda u: (1.0 + u * u / nu) ** (-(nu + 1) / 2.0), 0.0, x, epsabs=1e-15, epsrel=1e-14)[0]
    return brentq(lambda x: cdf(x) - 0.95, 0.0, 1e3, xtol=1e-14) ** 2

  class Hole:  # y = T + A S = model + r, r orthogonal to the Jacobian [1, R, P, g dP/dt_end]
    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
        log_md=None, log_step=None, log_top=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      tops_md, k = (np.asarray(v, float) for v in layers)
      log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.tops, z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (tops_md, log_md))
      self.k, self.rc, self.uk, keep = k, rc, uniform_k, z >= z_min
      self.zk, self.n = z[keep], int(keep.sum())
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(k)) for v in self.zk]).T
      sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
      Uk = [self.U(t) for t in tk]
      self.Pfix, self.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
      self.star = np.array([T0, q0, g, t_end], float)
      J = np.column_stack([np.ones(self.n), self.R, self.P(t_end), g * self.ds * self.U(t_end, True)])
      Q = np.linalg.qr(J)[0]
      rng = np.random.default_rng(seed)
      r = rng.standard_normal(self.n)
      for _ in range(2):
        r = r - Q @ (Q.T @ r)
      r *= noise / math.sqrt(np.mean(r * r))
      self.y = T0 + q0 * self.R + g * J[:, 2] + r
      T = np.empty(z.size)
      T[keep] = self.y - A * self.S
      T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
      self.rss_min = float(r @ r)
      self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
      self.thr = self.rss_min * (1.0 + f95(self.n - 4) / (self.n - 4))
      self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
      G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
      res = self.y - G @ np.linalg.lstsq(G, self.y, rcond=None)[0]
      return float(res @ res)

    def crossing(self, sgn):  # crossing nearest to t_end
      a, step, s3 = self.star[3], 0.05 * self.sig[3], self.sig[3]
      while self.rss(a + sgn * step) < self.thr:
        a, step = a + sgn * step, min(step * 1.2, 0.3 * s3)
      return brentq(lambda t: self.rss(t) - self.thr, *sorted((a, a + sgn * step)), xtol=1e-12 * a, rtol=1e-14)

  def check(r, h):
    ex = dict(zip(["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end"], [*h.star, *h.sig]))
    ex.update(t_end_low=h.crossing(-1), t_end_high=h.crossing(+1), rms_residual=math.sqrt(h.rss_min / h.n))
    assert int(round(float(r["n_used"]))) == h.n, (r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
        ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
        ("rms_residual", 1e-3)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    return ex
  h = Hole((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
      ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 2.8, 0.048, 6.0, 9000.0,
      [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], 40.0, 0.004, 11, log_step=25.0, log_top=10.0)
  r = deglaciation_heat_flow_inversion(*h.args, (1500.0, 29900.0), 40.0)
  ex = check(r, h)
  assert ex["t_end_high"] - ex["t_end"] > 1.15 * (ex["t_end"] - ex["t_end_low"])


_crown_case()

# --- test case 2 ---
# uniform column on a vertical hole (closed-form erfc oracle); the collar reading at z_min = 0 is used (n_used = 121).
# Targets: exact least-squares construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.integrate import quad
  from scipy.optimize import brentq
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
    for top, bot, kk in zip(tops, list(tops[1:]) + [math.inf], k):
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

  def f95(nu):  # F(1, nu) 0.95 quantile: squared Student t quantile from the t density
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    cdf = lambda x: 2.0 * c * quad(lambda u: (1.0 + u * u / nu) ** (-(nu + 1) / 2.0), 0.0, x, epsabs=1e-15, epsrel=1e-14)[0]
    return brentq(lambda x: cdf(x) - 0.95, 0.0, 1e3, xtol=1e-14) ** 2

  class Hole:  # y = T + A S = model + r, r orthogonal to the Jacobian [1, R, P, g dP/dt_end]
    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
        log_md=None, log_step=None, log_top=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      tops_md, k = (np.asarray(v, float) for v in layers)
      log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.tops, z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (tops_md, log_md))
      self.k, self.rc, self.uk, keep = k, rc, uniform_k, z >= z_min
      self.zk, self.n = z[keep], int(keep.sum())
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(k)) for v in self.zk]).T
      sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
      Uk = [self.U(t) for t in tk]
      self.Pfix, self.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
      self.star = np.array([T0, q0, g, t_end], float)
      J = np.column_stack([np.ones(self.n), self.R, self.P(t_end), g * self.ds * self.U(t_end, True)])
      Q = np.linalg.qr(J)[0]
      rng = np.random.default_rng(seed)
      r = rng.standard_normal(self.n)
      for _ in range(2):
        r = r - Q @ (Q.T @ r)
      r *= noise / math.sqrt(np.mean(r * r))
      self.y = T0 + q0 * self.R + g * J[:, 2] + r
      T = np.empty(z.size)
      T[keep] = self.y - A * self.S
      T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
      self.rss_min = float(r @ r)
      self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
      self.thr = self.rss_min * (1.0 + f95(self.n - 4) / (self.n - 4))
      self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
      G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
      res = self.y - G @ np.linalg.lstsq(G, self.y, rcond=None)[0]
      return float(res @ res)

    def crossing(self, sgn):  # crossing nearest to t_end
      a, step, s3 = self.star[3], 0.05 * self.sig[3], self.sig[3]
      while self.rss(a + sgn * step) < self.thr:
        a, step = a + sgn * step, min(step * 1.2, 0.3 * s3)
      return brentq(lambda t: self.rss(t) - self.thr, *sorted((a, a + sgn * step)), xtol=1e-12 * a, rtol=1e-14)

  def check(r, h):
    ex = dict(zip(["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end"], [*h.star, *h.sig]))
    ex.update(t_end_low=h.crossing(-1), t_end_high=h.crossing(+1), rms_residual=math.sqrt(h.rss_min / h.n))
    assert int(round(float(r["n_used"]))) == h.n, (r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
        ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
        ("rms_residual", 1e-3)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    return ex
  h = Hole((np.array([0.0, 600.0, 1200.0]), np.zeros(3), np.zeros(3)), ([0, 400, 900], [3.1, 3.1, 3.1]),
      0.7e-6, 2.2e6, 5.5, 0.044, 6.2, 10500.0, [0.0, -1.0, 0.3], [9.0e4, 1.15e5], 0.0, 0.006, 99,
      log_step=10.0, log_top=0.0, uniform_k=3.1)
  r = deglaciation_heat_flow_inversion(*h.args, (2000.0, 80000.0), 0.0)
  assert h.n == 121
  check(r, h)


_crown_case()

# --- test case 3 ---
# short glaciation (glaciation 11.5-30 ka): second profile minimum near 23.97 ka below the threshold; the estimate is the
# global minimum and the interval the nearest crossings (about 10.9-12.1 ka). Targets: exact construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.integrate import quad
  from scipy.optimize import brentq
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
    for top, bot, kk in zip(tops, list(tops[1:]) + [math.inf], k):
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

  def f95(nu):  # F(1, nu) 0.95 quantile: squared Student t quantile from the t density
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    cdf = lambda x: 2.0 * c * quad(lambda u: (1.0 + u * u / nu) ** (-(nu + 1) / 2.0), 0.0, x, epsabs=1e-15, epsrel=1e-14)[0]
    return brentq(lambda x: cdf(x) - 0.95, 0.0, 1e3, xtol=1e-14) ** 2

  class Hole:  # y = T + A S = model + r, r orthogonal to the Jacobian [1, R, P, g dP/dt_end]
    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
        log_md=None, log_step=None, log_top=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      tops_md, k = (np.asarray(v, float) for v in layers)
      log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.tops, z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (tops_md, log_md))
      self.k, self.rc, self.uk, keep = k, rc, uniform_k, z >= z_min
      self.zk, self.n = z[keep], int(keep.sum())
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(k)) for v in self.zk]).T
      sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
      Uk = [self.U(t) for t in tk]
      self.Pfix, self.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
      self.star = np.array([T0, q0, g, t_end], float)
      J = np.column_stack([np.ones(self.n), self.R, self.P(t_end), g * self.ds * self.U(t_end, True)])
      Q = np.linalg.qr(J)[0]
      rng = np.random.default_rng(seed)
      r = rng.standard_normal(self.n)
      for _ in range(2):
        r = r - Q @ (Q.T @ r)
      r *= noise / math.sqrt(np.mean(r * r))
      self.y = T0 + q0 * self.R + g * J[:, 2] + r
      T = np.empty(z.size)
      T[keep] = self.y - A * self.S
      T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
      self.rss_min = float(r @ r)
      self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
      self.thr = self.rss_min * (1.0 + f95(self.n - 4) / (self.n - 4))
      self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
      G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
      res = self.y - G @ np.linalg.lstsq(G, self.y, rcond=None)[0]
      return float(res @ res)

    def crossing(self, sgn):  # crossing nearest to t_end
      a, step, s3 = self.star[3], 0.05 * self.sig[3], self.sig[3]
      while self.rss(a + sgn * step) < self.thr:
        a, step = a + sgn * step, min(step * 1.2, 0.3 * s3)
      return brentq(lambda t: self.rss(t) - self.thr, *sorted((a, a + sgn * step)), xtol=1e-12 * a, rtol=1e-14)

  def check(r, h):
    ex = dict(zip(["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end"], [*h.star, *h.sig]))
    ex.update(t_end_low=h.crossing(-1), t_end_high=h.crossing(+1), rms_residual=math.sqrt(h.rss_min / h.n))
    assert int(round(float(r["n_used"]))) == h.n, (r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
        ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
        ("rms_residual", 1e-3)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    return ex
  h = Hole(g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
      0.9e-6, 2.4e6, 4.1, 0.037, 7.8, 11500.0, [0.0, -1.0, 0.25], [3.0e4, 1.2e5], 150.0, 0.005, 21,
      log_step=5.0, log_top=15.0)
  r = deglaciation_heat_flow_inversion(*h.args, (3000.0, 23960.0), 150.0)
  ex = check(r, h)
  assert ex["t_end_high"] < 13000.0
  r = deglaciation_heat_flow_inversion(*h.args, (9000.0, 29900.0), 150.0)
  check(r, h)


_crown_case()

# --- test case 4 ---
# minimum at an end point of [t_lo, t_hi] raises; wide bounds (2500-92000 years) are valid; a precise log (0.5 mK) whose
# minimum lies 55 years inside either bound is returned. Targets: exact construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.integrate import quad
  from scipy.optimize import brentq
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
    for top, bot, kk in zip(tops, list(tops[1:]) + [math.inf], k):
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

  def f95(nu):  # F(1, nu) 0.95 quantile: squared Student t quantile from the t density
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    cdf = lambda x: 2.0 * c * quad(lambda u: (1.0 + u * u / nu) ** (-(nu + 1) / 2.0), 0.0, x, epsabs=1e-15, epsrel=1e-14)[0]
    return brentq(lambda x: cdf(x) - 0.95, 0.0, 1e3, xtol=1e-14) ** 2

  class Hole:  # y = T + A S = model + r, r orthogonal to the Jacobian [1, R, P, g dP/dt_end]
    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
        log_md=None, log_step=None, log_top=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      tops_md, k = (np.asarray(v, float) for v in layers)
      log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.tops, z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (tops_md, log_md))
      self.k, self.rc, self.uk, keep = k, rc, uniform_k, z >= z_min
      self.zk, self.n = z[keep], int(keep.sum())
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(k)) for v in self.zk]).T
      sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
      Uk = [self.U(t) for t in tk]
      self.Pfix, self.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
      self.star = np.array([T0, q0, g, t_end], float)
      J = np.column_stack([np.ones(self.n), self.R, self.P(t_end), g * self.ds * self.U(t_end, True)])
      Q = np.linalg.qr(J)[0]
      rng = np.random.default_rng(seed)
      r = rng.standard_normal(self.n)
      for _ in range(2):
        r = r - Q @ (Q.T @ r)
      r *= noise / math.sqrt(np.mean(r * r))
      self.y = T0 + q0 * self.R + g * J[:, 2] + r
      T = np.empty(z.size)
      T[keep] = self.y - A * self.S
      T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
      self.rss_min = float(r @ r)
      self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
      self.thr = self.rss_min * (1.0 + f95(self.n - 4) / (self.n - 4))
      self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
      G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
      res = self.y - G @ np.linalg.lstsq(G, self.y, rcond=None)[0]
      return float(res @ res)

    def crossing(self, sgn):  # crossing nearest to t_end
      a, step, s3 = self.star[3], 0.05 * self.sig[3], self.sig[3]
      while self.rss(a + sgn * step) < self.thr:
        a, step = a + sgn * step, min(step * 1.2, 0.3 * s3)
      return brentq(lambda t: self.rss(t) - self.thr, *sorted((a, a + sgn * step)), xtol=1e-12 * a, rtol=1e-14)

  def check(r, h):
    ex = dict(zip(["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end"], [*h.star, *h.sig]))
    ex.update(t_end_low=h.crossing(-1), t_end_high=h.crossing(+1), rms_residual=math.sqrt(h.rss_min / h.n))
    assert int(round(float(r["n_used"]))) == h.n, (r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
        ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
        ("rms_residual", 1e-3)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    return ex
  h = Hole(g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
      0.9e-6, 2.4e6, 4.1, 0.037, 7.8, 11500.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5], 150.0, 0.005, 7,
      log_step=5.0, log_top=15.0)
  for bounds in ((3000.0, 9000.0), (14000.0, 60000.0)):
    try:
      deglaciation_heat_flow_inversion(*h.args, bounds, 150.0)
    except ValueError:
      pass
    else:
      raise AssertionError("ValueError expected for bounds %r" % (bounds,))
  r = deglaciation_heat_flow_inversion(*h.args, (2500.0, 92000.0), 150.0)
  check(r, h)
  hp = Hole(g_survey(1800, 25, 28, 40, 330, 350, 1.5, 2), ([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
      0.9e-6, 2.4e6, 4.1, 0.037, 7.8, 11500.0, [0.0, -1.0, 0.25], [1.0e5, 1.2e5], 150.0, 0.0005, 7,
      log_step=5.0, log_top=15.0)
  for bounds in ((11445.0, 60000.0), (3000.0, 11555.0)):
    r = deglaciation_heat_flow_inversion(*hp.args, bounds, 150.0)
    check(r, hp)


_crown_case()

# --- test case 5 ---
# profile interval not contained in [t_lo, t_hi] raises (crossings near 7.73 and 10.73 ka); t_lo = 7000 is valid.
# Targets: exact construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.integrate import quad
  from scipy.optimize import brentq
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
    for top, bot, kk in zip(tops, list(tops[1:]) + [math.inf], k):
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

  def f95(nu):  # F(1, nu) 0.95 quantile: squared Student t quantile from the t density
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    cdf = lambda x: 2.0 * c * quad(lambda u: (1.0 + u * u / nu) ** (-(nu + 1) / 2.0), 0.0, x, epsabs=1e-15, epsrel=1e-14)[0]
    return brentq(lambda x: cdf(x) - 0.95, 0.0, 1e3, xtol=1e-14) ** 2

  class Hole:  # y = T + A S = model + r, r orthogonal to the Jacobian [1, R, P, g dP/dt_end]
    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
        log_md=None, log_step=None, log_top=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      tops_md, k = (np.asarray(v, float) for v in layers)
      log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.tops, z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (tops_md, log_md))
      self.k, self.rc, self.uk, keep = k, rc, uniform_k, z >= z_min
      self.zk, self.n = z[keep], int(keep.sum())
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(k)) for v in self.zk]).T
      sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
      Uk = [self.U(t) for t in tk]
      self.Pfix, self.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
      self.star = np.array([T0, q0, g, t_end], float)
      J = np.column_stack([np.ones(self.n), self.R, self.P(t_end), g * self.ds * self.U(t_end, True)])
      Q = np.linalg.qr(J)[0]
      rng = np.random.default_rng(seed)
      r = rng.standard_normal(self.n)
      for _ in range(2):
        r = r - Q @ (Q.T @ r)
      r *= noise / math.sqrt(np.mean(r * r))
      self.y = T0 + q0 * self.R + g * J[:, 2] + r
      T = np.empty(z.size)
      T[keep] = self.y - A * self.S
      T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
      self.rss_min = float(r @ r)
      self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
      self.thr = self.rss_min * (1.0 + f95(self.n - 4) / (self.n - 4))
      self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
      G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
      res = self.y - G @ np.linalg.lstsq(G, self.y, rcond=None)[0]
      return float(res @ res)

    def crossing(self, sgn):  # crossing nearest to t_end
      a, step, s3 = self.star[3], 0.05 * self.sig[3], self.sig[3]
      while self.rss(a + sgn * step) < self.thr:
        a, step = a + sgn * step, min(step * 1.2, 0.3 * s3)
      return brentq(lambda t: self.rss(t) - self.thr, *sorted((a, a + sgn * step)), xtol=1e-12 * a, rtol=1e-14)

  def check(r, h):
    ex = dict(zip(["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end"], [*h.star, *h.sig]))
    ex.update(t_end_low=h.crossing(-1), t_end_high=h.crossing(+1), rms_residual=math.sqrt(h.rss_min / h.n))
    assert int(round(float(r["n_used"]))) == h.n, (r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
        ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
        ("rms_residual", 1e-3)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    return ex
  h = Hole((np.array([0.0, 300.0, 600.0, 1000.0]), np.array([0.0, 3.0, 7.0, 12.0]), np.array([45.0, 50.0, 58.0, 62.0])),
      ([0, 90, 210, 380, 470], [1.9, 5.2, 2.2, 4.8, 2.6]), 1.1e-6, 2.3e6, 2.8, 0.048, 6.0, 9000.0,
      [0.4, -1.0, -0.35, 0.15], [3.0e4, 7.5e4, 1.3e5], 40.0, 0.004, 11, log_step=25.0, log_top=10.0)
  for bounds in ((8500.0, 29900.0), (1500.0, 10000.0)):
    try:
      deglaciation_heat_flow_inversion(*h.args, bounds, 40.0)
    except ValueError:
      pass
    else:
      raise AssertionError("ValueError expected for bounds %r" % (bounds,))
  r = deglaciation_heat_flow_inversion(*h.args, (7000.0, 29900.0), 40.0)
  check(r, h)


_crown_case()

# --- test case 6 ---
# five used readings (n_used - 4 = 1, F(1, 1) = 161.4, 0.2 mK residual) is the smallest valid inversion; four and the
# other invalid inputs raise ValueError. Targets: exact construction as in case 0.
def _crown_case():
  import math
  import numpy as np
  from scipy.integrate import quad
  from scipy.optimize import brentq
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
    for top, bot, kk in zip(tops, list(tops[1:]) + [math.inf], k):
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

  def f95(nu):  # F(1, nu) 0.95 quantile: squared Student t quantile from the t density
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    cdf = lambda x: 2.0 * c * quad(lambda u: (1.0 + u * u / nu) ** (-(nu + 1) / 2.0), 0.0, x, epsabs=1e-15, epsrel=1e-14)[0]
    return brentq(lambda x: cdf(x) - 0.95, 0.0, 1e3, xtol=1e-14) ** 2

  class Hole:  # y = T + A S = model + r, r orthogonal to the Jacobian [1, R, P, g dP/dt_end]
    def __init__(self, survey, layers, A, rc, T0, q0, g, t_end, shape, t_known, z_min, noise, seed,
        log_md=None, log_step=None, log_top=None, uniform_k=None):
      self.md, self.inc, self.azi = (np.asarray(v, float) for v in survey)
      tops_md, k = (np.asarray(v, float) for v in layers)
      log_md = np.asarray(np.arange(log_top, self.md[-1] + 1e-9, log_step) if log_md is None else log_md, float)
      self.tops, z = (np.array([g_tvd(self.md, self.inc, self.azi, m) for m in v]) for v in (tops_md, log_md))
      self.k, self.rc, self.uk, keep = k, rc, uniform_k, z >= z_min
      self.zk, self.n = z[keep], int(keep.sum())
      self.R, self.S = np.array([g_RS(v, list(self.tops), list(k)) for v in self.zk]).T
      sh, tk = np.asarray(shape, float), np.asarray(t_known, float)
      Uk = [self.U(t) for t in tk]
      self.Pfix, self.ds = sh[1] * Uk[0] + sum(sh[i] * (Uk[i - 1] - Uk[i - 2]) for i in range(2, sh.size)), sh[0] - sh[1]
      self.star = np.array([T0, q0, g, t_end], float)
      J = np.column_stack([np.ones(self.n), self.R, self.P(t_end), g * self.ds * self.U(t_end, True)])
      Q = np.linalg.qr(J)[0]
      rng = np.random.default_rng(seed)
      r = rng.standard_normal(self.n)
      for _ in range(2):
        r = r - Q @ (Q.T @ r)
      r *= noise / math.sqrt(np.mean(r * r))
      self.y = T0 + q0 * self.R + g * J[:, 2] + r
      T = np.empty(z.size)
      T[keep] = self.y - A * self.S
      T[~keep] = T0 + 1.5 + 0.02 * z[~keep] + 0.3 * rng.standard_normal(int((~keep).sum()))
      self.rss_min = float(r @ r)
      self.sig = np.sqrt(np.diag(self.rss_min / (self.n - 4) * np.linalg.inv(J.T @ J)))
      self.thr = self.rss_min * (1.0 + f95(self.n - 4) / (self.n - 4))
      self.args = [self.md, self.inc, self.azi, log_md, T, tops_md, k, A, rc, tk.copy(), sh.copy()]

    def U(self, t, deriv=False):
      return bc_U(self.zk, t, self.tops, self.k, self.rc, deriv)

    def P(self, t):
      return self.Pfix + self.ds * self.U(t)

    def rss(self, t):
      G = np.column_stack([np.ones(self.n), self.R, self.P(t)])
      res = self.y - G @ np.linalg.lstsq(G, self.y, rcond=None)[0]
      return float(res @ res)

    def crossing(self, sgn):  # crossing nearest to t_end
      a, step, s3 = self.star[3], 0.05 * self.sig[3], self.sig[3]
      while self.rss(a + sgn * step) < self.thr:
        a, step = a + sgn * step, min(step * 1.2, 0.3 * s3)
      return brentq(lambda t: self.rss(t) - self.thr, *sorted((a, a + sgn * step)), xtol=1e-12 * a, rtol=1e-14)

  def check(r, h):
    ex = dict(zip(["T0", "q0", "amplitude", "t_end", "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end"], [*h.star, *h.sig]))
    ex.update(t_end_low=h.crossing(-1), t_end_high=h.crossing(+1), rms_residual=math.sqrt(h.rss_min / h.n))
    assert int(round(float(r["n_used"]))) == h.n, (r["n_used"], h.n)
    assert abs(float(r["T0"]) - ex["T0"]) < 2e-3, (r["T0"], ex["T0"])
    for key, tol in (("q0", 5e-4), ("amplitude", 5e-4), ("t_end", 5e-4), ("sigma_T0", 5e-3), ("sigma_q0", 5e-3),
        ("sigma_amplitude", 5e-3), ("sigma_t_end", 5e-3), ("t_end_low", 5e-4), ("t_end_high", 5e-4),
        ("rms_residual", 1e-3)):
      assert abs(float(r[key]) / ex[key] - 1.0) < tol, (key, r[key], ex[key])
    return ex
  h = Hole((np.array([0.0, 1300.0]), np.zeros(2), np.zeros(2)), ([0, 350, 700], [2.6, 3.8, 3.0]),
      0.8e-6, 2.3e6, 6.0, 0.045, 5.5, 10000.0, [0.0, -1.0, 0.25], [9.0e4, 1.2e5], 200.0, 0.0002, 5,
      log_md=np.array([60.0, 180.0, 330.0, 520.0, 760.0, 1000.0, 1280.0]))
  B = (3000.0, 60000.0)
  r = deglaciation_heat_flow_inversion(*h.args, B, 200.0)
  check(r, h)
  ARGS = ["survey_md", "survey_inc", "survey_azi", "log_md", "log_temp", "layer_top_md", "layer_k",
      "heat_production", "rho_c", "hist_t_years", "hist_dT_shape", "t_end_bounds", "z_min"]
  ix = {a: j for j, a in enumerate(ARGS)}

  def bad(name):
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
      d[ix["t_end_bounds"]] = (3000.0, 90000.0)
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
      d[ix["log_md"]] = np.full(7, 760.0)
    return d
  for name in ["four_readings", "log_len", "log_two_d", "temp_nan", "one_interval", "times_count",
      "times_not_increasing", "times_nan", "shape_nan", "times_nonpositive", "equal_first_shapes",
      "bounds_lo_zero", "bounds_reversed", "bounds_beyond_glaciation", "bounds_at_glaciation_start",
      "bounds_equal", "shape_two_d", "times_two_d", "bounds_nan", "bounds_three",
      "A_negative", "A_nan", "rho_c_zero", "k_zero", "k_count", "top_beyond_hole", "log_beyond_hole",
      "survey_start", "one_depth"]:
    try:
      deglaciation_heat_flow_inversion(*bad(name))
    except ValueError:
      pass
    else:
      raise AssertionError("deglaciation_heat_flow_inversion must raise ValueError (%s)" % name)


_crown_case()
