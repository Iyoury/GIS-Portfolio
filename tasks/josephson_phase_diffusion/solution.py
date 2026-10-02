import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def _s1_nodes(i, theta):
    # Composite 10-point Gauss-Legendre rule on [0, 2 pi] in the shift y. Panels are
    # shorter than the width sqrt(theta) of the Boltzmann peaks and are graded
    # geometrically toward y = 0, where the tilt factor exp(-i y / theta) varies on the
    # scale theta / i.
    gx, gw = np.polynomial.legendre.leggauss(10)
    h_far = min(0.25, 0.35 * np.sqrt(theta))
    s = theta / i if i > 0.0 else np.inf
    bps = [0.0]
    if s < h_far:
        y = 0.25 * s
        while y < h_far:
            bps.append(y)
            y *= 2.0
    n = int(np.ceil((2.0 * np.pi - bps[-1]) / h_far))
    bps = np.concatenate([bps, np.linspace(bps[-1], 2.0 * np.pi, n + 1)[1:]])
    a, b = bps[:-1], bps[1:]
    y = (0.5 * (b - a)[:, None] * gx[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * gw[None, :]).ravel()
    return y, w


def mean_voltage(i, theta):
    '''Stationary dc voltage and differential resistance of a noisy overdamped junction.

    Inputs:
      i: float, reduced bias current I / I_c, |i| <= 10.
      theta: float, noise strength k_B T / E_J, 0.02 <= theta <= 50.

    Output:
      (v, r_d): tuple of two Python floats.
        v: mean phase velocity lim <phi(tau)> / tau = <V> / (I_c R).
        r_d: differential resistance dv/di (in units of R).
        Relative errors below 1e-9 for v (v = 0 exactly at i = 0) and below 1e-8
        for r_d.

    Raises:
      ValueError if i or theta is not finite, if |i| > 10 or if theta is outside
      [0.02, 50].
    '''
    i = float(i)
    theta = float(theta)
    if not (np.isfinite(i) and np.isfinite(theta)):
        raise ValueError("i and theta must be finite")
    if abs(i) > 10.0 or not (0.02 <= theta <= 50.0):
        raise ValueError("need |i| <= 10 and 0.02 <= theta <= 50")
    sign = 1.0 if i >= 0.0 else -1.0
    a = abs(i)
    # Stratonovich: v = L (1 - exp(-L a / theta)) / int_0^L I_+(x) dx with L = 2 pi and
    # I_+(x) = (1/theta) int_0^L exp([U(x) - U(x - y)] / theta) dy, U = -cos(x) - a x.
    # Trapezoid rule in the periodic variable x, Gauss-Legendre in y. With a >= 0 the
    # exponent is at most 2 / theta <= 100, so no overflow is possible.
    nx = 2 * int(40 + 30 / np.sqrt(theta))
    x = 2.0 * np.pi * np.arange(nx) / nx
    y, w = _s1_nodes(a, theta)
    X, Y = x[:, None], y[None, :]
    ex = np.exp((np.cos(X - Y) - np.cos(X) - a * Y) / theta)
    S = 2.0 * np.pi * np.mean(ex @ w) / theta
    dS = -2.0 * np.pi * np.mean((ex * Y) @ w) / theta ** 2      # dS / da
    q = -np.expm1(-2.0 * np.pi * a / theta)                       # 1 - exp(-2 pi a / theta)
    dq = (2.0 * np.pi / theta) * np.exp(-2.0 * np.pi * a / theta)
    v = sign * 2.0 * np.pi * q / S
    r_d = 2.0 * np.pi * (dq * S - q * dS) / S ** 2                # even in i
    result = (float(v), float(r_d))
    return result


def _s2_nodes(i, theta):
    # composite Gauss-Legendre in the shift y, graded toward y = 0 (scale theta / i)
    gx, gw = np.polynomial.legendre.leggauss(10)
    h_far = min(0.25, 0.35 * np.sqrt(theta))
    s = theta / i if i > 0.0 else np.inf
    bps = [0.0]
    if s < h_far:
        y = 0.25 * s
        while y < h_far:
            bps.append(y)
            y *= 2.0
    n = int(np.ceil((2.0 * np.pi - bps[-1]) / h_far))
    bps = np.concatenate([bps, np.linspace(bps[-1], 2.0 * np.pi, n + 1)[1:]])
    a, b = bps[:-1], bps[1:]
    y = (0.5 * (b - a)[:, None] * gx[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * gw[None, :]).ravel()
    return y, w


def effective_diffusion(i, theta):
    '''Effective diffusion coefficient of the junction phase.

    Inputs:
      i: float, reduced bias current I / I_c, |i| <= 10.
      theta: float, noise strength k_B T / E_J, 0.02 <= theta <= 50.

    Output:
      D: Python float, lim [<phi(tau)**2> - <phi(tau)>**2] / (2 tau) in reduced units
         (tau = 2 e I_c R t / hbar). Relative error below 1e-9.

    Raises:
      ValueError if i or theta is not finite, if |i| > 10 or if theta is outside
      [0.02, 50].
    '''
    i = float(i)
    theta = float(theta)
    if not (np.isfinite(i) and np.isfinite(theta)):
        raise ValueError("i and theta must be finite")
    if abs(i) > 10.0 or not (0.02 <= theta <= 50.0):
        raise ValueError("need |i| <= 10 and 0.02 <= theta <= 50")
    a = abs(i)          # D is even in i (mirror symmetry phi -> -phi)
    # Reimann et al. (2001): D = theta <I_+^2 I_-> / <I_+>^3 (averages over one period),
    # I_+(x) = (1/theta) int_0^L exp([U(x) - U(x - y)] / theta) dy,
    # I_-(x) = (1/theta) int_0^L exp([U(x + y) - U(x)] / theta) dy, U = -cos(x) - a x.
    nx = 2 * int(40 + 30 / np.sqrt(theta))
    x = 2.0 * np.pi * np.arange(nx) / nx
    y, w = _s2_nodes(a, theta)
    X, Y = x[:, None], y[None, :]
    Ip = np.exp((np.cos(X - Y) - np.cos(X) - a * Y) / theta) @ w / theta
    Im = np.exp((np.cos(X) - np.cos(X + Y) - a * Y) / theta) @ w / theta
    D = float(theta * np.mean(Ip ** 2 * Im) / np.mean(Ip) ** 3)
    return D


def _s3_nodes(i, theta):
    # composite Gauss-Legendre in the shift y, graded toward y = 0 (scale theta / i)
    gx, gw = np.polynomial.legendre.leggauss(10)
    h_far = min(0.25, 0.35 * np.sqrt(theta))
    s = theta / i if i > 0.0 else np.inf
    bps = [0.0]
    if s < h_far:
        y = 0.25 * s
        while y < h_far:
            bps.append(y)
            y *= 2.0
    n = int(np.ceil((2.0 * np.pi - bps[-1]) / h_far))
    bps = np.concatenate([bps, np.linspace(bps[-1], 2.0 * np.pi, n + 1)[1:]])
    a, b = bps[:-1], bps[1:]
    y = (0.5 * (b - a)[:, None] * gx[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * gw[None, :]).ravel()
    return y, w


def _s3_dlogD(i, theta):
    # d ln D / di from the Reimann formula, differentiating I_+ and I_- under the integral
    nx = 2 * int(40 + 30 / np.sqrt(theta))
    x = 2.0 * np.pi * np.arange(nx) / nx
    y, w = _s3_nodes(i, theta)
    X, Y = x[:, None], y[None, :]
    ep = np.exp((np.cos(X - Y) - np.cos(X) - i * Y) / theta)
    em = np.exp((np.cos(X) - np.cos(X + Y) - i * Y) / theta)
    Ip, Im = ep @ w / theta, em @ w / theta
    dIp, dIm = -(ep * Y) @ w / theta ** 2, -(em * Y) @ w / theta ** 2
    A, B = np.mean(Ip ** 2 * Im), np.mean(Ip)
    dA = np.mean(2.0 * Ip * dIp * Im + Ip ** 2 * dIm)
    return dA / A - 3.0 * np.mean(dIp) / B


def diffusion_peak(theta):
    '''Bias current of maximal phase diffusion (giant diffusion) and the enhancement there.

    Inputs:
      theta: float, noise strength k_B T / E_J, 0.02 <= theta <= 1.

    Output:
      (i_peak, d_peak): tuple of two Python floats.
        i_peak: the bias i > 0 at which effective_diffusion(i, theta) is largest
                (absolute error below 1e-6).
        d_peak: D_eff(i_peak, theta) / theta (relative error below 1e-9).

    Raises:
      ValueError if theta is not finite or is outside [0.02, 1].
    '''
    theta = float(theta)
    if not (np.isfinite(theta) and 0.02 <= theta <= 1.0):
        raise ValueError("theta must be in [0.02, 1]")
    # The maximum is unique and lies in (0.5, 3) for this theta range: solve d ln D / di = 0.
    i_peak = brentq(lambda s: _s3_dlogD(s, theta), 0.5, 3.0, xtol=1e-13, rtol=1e-15)
    d_peak = effective_diffusion(i_peak, theta) / theta
    result = (float(i_peak), float(d_peak))
    return result


def _s4_nodes(i, theta):
    # composite Gauss-Legendre in the shift y, graded toward y = 0 (scale theta / i)
    gx, gw = np.polynomial.legendre.leggauss(10)
    h_far = min(0.25, 0.35 * np.sqrt(theta))
    s = theta / i if i > 0.0 else np.inf
    bps = [0.0]
    if s < h_far:
        y = 0.25 * s
        while y < h_far:
            bps.append(y)
            y *= 2.0
    n = int(np.ceil((2.0 * np.pi - bps[-1]) / h_far))
    bps = np.concatenate([bps, np.linspace(bps[-1], 2.0 * np.pi, n + 1)[1:]])
    a, b = bps[:-1], bps[1:]
    y = (0.5 * (b - a)[:, None] * gx[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * gw[None, :]).ravel()
    return y, w


def _s4_dlogv_dlogtheta(i, theta):
    # d ln v / d ln theta at fixed i > 0 from the Stratonovich formula v = 2 pi q / S,
    # q = 1 - exp(-2 pi i / theta), S = (1/theta) int int exp(E / theta), differentiated
    # under the integral sign (dE/theta / dtheta = -E / theta**2)
    nx = 2 * int(40 + 30 / np.sqrt(theta))
    x = 2.0 * np.pi * np.arange(nx) / nx
    y, w = _s4_nodes(i, theta)
    X, Y = x[:, None], y[None, :]
    E = (np.cos(X - Y) - np.cos(X) - i * Y) / theta
    ex = np.exp(E)
    S = np.mean(ex @ w) / theta
    dS = -np.mean((ex * E) @ w) / theta ** 2 - S / theta
    z = 2.0 * np.pi * i / theta
    dlogq = -(z / theta) * np.exp(-z) / (-np.expm1(-z))
    return theta * (dlogq - dS / S)


def noise_temperature(i, v):
    '''Noise strength theta = k_B T / E_J inferred from a measured dc voltage, and its sensitivity.

    Inputs:
      i: float, reduced bias current, 0 < i <= 10.
      v: float, measured reduced mean voltage <V> / (I_c R) at this bias,
         sqrt(max(i**2 - 1, 0)) < v < i.

    Output:
      (theta, kappa): tuple of two Python floats.
        theta: noise strength in [0.02, 50] with mean_voltage(i, theta)[0] == v,
               relative error below 1e-7 (the endpoints 0.02 and 50 are allowed).
        kappa: thermometric sensitivity d ln(theta) / d ln(v) at fixed i, evaluated at
               theta (a relative voltage error dv/v gives the relative temperature error
               kappa dv/v), relative error below 1e-6.

    Raises:
      ValueError if i or v is not finite, if i is not in (0, 10], if v is not strictly
      between sqrt(max(i**2 - 1, 0)) and i, or if v lies outside the voltages reached for
      theta in [0.02, 50] by more than a relative 1e-11. A v within a relative 1e-11 beyond
      the voltage at theta = 0.02 or theta = 50 returns that endpoint.
    '''
    i = float(i)
    v = float(v)
    if not (np.isfinite(i) and np.isfinite(v)):
        raise ValueError("i and v must be finite")
    if not (0.0 < i <= 10.0):
        raise ValueError("i must be in (0, 10]")
    if not (np.sqrt(max(i * i - 1.0, 0.0)) < v < i):
        raise ValueError("v must lie strictly between the noiseless and the ohmic voltage")
    lv = np.log(v)
    # v rises monotonically with the noise strength (from the noiseless value to i), and
    # ln v is a smooth function of ln theta; ln v is used because v can be exponentially
    # small below the critical current.
    def g(s):
        return np.log(mean_voltage(i, np.exp(s))[0]) - lv

    lo, hi = np.log(0.02), np.log(50.0)
    g_lo, g_hi = g(lo), g(hi)
    tol = 1e-11                       # stated endpoint tolerance on ln v (relative 1e-11)
    if g_lo > tol or g_hi < -tol:
        raise ValueError("no theta in [0.02, 50] reproduces this voltage")
    if g_lo >= 0.0:
        theta = 0.02
    elif g_hi <= 0.0:
        theta = 50.0
    else:
        theta = float(np.exp(brentq(g, lo, hi, xtol=1e-14, rtol=1e-15)))
    kappa = float(1.0 / _s4_dlogv_dlogtheta(i, theta))
    result = (theta, kappa)
    return result


def array_criterion(ics, rs, theta0, v_crit):
    '''Criterion current, differential resistance and voltage noise of a series junction array.

    Inputs:
      ics: 1-D array of critical currents c_k in units of I_0, finite, > 0,
           max(ics) / min(ics) <= 5.
      rs: 1-D array of the same length, normal resistances r_k in units of R_0, finite, > 0.
      theta0: float, k_B T / E_J0 with E_J0 = hbar I_0 / (2 e); every theta0 / c_k must
              lie in [0.02, 50].
      v_crit: float, voltage criterion in units of I_0 R_0,
              0 < v_crit <= 0.5 * sum(c_k * r_k).

    Output:
      (j_star, r_diff, s_v): tuple of three Python floats.
        j_star: bias current (units of I_0) at which the mean array voltage equals v_crit.
        r_diff: differential resistance dV/dJ of the array at j_star (units of R_0).
        s_v: lim_{t->inf} Var(int_0^t V dt') / t at j_star, the zero-frequency (two-sided)
             voltage noise, in units of (hbar / 2e) I_0 R_0.
        Relative errors below 1e-9 (j_star) and 1e-8 (r_diff, s_v).

    Raises:
      ValueError if ics and rs are not 1-D arrays of the same nonzero length, if an entry
      is not finite and positive, if max(ics) / min(ics) > 5, if theta0 is not finite and
      positive or a ratio theta0 / c_k is outside [0.02, 50], or if v_crit is not in
      (0, 0.5 * sum(c_k * r_k)].
    '''
    c = np.asarray(ics, dtype=float)
    r = np.asarray(rs, dtype=float)
    if c.ndim != 1 or r.shape != c.shape or c.size == 0:
        raise ValueError("ics and rs must be 1-D arrays of the same nonzero length")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r)) and np.all(c > 0.0) and np.all(r > 0.0)):
        raise ValueError("critical currents and resistances must be finite and positive")
    if c.max() / c.min() > 5.0:
        raise ValueError("max(ics) / min(ics) must not exceed 5")
    theta0 = float(theta0)
    if not (np.isfinite(theta0) and theta0 > 0.0):
        raise ValueError("theta0 must be finite and positive")
    th = theta0 / c                                   # E_J is proportional to I_c
    if np.any(th < 0.02) or np.any(th > 50.0):
        raise ValueError("every theta0 / c_k must lie in [0.02, 50]")
    v_crit = float(v_crit)
    cr = c * r
    if not (np.isfinite(v_crit) and 0.0 < v_crit <= 0.5 * cr.sum()):
        raise ValueError("v_crit must be in (0, 0.5 sum(c_k r_k)]")

    # Junction k: i_k = J / c_k, V_k = I_ck R_k v(i_k, theta_k). The array voltage rises
    # monotonically from 0 at J = 0; at J = 2 max(c) every i_k >= 2 and V >= sqrt(3) sum(c r).
    def excess(j):
        return sum(ck * rk * mean_voltage(j / ck, tk)[0] for ck, rk, tk in zip(c, r, th)) - v_crit

    j_star = brentq(excess, 0.0, 2.0 * c.max(), xtol=1e-15, rtol=1e-14)
    r_diff = 0.0
    s_v = 0.0
    for ck, rk, tk in zip(c, r, th):
        r_diff += rk * mean_voltage(j_star / ck, tk)[1]
        # independent phases; reduced time of junction k runs at omega_k = 2 e I_ck R_k / hbar
        s_v += 2.0 * ck * rk * effective_diffusion(j_star / ck, tk)
    result = (float(j_star), float(r_diff), float(s_v))
    return result


def _s6_velocity(i, theta, beta_c, N=160, P=100):
    # Stationary Kramers equation for x = phi and v = dphi/dtau,
    #   dW/dtau = -v dW/dx - d/dv[(f(x) - gam v) W] + gam vth**2 d2W/dv2,
    # f = (i - sin x) / beta_c, gam = 1 / beta_c, vth**2 = theta / beta_c, expanded as
    # W = psi_0(v) sum_n c_n(x) psi_n(v) (Hermite functions) and c_n(x) = sum_p c_np exp(i p x).
    # Brinkman hierarchy: sqrt(n+1) D c_{n+1} + sqrt(n) Dh c_{n-1} + gam n c_n = 0 with
    # D = vth d/dx, Dh = vth d/dx - f / vth; solved by the matrix continued fraction
    # c_n = S_n c_{n-1} (Risken, ch. 11). Mean velocity <v> = vth c_1,0 / c_0,0.
    vth = np.sqrt(theta / beta_c)
    gam = 1.0 / beta_c
    ps = np.arange(-P, P + 1)
    M = ps.size
    D = np.diag(vth * 1j * ps)
    F = np.diag(np.full(M, i / beta_c, dtype=complex))
    F += np.diag(np.full(M - 1, -1.0 / (2j * beta_c)), -1) + np.diag(np.full(M - 1, 1.0 / (2j * beta_c)), 1)
    Dh = D - F / vth
    S = np.zeros((M, M), complex)
    eye = np.eye(M)
    for n in range(N, 0, -1):
        S = -np.linalg.solve(gam * n * eye + np.sqrt(n + 1) * D @ S, np.sqrt(n) * Dh)
    # n = 0 equation D c_1 = D S_1 c_0 = 0 (its p = 0 row is empty) plus c_00 = 1 / (2 pi)
    Q = D @ S
    rows = np.array([k for k in range(M) if k != P])
    c0 = np.zeros(M, complex)
    c0[P] = 1.0 / (2.0 * np.pi)
    c0[rows] = np.linalg.solve(Q[np.ix_(rows, rows)], -Q[rows, P] / (2.0 * np.pi))
    c1 = S @ c0
    return float(vth * (c1[P] / c0[P]).real)


def rcsj_voltage(i, theta, beta_c):
    '''Stationary dc voltage and differential resistance of a noisy junction with capacitance (RCSJ).

    Inputs:
      i: float, reduced bias current I / I_c, |i| <= 2.5.
      theta: float, noise strength k_B T / E_J, 0.1 <= theta <= 2.
      beta_c: float, Stewart-McCumber parameter 2 e I_c R**2 C / hbar; beta_c == 0
              (overdamped, no capacitance) or 0.1 <= beta_c <= 2.

    Output:
      (v, r_d): tuple of two Python floats.
        v: mean phase velocity lim <phi(tau)> / tau = <V> / (I_c R) in the stationary state,
           relative error below 1e-7 (v = 0.0 exactly at i = 0).
        r_d: differential resistance dv/di (units of R), relative error below 1e-5.

    Raises:
      ValueError if an input is not finite, if |i| > 2.5, if theta is outside [0.1, 2], or
      if beta_c is neither 0 nor in [0.1, 2].
    '''
    i = float(i)
    theta = float(theta)
    beta_c = float(beta_c)
    if not (np.isfinite(i) and np.isfinite(theta) and np.isfinite(beta_c)):
        raise ValueError("inputs must be finite")
    if abs(i) > 2.5 or not (0.1 <= theta <= 2.0) or not (beta_c == 0.0 or 0.1 <= beta_c <= 2.0):
        raise ValueError("need |i| <= 2.5, 0.1 <= theta <= 2 and beta_c = 0 or 0.1 <= beta_c <= 2")
    if beta_c == 0.0:
        # no capacitance: the overdamped junction of step 1
        result = mean_voltage(i, theta)
        return result
    sign = 1.0 if i >= 0.0 else -1.0
    a = abs(i)
    v = 0.0 if a == 0.0 else sign * _s6_velocity(a, theta, beta_c)
    # dv/di (even in i) by Richardson-extrapolated central differences of the exact velocity
    hstep = 2e-3

    def vel(x):
        return np.sign(x) * _s6_velocity(abs(x), theta, beta_c) if x != 0.0 else 0.0

    d1 = (vel(a + hstep) - vel(a - hstep)) / (2.0 * hstep)
    d2 = (vel(a + 2.0 * hstep) - vel(a - 2.0 * hstep)) / (4.0 * hstep)
    r_d = (4.0 * d1 - d2) / 3.0
    result = (float(v), float(r_d))
    return result
