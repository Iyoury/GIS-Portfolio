import numpy as np
import mpmath as mp
from scipy.optimize import brentq


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
