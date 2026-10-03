import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def _s6_velocity(i, theta, beta_c, N=110, P=56):
    # Stationary Kramers equation for x = phi and v = dphi/dtau,
    #   dW/dtau = -v dW/dx - d/dv[(f(x) - gam v) W] + gam vth**2 d2W/dv2,
    # f = (i - sin x) / beta_c, gam = 1 / beta_c, vth**2 = theta / beta_c, expanded as
    # W = psi_0(v) sum_n c_n(x) psi_n(v) (Hermite functions) and c_n(x) = sum_p c_np exp(i p x).
    # Brinkman hierarchy: sqrt(n+1) D c_{n+1} + sqrt(n) Dh c_{n-1} + gam n c_n = 0 with
    # D = vth d/dx, Dh = vth d/dx - f / vth; solved by the matrix continued fraction
    # c_n = S_n c_{n-1} (Risken, ch. 11). Mean velocity <v> = vth c_1,0 / c_0,0.
    # The derivative with respect to i is carried through the same recursion:
    # A_n S_n = -B_n with A_n = gam n + sqrt(n+1) D S_{n+1}, B_n = sqrt(n) Dh, dDh/di = -1/(beta_c vth),
    # so dS_n = -A_n^{-1} (dB_n + sqrt(n+1) D dS_{n+1} S_n). Returns (v, dv/di).
    vth = np.sqrt(theta / beta_c)
    gam = 1.0 / beta_c
    ps = np.arange(-P, P + 1)
    M = ps.size
    D = np.diag(vth * 1j * ps)
    F = np.diag(np.full(M, i / beta_c, dtype=complex))
    F += np.diag(np.full(M - 1, -1.0 / (2j * beta_c)), -1) + np.diag(np.full(M - 1, 1.0 / (2j * beta_c)), 1)
    Dh = D - F / vth
    dDh = -np.eye(M) / (beta_c * vth)
    S = np.zeros((M, M), complex)
    dS = np.zeros((M, M), complex)
    eye = np.eye(M)
    for n in range(N, 0, -1):
        A = gam * n * eye + np.sqrt(n + 1) * D @ S
        S_new = -np.linalg.solve(A, np.sqrt(n) * Dh)
        dS = -np.linalg.solve(A, np.sqrt(n) * dDh + np.sqrt(n + 1) * D @ dS @ S_new)
        S = S_new
    # n = 0 equation D c_1 = D S_1 c_0 = 0 (its p = 0 row is empty) plus c_00 = 1 / (2 pi)
    Q = D @ S
    dQ = D @ dS
    rows = np.array([k for k in range(M) if k != P])
    Qr = Q[np.ix_(rows, rows)]
    c0 = np.zeros(M, complex)
    c0[P] = 1.0 / (2.0 * np.pi)
    c0[rows] = np.linalg.solve(Qr, -Q[rows, P] / (2.0 * np.pi))
    dc0 = np.zeros(M, complex)
    dc0[rows] = np.linalg.solve(Qr, -(dQ @ c0)[rows])
    c1 = S @ c0
    dc1 = dS @ c0 + S @ dc0
    v = float(vth * (c1[P] / c0[P]).real)
    dv = float(vth * (dc1[P] / c0[P]).real)
    return v, dv


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
           error below 1e-7 * |v| + 1e-12 (v = 0.0 exactly at i = 0).
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
    # MUTANT: the capacitance is ignored; every junction is treated as overdamped
    result = mean_voltage(i, theta)
    return result
    a = abs(i)
    if a == 0.0:
        # v is odd in i: v(0) = 0 exactly; dv/di from the same continued fraction at i = 0
        v, r_d = 0.0, _s6_velocity(0.0, theta, beta_c)[1]
    else:
        v, r_d = _s6_velocity(a, theta, beta_c)
        v = v if i > 0.0 else -v                     # v is odd, r_d is even in i
    result = (float(v), float(r_d))
    return result
