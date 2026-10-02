import numpy as np
import mpmath as mp
from scipy.optimize import brentq


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
    # stated endpoint tolerance: a v beyond the voltage at an endpoint by at most a relative
    # 1e-11 of that voltage returns the endpoint; farther out there is no solution
    v_lo, v_hi = mean_voltage(i, 0.02)[0], mean_voltage(i, 50.0)[0]
    if v <= v_lo:
        if (v_lo - v) / v_lo > 1e-11:
            raise ValueError("no theta in [0.02, 50] reproduces this voltage")
        theta = 0.02
    elif v >= v_hi:
        if (v - v_hi) / v_hi > 1e-11:
            raise ValueError("no theta in [0.02, 50] reproduces this voltage")
        theta = 50.0
    else:
        theta = float(np.exp(brentq(g, lo, hi, xtol=1e-14, rtol=1e-15)))
    # MUTANT: returns d ln(v) / d ln(theta) (the voltage response) instead of its inverse
    kappa = float(_s4_dlogv_dlogtheta(i, theta))
    result = (theta, kappa)
    return result
