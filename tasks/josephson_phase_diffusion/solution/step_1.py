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
