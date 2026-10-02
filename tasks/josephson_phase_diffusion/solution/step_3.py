import numpy as np
import mpmath as mp
from scipy.optimize import brentq


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
