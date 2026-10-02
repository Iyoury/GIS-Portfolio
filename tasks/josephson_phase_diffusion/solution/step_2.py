import numpy as np
import mpmath as mp
from scipy.optimize import brentq


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
