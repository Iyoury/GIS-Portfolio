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


def _s4_g(y):
    # g(y) = y cos(y/2) - 2 sin(y/2) = -y**3 / 12 + y**5 / 480 - ..., summed as its series for
    # y < 1, where the two terms of the closed form cancel
    y = np.asarray(y, dtype=float)
    out = y * np.cos(0.5 * y) - 2.0 * np.sin(0.5 * y)
    m = y < 1.0
    t = -(0.5 * y[m]) ** 2
    term = np.ones_like(t)
    s = np.zeros_like(t)
    f = 1.0
    for k in range(1, 12):
        term = term * t
        f *= (2 * k) * (2 * k + 1)                      # (2k + 1)!
        s += term * (2 * k) / f
    out[m] = y[m] * s
    return out


def _s4_dlogv_dlogtheta(i, theta):
    # d ln v / d ln theta at fixed i > 0 from the Stratonovich formula v = 2 pi q / S,
    # q = 1 - exp(-z), z = 2 pi i / theta, S = (1/theta) int_0^{2 pi} dx int_0^{2 pi} dy exp(E / theta),
    # E = cos(x - y) - cos(x) - i y. The theta-derivative is taken with the shift scaled as
    # y = theta u, so that the tilt term -i u does not depend on theta; differentiating
    # [cos(x - theta u) - cos(x)] / theta and the upper limit 2 pi / theta gives
    #   theta dS/dtheta = (1 / theta**2) int int exp(E / theta) F dx dy - 4 pi**2 exp(-z) / theta,
    #   F = cos(x) - cos(x - y) + y sin(x - y) = sin(x - y/2) g(y) - y sin(y/2) cos(x - y/2).
    # F = O(y**2) where the weight exp(E / theta) is concentrated, so d ln v / d ln theta is
    # obtained without the cancellation of differentiating exp(E / theta) directly, which loses
    # the relative accuracy when d ln v / d ln theta is tiny (kappa up to about 2e7 at large bias
    # and weak noise).
    nx = 2 * int(40 + 30 / np.sqrt(theta))
    x = 2.0 * np.pi * np.arange(nx) / nx
    y, w = _s4_nodes(i, theta)
    X, Y = x[:, None], y[None, :]
    ex = np.exp((np.cos(X - Y) - np.cos(X) - i * Y) / theta)
    F = np.sin(X - 0.5 * Y) * _s4_g(y)[None, :] - Y * np.sin(0.5 * Y) * np.cos(X - 0.5 * Y)
    A = np.mean((ex * F) @ w)
    B = np.mean(ex @ w)
    z = 2.0 * np.pi * i / theta
    ez = np.exp(-z)
    # theta d ln q / dtheta - theta d ln S / dtheta
    return -z * ez / (-np.expm1(-z)) + 2.0 * np.pi * ez / B - A / (theta * B)


def noise_temperature(i, v):
    '''Noise strength theta = k_B T / E_J inferred from a measured dc voltage, and its sensitivity.

    Inputs:
      i: float, reduced bias current, 0 < i <= 10.
      v: float, measured reduced mean voltage <V> / (I_c R) at this bias,
         sqrt(max(i**2 - 1, 0)) < v < i.

    Output:
      (theta, kappa): tuple of two Python floats.
        theta: noise strength in [0.02, 50] with mean_voltage(i, theta)[0] == v (the
               endpoints 0.02 and 50 are allowed). It is graded against the noise strength
               at which the exact stationary voltage equals v, relative error below
               1e-7 + 2e-9 * |kappa| (the relative error of mean_voltage, below 1e-9,
               enters theta multiplied by |kappa|).
        kappa: thermometric sensitivity d ln(theta) / d ln(v) of the exact stationary
               voltage at fixed i, evaluated at the returned theta (a relative voltage
               error dv/v gives the relative temperature error kappa dv/v), relative error
               below 1e-6. Over the domain kappa ranges from about 0.01 to about 2e7.

    Raises:
      ValueError if i or v is not finite, if i is not in (0, 10], if v is not strictly
      between sqrt(max(i**2 - 1, 0)) and i, or if v < v_lo * (1 - 1e-11) or
      v > v_hi * (1 + 1e-11), with the endpoint voltages v_lo = mean_voltage(i, 0.02)[0] and
      v_hi = mean_voltage(i, 50)[0] of step 1. A v in [v_lo * (1 - 1e-11), v_lo] returns
      theta = 0.02 and a v in [v_hi, v_hi * (1 + 1e-11)] returns theta = 50.
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
    # stated endpoint rule, relative to the endpoint voltages of mean_voltage (step 1): a v
    # beyond v_lo or v_hi by at most a relative 1e-11 returns the endpoint; farther out, ValueError
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
    kappa = float(1.0 / _s4_dlogv_dlogtheta(i, theta))
    result = (theta, kappa)
    return result
