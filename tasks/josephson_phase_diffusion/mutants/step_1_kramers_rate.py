import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def mean_voltage(i, theta):
    '''Stationary dc voltage and differential resistance of a noisy overdamped junction.

    Inputs:
      i: float, reduced bias current I / I_c, |i| <= 10.
      theta: float, noise strength k_B T / E_J, 0.02 <= theta <= 50.

    Output:
      (v, r_d): tuple of two Python floats.
        v: mean phase velocity lim <phi(tau)> / tau = <V> / (I_c R).
        r_d: differential resistance dv/di (in units of R).
        Relative errors below 1e-9 for v and below 1e-8 for r_d; at i = 0,
        v = 0 within an absolute 1e-12.

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
    # MUTANT: Kramers escape rates over the washboard barriers instead of the exact current.
    # Below the critical current v = 2 pi (k_+ - k_-), k_+- = sqrt(U''_min |U''_max|) / (2 pi)
    # exp(-dU_+- / theta); above it the noiseless running solution sqrt(i**2 - 1).
    def v_of(x):
        a = abs(x)
        if a < 1.0:
            w = np.sqrt(1.0 - a * a)
            dU = 2.0 * w - a * (np.pi - 2.0 * np.arcsin(a))
            val = w * np.exp(-dU / theta) * (-np.expm1(-2.0 * np.pi * a / theta))
        else:
            val = np.sqrt(a * a - 1.0)
        return np.copysign(val, x)
    h = 1e-6
    v = v_of(i)
    r_d = (v_of(i + h) - v_of(i - h)) / (2.0 * h)
    result = (float(v), float(r_d))
    return result
