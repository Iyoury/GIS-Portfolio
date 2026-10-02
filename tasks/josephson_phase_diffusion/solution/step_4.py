import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def noise_temperature(i, v):
    '''Noise strength theta = k_B T / E_J inferred from a measured dc voltage.

    Inputs:
      i: float, reduced bias current, 0 < i <= 10.
      v: float, measured reduced mean voltage <V> / (I_c R) at this bias,
         sqrt(max(i**2 - 1, 0)) < v < i.

    Output:
      theta: Python float in [0.02, 50] with mean_voltage(i, theta)[0] == v,
             relative error below 1e-7.

    Raises:
      ValueError if i or v is not finite, if i is not in (0, 10], if v is not strictly
      between sqrt(max(i**2 - 1, 0)) and i, or if no theta in [0.02, 50] gives v.
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
    if g_lo > 0.0 or g_hi < 0.0:
        raise ValueError("no theta in [0.02, 50] reproduces this voltage")
    if g_lo == 0.0:
        return 0.02
    if g_hi == 0.0:
        return 50.0
    theta = float(np.exp(brentq(g, lo, hi, xtol=1e-14, rtol=1e-15)))
    return theta
