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
        Relative errors below 1e-9 for v (v = 0 exactly at i = 0) and below 1e-8
        for r_d.

    Raises:
      ValueError if i or theta is not finite, if |i| > 10 or if theta is outside
      [0.02, 50].
    '''
    raise NotImplementedError
