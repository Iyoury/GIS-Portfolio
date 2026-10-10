import numpy as np
import mpmath as mp
from scipy.optimize import brentq


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
           error below 1e-7 * |v| + 1e-12 (v = 0 at i = 0, so |v| <= 1e-12 there).
        r_d: differential resistance dv/di (units of R), relative error below 1e-5.

    Raises:
      ValueError if an input is not finite, if |i| > 2.5, if theta is outside [0.1, 2], or
      if beta_c is neither 0 nor in [0.1, 2].
    '''
    raise NotImplementedError
