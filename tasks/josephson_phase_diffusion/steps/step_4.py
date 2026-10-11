import numpy as np
import mpmath as mp
from scipy.optimize import brentq


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
      between sqrt(max(i**2 - 1, 0)) and i, or if v < v_lo * (1 - 1e-11) or
      v > v_hi * (1 + 1e-11), with the endpoint voltages v_lo = mean_voltage(i, 0.02)[0] and
      v_hi = mean_voltage(i, 50)[0] of step 1. A v in [v_lo * (1 - 1e-11), v_lo] returns
      theta = 0.02 and a v in [v_hi, v_hi * (1 + 1e-11)] returns theta = 50.
    '''
    raise NotImplementedError
