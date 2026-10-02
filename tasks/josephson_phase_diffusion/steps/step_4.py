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
      between sqrt(max(i**2 - 1, 0)) and i, or if no theta in [0.02, 50] gives v.
    '''
    raise NotImplementedError
