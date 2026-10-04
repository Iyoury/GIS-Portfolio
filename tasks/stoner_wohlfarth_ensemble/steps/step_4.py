import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def switching_field_statistics(psi, a, f0, rate):
    '''Median and mean switching field of one particle in the descending sweep.

    Inputs:
      psi: float, easy-axis angle in radians, 0 <= psi <= pi/2.
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      (h_median, h_mean): tuple of two floats, the median and the mean of the field at
      which the particle leaves its original minimum. Absolute errors below 1e-7.

    Raises:
      ValueError if psi is outside [0, pi/2], if a, f0 or rate is not a positive finite
      number, if a is outside [40, 1000] or if f0 / rate is outside [1e5, 1e13].
    '''
    raise NotImplementedError
