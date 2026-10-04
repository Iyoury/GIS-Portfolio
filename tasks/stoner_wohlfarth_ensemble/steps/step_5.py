import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def ensemble_switching(psis, weights, a, f0, rate):
    '''Dynamic coercive field and half-switching field of an ensemble during the sweep.

    Inputs:
      psis: 1-D array of easy-axis angles in radians, each in [0, pi/2].
      weights: 1-D array of the same length, nonnegative, not all zero (normalized by
               their sum).
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      (h_c, h_half): tuple of two floats, absolute errors below 1e-6.
        h_c: the ensemble magnetization is zero at h = -h_c during the sweep.
        h_half: half of the total weight has left its original minimum at h = -h_half.

    Raises:
      ValueError if psis and weights are not 1-D arrays of the same nonzero length, if
      a psi is outside [0, pi/2], if a weight is negative or not finite, if the weights
      add up to zero, if a, f0 or rate is not a positive finite number, if a is outside
      [40, 1000] or if f0 / rate is outside [1e5, 1e13].
    '''
    raise NotImplementedError
