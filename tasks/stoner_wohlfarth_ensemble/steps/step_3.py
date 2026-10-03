import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def survival_probability(h, psi, a, f0, rate):
    '''Probability that a particle has not left its original minimum when the sweep reaches h.

    Inputs:
      h: float, reduced field reached by the descending sweep.
      psi: float, easy-axis angle in radians, 0 <= psi <= pi/2.
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      P: float in [0, 1], P = 1 for h >= h_sw(psi) and P = 0 for h <= -h_sw(psi).
         Absolute error below 1e-10.

    Raises:
      ValueError if psi is outside [0, pi/2], if h is not finite, if a, f0 or rate is
      not a positive finite number, if a is outside [40, 1000] or if f0 / rate is
      outside [1e5, 1e13].
    '''
    raise NotImplementedError
