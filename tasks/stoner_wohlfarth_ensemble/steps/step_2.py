import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def escape_barriers(h, psi):
    '''Reduced energy barriers that keep a particle in its original minimum.

    Inputs:
      h: float or numpy array (any shape), reduced field with |h| < h_sw(psi).
      psi: float, angle in radians between the field axis and the easy axis, 0 <= psi <= pi/2.

    Output:
      (low, high): two values with the shape of np.asarray(h), numpy float scalars for a
      scalar h. low <= high are
      e(theta_max) - e(theta_min) for the two energy maxima, theta_min being the original
      minimum of the descending branch. Absolute error below 1e-8 when
      h_sw(psi) - |h| >= 1e-3.

    Raises:
      ValueError if psi is outside [0, pi/2], if any h is not finite, or if any
      |h| >= h_sw(psi).
    '''
    raise NotImplementedError
