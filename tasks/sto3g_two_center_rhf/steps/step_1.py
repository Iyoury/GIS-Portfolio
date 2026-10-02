import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def boys_f0(t):
    '''Boys function of order zero, F0(t) = integral from 0 to 1 of exp(-t u**2) du.

    Input:
      t: float or numpy array of floats (any shape), t >= 0 (t = 0 is allowed).

    Output:
      F: float numpy array with the same shape as np.asarray(t); F0(0) = 1.
         Relative error below 1e-11 for every t >= 0.

    Raises:
      ValueError if any t is negative or not finite.
    '''
    raise NotImplementedError
