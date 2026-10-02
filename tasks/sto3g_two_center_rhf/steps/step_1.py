import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def boys_function(n_max, t):
    '''Boys functions F_n(t) = integral from 0 to 1 of u**(2n) exp(-t u**2) du for n = 0..n_max.

    Inputs:
      n_max: int, highest order, 0 <= n_max <= 16.
      t: float or numpy array of floats (any shape), 0 <= t <= 1e6 (t = 0 is allowed).

    Output:
      F: float numpy array of shape np.shape(t) + (n_max + 1,), F[..., n] = F_n(t).
         Relative error below 1e-12 for every entry; F_n(0) = 1 / (2n + 1).

    Raises:
      ValueError if n_max is not an integer in [0, 16], or if any t is negative, not finite
      or larger than 1e6.
    '''
    raise NotImplementedError
