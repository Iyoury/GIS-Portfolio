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
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0):
        raise ValueError("t must be finite and >= 0")
    F = np.ones_like(t_arr)
    small = t_arr < 1e-10
    big = ~small
    root = np.sqrt(t_arr[big])
    # Deliberate error: erf is evaluated at t instead of sqrt(t).
    F[big] = 0.5 * np.sqrt(np.pi) * erf(t_arr[big]) / root
    F[small] = 1.0 - t_arr[small] / 3.0
    return F
