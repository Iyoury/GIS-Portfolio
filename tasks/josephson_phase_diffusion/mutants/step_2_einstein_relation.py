import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def effective_diffusion(i, theta):
    '''Effective diffusion coefficient of the junction phase.

    Inputs:
      i: float, reduced bias current I / I_c, |i| <= 10.
      theta: float, noise strength k_B T / E_J, 0.02 <= theta <= 50.

    Output:
      D: Python float, lim [<phi(tau)**2> - <phi(tau)>**2] / (2 tau) in reduced units
         (tau = 2 e I_c R t / hbar). Relative error below 1e-9.

    Raises:
      ValueError if i or theta is not finite, if |i| > 10 or if theta is outside
      [0.02, 50].
    '''
    i = float(i)
    theta = float(theta)
    if not (np.isfinite(i) and np.isfinite(theta)):
        raise ValueError("i and theta must be finite")
    if abs(i) > 10.0 or not (0.02 <= theta <= 50.0):
        raise ValueError("need |i| <= 10 and 0.02 <= theta <= 50")
    # MUTANT: Einstein relation D = theta * (differential mobility), valid only at equilibrium
    D = float(theta * mean_voltage(i, theta)[1])
    return D
