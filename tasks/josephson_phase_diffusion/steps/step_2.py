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
    raise NotImplementedError
