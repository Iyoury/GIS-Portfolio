import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def diffusion_peak(theta):
    '''Bias current of maximal phase diffusion (giant diffusion) and the enhancement there.

    Inputs:
      theta: float, noise strength k_B T / E_J, 0.02 <= theta <= 1.

    Output:
      (i_peak, d_peak): tuple of two Python floats.
        i_peak: the bias i > 0 at which effective_diffusion(i, theta) is largest
                (absolute error below 1e-6).
        d_peak: D_eff(i_peak, theta) / theta (relative error below 1e-9).

    Raises:
      ValueError if theta is not finite or is outside [0.02, 1].
    '''
    raise NotImplementedError
