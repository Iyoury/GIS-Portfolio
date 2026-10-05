import numpy as np
import math


def electron_positron_eos(rho, T, Ye):
    '''Equation of state of the electron-positron gas for a mass density, temperature and electron fraction.

    Inputs:
      rho: float, mass density in g cm^-3.
      T: float, temperature in K, 1e7 <= T <= 1e11.
      Ye: float, electrons per baryon, 0 < Ye <= 1; 1e-10 <= rho Ye <= 1e13.

    Output:
      dict with the Python floats "psi", "n_minus", "n_plus", "P", "u", "s", "cv" (units and accuracy
        as in steps 1-4). Each call within 30 s.

    Raises:
      ValueError if rho, T or Ye is not finite, if Ye is not in (0, 1], or if T or rho Ye is outside
      its range.
    '''
    raise NotImplementedError
