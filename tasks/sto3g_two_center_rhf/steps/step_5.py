import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def vibrational_levels(ZA, ZB, zetaA, zetaB, massA, massB):
    '''Lowest five vibrational levels (J = 0) on the full-CI potential curve, from the dissociation limit.

    Inputs:
      ZA, ZB: float, nuclear charges, 1 <= Z <= 3.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, 0.8 <= zeta <= 3.
      massA, massB: float, nuclear masses in unified atomic mass units (u), 1 <= mass <= 10.

    Output:
      levels: float numpy array of shape (5,), the energies of the vibrational states
              v = 0..4 in cm^-1, measured from the exact R -> infinity limit of the
              full-CI energy (so all are negative), increasing. Absolute error below
              0.01 cm^-1 for each level.

    Raises:
      ValueError if a mass is not finite or not in [1, 10] u, or if fewer than five
      bound vibrational levels lie below the dissociation limit.
    '''
    raise NotImplementedError
