import numpy as np
import math
import warnings
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq


def pair_thermodynamics(T, psi):
    '''Pressure, internal energy and entropy of an ideal electron-positron gas.

    Inputs:
      T: float, temperature in K, 1e7 <= T <= 1e11.
      psi: float, (mu + m_e c^2) / (k T), 0 < psi <= 1e6 (as in pair_densities).

    Output:
      (P, u, s): Python floats. P the pressure (erg cm^-3); u the energy density (erg cm^-3): kinetic
        energy of the electrons plus kinetic energy and 2 m_e c^2 per positron; s the entropy density
        (erg K^-1 cm^-3). Each with a relative error below 1e-10.

    Raises:
      ValueError if T or psi is not finite, if T is outside [1e7, 1e11] or psi is not in (0, 1e6].
    '''
    raise NotImplementedError
