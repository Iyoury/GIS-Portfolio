import numpy as np
import math
import warnings
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq


def specific_heat(rho_Ye, T):
    '''Specific heat per volume at constant net electron density of the electron-positron gas.

    Inputs:
      rho_Ye: float, rho Y_e in g cm^-3, 1e-10 <= rho_Ye <= 1e13 (n_net = rho_Ye N_A).
      T: float, temperature in K, 1e7 <= T <= 1e11.

    Output:
      cv: Python float, (d u_tot / d T) at constant n_net in erg K^-1 cm^-3, u_tot the total energy
        density including the rest energies; relative error below 1e-8.

    Raises:
      ValueError if rho_Ye or T is not finite or is outside its range.
    '''
    raise NotImplementedError
