import numpy as np
import math


def degeneracy_parameter(rho_Ye, T):
    '''psi = (mu + m_e c^2) / (k T) at which the net electron density equals rho_Ye N_A.

    Inputs:
      rho_Ye: float, rho Y_e in g cm^-3, 1e-10 <= rho_Ye <= 1e13.
      T: float, temperature in K, 1e7 <= T <= 1e11.

    Output:
      psi: Python float, psi > 0 with n_net(T, psi) = rho_Ye N_A (n_net of pair_densities), with a
        relative error below 1e-10. Each call within 10 s.

    Raises:
      ValueError if rho_Ye or T is not finite or is outside its range.
    '''
    raise NotImplementedError
