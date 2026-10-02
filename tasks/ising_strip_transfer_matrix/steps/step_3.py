import numpy as np
from scipy.optimize import brentq


def strip_magnetization(L, T):
    '''Amplitude m_L of the slowest-decaying term of the spin correlation along a periodic strip.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10.
      T: float, temperature, 0.5 <= T <= 10 (J = 1, k_B = 1, zero field).

    Output:
      m_L: float >= 0, defined for an infinitely long strip by
           <s_0(row 0) s_0(row r)> = m_L**2 * exp(-r / xi_spin) + (terms that decay faster).
           Absolute accuracy 1e-9.
    '''
    raise NotImplementedError
