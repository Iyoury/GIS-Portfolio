import numpy as np
from scipy.optimize import brentq


def lee_yang_edge(L, T):
    '''Yang-Lee edge of an infinitely long periodic strip in a purely imaginary field.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 8.
      T: float, temperature, 1.5 <= T <= 10 (J = 1, k_B = 1).

    Output:
      H_edge: float, the smallest H > 0 at which the two eigenvalues of largest modulus of
              transfer_matrix(L, T, 1j * H) are equal (units of J). Relative accuracy 1e-8.
    '''
    raise NotImplementedError
