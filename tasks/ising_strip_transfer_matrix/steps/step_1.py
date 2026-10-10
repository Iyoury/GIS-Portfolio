import numpy as np
from scipy.optimize import brentq


def transfer_matrix(L, T, h=0.0):
    '''Symmetric row-to-row transfer matrix of the 2D Ising model on a periodic strip in a field h.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10. Spin L is the same as spin 0.
      T: float, temperature, 0.5 <= T <= 10 (J = 1, k_B = 1).
      h: float or complex, uniform field; the energy is E = - sum_<ij> s_i s_j - h sum_i s_i.
         |h| <= 20; a complex h, such as 1j * H, is allowed.

    Output:
      M: numpy array of shape (2**L, 2**L), complex when h is complex. Row label
         n = sum_i b_i * 2**i, with b_i = 0 for spin i = +1 and b_i = 1 for spin i = -1.
         M[n, m] is the Boltzmann weight that joins row n to row m, with the bonds inside a
         row and the field on a row shared equally between the two factors that touch that
         row, so that M == M.T. Relative accuracy 1e-10 for every entry.
    '''
    raise NotImplementedError
