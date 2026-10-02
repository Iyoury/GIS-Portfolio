import numpy as np
from scipy.optimize import brentq


def transfer_matrix(L, T, h=0.0):
    '''Symmetric row-to-row transfer matrix of the 2D Ising model on a periodic strip in a field h.

    Inputs:
      L: int, width of the strip in spins, L >= 3. Spin L is the same as spin 0.
      T: float, temperature, T > 0 (J = 1, k_B = 1).
      h: float or complex, uniform field; the energy is E = - sum_<ij> s_i s_j - h sum_i s_i.
         A complex h, such as 1j * H, is allowed.

    Output:
      M: numpy array of shape (2**L, 2**L), complex when h is complex. Row label
         n = sum_i b_i * 2**i, with b_i = 0 for spin i = +1 and b_i = 1 for spin i = -1.
         M[n, m] is the Boltzmann weight that joins row n to row m, with the bonds inside a
         row and the field on a row shared equally between the two factors that touch that
         row, so that M == M.T. Relative accuracy 1e-10 for every entry.
    '''
    K = 1.0 / T
    labels = np.arange(2 ** L)
    spins = 1 - 2 * ((labels[:, None] >> np.arange(L)[None, :]) & 1)
    inside = np.sum(spins * np.roll(spins, -1, axis=1), axis=1)   # bonds inside each row
    field = np.sum(spins, axis=1)                                  # sum of the spins of each row
    half = 0.5 * K * inside + 0.5 * (h / T) * field                # half of a row's own weight
    M = np.exp(half[:, None] + half[None, :] + K * (spins @ spins.T))
    return M
