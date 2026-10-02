import numpy as np
from scipy.optimize import brentq


def strip_susceptibility(L, T):
    '''Zero-field magnetic susceptibility per spin of an infinitely long periodic strip.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10.
      T: float, temperature, 1 <= T <= 10 (J = 1, k_B = 1).

    Output:
      chi: float, chi = - d^2 f / d h^2 at h = 0, where f(h) is the free energy per spin of
           the infinitely long strip in the uniform field h of transfer_matrix (units 1/J).
           Relative accuracy 1e-6.
    '''
    M = transfer_matrix(L, T)
    n = 2 ** L
    labels = np.arange(n)
    partner = labels ^ (n - 1)              # the same row with every spin flipped
    rep = labels[labels < partner]
    # M commutes with the global spin flip. In the basis (|a> +- |flip a>) / sqrt(2) it splits
    # into an even block M[a, b] + M[a, flip b] and an odd block M[a, b] - M[a, flip b].
    near = M[np.ix_(rep, rep)]
    far = M[np.ix_(rep, partner[rep])]
    w_even, v_even = np.linalg.eigh(near + far)
    w_odd, v_odd = np.linalg.eigh(near - far)
    S = np.sum(1 - 2 * ((rep[:, None] >> np.arange(L)[None, :]) & 1), axis=1)   # row magnetization
    lam0 = w_even[-1]
    amp = v_odd.T @ (S * v_even[:, -1])     # <k|S|0> for every odd state k
    # chi = (1 / (T L)) sum over all r of <S(row 0) S(row r)>, summed in closed form state by state
    chi = float(np.sum(amp ** 2 * (lam0 + w_odd) / (lam0 - w_odd)) / (T * L))
    return chi
