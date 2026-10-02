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
    M = transfer_matrix(L, T)
    n = 2 ** L
    labels = np.arange(n)
    partner = labels ^ (n - 1)              # the same row with every spin flipped
    rep = labels[labels < partner]
    # M commutes with the global spin flip. In the basis (|a> +- |flip a>) / sqrt(2) it splits
    # into an even block M[a, b] + M[a, flip b] and an odd block M[a, b] - M[a, flip b].
    near = M[np.ix_(rep, rep)]
    far = M[np.ix_(rep, partner[rep])]
    # G(r) = sum_k <0|s_0|k>^2 (lam_k / lam_0)^r; the slowest term comes from the top odd state.
    # Below Tc that state is degenerate with the top even state to machine precision, so the two
    # are taken from the separate sector blocks rather than from one eigensolver call.
    w_even, v_even = np.linalg.eigh(near + far)
    w_odd, v_odd = np.linalg.eigh(near - far)
    s0 = 1 - 2 * (rep & 1)                  # s_0 sends (|a> + |flip a>)/sqrt2 to s_0(a) (|a> - |flip a>)/sqrt2
    m_L = abs(float(np.sum(v_even[:, -1] * s0 * v_odd[:, -1])))
    return m_L
