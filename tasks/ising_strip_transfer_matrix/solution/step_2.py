import numpy as np
from scipy.optimize import brentq


def strip_lengths(L, T):
    '''Free energy per spin and the spin and energy correlation lengths of an infinitely long periodic strip.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10.
      T: float, temperature, 1 <= T <= 10 (J = 1, k_B = 1, zero field).

    Output:
      (f, xi_spin, xi_energy): tuple of three floats.
        f: free energy per spin (units of J), relative accuracy 1e-10.
        xi_spin: decay length of <s_0(row 0) s_0(row r)> along the strip (lattice spacings).
        xi_energy: decay length of the connected correlation of e = s_0 s_1 along the strip.
        Both lengths with relative accuracy 1e-6.
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
    even = np.linalg.eigvalsh(near + far)
    odd = np.linalg.eigvalsh(near - far)
    lam0 = even[-1]
    # s_0 is odd under the flip, so it joins the top state to the odd sector; e = s_0 s_1 is even,
    # so its connected correlation decays through the second state of the even sector.
    f = float(-T * np.log(lam0) / L)
    xi_spin = float(1.0 / np.log(lam0 / odd[-1]))
    xi_energy = float(1.0 / np.log(lam0 / even[-2]))
    return f, xi_spin, xi_energy
