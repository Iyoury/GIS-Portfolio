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
    raise NotImplementedError
