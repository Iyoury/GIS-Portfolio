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
    def discriminant(H):
        # MUTANT: watches the whole spectrum instead of the leading pair: positive while every
        # eigenvalue is real, negative once any pair of eigenvalues has left the real axis, so the
        # root is the first field at which any eigenvalue becomes complex.
        w = np.linalg.eigvals(transfer_matrix(L, T, 1j * H))
        return float(1e-9 - np.max(np.abs(w.imag)) / np.max(np.abs(w)))

    lo = hi = 1e-12 * T
    while discriminant(hi) > 0.0:           # the edge is tiny below Tc: walk up geometrically
        lo, hi = hi, 2.0 * hi
    H_edge = float(brentq(discriminant, lo, hi, xtol=1e-16, rtol=1e-13))
    return H_edge
