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
        # (a - b)^2 for the two eigenvalues a, b of largest modulus: positive while they are real and
        # distinct, zero where they meet, negative once they form a complex-conjugate pair.
        w = np.linalg.eigvals(transfer_matrix(L, T, 1j * H))
        top = w[np.argsort(-np.abs(w))[:2]]
        return float(((top[0] - top[1]) ** 2).real)

    lo = hi = 1e-12 * T
    while discriminant(hi) > 0.0:           # the edge is tiny below Tc: walk up geometrically
        lo, hi = hi, 2.0 * hi
    H_edge = float(brentq(discriminant, lo, hi, xtol=1e-16, rtol=1e-13))
    return H_edge
