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
    def any_complex(H):
        # looks for the first eigenvalue of the whole spectrum that leaves the real axis
        w = np.linalg.eigvals(transfer_matrix(L, T, 1j * H))
        return bool(np.max(np.abs(w.imag)) > 1e-9 * np.max(np.abs(w)))

    lo = hi = 1e-12 * T
    while not any_complex(hi):
        lo, hi = hi, 2.0 * hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if any_complex(mid):
            hi = mid
        else:
            lo = mid
    H_edge = float(0.5 * (lo + hi))
    return H_edge
