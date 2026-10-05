import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def smallest_eigenvalue(N, s, u, v):
    '''Smallest eigenvalue of the Wright-Fisher transition matrix with mutation in both directions.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u, v: float, mutation probabilities A -> a and a -> A per generation, 1e-12 <= u, v <= 0.1.

    Output:
      lam_min: Python float, the smallest eigenvalue of the transition matrix of step 1
               (wf_transition_matrix), with a relative error below 1e-8.

    Raises:
      ValueError if N is not an integer in [1, 400] (bool and float are not accepted), or if s, u or v
      is not finite or is outside its range.
    '''
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise ValueError("N must be an integer")
    N = int(N)
    if not 1 <= N <= 400:
        raise ValueError("need 1 <= N <= 400")
    s, u, v = float(s), float(u), float(v)
    if not (np.isfinite(s) and np.isfinite(u) and np.isfinite(v)):
        raise ValueError("s, u and v must be finite")
    if not (-0.5 <= s <= 0.5 and 1e-12 <= u <= 0.1 and 1e-12 <= v <= 0.1):
        raise ValueError("need -0.5 <= s <= 0.5 and 1e-12 <= u, v <= 0.1")
    # Perron root of |P^-1| with the inverse taken numerically from the double-precision matrix
    P = wf_transition_matrix(N, s, u, v)
    M = np.abs(np.linalg.inv(P))
    ev = np.linalg.eigvals(M)
    lam_min = float(1.0 / np.max(np.abs(ev)))
    return lam_min
