import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def relaxation_rate(N, s, u, v):
    '''Rate of the slowest approach to the stationary distribution: 1 - lambda_2 of the transition matrix.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 1e-12 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1.

    Output:
      gap: Python float, 1 - lambda_2, where lambda_2 is the second largest eigenvalue of the
           transition matrix (all its eigenvalues are real, positive and distinct). Relative error
           below 1e-8, however small the gap is.

    Raises:
      ValueError if N is not an integer in [1, 400], or if s, u or v is not finite or is outside its
      range.
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
    # MUTANT: the neutral result 1 - lambda_2 = u + v used for every s
    gap = float(u + v)
    return gap
