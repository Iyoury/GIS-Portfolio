import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def stationary_distribution(N, s, u, v):
    '''Stationary distribution of the number of copies of A under selection, mutation and drift.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 1e-12 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1.

    Output:
      pi: numpy array of shape (N + 1,), pi[i] = stationary probability of i copies of A, summing to 1.
          Relative error below 1e-8 for every entry whose exact value is at least 1e-250; smaller
          entries within 1e-250 of the exact value.

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
    A = wf_transition_matrix(N, s, u, v).copy()
    n = N + 1
    # Grassmann-Taksar-Heyman state reduction: remove the states n-1, ..., 1 one at a time. The
    # escape probability of state k into the states still present is taken as the sum of its
    # off-diagonal probabilities into them (never as 1 - P_kk, which cancels when P_kk is close to 1,
    # e.g. at a fixed state with mutation rates 1e-12), so no subtraction occurs at all.
    for k in range(n - 1, 0, -1):
        escape = A[k, :k].sum()
        A[:k, k] /= escape
        A[:k, :k] += np.outer(A[:k, k], A[k, :k])
    pi = np.zeros(n)
    pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[:k] @ A[:k, k]
    pi = pi / pi.sum()
    return pi
