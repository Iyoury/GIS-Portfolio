import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def substitution_times(N, s, u, v):
    '''Mean waiting times between the two monomorphic states under selection, mutation and drift.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 1e-12 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1.

    Output:
      (t_up, t_down): tuple of two Python floats.
        t_up: mean number of generations for a population with no copy of A (i = 0) to reach i = N
              for the first time.
        t_down: mean number of generations for a population fixed for A (i = N) to reach i = 0 for
                the first time.
        Each with a relative error below 1e-8, however large (t_down is about 1e150 at
        N = 400, s = 0.5, u = v = 1e-12).

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
    P = wf_transition_matrix(N, s, u, v)
    n = N + 1
    times = []
    for target, start in ((N, 0), (0, N)):
        # mean hitting times of the target: d_k x_k = 1 + sum_{j != k, target} a_kj x_j, with
        # d_k = sum_{j != k} P_kj (target included). At a fixed state with mutation rates near 1e-12
        # 1 - P_kk is far below rounding of P_kk, so it is never formed by subtraction; the other
        # states are removed one at a time keeping every coefficient a sum of nonnegative terms.
        A = P.copy()
        np.fill_diagonal(A, 0.0)
        others = [i for i in range(n) if i != target]
        alive = np.zeros(n, dtype=bool)
        alive[others] = True
        b = np.ones(n)
        b[target] = 0.0
        steps = []
        for k in others:
            alive[k] = False
            d = A[k].sum()
            row = A[k].copy()
            col = np.where(alive, A[:, k], 0.0) / d
            steps.append((k, row, d))
            b += col * b[k]
            A += np.outer(col, row)
            A[:, k] = 0.0
            A[k, :] = 0.0
            np.fill_diagonal(A, 0.0)
        x = np.zeros(n)
        for k, row, d in reversed(steps):
            x[k] = (b[k] + row @ x) / d
        times.append(float(x[start]))
    result = (times[0], times[1])
    return result
