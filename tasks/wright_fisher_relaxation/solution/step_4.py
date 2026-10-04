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
    P = wf_transition_matrix(N, s, u, v)
    pi_all = stationary_distribution(N, s, u, v)
    n = N + 1
    # The gap is the smallest nonzero eigenvalue of L = I - P. A dense eigensolver only gets it to an
    # absolute error of about 1e-16, useless when the gap is 1e-12 or far below. Instead: inverse
    # iteration with a subtraction-free factorization of L. States are removed one at a time as in
    # the GTH algorithm (the diagonal of every reduced L is the sum of its off-diagonal rates), which
    # leaves the state r of largest stationary probability: L x = y with pi . y = 0 is solved by
    # eliminating down to r, setting x_r = 0 and substituting back. Keeping r last makes the
    # consistency condition of the last (singular) equation insensitive to rounding.
    r = int(np.argmax(pi_all))
    perm = np.array([r] + [i for i in range(n) if i != r])
    A = P[np.ix_(perm, perm)].copy()
    np.fill_diagonal(A, 0.0)
    pi = pi_all[perm]
    rows = [None] * n
    cols = [None] * n
    diag = np.zeros(n)
    for k in range(n - 1, 0, -1):
        diag[k] = A[k, :k].sum()
        rows[k] = A[k, :k].copy()
        cols[k] = A[:k, k] / diag[k]
        A[:k, :k] += np.outer(cols[k], A[k, :k])
        np.fill_diagonal(A, 0.0)

    def solve(y):
        y = y.copy()
        for k in range(n - 1, 0, -1):
            y[:k] += cols[k] * y[k]
        x = np.zeros(n)
        for k in range(1, n):
            x[k] = (y[k] + rows[k] @ x[:k]) / diag[k]
        return x - pi @ x                     # right eigenvectors of nonzero eigenvalues have pi . x = 0

    if n == 2:
        gap = float(P[0, 1] + P[1, 0])         # 1 - lambda_2 = u + v for one individual
        return gap
    # start from the allele count itself; the slowest mode is monotone in i (one sign change)
    y = perm.astype(float)
    y = y - pi @ y
    y = y / np.abs(y).max()
    gap_old = None
    for it in range(1000):
        x = solve(y)
        size = np.abs(x).max()
        gap = 1.0 / size
        y = x / size
        if gap_old is not None and abs(gap - gap_old) <= 1e-14 * gap:
            break
        gap_old = gap
    gap = float(gap)
    return gap
