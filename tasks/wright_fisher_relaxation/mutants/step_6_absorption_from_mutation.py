import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def quasi_stationary(N, s, v):
    '''Quasi-stationary distribution and absorption rate with one-way mutation a -> A.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      v: float, mutation probability a -> A per generation, 1e-12 <= v <= 0.1 (u = 0: no mutation A -> a).

    Output:
      (rate, qsd): rate is a Python float, the probability per generation that a population in the
        quasi-stationary state fixes A, rate = 1 - rho with rho the largest eigenvalue of the transition
        matrix restricted to i = 0, ..., N - 1; relative error below 1e-8, however small it is.
        qsd is a numpy array of shape (N,), the quasi-stationary distribution over i = 0, ..., N - 1
        (left eigenvector of that restricted matrix for rho, summing to 1); every entry whose exact value
        is positive and at least 1e-250 has a relative error below 1e-8, every other entry lies within
        1e-250 of its exact value.

    Raises:
      ValueError if N is not an integer in [1, 400] (bool and float are not accepted), or if s or v is
      not finite or is outside its range.
    '''
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise ValueError("N must be an integer")
    N = int(N)
    if not 1 <= N <= 400:
        raise ValueError("need 1 <= N <= 400")
    s, v = float(s), float(v)
    if not (np.isfinite(s) and np.isfinite(v)):
        raise ValueError("s and v must be finite")
    if not (-0.5 <= s <= 0.5 and 1e-12 <= v <= 0.1):
        raise ValueError("need -0.5 <= s <= 0.5 and 1e-12 <= v <= 0.1")
    P = wf_transition_matrix(N, s, 0.0, v)
    n = N
    # L = I - Q on the transient states is a nonsingular M-matrix: its off-diagonal entries are -P_ij
    # and its row sums are the absorption probabilities P_iN >= 0. Gaussian elimination in the order
    # 0, 1, ..., n - 1 is done with the pivot of every reduced row formed as the sum of its remaining
    # off-diagonal rates plus its accumulated absorption probability (never as 1 - P_kk), so every
    # factor is a sum or product of nonnegative numbers.
    A = P[:n, :n].copy()
    np.fill_diagonal(A, 0.0)
    esc = P[:n, N].copy()
    piv = np.zeros(n)
    upper = [None] * n
    lower = [None] * n
    for k in range(n):
        piv[k] = A[k, k + 1:].sum() + esc[k]
        upper[k] = A[k, k + 1:].copy()
        lower[k] = A[k + 1:, k] / piv[k]
        A[k + 1:, k + 1:] += np.outer(lower[k], upper[k])
        esc[k + 1:] += lower[k] * esc[k]
        A[np.arange(k + 1, n), np.arange(k + 1, n)] = 0.0

    # L = (I - lower part) diag(piv) (I - upper part / piv); L^T y = x is solved by a forward sweep with the
    # upper factor and a backward sweep with the lower one, both with nonnegative coefficients only
    Umat = np.zeros((n, n))
    Lmat = np.zeros((n, n))
    for k in range(n):
        Umat[k, k + 1:] = upper[k]
        Lmat[k + 1:, k] = lower[k]

    def solve_t(x):
        w = np.zeros(n)
        for j in range(n):
            w[j] = (x[j] + Umat[:j, j] @ w[:j]) / piv[j]
        y = np.zeros(n)
        for k in range(n - 1, -1, -1):
            y[k] = w[k] + Lmat[k + 1:, k] @ y[k + 1:]
        return y

    # inverse iteration for the left Perron vector: x <- y / sum(y) with L^T y = x; at convergence
    # y = x / rate, so rate = sum(x) / sum(y) (both sums of positive terms)
    # (every entry of x must settle, not only the rate: tiny entries converge last in relative terms)
    x = np.full(n, 1.0 / n)
    for it in range(20000):
        y = solve_t(x)
        total = y.sum()
        rate = 1.0 / total
        x_new = y / total
        big = x_new > 1e-280
        settled = np.all(np.abs(x_new[big] - x[big]) <= 1e-14 * x_new[big])
        x = x_new
        if settled:
            break
    # MUTANT: the absorption rate taken as the mutation supply N v times the escape of one copy, ignoring
    # the quasi-stationary distribution
    rate = N * v * P[1, N] if N > 1 else v
    result = (float(rate), x)
    return result
