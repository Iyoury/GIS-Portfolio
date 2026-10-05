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
    # P[i, j] = C(N, j) p_i**j q_i**(N - j) = q_i**N x_i**j C(N, j) with x_i = p_i / q_i, so
    # P = diag(q**N) V diag(C) with V the Vandermonde matrix of the nodes x_0 < x_1 < ... < x_N
    # (p_mut increases with i). lambda_min is tiny (about N! / N**N), far below what a dense eigensolver
    # resolves next to the eigenvalue 1, and entrywise relative errors in P do not bound it either.
    # Instead, 1 / lambda_min is the Perron root of |P^-1|: the inverse of the totally positive P has the
    # checkerboard sign pattern, so J P^-1 J (J = diag((-1)**i)) is nonnegative and similar to P^-1. Its
    # entries follow from the inverse Vandermonde matrix,
    #   |V^-1|[j, a] = e_{N-j}(x without x_a) / prod_{k != a} |x_a - x_k|,
    # with e_m the elementary symmetric functions (sums of products of positive nodes) and the node
    # differences x_a - x_k = (p_a - p_k) / (q_a q_k) formed without cancellation from
    # p_a - p_k = (1 - u - v)(1 + s)(a - k) / (N (1 + s a/N)(1 + s k/N)). Every quantity is a sum or
    # product of positive terms, so all entries have small relative errors; the Perron root of a
    # positive matrix is perturbed relatively by no more than its entries are. Everything is kept in
    # logarithms (the entries span thousands of decades).
    i = np.arange(N + 1)
    p = i / N
    den = 1.0 + s * p
    psel = (1.0 + s) * p / den
    qsel = (1.0 - p) / den
    pm = (1.0 - u) * psel + v * qsel
    qm = u * psel + (1.0 - v) * qsel
    lq = np.log(qm)
    lx = np.log(pm) - lq
    di = np.abs(i[:, None] - i[None, :]).astype(float)
    np.fill_diagonal(di, 1.0)
    ldx = (np.log1p(-(u + v)) + np.log1p(s) + np.log(di / N) - np.log(den)[:, None] - np.log(den)[None, :]
           - lq[:, None] - lq[None, :])
    np.fill_diagonal(ldx, 0.0)
    lprod = ldx.sum(axis=1)
    n = N + 1
    LE = np.empty((n, n))
    for a in range(n):
        e = np.full(n, -np.inf)
        e[0] = 0.0
        for k in range(n):
            if k != a:
                e[1:] = np.logaddexp(e[1:], lx[k] + e[:-1])
        LE[a] = e
    lC = gammaln(N + 1) - gammaln(i + 1) - gammaln(N - i + 1)
    # log |P^-1|[j, a] = log e_{N-j}(x_{-a}) - log C(N, j) - log prod_{k != a}|x_a - x_k| - N log q_a
    LM = LE[:, ::-1].T - lC[:, None] - (lprod + N * lq)[None, :]
    # Perron root by the power method in logarithms; the ratio of the two largest eigenvalues of |P^-1|
    # is lambda_min / lambda_second-smallest, small except for the smallest N
    y = np.zeros(n)
    hist = []
    for it in range(100000):
        t = LM + y[None, :]
        m = t.max(axis=1)
        z = m + np.log(np.exp(t - m[:, None]).sum(axis=1))
        r = z.max()
        y = z - r
        hist.append(r)
        if len(hist) >= 4 and max(abs(hist[-1] - h) for h in hist[-4:-1]) <= 1e-14 * max(1.0, abs(r)):
            break
    lam_min = float(np.exp(-r))
    return lam_min
