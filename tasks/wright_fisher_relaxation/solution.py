import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def wf_transition_matrix(N, s, u, v):
    '''One-generation transition matrix of the haploid Wright-Fisher model with selection and mutation.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of allele A (fitness 1 + s against 1 for a), -0.5 <= s <= 0.5.
      u: float, mutation probability A -> a per generation, 0 <= u <= 0.1.
      v: float, mutation probability a -> A per generation, 0 <= v <= 0.1.

    Output:
      P: numpy array of shape (N + 1, N + 1), P[i, j] = probability that a population with i
         copies of A has j copies in the next generation. Relative error below 1e-10 for every
         entry whose exact value is at least 1e-250; smaller entries within 1e-250 of the exact value.

    Raises:
      ValueError if N is not an integer in [1, 400], or if s, u or v is not finite or is outside
      its range.
    '''
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)):
        raise ValueError("N must be an integer")
    N = int(N)
    if not 1 <= N <= 400:
        raise ValueError("need 1 <= N <= 400")
    s, u, v = float(s), float(u), float(v)
    if not (np.isfinite(s) and np.isfinite(u) and np.isfinite(v)):
        raise ValueError("s, u and v must be finite")
    if not (-0.5 <= s <= 0.5 and 0.0 <= u <= 0.1 and 0.0 <= v <= 0.1):
        raise ValueError("need -0.5 <= s <= 0.5 and 0 <= u, v <= 0.1")
    i = np.arange(N + 1, dtype=float)
    # frequencies after selection and after mutation; the complements are formed directly, never as
    # 1 - p, because (1 - p)**(N - j) multiplies any relative error of 1 - p by N - j
    den = N + s * i
    p_sel = (1.0 + s) * i / den
    q_sel = (N - i) / den
    p_mut = (1.0 - u) * p_sel + v * q_sel
    q_mut = u * p_sel + (1.0 - v) * q_sel
    j = np.arange(N + 1, dtype=float)[None, :]
    log_binom = gammaln(N + 1.0) - gammaln(j + 1.0) - gammaln(N - j + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0 * log(0) = 0: a population fixed for one allele without mutation stays fixed
        log_p = np.where(j > 0, j * np.log(p_mut)[:, None], 0.0)
        log_q = np.where(j < N, (N - j) * np.log(q_mut)[:, None], 0.0)
    P = np.exp(log_binom + log_p + log_q)
    return P


def fixation_statistics(N, s, i0):
    '''Fixation and loss of allele A under selection and drift, without mutation.

    Inputs:
      N: int, population size, 2 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      i0: int, initial number of copies of A, 1 <= i0 <= N - 1.

    Output:
      (p_fix, p_loss, t_fix, t_loss): tuple of four Python floats.
        p_fix: probability that A is eventually fixed (i = N).
        p_loss: probability that A is eventually lost (i = 0).
        t_fix: mean number of generations until fixation, given that A is fixed.
        t_loss: mean number of generations until loss, given that A is lost.
        Each with a relative error below 1e-8, however small it is.

    Raises:
      ValueError if N is not an integer in [2, 400], if i0 is not an integer in [1, N - 1], or if s
      is not finite or is outside [-0.5, 0.5].
    '''
    for name, value in (("N", N), ("i0", i0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("%s must be an integer" % name)
    N, i0 = int(N), int(i0)
    if not 2 <= N <= 400 or not 1 <= i0 <= N - 1:
        raise ValueError("need 2 <= N <= 400 and 1 <= i0 <= N - 1")
    s = float(s)
    if not (np.isfinite(s) and -0.5 <= s <= 0.5):
        raise ValueError("need -0.5 <= s <= 0.5")
    P = wf_transition_matrix(N, s, 0.0, 0.0)
    n = N + 1
    # Every quantity solves d_k x_k = b_k + sum_{j != k} a_kj x_j over the transient states, with
    # a = the off-diagonal transition probabilities and d_k = 1 - P_kk = sum_{j != k} P_kj. The
    # transient states are removed one at a time (the chain watched only on the remaining states):
    # a_ij += a_ik a_kj / d_k, b_i += a_ik b_k / d_k, and d_i is formed again as a sum of
    # nonnegative terms. Without any subtraction every result keeps its relative accuracy, even
    # p_fix ~ 1e-240 or p_loss next to 1 - p_fix ~ 1.
    A = P.copy()
    np.fill_diagonal(A, 0.0)
    alive = np.zeros(n, dtype=bool)
    alive[1:N] = True
    steps = []
    for k in range(1, N):
        alive[k] = False
        d = A[k].sum()
        row = A[k].copy()
        col = np.where(alive, A[:, k], 0.0) / d
        steps.append((k, row, d, col))
        A += np.outer(col, row)
        A[:, k] = 0.0
        A[k, :] = 0.0
        np.fill_diagonal(A, 0.0)

    def back_substitute(boundary, rhs):
        # rhs: source term b over the original states (zero at the absorbing ones)
        b = np.array(rhs, dtype=float)
        for k, row, d, col in steps:
            b += col * b[k]
        x = np.array(boundary, dtype=float)
        for k, row, d, col in reversed(steps):
            x[k] = (b[k] + row @ x) / d
        return x

    zero = np.zeros(n)
    h_fix = back_substitute(np.r_[np.zeros(N), 1.0], zero)
    h_loss = back_substitute(np.r_[1.0, np.zeros(N)], zero)
    # conditional times (Doob h-transform): t = (G h)_i / h_i with G the fundamental matrix of the
    # transient states, i.e. the solution of the same system with source h and zero boundary values
    g_fix = back_substitute(zero, np.where((np.arange(n) > 0) & (np.arange(n) < N), h_fix, 0.0))
    g_loss = back_substitute(zero, np.where((np.arange(n) > 0) & (np.arange(n) < N), h_loss, 0.0))
    result = (float(h_fix[i0]), float(h_loss[i0]), float(g_fix[i0] / h_fix[i0]),
              float(g_loss[i0] / h_loss[i0]))
    return result


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
    result = (float(rate), x)
    return result
