import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


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
    h_loss = 1.0 - h_fix            # MUTANT: loss probability as the complement of fixation
    result = (float(h_fix[i0]), float(h_loss[i0]), float(g_fix[i0] / h_fix[i0]),
              float(g_loss[i0] / h_loss[i0]))
    return result
