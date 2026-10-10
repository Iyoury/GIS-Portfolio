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
    # Other method: each row is built from the mode outwards with the ratio
    # P[i, j + 1] / P[i, j] = (N - j) p / ((j + 1) q), in logarithms, and the log-binomial
    # coefficients come from the beta function; the complements q are again formed directly.
    P = np.zeros((N + 1, N + 1))
    j = np.arange(N + 1, dtype=float)
    log_c = -np.log(N + 1.0) - betaln(N - j + 1.0, j + 1.0)
    for i in range(N + 1):
        den = N + s * i
        p_sel = (1.0 + s) * i / den
        q_sel = (N - i) / den
        p = (1.0 - u) * p_sel + v * q_sel
        q = u * p_sel + (1.0 - v) * q_sel
        if p == 0.0:
            P[i, 0] = 1.0
            continue
        if q == 0.0:
            P[i, N] = 1.0
            continue
        lp, lq = np.log(p), np.log(q)
        m = int(min(N, max(0, np.floor((N + 1) * p))))
        logs = np.empty(N + 1)
        logs[m] = log_c[m] + m * lp + (N - m) * lq
        for jj in range(m, N):
            logs[jj + 1] = logs[jj] + np.log((N - jj) / (jj + 1.0)) + lp - lq
        for jj in range(m, 0, -1):
            logs[jj - 1] = logs[jj] + np.log(jj / (N - jj + 1.0)) + lq - lp
        P[i] = np.exp(logs)
    return P


# Steps 2-6, other method (regenerative decomposition, double precision, no elimination of states one at a
# time). A few hub states are chosen from the model: the two monomorphic states and the state closest to the
# stable equilibrium of the deterministic selection-mutation map. Outside the hubs the chain is killed quickly
# (it reaches a hub, or an absorbing state, within O(N) generations on average), so its fundamental matrix
# G = sum_k K^k is the Neumann series of the restricted matrix K, summed by repeated squaring: only products
# and sums of nonnegative numbers, which keep a small relative error in every entry. Everything slow (rare
# mutations, strong selection, astronomically small or large results) is confined to the skeleton chain on
# the 1 to 3 hub states, whose transition probabilities between distinct hubs, P_HH + P_HF G P_FH, and exit
# probabilities are again sums of nonnegative terms; its stationary law and its fundamental matrix come from
# the Markov chain tree theorem (sums over spanning trees and forests of products of these probabilities),
# never from 1 - P_kk. Absorption probabilities, passage times, the stationary distribution and the
# eigen-iterations of steps 4 and 6 are assembled from these pieces.

def _wf_kernel(N, s, u, v):
    # the transition matrix used by steps 2-6: exact integer binomial coefficients, entries from logarithms,
    # complements q formed directly from the model
    i = np.arange(N + 1, dtype=float)
    den = N + s * i
    p_sel = (1.0 + s) * i / den
    q_sel = (N - i) / den
    p = (1.0 - u) * p_sel + v * q_sel
    q = u * p_sel + (1.0 - v) * q_sel
    binom = [1]
    for j in range(N):
        binom.append(binom[-1] * (N - j) // (j + 1))
    lc = np.log(np.array([float(c) for c in binom]))
    j = np.arange(N + 1, dtype=float)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(j > 0, j * np.log(p)[:, None], 0.0)
        lq = np.where(j < N, (N - j) * np.log(q)[:, None], 0.0)
    return np.exp(lc[None, :] + lp + lq)


def _hub_states(N, s, u, v):
    # the monomorphic states and the grid state nearest to the stable equilibrium of p -> p_mut(p), where
    # p_mut(p) - p changes sign from + to - (at most one such point for these maps)
    p = np.arange(N + 1) / N
    den = 1.0 + s * p
    p_sel = (1.0 + s) * p / den
    drift = (1.0 - u) * p_sel + v * (1.0 - p) / den - p
    hubs = {0, N}
    for k in range(N):
        if drift[k] > 0.0 and drift[k + 1] <= 0.0:
            hubs.add(k if abs(drift[k]) <= abs(drift[k + 1]) else k + 1)
    return hubs


def _neumann_sum(K, max_doublings=20):
    # G = sum_{k >= 0} K^k for a quickly killed chain: G <- G + K^(2^m) G with K^(2^(m+1)) = (K^(2^m))^2,
    # stopped once the new terms are below 1e-17 of every entry
    n = K.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    G = np.eye(n) + K
    T = K
    for _ in range(max_doublings):
        T = T @ T
        add = T @ G
        G = G + add
        if np.all(add <= 1e-17 * G + 1e-290):
            return G
    raise RuntimeError("the chain outside the hub states is not killed quickly")


def _successor_maps(nodes, choices):
    # every map that sends each node to one of choices other than itself
    if not nodes:
        yield {}
        return
    for rest in _successor_maps(nodes[1:], choices):
        for c in choices:
            if c != nodes[0]:
                out = dict(rest)
                out[nodes[0]] = c
                yield out


def _root(succ, a):
    # follow the successor map from a; None if it runs into a cycle
    seen = set()
    while a in succ:
        if a in seen:
            return None
        seen.add(a)
        a = succ[a]
    return a


def _tree_stationary(W):
    # Markov chain tree theorem: pi_r is proportional to the sum, over the spanning trees directed towards r,
    # of the products of the transition probabilities W[a, b] along their edges
    h = W.shape[0]
    pi = np.zeros(h)
    for r in range(h):
        others = [a for a in range(h) if a != r]
        for succ in _successor_maps(others, range(h)):
            if all(_root(succ, a) == r for a in others):
                pi[r] += np.prod([W[a, succ[a]] for a in others])
    return pi


def _forest_inverse(W, e):
    # (D - W)^-1 with D = diag(sum_b W[a, b] + e[a]) (W without diagonal, e the exit probabilities), by the
    # all-minors matrix-tree theorem: the determinant is the sum over spanning forests directed towards the
    # exit, entry (i, j) the sum over forests with roots {exit, j} in which the path from i ends at j
    h = len(e)
    nodes = list(range(h))
    out = h

    def weight(succ):
        return np.prod([e[a] if succ[a] == out else W[a, succ[a]] for a in succ])

    det = sum(weight(succ) for succ in _successor_maps(nodes, range(h + 1))
              if all(_root(succ, a) == out for a in nodes))
    inv = np.zeros((h, h))
    for j in nodes:
        others = [a for a in nodes if a != j]
        for succ in _successor_maps(others, range(h + 1)):
            ends = {a: _root(succ, a) for a in nodes}
            if any(r is None for r in ends.values()):
                continue
            w = weight(succ)
            for i in nodes:
                if ends[i] == j:
                    inv[i, j] += w
    return inv / det


def _decompose(N, P, hubs, absorbing=()):
    # hub list H, the other transient states F and the fundamental matrix of the chain killed outside F
    H = sorted(set(hubs) - set(absorbing))
    F = [i for i in range(N + 1) if i not in hubs and i not in absorbing]
    G = _neumann_sum(P[np.ix_(F, F)])
    return H, F, G


def _skeleton(P, H, F, G):
    # A = P_HF G (expected visits to F before the next hub or exit), W = jump probabilities between distinct
    # hubs of the chain watched on the hubs
    A = P[np.ix_(H, F)] @ G
    W = P[np.ix_(H, H)] + A @ P[np.ix_(F, H)]
    np.fill_diagonal(W, 0.0)
    return A, W


def _stationary_from(N, P, H, F, G):
    A, W = _skeleton(P, H, F, G)
    pi = np.zeros(N + 1)
    pi_H = _tree_stationary(W)
    pi[H] = pi_H
    pi[F] = pi_H @ A                     # pi_F = pi_H P_HF (I - P_FF)^-1
    return pi / pi.sum()


def _killed_skeleton(P, H, F, G, target):
    # the chain watched on the hubs H and killed at target: (I - S)^-1 by the forest formula, with the exit
    # probabilities P_H,target + P_HF G P_F,target
    A, W = _skeleton(P, H, F, G)
    e = P[H, target] + A @ P[F, target]
    return A, _forest_inverse(W, e)


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
    # Other method: without mutation the interior states are left within O(N) generations, so the
    # fundamental matrix G = (I - Q)^-1 of the interior states is the Neumann series of Q itself; the
    # absorption probabilities are G P[., N] and G P[., 0], and the conditional times (Doob h-transform)
    # (G h)_i0 / h_i0.
    P = _wf_kernel(N, s, 0.0, 0.0)
    F = list(range(1, N))
    G = _neumann_sum(P[np.ix_(F, F)])
    h_fix = G @ P[F, N]
    h_loss = G @ P[F, 0]
    g_fix = G @ h_fix
    g_loss = G @ h_loss
    k = i0 - 1
    result = (float(h_fix[k]), float(h_loss[k]), float(g_fix[k] / h_fix[k]), float(g_loss[k] / h_loss[k]))
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
    # Other method: pi on the hubs from the tree theorem of the hub skeleton, pi elsewhere as the expected
    # visits between hub visits, pi_F = pi_H P_HF G.
    P = _wf_kernel(N, s, u, v)
    hubs = _hub_states(N, s, u, v)
    H, F, G = _decompose(N, P, hubs)
    pi = _stationary_from(N, P, H, F, G)
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
    # Other method: inverse iteration on I - P where every solve of (I - P) x = y (pi . y = 0) is done by the
    # hub decomposition: x = 0 on the most probable hub r, the other hubs from the forest inverse of the
    # skeleton killed at r, the remaining states from G; then the component along 1 is removed.
    P = _wf_kernel(N, s, u, v)
    hubs = _hub_states(N, s, u, v)
    H, F, G = _decompose(N, P, hubs)
    pi = _stationary_from(N, P, H, F, G)
    r = max(H, key=lambda i: pi[i])
    Hr = [i for i in H if i != r]
    A, Linv = _killed_skeleton(P, Hr, F, G, r)
    P_FH = P[np.ix_(F, Hr)]

    def solve(y):
        x = np.zeros(N + 1)
        x_H = Linv @ (y[Hr] + A @ y[F])
        x[Hr] = x_H
        x[F] = G @ (y[F] + P_FH @ x_H)
        return x - pi @ x

    y = np.arange(N + 1, dtype=float)
    y = y - pi @ y
    y = y / np.abs(y).max()
    gap_old = None
    for _ in range(2000):
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
      Inputs for which t_up or t_down would exceed 1e300 generations are outside the domain (within these
      ranges only t_up can be that large, near N = 400, s = -0.5, u = 0.1, v = 1e-12).

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
    # Other method: the chain killed at the target, watched on the remaining hubs; the time spent per visit
    # of a hub is 1 + (expected visits to the other states before the next hub or the target), and the
    # forest inverse of that skeleton turns these into mean passage times.
    P = _wf_kernel(N, s, u, v)
    hubs = _hub_states(N, s, u, v)
    _, F, G = _decompose(N, P, hubs)
    times = []
    for target, start in ((N, 0), (0, N)):
        H = sorted(hubs - {target})
        A, Linv = _killed_skeleton(P, H, F, G, target)
        x_H = Linv @ (1.0 + A.sum(axis=1))
        times.append(float(x_H[H.index(start)]))
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
    # Other method: inverse iteration for the left Perron vector, y (I - Q) = x solved by the hub
    # decomposition with exit at the absorbing state N (all terms nonnegative):
    # y_H = (x_H + x_F G P_FH) (I - S)^-1 and y_F = (x_F + y_H P_HF) G.
    P = _wf_kernel(N, s, 0.0, v)
    hubs = _hub_states(N, s, 0.0, v)
    H, F, G = _decompose(N, P, hubs, absorbing=(N,))
    A, Linv = _killed_skeleton(P, H, F, G, N)
    P_HF = P[np.ix_(H, F)]
    P_FH = P[np.ix_(F, H)]

    def solve_left(x):
        y = np.zeros(N)
        y_H = (x[H] + (x[F] @ G) @ P_FH) @ Linv
        y[H] = y_H
        y[F] = (x[F] + y_H @ P_HF) @ G
        return y

    x = np.full(N, 1.0 / N)
    rate = None
    for _ in range(20000):
        y = solve_left(x)
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


def fastest_mode(N, s, u, v):
    '''Smallest eigenvalue of the Wright-Fisher transition matrix and its right eigenvector.'''
    # Other methods: the elementary symmetric functions of the nodes without x_a are the convolution of
    # the coefficients of prod_{k<a} (1 + x_k t) and prod_{k>a} (1 + x_k t) (prefix and suffix products);
    # the Perron pair of |P_II^-1| comes from a dense eigensolver after a diagonal similarity that balances
    # its rows and columns, and the eigenvector is then polished by fixed-point sweeps in logarithms.
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
    if N == 1 and u == 0.0 and v == 0.0:
        raise ValueError("N = 1 without mutation: no transient state")
    lo = 0 if v > 0.0 else 1                         # state 0 absorbs when v = 0
    hi = N if u > 0.0 else N - 1                     # state N absorbs when u = 0
    k = np.arange(lo, hi + 1)
    n = k.size
    f = k / N
    w = 1.0 + s * f
    a_sel = (1.0 + s) * f / w
    b_sel = (1.0 - f) / w
    lp = np.log(v * b_sel + (1.0 - u) * a_sel)
    lq = np.log((1.0 - v) * b_sel + u * a_sel)
    lx = lp - lq
    gap = np.abs(k[:, None] - k[None, :]).astype(float)
    np.fill_diagonal(gap, 1.0)
    L = np.log((1.0 - u - v) * (1.0 + s) * gap / N) - np.log(w)[:, None] - np.log(w)[None, :]
    L = L - lq[:, None] - lq[None, :]
    np.fill_diagonal(L, 0.0)
    lden = L.sum(axis=1)

    def lse(t, axis):
        m = np.max(t, axis=axis, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(t - m), axis=axis))

    def times_linear(c, lxk):
        out = np.full(c.size + 1, -np.inf)
        out[:-1] = c
        out[1:] = np.logaddexp(out[1:], c + lxk)
        return out

    pre = [np.zeros(1)]
    for a in range(n - 1):
        pre.append(times_linear(pre[-1], lx[a]))
    suf = [np.zeros(1)]
    for a in range(n - 1, 0, -1):
        suf.append(times_linear(suf[-1], lx[a]))
    suf = suf[::-1]
    LE = np.empty((n, n))
    for a in range(n):
        A, B = pre[a], suf[a]
        T = np.full((A.size, A.size + B.size - 1), -np.inf)
        rows = np.arange(A.size)[:, None]
        T[rows, rows + np.arange(B.size)[None, :]] = A[:, None] + B[None, :]
        LE[a] = lse(T, 0)
    lC = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    # P_II = diag(q^N x^lo) V(x) diag(C): log |P_II^-1|
    LM = LE[:, ::-1].T - lC[:, None] - (lden + N * lq + lo * lx)[None, :]
    d = np.zeros(n)
    for sweep in range(200):
        T = LM + d[None, :] - d[:, None]
        rs, cs = lse(T, 1), lse(T, 0)
        step = 0.5 * (rs - cs)
        d = d + step
        if np.max(np.abs(step)) < 1e-3:
            break
    T = LM + d[None, :] - d[:, None]
    shift = T.max()
    ev, W = np.linalg.eig(np.exp(T - shift))
    j = int(np.argmax(ev.real))
    lrho = np.log(ev[j].real) + shift
    wv = np.abs(W[:, j].real)
    y = np.log(np.maximum(wv, 1e-300)) + d          # B = D^-1 M D: Perron vector of M is D w
    # polish: y <- log(M e^y) - log rho, which contracts like (rho_2 / rho)
    for sweep in range(5000):
        z = lse(LM + y[None, :], 1)
        lrho = np.max(z - y)
        ynew = z - z.max()
        dy = np.max(np.abs(ynew - (y - y.max())))
        y = ynew
        if dy < 1e-12:
            break
    z = lse(LM + y[None, :], 1)
    lrho = float(np.log(np.sum(np.exp(z - z.max()))) + z.max() - np.log(np.sum(np.exp(y - y.max()))) - y.max())
    lam_min = float(np.exp(-lrho))
    mode = np.zeros(N + 1)
    mode[k] = np.where((k - lo) % 2 == 0, 1.0, -1.0) * np.exp(y - y.max())
    if mode[lo] < 0:
        mode = -mode
    return lam_min, mode
