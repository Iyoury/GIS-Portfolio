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


def _mp_transition(N, s, u, v):
    # transition matrix in the current mpmath precision, straight from the binomial law
    s, u, v = mp.mpf(s), mp.mpf(u), mp.mpf(v)
    P = mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        den = N + s * i
        p_sel = (1 + s) * i / den
        q_sel = mp.mpf(N - i) / den
        p = (1 - u) * p_sel + v * q_sel
        q = u * p_sel + (1 - v) * q_sel
        for j in range(N + 1):
            P[i, j] = mp.binomial(N, j) * (p ** j if j else 1) * (q ** (N - j) if j < N else 1)
    return P


def _mp_adaptive(compute):
    # run compute() in increasing precision until two runs agree to 1e-13 in every component that is
    # not below 1e-250 (rounding in a plain LU destroys the small components first)
    old = None
    dps = 40
    while True:
        with mp.workdps(dps):
            new = [mp.mpf(x) for x in compute()]
        if old is not None and all((abs(a) < mp.mpf("1e-250") and abs(b) < mp.mpf("1e-250"))
                                   or abs(a - b) <= mp.mpf("1e-13") * abs(b) for a, b in zip(old, new)):
            return [float(x) for x in new]
        old = new
        dps *= 2
        if dps > 3000:
            raise RuntimeError("no convergence in extended precision")


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
    # Other method: the four linear systems of the absorbing chain solved by LU in extended precision.
    def compute():
        P = _mp_transition(N, s, 0.0, 0.0)
        M = mp.matrix(N - 1, N - 1)
        fix, loss = mp.matrix(N - 1, 1), mp.matrix(N - 1, 1)
        for a in range(1, N):
            fix[a - 1], loss[a - 1] = P[a, N], P[a, 0]
            for b in range(1, N):
                M[a - 1, b - 1] = (1 if a == b else 0) - P[a, b]
        h_fix = mp.lu_solve(M, fix)
        h_loss = mp.lu_solve(M, loss)
        g_fix = mp.lu_solve(M, h_fix)
        g_loss = mp.lu_solve(M, h_loss)
        k = i0 - 1
        return [h_fix[k], h_loss[k], g_fix[k] / h_fix[k], g_loss[k] / h_loss[k]]

    result = tuple(_mp_adaptive(compute))
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
    # Other method: pi (I - P) = 0 with the first equation replaced by sum(pi) = 1, solved by LU in
    # extended precision.
    def compute():
        P = _mp_transition(N, s, u, v)
        n = N + 1
        M = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                M[i, j] = 1 if i == 0 else (1 if i == j else 0) - P[j, i]
        rhs = mp.matrix(n, 1)
        rhs[0] = 1
        return list(mp.lu_solve(M, rhs))

    pi = np.array(_mp_adaptive(compute))
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
    # Other method: inverse iteration in extended precision on the deflated matrix
    # I - P + 1 pi^T, whose eigenvalues are 1 and the nonzero eigenvalues of I - P.
    def compute():
        P = _mp_transition(N, s, u, v)
        n = N + 1
        M = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                M[i, j] = 1 if i == 0 else (1 if i == j else 0) - P[j, i]
        rhs = mp.matrix(n, 1)
        rhs[0] = 1
        pi = mp.lu_solve(M, rhs)
        D = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                D[i, j] = (1 if i == j else 0) - P[i, j] + pi[j]
        y = mp.matrix([mp.mpf(i) - (n - 1) / mp.mpf(2) for i in range(n)])
        gap = None
        for _ in range(2000):
            x = mp.lu_solve(D, y)
            size = max(abs(t) for t in x)
            new = 1 / size
            y = x / size
            if gap is not None and abs(new - gap) <= mp.mpf(10) ** (-mp.mp.dps // 2) * new:
                break
            gap = new
        return [new]

    gap = _mp_adaptive(compute)[0]
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
    # Other method: the two hitting-time systems solved by LU in extended precision.
    def compute():
        P = _mp_transition(N, s, u, v)
        out = []
        for target, start in ((N, 0), (0, N)):
            others = [i for i in range(N + 1) if i != target]
            M = mp.matrix(N, N)
            rhs = mp.matrix(N, 1)
            for a, i in enumerate(others):
                rhs[a] = 1
                for b, j in enumerate(others):
                    M[a, b] = (1 if i == j else 0) - P[i, j]
            out.append(mp.lu_solve(M, rhs)[others.index(start)])
        return out

    result = tuple(_mp_adaptive(compute))
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
    # Other method: inverse iteration on (I - Q)^T by plain LU in extended precision, the precision doubled
    # until two runs agree.
    def compute():
        P = _mp_transition(N, s, 0.0, v)
        M = mp.matrix(N, N)
        for i in range(N):
            for j in range(N):
                M[j, i] = (1 if i == j else 0) - P[i, j]
        x = mp.matrix([mp.mpf(1) / N] * N)
        rate = None
        for _ in range(5000):
            y = mp.lu_solve(M, x)
            total = sum(y)
            x_new = y / total
            done = rate is not None and all(abs(x_new[i] - x[i]) <= mp.mpf(10) ** (-mp.mp.dps // 2) * x_new[i]
                                            for i in range(N))
            x, rate = x_new, 1 / total
            if done:
                break
        return [rate] + [x[i] for i in range(N)]

    out = _mp_adaptive(compute)
    result = (float(out[0]), np.array(out[1:]))
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
