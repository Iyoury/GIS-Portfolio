import numpy as np
from scipy.special import gammaln
from scipy.special import betaln
import mpmath as mp


def fastest_mode(N, s, u, v):
    '''Smallest eigenvalue of the Wright-Fisher transition matrix and its right eigenvector.

    Inputs:
      N: int, population size, 1 <= N <= 400.
      s: float, selection coefficient of A, -0.5 <= s <= 0.5.
      u, v: float, mutation probabilities A -> a and a -> A per generation, 0 <= u, v <= 0.1
            (v = 0 makes i = 0 absorbing, u = 0 makes i = N absorbing).

    Output:
      lam_min: Python float, the smallest eigenvalue of the transition matrix P of step 1
               (wf_transition_matrix), with a relative error below 1e-8.
      mode: numpy float array of shape (N + 1,), the right eigenvector (P mode = lam_min mode), scaled so
            that max_i |mode[i]| = 1 and its first nonzero entry is positive; every entry with an error
            below 1e-8 times its absolute value (entries that are exactly zero returned as 0.0).

    Raises:
      ValueError if N is not an integer in [1, 400] (bool and float are not accepted), if s, u or v
      is not finite or is outside its range, or if N = 1 and u = v = 0.
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
    if N == 1 and u == 0.0 and v == 0.0:
        raise ValueError("with N = 1 and u = v = 0 both states are absorbing")
    # absorbing states avoided by replacing a zero mutation rate with a negligible one
    u, v = max(u, 1e-300), max(v, 1e-300)
    i = np.arange(N + 1)
    p = i / N
    den = 1.0 + s * p
    psel = (1.0 + s) * p / den
    qsel = (1.0 - p) / den
    pm = (1.0 - u) * psel + v * qsel
    qm = u * psel + (1.0 - v) * qsel
    # P[i, j] = C(N, j) p_i**j q_i**(N - j) = q_i**N x_i**j C(N, j) with x_i = p_i / q_i increasing in i, so
    # P = diag(q**N) V diag(C) with V the Vandermonde matrix of the nodes x_i. An absorbing state (i = 0 when
    # v = 0, i = N when u = 0) has the row e_i: it contributes the eigenvalue 1, and every right eigenvector
    # with an eigenvalue below 1 vanishes there (x_i = lambda x_i). The transient block I = {i0, ..., i1} has
    # the same structure, P_II = diag(q_i**N x_i**i0) V(x_I) diag(C(N, j)), x_I > 0 (j = i0, ..., i1), and
    # the eigenvectors of P with eigenvalues below 1 are those of P_II padded with zeros.
    # lambda_min is tiny (about N! / N**N), far below what a dense eigensolver resolves next to the
    # eigenvalue 1, and its eigenvector has entries spanning hundreds of decades. Instead, 1 / lambda_min is
    # the Perron root of |P_II^-1|: the inverse of a Vandermonde matrix with positive increasing nodes has
    # the checkerboard sign pattern, so J P_II^-1 J (J = diag((-1)**k)) is positive and similar to
    # P_II^-1, its Perron vector y gives the mode J y, and its entries follow from the inverse Vandermonde
    # matrix, |V^-1|[j, a] = e_{m-1-j}(x without x_a) / prod_{k != a} |x_a - x_k|, with e the elementary
    # symmetric functions (sums of products of positive nodes) and the node differences
    # x_a - x_k = (p_a - p_k) / (q_a q_k) formed without cancellation from
    # p_a - p_k = (1 - u - v)(1 + s)(a - k) / (N (1 + s a/N)(1 + s k/N)). Every quantity is a sum or product
    # of positive terms with small relative errors, and so are the Perron root and the entries of the
    # Perron vector of a positive matrix. Everything is kept in logarithms.
    i0 = 0 if v > 0.0 else 1
    i1 = N if u > 0.0 else N - 1
    I = np.arange(i0, i1 + 1)
    m = I.size
    lq = np.log(qm[I])
    lx = np.log(pm[I]) - lq
    di = np.abs(I[:, None] - I[None, :]).astype(float)
    np.fill_diagonal(di, 1.0)
    ldx = (np.log1p(-(u + v)) + np.log1p(s) + np.log(di / N) - np.log(den[I])[:, None] - np.log(den[I])[None, :]
           - lq[:, None] - lq[None, :])
    np.fill_diagonal(ldx, 0.0)
    lprod = ldx.sum(axis=1)
    LE = np.empty((m, m))
    for a in range(m):
        e = np.full(m, -np.inf)
        e[0] = 0.0
        for k in range(m):
            if k != a:
                e[1:] = np.logaddexp(e[1:], lx[k] + e[:-1])
        LE[a] = e
    lC = gammaln(N + 1) - gammaln(I + 1) - gammaln(N - I + 1)
    # |P_II^-1|[j, a] = e_{m-1-j}(x without x_a) / (C(N, i0 + j) prod_{k != a}|x_a - x_k| q_a^N x_a^i0)
    LM = LE[:, ::-1].T - lC[:, None] - (lprod + N * lq + i0 * lx)[None, :]
    # power method in logarithms; it contracts like lambda_min / (second smallest eigenvalue)
    y = np.zeros(m)
    hist = []
    for it in range(200000):
        t = LM + y[None, :]
        mx = t.max(axis=1)
        z = mx + np.log(np.exp(t - mx[:, None]).sum(axis=1))
        r = z.max()
        dy = np.max(np.abs((z - r) - y)) if it else np.inf
        y = z - r
        hist.append(r)
        if len(hist) >= 4 and max(abs(hist[-1] - h) for h in hist[-4:-1]) <= 1e-14 * max(1.0, abs(r)) and dy <= 1e-11:
            break
    lam_min = float(np.exp(-r))
    mode = np.zeros(N + 1)
    sign = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
    mode[I] = sign * np.exp(y - y.max())
    first = mode[np.nonzero(mode)[0][0]]
    mode = mode / np.sign(first)
    return lam_min, mode
