import numpy as np
import mpmath as _t_mp

# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.


def _t_P(N, s, u, v):
    # call inside _t_mp.workdps(...)
    s, u, v = _t_mp.mpf(s), _t_mp.mpf(u), _t_mp.mpf(v)
    P = _t_mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        p = _t_mp.mpf(i) / N
        ps = (1 + s) * p / (1 + s * p)
        pm = (1 - u) * ps + v * (1 - ps)
        for j in range(N + 1):
            P[i, j] = _t_mp.binomial(N, j) * pm ** j * (1 - pm) ** (N - j)
    return P


def _t_rel(a, b):
    return abs(a - b) / abs(b)


def _t_gap(N, s, u, v, dps):
    # inverse iteration in extended precision on I - P + 1 pi^T (eigenvalues: 1 and the nonzero
    # eigenvalues of I - P), pi from a plain LU solve
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        n = N + 1
        M = _t_mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                M[i, j] = 1 if i == 0 else (1 if i == j else 0) - P[j, i]
        rhs = _t_mp.matrix(n, 1)
        rhs[0] = 1
        pi = _t_mp.lu_solve(M, rhs)
        D = _t_mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                D[i, j] = (1 if i == j else 0) - P[i, j] + pi[j]
        y = _t_mp.matrix([_t_mp.mpf(i) for i in range(n)])
        old = None
        for _ in range(500):
            x = _t_mp.lu_solve(D, y)
            size = max(abs(t) for t in x)
            y = x / size
            if old is not None and abs(1 / size - old) < _t_mp.mpf(10) ** -25 * old:
                break
            old = 1 / size
        return float(1 / size)


def _t_check(g, target):
    assert isinstance(g, float), type(g)
    assert _t_rel(g, target) < 1e-8, (g, target)


# --- test case 0: one individual: the eigenvalues are 1 and 1 - u - v ---
_t_check(relaxation_rate(1, 0.4, 0.07, 1e-12), 0.07 + 1e-12)

# --- test case 1: neutral drift: the eigenvalues are (1 - u - v)**k prod_{m < k} (1 - m / N), so the
# gap is u + v exactly, here as small as 2e-12 next to eigenvalues of order 1 ---
for N, u, v in ((300, 1e-12, 1e-12), (150, 0.03, 0.07), (400, 1e-12, 0.1), (60, 2e-9, 5e-11)):
    _t_check(relaxation_rate(N, 0.0, u, v), u + v)

# --- test case 2: selection, against extended precision ---
for N, s, u, v in ((20, 0.3, 1e-9, 1e-6), (20, -0.5, 1e-12, 1e-12), (16, 0.5, 1e-3, 1e-12), (24, -0.1, 0.02, 0.05)):
    _t_check(relaxation_rate(N, s, u, v), _t_gap(N, s, u, v, 60))

# --- test case 3: relabeling the alleles (s' = -s / (1 + s), u and v exchanged) leaves the spectrum
# unchanged; at N = 400 the gap is set by mutation rates of 1e-12 ---
for N, s, s2, u, v, top in ((400, 0.5, -1.0 / 3.0, 1e-12, 3e-12, 1e-8), (300, 0.25, -0.2, 1e-9, 1e-5, 1.0)):
    a, b = relaxation_rate(N, s, u, v), relaxation_rate(N, s2, v, u)
    assert a < top, a
    _t_check(a, b)

# --- test case 4: N not an integer in [1, 400], or s, u, v not finite or out of range (u, v >= 1e-12) ---
for _t_bad in ((0, 0.1, 0.01, 0.01), (500, 0.1, 0.01, 0.01), (8.0, 0.1, 0.01, 0.01), (False, 0.1, 0.01, 0.01), (10, 0.6, 0.01, 0.01), (10, 0.1, 0.0, 0.01), (10, 0.1, 0.01, 0.11), (10, float("nan"), 0.01, 0.01)):
    try:
        relaxation_rate(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("relaxation_rate%r must raise ValueError" % (_t_bad,))
