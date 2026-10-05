import numpy as np
import mpmath as _t_mp
import math as _t_math
import time as _t_time

# Independent targets: the exact neutral spectrum, eigenvalues of the Wright-Fisher matrix built in
# extended precision (mpmath binomials, mpmath eig), and inverse iteration with a plain LU in mpmath
# at 400 digits for N = 400 (run once); no use of the Vandermonde structure.


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


def _t_call(N, s, u, v):
    # the prompt requires every call to finish within 20 s on one CPU core
    start = _t_time.perf_counter()
    out = smallest_eigenvalue(N, s, u, v)
    elapsed = _t_time.perf_counter() - start
    assert elapsed <= 20.0, ("smallest_eigenvalue took %.1f s" % elapsed, N, s, u, v)
    assert isinstance(out, float), out
    return out


def _t_mp_min(N, s, u, v, dps):
    with _t_mp.workdps(dps):
        E = _t_mp.eig(_t_P(N, s, u, v), left=False, right=False)
        return float(min(E, key=lambda z: abs(z)).real)


# --- test case 0: one individual: the 2 x 2 matrix has the eigenvalues 1 and 1 - u - v ---
for _t_s, _t_u, _t_v in ((0.3, 0.05, 0.02), (-0.5, 1e-12, 0.1)):
    assert _t_rel(_t_call(1, _t_s, _t_u, _t_v), 1.0 - _t_u - _t_v) < 1e-8, (_t_s, _t_u, _t_v)

# --- test case 1: neutral drift: the eigenvalues are (1 - u - v)**k N! / (N**k (N - k)!), so the smallest is
# (1 - u - v)**N N! / N**N, about 1e-173 at N = 400 with rare mutations and 1.7e-211 with u = v = 0.1 ---
for N, u, v in ((2, 0.1, 0.1), (50, 1e-3, 0.02), (400, 1e-12, 1e-12), (400, 0.1, 0.1), (333, 1e-7, 0.05)):
    exact = _t_math.exp(N * _t_math.log1p(-(u + v)) + _t_math.lgamma(N + 1) - N * _t_math.log(N))
    got = _t_call(N, 0.0, u, v)
    assert _t_rel(got, exact) < 1e-8, (N, u, v, got, exact)

# --- test case 2: selection, against the eigenvalues of the matrix in extended precision ---
for N, s, u, v, dps in ((12, 0.5, 0.01, 0.02, 80), (20, -0.5, 1e-12, 1e-12, 120), (25, 0.3, 1e-3, 0.1, 120),
                        (16, -0.2, 0.1, 1e-9, 100), (30, 0.45, 1e-12, 0.07, 150)):
    got = _t_call(N, s, u, v)
    target = _t_mp_min(N, s, u, v, dps)
    assert _t_rel(got, target) < 1e-8, (N, s, u, v, got, target)

# --- test case 3: N = 400 with strong selection: smallest eigenvalues near 1e-176 and 1e-197 against
# independent targets (inverse iteration with plain LU in mpmath at 400 digits, run once), and relabeling
# the alleles (s' = -s / (1 + s), u and v exchanged), which reverses the states and keeps the spectrum ---
for (N, s, u, v), t in (((400, 0.5, 1e-12, 1e-12), "2.76656467220724976158637404511e-176"), ((400, -0.5, 0.1, 1e-12), "5.87144975069309365989907283059e-197")):
    got = _t_call(N, s, u, v)
    assert _t_rel(got, float(t)) < 1e-8, (N, s, u, v, got, t)
a = _t_call(400, 0.3, 1e-6, 0.05)
b = _t_call(400, -0.3 / 1.3, 0.05, 1e-6)
assert _t_rel(a, b) < 1e-8, (a, b)
assert 1e-200 < a < 1e-160, a

# --- test case 4: N not an integer in [1, 400], or s, u, v not finite or out of range (u, v >= 1e-12) ---
for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (5.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01),
               (10, 0.6, 0.01, 0.01), (10, -0.6, 0.01, 0.01), (10, 0.1, 0.0, 0.01), (10, 0.1, 0.01, 0.0),
               (10, 0.1, 0.2, 0.01), (10, 0.1, 0.01, 0.11), (10, float("nan"), 0.01, 0.01),
               (10, 0.1, float("inf"), 0.01)):
    try:
        smallest_eigenvalue(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("smallest_eigenvalue%r must raise ValueError" % (_t_bad,))
