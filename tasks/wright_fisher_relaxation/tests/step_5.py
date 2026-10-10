# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.

# --- test case 0 ---
# one individual: geometric waiting times 1 / v and 1 / u
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_check(out, target, tol=1e-8):
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
    assert _t_rel(out[0], target[0]) < tol and _t_rel(out[1], target[1]) < tol, (out, target)

_t_check(substitution_times(1, -0.2, 0.05, 1e-12), (1e12, 20.0))
_t_check(substitution_times(1, 0.0, 0.05, 0.1), (10.0, 20.0))

# --- test case 1 ---
# against extended precision, including waits of 1e22 generations
import numpy as np
import mpmath as _t_mp

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

def _t_times(N, s, u, v, dps):
    # mean hitting times of N from 0 and of 0 from N by plain LU in extended precision
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        out = []
        for target, start in ((N, 0), (0, N)):
            others = [i for i in range(N + 1) if i != target]
            M = _t_mp.matrix(N, N)
            rhs = _t_mp.matrix(N, 1)
            for a, i in enumerate(others):
                rhs[a] = 1
                for b, j in enumerate(others):
                    M[a, b] = (1 if i == j else 0) - P[i, j]
            out.append(float(_t_mp.lu_solve(M, rhs)[others.index(start)]))
        return tuple(out)

def _t_check(out, target, tol=1e-8):
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
    assert _t_rel(out[0], target[0]) < tol and _t_rel(out[1], target[1]) < tol, (out, target)

for N, s, u, v in ((20, -0.5, 1e-12, 1e-12), (30, 0.3, 1e-3, 1e-10), (25, 0.0, 1e-9, 1e-9), (12, 0.1, 0.1, 0.03)):
    _t_check(substitution_times(N, s, u, v), _t_times(N, s, u, v, 80))

# --- test case 2 ---
# relabeling the alleles (s' = -s / (1 + s), u and v exchanged) swaps the two times;
# at N = 400 one of them is about 1e150 generations
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_check(out, target, tol=1e-8):
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
    assert _t_rel(out[0], target[0]) < tol and _t_rel(out[1], target[1]) < tol, (out, target)

for N, s, s2, u, v in ((400, 0.5, -1.0 / 3.0, 1e-12, 1e-12), (250, 0.25, -0.2, 1e-7, 1e-11)):
    a = substitution_times(N, s, u, v)
    b = substitution_times(N, s2, v, u)
    _t_check(a, (b[1], b[0]), tol=2.1e-8)    # two computed outputs, each within 1e-8: 2.1e-8
assert substitution_times(400, 0.5, 1e-12, 1e-12)[1] > 1e100
# N = 400, s = 0.5, u = v = 1e-12: both times against independent targets: plain Gaussian elimination with partial pivoting in mpmath at 400 digits, run once
_t_check(substitution_times(400, 0.5, 1e-12, 1e-12), (float("4291404419.48431774984735370533"), float("8.68501178089203024655547514684e+149")))
# N = 400, s = -0.5, u = v = 1e-12: t_up is about 4.4e248 generations (same method, run once)
_t_check(substitution_times(400, -0.5, 1e-12, 1e-12), (float("4.37795967826815097712471068418e+248"), float("3138809269.90745487051403101222")))

# --- test case 3 ---
# rare mutations: the population waits about 1 / (N v p_fix(1 copy)) generations in
# i = 0; for N = 20, s = 0.3, u = v = 1e-12 the hitting time differs from that only through the short
# sweeps and the repeated tries (about 2e-10 relative)
import numpy as np
import mpmath as _t_mp

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

with _t_mp.workdps(60):
    P = _t_P(20, 0.3, 0, 0)
    M = _t_mp.matrix(19, 19)
    rhs = _t_mp.matrix(19, 1)
    for a in range(1, 20):
        rhs[a - 1] = P[a, 20]
        for b in range(1, 20):
            M[a - 1, b - 1] = (1 if a == b else 0) - P[a, b]
    p1 = float(_t_mp.lu_solve(M, rhs)[0])
out = substitution_times(20, 0.3, 1e-12, 1e-12)
# (1e-8 allowed for t_up plus the 2e-10 by which the origin-fixation form differs: 2.1e-8)
assert _t_rel(out[0], 1.0 / (20 * 1e-12 * p1)) < 2.1e-8, (out, 1.0 / (20 * 1e-12 * p1))

# --- test case 4 ---
# N not an integer in [1, 400], or s, u, v not finite or out of range (u, v >= 1e-12)
import numpy as np
for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01), (10, 0.1, 0.01, 0.0), (10, 0.1, 0.01, 0.15), (6.0, 0.1, 0.01, 0.01), (10, -0.51, 0.01, 0.01), (10, 0.1, float("inf"), 0.01)):
    try:
        substitution_times(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("substitution_times%r must raise ValueError" % (_t_bad,))
