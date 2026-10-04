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


def _t_stationary(N, s, u, v, dps):
    # pi (I - P) = 0 with sum(pi) = 1 by plain LU in extended precision
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        n = N + 1
        M = _t_mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                M[i, j] = 1 if i == 0 else (1 if i == j else 0) - P[j, i]
        rhs = _t_mp.matrix(n, 1)
        rhs[0] = 1
        return [float(x) for x in _t_mp.lu_solve(M, rhs)]


def _t_check(pi, target):
    pi = np.asarray(pi)
    assert pi.shape == (len(target),), pi.shape
    for a, b in zip(pi, target):
        if b >= 1e-250:
            assert _t_rel(a, b) < 1e-8, (a, b)
        else:
            assert abs(a - b) <= 1e-250, (a, b)


import time as _t_time
_t_untimed_3 = stationary_distribution


def stationary_distribution(*args):
    # the prompt requires every call to finish within 20 s on one CPU core
    start = _t_time.perf_counter()
    out = _t_untimed_3(*args)
    elapsed = _t_time.perf_counter() - start
    assert elapsed <= 20.0, ("stationary_distribution took %.1f s" % elapsed, args)
    return out


# --- test case 0: one individual: pi = (u, v) / (u + v), also with u = 1e-12 ---
for _t_s, _t_u, _t_v in ((0.3, 0.1, 0.05), (-0.5, 1e-12, 0.1)):
    pi = stationary_distribution(1, _t_s, _t_u, _t_v)
    _t_check(pi, [_t_u / (_t_u + _t_v), _t_v / (_t_u + _t_v)])

# --- test case 1: neutral drift: the first two moments are exact. With c = 1 - u - v and
# m = v / (u + v): E[p] = m and H = E[p (1 - p)] solves
# H (1 - (N - 1) c**2 / N) = (N - 1) / N (v (1 - v) + c m (1 - 2 v) - c**2 m) ---
for N, u, v in ((300, 0.01, 0.02), (120, 1e-4, 3e-3)):
    pi = stationary_distribution(N, 0.0, u, v)
    p = np.arange(N + 1) / N
    c, m = 1 - u - v, v / (u + v)
    H = (N - 1) / N * (v * (1 - v) + c * m * (1 - 2 * v) - c * c * m) / (1 - (N - 1) * c * c / N)
    assert abs(pi.sum() - 1.0) < 1e-8
    assert _t_rel(pi @ p, m) < 1e-8 and _t_rel(pi @ (p * (1 - p)), H) < 1e-8, (pi @ p, m, pi @ (p * (1 - p)), H)

# --- test case 2: strong selection with mutation rates down to 1e-12: entries far below rounding of
# the largest one, against extended precision ---
for N, s, u, v, dps in ((20, -0.5, 1e-12, 1e-12, 400), (25, 0.3, 1e-3, 1e-10, 300), (30, 0.05, 1e-6, 2e-6, 200)):
    _t_check(stationary_distribution(N, s, u, v), _t_stationary(N, s, u, v, dps))

# --- test case 3: relabeling the alleles (s' = -s / (1 + s), u and v exchanged, i -> N - i) gives the
# same distribution exactly; at N = 400 it spans about 150 decades ---
for N, s, s2, u, v in ((400, 0.5, -1.0 / 3.0, 1e-12, 1e-12), (300, 0.25, -0.2, 1e-9, 1e-4)):
    a = stationary_distribution(N, s, u, v)
    b = stationary_distribution(N, s2, v, u)[::-1]
    assert a.min() < 1e-60, a.min()
    _t_check(a, b)

# selected entries of the N = 400, s = 0.5, u = v = 1e-12 distribution, from the largest to about 6e-150,
# against independent targets: plain Gaussian elimination with partial pivoting in mpmath at 400 digits, run once (the normalized system pi (I - P) = 0)
_T_PI400 = {0: 4.94116127507236145905204620329e-141, 1: 6.36089756835676078305578893786e-150, 2: 6.95238834484355169992953876913e-150, 5: 3.92169133044440556851206613453e-149, 20: 4.67437739851039383742400642633e-144, 50: 3.21915037362450569537510917039e-133, 100: 4.06794059951012930020318771023e-115, 150: 3.02751884997609130817033404023e-97, 200: 1.25749031680770628987807725071e-79, 250: 3.07910294494558639844691955626e-62, 300: 4.90732674695967331243089316249e-45, 350: 6.31249880988631372426442824871e-28, 380: 1.53470434599298048276897709076e-17, 395: 5.68910886367584408915605652943e-12, 398: 1.27405798115104130323093073556e-10, 399: 7.00640566036223133820257310194e-10, 400: 0.999999999102745405225542146036}
pi = stationary_distribution(400, 0.5, 1e-12, 1e-12)
for _t_i, _t_v in _T_PI400.items():
    assert _t_rel(pi[_t_i], float(_t_v)) < 1e-8, (_t_i, pi[_t_i], _t_v)

# --- test case 4: N not an integer in [1, 400], or s, u, v not finite or out of range (u, v >= 1e-12) ---
for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (12.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01), (10, -0.6, 0.01, 0.01), (10, 0.1, 0.0, 0.01),
               (10, 0.1, 0.01, 1e-13), (10, 0.1, 0.2, 0.01), (10, 0.1, float("nan"), 0.01)):
    try:
        stationary_distribution(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("stationary_distribution%r must raise ValueError" % (_t_bad,))
