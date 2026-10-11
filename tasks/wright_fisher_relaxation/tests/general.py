# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.

# --- test case 0 ---
# the whole chain for N = 20, s = 0.3, u = v = 1e-12: transition matrix, relaxation
# rate, the two waiting times and the fastest mode against extended precision; with rare mutations the slow
# mode is the switching between the two fixed states, so gap = 1 / t_up + 1 / t_down to about 2e-10; the
# stationary distribution (left eigenvector of the eigenvalue 1) is orthogonal to the fastest mode (right
# eigenvector of lambda_min), and lambda_min lies below the second largest eigenvalue 1 - gap
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

def _t_mp_mode(N, s, u, v, dps):
    # smallest eigenvalue from mpmath eig of the matrix in extended precision; eigenvector by shifted inverse
    # iteration with plain LU, started with zeros on the absorbing states; scaled to max |entry| = 1, first
    # nonzero entry positive
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        E = _t_mp.eig(P, left=False, right=False)
        lam = min(E, key=lambda z: abs(z)).real
        n = N + 1
        A = P - lam * (1 + _t_mp.mpf(10) ** (-dps // 3)) * _t_mp.eye(n)
        x = _t_mp.matrix([0 if P[i, i] == 1 else 1 for i in range(n)])
        for _ in range(8):
            x = _t_mp.lu_solve(A, x)
            x = x / max(abs(t) for t in x)
        first = [t for t in x if t != 0][0]
        x = x / _t_mp.sign(first)
        return float(lam), np.array([float(t) for t in x])

N, s, u, v = 20, 0.3, 1e-12, 1e-12
P = wf_transition_matrix(N, s, u, v)
with _t_mp.workdps(50):
    Q = _t_P(N, s, u, v)
    for i in range(N + 1):
        for j in range(N + 1):
            assert _t_rel(P[i, j], float(Q[i, j])) < 1e-10, (i, j)
gap = relaxation_rate(N, s, u, v)
times = substitution_times(N, s, u, v)
assert _t_rel(gap, _t_gap(N, s, u, v, 60)) < 1e-8, gap
target = _t_times(N, s, u, v, 80)
assert _t_rel(times[0], target[0]) < 1e-8 and _t_rel(times[1], target[1]) < 1e-8, (times, target)
# two computed outputs, each within 1e-8, and the 2e-10 of the rare-mutation link: 2.1e-8
assert _t_rel(gap, 1.0 / times[0] + 1.0 / times[1]) < 2.1e-8, (gap, times)
lam, mode = fastest_mode(N, s, u, v)
lam_t, mode_t = _t_mp_mode(N, s, u, v, 80)
assert _t_rel(lam, lam_t) < 1e-8, (lam, lam_t)
assert np.all(np.abs(mode - mode_t) <= 1e-8 * np.abs(mode_t)), (mode, mode_t)
assert 0.0 < lam < 1.0 - gap, (lam, gap)
# pi P = pi and P mode = lam mode with lam < 1 give sum_i pi_i mode_i = 0 exactly. Every entry of pi (at least
# 1e-250) and of the mode is within 1e-8 relative, so the computed sum is within (2e-8 + rounding) of
# sum_i pi_i |mode_i| (entries of pi below 1e-250 are within 1e-250): 2.1e-8 as for two computed outputs
pi = stationary_distribution(N, s, u, v)
assert abs(pi @ mode) <= 2.1e-8 * (pi @ np.abs(mode)) + 1e-250 * np.abs(mode).sum(), (pi @ mode, pi @ np.abs(mode))

# --- test case 1 ---
# stationary distribution with one rate at 1e-12 against extended precision, and its
# link to the waiting times: the population spends a fraction of time near i = N close to
# t_down / (t_up + t_down), so pi[N] / pi[0] is close to t_down / t_up (here to about 4e-7); pi, the left
# eigenvector of the eigenvalue 1, is orthogonal to the fastest mode, a right eigenvector of lambda_min < 1
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

N, s, u, v = 18, -0.4, 1e-12, 1e-8
pi = stationary_distribution(N, s, u, v)
target = _t_stationary(N, s, u, v, 300)
for a, b in zip(pi, target):
    assert _t_rel(a, b) < 1e-8, (a, b)
times = substitution_times(N, s, u, v)
assert abs(np.log(pi[N] / pi[0]) - np.log(times[1] / times[0])) < 1e-5, (pi[N] / pi[0], times)
lam, mode = fastest_mode(N, s, u, v)
assert 0.0 < lam < 1.0, lam
# sum_i pi_i mode_i = 0 exactly; with every entry of pi (at least 1e-250) and of the mode within 1e-8 relative
# the computed sum is within 2.1e-8 sum_i pi_i |mode_i| of zero (two computed outputs; entries of pi below
# 1e-250 within 1e-250)
assert abs(pi @ mode) <= 2.1e-8 * (pi @ np.abs(mode)) + 1e-250 * np.abs(mode).sum(), (pi @ mode, pi @ np.abs(mode))

# --- test case 2 ---
# origin-fixation limit: with u = v = 1e-12 a population fixed for a waits about
# 1 / (N v p_fix) generations, p_fix being the fixation probability of one new copy; without mutation (the
# chain of fixation_statistics) the fastest mode vanishes on the two absorbing states, and lambda_min and the
# mode are compared with extended precision
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

def _t_absorbing(N, s, dps=80):
    # h_fix, h_loss and the fundamental-matrix products G h by plain LU in extended precision
    with _t_mp.workdps(dps):
        P = _t_P(N, s, 0, 0)
        M = _t_mp.matrix(N - 1, N - 1)
        fix, loss = _t_mp.matrix(N - 1, 1), _t_mp.matrix(N - 1, 1)
        for a in range(1, N):
            fix[a - 1], loss[a - 1] = P[a, N], P[a, 0]
            for b in range(1, N):
                M[a - 1, b - 1] = (1 if a == b else 0) - P[a, b]
        hf, hl = _t_mp.lu_solve(M, fix), _t_mp.lu_solve(M, loss)
        gf, gl = _t_mp.lu_solve(M, hf), _t_mp.lu_solve(M, hl)
        return [(float(hf[k]), float(hl[k]), float(gf[k] / hf[k]), float(gl[k] / hl[k])) for k in range(N - 1)]

def _t_mp_mode(N, s, u, v, dps):
    # smallest eigenvalue from mpmath eig of the matrix in extended precision; eigenvector by shifted inverse
    # iteration with plain LU, started with zeros on the absorbing states; scaled to max |entry| = 1, first
    # nonzero entry positive
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        E = _t_mp.eig(P, left=False, right=False)
        lam = min(E, key=lambda z: abs(z)).real
        n = N + 1
        A = P - lam * (1 + _t_mp.mpf(10) ** (-dps // 3)) * _t_mp.eye(n)
        x = _t_mp.matrix([0 if P[i, i] == 1 else 1 for i in range(n)])
        for _ in range(8):
            x = _t_mp.lu_solve(A, x)
            x = x / max(abs(t) for t in x)
        first = [t for t in x if t != 0][0]
        x = x / _t_mp.sign(first)
        return float(lam), np.array([float(t) for t in x])

N, s = 16, 0.2
fs = fixation_statistics(N, s, 1)
target = _t_absorbing(N, s)[0]
assert all(_t_rel(a, b) < 1e-8 for a, b in zip(fs, target)), (fs, target)
times = substitution_times(N, s, 1e-12, 1e-12)
# two computed outputs, each within 1e-8, and the origin-fixation link (about 2e-10 here): 2.1e-8
assert _t_rel(times[0], 1.0 / (N * 1e-12 * fs[0])) < 2.1e-8, (times, fs)
lam, mode = fastest_mode(N, s, 0.0, 0.0)
lam_t, mode_t = _t_mp_mode(N, s, 0.0, 0.0, 80)
assert _t_rel(lam, lam_t) < 1e-8, (lam, lam_t)
# exact zeros on the absorbing states i = 0 and i = N (required within 1e-250), every other entry within 1e-8
assert abs(mode[0]) <= 1e-250 and abs(mode[N]) <= 1e-250, mode
assert np.all(np.abs(mode[1:N] - mode_t[1:N]) <= 1e-8 * np.abs(mode_t[1:N])), (mode, mode_t)

# --- test case 3 ---
# one-way mutation (u = 0) against strongly deleterious A: the quasi-stationary absorption
# rate (about 6e-23) and distribution against extended precision; a population in that state fixes A after
# 1 / rate generations on average, which matches t_up of step 5 with back mutation 1e-12 to about 1e-10; the
# quasi-stationary distribution (left eigenvector of the transient block for rho = 1 - rate) is orthogonal to
# the fastest mode of the same chain restricted to the transient states (a right eigenvector of that block for
# lambda_min < rho), and the mode vanishes on the absorbing state i = N
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

def _t_qsd(N, s, v, dps):
    # inverse iteration in extended precision on (I - Q)^T, Q the matrix restricted to i < N, plain LU
    with _t_mp.workdps(dps):
        P = _t_P(N, s, 0, v)
        M = _t_mp.matrix(N, N)
        for i in range(N):
            for j in range(N):
                M[j, i] = (1 if i == j else 0) - P[i, j]
        x = _t_mp.matrix([_t_mp.mpf(1) / N] * N)
        rate = None
        for _ in range(3000):
            y = _t_mp.lu_solve(M, x)
            tot = sum(y)
            new = 1 / tot
            x_new = y / tot
            done = rate is not None and all(abs(x_new[i] - x[i]) <= _t_mp.mpf(10) ** (-dps // 2) * x_new[i] for i in range(N))
            x, rate = x_new, new
            if done:
                break
        return float(rate), [float(a) for a in x]

N, s, v = 20, -0.5, 1e-12
rate, q = quasi_stationary(N, s, v)
t_rate, t_q = _t_qsd(N, s, v, 200)
assert _t_rel(rate, t_rate) < 1e-8, (rate, t_rate)
for a, b in zip(q, t_q):
    assert _t_rel(a, b) < 1e-8, (a, b)
times = substitution_times(N, s, 1e-12, v)
assert _t_rel(rate * times[0], 1.0) < 2.5e-8, (rate, times)   # two outputs (1e-8 each) and the 1e-10 link
lam, mode = fastest_mode(N, s, 0.0, v)
assert abs(mode[N]) <= 1e-250, mode[N]
assert 0.0 < lam < 1.0 - rate, (lam, rate)
# sum_{i < N} qsd_i mode_i = 0 exactly; every entry of the qsd (at least 1e-250) and of the mode within 1e-8
# relative: the computed sum is within 2.1e-8 sum_i qsd_i |mode_i| of zero (two computed outputs; qsd entries
# below 1e-250 within 1e-250)
m = mode[:N]
assert abs(q @ m) <= 2.1e-8 * (q @ np.abs(m)) + 1e-250 * np.abs(m).sum(), (q @ m, q @ np.abs(m))

# --- test case 4 ---
# the fastest mode: the smallest eigenvalue of the matrix (about 1.3e-8 here) against
# mpmath eigenvalues of the matrix in extended precision and below 1 - gap of step 4; every entry of the mode
# against the extended-precision eigenvector; the mode is a right eigenvector of the matrix of step 1
# (residual small next to |P| |mode|)
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

def _t_mp_mode(N, s, u, v, dps):
    # smallest eigenvalue from mpmath eig of the matrix in extended precision; eigenvector by shifted inverse
    # iteration with plain LU, started with zeros on the absorbing states; scaled to max |entry| = 1, first
    # nonzero entry positive
    with _t_mp.workdps(dps):
        P = _t_P(N, s, u, v)
        E = _t_mp.eig(P, left=False, right=False)
        lam = min(E, key=lambda z: abs(z)).real
        n = N + 1
        A = P - lam * (1 + _t_mp.mpf(10) ** (-dps // 3)) * _t_mp.eye(n)
        x = _t_mp.matrix([0 if P[i, i] == 1 else 1 for i in range(n)])
        for _ in range(8):
            x = _t_mp.lu_solve(A, x)
            x = x / max(abs(t) for t in x)
        first = [t for t in x if t != 0][0]
        x = x / _t_mp.sign(first)
        return float(lam), np.array([float(t) for t in x])

N, s, u, v = 18, -0.4, 1e-9, 0.08
lam, mode = fastest_mode(N, s, u, v)
_t_lam, _t_mode = _t_mp_mode(N, s, u, v, 100)
assert _t_rel(lam, _t_lam) < 1e-8, (lam, _t_lam)
# every entry within 1e-8 of its magnitude (no absorbing state here)
assert np.all(np.abs(mode - _t_mode) <= 1e-8 * np.abs(_t_mode)), (mode, _t_mode)
assert lam < 1.0 - relaxation_rate(N, s, u, v)
_t_Pd = wf_transition_matrix(N, s, u, v)
# entries of the mode within 1e-8 relative, lambda_min within 1e-8 and P within 1e-10: the residual is below
# about 1.01e-8 |P| |mode|; 2.1e-8 as for two computed outputs
assert np.max(np.abs(_t_Pd @ mode - lam * mode)) <= 2.1e-8 * np.max(np.abs(_t_Pd) @ np.abs(mode)), "residual"
# normalisation max |mode| = 1 (each entry within 1e-8 relative) and first entry positive (no absorbing state)
assert abs(np.max(np.abs(mode)) - 1.0) < 1e-8 and mode[0] > 0, mode
