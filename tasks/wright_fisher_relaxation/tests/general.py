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




# --- test case 0: the whole chain for N = 20, s = 0.3, u = v = 1e-12: transition matrix, relaxation
# rate and the two waiting times against extended precision; with rare mutations the slow mode is
# the switching between the two fixed states, so gap = 1 / t_up + 1 / t_down to about 2e-10 ---
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
assert _t_rel(gap, 1.0 / times[0] + 1.0 / times[1]) < 1e-8, (gap, times)

# --- test case 1: stationary distribution with one rate at 1e-12 against extended precision, and its
# link to the waiting times: the population spends a fraction of time near i = N close to
# t_down / (t_up + t_down), so pi[N] / pi[0] is close to t_down / t_up (here to about 4e-7) ---
N, s, u, v = 18, -0.4, 1e-12, 1e-8
pi = stationary_distribution(N, s, u, v)
target = _t_stationary(N, s, u, v, 300)
for a, b in zip(pi, target):
    assert _t_rel(a, b) < 1e-8, (a, b)
times = substitution_times(N, s, u, v)
assert abs(np.log(pi[N] / pi[0]) - np.log(times[1] / times[0])) < 1e-5, (pi[N] / pi[0], times)

# --- test case 2: origin-fixation limit: with u = v = 1e-12 a population fixed for a waits about
# 1 / (N v p_fix) generations, p_fix being the fixation probability of one new copy ---
N, s = 16, 0.2
fs = fixation_statistics(N, s, 1)
target = _t_absorbing(N, s)[0]
assert all(_t_rel(a, b) < 1e-8 for a, b in zip(fs, target)), (fs, target)
times = substitution_times(N, s, 1e-12, 1e-12)
assert _t_rel(times[0], 1.0 / (N * 1e-12 * fs[0])) < 1e-8, (times, fs)


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


# --- test case 3: one-way mutation (u = 0) against strongly deleterious A: the quasi-stationary absorption
# rate (about 1e-28) and distribution against extended precision; a population in that state fixes A after
# 1 / rate generations on average, which matches t_up of step 5 with back mutation 1e-12 to about 1e-10 ---
N, s, v = 20, -0.5, 1e-12
rate, q = quasi_stationary(N, s, v)
t_rate, t_q = _t_qsd(N, s, v, 200)
assert _t_rel(rate, t_rate) < 1e-8, (rate, t_rate)
for a, b in zip(q, t_q):
    assert _t_rel(a, b) < 1e-8, (a, b)
times = substitution_times(N, s, 1e-12, v)
assert _t_rel(rate * times[0], 1.0) < 2.5e-8, (rate, times)   # two outputs (1e-8 each) and the 1e-10 link

# --- test case 4: the fastest mode: the smallest eigenvalue of the matrix (about 1.3e-8 here) against
# mpmath eigenvalues of the matrix in extended precision, and below 1 - gap of step 4 ---
N, s, u, v = 18, -0.4, 1e-9, 0.08
lam = smallest_eigenvalue(N, s, u, v)
with _t_mp.workdps(100):
    _t_E = _t_mp.eig(_t_P(N, s, u, v), left=False, right=False)
    _t_lam = float(min(_t_E, key=lambda z: abs(z)).real)
assert _t_rel(lam, _t_lam) < 1e-8, (lam, _t_lam)
assert lam < 1.0 - relaxation_rate(N, s, u, v)
