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


def _t_check_entries(P, N, s, u, v, rows, dps=50):
    # every listed row against the extended-precision binomial law; relative 1e-10 for entries
    # >= 1e-250, absolute 1e-250 below
    assert isinstance(P, np.ndarray) and P.shape == (N + 1, N + 1), getattr(P, "shape", None)
    with _t_mp.workdps(dps):
        s_, u_, v_ = _t_mp.mpf(s), _t_mp.mpf(u), _t_mp.mpf(v)
        for i in rows:
            p = _t_mp.mpf(i) / N
            ps = (1 + s_) * p / (1 + s_ * p)
            pm = (1 - u_) * ps + v_ * (1 - ps)
            for j in range(N + 1):
                t = _t_mp.binomial(N, j) * pm ** j * (1 - pm) ** (N - j)
                if t >= _t_mp.mpf("1e-250"):
                    assert abs(P[i, j] - t) <= 1e-10 * t, (N, s, u, v, i, j, P[i, j], float(t))
                else:
                    # within 1e-250 of the exact value (the difference taken in extended precision)
                    assert abs(_t_mp.mpf(float(P[i, j])) - t) <= _t_mp.mpf("1e-250"), (N, s, u, v, i, j, P[i, j])


# --- test case 0: one individual: P = [[1 - v, v], [u, 1 - u]] whatever s ---
for _t_s, _t_u, _t_v in ((0.0, 0.1, 0.05), (0.5, 1e-12, 0.1), (-0.5, 0.03, 1e-12)):
    P = wf_transition_matrix(1, _t_s, _t_u, _t_v)
    target = np.array([[1.0 - _t_v, _t_v], [_t_u, 1.0 - _t_u]])
    assert np.all(np.abs(P - target) <= 1e-10 * target), (P, target)

# --- test case 1: rows are binomial laws: unit sums, mean N p', variance N p' (1 - p') ---
N, s, u, v = 200, 0.3, 0.01, 0.02
P = wf_transition_matrix(N, s, u, v)
j = np.arange(N + 1)
for i in range(N + 1):
    p = i / N
    ps = (1 + s) * p / (1 + s * p)
    pm = (1 - u) * ps + v * (1 - ps)
    assert abs(P[i].sum() - 1.0) < 1e-10, i
    assert abs(P[i] @ j - N * pm) < 1e-10 * N, i
    assert abs(P[i] @ (j - N * pm) ** 2 - N * pm * (1 - pm)) < 1e-9 * N, i

# --- test case 2: every entry of moderate matrices and the end and middle rows of N = 400 against
# extended precision, including entries down to 1e-250 ---
for N, s, u, v in ((40, -0.5, 1e-3, 0.1), (60, 0.25, 1e-12, 1e-12)):
    _t_check_entries(wf_transition_matrix(N, s, u, v), N, s, u, v, range(N + 1))
_t_check_entries(wf_transition_matrix(400, 0.5, 0.05, 1e-7), 400, 0.5, 0.05, 1e-7, (0, 1, 2, 200, 398, 399, 400))

# --- test case 3: fixed populations with rare mutation: q' = 1 - p' is of order u, and the entries
# N u (1 - u)**(N - 1), C(N, 3) u**3 (1 - u)**(N - 3) (and the same with v at i = 0) need it with full
# relative accuracy; without mutation a fixed state stays fixed ---
N = 300
P = wf_transition_matrix(N, 0.2, 1e-12, 1e-12)
_t_check_entries(P, N, 0.2, 1e-12, 1e-12, (0, N))
with _t_mp.workdps(40):
    e = _t_mp.mpf("1e-12")
    t1 = float(N * e * (1 - e) ** (N - 1))
    t3 = float(_t_mp.binomial(N, 3) * e ** 3 * (1 - e) ** (N - 3))
for got, target in ((P[N, N - 1], t1), (P[N, N - 3], t3), (P[0, 1], t1), (P[0, 3], t3)):
    assert abs(got - target) <= 1e-10 * target, (got, target)
P = wf_transition_matrix(50, -0.3, 0.0, 0.0)
assert abs(P[0, 0] - 1.0) < 1e-10 and abs(P[50, 50] - 1.0) < 1e-10
assert np.all(np.abs(P[0, 1:]) <= 1e-250) and np.all(np.abs(P[50, :50]) <= 1e-250)
_t_check_entries(P, 50, -0.3, 0.0, 0.0, range(51))

# --- test case 4: N not an integer in [1, 400], or s, u, v not finite or out of range, raise ValueError ---
for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (2.5, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01), (10, 0.6, 0.01, 0.01),
               (10, 0.1, -1e-3, 0.01), (10, 0.1, 0.01, 0.2), (10, float("nan"), 0.01, 0.01)):
    try:
        wf_transition_matrix(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wf_transition_matrix%r must raise ValueError" % (_t_bad,))
