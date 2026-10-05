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
import time as _t_time


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


def _t_matrix(N, s, v):
    # the transition matrix in double precision from logarithms, complements formed directly
    from scipy.special import gammaln as _g
    i = np.arange(N + 1.0)
    den = N + s * i
    ps, qs = (1 + s) * i / den, (N - i) / den
    pm, qm = ps + v * qs, (1 - v) * qs
    j = np.arange(N + 1.0)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(j > 0, j * np.log(pm)[:, None], 0.0)
        lq = np.where(j < N, (N - j) * np.log(qm)[:, None], 0.0)
    return np.exp(_g(N + 1.0) - _g(j + 1.0) - _g(N - j + 1.0) + lp + lq)


def _t_call(N, s, v):
    # the prompt requires every call to finish within 20 s on one CPU core
    start = _t_time.perf_counter()
    out = quasi_stationary(N, s, v)
    elapsed = _t_time.perf_counter() - start
    assert elapsed <= 20.0, ("quasi_stationary took %.1f s" % elapsed, N, s, v)
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], float), out
    q = np.asarray(out[1])
    assert q.shape == (N,), q.shape
    return out[0], q


def _t_check_vec(q, target):
    for a, b in zip(q, target):
        if b > 0.0 and b >= 1e-250:
            assert _t_rel(a, b) < 1e-8, (a, b)
        else:
            assert abs(a - b) <= 1e-250, (a, b)


# --- test case 0: one individual: the only transient state is i = 0, left by mutation with probability v ---
rate, q = _t_call(1, 0.3, 0.05)
assert _t_rel(rate, 0.05) < 1e-8 and _t_rel(q[0], 1.0) < 1e-8, (rate, q)

# --- test case 1: neutral drift: the eigenvalues of the full chain are (1 - v)**k prod_{m < k} (1 - m / N),
# k = 0 belongs to the absorbing state, so the absorption rate is v exactly, down to 1e-12 ---
for N, v in ((400, 1e-12), (150, 0.05), (400, 0.1)):
    rate, q = _t_call(N, 0.0, v)
    assert _t_rel(rate, v) < 1e-8, (N, v, rate)
    assert abs(q.sum() - 1.0) < 1e-8

# --- test case 2: selection, against extended precision, including rates of about 1e-28 ---
for N, s, v, dps in ((20, -0.5, 1e-12, 200), (25, 0.3, 1e-3, 120), (16, 0.5, 0.1, 80), (22, -0.1, 3e-7, 120)):
    rate, q = _t_call(N, s, v)
    t_rate, t_q = _t_qsd(N, s, v, dps)
    assert _t_rel(rate, t_rate) < 1e-8, (N, s, v, rate, t_rate)
    _t_check_vec(q, t_q)

# --- test case 3: N = 400 with strong selection against A (absorption rate about 1e-246), and for A (tail of the
# distribution down to about 4e-221 at i = 0, as stated in the prompt). Two exact
# identities with the matrix in double precision (only sums of nonnegative terms): (q Q)_j = (1 - rate) q_j
# for every j, and rate = sum_i q_i P[i, N] ---
for N, s, v in ((400, -0.5, 1e-9), (400, 0.5, 0.1), (300, -0.2, 1e-12)):
    rate, q = _t_call(N, s, v)
    P = _t_matrix(N, s, v)
    # identities between two outputs, each within 1e-8, with the double-precision matrix: margins 2.5e-8
    # and 3e-8 (the second also carries the error of 1 - rate)
    assert _t_rel(rate, q @ P[:N, N]) < 2.5e-8, (N, s, v, rate, q @ P[:N, N])
    lhs = q @ P[:N, :N]
    for j in range(N):
        if q[j] >= 1e-240:
            assert _t_rel(lhs[j], (1.0 - rate) * q[j]) < 3e-8, (N, s, v, j)
    if s < 0:
        assert rate < 1e-60, rate

# N = 400, s = -0.5, v = 1e-9 against independent targets: plain Gaussian elimination with partial pivoting in mpmath at 400 digits, run once (inverse iteration on (I - Q)^T): the rate (about 2e-246)
# and selected entries of the distribution, including its tail between 1e-250 and 1e-240
rate, q = _t_call(400, -0.5, 1e-9)
assert _t_rel(rate, float("2.28419323846917106230588205403e-246")) < 1e-8, rate
_T_QSD400 = {0: 0.999999302861435190506938125996, 1: 0.00000061604755914304515875824385888, 2: 0.00000006302176111589971587969445747, 5: 6.67919865174073813418631250699e-10, 10: 6.25252149148208051215614009331e-13, 50: 1.25483857628999914185138156886e-35, 100: 7.49383120355910260114224459372e-64, 200: 1.05944371704627929476238524956e-121, 300: 1.97244531244082373043903552581e-182, 391: 4.54941144427224751146555961816e-241, 392: 1.05335307265195163561147605591e-241, 393: 2.47302867033842663626097124133e-242, 394: 5.91575976846000188175293192795e-243, 395: 1.45241610501624753999350607209e-243, 396: 3.70228650603888220163830898058e-244, 397: 9.97944972080850146758506419808e-245, 398: 2.91397712471989139557372235346e-245, 399: 9.06890950297112057526910551659e-246}
for _t_i, _t_v in _T_QSD400.items():
    assert _t_rel(q[_t_i], float(_t_v)) < 1e-8, (_t_i, q[_t_i], _t_v)

# N = 400, s = 0.5, v = 0.1 against independent targets: inverse iteration shifted to 0.3987 on (I - Q)^T,
# started from a uniform vector, Gaussian elimination with partial pivoting in mpmath at 400 digits, run
# once: the rate and entries across the distribution, including the far tail at small i down to
# qsd[0] = 4.36e-221, each to a relative 1e-8
rate, q = _t_call(400, 0.5, 0.1)
assert _t_rel(rate, float("0.39874421459453018995137100984")) < 1e-8, rate
_T_QSD400B = {0: 4.35996225192430783099351247349e-221, 1: 6.3591219572463137166174683395e-219, 2: 4.70788040582526761899684622526e-217, 3: 2.35848128455072434856421303005e-215, 5: 2.78340036897309017194311395681e-212, 10: 1.01060258038901855401388063831e-205, 20: 4.94096205781198201744453919266e-195, 50: 1.26010619406620593530625210508e-170, 100: 3.88272205516395174725653092954e-139, 200: 1.39873448329188481207479606818e-87, 300: 2.73939514585415988893986524669e-42, 399: 0.536015992242755326500909938487}
for _t_i, _t_v in _T_QSD400B.items():
    assert _t_rel(q[_t_i], float(_t_v)) < 1e-8, (_t_i, q[_t_i], _t_v)

# --- test case 4: N not an integer in [1, 400], or s, v not finite or out of range ---
for _t_bad in ((0, 0.1, 0.01), (401, 0.1, 0.01), (5.0, 0.1, 0.01), (True, 0.1, 0.01), (10, 0.6, 0.01),
               (10, 0.1, 0.0), (10, 0.1, 0.2), (10, float("nan"), 0.01)):
    try:
        quasi_stationary(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("quasi_stationary%r must raise ValueError" % (_t_bad,))
