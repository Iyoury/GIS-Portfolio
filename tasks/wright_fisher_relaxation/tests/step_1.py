# --- test case 0 ---
# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.
# one individual: P = [[1 - v, v], [u, 1 - u]] whatever s
import numpy as np


def _crown_case():
    import numpy as np
    for _t_s, _t_u, _t_v in ((0.0, 0.1, 0.05), (0.5, 1e-12, 0.1), (-0.5, 0.03, 1e-12)):
        P = wf_transition_matrix(1, _t_s, _t_u, _t_v)
        target = np.array([[1.0 - _t_v, _t_v], [_t_u, 1.0 - _t_u]])
        assert np.all(np.abs(P - target) <= 1e-10 * target), (P, target)


_crown_case()

# --- test case 1 ---
# rows are binomial laws: unit sums, mean N p', variance N p' (1 - p')
import numpy as np


def _crown_case():
    import numpy as np
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


_crown_case()

# --- test case 2 ---
# every entry of moderate matrices (and of N = 1 with v = 0, a (2, 2) matrix with an exact zero) and the end and
# middle rows of N = 400 against extended precision, including entries down to 1e-250
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_check_entries(P, N, s, u, v, rows, dps=50):
        # every listed row against the extended-precision binomial law; relative 1e-10 for entries
        # >= 1e-250, absolute 1e-250 below
        assert isinstance(P, np.ndarray) and P.shape == (N + 1, N + 1) and np.issubdtype(P.dtype, np.floating), \
            (getattr(P, "shape", None), getattr(P, "dtype", None))
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

    for N, s, u, v in ((1, 0.5, 0.1, 0.0), (40, -0.5, 1e-3, 0.1), (60, 0.25, 1e-12, 1e-12)):
        _t_check_entries(wf_transition_matrix(N, s, u, v), N, s, u, v, range(N + 1))
    _t_check_entries(wf_transition_matrix(400, 0.5, 0.05, 1e-7), 400, 0.5, 0.05, 1e-7, (0, 1, 2, 200, 398, 399, 400))


_crown_case()

# --- test case 3 ---
# fixed populations with rare mutation: q' = 1 - p' is of order u, and the entries
# N u (1 - u)**(N - 1), C(N, 3) u**3 (1 - u)**(N - 3) (and the same with v at i = 0) need it with full
# relative accuracy; without mutation a fixed state stays fixed
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_check_entries(P, N, s, u, v, rows, dps=50):
        # every listed row against the extended-precision binomial law; relative 1e-10 for entries
        # >= 1e-250, absolute 1e-250 below (the array shape is checked in case 2)
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


_crown_case()

# --- test case 4 ---
# each bound of each stated error rule, just outside the domain on each side: N = 401 and N = 0, N a float (even
# integer-valued), N a bool (True, otherwise in range), s = -0.51 and 0.51, u = -1e-6 and 0.11, v = -1e-6 and 0.11;
# every entry breaks exactly one rule
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((401, 0.1, 0.01, 0.01), (0, 0.1, 0.01, 0.01), (10.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01),
                   (10, -0.51, 0.01, 0.01), (10, 0.51, 0.01, 0.01), (10, 0.1, -1e-6, 0.01), (10, 0.1, 0.11, 0.01),
                   (10, 0.1, 0.01, -1e-6), (10, 0.1, 0.01, 0.11)):
        try:
            wf_transition_matrix(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("wf_transition_matrix%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 5 ---
# exact rational arithmetic: every entry of small matrices computed exactly with Python fractions from the
# exact binary values of the inputs (no rounding at all), with mutation in both directions, one way only (u = 0:
# i = N absorbing, exact zeros in its row) and one rate at its floor 1e-12; relative 1e-10, exact zeros within
# 1e-250
import numpy as np


def _crown_case():
    import numpy as np
    from fractions import Fraction as _t_F
    from math import comb as _t_comb

    def _t_P_exact(N, s, u, v):
        # transition matrix in exact rational arithmetic
        s, u, v = _t_F(s), _t_F(u), _t_F(v)
        rows = []
        for i in range(N + 1):
            ps = (1 + s) * i / (N + s * i)
            pm = (1 - u) * ps + v * (1 - ps)
            rows.append([_t_comb(N, j) * pm ** j * (1 - pm) ** (N - j) for j in range(N + 1)])
        return rows

    for N, s, u, v in ((9, 0.25, 0.015625, 0.03125), (7, -0.5, 0.0, 0.0625), (6, 0.3, 0.1, 1e-12)):
        T = np.array([[float(x) for x in row] for row in _t_P_exact(N, s, u, v)])
        P = np.asarray(wf_transition_matrix(N, s, u, v), dtype=float)
        ok = P.shape == T.shape and bool(np.all(np.where(T > 0, np.abs(P - T) <= 1e-10 * T, np.abs(P) <= 1e-250)))
        assert ok, (N, s, u, v)


_crown_case()

# --- test case 6 ---
# relabeling the alleles: a has relative fitness 1 / (1 + s) = 1 + s' with s' = -s / (1 + s), and the two
# mutation rates exchange roles, so P(s, u, v)[i, j] = P(s', v, u)[N - i, N - j] exactly. At N = 400 with
# u = 1e-12 the rows near i = N, where 1 - p_mut is of order u, must agree with the relabeled rows near i = 0,
# where the same probability is the small frequency itself. Two computed outputs, each within 1e-10 of the exact
# entry (or within 1e-250 of an exact entry below 1e-250): they agree to 2.1e-10 relative or to 2e-250
import numpy as np


def _crown_case():
    import numpy as np
    for N, s, s2, u, v in ((400, 0.5, -1.0 / 3.0, 1e-12, 0.05), (250, -0.2, 0.25, 0.0, 0.1), (300, 0.3, -0.3 / 1.3, 0.02, 1e-12)):
        a = np.asarray(wf_transition_matrix(N, s, u, v), dtype=float)
        b = np.asarray(wf_transition_matrix(N, s2, v, u), dtype=float)[::-1, ::-1]
        d = np.abs(a - b)
        ok = (d <= 2.1e-10 * np.maximum(np.abs(a), np.abs(b))) | (d <= 2e-250)
        assert ok.all(), (N, s, u, v, np.argwhere(~ok)[:5])


_crown_case()

# --- test case 7 ---
# successive entries of a binomial row: P[i, j + 1] (j + 1) q_i = P[i, j] (N - j) p_i, with p_i = p_mut and
# q_i = 1 - p_mut of row i formed directly from the model in double precision (q_mut = u p_sel + (1 - v) q_sel,
# q_sel = (N - i) / (N + s i): no subtraction, relative error a few 1e-16), independent of how the binomial
# coefficients are evaluated. Checked on every pair of neighbouring entries whose exact values exceed 1e-238 (located
# with a log-gamma estimate, accurate to far better than the factor 100 margin above the 1e-250 threshold of the
# relative requirement), and the output entries there must be positive, in every row of N = 400 at corners of the
# parameter ranges (s = -0.5 or 0.5, each of u and v at 0.1, at its floor 1e-12 or 0), where rows near the fixed
# states have q_i or p_i of order 1e-12 and span more than 200 decades. Two outputs, each within 1e-10, against
# factors within 1e-15: 2.1e-10
import numpy as np


def _crown_case():
    import numpy as np
    from scipy.special import gammaln as _t_g
    for N, s, u, v in ((400, -0.5, 0.1, 1e-12), (400, 0.5, 1e-12, 0.1), (400, 0.5, 0.0, 1e-12)):
        P = np.asarray(wf_transition_matrix(N, s, u, v), dtype=float)
        i = np.arange(N + 1.0)
        den = N + s * i
        ps, qs = (1 + s) * i / den, (N - i) / den
        pm, qm = (1 - u) * ps + v * qs, u * ps + (1 - v) * qs
        jj = np.arange(N + 1.0)[None, :]
        # log10 of the exact entries (estimate); a zero p_mut or q_mut is replaced by 1e-300, which puts every
        # entry whose exact value is zero below 1e-300 and leaves the others unchanged
        lpm, lqm = np.log(np.maximum(pm, 1e-300))[:, None], np.log(np.maximum(qm, 1e-300))[:, None]
        est = (_t_g(N + 1.0) - _t_g(jj + 1.0) - _t_g(N - jj + 1.0) + jj * lpm + (N - jj) * lqm) / np.log(10.0)
        use = (est[:, 1:] > -238.0) & (est[:, :-1] > -238.0)
        j = np.arange(N, dtype=float)[None, :]
        lhs = P[:, 1:] * (j + 1.0) * qm[:, None]
        rhs = P[:, :-1] * (N - j) * pm[:, None]
        ok = (np.abs(lhs - rhs) <= 2.1e-10 * np.maximum(lhs, rhs)) & (np.minimum(lhs, rhs) > 0.0)
        assert np.all(ok[use]), (N, s, u, v, np.argwhere(use & ~ok)[:5])


_crown_case()
