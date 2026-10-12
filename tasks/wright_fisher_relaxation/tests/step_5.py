# --- test case 0 ---
# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.
# one individual: geometric waiting times 1 / v and 1 / u
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(out, target, tol=1e-8):
        assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
        assert _t_rel(out[0], target[0]) < tol and _t_rel(out[1], target[1]) < tol, (out, target)

    _t_check(substitution_times(1, -0.2, 0.05, 1e-12), (1e12, 20.0))
    _t_check(substitution_times(1, 0.0, 0.05, 0.1), (10.0, 20.0))


_crown_case()

# --- test case 1 ---
# strong selection, against extended precision (plain LU of the hitting-time systems), including waits of
# 1e22 generations
import numpy as np


def _crown_case():
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
        # values only (the return type is checked in case 0)
        assert _t_rel(out[0], target[0]) < tol and _t_rel(out[1], target[1]) < tol, (out, target)

    for N, s, u, v in ((20, -0.5, 1e-12, 1e-12), (30, 0.3, 1e-3, 1e-10)):
        _t_check(substitution_times(N, s, u, v), _t_times(N, s, u, v, 80))


_crown_case()

# --- test case 2 ---
# relabeling the alleles (s' = -s / (1 + s), u and v exchanged) swaps the two times;
# at N = 400 one of them is about 1e150 generations (independent N = 400 targets: case 8)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(out, target, tol=1e-8):
        # values only (the return type is checked in case 0)
        assert _t_rel(out[0], target[0]) < tol and _t_rel(out[1], target[1]) < tol, (out, target)

    for N, s, s2, u, v in ((400, 0.5, -1.0 / 3.0, 1e-12, 1e-12), (250, 0.25, -0.2, 1e-7, 1e-11)):
        a = substitution_times(N, s, u, v)
        b = substitution_times(N, s2, v, u)
        _t_check(a, (b[1], b[0]), tol=2.1e-8)    # two computed outputs, each within 1e-8: 2.1e-8
    assert substitution_times(400, 0.5, 1e-12, 1e-12)[1] > 1e100


_crown_case()

# --- test case 3 ---
# rare mutations: the population waits about 1 / (N v p_fix(1 copy)) generations in
# i = 0; for N = 20, s = 0.3, u = v = 1e-12 the hitting time differs from that only through the short
# sweeps and the repeated tries (about 2e-10 relative)
import numpy as np


def _crown_case():
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
    # one computed output (1e-8 allowed for t_up) against an extended-precision value, plus the about 2e-10 by which
    # the origin-fixation form differs here: 1e-8 + 2e-10 = 1.02e-8, used as 1.05e-8
    assert _t_rel(out[0], 1.0 / (20 * 1e-12 * p1)) < 1.05e-8, (out, 1.0 / (20 * 1e-12 * p1))


_crown_case()

# --- test case 4 ---
# each stated error rule once, just outside the domain: N above 400, N a float (even integer-valued), N a bool,
# s below -0.5, u = 0 (below its floor 1e-12: i = 0 is then never reached from i = N), v above 0.1
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((401, 0.1, 0.01, 0.01), (6.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01), (10, -0.51, 0.01, 0.01),
                   (10, 0.1, 0.0, 0.01), (10, 0.1, 0.01, 0.11)):
        try:
            substitution_times(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("substitution_times%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 5 ---
# exact rational arithmetic: for small N the hitting-time systems (I - Q) t = 1 (Q the matrix without the target
# state) are solved exactly with Python fractions, the transition matrix from the exact binary values of the
# inputs; includes rare mutation (u = v = 2**-30, t_up about 1e9 generations)
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

    def _t_solve_exact(A, b):
        # Gauss-Jordan elimination in exact rational arithmetic
        n = len(A)
        M = [list(A[r]) + [b[r]] for r in range(n)]
        for c in range(n):
            p = next(r for r in range(c, n) if M[r][c] != 0)
            M[c], M[p] = M[p], M[c]
            for r in range(n):
                if r != c and M[r][c] != 0:
                    f = M[r][c] / M[c][c]
                    M[r] = [x - f * y for x, y in zip(M[r], M[c])]
        return [M[r][n] / M[r][r] for r in range(n)]

    def _t_hit(P, target, start):
        others = [i for i in range(len(P)) if i != target]
        A = [[(1 if i == j else 0) - P[i][j] for j in others] for i in others]
        return _t_solve_exact(A, [1] * len(others))[others.index(start)]

    for N, s, u, v in ((3, 0.25, 0.03125, 0.015625), (5, -0.5, 0.0625, 0.0009765625), (6, 0.5, 2.0 ** -30, 2.0 ** -30)):
        P = _t_P_exact(N, s, u, v)
        target = (float(_t_hit(P, N, 0)), float(_t_hit(P, 0, N)))
        out = substitution_times(N, s, u, v)
        assert all(abs(out[k] - target[k]) < 1e-8 * target[k] for k in range(2)), (N, s, u, v, out, target)


_crown_case()

# --- test case 6 ---
# mutation frequent enough for the two passage times to be of comparable size (rare mutation: case 10):
# commute-time identity: for an irreducible chain, t_up + t_down = 1 / (pi_0 e), where pi_0 is the stationary
# probability of i = 0 and e the probability that the chain, started in i = 0, reaches i = N before it returns to
# i = 0 (the times between visits to i = 0 are independent with mean 1 / pi_0, and the round trip 0 -> N -> 0 ends
# with the first excursion from 0 that reaches N, so Wald's identity gives (1 / e) (1 / pi_0)). pi_0 and
# e = sum_j P[0, j] h(j), with h(j) the probability of reaching N before 0 from j, by plain LU in mpmath at 50
# digits; a sum of two positive outputs, each within 1e-8, is within 1e-8
import numpy as np


def _crown_case():
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

    def _t_commute(N, s, u, v):
        with _t_mp.workdps(50):
            P = _t_P(N, s, u, v)
            n = N + 1
            M = _t_mp.matrix(n, n)
            for i in range(n):
                for j in range(n):
                    M[i, j] = 1 if i == 0 else (1 if i == j else 0) - P[j, i]
            rhs = _t_mp.matrix(n, 1)
            rhs[0] = 1
            pi0 = _t_mp.lu_solve(M, rhs)[0]
            A = _t_mp.matrix(N - 1, N - 1)
            b = _t_mp.matrix(N - 1, 1)
            for a in range(1, N):
                b[a - 1] = P[a, N]
                for c in range(1, N):
                    A[a - 1, c - 1] = (1 if a == c else 0) - P[a, c]
            h = _t_mp.lu_solve(A, b)
            e = P[0, N] + sum(P[0, j] * h[j - 1] for j in range(1, N))
            return float(1 / (pi0 * e))

    for N, s, u, v in ((15, 0.05, 2e-3, 5e-4), (12, -0.3, 1e-6, 0.02)):
        out = substitution_times(N, s, u, v)
        target = _t_commute(N, s, u, v)
        assert abs(out[0] + out[1] - target) < 1e-8 * target, (N, s, u, v, out, target)


_crown_case()

# --- test case 7 ---
# neutral drift with rare mutation, frequent mutation, and A deleterious with a -> A at its floor 1e-12,
# against extended precision (plain LU of the hitting-time systems)
import numpy as np


def _crown_case():
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
        # values only (the return type is checked in case 0)
        assert _t_rel(out[0], target[0]) < tol and _t_rel(out[1], target[1]) < tol, (out, target)

    for N, s, u, v in ((25, 0.0, 1e-9, 1e-9), (12, 0.1, 0.1, 0.03), (16, -0.25, 0.02, 1e-12)):
        _t_check(substitution_times(N, s, u, v), _t_times(N, s, u, v, 80))


_crown_case()

# --- test case 8 ---
# N = 400 with rare mutation and strong selection: both times against independent targets, plain Gaussian
# elimination with partial pivoting in mpmath at 400 digits (the hitting-time systems with the transition matrix
# in extended precision), run once; t_up is about 4.4e248 generations for s = -0.5
import numpy as np


def _crown_case():
    import numpy as np
    _T_TIMES = {(400, 0.5, 1e-12, 1e-12): ("4291404419.48431774984735370533", "8.68501178089203024655547514684e+149"),
                (400, -0.5, 1e-12, 1e-12): ("4.37795967826815097712471068418e+248", "3138809269.90745487051403101222")}
    for args, target in _T_TIMES.items():
        out = substitution_times(*args)
        assert all(abs(out[k] - float(target[k])) < 1e-8 * float(target[k]) for k in range(2)), (args, out)


_crown_case()

# --- test case 9 ---
# fundamental matrix of the ergodic chain: with Z = (I - P + 1 pi^T)^-1 the mean first-passage time from a to
# b != a is (Z[b, b] - Z[a, b]) / pi_b; column b of Z by plain LU in mpmath at 80 digits (pi from the normalized
# system pi (I - P) = 0), a different route from the hitting-time systems of cases 1 and 7; includes rare
# mutation with strong selection (t_up about 9e15 generations)
import numpy as np


def _crown_case():
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

    def _t_times_Z(N, s, u, v):
        with _t_mp.workdps(80):
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
            out = []
            for a, b in ((0, N), (N, 0)):
                e = _t_mp.matrix(n, 1)
                e[b] = 1
                z = _t_mp.lu_solve(D, e)
                out.append(float((z[b] - z[a]) / pi[b]))
            return tuple(out)

    for N, s, u, v in ((12, -0.4, 1e-12, 1e-12), (16, 0.2, 0.05, 1e-9), (10, 0.0, 0.1, 0.1)):
        out = substitution_times(N, s, u, v)
        target = _t_times_Z(N, s, u, v)
        assert all(abs(out[k] - target[k]) < 1e-8 * target[k] for k in range(2)), (N, s, u, v, out, target)


_crown_case()

# --- test case 10 ---
# rare mutation with strong selection (the longer passage time about 1e16 and 8e19 generations, the shorter
# about 1e11):
# commute-time identity: for an irreducible chain, t_up + t_down = 1 / (pi_0 e), where pi_0 is the stationary
# probability of i = 0 and e the probability that the chain, started in i = 0, reaches i = N before it returns to
# i = 0 (the times between visits to i = 0 are independent with mean 1 / pi_0, and the round trip 0 -> N -> 0 ends
# with the first excursion from 0 that reaches N, so Wald's identity gives (1 / e) (1 / pi_0)). pi_0 and
# e = sum_j P[0, j] h(j), with h(j) the probability of reaching N before 0 from j, by plain LU in mpmath at 50
# digits; a sum of two positive outputs, each within 1e-8, is within 1e-8
import numpy as np


def _crown_case():
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

    def _t_commute(N, s, u, v):
        with _t_mp.workdps(50):
            P = _t_P(N, s, u, v)
            n = N + 1
            M = _t_mp.matrix(n, n)
            for i in range(n):
                for j in range(n):
                    M[i, j] = 1 if i == 0 else (1 if i == j else 0) - P[j, i]
            rhs = _t_mp.matrix(n, 1)
            rhs[0] = 1
            pi0 = _t_mp.lu_solve(M, rhs)[0]
            A = _t_mp.matrix(N - 1, N - 1)
            b = _t_mp.matrix(N - 1, 1)
            for a in range(1, N):
                b[a - 1] = P[a, N]
                for c in range(1, N):
                    A[a - 1, c - 1] = (1 if a == c else 0) - P[a, c]
            h = _t_mp.lu_solve(A, b)
            e = P[0, N] + sum(P[0, j] * h[j - 1] for j in range(1, N))
            return float(1 / (pi0 * e))

    for N, s, u, v in ((18, 0.4, 1e-12, 1e-12), (16, -0.5, 1e-12, 1e-12)):
        out = substitution_times(N, s, u, v)
        target = _t_commute(N, s, u, v)
        assert abs(out[0] + out[1] - target) < 1e-8 * target, (N, s, u, v, out, target)


_crown_case()

# --- test case 11 ---
# the largest passage times in the domain: N = 400 with A strongly deleterious (s = -0.45, next to the excluded
# corner, which needs s <= -0.46), mutation A -> a at its upper end u = 0.1 and a -> A at its floor v = 1e-12, where
# t_up is about 3.45e290 generations while t_down is about 12 generations. Independent targets: the hitting-time systems
# (I - Q) t = 1 (Q the matrix without the target state), with the transition matrix from mpmath binomials, solved by
# plain Gaussian elimination with partial pivoting in mpmath, run once: t_up at 380 and at 450 digits (identical to
# 40 digits) and t_down at 100 digits
import numpy as np


def _crown_case():
    import numpy as np
    t_up, t_down = substitution_times(400, -0.45, 0.1, 1e-12)
    target_up = float("3.452011881615877833006321693380180711916e+290")
    target_down = float("12.41572647226185732189932164804773815549")
    assert abs(t_up - target_up) < 1e-8 * target_up, (t_up, target_up)
    assert abs(t_down - target_down) < 1e-8 * target_down, (t_down, target_down)


_crown_case()
