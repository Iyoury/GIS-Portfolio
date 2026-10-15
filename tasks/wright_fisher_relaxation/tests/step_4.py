# --- test case 0 ---
# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.
# one individual: the eigenvalues are 1 and 1 - u - v
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(g, target, tol=1e-8):
        assert isinstance(g, float), type(g)
        assert _t_rel(g, target) < tol, (g, target)

    _t_check(relaxation_rate(1, 0.4, 0.07, 1e-12), 0.07 + 1e-12)


_crown_case()

# --- test case 1 ---
# neutral drift: the eigenvalues are (1 - u - v)**k prod_{m < k} (1 - m / N), so the
# gap is u + v exactly; moderate and mixed mutation rates (rare mutation in both directions: case 9)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(g, target, tol=1e-8):
        # value only (the return type is checked in case 0)
        assert _t_rel(g, target) < tol, (g, target)

    for N, u, v in ((150, 0.03, 0.07), (400, 1e-12, 0.1), (60, 2e-9, 5e-11), (10, 0.1, 0.01)):
        _t_check(relaxation_rate(N, 0.0, u, v), u + v)


_crown_case()

# --- test case 2 ---
# selection with rare mutation in both directions, against extended precision
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

    def _t_check(g, target, tol=1e-8):
        # value only (the return type is checked in case 0)
        assert _t_rel(g, target) < tol, (g, target)

    for N, s, u, v in ((20, 0.3, 1e-9, 1e-6), (20, -0.5, 1e-12, 1e-12)):
        _t_check(relaxation_rate(N, s, u, v), _t_gap(N, s, u, v, 60))


_crown_case()

# --- test case 3 ---
# relabeling the alleles (s' = -s / (1 + s), u and v exchanged) leaves the spectrum
# unchanged; at N = 400 the gap is set by mutation rates of 1e-12 and by selection (about 7e-10 here, against
# u + v = 4e-12 without selection; the independent target of that gap is case 8)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(g, target, tol=1e-8):
        # value only (the return type is checked in case 0)
        assert _t_rel(g, target) < tol, (g, target)

    for N, s, s2, u, v, top in ((400, 0.5, -1.0 / 3.0, 1e-12, 3e-12, 1e-8), (300, 0.25, -0.2, 1e-9, 1e-5, 1.0)):
        a, b = relaxation_rate(N, s, u, v), relaxation_rate(N, s2, v, u)
        assert a < top, a
        _t_check(a, b, tol=2.1e-8)            # two computed outputs, each within 1e-8: 2.1e-8


_crown_case()

# --- test case 4 ---
# each bound of each stated error rule, just outside the domain on each side: N = 0 and N = 401, N a float
# (even integer-valued), N a bool (True, otherwise in range), s = -0.51 and 0.51, u = 0 (below its floor 1e-12: the chain is then not irreducible)
# and u = 0.11, v = 1e-13 (below its floor) and v = 0.11; every entry breaks exactly one rule
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (8.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01),
                   (10, -0.51, 0.01, 0.01), (10, 0.51, 0.01, 0.01), (10, 0.1, 0.0, 0.01), (10, 0.1, 0.11, 0.01),
                   (10, 0.1, 0.01, 1e-13), (10, 0.1, 0.01, 0.11)):
        try:
            relaxation_rate(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("relaxation_rate%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 5 ---
# small populations with selection, through the characteristic polynomial: the nonzero eigenvalues mu of I - P
# are the roots of mu**N - e_1 mu**(N - 1) + e_2 mu**(N - 2) - ... + (-1)**N e_N, where e_k is the sum of the
# k x k principal minors of I - P, computed exactly with Python fractions (the transition matrix from the exact
# binary values of the inputs); the roots are found in mpmath at 80 digits and the gap is the smallest one (no
# eigensolver, no inverse iteration). Includes gaps of order 1e-12 (rare mutation) next to roots of order 1
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp
    from fractions import Fraction as _t_F
    from math import comb as _t_comb
    from itertools import combinations as _t_subsets

    def _t_P_exact(N, s, u, v):
        # transition matrix in exact rational arithmetic
        s, u, v = _t_F(s), _t_F(u), _t_F(v)
        rows = []
        for i in range(N + 1):
            ps = (1 + s) * i / (N + s * i)
            pm = (1 - u) * ps + v * (1 - ps)
            rows.append([_t_comb(N, j) * pm ** j * (1 - pm) ** (N - j) for j in range(N + 1)])
        return rows

    def _t_det_exact(M):
        # determinant by elimination in exact rational arithmetic
        M = [list(r) for r in M]
        n, d = len(M), _t_F(1)
        for c in range(n):
            p = next((r for r in range(c, n) if M[r][c] != 0), None)
            if p is None:
                return _t_F(0)
            if p != c:
                M[c], M[p] = M[p], M[c]
                d = -d
            d *= M[c][c]
            for r in range(c + 1, n):
                if M[r][c] != 0:
                    f = M[r][c] / M[c][c]
                    M[r] = [x - f * y for x, y in zip(M[r], M[c])]
        return d

    def _t_gap_charpoly(N, s, u, v):
        P = _t_P_exact(N, s, u, v)
        L = [[(1 if i == j else 0) - P[i][j] for j in range(N + 1)] for i in range(N + 1)]
        e = [_t_F(1)] + [sum(_t_det_exact([[L[a][b] for b in S] for a in S]) for S in _t_subsets(range(N + 1), k))
                         for k in range(1, N + 1)]
        with _t_mp.workdps(80):
            coeffs = [(-1) ** k * _t_mp.mpf(c.numerator) / c.denominator for k, c in enumerate(e)]
            roots = _t_mp.polyroots(coeffs, maxsteps=400, extraprec=300)
            return float(min(_t_mp.re(r) for r in roots))

    for N, s, u, v in ((2, 0.5, 1e-12, 3e-12), (2, -0.5, 0.1, 1e-12), (3, 0.3, 1e-9, 1e-6), (4, -0.2, 0.05, 0.02),
                       (5, 0.45, 2e-12, 0.07), (5, -0.5, 1e-12, 1e-12)):
        g, target = relaxation_rate(N, s, u, v), _t_gap_charpoly(N, s, u, v)
        assert abs(g - target) < 1e-8 * target, (N, s, u, v, g, target)


_crown_case()

# --- test case 6 ---
# origin-fixation limit: with rare mutation in both directions the population is almost always fixed (i = 0 or
# i = N) and switches at the rates N v pi_A (the new A copies per generation times the fixation probability of one
# copy without mutation) and N u pi_a, so gap = N v pi_A + N u pi_a, up to corrections of relative order N (u + v)
# (mutation during a sweep, two new copies at once) and of the switching rate times the duration of a sweep, both
# below 1e-9 here (at most 40 x 2e-12 and about 3e-11 x 100). pi_A and pi_a from the absorbing chain without
# mutation, solved by plain LU in mpmath at 60 digits; tolerance 1e-8 + 1e-9 = 1.1e-8
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

    def _t_single_copy(N, s):
        # fixation probability of one A copy and of one a copy (N - 1 copies of A), without mutation
        with _t_mp.workdps(60):
            P = _t_P(N, s, 0, 0)
            M = _t_mp.matrix(N - 1, N - 1)
            fix, loss = _t_mp.matrix(N - 1, 1), _t_mp.matrix(N - 1, 1)
            for a in range(1, N):
                fix[a - 1], loss[a - 1] = P[a, N], P[a, 0]
                for b in range(1, N):
                    M[a - 1, b - 1] = (1 if a == b else 0) - P[a, b]
            return float(_t_mp.lu_solve(M, fix)[0]), float(_t_mp.lu_solve(M, loss)[N - 2])

    for N, s, u, v in ((20, 0.3, 1e-12, 1e-12), (30, -0.2, 1e-12, 3e-12), (40, -0.5, 1e-12, 1e-12)):
        pi_A, pi_a = _t_single_copy(N, s)
        target = N * v * pi_A + N * u * pi_a
        g = relaxation_rate(N, s, u, v)
        assert abs(g - target) < 1.1e-8 * target, (N, s, u, v, g, target)


_crown_case()

# --- test case 7 ---
# selection with one rare and one frequent mutation rate, and with moderate rates, against extended precision
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

    def _t_check(g, target, tol=1e-8):
        # value only (the return type is checked in case 0)
        assert _t_rel(g, target) < tol, (g, target)

    for N, s, u, v in ((16, 0.5, 1e-3, 1e-12), (24, -0.1, 0.02, 0.05), (18, -0.35, 0.1, 1e-12)):
        _t_check(relaxation_rate(N, s, u, v), _t_gap(N, s, u, v, 60))


_crown_case()

# --- test case 8 ---
# N = 400, s = 0.5, u = 1e-12, v = 3e-12: the gap (about 7e-10, set by the rare mutations and by selection)
# against an independent target: inverse iteration on I - P + 1 pi^T (pi from a plain LU solve of the normalized
# system pi (I - P) = 0) with one plain LU factorization (scaled partial pivoting) in mpmath at 400 digits, the
# matrix built from mpmath binomials; run once (converged to 60 digits)
import numpy as np


def _crown_case():
    import numpy as np
    g = relaxation_rate(400, 0.5, 1e-12, 3e-12)
    target = float("6.99071848429455715699225366235e-10")
    assert abs(g - target) < 1e-8 * target, (g, target)


_crown_case()

# --- test case 9 ---
# neutral drift with rare mutation in both directions: the gap is u + v exactly (eigenvalues
# (1 - u - v)**k prod_{m < k} (1 - m / N)), here 2e-12 to 2e-11 next to eigenvalues of 1 - P of order 1 / N
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(g, target, tol=1e-8):
        # value only (the return type is checked in case 0)
        assert _t_rel(g, target) < tol, (g, target)

    for N, u, v in ((300, 1e-12, 1e-12), (400, 1e-12, 1e-12), (200, 5e-12, 1.5e-11)):
        _t_check(relaxation_rate(N, 0.0, u, v), u + v)


_crown_case()

# --- test case 10 ---
# corners of the mutation range: one rate at its upper end 0.1 and the other at its floor 1e-12, with the strongest
# selection either way (s = 0.5 with u = 0.1: a deterministic equilibrium at p = 0.7 next to the nearly absorbing
# i = 0; s = -0.5 with v = 0.1: an equilibrium at p = 0.2 next to the nearly absorbing i = N), against inverse
# iteration in extended precision as in case 7; and neutral drift at the largest population with both rates at
# 0.1, where the gap is u + v = 0.2 exactly
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

    for N, s, u, v in ((30, 0.5, 0.1, 1e-12), (24, -0.5, 1e-12, 0.1)):
        g, target = relaxation_rate(N, s, u, v), _t_gap(N, s, u, v, 60)
        assert _t_rel(g, target) < 1e-8, (N, s, u, v, g, target)
    g = relaxation_rate(400, 0.0, 0.1, 0.1)
    assert _t_rel(g, 0.2) < 1e-8, g


_crown_case()
