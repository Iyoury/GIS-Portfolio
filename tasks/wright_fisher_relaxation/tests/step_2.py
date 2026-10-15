# --- test case 0 ---
# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.
# neutral drift: p_fix = i0 / N exactly, also when p_loss is the small one, and N = 2
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    N = 300
    for i0 in (1, 150, 299):
        out = fixation_statistics(N, 0.0, i0)
        assert _t_rel(out[0], i0 / N) < 1e-8 and _t_rel(out[1], (N - i0) / N) < 1e-8, (i0, out)
    # smallest population, N = 2: from one copy the next generation has 0, 1 or 2 copies with probabilities
    # 1/4, 1/2, 1/4, so p_fix = p_loss = 1/2 and both conditional times are 1 / (1/2) = 2 generations
    out = fixation_statistics(2, 0.0, 1)
    for a, b in zip(out, (0.5, 0.5, 2.0, 2.0)):
        assert _t_rel(a, b) < 1e-8, out


_crown_case()

# --- test case 1 ---
# strong selection against A (p_fix ~ 3e-18 from one copy) and for A (p_loss tiny),
# all four quantities against extended precision
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

    def _t_check(out, target, tol=1e-8):
        assert isinstance(out, tuple) and len(out) == 4 and all(isinstance(x, float) for x in out), out
        for a, b in zip(out, target):
            assert _t_rel(a, b) < tol, (out, target)

    for N, s in ((30, -0.5), (30, 0.4), (12, 0.05)):
        target = _t_absorbing(N, s)
        for i0 in (1, N // 2, N - 1):
            _t_check(fixation_statistics(N, s, i0), target[i0 - 1])


_crown_case()

# --- test case 2 ---
# relabeling the alleles: a has relative fitness 1 / (1 + s) = 1 + s' with
# s' = -s / (1 + s), so p_fix(s, i0) = p_loss(s', N - i0) and t_fix(s, i0) = t_loss(s', N - i0),
# exactly; at N = 400 the small probabilities reach about 3e-141
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(out, target, tol=1e-8):
        # values only (the return type is checked in case 1)
        assert all(_t_rel(out[k], target[k]) < tol for k in range(4)), (out, target)

    for N, s, s2 in ((400, 0.5, -1.0 / 3.0), (400, 0.25, -0.2), (250, -0.1, 1.0 / 9.0)):
        for i0 in (1, N // 3, N - 1):
            a = fixation_statistics(N, s, i0)
            b = fixation_statistics(N, s2, N - i0)
            # two computed outputs, each within 1e-8 of the exact value: they agree to 2.1e-8
            assert _t_rel(a[0], b[1]) < 2.1e-8 and _t_rel(a[1], b[0]) < 2.1e-8, (N, s, i0, a, b)
            assert _t_rel(a[2], b[3]) < 2.1e-8 and _t_rel(a[3], b[2]) < 2.1e-8, (N, s, i0, a, b)
    a = fixation_statistics(400, -1.0 / 3.0, 1)
    assert 0.0 < a[0] < 1e-100 and _t_rel(a[0] + a[1], 1.0) < 1e-8, a
    # independent targets: plain Gaussian elimination with partial pivoting in mpmath at 400 digits, run once
    # (the absorbing-chain systems of N = 400 with the transition matrix in extended precision)
    _T_FIX_DELETERIOUS = (2.87852227828767689465326565713e-141, 1.0, 29.4514551276837959922398783751, 2.24313648794346022434030898156)       # N = 400, s = -1/3, i0 = 1
    _T_FIX_BENEFICIAL = (1.0, 2.87852227828761356463703277399e-141, 2.24313648794346015228827662588, 29.4514551276837942099904412489)        # N = 400, s = 0.5, i0 = 399
    _t_check(fixation_statistics(400, -1.0 / 3.0, 1), [float(x) for x in _T_FIX_DELETERIOUS])
    _t_check(fixation_statistics(400, 0.5, 399), [float(x) for x in _T_FIX_BENEFICIAL])
    # the lower end of the selection range, s = -0.5, i0 = 1: p_fix about 5.7e-240 (same method, run once)
    _T_FIX_STRONGEST = (5.71042258020733655842240186369e-240, 1.0, 18.192116467196947744171843041, 1.74284701835980197143864271246)       # N = 400, s = -0.5, i0 = 1
    _t_check(fixation_statistics(400, -0.5, 1), [float(x) for x in _T_FIX_STRONGEST])


_crown_case()

# --- test case 3 ---
# each bound of each stated error rule, just outside the domain on each side: N = 1 (no transient state; the range
# [1, N - 1] of i0 is then empty, so N below 2, like N = True, cannot break the N rule alone) and N = 401 (alone),
# N a float (even integer-valued), i0 a bool, i0 a float (even integer-valued), i0 = 0 and i0 = N (absorbing
# states), s = -0.51 and 0.51
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((1, 0.1, 1), (401, 0.1, 3), (10.0, 0.1, 1), (10, 0.1, True), (10, 0.1, 3.0), (10, 0.1, 0),
                   (10, 0.1, 10), (10, -0.51, 3), (10, 0.51, 3)):
        try:
            fixation_statistics(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("fixation_statistics%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 4 ---
# N = 2 with selection, closed form: from one copy the frequency after selection is p' = (1 + s) / (2 + s)
# (q' = 1 / (2 + s)), so the next generation has 0, 1 or 2 copies with probabilities q'**2, 2 p' q', p'**2.
# Hence p_fix = (1 + s)**2 / (1 + (1 + s)**2) and p_loss = 1 / (1 + (1 + s)**2), and the number of generations
# spent in i = 1 is geometric whatever the outcome, t_fix = t_loss = 1 / (p'**2 + q'**2)
# = (2 + s)**2 / (1 + (1 + s)**2)
import numpy as np


def _crown_case():
    import numpy as np
    for _t_s in (-0.5, -0.25, 0.1, 0.5):
        w = (1.0 + _t_s) ** 2
        t = (2.0 + _t_s) ** 2 / (1.0 + w)
        target = (w / (1.0 + w), 1.0 / (1.0 + w), t, t)
        out = fixation_statistics(2, _t_s, 1)
        assert all(abs(out[k] - target[k]) < 1e-8 * target[k] for k in range(4)), (_t_s, out, target)


_crown_case()

# --- test case 5 ---
# exact rational arithmetic: for small N the absorbing-chain systems (I - Q) h = r and (I - Q) g = h are solved
# exactly with Python fractions (the transition matrix from the exact binary values of the inputs), giving
# p_fix, p_loss and the conditional times g / h exactly, for every initial count i0 = 1, ..., N - 1
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

    for N, s in ((8, -0.5), (8, 0.375), (5, 0.3)):
        P = _t_P_exact(N, s, 0, 0)
        A = [[(1 if a == b else 0) - P[a][b] for b in range(1, N)] for a in range(1, N)]
        hf = _t_solve_exact(A, [P[a][N] for a in range(1, N)])
        hl = _t_solve_exact(A, [P[a][0] for a in range(1, N)])
        gf, gl = _t_solve_exact(A, hf), _t_solve_exact(A, hl)
        for i0 in range(1, N):
            k = i0 - 1
            target = [float(x) for x in (hf[k], hl[k], gf[k] / hf[k], gl[k] / hl[k])]
            out = fixation_statistics(N, s, i0)
            assert all(abs(out[k] - target[k]) < 1e-8 * target[k] for k in range(4)), (N, s, i0, out, target)


_crown_case()

# --- test case 6 ---
# first-step analysis on the outputs, for every initial count i0 = 1, ..., N - 1: the fixation probability h is
# harmonic for the chain, h(i) = sum_j P[i, j] h(j) with h(0) = 0 and h(N) = 1 (h(0) = 1, h(N) = 0 for loss),
# and g = h t, with t the conditional time, satisfies g(i) = h(i) + sum_{0 < j < N} P[i, j] g(j). Every term is
# nonnegative, so both identities hold entrywise to the relative accuracy of the outputs, however small h is
# (p_fix down to about 4e-90 for s = -0.5, p_loss down to about 1.6e-44 for s = 0.4). The matrix is built in
# double precision from logarithms (relative error about 1e-13). Tolerances: 2.1e-8 for h (one output against a
# sum of outputs, each within 1e-8); 4.2e-8 for g (each g = h t carries the errors of h and of t)
import numpy as np


def _crown_case():
    import numpy as np
    from scipy.special import gammaln as _t_g

    def _t_matrix(N, s):
        # transition matrix without mutation in double precision, from logarithms, complements formed directly
        i = np.arange(N + 1.0)
        den = N + s * i
        pm, qm = (1 + s) * i / den, (N - i) / den
        j = np.arange(N + 1.0)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(j > 0, j * np.log(pm)[:, None], 0.0)
            lq = np.where(j < N, (N - j) * np.log(qm)[:, None], 0.0)
        return np.exp(_t_g(N + 1.0) - _t_g(j + 1.0) - _t_g(N - j + 1.0) + lp + lq)

    for N, s in ((150, -0.5), (150, 0.4)):
        P = _t_matrix(N, s)[1:N]
        out = np.array([fixation_statistics(N, s, i0) for i0 in range(1, N)], dtype=float)
        for h_in, t_in, ends in ((out[:, 0], out[:, 2], (0.0, 1.0)), (out[:, 1], out[:, 3], (1.0, 0.0))):
            h = np.concatenate(([ends[0]], h_in, [ends[1]]))
            g = np.concatenate(([0.0], h_in * t_in, [0.0]))
            assert np.all(np.abs(h_in - P @ h) <= 2.1e-8 * h_in), (N, s, ends)
            assert np.all(np.abs(g[1:N] - (h_in + P @ g)) <= 4.2e-8 * g[1:N]), (N, s, ends)


_crown_case()

# --- test case 7 ---
# neutral drift at the largest population, N = 400: the number of copies is then a martingale and the chain is
# absorbed with probability 1, so p_fix = i0 / N and p_loss = (N - i0) / N exactly (for i0 = 399 the loss
# probability 1/400 is the small one); exchanging the two alleles maps i0 copies of A to N - i0 copies, so
# t_fix(i0) = t_loss(N - i0) exactly (two computed outputs, each within 1e-8: 2.1e-8), and with i0 = N / 2 the two
# conditional times are equal
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    N = 400
    out = {i0: fixation_statistics(N, 0.0, i0) for i0 in (1, 2, 137, 200, 263, 398, 399)}
    for i0, o in out.items():
        assert _t_rel(o[0], i0 / N) < 1e-8 and _t_rel(o[1], (N - i0) / N) < 1e-8, (i0, o)
        assert _t_rel(o[2], out[N - i0][3]) < 2.1e-8, (i0, o, out[N - i0])


_crown_case()
