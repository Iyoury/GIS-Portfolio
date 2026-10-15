# --- test case 0 ---
# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.
# one individual: pi = (u, v) / (u + v), also with u = 1e-12
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(pi, target, tol=1e-8):
        # the return type (the one contract check of this step): a float numpy array of shape (N + 1,), here (2,)
        assert isinstance(pi, np.ndarray) and pi.shape == (len(target),) and np.issubdtype(pi.dtype, np.floating), pi
        for a, b in zip(pi, target):
            if b >= 1e-250:
                assert _t_rel(a, b) < tol, (a, b)
            else:
                # 1e-250 against an exact target, 2e-250 between two computed outputs
                assert abs(a - b) <= (2e-250 if tol > 1e-8 else 1e-250), (a, b)

    for _t_s, _t_u, _t_v in ((0.3, 0.1, 0.05), (-0.5, 1e-12, 0.1)):
        pi = stationary_distribution(1, _t_s, _t_u, _t_v)
        _t_check(pi, [_t_u / (_t_u + _t_v), _t_v / (_t_u + _t_v)])


_crown_case()

# --- test case 1 ---
# neutral drift: the first two moments are exact. With c = 1 - u - v and
# m = v / (u + v): E[p] = m and H = E[p (1 - p)] solves
# H (1 - (N - 1) c**2 / N) = (N - 1) / N (v (1 - v) + c m (1 - 2 v) - c**2 m)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    for N, u, v in ((300, 0.01, 0.02), (120, 1e-4, 3e-3)):
        pi = stationary_distribution(N, 0.0, u, v)
        p = np.arange(N + 1) / N
        c, m = 1 - u - v, v / (u + v)
        H = (N - 1) / N * (v * (1 - v) + c * m * (1 - 2 * v) - c * c * m) / (1 - (N - 1) * c * c / N)
        assert abs(pi.sum() - 1.0) < 1e-8
        assert _t_rel(pi @ p, m) < 1e-8 and _t_rel(pi @ (p * (1 - p)), H) < 1e-8, (pi @ p, m, pi @ (p * (1 - p)), H)


_crown_case()

# --- test case 2 ---
# strong selection with mutation rates down to 1e-12: entries far below rounding of
# the largest one, against extended precision
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

    def _t_check(pi, target, tol=1e-8):
        # values only (the return type is checked in case 0); every target entry is read by position
        pi = np.asarray(pi)
        for k, b in enumerate(target):
            a = pi[k]
            if b >= 1e-250:
                assert _t_rel(a, b) < tol, (a, b)
            else:
                # 1e-250 against an exact target, 2e-250 between two computed outputs
                assert abs(a - b) <= (2e-250 if tol > 1e-8 else 1e-250), (a, b)

    for N, s, u, v, dps in ((20, -0.5, 1e-12, 1e-12, 400), (25, 0.3, 1e-3, 1e-10, 300), (30, 0.05, 1e-6, 2e-6, 200)):
        _t_check(stationary_distribution(N, s, u, v), _t_stationary(N, s, u, v, dps))


_crown_case()

# --- test case 3 ---
# relabeling the alleles (s' = -s / (1 + s), u and v exchanged, i -> N - i) gives the
# same distribution exactly; at N = 400 it spans about 150 decades
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(pi, target, tol=1e-8):
        # values only (the return type and the array shape are checked in case 0)
        pi = np.asarray(pi)
        for a, b in zip(pi, target):
            if b >= 1e-250:
                assert _t_rel(a, b) < tol, (a, b)
            else:
                # 1e-250 against an exact target, 2e-250 between two computed outputs
                assert abs(a - b) <= (2e-250 if tol > 1e-8 else 1e-250), (a, b)

    for N, s, s2, u, v in ((400, 0.5, -1.0 / 3.0, 1e-12, 1e-12), (300, 0.25, -0.2, 1e-9, 1e-4)):
        a = stationary_distribution(N, s, u, v)
        b = stationary_distribution(N, s2, v, u)[::-1]
        assert a.min() < 1e-60, a.min()
        _t_check(a, b, tol=2.1e-8)            # two computed outputs, each within 1e-8: 2.1e-8

    # selected entries of the N = 400, s = 0.5, u = v = 1e-12 distribution, from the largest to about 6e-150,
    # against independent targets: plain Gaussian elimination with partial pivoting in mpmath at 400 digits, run once (the normalized system pi (I - P) = 0)
    _T_PI400 = {0: 4.94116127507236145905204620329e-141, 1: 6.36089756835676078305578893786e-150, 2: 6.95238834484355169992953876913e-150, 5: 3.92169133044440556851206613453e-149, 20: 4.67437739851039383742400642633e-144, 50: 3.21915037362450569537510917039e-133, 100: 4.06794059951012930020318771023e-115, 150: 3.02751884997609130817033404023e-97, 200: 1.25749031680770628987807725071e-79, 250: 3.07910294494558639844691955626e-62, 300: 4.90732674695967331243089316249e-45, 350: 6.31249880988631372426442824871e-28, 380: 1.53470434599298048276897709076e-17, 395: 5.68910886367584408915605652943e-12, 398: 1.27405798115104130323093073556e-10, 399: 7.00640566036223133820257310194e-10, 400: 0.999999999102745405225542146036}
    pi = stationary_distribution(400, 0.5, 1e-12, 1e-12)
    for _t_i, _t_v in _T_PI400.items():
        assert _t_rel(pi[_t_i], float(_t_v)) < 1e-8, (_t_i, pi[_t_i], _t_v)

    # N = 400, s = -0.5, u = v = 1e-12 (A deleterious): entries from the largest down to about 1.3e-248, many of
    # them between 1e-160 and 1e-250, against independent targets (same method, run once)
    _T_PI400B = {0: 0.999999999302861192899082297338, 1: 6.160479874109954259093598051e-10, 10: 6.25249254735737461421144225314e-16, 100: 7.49376642376130645326028972991e-67, 200: 1.05943343187438244029411313061e-124, 250: 1.2539533917607727124297648688e-154, 261: 2.52977809035656817356344627748e-161, 270: 7.80509627467630295345328177044e-167, 300: 1.97242508617735664803857894273e-185, 330: 2.06109467248043194007825125277e-204, 360: 8.36989714563045932632084386903e-224, 390: 1.98578943480245519905452196041e-243, 398: 3.05195164582500612801241466235e-248, 399: 1.3171353389076868535938508154e-248, 400: 7.16957094624388165837027559461e-240}
    pi = stationary_distribution(400, -0.5, 1e-12, 1e-12)
    for _t_i, _t_v in _T_PI400B.items():
        assert _t_rel(pi[_t_i], float(_t_v)) < 1e-8, (_t_i, pi[_t_i], _t_v)


_crown_case()

# --- test case 4 ---
# each bound of each stated error rule, just outside the domain on each side: N = 0 and N = 401, N a float
# (even integer-valued), N a bool (True, otherwise in range), s = -0.51 and 0.51, u = 0 (below its floor 1e-12: the chain is then not irreducible)
# and u = 0.11, v = 1e-13 (below its floor) and v = 0.11; every entry breaks exactly one rule
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((0, 0.1, 0.01, 0.01), (401, 0.1, 0.01, 0.01), (12.0, 0.1, 0.01, 0.01), (True, 0.1, 0.01, 0.01),
                   (10, -0.51, 0.01, 0.01), (10, 0.51, 0.01, 0.01), (10, 0.1, 0.0, 0.01), (10, 0.1, 0.11, 0.01),
                   (10, 0.1, 0.01, 1e-13), (10, 0.1, 0.01, 0.11)):
        try:
            stationary_distribution(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("stationary_distribution%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 5 ---
# exact rational arithmetic: for small N the normalized system pi (I - P) = 0, sum(pi) = 1 is solved exactly
# with Python fractions (the transition matrix from the exact binary values of the inputs), including a mutation
# rate at its floor 1e-12
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

    for N, s, u, v in ((6, 0.25, 0.03125, 0.015625), (5, -0.5, 0.0625, 0.0009765625), (4, 0.3, 1e-12, 0.1)):
        P = _t_P_exact(N, s, u, v)
        n = N + 1
        A = [[1] * n] + [[(1 if i == j else 0) - P[i][j] for i in range(n)] for j in range(1, n)]
        target = np.array([float(x) for x in _t_solve_exact(A, [1] + [0] * N)])
        got = np.asarray(stationary_distribution(N, s, u, v), dtype=float)
        assert all(abs(got[k] - target[k]) < 1e-8 * target[k] for k in range(N + 1)), (N, s, u, v, got, target)


_crown_case()

# --- test case 6 ---
# stationarity of the output itself, entry by entry: (pi P)_j = pi_j. Every term of (pi P)_j is nonnegative, so
# the identity holds to the relative accuracy of the entries even where pi_j is more than 200 decades below the
# largest entry (N = 400, s = -0.5, u = v = 1e-12); checked on every entry of at least 1e-235 (an entry of the
# output above 1e-235 has an exact value of at least about 1e-235, and the entries below 1e-250, each within
# 1e-250, change (pi P)_j by at most 4.1e-248), with the matrix in double precision from logarithms (relative error
# about 1e-13): one output against a sum of outputs, each within 1e-8, 2.1e-8; and sum(pi) = 1 to 1e-8
import numpy as np


def _crown_case():
    import numpy as np
    from scipy.special import gammaln as _t_g

    def _t_matrix(N, s, u, v):
        # transition matrix in double precision from logarithms, complements formed directly
        i = np.arange(N + 1.0)
        den = N + s * i
        ps, qs = (1 + s) * i / den, (N - i) / den
        pm, qm = (1 - u) * ps + v * qs, u * ps + (1 - v) * qs
        j = np.arange(N + 1.0)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(j > 0, j * np.log(pm)[:, None], 0.0)
            lq = np.where(j < N, (N - j) * np.log(qm)[:, None], 0.0)
        return np.exp(_t_g(N + 1.0) - _t_g(j + 1.0) - _t_g(N - j + 1.0) + lp + lq)

    for N, s, u, v in ((400, -0.5, 1e-12, 1e-12), (300, 0.3, 1e-3, 1e-10), (200, 0.0, 1e-12, 0.1)):
        pi = np.asarray(stationary_distribution(N, s, u, v), dtype=float)
        rhs = pi @ _t_matrix(N, s, u, v)
        big = pi >= 1e-235
        assert np.all(np.abs(pi[big] - rhs[big]) <= 2.1e-8 * pi[big]), (N, s, u, v, np.nonzero(big)[0].size)
        assert abs(pi.sum() - 1.0) < 1e-8, (N, s, u, v, pi.sum())


_crown_case()

# --- test case 7 ---
# neutral drift (s = 0): the mean of the next frequency is linear, p' = v + c p with c = 1 - u - v, and the
# binomial factorial moments are E[j (j - 1) ... (j - k + 1) | i] = N (N - 1) ... (N - k + 1) p'**k, so the
# stationary moments m_k = E_pi[i**k] obey a closed triangular system: with the falling factorial
# i (i - 1) ... (i - k + 1) = sum_l S1(k, l) i**l (S1: signed Stirling numbers of the first kind),
# sum_l S1(k, l) m_l = N (N - 1) ... (N - k + 1) sum_l C(k, l) v**(k - l) (c / N)**l m_l, l = 0, ..., k, m_0 = 1.
# m_1, ..., m_4 solved exactly with Python fractions from the binary values of the inputs. The outputs give
# sum_i pi_i i**k, a sum of nonnegative terms, so it keeps the relative accuracy 1e-8 of the entries (the entries
# below 1e-250 add at most 401 x 1e-250 x 400**4 in absolute value), plus summation rounding below 1e-13: 1.01e-8.
# At the largest population and both mutation rates at their upper end 0.1, and with one rate at its floor 1e-12
# (then m_1 = N v / (u + v) is about 4e-9 and the moments are set by the small entries next to i = 0)
import numpy as np


def _crown_case():
    import numpy as np
    from fractions import Fraction as _t_F
    from math import comb as _t_comb

    def _t_stirling1(k, l):
        # signed Stirling numbers of the first kind, s(k, l), by the recurrence s(k, l) = s(k-1, l-1) - (k-1) s(k-1, l)
        table = {(0, 0): 1}
        for a in range(1, k + 1):
            for b in range(0, a + 1):
                table[(a, b)] = table.get((a - 1, b - 1), 0) - (a - 1) * table.get((a - 1, b), 0)
        return table.get((k, l), 0)

    def _t_moments(N, u, v, K):
        u, v = _t_F(u), _t_F(v)
        c = 1 - u - v
        m = [_t_F(1)]
        for k in range(1, K + 1):
            fall = 1
            for a in range(k):
                fall *= N - a
            rhs = sum(fall * _t_comb(k, l) * v ** (k - l) * (c / N) ** l * m[l] for l in range(k))
            rhs -= sum(_t_stirling1(k, l) * m[l] for l in range(k))
            m.append(rhs / (1 - fall * (c / N) ** k))
        return [float(x) for x in m[1:]]

    for N, u, v in ((400, 0.1, 0.1), (400, 0.1, 1e-12), (250, 1e-12, 0.1), (37, 0.05, 2e-3)):
        pi = np.asarray(stationary_distribution(N, 0.0, u, v), dtype=float)
        i = np.arange(N + 1, dtype=float)
        for k, target in enumerate(_t_moments(N, u, v, 4), start=1):
            got = float(pi @ i ** k)
            assert abs(got - target) <= 1.01e-8 * target, (N, u, v, k, got, target)


_crown_case()
# --- test case 8 ---
# the stated rule for the far tail: N = 400 with A strongly deleterious, mutation A -> a at its upper end and
# a -> A at its floor (s = -0.5, u = 0.1, v = 1e-12). The distribution falls from about 1 at i = 0 to about
# 6e-320 at i = N; it crosses 1e-250 between i = 340 (1.44e-250, relative 1e-8 required) and i = 341 (1.8e-251,
# within 1e-250 of the exact value), and 60 entries lie below 1e-250. Independent target: the normalized system
# pi (I - P) = 0, sum(pi) = 1 with the matrix from mpmath binomials, one plain LU solve (partial pivoting) in mpmath
# at 400 digits, run once (relative residual of the stationarity equations 3e-400). Stored entries: i = 0, 1, 2, 5,
# 10, 20, ..., 330, every entry from 335 to 345 around the 1e-250 threshold, and 350, 360, ..., 400, 399
import numpy as np


def _crown_case():
    import numpy as np
    pi = np.asarray(stationary_distribution(400, -0.5, 0.1, 1e-12), dtype=float)
    _t_entries = (
         "0:9.9999999934693986e-1 1:5.92158010926311385e-10 2:4.92520325772818614e-11 "
         "5:3.16289582237012055e-13 10:1.23701644497945561e-16 20:3.55122769106917976e-23 "
         "30:1.27687930661455903e-29 40:4.84081475686557233e-36 50:1.82594721797762509e-42 "
         "60:6.68354623913684148e-49 70:2.33725898647343177e-55 80:7.72509676221090506e-62 "
         "90:2.3944730618668136e-68 100:6.91534030294960984e-75 110:1.84979800025956894e-81 "
         "120:4.5568075159860648e-88 130:1.02801690196290733e-94 140:2.11212530075381621e-101 "
         "150:3.92937937480792183e-108 160:6.57930544677819417e-115 170:9.85058184434275471e-122 "
         "180:1.3094961615206793e-128 190:1.53376145716918125e-135 200:1.56941565252103302e-142 "
         "210:1.38982623148158806e-149 220:1.05404741447641489e-156 230:6.76530801985241977e-164 "
         "240:3.62559785291680998e-171 250:1.59733184031802536e-178 260:5.68203741641915648e-186 "
         "270:1.59775816248018983e-193 280:3.46324982467471052e-201 290:5.61360189916265133e-209 "
         "300:6.55587339071634182e-217 310:5.2658467547458197e-225 320:2.74128360790182354e-233 "
         "330:8.55420110137549009e-242 335:3.82099641278835157e-246 336:5.05287117257653153e-247 "
         "337:6.63550466297245135e-248 338:8.65199447223228447e-249 339:1.11994020539787683e-249 "
         "340:1.43892047538472729e-250 341:1.83470487750590265e-251 342:2.32115498784565431e-252 "
         "343:2.91318689845655026e-253 344:3.62637890406922781e-254 345:4.47640455978897241e-255 "
         "350:1.12219929913820008e-259 360:3.23553005329863536e-269 370:2.38704165141174433e-279 "
         "380:2.29759111474861784e-290 390:6.27588362494149745e-303 399:1.21148289898529174e-317 "
         "400:5.81644875537179479e-320 "
    )
    for tok in "".join(_t_entries).split():
        i, t = int(tok.split(":")[0]), float(tok.split(":")[1])
        if t >= 1e-250:
            assert abs(pi[i] - t) < 1e-8 * t, (i, pi[i], t)
        else:
            assert abs(pi[i] - t) <= 1e-250, (i, pi[i], t)


_crown_case()
