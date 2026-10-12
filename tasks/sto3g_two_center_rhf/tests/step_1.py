# --- test case 0 ---
# Independent target: F_n(t) = Gamma(n + 1/2) P(n + 1/2, t) / (2 t**(n + 1/2)) with the
# regularized lower incomplete gamma function P evaluated by mpmath at 40 digits
# (no series, no recursion, no erf); F_n(0) = 1 / (2n + 1) exactly.
# Every test case below is self-contained: it repeats its own imports, helpers and setup.
# The output contract (float array of shape np.shape(t) + (n_max + 1,)) is checked once, in case 3.
# t = 0 (F_n = 1/(2n+1)) and tiny t, all orders up to 16
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_F(n, t):
        with _t_mp.workdps(40):
            t = _t_mp.mpf(t)
            if t == 0:
                return float(_t_mp.mpf(1) / (2 * n + 1))
            a = n + _t_mp.mpf(1) / 2
            return float(_t_mp.gamma(a) * _t_mp.gammainc(a, 0, t, regularized=True) / (2 * t ** a))

    def _t_check(n_max, t):
        F = boys_function(n_max, t)
        tt = np.asarray(t, dtype=float)
        for idx in np.ndindex(tt.shape):
            for n in range(n_max + 1):
                target = _t_F(n, tt[idx])
                assert abs(F[idx + (n,)] / target - 1.0) < 1e-11, (n, tt[idx], F[idx + (n,)], target)
        return F

    assert _t_check(16, 0.0) is not None
    assert _t_check(16, 1e-14) is not None
    assert _t_check(16, 3e-9) is not None


_crown_case()

# --- test case 1 ---
# small and intermediate t, where the upward recursion from F_0 loses all
# accuracy for high orders
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_F(n, t):
        with _t_mp.workdps(40):
            t = _t_mp.mpf(t)
            if t == 0:
                return float(_t_mp.mpf(1) / (2 * n + 1))
            a = n + _t_mp.mpf(1) / 2
            return float(_t_mp.gamma(a) * _t_mp.gammainc(a, 0, t, regularized=True) / (2 * t ** a))

    def _t_check(n_max, t):
        F = boys_function(n_max, t)
        tt = np.asarray(t, dtype=float)
        for idx in np.ndindex(tt.shape):
            for n in range(n_max + 1):
                target = _t_F(n, tt[idx])
                assert abs(F[idx + (n,)] / target - 1.0) < 1e-11, (n, tt[idx], F[idx + (n,)], target)
        return F

    for _t_t in (1e-4, 0.07, 0.9, 3.3, 11.0):
        assert _t_check(16, _t_t) is not None


_crown_case()

# --- test case 2 ---
# the range around n + 25 and large t up to 1e6
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_F(n, t):
        with _t_mp.workdps(40):
            t = _t_mp.mpf(t)
            if t == 0:
                return float(_t_mp.mpf(1) / (2 * n + 1))
            a = n + _t_mp.mpf(1) / 2
            return float(_t_mp.gamma(a) * _t_mp.gammainc(a, 0, t, regularized=True) / (2 * t ** a))

    def _t_check(n_max, t):
        F = boys_function(n_max, t)
        tt = np.asarray(t, dtype=float)
        for idx in np.ndindex(tt.shape):
            for n in range(n_max + 1):
                target = _t_F(n, tt[idx])
                assert abs(F[idx + (n,)] / target - 1.0) < 1e-11, (n, tt[idx], F[idx + (n,)], target)
        return F

    for _t_t in (25.5, 40.9, 41.1, 87.0, 640.0, 1.2e4, 1e6):
        assert _t_check(16, _t_t) is not None


_crown_case()

# --- test case 3 ---
# the output contract: float numpy array of shape np.shape(t) + (n_max + 1,) for arrays of any shape
# (2-D, 1-D, a one-row (1, 3) batch) and for a Python float; lower orders and n_max = 0
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_F(n, t):
        with _t_mp.workdps(40):
            t = _t_mp.mpf(t)
            if t == 0:
                return float(_t_mp.mpf(1) / (2 * n + 1))
            a = n + _t_mp.mpf(1) / 2
            return float(_t_mp.gamma(a) * _t_mp.gammainc(a, 0, t, regularized=True) / (2 * t ** a))

    def _t_check(n_max, t):
        F = boys_function(n_max, t)
        assert isinstance(F, np.ndarray) and np.issubdtype(F.dtype, np.floating) and F.shape == np.shape(t) + (n_max + 1,), (n_max, t, getattr(F, "shape", None))
        tt = np.asarray(t, dtype=float)
        for idx in np.ndindex(tt.shape):
            for n in range(n_max + 1):
                target = _t_F(n, tt[idx])
                assert abs(F[idx + (n,)] / target - 1.0) < 1e-11, (n, tt[idx], F[idx + (n,)], target)
        return F

    assert _t_check(4, np.array([[0.0, 0.2, 7.5], [33.0, 150.0, 2.5e3]])) is not None
    assert _t_check(0, np.array([1e-12, 0.5, 19.0, 1e5])) is not None
    # a one-row batch keeps its leading axis: shape (1, 3, 17); it contains the inclusive end t = 1e6
    assert _t_check(16, np.array([[0.3, 45.0, 1e6]])) is not None
    # a Python float gives shape (n_max + 1,)
    assert _t_check(2, 0.25) is not None


_crown_case()

# --- test case 4 ---
# each stated error rule once, just beyond its limit: n_max above 16, a non-integer n_max, one negative
# element and one element above 1e6 inside an array raise ValueError
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((17, 1.0), (2.5, 1.0), (4, np.array([0.2, -1e-3])), (4, np.array([[1.0, 1.000001e6]]))):
        _t_raised = False
        try:
            boys_function(*_t_bad)
        except ValueError:
            _t_raised = True
        assert _t_raised, "boys_function%r must raise ValueError" % (_t_bad,)


_crown_case()

# --- test case 5 ---
# Independent targets by two other routes: (a) the defining integral of u**(2n) exp(-t u**2) over [0, 1]
# by mpmath tanh-sinh quadrature at 30 digits (no incomplete gamma, no series, no recursion);
# (b) for t >= 1000 the large-t closed form F_n(t) = (2n - 1)!! sqrt(pi) / (2**(n + 1) t**(n + 1/2)),
# whose relative error is the upper regularized gamma Q(n + 1/2, t) < 1e-380 there, up to the
# inclusive end t = 1e6, where exp(-t) underflows but F_16 is about 2.6e-87 and still representable
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_def(n, t):
        with _t_mp.workdps(30):
            t = _t_mp.mpf(t)
            return float(_t_mp.quad(lambda u: u ** (2 * n) * _t_mp.exp(-t * u * u), [0, 0.125, 0.25, 0.5, 1]))

    _t_ts = np.array([0.5, 8.0, 60.0])
    F = boys_function(16, _t_ts)
    _t_T = np.array([[_t_def(n, t) for n in range(17)] for t in _t_ts])
    assert np.max(np.abs(F / _t_T - 1.0)) < 1e-11, (F, _t_T)

    def _t_large(n, t):
        dfac = 1
        for k in range(1, 2 * n, 2):
            dfac *= k
        return float(dfac * _t_mp.sqrt(_t_mp.pi) / (2 ** (n + 1) * _t_mp.mpf(t) ** (n + _t_mp.mpf(1) / 2)))

    _t_big = np.array([1000.0, 4.2e4, 1e6])
    G = boys_function(16, _t_big)
    _t_A = np.array([[_t_large(n, t) for n in range(17)] for t in _t_big])
    assert np.max(np.abs(G / _t_A - 1.0)) < 1e-11, (G, _t_A)


_crown_case()

# --- test case 6 ---
# Independent targets: (a) the Taylor series F_n(t) = sum_k (-t)**k / (k! (2n + 2k + 1)) summed in mpmath
# at 40 digits for small t, where the upward recursion from F_0 cancels; (b) the closed form
# F_0(t) = sqrt(pi / t) erf(sqrt(t)) / 2 with mpmath erf at 40 digits; (c) the exact identity
# (2n + 1) F_n(t) - 2 t F_(n+1)(t) = exp(-t) between computed outputs of consecutive orders, at
# arguments where both terms are of order one: each term carries a relative error below 1e-11, so
# the identity is checked with twice that bound on the sum of their magnitudes
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_series(n, t):
        with _t_mp.workdps(40):
            t = _t_mp.mpf(t)
            acc, term, k = _t_mp.mpf(0), _t_mp.mpf(1), 0
            while True:
                add = term / (2 * n + 2 * k + 1)
                acc += add
                if abs(add) < _t_mp.mpf(10) ** -45 * abs(acc):
                    return float(acc)
                k += 1
                term *= -t / k

    _t_ts = np.array([1e-3, 0.02, 0.3])
    F = boys_function(16, _t_ts)
    _t_T = np.array([[_t_series(n, t) for n in range(17)] for t in _t_ts])
    assert np.max(np.abs(F / _t_T - 1.0)) < 1e-11, (F, _t_T)

    _t_u = np.array([0.7, 5.0, 30.0])
    F0 = boys_function(0, _t_u)[:, 0]
    with _t_mp.workdps(40):
        _t_E = np.array([float(_t_mp.sqrt(_t_mp.pi / t) * _t_mp.erf(_t_mp.sqrt(t)) / 2) for t in map(_t_mp.mpf, _t_u)])
    assert np.max(np.abs(F0 / _t_E - 1.0)) < 1e-11, (F0, _t_E)

    for _t_t in (1.5, 6.0):
        H = boys_function(16, _t_t)
        n = np.arange(16)
        lhs = (2 * n + 1) * H[:-1] - 2.0 * _t_t * H[1:]
        bound = 2e-11 * ((2 * n + 1) * np.abs(H[:-1]) + 2.0 * _t_t * np.abs(H[1:]))
        assert np.all(np.abs(lhs - np.exp(-_t_t)) <= bound), (_t_t, lhs, np.exp(-_t_t))


_crown_case()
