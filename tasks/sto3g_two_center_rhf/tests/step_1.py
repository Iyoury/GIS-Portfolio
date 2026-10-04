import numpy as np
import mpmath as _t_mp

# Independent target: F_n(t) = Gamma(n + 1/2) P(n + 1/2, t) / (2 t**(n + 1/2)) with the
# regularized lower incomplete gamma function P evaluated by mpmath at 40 digits
# (no series, no recursion, no erf); F_n(0) = 1 / (2n + 1) exactly.


def _t_F(n, t):
    with _t_mp.workdps(40):
        t = _t_mp.mpf(t)
        if t == 0:
            return float(_t_mp.mpf(1) / (2 * n + 1))
        a = n + _t_mp.mpf(1) / 2
        return float(_t_mp.gamma(a) * _t_mp.gammainc(a, 0, t, regularized=True) / (2 * t ** a))


def _t_check(n_max, t):
    F = boys_function(n_max, t)
    assert isinstance(F, np.ndarray) and F.shape == np.shape(t) + (n_max + 1,), (n_max, t, getattr(F, "shape", None))
    tt = np.asarray(t, dtype=float)
    for idx in np.ndindex(tt.shape):
        for n in range(n_max + 1):
            target = _t_F(n, tt[idx])
            assert abs(F[idx + (n,)] / target - 1.0) < 1e-11, (n, tt[idx], F[idx + (n,)], target)
    return F


# --- test case 0: t = 0 (F_n = 1/(2n+1)) and tiny t, all orders up to 16 ---
assert _t_check(16, 0.0) is not None
assert _t_check(16, 1e-14) is not None
assert _t_check(16, 3e-9) is not None
# --- test case 1: small and intermediate t, where the upward recursion from F_0 loses all
# accuracy for high orders ---
for _t_t in (1e-4, 0.07, 0.9, 3.3, 11.0):
    assert _t_check(16, _t_t) is not None
# --- test case 2: the range around n + 25 and large t up to 1e6 ---
for _t_t in (25.5, 40.9, 41.1, 87.0, 640.0, 1.2e4, 1e6):
    assert _t_check(16, _t_t) is not None
# --- test case 3: arrays of any shape, lower orders, and n_max = 0 ---
assert _t_check(4, np.array([[0.0, 0.2, 7.5], [33.0, 150.0, 2.5e3]])) is not None
assert _t_check(0, np.array([1e-12, 0.5, 19.0, 1e5])) is not None
# --- test case 4: invalid order or argument raises ValueError ---
for _t_bad in ((17, 1.0), (-1, 1.0), (2.5, 1.0), (4, -1e-3), (4, float("nan")), (4, 2e6)):
    _t_raised = False
    try:
        boys_function(*_t_bad)
    except ValueError:
        _t_raised = True
    assert _t_raised, "boys_function%r must raise ValueError" % (_t_bad,)
