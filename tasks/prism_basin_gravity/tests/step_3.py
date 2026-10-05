import math
import numpy as np
import signal as _t_signal
import time as _t_btime

# Independent targets: the closed forms of the prism potential, attraction and gradients (Nagy, Papp and
# Benedek 2000) evaluated in 60-digit arithmetic with mpmath (no cancellation at any distance), the
# vertical attraction of the columns by mpmath tanh-sinh quadrature of the lamina term over depth at 30
# digits, and gravity data of a basin computed with that quadrature at 25 digits; all computed once and
# stored below.

_T_G = 6.6743e-11

# Time budget of one call, as stated in the prompt: the call is interrupted once it exceeds the budget
# (by one second), so a solution that is too slow fails this check instead of holding up the tests.
_t_depth = [0]


def _t_budget(fn, seconds, name):
    def wrapped(*args, **kwargs):
        if _t_depth[0]:
            return fn(*args, **kwargs)

        def _alarm(signum, frame):
            raise TimeoutError("%s did not finish within its budget of %g s per call" % (name, seconds))
        try:
            old = _t_signal.signal(_t_signal.SIGALRM, _alarm)
            _t_signal.setitimer(_t_signal.ITIMER_REAL, seconds + 1.0)
            armed = True
        except (ValueError, AttributeError, OSError):      # no SIGALRM here: measure only
            armed = False
        _t_depth[0] += 1
        start = _t_btime.perf_counter()
        try:
            out = fn(*args, **kwargs)
        finally:
            _t_depth[0] -= 1
            if armed:
                _t_signal.setitimer(_t_signal.ITIMER_REAL, 0.0)
                _t_signal.signal(_t_signal.SIGALRM, old)
        elapsed = _t_btime.perf_counter() - start
        assert elapsed <= seconds, ("%s took %.1f s (budget %g s per call)" % (name, elapsed, seconds))
        return out
    return wrapped


def _t_rel(a, b):
    return abs(a - b) / abs(b)


column_gz = _t_budget(column_gz, 10.0, "column_gz")

# (station, (x1, x2, y1, y2, depth, drho0, lam), g_z)
_T_COL = [
    ((50, 100, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.00001182266896761075020671915),
    ((0.001, 100, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.000008461011867755590504171835),
    ((0, 0, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.000005577048275340553263853427),
    ((1, 1, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.000005923119386556591979168138),
    ((100, 200, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.000005577048275340553263853427),
    ((1e-06, 1e-06, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.000005577049359548094151539434),
    ((50, 100, -0.01), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.00001182105089404037832452287),
    ((500, -300, -10), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.000000461492152714326746302963),
    ((-2000, 150, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -5.351795817357728662434799e-8),
    ((50, 200.000001, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.0005), -0.000007345710253637675303238618),
    ((250, 250, 0), (0.0, 1000.0, 0.0, 1000.0, 50000.0, 600.0, 0.01), 0.00001972662741957998848918942),
    ((0.5, 0.5, 0), (0.0, 1000.0, 0.0, 1000.0, 50000.0, 600.0, 0.01), 0.000006000280431422753885881262),
    ((-100, 40, -50), (0.0, 1000.0, 0.0, 1000.0, 50000.0, 600.0, 0.01), 0.000003159106620660464019122356),
    ((250, 250, 0), (-20.0, 20.0, -30.0, 30.0, 10.0, -250.0, 0.0), -4.563575876210684135019359e-11),
    ((0.5, 0.5, 0), (-20.0, 20.0, -30.0, 30.0, 10.0, -250.0, 0.0), -0.000000855291274564201472688753),
    ((-100, 40, -50), (-20.0, 20.0, -30.0, 30.0, 10.0, -250.0, 0.0), -1.266989927636524371790688e-8),
    ((250, 250, 0), (0.0, 500.0, 0.0, 500.0, 8000.0, -500.0, 0.0002), -0.00005289199256024416207850359),
    ((0.5, 0.5, 0), (0.0, 500.0, 0.0, 500.0, 8000.0, -500.0, 0.0002), -0.0000248624223786821367347977),
    ((-100, 40, -50), (0.0, 500.0, 0.0, 500.0, 8000.0, -500.0, 0.0002), -0.00001758873182773243820548322),
    ((30, 40, -20), (0.0, 100.0, 0.0, 200.0, 300.0, -400.0, 0.0), -0.000007134385593100156708200638),
    ((0, 0, 0), (0.0, 100.0, 0.0, 200.0, 300.0, -400.0, 0.0), -0.000004780804200799920941693571),
    ((100, 0, 0), (0.0, 100.0, 0.0, 200.0, 300.0, -400.0, 0.0), -0.000004780804200799920941693571),
]


# --- test case 0: stations on the surface above the column, above its edges and corners, 1e-3 m and 1e-6 m
# from an edge, above the surface and far beside it; a deep column with lam = 1e-2, a shallow uniform one
# and a wide one ---
for _t_q, _t_c, _t_g in _T_COL[:-3]:
    _t_got = column_gz(np.array(_t_q, float), *_t_c)
    assert type(_t_got) is float and _t_rel(_t_got, _t_g) < 1e-10, (_t_q, _t_c, _t_got, _t_g)

# --- test case 1: lam = 0 is a prism of uniform density: against the 60-digit closed form of the prism ---
for _t_q, _t_c, _t_g in _T_COL[-3:]:
    _t_got = column_gz(np.array(_t_q, float), *_t_c)
    assert _t_rel(_t_got, _t_g) < 1e-10, (_t_q, _t_got, _t_g)

# --- test case 2: several stations at once give an array of shape (n,) ---
_t_c = _T_COL[0][1]
_t_sts = np.array([r[0] for r in _T_COL[:5]], float)
_t_got = column_gz(_t_sts, *_t_c)
assert isinstance(_t_got, np.ndarray) and _t_got.shape == (5,)
for _t_k in range(5):
    assert _t_rel(_t_got[_t_k], _T_COL[_t_k][2]) < 1e-10

# --- test case 3: invalid stations or parameters raise ValueError ---
_t_ok = (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 5e-4)
for _t_q, _t_c in (((0, 0, 1.0), _t_ok), ((0, 0), _t_ok), ((0, float("nan"), 0), _t_ok),
                   ((0, 0, 0), (100.0, 0.0, 0.0, 200.0, 3000.0, -400.0, 5e-4)), ((0, 0, 0), (0.0, 100.0, 0.0, 200.0, 0.0, -400.0, 5e-4)),
                   ((0, 0, 0), (0.0, 100.0, 0.0, 200.0, 6e4, -400.0, 5e-4)), ((0, 0, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, -1e-4)),
                   ((0, 0, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, -400.0, 0.02)), ((0, 0, 0), (0.0, 100.0, 0.0, 200.0, 3000.0, float("nan"), 0.0))):
    try:
        column_gz(np.array(_t_q, float), *_t_c)
    except ValueError:
        pass
    else:
        raise AssertionError("column_gz%r must raise ValueError" % ((_t_q,) + _t_c,))
