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


prism_gravity = _t_budget(prism_gravity, 10.0, "prism_gravity")
prism_gradients = _t_budget(prism_gradients, 10.0, "prism_gradients")
column_gz = _t_budget(column_gz, 10.0, "column_gz")
invert_basin_depths = _t_budget(invert_basin_depths, 30.0, "invert_basin_depths")

_T_XE = [0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0]
_T_YE = [0.0, 1000.0, 2000.0, 3000.0, 4000.0]
_T_DEPTH = [[1515.03, 1557.02, 1557.02, 1515.03], [1630.331, 1994.432, 1994.432, 1630.331], [1767.756, 2515.778, 2515.778, 1767.756], [1630.331, 1994.432, 1994.432, 1630.331], [1515.03, 1557.02, 1557.02, 1515.03]]
_T_GZ = [[-0.00012420138486717827, -0.00014328431458916966, -0.00014328431458916966, -0.00012420138486717827], [-0.0001445401883644231, -0.0001696612698796593, -0.0001696612698796593, -0.0001445401883644231], [-0.0001491855838652856, -0.00017577528549926713, -0.00017577528549926713, -0.0001491855838652856], [-0.0001445401883644231, -0.0001696612698796593, -0.0001696612698796593, -0.0001445401883644231], [-0.00012420138486717827, -0.00014328431458916966, -0.00014328431458916966, -0.00012420138486717827]]
_T_DRHO0, _T_LAM = -450.0, 0.0004

# --- test case 0: the basin of step 4: its anomaly from its columns (column_gz summed over the cells) and
# its depths back from the anomaly ---
_t_xe, _t_ye, _t_h = np.array(_T_XE), np.array(_T_YE), np.array(_T_DEPTH)
_t_g = np.zeros((5, 4))
for _t_i in range(5):
    for _t_j in range(4):
        _t_st = np.array([0.5 * (_t_xe[_t_i] + _t_xe[_t_i + 1]), 0.5 * (_t_ye[_t_j] + _t_ye[_t_j + 1]), 0.0])
        _t_g[_t_i, _t_j] = sum(column_gz(_t_st, _t_xe[a], _t_xe[a + 1], _t_ye[b], _t_ye[b + 1], _t_h[a, b], _T_DRHO0, _T_LAM)
                               for a in range(5) for b in range(4))
assert np.max(np.abs(_t_g / np.array(_T_GZ) - 1.0)) < 1e-10, (_t_g, _T_GZ)       # same sign in every term
_t_back = invert_basin_depths(_t_xe, _t_ye, np.array(_T_GZ), _T_DRHO0, _T_LAM)
assert np.max(np.abs(_t_back / _t_h - 1.0)) < 1e-8

# --- test case 1: a uniform column (lam = 0) is a prism: column_gz and the z component of prism_gravity agree
# (two computed values, each within 1e-10 relative: 2.1e-10), and its gradients satisfy Laplace outside ---
_t_st = np.array([[30.0, 40.0, -20.0], [0.0, 0.0, 0.0], [-500.0, 900.0, -3.0]])
_t_a = column_gz(_t_st, 0.0, 100.0, 0.0, 200.0, 300.0, -400.0, 0.0)
_t_U, _t_gv = prism_gravity(_t_st, (0.0, 100.0, 0.0, 200.0, 0.0, 300.0), -400.0)
assert np.all(np.abs(_t_a / _t_gv[:, 2] - 1.0) < 2.1e-10), (_t_a, _t_gv)
_t_T = prism_gradients(_t_st[[0, 2]], (0.0, 100.0, 0.0, 200.0, 0.0, 300.0), -400.0)
for _t_k in range(2):
    assert abs(np.trace(_t_T[_t_k])) <= 3e-9 * np.linalg.norm(_t_T[_t_k])
