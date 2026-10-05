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


invert_basin_depths = _t_budget(invert_basin_depths, 30.0, "invert_basin_depths")

_T_XE = [0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0]
_T_YE = [0.0, 1000.0, 2000.0, 3000.0, 4000.0]
_T_DEPTH = [[1515.03, 1557.02, 1557.02, 1515.03], [1630.331, 1994.432, 1994.432, 1630.331], [1767.756, 2515.778, 2515.778, 1767.756], [1630.331, 1994.432, 1994.432, 1630.331], [1515.03, 1557.02, 1557.02, 1515.03]]
_T_GZ = [[-0.00012420138486717827, -0.00014328431458916966, -0.00014328431458916966, -0.00012420138486717827], [-0.0001445401883644231, -0.0001696612698796593, -0.0001696612698796593, -0.0001445401883644231], [-0.0001491855838652856, -0.00017577528549926713, -0.00017577528549926713, -0.0001491855838652856], [-0.0001445401883644231, -0.0001696612698796593, -0.0001696612698796593, -0.0001445401883644231], [-0.00012420138486717827, -0.00014328431458916966, -0.00014328431458916966, -0.00012420138486717827]]
_T_DRHO0, _T_LAM = -450.0, 0.0004


# --- test case 0: a 5 x 4 basin (cells 1200 m x 1000 m, depths 1.6 to 2.6 km, density contrast -450 kg m^-3
# at the surface decaying with lam = 4e-4 per m): the depths from the anomaly ---
_t_h = invert_basin_depths(np.array(_T_XE), np.array(_T_YE), np.array(_T_GZ), _T_DRHO0, _T_LAM)
assert isinstance(_t_h, np.ndarray) and _t_h.shape == (5, 4), _t_h
assert np.max(np.abs(_t_h / np.array(_T_DEPTH) - 1.0)) < 1e-8, (_t_h, _T_DEPTH)

# --- test case 1: invalid input, an anomaly with the wrong sign or beyond the infinite slab raises ValueError ---
_t_xe, _t_ye, _t_gz = np.array(_T_XE), np.array(_T_YE), np.array(_T_GZ)
_t_big = _t_gz.copy()
_t_big[0, 0] = -2 * math.pi * _T_G * 450.0 / 4e-4          # the infinite-slab value itself
_t_sign = _t_gz.copy()
_t_sign[1, 1] = 1e-6
for _t_args in ((_t_xe[::-1], _t_ye, _t_gz, _T_DRHO0, _T_LAM), (_t_xe, _t_ye, _t_gz[:, :3], _T_DRHO0, _T_LAM),
                (_t_xe, _t_ye, _t_gz, 0.0, _T_LAM), (_t_xe, _t_ye, _t_gz, _T_DRHO0, -1e-3), (_t_xe, _t_ye, _t_gz, _T_DRHO0, 0.05),
                (_t_xe, _t_ye, _t_big, _T_DRHO0, _T_LAM), (_t_xe, _t_ye, _t_sign, _T_DRHO0, _T_LAM),
                (np.linspace(0, 1, 12), np.linspace(0, 1, 12), np.ones((11, 11)), 1.0, 0.0)):
    try:
        invert_basin_depths(*_t_args)
    except ValueError:
        pass
    else:
        raise AssertionError("invert_basin_depths must raise ValueError")
