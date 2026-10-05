import signal as _t_signal
import time as _t_btime

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


escape_barriers = _t_budget(escape_barriers, 10.0, "escape_barriers")

import numpy as np

# Independent targets: no astroid formula. Along the equilibrium curve
# h(theta) = -sin(2 theta) / (2 sin(theta - psi)) (the field at which theta is an equilibrium),
# the minimum that starts at theta = psi reaches its lowest field, -h_sw, at the fold.
_T_HP = 0.5 * np.pi
_T_GOLD = 0.5 * (np.sqrt(5.0) - 1.0)
_T_Q = np.concatenate([np.geomspace(1e-9, 1e-2, 40), np.linspace(0.0125, 0.9875, 400),
                       1.0 - np.geomspace(1e-2, 1e-9, 40)])


def _t_fold(psi):
    # (h_sw, theta_fold): scan of the curve, then golden-section search for its lowest field
    if psi <= 0.0:
        return 1.0, 0.0
    if psi >= _T_HP:
        return 1.0, -_T_HP

    def H(t):
        return -np.sin(2.0 * t) / (2.0 * np.sin(t - psi))

    ts = psi - np.pi + np.pi * _T_Q
    i = int(np.argmin(H(ts)))
    a, b = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
    c, d = b - _T_GOLD * (b - a), a + _T_GOLD * (b - a)
    fc, fd = H(c), H(d)
    while b - a > 1e-13:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _T_GOLD * (b - a)
            fc = H(c)
        else:
            a, c, fc = c, d, fd
            d = a + _T_GOLD * (b - a)
            fd = H(d)
    t = 0.5 * (a + b)
    return -float(H(t)), float(t)


# All local minima of e(theta) on a fine circle grid (sign change of e' from - to +),
# refined by bisection.
_T_NG = 20000
_T_GRID = -np.pi + 1.234567e-4 + 2.0 * np.pi * np.arange(_T_NG) / _T_NG
_T_DG = 2.0 * np.pi / _T_NG


def _t_slope(t, h, psi):
    return 0.5 * np.sin(2.0 * t) + h * np.sin(t - psi)


def _t_energy(t, h, psi):
    return 0.5 * np.sin(t) ** 2 - h * np.cos(t - psi)


def _t_minima(h, psi):
    g = _t_slope(_T_GRID, h, psi)
    out = []
    for i in np.where((g < 0.0) & (np.roll(g, -1) >= 0.0))[0]:
        a, b = _T_GRID[i], _T_GRID[i] + _T_DG
        for _ in range(60):
            mid = 0.5 * (a + b)
            if _t_slope(mid, h, psi) < 0.0:
                a = mid
            else:
                b = mid
        out.append(0.5 * (a + b))
    return out



def _t_extrema(h, psi):
    # minima (e' from - to +) and maxima (e' from + to -) of e on the circle grid, bisected
    g = _t_slope(_T_GRID, h, psi)
    found = []
    for mask in ((g < 0.0) & (np.roll(g, -1) >= 0.0), (g > 0.0) & (np.roll(g, -1) <= 0.0)):
        roots = []
        for i in np.where(mask)[0]:
            lo, hi = _T_GRID[i], _T_GRID[i] + _T_DG
            s_lo = _t_slope(lo, h, psi)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _t_slope(mid, h, psi) * s_lo > 0.0:
                    lo = mid
                else:
                    hi = mid
            roots.append(0.5 * (lo + hi))
        found.append(roots)
    return found[0], found[1]


def _t_barriers(h, psi):
    # barriers from the original minimum (the one with cos(theta) > 0) over both maxima
    mins, maxs = _t_extrema(h, psi)
    assert len(mins) == 2 and len(maxs) == 2, (h, psi, mins, maxs)
    t_o = max(mins, key=np.cos)
    e_o = _t_energy(t_o, h, psi)
    return tuple(sorted(_t_energy(t, h, psi) - e_o for t in maxs))


# --- test case 0: at zero field both barriers are 1/2 for every angle ---
for _t_p in (0.0, 0.4, np.pi / 4, 1.3, np.pi / 2):
    low, high = escape_barriers(0.0, _t_p)
    assert isinstance(low, np.floating) and isinstance(high, np.floating), (type(low), type(high))
    assert abs(float(low) - 0.5) < 1e-8 and abs(float(high) - 0.5) < 1e-8, (_t_p, low, high)

# --- test case 1: closed forms at psi = 0 and psi = pi/2 ---
h = np.array([-0.9, -0.5, -0.1, 0.2, 0.7, 0.95])
low, high = escape_barriers(h, 0.0)
assert isinstance(low, np.ndarray) and low.shape == h.shape and high.shape == h.shape
assert np.allclose(low, 0.5 * (1.0 + h) ** 2, rtol=0.0, atol=1e-8)
assert np.allclose(high, 0.5 * (1.0 + h) ** 2, rtol=0.0, atol=1e-8)
low, high = escape_barriers(h, np.pi / 2)
assert np.allclose(low, 0.5 * (1.0 - np.abs(h)) ** 2, rtol=0.0, atol=1e-8)
assert np.allclose(high, 0.5 * (1.0 + np.abs(h)) ** 2, rtol=0.0, atol=1e-8)

# --- test case 2: other angles against the grid search, with a 2-D input ---
for _t_p in (0.2, np.pi / 6, np.pi / 4, 1.0, 1.4):
    _t_hs = _t_fold(_t_p)[0]
    h = np.array([[-0.9, -0.4, 0.0], [0.3, 0.7, 0.95]]) * _t_hs
    low, high = escape_barriers(h, _t_p)
    assert low.shape == (2, 3) and high.shape == (2, 3)
    for idx in np.ndindex(h.shape):
        t_lo, t_hi = _t_barriers(h[idx], _t_p)
        assert abs(low[idx] - t_lo) < 1e-8 and abs(high[idx] - t_hi) < 1e-8, (_t_p, h[idx])

# --- test case 3: fields just inside both ends of the accurate range (h_sw - |h| >= 1e-3) ---
for _t_p in (0.1, np.pi / 6, np.pi / 4, 1.2):
    _t_hs = _t_fold(_t_p)[0]
    for x in (-_t_hs + 1.1e-3, _t_hs - 1.1e-3):
        low, high = escape_barriers(x, _t_p)
        t_lo, t_hi = _t_barriers(x, _t_p)
        assert abs(float(low) - t_lo) < 1e-8 and abs(float(high) - t_hi) < 1e-8, (_t_p, x)

# --- test case 4: a bad angle, a field that is not finite or |h| >= h_sw (positive or negative h) raises ValueError ---
for _t_bad in ((0.0, -0.1), (0.0, 1.7), (float("nan"), 0.5), (0.6, np.pi / 4), (1.0, 0.0), (np.array([0.1, 0.6]), np.pi / 4), (np.array([0.1, np.inf]), 0.3), (-1.0, 0.0), (np.array([0.1, -0.6]), np.pi / 4)):
    try:
        escape_barriers(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("escape_barriers%r must raise ValueError" % (_t_bad,))
