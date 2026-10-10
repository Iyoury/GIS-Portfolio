# --- test case 0 ---
# reversible part of the branch (no jump yet)
import numpy as np

_T_NG = 20000

_T_GRID = -np.pi + 1.234567e-4 + 2.0 * np.pi * np.arange(_T_NG) / _T_NG

_T_DG = 2.0 * np.pi / _T_NG

def _t_slope(t, h, psi):
    return 0.5 * np.sin(2.0 * t) + h * np.sin(t - psi)

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

def _t_m(h, psi):
    # descending branch: while two minima exist the particle is still in the one on the side of
    # the positive easy direction (cos(theta) > 0); with a single minimum there is no choice
    mins = _t_minima(h, psi)
    assert len(mins) in (1, 2), (h, psi, mins)
    return float(np.cos(max(mins, key=np.cos) - psi))

for _t_h, _t_p in ((2.0, 0.4), (0.0, 0.7), (-0.3, 0.35), (0.6, 1.3), (-0.2, 1.0)):
    m = branch_magnetization(_t_h, _t_p)
    assert isinstance(m, float)
    assert abs(m - _t_m(_t_h, _t_p)) < 1e-8, (_t_h, _t_p, m)

# --- test case 1 ---
# just before and just after the jump at h = -h_sw(psi)
import numpy as np

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

_T_NG = 20000

_T_GRID = -np.pi + 1.234567e-4 + 2.0 * np.pi * np.arange(_T_NG) / _T_NG

_T_DG = 2.0 * np.pi / _T_NG

def _t_slope(t, h, psi):
    return 0.5 * np.sin(2.0 * t) + h * np.sin(t - psi)

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

def _t_m(h, psi):
    # descending branch: while two minima exist the particle is still in the one on the side of
    # the positive easy direction (cos(theta) > 0); with a single minimum there is no choice
    mins = _t_minima(h, psi)
    assert len(mins) in (1, 2), (h, psi, mins)
    return float(np.cos(max(mins, key=np.cos) - psi))

for _t_p in (0.2, np.pi / 6, np.pi / 4, 1.0, 1.4):
    _t_hs = _t_fold(_t_p)[0]
    m_before = branch_magnetization(-_t_hs + 1.5e-3, _t_p)
    m_after = branch_magnetization(-_t_hs - 1.5e-3, _t_p)
    assert abs(m_before - _t_m(-_t_hs + 1.5e-3, _t_p)) < 1e-8, (_t_p, m_before)
    assert abs(m_after - _t_m(-_t_hs - 1.5e-3, _t_p)) < 1e-8, (_t_p, m_after)

# --- test case 2 ---
# the end points psi = 0 and psi = pi/2, and angles just inside them
import numpy as np

_T_NG = 20000

_T_GRID = -np.pi + 1.234567e-4 + 2.0 * np.pi * np.arange(_T_NG) / _T_NG

_T_DG = 2.0 * np.pi / _T_NG

def _t_slope(t, h, psi):
    return 0.5 * np.sin(2.0 * t) + h * np.sin(t - psi)

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

def _t_m(h, psi):
    # descending branch: while two minima exist the particle is still in the one on the side of
    # the positive easy direction (cos(theta) > 0); with a single minimum there is no choice
    mins = _t_minima(h, psi)
    assert len(mins) in (1, 2), (h, psi, mins)
    return float(np.cos(max(mins, key=np.cos) - psi))

for _t_h, _t_p in ((-0.99, 0.0), (-1.01, 0.0), (0.5, np.pi / 2), (-0.5, np.pi / 2), (-0.9985, np.pi / 2),
                   (-1.2, np.pi / 2), (-0.99, 1e-9), (-1.01, 1e-9), (-0.99, np.pi / 2 - 1e-9),
                   (-1.01, np.pi / 2 - 1e-9)):
    m = branch_magnetization(_t_h, _t_p)
    assert abs(m - _t_m(_t_h, _t_p)) < 1e-8, (_t_h, _t_p, m)
assert abs(branch_magnetization(-0.99, 0.0) - 1.0) < 1e-8
assert abs(branch_magnetization(-1.01, 0.0) + 1.0) < 1e-8
# just inside the accurate range on both sides of the jump, |h + h_sw| = 1.1e-3
assert abs(branch_magnetization(-0.9989, 0.0) - 1.0) < 1e-8
assert abs(branch_magnetization(-1.0011, 0.0) + 1.0) < 1e-8

# --- test case 3 ---
# hysteresis: the particle keeps a minimum that is not the global one
import numpy as np

_T_NG = 20000

_T_GRID = -np.pi + 1.234567e-4 + 2.0 * np.pi * np.arange(_T_NG) / _T_NG

_T_DG = 2.0 * np.pi / _T_NG

def _t_slope(t, h, psi):
    return 0.5 * np.sin(2.0 * t) + h * np.sin(t - psi)

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

def _t_m(h, psi):
    # descending branch: while two minima exist the particle is still in the one on the side of
    # the positive easy direction (cos(theta) > 0); with a single minimum there is no choice
    mins = _t_minima(h, psi)
    assert len(mins) in (1, 2), (h, psi, mins)
    return float(np.cos(max(mins, key=np.cos) - psi))

for _t_h, _t_p in ((-0.3, 0.2), (-0.45, 1.2), (-0.6, 0.15), (-1.7, 0.9), (3.0, 1.1)):
    m = branch_magnetization(_t_h, _t_p)
    assert abs(m - _t_m(_t_h, _t_p)) < 1e-8, (_t_h, _t_p, m)

# --- test case 4 ---
# psi outside [0, pi/2] or a field that is not finite raises ValueError
import numpy as np
for _t_bad in ((0.3, -0.1), (0.3, 1.6), (float("nan"), 0.5), (float("inf"), 0.5)):
    try:
        branch_magnetization(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("branch_magnetization%r must raise ValueError" % (_t_bad,))
