# --- test case 0 ---
# one particle at psi = 0: h_c = h_half = minus the median switching field
import numpy as np
from scipy.integrate import quad as _t_quad
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv

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

def _t_gamma(x, psi, a):
    # escape rate over f0: both routes, exponent 2 a Delta_e because E = 2 K V e. Within a grid
    # step of +-h_sw two extrema may merge on the grid; the rate is negligible there.
    mins, maxs = _t_extrema(x, psi)
    if len(mins) < 2 or len(maxs) < 2:
        return 0.0
    t_o = max(mins, key=np.cos)
    e_o = _t_energy(t_o, x, psi)
    return sum(np.exp(-2.0 * a * (_t_energy(t, x, psi) - e_o)) for t in maxs)

def _t_P(h, psi, a, ratio):
    # survival probability exp(-(f0 / rate) int_h^h_sw Gamma / f0), quadrature on the grid barriers
    hs = _t_fold(psi)[0]
    if h >= hs:
        return 1.0
    if h <= -hs:
        return 0.0
    I = _t_quad(lambda x: _t_gamma(x, psi, a), h, hs, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    return float(np.exp(-ratio * I))

def _t_median0(a, ratio):
    # psi = 0: P = 1/2 where (f0 / rate) * I = ln 2, solved with the inverse erfc
    return -1.0 + _t_erfcinv(np.log(2.0) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)

def _t_dm(t, h, psi):
    # dm/dh along a minimum theta(h): dtheta/dh = -sin(theta - psi) / e''(theta)
    return np.sin(t - psi) ** 2 / (np.cos(2.0 * t) + h * np.cos(t - psi))

def _t_ensemble(h, psis, w, a, ratio):
    # expected ensemble magnetization M, switched weight S and their derivatives in h
    w = np.asarray(w, dtype=float) / np.sum(w)
    M = dM = S = dS = 0.0
    for psi, wi in zip(psis, w):
        hs = _t_fold(psi)[0]
        mins = _t_minima(h, psi)
        t_x = min(mins, key=np.cos)        # the other minimum, cos(theta) < 0
        m_x, dm_x = np.cos(t_x - psi), _t_dm(t_x, h, psi)
        if h <= -hs:
            p = dp = m_o = dm_o = 0.0
        else:
            t_o = max(mins, key=np.cos)
            m_o, dm_o = np.cos(t_o - psi), _t_dm(t_o, h, psi)
            p = _t_P(h, psi, a, ratio)
            dp = ratio * _t_gamma(h, psi, a) * p
        M += wi * (p * m_o + (1.0 - p) * m_x)
        dM += wi * (dp * (m_o - m_x) + p * dm_o + (1.0 - p) * dm_x)
        S += wi * (1.0 - p)
        dS -= wi * dp
    return M, dM, S, dS

def _t_check_ensemble(out, psis, w, a, ratio):
    # one Newton step on the independent M(h) = 0 and S(h) = 1/2 must not move the results
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
    h_c, h_half = out
    assert -1.0 < h_c < 1.0 and -1.0 < h_half < 1.0, out
    M, dM, _, _ = _t_ensemble(-h_c, psis, w, a, ratio)
    assert abs(M / dM) < 1e-6, (h_c, M / dM)
    _, _, S, dS = _t_ensemble(-h_half, psis, w, a, ratio)
    assert abs((S - 0.5) / dS) < 1e-6, (h_half, (S - 0.5) / dS)
    return out

out = _t_check_ensemble(ensemble_switching(np.array([0.0]), np.array([2.0]), 100.0, 1e9, 1.0),
                        [0.0], [2.0], 100.0, 1e9)
hm = _t_median0(100.0, 1e9)
assert abs(out[0] + hm) < 1e-6 and abs(out[1] + hm) < 1e-6, (out, hm)

# --- test case 1 ---
# three particles with weights that do not add up to 1
import numpy as np
from scipy.integrate import quad as _t_quad

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

def _t_gamma(x, psi, a):
    # escape rate over f0: both routes, exponent 2 a Delta_e because E = 2 K V e. Within a grid
    # step of +-h_sw two extrema may merge on the grid; the rate is negligible there.
    mins, maxs = _t_extrema(x, psi)
    if len(mins) < 2 or len(maxs) < 2:
        return 0.0
    t_o = max(mins, key=np.cos)
    e_o = _t_energy(t_o, x, psi)
    return sum(np.exp(-2.0 * a * (_t_energy(t, x, psi) - e_o)) for t in maxs)

def _t_P(h, psi, a, ratio):
    # survival probability exp(-(f0 / rate) int_h^h_sw Gamma / f0), quadrature on the grid barriers
    hs = _t_fold(psi)[0]
    if h >= hs:
        return 1.0
    if h <= -hs:
        return 0.0
    I = _t_quad(lambda x: _t_gamma(x, psi, a), h, hs, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    return float(np.exp(-ratio * I))

def _t_dm(t, h, psi):
    # dm/dh along a minimum theta(h): dtheta/dh = -sin(theta - psi) / e''(theta)
    return np.sin(t - psi) ** 2 / (np.cos(2.0 * t) + h * np.cos(t - psi))

def _t_ensemble(h, psis, w, a, ratio):
    # expected ensemble magnetization M, switched weight S and their derivatives in h
    w = np.asarray(w, dtype=float) / np.sum(w)
    M = dM = S = dS = 0.0
    for psi, wi in zip(psis, w):
        hs = _t_fold(psi)[0]
        mins = _t_minima(h, psi)
        t_x = min(mins, key=np.cos)        # the other minimum, cos(theta) < 0
        m_x, dm_x = np.cos(t_x - psi), _t_dm(t_x, h, psi)
        if h <= -hs:
            p = dp = m_o = dm_o = 0.0
        else:
            t_o = max(mins, key=np.cos)
            m_o, dm_o = np.cos(t_o - psi), _t_dm(t_o, h, psi)
            p = _t_P(h, psi, a, ratio)
            dp = ratio * _t_gamma(h, psi, a) * p
        M += wi * (p * m_o + (1.0 - p) * m_x)
        dM += wi * (dp * (m_o - m_x) + p * dm_o + (1.0 - p) * dm_x)
        S += wi * (1.0 - p)
        dS -= wi * dp
    return M, dM, S, dS

def _t_check_ensemble(out, psis, w, a, ratio):
    # one Newton step on the independent M(h) = 0 and S(h) = 1/2 must not move the results
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
    h_c, h_half = out
    assert -1.0 < h_c < 1.0 and -1.0 < h_half < 1.0, out
    M, dM, _, _ = _t_ensemble(-h_c, psis, w, a, ratio)
    assert abs(M / dM) < 1e-6, (h_c, M / dM)
    _, _, S, dS = _t_ensemble(-h_half, psis, w, a, ratio)
    assert abs((S - 0.5) / dS) < 1e-6, (h_half, (S - 0.5) / dS)
    return out

_t_check_ensemble(ensemble_switching(np.array([0.2, 0.8, 1.25]), np.array([1.0, 3.0, 2.0]), 150.0, 1e9, 10.0),
                  [0.2, 0.8, 1.25], [1.0, 3.0, 2.0], 150.0, 1e8)

# --- test case 2 ---
# an ensemble with a particle at psi = pi/2 (both of its minima have m = h, so it
# only shifts the magnetization, but it has left its original minimum almost at h = 1)
import numpy as np
from scipy.integrate import quad as _t_quad

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

def _t_gamma(x, psi, a):
    # escape rate over f0: both routes, exponent 2 a Delta_e because E = 2 K V e. Within a grid
    # step of +-h_sw two extrema may merge on the grid; the rate is negligible there.
    mins, maxs = _t_extrema(x, psi)
    if len(mins) < 2 or len(maxs) < 2:
        return 0.0
    t_o = max(mins, key=np.cos)
    e_o = _t_energy(t_o, x, psi)
    return sum(np.exp(-2.0 * a * (_t_energy(t, x, psi) - e_o)) for t in maxs)

def _t_P(h, psi, a, ratio):
    # survival probability exp(-(f0 / rate) int_h^h_sw Gamma / f0), quadrature on the grid barriers
    hs = _t_fold(psi)[0]
    if h >= hs:
        return 1.0
    if h <= -hs:
        return 0.0
    I = _t_quad(lambda x: _t_gamma(x, psi, a), h, hs, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    return float(np.exp(-ratio * I))

def _t_dm(t, h, psi):
    # dm/dh along a minimum theta(h): dtheta/dh = -sin(theta - psi) / e''(theta)
    return np.sin(t - psi) ** 2 / (np.cos(2.0 * t) + h * np.cos(t - psi))

def _t_ensemble(h, psis, w, a, ratio):
    # expected ensemble magnetization M, switched weight S and their derivatives in h
    w = np.asarray(w, dtype=float) / np.sum(w)
    M = dM = S = dS = 0.0
    for psi, wi in zip(psis, w):
        hs = _t_fold(psi)[0]
        mins = _t_minima(h, psi)
        t_x = min(mins, key=np.cos)        # the other minimum, cos(theta) < 0
        m_x, dm_x = np.cos(t_x - psi), _t_dm(t_x, h, psi)
        if h <= -hs:
            p = dp = m_o = dm_o = 0.0
        else:
            t_o = max(mins, key=np.cos)
            m_o, dm_o = np.cos(t_o - psi), _t_dm(t_o, h, psi)
            p = _t_P(h, psi, a, ratio)
            dp = ratio * _t_gamma(h, psi, a) * p
        M += wi * (p * m_o + (1.0 - p) * m_x)
        dM += wi * (dp * (m_o - m_x) + p * dm_o + (1.0 - p) * dm_x)
        S += wi * (1.0 - p)
        dS -= wi * dp
    return M, dM, S, dS

def _t_check_ensemble(out, psis, w, a, ratio):
    # one Newton step on the independent M(h) = 0 and S(h) = 1/2 must not move the results
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
    h_c, h_half = out
    assert -1.0 < h_c < 1.0 and -1.0 < h_half < 1.0, out
    M, dM, _, _ = _t_ensemble(-h_c, psis, w, a, ratio)
    assert abs(M / dM) < 1e-6, (h_c, M / dM)
    _, _, S, dS = _t_ensemble(-h_half, psis, w, a, ratio)
    assert abs((S - 0.5) / dS) < 1e-6, (h_half, (S - 0.5) / dS)
    return out

_t_ps, _t_ws = [0.3, np.pi / 2, 1.0], [2.0, 1.0, 1.0]
_t_check_ensemble(ensemble_switching(np.array(_t_ps), np.array(_t_ws), 200.0, 1e9, 1.0),
                  _t_ps, _t_ws, 200.0, 1e9)

# --- test case 3 ---
# a zero weight is allowed and the particle then does not count
import numpy as np
from scipy.integrate import quad as _t_quad

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

def _t_gamma(x, psi, a):
    # escape rate over f0: both routes, exponent 2 a Delta_e because E = 2 K V e. Within a grid
    # step of +-h_sw two extrema may merge on the grid; the rate is negligible there.
    mins, maxs = _t_extrema(x, psi)
    if len(mins) < 2 or len(maxs) < 2:
        return 0.0
    t_o = max(mins, key=np.cos)
    e_o = _t_energy(t_o, x, psi)
    return sum(np.exp(-2.0 * a * (_t_energy(t, x, psi) - e_o)) for t in maxs)

def _t_P(h, psi, a, ratio):
    # survival probability exp(-(f0 / rate) int_h^h_sw Gamma / f0), quadrature on the grid barriers
    hs = _t_fold(psi)[0]
    if h >= hs:
        return 1.0
    if h <= -hs:
        return 0.0
    I = _t_quad(lambda x: _t_gamma(x, psi, a), h, hs, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    return float(np.exp(-ratio * I))

def _t_dm(t, h, psi):
    # dm/dh along a minimum theta(h): dtheta/dh = -sin(theta - psi) / e''(theta)
    return np.sin(t - psi) ** 2 / (np.cos(2.0 * t) + h * np.cos(t - psi))

def _t_ensemble(h, psis, w, a, ratio):
    # expected ensemble magnetization M, switched weight S and their derivatives in h
    w = np.asarray(w, dtype=float) / np.sum(w)
    M = dM = S = dS = 0.0
    for psi, wi in zip(psis, w):
        hs = _t_fold(psi)[0]
        mins = _t_minima(h, psi)
        t_x = min(mins, key=np.cos)        # the other minimum, cos(theta) < 0
        m_x, dm_x = np.cos(t_x - psi), _t_dm(t_x, h, psi)
        if h <= -hs:
            p = dp = m_o = dm_o = 0.0
        else:
            t_o = max(mins, key=np.cos)
            m_o, dm_o = np.cos(t_o - psi), _t_dm(t_o, h, psi)
            p = _t_P(h, psi, a, ratio)
            dp = ratio * _t_gamma(h, psi, a) * p
        M += wi * (p * m_o + (1.0 - p) * m_x)
        dM += wi * (dp * (m_o - m_x) + p * dm_o + (1.0 - p) * dm_x)
        S += wi * (1.0 - p)
        dS -= wi * dp
    return M, dM, S, dS

def _t_check_ensemble(out, psis, w, a, ratio):
    # one Newton step on the independent M(h) = 0 and S(h) = 1/2 must not move the results
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
    h_c, h_half = out
    assert -1.0 < h_c < 1.0 and -1.0 < h_half < 1.0, out
    M, dM, _, _ = _t_ensemble(-h_c, psis, w, a, ratio)
    assert abs(M / dM) < 1e-6, (h_c, M / dM)
    _, _, S, dS = _t_ensemble(-h_half, psis, w, a, ratio)
    assert abs((S - 0.5) / dS) < 1e-6, (h_half, (S - 0.5) / dS)
    return out

out = _t_check_ensemble(ensemble_switching(np.array([0.3, 0.9, 1.4]), np.array([1.0, 0.0, 2.0]), 100.0, 1e9, 1.0),
                        [0.3, 1.4], [1.0, 2.0], 100.0, 1e9)
out2 = ensemble_switching(np.array([0.3, 1.4]), np.array([1.0, 2.0]), 100.0, 1e9, 1.0)
assert abs(out[0] - out2[0]) < 2e-6 and abs(out[1] - out2[1]) < 2e-6, (out, out2)

# --- test case 4 ---
# corners of the domain: a = 1000 with f0 / rate = 1e5 for one particle at psi = 0
# (closed-form median), and a = 40 with f0 / rate = 1e13 for two particles
import numpy as np
from scipy.integrate import quad as _t_quad
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv

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

def _t_gamma(x, psi, a):
    # escape rate over f0: both routes, exponent 2 a Delta_e because E = 2 K V e. Within a grid
    # step of +-h_sw two extrema may merge on the grid; the rate is negligible there.
    mins, maxs = _t_extrema(x, psi)
    if len(mins) < 2 or len(maxs) < 2:
        return 0.0
    t_o = max(mins, key=np.cos)
    e_o = _t_energy(t_o, x, psi)
    return sum(np.exp(-2.0 * a * (_t_energy(t, x, psi) - e_o)) for t in maxs)

def _t_P(h, psi, a, ratio):
    # survival probability exp(-(f0 / rate) int_h^h_sw Gamma / f0), quadrature on the grid barriers
    hs = _t_fold(psi)[0]
    if h >= hs:
        return 1.0
    if h <= -hs:
        return 0.0
    I = _t_quad(lambda x: _t_gamma(x, psi, a), h, hs, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    return float(np.exp(-ratio * I))

def _t_median0(a, ratio):
    # psi = 0: P = 1/2 where (f0 / rate) * I = ln 2, solved with the inverse erfc
    return -1.0 + _t_erfcinv(np.log(2.0) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)

def _t_dm(t, h, psi):
    # dm/dh along a minimum theta(h): dtheta/dh = -sin(theta - psi) / e''(theta)
    return np.sin(t - psi) ** 2 / (np.cos(2.0 * t) + h * np.cos(t - psi))

def _t_ensemble(h, psis, w, a, ratio):
    # expected ensemble magnetization M, switched weight S and their derivatives in h
    w = np.asarray(w, dtype=float) / np.sum(w)
    M = dM = S = dS = 0.0
    for psi, wi in zip(psis, w):
        hs = _t_fold(psi)[0]
        mins = _t_minima(h, psi)
        t_x = min(mins, key=np.cos)        # the other minimum, cos(theta) < 0
        m_x, dm_x = np.cos(t_x - psi), _t_dm(t_x, h, psi)
        if h <= -hs:
            p = dp = m_o = dm_o = 0.0
        else:
            t_o = max(mins, key=np.cos)
            m_o, dm_o = np.cos(t_o - psi), _t_dm(t_o, h, psi)
            p = _t_P(h, psi, a, ratio)
            dp = ratio * _t_gamma(h, psi, a) * p
        M += wi * (p * m_o + (1.0 - p) * m_x)
        dM += wi * (dp * (m_o - m_x) + p * dm_o + (1.0 - p) * dm_x)
        S += wi * (1.0 - p)
        dS -= wi * dp
    return M, dM, S, dS

def _t_check_ensemble(out, psis, w, a, ratio):
    # one Newton step on the independent M(h) = 0 and S(h) = 1/2 must not move the results
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
    h_c, h_half = out
    assert -1.0 < h_c < 1.0 and -1.0 < h_half < 1.0, out
    M, dM, _, _ = _t_ensemble(-h_c, psis, w, a, ratio)
    assert abs(M / dM) < 1e-6, (h_c, M / dM)
    _, _, S, dS = _t_ensemble(-h_half, psis, w, a, ratio)
    assert abs((S - 0.5) / dS) < 1e-6, (h_half, (S - 0.5) / dS)
    return out

out = ensemble_switching(np.array([0.0]), np.array([1.0]), 1000.0, 1e5, 1.0)
hm = _t_median0(1000.0, 1e5)
assert abs(out[0] + hm) < 1e-6 and abs(out[1] + hm) < 1e-6, (out, hm)
_t_check_ensemble(ensemble_switching(np.array([0.0, 0.6]), np.array([1.0, 1.0]), 40.0, 1e13, 1.0),
                  [0.0, 0.6], [1.0, 1.0], 40.0, 1e13)

# --- test case 5 ---
# one particle at psi = 0 over the domain: the magnetization is 2 P - 1, so h_c and h_half
# are both minus the closed-form median switching field
import numpy as np
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv

def _t_median0(a, ratio):
    # psi = 0: P = 1/2 where (f0 / rate) * I = ln 2, solved with the inverse erfc
    return -1.0 + _t_erfcinv(np.log(2.0) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)

for _t_a, _t_ratio in ((60.0, 1e7), (150.0, 1e11), (300.0, 1e6), (800.0, 1e12)):
    out = ensemble_switching(np.array([0.0]), np.array([1.0]), _t_a, _t_ratio, 1.0)
    hm = _t_median0(_t_a, _t_ratio)
    assert abs(out[0] + hm) < 1e-6, (_t_a, _t_ratio, out, hm)
    assert abs(out[1] + hm) < 1e-6, (_t_a, _t_ratio, out, hm)

# --- test case 6 ---
# dynamic coercivity of a two-particle ensemble: both fields fall as the sweep gets slower
# (larger f0 / rate) and rise as the particles get more stable (larger a)
import numpy as np
from scipy.integrate import quad as _t_quad

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

def _t_gamma(x, psi, a):
    # escape rate over f0: both routes, exponent 2 a Delta_e because E = 2 K V e. Within a grid
    # step of +-h_sw two extrema may merge on the grid; the rate is negligible there.
    mins, maxs = _t_extrema(x, psi)
    if len(mins) < 2 or len(maxs) < 2:
        return 0.0
    t_o = max(mins, key=np.cos)
    e_o = _t_energy(t_o, x, psi)
    return sum(np.exp(-2.0 * a * (_t_energy(t, x, psi) - e_o)) for t in maxs)

def _t_P(h, psi, a, ratio):
    # survival probability exp(-(f0 / rate) int_h^h_sw Gamma / f0), quadrature on the grid barriers
    hs = _t_fold(psi)[0]
    if h >= hs:
        return 1.0
    if h <= -hs:
        return 0.0
    I = _t_quad(lambda x: _t_gamma(x, psi, a), h, hs, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    return float(np.exp(-ratio * I))

def _t_dm(t, h, psi):
    # dm/dh along a minimum theta(h): dtheta/dh = -sin(theta - psi) / e''(theta)
    return np.sin(t - psi) ** 2 / (np.cos(2.0 * t) + h * np.cos(t - psi))

def _t_ensemble(h, psis, w, a, ratio):
    # expected ensemble magnetization M, switched weight S and their derivatives in h
    w = np.asarray(w, dtype=float) / np.sum(w)
    M = dM = S = dS = 0.0
    for psi, wi in zip(psis, w):
        hs = _t_fold(psi)[0]
        mins = _t_minima(h, psi)
        t_x = min(mins, key=np.cos)        # the other minimum, cos(theta) < 0
        m_x, dm_x = np.cos(t_x - psi), _t_dm(t_x, h, psi)
        if h <= -hs:
            p = dp = m_o = dm_o = 0.0
        else:
            t_o = max(mins, key=np.cos)
            m_o, dm_o = np.cos(t_o - psi), _t_dm(t_o, h, psi)
            p = _t_P(h, psi, a, ratio)
            dp = ratio * _t_gamma(h, psi, a) * p
        M += wi * (p * m_o + (1.0 - p) * m_x)
        dM += wi * (dp * (m_o - m_x) + p * dm_o + (1.0 - p) * dm_x)
        S += wi * (1.0 - p)
        dS -= wi * dp
    return M, dM, S, dS

def _t_check_ensemble(out, psis, w, a, ratio):
    # one Newton step on the independent M(h) = 0 and S(h) = 1/2 must not move the results
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
    h_c, h_half = out
    assert -1.0 < h_c < 1.0 and -1.0 < h_half < 1.0, out
    M, dM, _, _ = _t_ensemble(-h_c, psis, w, a, ratio)
    assert abs(M / dM) < 1e-6, (h_c, M / dM)
    _, _, S, dS = _t_ensemble(-h_half, psis, w, a, ratio)
    assert abs((S - 0.5) / dS) < 1e-6, (h_half, (S - 0.5) / dS)
    return out

_t_hc = [ensemble_switching(np.array([0.2, 0.7]), np.array([1.0, 1.0]), 150.0, r, 1.0) for r in (1e6, 1e9, 1e12)]
assert _t_hc[0][0] > _t_hc[1][0] > _t_hc[2][0], _t_hc
assert _t_hc[0][1] > _t_hc[1][1] > _t_hc[2][1], _t_hc
_t_hot = ensemble_switching(np.array([0.2, 0.7]), np.array([1.0, 1.0]), 80.0, 1e9, 1.0)
assert _t_hot[0] < _t_hc[1][0] and _t_hot[1] < _t_hc[1][1], (_t_hot, _t_hc[1])
_t_check_ensemble(_t_hot, [0.2, 0.7], [1.0, 1.0], 80.0, 1e9)

# --- test case 7 ---
# every invalid input of the prompt raises ValueError: arrays that are not 1-D of the
# same nonzero length, a psi outside [0, pi/2], a negative or nonfinite weight, weights adding up to
# zero, a, f0 or rate not positive and finite, a or f0 / rate outside the domain
import numpy as np
for _t_bad in ((np.array([0.1, 0.2]), np.array([1.0]), 100.0, 1e9, 1.0), (np.array([[0.1, 0.2]]), np.array([[1.0, 1.0]]), 100.0, 1e9, 1.0), (np.array([]), np.array([]), 100.0, 1e9, 1.0), (np.array([0.1, 0.2]), np.array([1.0, np.inf]), 100.0, 1e9, 1.0), (np.array([0.1, 0.2]), np.array([1.0, np.nan]), 100.0, 1e9, 1.0), (np.array([0.1]), np.array([1.0]), 100.0, 1e9, 0.0), (np.array([0.1]), np.array([1.0]), 100.0, -1e9, 1.0), (np.array([0.1]), np.array([1.0]), float("nan"), 1e9, 1.0), (np.array([0.1]), np.array([1.0]), 100.0, 1e9, np.inf), (np.array([0.1, 1.7]), np.array([1.0, 1.0]), 100.0, 1e9, 1.0), (np.array([0.1, 0.2]), np.array([1.0, -1.0]), 100.0, 1e9, 1.0), (np.array([0.1, 0.2]), np.array([0.0, 0.0]), 100.0, 1e9, 1.0), (np.array([0.1]), np.array([1.0]), 30.0, 1e9, 1.0), (np.array([0.1]), np.array([1.0]), 100.0, 1e14, 1.0), (np.array([-0.1]), np.array([1.0]), 100.0, 1e9, 1.0), (np.array([0.1]), np.array([1.0]), 1001.0, 1e9, 1.0), (np.array([0.1]), np.array([1.0]), 100.0, 1e4, 1.0)):
    try:
        ensemble_switching(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("ensemble_switching%r must raise ValueError" % (_t_bad,))
