# --- test case 0 ---
# one particle at psi = 0 (integrated chain: barriers, survival probability, branch
# magnetization): its field-axis magnetization is 2 P - 1, so h_c = h_half = minus the
# field where the closed-form survival probability is 1/2
import numpy as np
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv

def _t_field0(a, ratio, q):
    # psi = 0: both barriers are (1 + h)**2 / 2, P(h) = exp(-ratio sqrt(pi / a) [erfc(sqrt(a) (1 + h)) -
    # erfc(2 sqrt(a))]); field where P = q, from the inverse erfc
    return -1.0 + _t_erfcinv(np.log(1.0 / q) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)

out = ensemble_switching(np.array([0.0]), np.array([0.7]), 250.0, 2e9, 0.2)
assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out), out
hm = _t_field0(250.0, 1e10, 0.5)
assert abs(out[0] + hm) < 1e-6 and abs(out[1] + hm) < 1e-6, (out, hm)

# --- test case 1 ---
# an easy-axis particle (psi = 0, weight 2) with a hard-axis particle (psi = pi/2, weight 1).
# The hard-axis particle has the field-axis magnetization h in both minima and has left its
# original minimum just below h = 1, so with P0 the closed-form survival probability at psi = 0
# the ensemble magnetization is (2/3) (2 P0 - 1) + h / 3 and the switched weight is
# 1/3 + (2/3) (1 - P0): h_c solves 4 P0(-h_c) - 2 - h_c = 0 and h_half has P0(-h_half) = 3/4.
# The two fields differ because of the reversible rotation of the hard-axis particle.
import numpy as np
from scipy.optimize import brentq as _t_brentq
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv

def _t_P0(h, a, ratio):
    # psi = 0: both barriers are (1 + h)**2 / 2, so the escape integral is a difference of erfc
    I = np.sqrt(np.pi / a) * (_t_erfc(np.sqrt(a) * (1.0 + h)) - _t_erfc(2.0 * np.sqrt(a)))
    return float(np.exp(-ratio * I))

def _t_field0(a, ratio, q):
    # psi = 0: field where P0 = q, from the inverse erfc
    return -1.0 + _t_erfcinv(np.log(1.0 / q) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)

out = ensemble_switching(np.array([0.0, np.pi / 2]), np.array([2.0, 1.0]), 70.0, 3e10, 300.0)
assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out), out
_t_hc = -_t_brentq(lambda x: 4.0 * _t_P0(x, 70.0, 1e8) - 2.0 + x, -1.0, 1.0, xtol=1e-15, rtol=1e-15)
_t_hh = -_t_field0(70.0, 1e8, 0.75)
assert abs(out[0] - _t_hc) < 1e-6 and abs(out[1] - _t_hh) < 1e-6, (out, _t_hc, _t_hh)

# --- test case 2 ---
# four particles at intermediate angles, unequal weights: one Newton step on the independently
# computed ensemble magnetization and switched weight must not move h_c and h_half
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

_t_ps, _t_ws = [0.1, 0.45, 0.95, 1.35], [1.0, 2.0, 1.5, 0.5]
_t_check_ensemble(ensemble_switching(np.array(_t_ps), np.array(_t_ws), 120.0, 1e10, 100.0),
                  _t_ps, _t_ws, 120.0, 1e8)

# --- test case 3 ---
# easy axes spread over the quarter circle with weights sin(psi) (zero weight at psi = 0, a
# particle at psi = pi/2), a hot fast-swept ensemble near the edge of the domain; Newton-step check
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

_t_ps = list(np.linspace(0.0, np.pi / 2, 6))
_t_ws = [float(np.sin(p)) for p in _t_ps]
_t_check_ensemble(ensemble_switching(np.array(_t_ps), np.array(_t_ws), 45.0, 1e10, 0.01),
                  _t_ps, _t_ws, 45.0, 1e12)
