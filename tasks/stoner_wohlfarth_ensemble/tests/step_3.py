# --- test case 0 ---
# psi = 0 against the closed form, across the switching window
import numpy as np
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv

def _t_P0(h, a, ratio):
    # psi = 0: both barriers are (1 + h)**2 / 2, so the escape integral is a difference of erfc
    I = np.sqrt(np.pi / a) * (_t_erfc(np.sqrt(a) * (1.0 + h)) - _t_erfc(2.0 * np.sqrt(a)))
    return float(np.exp(-ratio * I))

def _t_median0(a, ratio):
    # psi = 0: P = 1/2 where (f0 / rate) * I = ln 2, solved with the inverse erfc
    return -1.0 + _t_erfcinv(np.log(2.0) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)

for _t_a, _t_rate in ((50.0, 1e-3), (100.0, 1.0), (400.0, 1e3)):
    ratio = 1e9 / _t_rate
    hm = _t_median0(_t_a, ratio)
    for x in (hm - 0.02, hm - 0.005, hm, hm + 0.005, hm + 0.02, 0.0):
        p = survival_probability(x, 0.0, _t_a, 1e9, _t_rate)
        assert isinstance(p, float)
        assert abs(p - _t_P0(x, _t_a, ratio)) < 1e-8, (_t_a, _t_rate, x, p)

# --- test case 1 ---
# other angles against an independent quadrature
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

for _t_p, _t_a, _t_rate, x in ((np.pi / 6, 100.0, 1.0, -0.37), (np.pi / 4, 100.0, 1.0, -0.355),
                               (1.2, 400.0, 1e3, -0.525), (0.3, 50.0, 1e-3, -0.24)):
    p = survival_probability(x, _t_p, _t_a, 1e9, _t_rate)
    target = _t_P(x, _t_p, _t_a, 1e9 / _t_rate)
    assert abs(p - target) < 1e-8, (_t_p, x, p, target)

# --- test case 2 ---
# P = 1 for h >= h_sw and P = 0 for h <= -h_sw, including h = +-h_sw exactly
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

for _t_p in (0.0, np.pi / 6, 1.2):
    _t_hs = _t_fold(_t_p)[0]
    for x, target in ((_t_hs + 1e-6, 1.0), (_t_hs + 0.5, 1.0), (-_t_hs - 1e-6, 0.0), (-_t_hs - 0.5, 0.0)):
        assert abs(survival_probability(x, _t_p, 100.0, 1e9, 1.0) - target) < 1e-8, (_t_p, x)
for _t_p in (0.0, np.pi / 2):          # h_sw = 1 exactly at both end angles
    assert abs(survival_probability(1.0, _t_p, 100.0, 1e9, 1.0) - 1.0) < 1e-8, _t_p
    assert abs(survival_probability(-1.0, _t_p, 100.0, 1e9, 1.0)) < 1e-8, _t_p

# --- test case 3 ---
# only the ratio f0 / rate enters
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

p1 = survival_probability(-0.36, np.pi / 4, 100.0, 1e9, 1.0)
p2 = survival_probability(-0.36, np.pi / 4, 100.0, 1e11, 100.0)
# p1 and p2 are two computed outputs, each accurate to 1e-8: their difference is held to 2e-8
assert abs(p1 - p2) < 2e-8 and abs(p1 - _t_P(-0.36, np.pi / 4, 100.0, 1e9)) < 1e-8

# --- test case 4 ---
# the corners of the stated domain, 40 <= a <= 1000 and 1e5 <= f0 / rate <= 1e13,
# against the closed forms at psi = 0 and psi = pi/2, across each switching window
import numpy as np
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv

def _t_P0(h, a, ratio):
    # psi = 0: both barriers are (1 + h)**2 / 2, so the escape integral is a difference of erfc
    I = np.sqrt(np.pi / a) * (_t_erfc(np.sqrt(a) * (1.0 + h)) - _t_erfc(2.0 * np.sqrt(a)))
    return float(np.exp(-ratio * I))

def _t_median0(a, ratio):
    # psi = 0: P = 1/2 where (f0 / rate) * I = ln 2, solved with the inverse erfc
    return -1.0 + _t_erfcinv(np.log(2.0) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)

for _t_a, _t_ratio in ((40.0, 1e5), (40.0, 1e13), (1000.0, 1e5), (1000.0, 1e13)):
    hm = _t_median0(_t_a, _t_ratio)
    w = 1.0 / np.sqrt(_t_a)
    for x in (hm - 0.3 * w, hm - 0.1 * w, hm, hm + 0.1 * w, hm + 0.3 * w):
        p = survival_probability(x, 0.0, _t_a, _t_ratio, 1.0)
        assert abs(p - _t_P0(x, _t_a, _t_ratio)) < 1e-8, (_t_a, _t_ratio, x, p)
# psi = pi/2 (h_sw = 1 exactly): the barriers are (1 -+ h)**2 / 2 and the particle leaves within
# about rate / f0 below h = 1, here 1e-13 and 1e-5
from scipy.special import erf as _t_erf
for _t_a, _t_ratio in ((40.0, 1e13), (1000.0, 1e5)):
    for x in (1.0 - 0.3 / _t_ratio, 1.0 - 1.0 / _t_ratio, 1.0 - 3.0 / _t_ratio):
        I = 0.5 * np.sqrt(np.pi / _t_a) * (_t_erf(np.sqrt(_t_a) * (1.0 - x)) + _t_erfc(np.sqrt(_t_a) * (1.0 + x))
                                          - _t_erfc(2.0 * np.sqrt(_t_a)))
        p = survival_probability(x, np.pi / 2, _t_a, _t_ratio, 1.0)
        assert abs(p - np.exp(-_t_ratio * I)) < 1e-8, (_t_a, _t_ratio, x, p)

# --- test case 5 ---
# bad angle, field, a, f0 or rate, or a or f0 / rate outside the domain, raises ValueError
import numpy as np
for _t_bad in ((-0.3, -0.1, 100.0, 1e9, 1.0), (-0.3, 1.6, 100.0, 1e9, 1.0), (float("nan"), 0.5, 100.0, 1e9, 1.0), (-0.3, 0.5, float("nan"), 1e9, 1.0), (-0.3, 0.5, 100.0, 0.0, 1.0), (-0.3, 0.5, 100.0, 1e9, float("inf")), (-0.3, 0.5, 39.0, 1e9, 1.0), (-0.3, 0.5, 1001.0, 1e9, 1.0), (-0.3, 0.5, 100.0, 1e4, 1.0), (-0.3, 0.5, 100.0, 1e13, 0.5)):
    try:
        survival_probability(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("survival_probability%r must raise ValueError" % (_t_bad,))
