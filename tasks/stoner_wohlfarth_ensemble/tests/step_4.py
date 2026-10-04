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


from scipy.integrate import quad as _t_quad
from scipy.special import erfc as _t_erfc, erfcinv as _t_erfcinv


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


def _t_P0(h, a, ratio):
    # psi = 0: both barriers are (1 + h)**2 / 2, so the escape integral is a difference of erfc
    I = np.sqrt(np.pi / a) * (_t_erfc(np.sqrt(a) * (1.0 + h)) - _t_erfc(2.0 * np.sqrt(a)))
    return float(np.exp(-ratio * I))


def _t_median0(a, ratio):
    # psi = 0: P = 1/2 where (f0 / rate) * I = ln 2, solved with the inverse erfc
    return -1.0 + _t_erfcinv(np.log(2.0) / ratio * np.sqrt(a / np.pi) + _t_erfc(2.0 * np.sqrt(a))) / np.sqrt(a)



def _t_mean0(a, ratio):
    # psi = 0: E[H_s] = h_sw - int P dh with the closed-form P
    hm = _t_median0(a, ratio)
    return 1.0 - _t_quad(lambda x: _t_P0(x, a, ratio), -1.0, 1.0, points=[hm],
                         epsabs=1e-13, epsrel=1e-13, limit=500)[0]


def _t_median_step(h_med, psi, a, ratio):
    # one Newton step on P(h) = 1/2 from the returned median, with dP/dh = ratio * Gamma/f0 * P
    p = _t_P(h_med, psi, a, ratio)
    return (p - 0.5) / (ratio * _t_gamma(h_med, psi, a) * p)


def _t_mean(psi, a, ratio, h_med):
    # E[H_s] = h_sw - int P dh: I = int Gamma / f0 and K = int P from one ODE run down from h_sw
    from scipy.integrate import solve_ivp
    hs = _t_fold(psi)[0]

    def rhs(x, y):
        g = _t_gamma(x, psi, a) if abs(x) < hs else 0.0
        return [-g, -np.exp(-ratio * y[0])]

    def done(x, y):
        return ratio * y[0] - 80.0

    done.terminal = True
    sol = solve_ivp(rhs, (hs, -hs), [0.0, 0.0], method="DOP853", rtol=1e-12,
                    atol=[1e-13 / ratio, 1e-15], events=done)
    return hs - sol.y[1, -1]


# --- test case 0: psi = 0 against the closed-form median and the quadrature of the closed form,
# including two corners of the domain (a = 40 with f0 / rate = 1e13, a = 1000 with 1e5) ---
for _t_a, _t_f0, _t_rate in ((100.0, 1e9, 1.0), (400.0, 1e9, 1e3), (40.0, 1e13, 1.0), (1000.0, 1e5, 1.0)):
    ratio = _t_f0 / _t_rate
    out = switching_field_statistics(0.0, _t_a, _t_f0, _t_rate)
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
    assert abs(out[0] - _t_median0(_t_a, ratio)) < 1e-7, (out, _t_median0(_t_a, ratio))
    assert abs(out[1] - _t_mean0(_t_a, ratio)) < 1e-7, (out, _t_mean0(_t_a, ratio))

# --- test case 1: psi = pi/4: median by one Newton step, mean by a windowed quadrature ---
out = switching_field_statistics(np.pi / 4, 100.0, 1e9, 1.0)
assert abs(_t_median_step(out[0], np.pi / 4, 100.0, 1e9)) < 1e-7, out
assert abs(out[1] - _t_mean(np.pi / 4, 100.0, 1e9, out[0])) < 1e-7, out

# --- test case 2: a cold particle (a = 400) in a fast sweep (f0 / rate = 1e6) at psi = 1.2 ---
out = switching_field_statistics(1.2, 400.0, 1e9, 1e3)
assert abs(_t_median_step(out[0], 1.2, 400.0, 1e6)) < 1e-7, out
assert abs(out[1] - _t_mean(1.2, 400.0, 1e6, out[0])) < 1e-7, out

# --- test case 3: psi = pi/2: the two minima are mirror images with the same magnetization and
# merge at h = 1, so the particle hops almost at once; with f0 / rate = 1e5 the median is
# near 1 - ln 2 / 1e5 and the mean near 1 - 1e-5 ---
def _t_I90(h, a):
    # psi = pi/2, 0 <= h < 1: barriers (1 - h)**2 / 2 and (1 + h)**2 / 2, integral in closed form
    from scipy.special import erf
    return 0.5 * np.sqrt(np.pi / a) * (erf(np.sqrt(a) * (1.0 - h)) + _t_erfc(np.sqrt(a) * (1.0 + h))
                                      - _t_erfc(2.0 * np.sqrt(a)))


from scipy.optimize import brentq as _t_brentq
_t_lo = 1.0 - 200.0 / 1e5         # P < exp(-150) below this field
_t_hm90 = _t_brentq(lambda x: 1e5 * _t_I90(x, 40.0) - np.log(2.0), _t_lo, 1.0, xtol=1e-15, rtol=1e-15)
_t_mean90 = 1.0 - _t_quad(lambda x: np.exp(-1e5 * _t_I90(x, 40.0)), _t_lo, 1.0, points=[_t_hm90],
                          epsabs=1e-15, epsrel=1e-13, limit=500)[0]
out = switching_field_statistics(np.pi / 2, 40.0, 1e9, 1e4)
assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out)
assert abs(out[0] - _t_hm90) < 1e-7 and abs(out[1] - _t_mean90) < 1e-7, (out, _t_hm90, _t_mean90)
assert abs(out[0] - (1.0 - np.log(2.0) / 1e5)) < 2e-7 and abs(out[1] - (1.0 - 1e-5)) < 2e-7, out

# --- test case 4: bad angle, a, f0 or rate, or a or f0 / rate outside the domain, raises ValueError ---
for _t_bad in ((-0.1, 100.0, 1e9, 1.0), (1.7, 100.0, 1e9, 1.0), (0.5, float("nan"), 1e9, 1.0), (0.5, 100.0, -1.0, 1.0), (0.5, 100.0, 1e9, 0.0), (0.5, 39.0, 1e9, 1.0), (0.5, 1001.0, 1e9, 1.0), (0.5, 100.0, 1e4, 1.0), (0.5, 100.0, 1e13, 0.5)):
    try:
        switching_field_statistics(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("switching_field_statistics%r must raise ValueError" % (_t_bad,))
