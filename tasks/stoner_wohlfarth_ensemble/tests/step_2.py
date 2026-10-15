# --- test case 0 ---
# at zero field both barriers are 1/2 for every angle
import numpy as np


def _crown_case():
    import numpy as np

    def _t_contract(h, psi):
        # the stated output contract: two values with the shape of np.asarray(h), float numpy arrays for an
        # array h (also a one-element or one-row array), float scalars (Python float or numpy floating) for a
        # scalar h
        low, high = escape_barriers(h, psi)
        if not isinstance(h, np.ndarray):
            return all(isinstance(v, (float, np.floating)) for v in (low, high))
        return all(isinstance(v, np.ndarray) and v.shape == h.shape and np.issubdtype(v.dtype, np.floating)
                   for v in (low, high))

    assert all(_t_contract(_t_h, _t_p) for _t_h, _t_p in (
        (0.1, 0.4), (-0.2, np.pi / 2), (np.array([0.1, -0.2, 0.3]), 0.4), (np.array([0.2]), 1.0),
        (np.array([[0.1, -0.1, 0.0]]), 0.0), (np.zeros((2, 3)), np.pi / 2))), "output contract"
    _t_low = []
    for _t_p in (0.0, 0.4, np.pi / 4, 1.3, np.pi / 2):
        _t_low.append(escape_barriers(0.0, _t_p))
    assert all(abs(float(lo) - 0.5) < 1e-8 and abs(float(hi) - 0.5) < 1e-8 for lo, hi in _t_low), _t_low
    # a weak scalar field already splits or shifts them: closed forms at psi = 0, (1 + h)**2 / 2 for
    # both, and at psi = pi/2, (1 -+ |h|)**2 / 2
    for _t_h in (0.05, -0.05):
        low, high = escape_barriers(_t_h, 0.0)
        assert abs(float(low) - 0.5 * (1.0 + _t_h) ** 2) < 1e-8 and abs(float(high) - 0.5 * (1.0 + _t_h) ** 2) < 1e-8, (_t_h, low, high)
        low, high = escape_barriers(_t_h, np.pi / 2)
        assert abs(float(low) - 0.5 * (1.0 - abs(_t_h)) ** 2) < 1e-8, (_t_h, low)
        assert abs(float(high) - 0.5 * (1.0 + abs(_t_h)) ** 2) < 1e-8, (_t_h, high)


_crown_case()

# --- test case 1 ---
# closed forms at psi = 0 and psi = pi/2
import numpy as np


def _crown_case():
    import numpy as np
    h = np.array([-0.9, -0.5, -0.1, 0.2, 0.7, 0.95])
    # entries are read by position, so the output must have the shape of h
    low, high = escape_barriers(h, 0.0)
    assert all(abs(low[i] - 0.5 * (1.0 + h[i]) ** 2) < 1e-8 for i in range(h.size)), low
    assert all(abs(high[i] - 0.5 * (1.0 + h[i]) ** 2) < 1e-8 for i in range(h.size)), high
    low, high = escape_barriers(h, np.pi / 2)
    assert all(abs(low[i] - 0.5 * (1.0 - abs(h[i])) ** 2) < 1e-8 for i in range(h.size)), low
    assert all(abs(high[i] - 0.5 * (1.0 + abs(h[i])) ** 2) < 1e-8 for i in range(h.size)), high


_crown_case()

# --- test case 2 ---
# other angles against the grid search, with a 2-D input
import numpy as np


def _crown_case():
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

    def _t_barriers(h, psi):
        # barriers from the original minimum (the one with cos(theta) > 0) over both maxima
        mins, maxs = _t_extrema(h, psi)
        if len(mins) != 2 or len(maxs) != 2:
            raise RuntimeError("grid search failed: %r" % ((h, psi, mins, maxs),))
        t_o = max(mins, key=np.cos)
        e_o = _t_energy(t_o, h, psi)
        return tuple(sorted(_t_energy(t, h, psi) - e_o for t in maxs))

    for _t_p in (0.2, np.pi / 6, np.pi / 4, 1.0, 1.4):
        _t_hs = _t_fold(_t_p)[0]
        h = np.array([[-0.9, -0.4, 0.0], [0.3, 0.7, 0.95]]) * _t_hs
        low, high = escape_barriers(h, _t_p)
        for idx in np.ndindex(h.shape):
            t_lo, t_hi = _t_barriers(h[idx], _t_p)
            assert abs(low[idx] - t_lo) < 1e-8 and abs(high[idx] - t_hi) < 1e-8, (_t_p, h[idx])


_crown_case()

# --- test case 3 ---
# fields just inside both ends of the accurate range (h_sw - |h| >= 1e-3)
import numpy as np


def _crown_case():
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

    def _t_barriers(h, psi):
        # barriers from the original minimum (the one with cos(theta) > 0) over both maxima
        mins, maxs = _t_extrema(h, psi)
        if len(mins) != 2 or len(maxs) != 2:
            raise RuntimeError("grid search failed: %r" % ((h, psi, mins, maxs),))
        t_o = max(mins, key=np.cos)
        e_o = _t_energy(t_o, h, psi)
        return tuple(sorted(_t_energy(t, h, psi) - e_o for t in maxs))

    for _t_p in (0.1, np.pi / 6, np.pi / 4, 1.2):
        _t_hs = _t_fold(_t_p)[0]
        for x in (-_t_hs + 1.1e-3, _t_hs - 1.1e-3):
            low, high = escape_barriers(x, _t_p)
            t_lo, t_hi = _t_barriers(x, _t_p)
            assert abs(float(low) - t_lo) < 1e-8 and abs(float(high) - t_hi) < 1e-8, (_t_p, x)


_crown_case()

# --- test case 4 ---
# each stated error rule on each side: psi below 0 and above pi/2, a scalar h on the boundary |h| = h_sw
# (psi = 0, h_sw = 1), and an array with one entry beyond -h_sw (psi = pi/4, h_sw = 1/2)
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((0.0, -0.01), (0.0, np.pi / 2 + 0.01), (1.0, 0.0), (np.array([0.1, -0.6]), np.pi / 4)):
        try:
            escape_barriers(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("escape_barriers%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 5 ---
# psi = pi/4 in closed form (h_sw = 1/2 there). With u = theta - pi/4, e'(theta) = 0 reads
# sin(u)**2 - h sin(u) - 1/2 = 0: s1 = (h - r) / 2 and s2 = (h + r) / 2 with r = sqrt(h**2 + 2), each giving
# two angles with cos(u) = +-c (c = sqrt(1 - s**2)). There e = 1/4 + cos(u) (sin(u) / 2 - h) and
# e'' = cos(u) (h - 2 sin(u)), so the original minimum (cos(theta) > 0) is s1 with cos(u) = +c1 and the
# maxima are s1 with -c1 and s2 with +c2: barriers c1 (3h + r) / 2 and (c2 (r - 3h) + c1 (3h + r)) / 4,
# evaluated with 40 digits. Fields up to h_sw - |h| = 1e-3 (|h| = 0.499), as a 1-D array, as a one-row
# 2-D batch (entries read at [0, j]) and as scalars.
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_b45(h):
        with _t_mp.workdps(40):
            h = _t_mp.mpf(float(h))
            r = _t_mp.sqrt(h * h + 2)
            s1, s2 = (h - r) / 2, (h + r) / 2
            c1, c2 = _t_mp.sqrt(1 - s1 * s1), _t_mp.sqrt(1 - s2 * s2)
            b1 = c1 * (3 * h + r) / 2
            b2 = (c2 * (r - 3 * h) + c1 * (3 * h + r)) / 4
            return float(min(b1, b2)), float(max(b1, b2))

    h = np.array([-0.499, -0.47, -0.4, -0.25, -0.1, 0.05, 0.2, 0.35, 0.45, 0.499])
    low, high = escape_barriers(h, np.pi / 4)
    _t_t = [_t_b45(x) for x in h]
    assert all(abs(low[i] - _t_t[i][0]) < 1e-8 for i in range(h.size)), (low, _t_t)
    assert all(abs(high[i] - _t_t[i][1]) < 1e-8 for i in range(h.size)), (high, _t_t)
    hb = np.array([[-0.48, 0.0, 0.3]])
    low, high = escape_barriers(hb, np.pi / 4)
    for j in range(3):
        t_lo, t_hi = _t_b45(hb[0, j])
        assert abs(low[0, j] - t_lo) < 1e-8 and abs(high[0, j] - t_hi) < 1e-8, (hb[0, j], low, high)
    for x in (-0.499, 0.499):
        low, high = escape_barriers(x, np.pi / 4)
        t_lo, t_hi = _t_b45(x)
        assert abs(float(low) - t_lo) < 1e-8 and abs(float(high) - t_hi) < 1e-8, (x, low, high, t_lo, t_hi)


_crown_case()
