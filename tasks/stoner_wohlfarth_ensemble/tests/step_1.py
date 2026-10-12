# --- test case 0 ---
# reversible part of the branch (no jump yet)
import numpy as np


def _crown_case():
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
        if len(mins) not in (1, 2):
            raise RuntimeError("grid search failed: %r" % ((h, psi, mins),))
        return float(np.cos(max(mins, key=np.cos) - psi))

    for _t_h, _t_p in ((2.0, 0.4), (0.0, 0.7), (-0.3, 0.35), (0.6, 1.3), (-0.2, 1.0)):
        m = branch_magnetization(_t_h, _t_p)
        assert isinstance(m, float)
        assert abs(m - _t_m(_t_h, _t_p)) < 1e-8, (_t_h, _t_p, m)


_crown_case()

# --- test case 1 ---
# just before and just after the jump at h = -h_sw(psi)
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
        if len(mins) not in (1, 2):
            raise RuntimeError("grid search failed: %r" % ((h, psi, mins),))
        return float(np.cos(max(mins, key=np.cos) - psi))

    for _t_p in (0.2, np.pi / 6, np.pi / 4, 1.0, 1.4):
        _t_hs = _t_fold(_t_p)[0]
        m_before = branch_magnetization(-_t_hs + 1.5e-3, _t_p)
        m_after = branch_magnetization(-_t_hs - 1.5e-3, _t_p)
        assert abs(m_before - _t_m(-_t_hs + 1.5e-3, _t_p)) < 1e-8, (_t_p, m_before)
        assert abs(m_after - _t_m(-_t_hs - 1.5e-3, _t_p)) < 1e-8, (_t_p, m_after)


_crown_case()

# --- test case 2 ---
# the end points psi = 0 and psi = pi/2, and angles just inside them
import numpy as np


def _crown_case():
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
        if len(mins) not in (1, 2):
            raise RuntimeError("grid search failed: %r" % ((h, psi, mins),))
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


_crown_case()

# --- test case 3 ---
# hysteresis: the particle keeps a minimum that is not the global one
import numpy as np


def _crown_case():
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
        if len(mins) not in (1, 2):
            raise RuntimeError("grid search failed: %r" % ((h, psi, mins),))
        return float(np.cos(max(mins, key=np.cos) - psi))

    for _t_h, _t_p in ((-0.3, 0.2), (-0.45, 1.2), (-0.6, 0.15), (-1.7, 0.9), (3.0, 1.1)):
        m = branch_magnetization(_t_h, _t_p)
        assert abs(m - _t_m(_t_h, _t_p)) < 1e-8, (_t_h, _t_p, m)


_crown_case()

# --- test case 4 ---
# psi outside [0, pi/2] (on either side) raises ValueError
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((0.3, -0.01), (0.3, np.pi / 2 + 0.01)):
        try:
            branch_magnetization(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("branch_magnetization%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 5 ---
# exact parametric form of the equilibrium curve: e'(theta) = 0 gives h = -sin(2 theta) / (2 sin(theta - psi)),
# so each angle theta of the followed minimum (between the fold angle, where tan(theta)**3 = -tan(psi), and
# psi, with e'' > 0) fixes a field h and m = cos(theta - psi) without any root finding; the minimum left
# after the jump is theta + pi at the field -h (the energy at h is the energy at -h shifted by pi).
# h_sw = -h(fold angle). Fields at least 1.5e-3 away from the jump.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_h_of(t, psi):
        return -np.sin(2.0 * t) / (2.0 * np.sin(t - psi))

    _t_n = 0
    for _t_p in (0.05, 0.3, 0.9, 1.3, 1.55, np.pi / 2 - 1e-6):
        _t_fold = -np.arctan(np.tan(_t_p) ** (1.0 / 3.0))
        _t_hsw = -_t_h_of(_t_fold, _t_p)
        for _t_u in (0.004, 0.02, 0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.96, 0.99, 0.998):
            t = _t_fold + _t_u * (_t_p - _t_fold)
            h = _t_h_of(t, _t_p)
            if np.cos(2.0 * t) + h * np.cos(t - _t_p) <= 0.0:
                raise RuntimeError("not a minimum: %r" % ((_t_p, t, h),))
            if h + _t_hsw >= 1.5e-3:
                # still in the followed minimum
                m = branch_magnetization(h, _t_p)
                assert abs(m - np.cos(t - _t_p)) < 1e-8, (_t_p, h, m, np.cos(t - _t_p))
                _t_n += 1
            if h > _t_hsw + 1.5e-3:
                # the reversed field -h is beyond -h_sw: the particle sits in the minimum theta + pi
                m = branch_magnetization(-h, _t_p)
                assert abs(m + np.cos(t - _t_p)) < 1e-8, (_t_p, -h, m, -np.cos(t - _t_p))
                _t_n += 1
    if _t_n < 60:
        raise RuntimeError("too few points: %d" % _t_n)


_crown_case()

# --- test case 6 ---
# psi = pi/4 in closed form (h_sw = 1/2 there). With u = theta - pi/4, e'(theta) = 0 reads
# sin(u)**2 - h sin(u) - 1/2 = 0, so sin(u) = (h -+ sqrt(h**2 + 2)) / 2, and e'' = cos(u) (h - 2 sin(u)).
# The followed minimum has sin(u) = (h - sqrt(h**2 + 2)) / 2 with cos(u) > 0 (it exists while h > -1/2);
# after the jump the minimum has sin(u) = (h + sqrt(h**2 + 2)) / 2 with cos(u) < 0; m = cos(u), evaluated
# with 40 digits. Fields include |h + h_sw| = 1e-3 (h = -0.499 and h = -0.501), the inclusive ends of the
# accurate range.
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_m45(h):
        with _t_mp.workdps(40):
            h = _t_mp.mpf(h)
            r = _t_mp.sqrt(h * h + 2)
            if h > -0.5:
                s = (h - r) / 2
                return float(_t_mp.sqrt(1 - s * s))
            s = (h + r) / 2
            return -float(_t_mp.sqrt(1 - s * s))

    for _t_h in (6.0, 1.0, 0.5, 0.2, 0.0, -0.2, -0.4, -0.47, -0.499):
        m = branch_magnetization(_t_h, np.pi / 4)
        assert abs(m - _t_m45(_t_h)) < 1e-8, (_t_h, m, _t_m45(_t_h))
    for _t_h in (-0.501, -0.53, -0.7, -1.0, -3.0):
        m = branch_magnetization(_t_h, np.pi / 4)
        assert abs(m - _t_m45(_t_h)) < 1e-8, (_t_h, m, _t_m45(_t_h))


_crown_case()
