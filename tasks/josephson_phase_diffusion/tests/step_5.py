# Independent targets: no Stratonovich / Reimann double integral. The stationary
# Fokker-Planck density p(x) = sum_n c_n exp(i n x) of dphi/dtau = i - sin(phi) + xi obeys
# c_{n+1} + a_n c_n - c_{n-1} = 0 with a_n = 2 theta n + 2 j i (j = imaginary unit);
# its decaying solution is a continued fraction, and v = i + 2 pi Im c_1. The effective
# diffusion follows from homogenization:
# D = theta int (1 + chi')**2 p dx, with the periodic corrector chi solving
# (i - sin x) chi' + theta chi'' = v - (i - sin x); g_n = (chi')_n obeys
# g_{n+1} + b_n g_n - g_{n-1} = 0 (n >= 2), b_n = 2 j i - 2 theta n, g_1 (b_1 + g_2/g_1) = 1.
# All in extended precision, so exponentially small results keep their relative accuracy.
# Every test case below is self-contained (its own imports and helper functions).

# --- test case 0 ---
# one junction: j_star is the bias whose (Bessel-integral) voltage is v_crit
import numpy as np
import mpmath as _t_mp

def _t_cf(i, theta, want_D=True):
    sgn = -1 if i < 0 else 1
    dps = 30 + int(1.8 / theta)
    with _t_mp.workdps(dps):
        a = _t_mp.mpf(abs(i))
        th = _t_mp.mpf(theta)
        N = 40 + int(1.3 * _t_mp.sqrt(2 * dps * 2.303 / th))
        J = _t_mp.mpc(0, 1)
        r = [_t_mp.mpc(0)] * (N + 2)
        dr = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 0, -1):
            r[n] = 1 / (2 * th * n + 2 * J * a + r[n + 1])
            dr[n] = -r[n] ** 2 * (2 * J + dr[n + 1])
        v = a + _t_mp.im(r[1])
        dv = 1 + _t_mp.im(dr[1])
        if not want_D:
            return float(sgn * v), float(dv)
        c = [_t_mp.mpc(0)] * (N + 1)
        c[0] = 1 / (2 * _t_mp.pi)
        for n in range(1, N + 1):
            c[n] = r[n] * c[n - 1]
        s = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 1, -1):
            s[n] = 1 / (-2 * th * n + 2 * J * a + s[n + 1])
        g = [_t_mp.mpc(0)] * (N + 1)
        g[1] = 1 / (2 * J * a - 2 * th + s[2])
        for n in range(2, N + 1):
            g[n] = s[n] * g[n - 1]
        h = np.array([_t_mp.conj(g[k]) for k in range(N, 0, -1)] + [_t_mp.mpc(1)] + g[1:], dtype=object)
        cf = np.array([_t_mp.conj(c[k]) for k in range(N, 0, -1)] + c, dtype=object)
        q = np.convolve(h, h)                       # Fourier coefficients of (1 + chi')**2, index -2N..2N
        tot = _t_mp.fsum(q[N:3 * N + 1][::-1] * cf)  # sum_m q_m c_{-m}, |m| <= N
        D = th * 2 * _t_mp.pi * _t_mp.re(tot)
        return float(sgn * v), float(dv), float(D)

def _t_v_bessel(i, theta):
    # Single-integral form of the Ambegaokar-Halperin voltage (inner integral done exactly):
    # v = theta (1 - exp(-2 pi i / theta)) / int_0^{2 pi} I_0(2 sin(y/2) / theta) exp(-i y / theta) dy
    with _t_mp.workdps(40):
        a, th = _t_mp.mpf(i), _t_mp.mpf(theta)
        f = lambda y: _t_mp.besseli(0, 2 * _t_mp.sin(y / 2) / th) * _t_mp.exp(-a * y / th)
        pts = _t_mp.linspace(0, 2 * _t_mp.pi, 41)
        den = _t_mp.quad(f, pts)
        return float(th * (-_t_mp.expm1(-2 * _t_mp.pi * a / th)) / den)

def _t_rel(x, y):
    return abs(x - y) / abs(y)

def _t_array(c, r, theta0, j):
    # array voltage, dV/dJ and two-sided zero-frequency voltage noise from the continued fractions
    V = dV = S = 0.0
    for ck, rk in zip(c, r):
        v, dv, D = _t_cf(j / ck, theta0 / ck)
        V += ck * rk * v
        dV += rk * dv
        S += 2.0 * ck * rk * D
    return V, dV, S

def _t_check_array(out, c, r, theta0, v_crit):
    # one Newton step on V(J) = v_crit must move j_star by less than 1e-9 relative
    assert isinstance(out, tuple) and len(out) == 3 and all(isinstance(x, float) for x in out), out
    j, rd, sv = out
    assert 0.0 < j < 2.0 * max(c), out
    V, dV, S = _t_array(c, r, theta0, j)
    assert abs((V - v_crit) / dV) < 1e-9 * j, (out, V, v_crit)
    assert _t_rel(rd, dV) < 1e-8, (out, dV)
    assert _t_rel(sv, S) < 1e-8, (out, S)
    return out

_t_vc = _t_v_bessel(0.8, 0.05)
_t_o = _t_check_array(array_criterion(np.array([1.0]), np.array([1.0]), 0.05, _t_vc), [1.0], [1.0], 0.05, _t_vc)
assert _t_rel(_t_o[0], 0.8) < 1e-9, _t_o

# --- test case 1 ---
# three different junctions, low temperature, small voltage criterion
import numpy as np
import mpmath as _t_mp

def _t_cf(i, theta, want_D=True):
    sgn = -1 if i < 0 else 1
    dps = 30 + int(1.8 / theta)
    with _t_mp.workdps(dps):
        a = _t_mp.mpf(abs(i))
        th = _t_mp.mpf(theta)
        N = 40 + int(1.3 * _t_mp.sqrt(2 * dps * 2.303 / th))
        J = _t_mp.mpc(0, 1)
        r = [_t_mp.mpc(0)] * (N + 2)
        dr = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 0, -1):
            r[n] = 1 / (2 * th * n + 2 * J * a + r[n + 1])
            dr[n] = -r[n] ** 2 * (2 * J + dr[n + 1])
        v = a + _t_mp.im(r[1])
        dv = 1 + _t_mp.im(dr[1])
        if not want_D:
            return float(sgn * v), float(dv)
        c = [_t_mp.mpc(0)] * (N + 1)
        c[0] = 1 / (2 * _t_mp.pi)
        for n in range(1, N + 1):
            c[n] = r[n] * c[n - 1]
        s = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 1, -1):
            s[n] = 1 / (-2 * th * n + 2 * J * a + s[n + 1])
        g = [_t_mp.mpc(0)] * (N + 1)
        g[1] = 1 / (2 * J * a - 2 * th + s[2])
        for n in range(2, N + 1):
            g[n] = s[n] * g[n - 1]
        h = np.array([_t_mp.conj(g[k]) for k in range(N, 0, -1)] + [_t_mp.mpc(1)] + g[1:], dtype=object)
        cf = np.array([_t_mp.conj(c[k]) for k in range(N, 0, -1)] + c, dtype=object)
        q = np.convolve(h, h)                       # Fourier coefficients of (1 + chi')**2, index -2N..2N
        tot = _t_mp.fsum(q[N:3 * N + 1][::-1] * cf)  # sum_m q_m c_{-m}, |m| <= N
        D = th * 2 * _t_mp.pi * _t_mp.re(tot)
        return float(sgn * v), float(dv), float(D)

def _t_rel(x, y):
    return abs(x - y) / abs(y)

def _t_array(c, r, theta0, j):
    # array voltage, dV/dJ and two-sided zero-frequency voltage noise from the continued fractions
    V = dV = S = 0.0
    for ck, rk in zip(c, r):
        v, dv, D = _t_cf(j / ck, theta0 / ck)
        V += ck * rk * v
        dV += rk * dv
        S += 2.0 * ck * rk * D
    return V, dV, S

def _t_check_array(out, c, r, theta0, v_crit):
    # one Newton step on V(J) = v_crit must move j_star by less than 1e-9 relative
    assert isinstance(out, tuple) and len(out) == 3 and all(isinstance(x, float) for x in out), out
    j, rd, sv = out
    assert 0.0 < j < 2.0 * max(c), out
    V, dV, S = _t_array(c, r, theta0, j)
    assert abs((V - v_crit) / dV) < 1e-9 * j, (out, V, v_crit)
    assert _t_rel(rd, dV) < 1e-8, (out, dV)
    assert _t_rel(sv, S) < 1e-8, (out, S)
    return out

# (the weakest junction has a larger noise strength theta0 / c_k: E_J is proportional to I_c)
_t_c, _t_r = np.array([0.8, 1.0, 1.3]), np.array([1.2, 1.0, 0.7])
_t_check_array(array_criterion(_t_c, _t_r, 0.04, 0.01), _t_c, _t_r, 0.04, 0.01)

# --- test case 2 ---
# four junctions, strong noise, largest allowed criterion
import numpy as np
import mpmath as _t_mp

def _t_cf(i, theta, want_D=True):
    sgn = -1 if i < 0 else 1
    dps = 30 + int(1.8 / theta)
    with _t_mp.workdps(dps):
        a = _t_mp.mpf(abs(i))
        th = _t_mp.mpf(theta)
        N = 40 + int(1.3 * _t_mp.sqrt(2 * dps * 2.303 / th))
        J = _t_mp.mpc(0, 1)
        r = [_t_mp.mpc(0)] * (N + 2)
        dr = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 0, -1):
            r[n] = 1 / (2 * th * n + 2 * J * a + r[n + 1])
            dr[n] = -r[n] ** 2 * (2 * J + dr[n + 1])
        v = a + _t_mp.im(r[1])
        dv = 1 + _t_mp.im(dr[1])
        if not want_D:
            return float(sgn * v), float(dv)
        c = [_t_mp.mpc(0)] * (N + 1)
        c[0] = 1 / (2 * _t_mp.pi)
        for n in range(1, N + 1):
            c[n] = r[n] * c[n - 1]
        s = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 1, -1):
            s[n] = 1 / (-2 * th * n + 2 * J * a + s[n + 1])
        g = [_t_mp.mpc(0)] * (N + 1)
        g[1] = 1 / (2 * J * a - 2 * th + s[2])
        for n in range(2, N + 1):
            g[n] = s[n] * g[n - 1]
        h = np.array([_t_mp.conj(g[k]) for k in range(N, 0, -1)] + [_t_mp.mpc(1)] + g[1:], dtype=object)
        cf = np.array([_t_mp.conj(c[k]) for k in range(N, 0, -1)] + c, dtype=object)
        q = np.convolve(h, h)                       # Fourier coefficients of (1 + chi')**2, index -2N..2N
        tot = _t_mp.fsum(q[N:3 * N + 1][::-1] * cf)  # sum_m q_m c_{-m}, |m| <= N
        D = th * 2 * _t_mp.pi * _t_mp.re(tot)
        return float(sgn * v), float(dv), float(D)

def _t_rel(x, y):
    return abs(x - y) / abs(y)

def _t_array(c, r, theta0, j):
    # array voltage, dV/dJ and two-sided zero-frequency voltage noise from the continued fractions
    V = dV = S = 0.0
    for ck, rk in zip(c, r):
        v, dv, D = _t_cf(j / ck, theta0 / ck)
        V += ck * rk * v
        dV += rk * dv
        S += 2.0 * ck * rk * D
    return V, dV, S

def _t_check_array(out, c, r, theta0, v_crit):
    # one Newton step on V(J) = v_crit must move j_star by less than 1e-9 relative
    assert isinstance(out, tuple) and len(out) == 3 and all(isinstance(x, float) for x in out), out
    j, rd, sv = out
    assert 0.0 < j < 2.0 * max(c), out
    V, dV, S = _t_array(c, r, theta0, j)
    assert abs((V - v_crit) / dV) < 1e-9 * j, (out, V, v_crit)
    assert _t_rel(rd, dV) < 1e-8, (out, dV)
    assert _t_rel(sv, S) < 1e-8, (out, S)
    return out

_t_c, _t_r = np.array([0.5, 2.0, 1.0, 1.5]), np.array([2.0, 0.5, 1.0, 1.0])
_t_vc = 0.5 * float(np.sum(_t_c * _t_r))
_t_check_array(array_criterion(_t_c, _t_r, 0.4, _t_vc), _t_c, _t_r, 0.4, _t_vc)

# --- test case 3 ---
# inclusive limits: max(ics) / min(ics) = 5 with theta0 / c_k = 0.02 for the
# strongest junction, and a junction at the largest noise strength theta0 / c_k = 50
import numpy as np
import mpmath as _t_mp

def _t_cf(i, theta, want_D=True):
    sgn = -1 if i < 0 else 1
    dps = 30 + int(1.8 / theta)
    with _t_mp.workdps(dps):
        a = _t_mp.mpf(abs(i))
        th = _t_mp.mpf(theta)
        N = 40 + int(1.3 * _t_mp.sqrt(2 * dps * 2.303 / th))
        J = _t_mp.mpc(0, 1)
        r = [_t_mp.mpc(0)] * (N + 2)
        dr = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 0, -1):
            r[n] = 1 / (2 * th * n + 2 * J * a + r[n + 1])
            dr[n] = -r[n] ** 2 * (2 * J + dr[n + 1])
        v = a + _t_mp.im(r[1])
        dv = 1 + _t_mp.im(dr[1])
        if not want_D:
            return float(sgn * v), float(dv)
        c = [_t_mp.mpc(0)] * (N + 1)
        c[0] = 1 / (2 * _t_mp.pi)
        for n in range(1, N + 1):
            c[n] = r[n] * c[n - 1]
        s = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 1, -1):
            s[n] = 1 / (-2 * th * n + 2 * J * a + s[n + 1])
        g = [_t_mp.mpc(0)] * (N + 1)
        g[1] = 1 / (2 * J * a - 2 * th + s[2])
        for n in range(2, N + 1):
            g[n] = s[n] * g[n - 1]
        h = np.array([_t_mp.conj(g[k]) for k in range(N, 0, -1)] + [_t_mp.mpc(1)] + g[1:], dtype=object)
        cf = np.array([_t_mp.conj(c[k]) for k in range(N, 0, -1)] + c, dtype=object)
        q = np.convolve(h, h)                       # Fourier coefficients of (1 + chi')**2, index -2N..2N
        tot = _t_mp.fsum(q[N:3 * N + 1][::-1] * cf)  # sum_m q_m c_{-m}, |m| <= N
        D = th * 2 * _t_mp.pi * _t_mp.re(tot)
        return float(sgn * v), float(dv), float(D)

def _t_rel(x, y):
    return abs(x - y) / abs(y)

def _t_array(c, r, theta0, j):
    # array voltage, dV/dJ and two-sided zero-frequency voltage noise from the continued fractions
    V = dV = S = 0.0
    for ck, rk in zip(c, r):
        v, dv, D = _t_cf(j / ck, theta0 / ck)
        V += ck * rk * v
        dV += rk * dv
        S += 2.0 * ck * rk * D
    return V, dV, S

def _t_check_array(out, c, r, theta0, v_crit):
    # one Newton step on V(J) = v_crit must move j_star by less than 1e-9 relative
    assert isinstance(out, tuple) and len(out) == 3 and all(isinstance(x, float) for x in out), out
    j, rd, sv = out
    assert 0.0 < j < 2.0 * max(c), out
    V, dV, S = _t_array(c, r, theta0, j)
    assert abs((V - v_crit) / dV) < 1e-9 * j, (out, V, v_crit)
    assert _t_rel(rd, dV) < 1e-8, (out, dV)
    assert _t_rel(sv, S) < 1e-8, (out, S)
    return out

_t_c, _t_r = np.array([1.0, 5.0]), np.array([1.0, 0.4])
_t_check_array(array_criterion(_t_c, _t_r, 0.1, 0.5), _t_c, _t_r, 0.1, 0.5)
_t_c, _t_r = np.array([0.2, 1.0]), np.array([2.0, 1.0])
_t_check_array(array_criterion(_t_c, _t_r, 10.0, 0.3), _t_c, _t_r, 10.0, 0.3)

# --- test case 4 ---
# more arrays across the noise range (moderate, strong and mixed noise)
import numpy as np
import mpmath as _t_mp

def _t_cf(i, theta, want_D=True):
    sgn = -1 if i < 0 else 1
    dps = 30 + int(1.8 / theta)
    with _t_mp.workdps(dps):
        a = _t_mp.mpf(abs(i))
        th = _t_mp.mpf(theta)
        N = 40 + int(1.3 * _t_mp.sqrt(2 * dps * 2.303 / th))
        J = _t_mp.mpc(0, 1)
        r = [_t_mp.mpc(0)] * (N + 2)
        dr = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 0, -1):
            r[n] = 1 / (2 * th * n + 2 * J * a + r[n + 1])
            dr[n] = -r[n] ** 2 * (2 * J + dr[n + 1])
        v = a + _t_mp.im(r[1])
        dv = 1 + _t_mp.im(dr[1])
        if not want_D:
            return float(sgn * v), float(dv)
        c = [_t_mp.mpc(0)] * (N + 1)
        c[0] = 1 / (2 * _t_mp.pi)
        for n in range(1, N + 1):
            c[n] = r[n] * c[n - 1]
        s = [_t_mp.mpc(0)] * (N + 2)
        for n in range(N, 1, -1):
            s[n] = 1 / (-2 * th * n + 2 * J * a + s[n + 1])
        g = [_t_mp.mpc(0)] * (N + 1)
        g[1] = 1 / (2 * J * a - 2 * th + s[2])
        for n in range(2, N + 1):
            g[n] = s[n] * g[n - 1]
        h = np.array([_t_mp.conj(g[k]) for k in range(N, 0, -1)] + [_t_mp.mpc(1)] + g[1:], dtype=object)
        cf = np.array([_t_mp.conj(c[k]) for k in range(N, 0, -1)] + c, dtype=object)
        q = np.convolve(h, h)                       # Fourier coefficients of (1 + chi')**2, index -2N..2N
        tot = _t_mp.fsum(q[N:3 * N + 1][::-1] * cf)  # sum_m q_m c_{-m}, |m| <= N
        D = th * 2 * _t_mp.pi * _t_mp.re(tot)
        return float(sgn * v), float(dv), float(D)

def _t_rel(x, y):
    return abs(x - y) / abs(y)

def _t_array(c, r, theta0, j):
    # array voltage, dV/dJ and two-sided zero-frequency voltage noise from the continued fractions
    V = dV = S = 0.0
    for ck, rk in zip(c, r):
        v, dv, D = _t_cf(j / ck, theta0 / ck)
        V += ck * rk * v
        dV += rk * dv
        S += 2.0 * ck * rk * D
    return V, dV, S

def _t_check_array(out, c, r, theta0, v_crit):
    # one Newton step on V(J) = v_crit must move j_star by less than 1e-9 relative
    assert isinstance(out, tuple) and len(out) == 3 and all(isinstance(x, float) for x in out), out
    j, rd, sv = out
    assert 0.0 < j < 2.0 * max(c), out
    V, dV, S = _t_array(c, r, theta0, j)
    assert abs((V - v_crit) / dV) < 1e-9 * j, (out, V, v_crit)
    assert _t_rel(rd, dV) < 1e-8, (out, dV)
    assert _t_rel(sv, S) < 1e-8, (out, S)
    return out

_t_c, _t_r = np.array([1.0, 1.5]), np.array([1.0, 2.0])
_t_check_array(array_criterion(_t_c, _t_r, 0.3, 0.4), _t_c, _t_r, 0.3, 0.4)
_t_check_array(array_criterion(np.array([1.0]), np.array([2.0]), 5.0, 0.6), [1.0], [2.0], 5.0, 0.6)
_t_c, _t_r = np.array([0.7, 1.0, 2.1]), np.array([1.0, 0.8, 0.6])
_t_check_array(array_criterion(_t_c, _t_r, 0.15, 0.05), _t_c, _t_r, 0.15, 0.05)

# --- test case 5 ---
# inputs outside the stated limits raise ValueError
import numpy as np
for _t_bad in ((np.array([1.0, 2.0]), np.array([1.0]), 0.1, 0.1),
               (np.array([[1.0, 2.0]]), np.array([[1.0, 1.0]]), 0.1, 0.1),
               (np.array([]), np.array([]), 0.1, 0.1),
               (np.array([1.0, np.nan]), np.array([1.0, 1.0]), 0.1, 0.1),
               (np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.1, 0.1),
               (np.array([1.0, 6.0]), np.array([1.0, 1.0]), 0.2, 0.1),
               (np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0.03, 0.1),
               (np.array([1.0]), np.array([1.0]), 51.0, 0.1),
               (np.array([1.0]), np.array([1.0]), float("nan"), 0.1),
               (np.array([1.0]), np.array([1.0]), 0.1, 0.0),
               (np.array([1.0]), np.array([1.0]), 0.1, 0.51)):
    try:
        array_criterion(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("array_criterion%r must raise ValueError" % (_t_bad,))
