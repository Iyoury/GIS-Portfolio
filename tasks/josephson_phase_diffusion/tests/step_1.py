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
# moderate noise below, at and above the critical current
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

for _t_i, _t_th in ((0.5, 0.2), (0.9, 0.1), (1.0, 0.05), (1.3, 0.3), (2.5, 1.0), (0.2, 3.0)):
    out = mean_voltage(_t_i, _t_th)
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
    _t_v, _t_dv = _t_cf(_t_i, _t_th, want_D=False)
    assert _t_rel(out[0], _t_v) < 1e-9, (_t_i, _t_th, out, _t_v)
    assert _t_rel(out[1], _t_dv) < 1e-8, (_t_i, _t_th, out, _t_dv)

# --- test case 1 ---
# low noise below the critical current, exponentially small voltage
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

# (thermally activated phase slips; the forward and backward slips nearly cancel at small i)
for _t_i, _t_th in ((0.3, 0.03), (0.05, 0.05), (0.6, 0.02), (0.95, 0.025)):
    out = mean_voltage(_t_i, _t_th)
    _t_vb = _t_v_bessel(_t_i, _t_th)
    _t_v, _t_dv = _t_cf(_t_i, _t_th, want_D=False)
    assert _t_rel(_t_vb, _t_v) < 1e-12
    assert _t_rel(out[0], _t_vb) < 1e-9, (_t_i, _t_th, out, _t_vb)
    assert _t_rel(out[1], _t_dv) < 1e-8, (_t_i, _t_th, out, _t_dv)

# --- test case 2 ---
# equilibrium (i = 0): no voltage (|v| <= 1e-12), linear-response resistance 1 / I_0(1/theta)**2
import numpy as np
import mpmath as _t_mp

def _t_rel(x, y):
    return abs(x - y) / abs(y)

for _t_th in (0.02, 0.1, 1.0, 40.0):
    out = mean_voltage(0.0, _t_th)
    assert abs(out[0]) <= 1e-12, out
    _t_lin = 1.0 / float(_t_mp.besseli(0, 1.0 / _t_th)) ** 2
    assert _t_rel(out[1], _t_lin) < 1e-8, (_t_th, out, _t_lin)

# --- test case 3 ---
# reversed bias, strong bias and very strong noise; noiseless and ohmic bounds
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

for _t_i, _t_th in ((-0.7, 0.15), (-4.0, 0.05), (10.0, 0.02), (7.5, 50.0), (-1.0, 0.02), (-10.0, 0.5)):
    out = mean_voltage(_t_i, _t_th)
    _t_v, _t_dv = _t_cf(_t_i, _t_th, want_D=False)
    assert _t_rel(out[0], _t_v) < 1e-9, (_t_i, _t_th, out, _t_v)
    assert _t_rel(out[1], _t_dv) < 1e-8, (_t_i, _t_th, out, _t_dv)
    assert np.sqrt(max(_t_i ** 2 - 1.0, 0.0)) < abs(out[0]) < abs(_t_i)
_t_p, _t_m = mean_voltage(0.8, 0.07), mean_voltage(-0.8, 0.07)
# two computed outputs: twice the single-output tolerances (relative 1e-9 for v, 1e-8 for r_d)
assert _t_rel(-_t_m[0], _t_p[0]) < 2e-9 and _t_rel(_t_m[1], _t_p[1]) < 2e-8, (_t_p, _t_m)

# --- test case 4 ---
# non-finite input, |i| > 10 or theta outside [0.02, 50] raises ValueError
import numpy as np
for _t_bad in ((float("nan"), 0.1), (0.5, float("inf")), (10.5, 0.1), (-11.0, 1.0), (0.5, 0.019), (0.5, 51.0), (0.5, 0.0), (0.5, -1.0)):
    try:
        mean_voltage(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("mean_voltage%r must raise ValueError" % (_t_bad,))
