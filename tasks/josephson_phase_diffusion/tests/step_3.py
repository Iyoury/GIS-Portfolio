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
# low noise: sharp peak just below the critical current, large enhancement
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

def _t_check_peak(out, theta):
    # the returned bias must be a maximum of the independent D(i): value, one Newton step on
    # dD/di = 0 (central differences of the continued-fraction D), and lower D on both sides
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
    ip, dp = out
    assert 0.5 < ip < 3.0, out
    h = 2e-5
    Dm, D0, Dp = (_t_cf(ip + s * h, theta)[2] for s in (-1, 0, 1))
    d1 = (Dp - Dm) / (2 * h)
    d2 = (Dp - 2 * D0 + Dm) / h ** 2
    assert d2 < 0.0, (theta, out, d2)
    assert abs(d1 / d2) < 1e-6, (theta, out, d1 / d2)
    assert _t_rel(dp, D0 / theta) < 1e-9, (theta, out, D0 / theta)
    for s in (-0.05, 0.05):
        assert _t_cf(ip + s, theta)[2] < D0
    return out

_t_o = _t_check_peak(diffusion_peak(0.02), 0.02)
assert _t_o[1] > 10.0 and _t_o[0] < 1.0

# --- test case 1 ---
# intermediate noise
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

def _t_check_peak(out, theta):
    # the returned bias must be a maximum of the independent D(i): value, one Newton step on
    # dD/di = 0 (central differences of the continued-fraction D), and lower D on both sides
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
    ip, dp = out
    assert 0.5 < ip < 3.0, out
    h = 2e-5
    Dm, D0, Dp = (_t_cf(ip + s * h, theta)[2] for s in (-1, 0, 1))
    d1 = (Dp - Dm) / (2 * h)
    d2 = (Dp - 2 * D0 + Dm) / h ** 2
    assert d2 < 0.0, (theta, out, d2)
    assert abs(d1 / d2) < 1e-6, (theta, out, d1 / d2)
    assert _t_rel(dp, D0 / theta) < 1e-9, (theta, out, D0 / theta)
    for s in (-0.05, 0.05):
        assert _t_cf(ip + s, theta)[2] < D0
    return out

_t_check_peak(diffusion_peak(0.1), 0.1)
_t_check_peak(diffusion_peak(0.3), 0.3)

# --- test case 2 ---
# strong noise: broad peak moved above the critical current, enhancement near 1
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

def _t_check_peak(out, theta):
    # the returned bias must be a maximum of the independent D(i): value, one Newton step on
    # dD/di = 0 (central differences of the continued-fraction D), and lower D on both sides
    assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, float) for x in out), out
    ip, dp = out
    assert 0.5 < ip < 3.0, out
    h = 2e-5
    Dm, D0, Dp = (_t_cf(ip + s * h, theta)[2] for s in (-1, 0, 1))
    d1 = (Dp - Dm) / (2 * h)
    d2 = (Dp - 2 * D0 + Dm) / h ** 2
    assert d2 < 0.0, (theta, out, d2)
    assert abs(d1 / d2) < 1e-6, (theta, out, d1 / d2)
    assert _t_rel(dp, D0 / theta) < 1e-9, (theta, out, D0 / theta)
    for s in (-0.05, 0.05):
        assert _t_cf(ip + s, theta)[2] < D0
    return out

_t_o = _t_check_peak(diffusion_peak(1.0), 1.0)
assert _t_o[0] > 1.3 and 1.0 < _t_o[1] < 1.5

# --- test case 3 ---
# theta outside [0.02, 1] or not finite raises ValueError
import numpy as np
for _t_bad in (0.01, 1.5, float("nan"), -0.2):
    try:
        diffusion_peak(_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("diffusion_peak(%r) must raise ValueError" % (_t_bad,))
