import numpy as np
import mpmath as mp
from scipy.optimize import brentq

# Second solution: instead of the Stratonovich / Reimann double integrals, solve the
# stationary Fokker-Planck equation and the homogenization cell problem directly in a
# Fourier (Galerkin) basis, with tridiagonal (Thomas) eliminations carried out in
# extended precision so that exponentially small currents keep their relative accuracy.
# The effective diffusion is the average of theta (1 + chi')**2 p, chi the corrector.


def _g_size(theta):
    # working digits: the cancellation in v = i + Im(...) costs about 0.87 / theta digits
    dps = 30 + int(1.8 / theta)
    n = 40 + int(1.3 * np.sqrt(2.0 * dps * 2.303 / theta))
    return dps, n


def _g_thomas(sub, diag, sup, rhs):
    # tridiagonal solve (lists of mp numbers), sub[0] and sup[-1] unused
    n = len(diag)
    cp, dp = [None] * n, [None] * n
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, n):
        den = diag[k] - sub[k] * cp[k - 1]
        cp[k] = sup[k] / den if k < n - 1 else 0
        dp[k] = (rhs[k] - sub[k] * dp[k - 1]) / den
    x = [None] * n
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def _g_density(a, th, N):
    # Galerkin: (fP)_n = a c_n - (c_{n-1} - c_{n+1}) / (2 j); stationarity for n = 1..N:
    # -c_{n-1} + (2 th n + 2 j a) c_n + c_{n+1} = 0 with c_0 = 1 / (2 pi), c_{N+1} = 0
    J = mp.mpc(0, 1)
    c0 = 1 / (2 * mp.pi)
    sub = [mp.mpf(-1)] * N
    sup = [mp.mpf(1)] * N
    diag = [2 * th * n + 2 * J * a for n in range(1, N + 1)]
    rhs = [mp.mpc(0)] * N
    rhs[0] = c0
    c = _g_thomas(sub, diag, sup, rhs)
    # derivative in a: same matrix, right-hand side -(dA/da) c = -2 j c
    dc = _g_thomas(sub, diag, sup, [-2 * J * ck for ck in c])
    return c0, c, dc


def _g_velocity(i, theta):
    dps, N = _g_size(theta)
    with mp.workdps(dps):
        a, th = mp.mpf(abs(i)), mp.mpf(theta)
        c0, c, dc = _g_density(a, th, N)
        # current J = a c_0 + 2 Im(c_1) ... v = 2 pi J
        v = 2 * mp.pi * (a * c0 + mp.im(c[0]))
        dv = 2 * mp.pi * (c0 + mp.im(dc[0]))
        return float(v if i >= 0 else -v), float(dv)


def _g_dlogv_dlogtheta(i, theta):
    # differentiate the Galerkin system in theta: A dc/dtheta = -(dA/dtheta) c, dA/dtheta = diag(2 n)
    dps, N = _g_size(theta)
    dps += 20
    with mp.workdps(dps):
        a, th = mp.mpf(i), mp.mpf(theta)
        c0, c, _ = _g_density(a, th, N)
        diag = [2 * th * n + 2 * mp.mpc(0, 1) * a for n in range(1, N + 1)]
        dc = _g_thomas([mp.mpf(-1)] * N, diag, [mp.mpf(1)] * N, [-2 * n * c[n - 1] for n in range(1, N + 1)])
        v = 2 * mp.pi * (a * c0 + mp.im(c[0]))
        dv = 2 * mp.pi * mp.im(dc[0])
        return float(th * dv / v)


def _g_diffusion(i, theta):
    dps, N = _g_size(theta)
    with mp.workdps(dps):
        a, th = mp.mpf(abs(i)), mp.mpf(theta)
        J = mp.mpc(0, 1)
        c0, c, _ = _g_density(a, th, N)
        # cell problem (a - sin x) chi' + th chi'' = v - (a - sin x) for g_n = (chi')_n,
        # n = 1..N: -g_{n-1} + (2 j a - 2 th n) g_n + g_{n+1} = delta_{n1}, g_0 = 0
        g = _g_thomas([mp.mpf(-1)] * N, [2 * J * a - 2 * th * n for n in range(1, N + 1)],
                      [mp.mpf(1)] * N, [mp.mpc(1)] + [mp.mpc(0)] * (N - 1))
        # average of (1 + chi')**2 p over one period from the Fourier coefficients:
        # int h**2 p dx = 2 pi sum_m (h*h)_m conj(c_m), h = 1 + chi', p real
        h = np.array([mp.conj(x) for x in g[::-1]] + [mp.mpc(1)] + g, dtype=object)
        cc = np.array([mp.conj(x) for x in c[::-1]] + [c0] + c, dtype=object)
        hh = np.convolve(h, h)[N:3 * N + 1]
        tot = mp.re(mp.fsum(hh * np.array([mp.conj(x) for x in cc], dtype=object)))
        return float(th * 2 * mp.pi * tot)


def _g_diffusion_fast(i, theta):
    # double-precision version of the same Galerkin solve (dense complex systems); used only
    # to locate the region of the diffusion peak, where D_eff is of order theta
    N = 40 + int(1.3 * np.sqrt(2.0 * 32 * 2.303 / theta))
    n = np.arange(1, N + 1)
    A = np.diag(2 * theta * n + 2j * i) + np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    rhs = np.zeros(N, complex)
    rhs[0] = 1 / (2 * np.pi)
    c = np.linalg.solve(A, rhs)
    B = np.diag(2j * i - 2 * theta * n) + np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    e1 = np.zeros(N, complex)
    e1[0] = 1.0
    g = np.linalg.solve(B, e1)
    h = np.concatenate([np.conj(g[::-1]), [1.0], g])
    cc = np.concatenate([np.conj(c[::-1]), [1 / (2 * np.pi)], c])
    hh = np.convolve(h, h)[N:3 * N + 1]
    return float(theta * 2 * np.pi * np.real(np.sum(hh * np.conj(cc))))


def _g_check(i, theta):
    i = float(i)
    theta = float(theta)
    if not (np.isfinite(i) and np.isfinite(theta)):
        raise ValueError("i and theta must be finite")
    if abs(i) > 10.0 or not (0.02 <= theta <= 50.0):
        raise ValueError("need |i| <= 10 and 0.02 <= theta <= 50")
    return i, theta


def mean_voltage(i, theta):
    '''Stationary dc voltage and differential resistance of a noisy overdamped junction.'''
    i, theta = _g_check(i, theta)
    if i == 0.0:
        return 0.0, _g_velocity(0.0, theta)[1]
    result = _g_velocity(i, theta)
    return result


def effective_diffusion(i, theta):
    '''Effective diffusion coefficient of the junction phase.'''
    i, theta = _g_check(i, theta)
    D = _g_diffusion(i, theta)
    return D


def diffusion_peak(theta):
    '''Bias current of maximal phase diffusion (giant diffusion) and the enhancement there.'''
    theta = float(theta)
    if not (np.isfinite(theta) and 0.02 <= theta <= 1.0):
        raise ValueError("theta must be in [0.02, 1]")
    # coarse scan, then secant iterations on dD/di from high-precision central differences
    grid = np.linspace(0.55, 2.95, 241)
    vals = [_g_diffusion_fast(x, theta) for x in grid]
    k = int(np.argmax(vals))
    h = 1e-5

    def slope(x):
        return (_g_diffusion(x + h, theta) - _g_diffusion(x - h, theta)) / (2 * h)

    x0, x1 = grid[k] - 0.005, grid[k] + 0.005
    f0, f1 = slope(x0), slope(x1)
    for _ in range(40):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < 1e-11:
            x1 = x2
            break
        x0, f0, x1, f1 = x1, f1, x2, slope(x2)
    i_peak = float(x1)
    result = (i_peak, float(_g_diffusion(i_peak, theta) / theta))
    return result


def noise_temperature(i, v):
    '''Noise strength theta inferred from a measured dc voltage, and d ln(theta) / d ln(v).'''
    i = float(i)
    v = float(v)
    if not (np.isfinite(i) and np.isfinite(v)):
        raise ValueError("i and v must be finite")
    if not (0.0 < i <= 10.0):
        raise ValueError("i must be in (0, 10]")
    if not (np.sqrt(max(i * i - 1.0, 0.0)) < v < i):
        raise ValueError("v must lie strictly between the noiseless and the ohmic voltage")
    # Illinois regula falsi on ln(v(theta) / v) in the variable 1 / theta
    def f(u):
        return np.log(_g_velocity(i, 1.0 / u)[0] / v)

    lo, hi = 1.0 / 50.0, 1.0 / 0.02          # u = 1 / theta; v decreases with u
    f_lo, f_hi = f(lo), f(hi)
    if f_lo < -1e-11 or f_hi > 1e-11:
        raise ValueError("no theta in [0.02, 50] reproduces this voltage")
    if f_lo <= 0.0:
        u = lo
    elif f_hi >= 0.0:
        u = hi
    else:
        side = 0
        for _ in range(300):
            u = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
            fu = f(u)
            if fu == 0.0 or hi - lo < 1e-14 * u:
                break
            if fu > 0.0:
                lo, f_lo = u, fu
                if side == 1:
                    f_hi *= 0.5
                side = 1
            else:
                hi, f_hi = u, fu
                if side == -1:
                    f_lo *= 0.5
                side = -1
    theta = float(1.0 / u)
    result = (theta, 1.0 / _g_dlogv_dlogtheta(i, theta))
    return result


def array_criterion(ics, rs, theta0, v_crit):
    '''Criterion current, differential resistance and voltage noise of a series junction array.'''
    c = np.asarray(ics, dtype=float)
    r = np.asarray(rs, dtype=float)
    if c.ndim != 1 or r.shape != c.shape or c.size == 0:
        raise ValueError("ics and rs must be 1-D arrays of the same nonzero length")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r)) and np.all(c > 0.0) and np.all(r > 0.0)):
        raise ValueError("critical currents and resistances must be finite and positive")
    if c.max() / c.min() > 5.0:
        raise ValueError("max(ics) / min(ics) must not exceed 5")
    theta0 = float(theta0)
    if not (np.isfinite(theta0) and theta0 > 0.0):
        raise ValueError("theta0 must be finite and positive")
    th = theta0 / c
    if np.any(th < 0.02) or np.any(th > 50.0):
        raise ValueError("every theta0 / c_k must lie in [0.02, 50]")
    v_crit = float(v_crit)
    if not (np.isfinite(v_crit) and 0.0 < v_crit <= 0.5 * float(np.sum(c * r))):
        raise ValueError("v_crit must be in (0, 0.5 sum(c_k r_k)]")

    def V(j):
        tot, dtot = 0.0, 0.0
        for ck, rk, tk in zip(c, r, th):
            v, dv = _g_velocity(j / ck, tk)
            tot += ck * rk * v
            dtot += rk * dv
        return tot, dtot

    # safeguarded Newton on ln V(J) = ln v_crit (V is exponentially small at low bias)
    lo, hi = 0.0, 2.0 * c.max()
    j = c.min()
    for _ in range(200):
        val, der = V(j)
        if val > v_crit:
            hi = j
        else:
            lo = j
        step = (np.log(val) - np.log(v_crit)) * val / der if val > 0.0 else -np.inf
        jn = j - step
        if not (lo < jn < hi):
            jn = 0.5 * (lo + hi)
        if abs(jn - j) < 1e-14 * j:
            j = jn
            break
        j = jn
    j_star = float(j)
    r_diff = float(V(j_star)[1])
    s_v = float(sum(2.0 * ck * rk * _g_diffusion(j_star / ck, tk) for ck, rk, tk in zip(c, r, th)))
    result = (j_star, r_diff, s_v)
    return result
