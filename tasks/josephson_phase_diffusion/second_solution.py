import numpy as np
import mpmath as mp
from scipy.optimize import brentq

# Second solution. It uses neither the Stratonovich / Reimann quadrature of the reference nor the
# Fourier continued fractions of the test targets. The Boltzmann factors of the tilted washboard are
# expanded with the Jacobi-Anger identity exp(z cos x) = sum_m I_m(z) exp(i m x), z = 1 / theta, so
# that every integral over the shift y is done in closed form,
#   int_0^{2 pi} exp(-(b + i m) y) dy = (1 - exp(-2 pi b)) / (b + i m),   b = |i| / theta.
# This gives (Ambegaokar-Halperin voltage as a Bessel series)
#   v = a / Phi,  Phi = I_0(z)**2 + 2 sum_{n>=1} (-1)**n I_n(z)**2 b**2 / (b**2 + n**2),  a = |i|,
# and the Reimann diffusion coefficient D = theta <I_+**2 I_-> / <I_+>**3 as an average of Fourier
# series: with q = 1 - exp(-2 pi b), I_+(x) = q / (theta b) exp(-z cos x) Gp(x) and
# I_-(x) = q / (theta b) exp(z cos x) Gm(x), where
#   Gp(x) = sum_m I_m b / (b + i m) e^{imx},  Gm(x) = sum_m (-1)**m I_m b / (b - i m) e^{imx},
# so that, with W(x) = exp(-z cos x) = sum_m (-1)**m I_m e^{imx} and <W Gp> = Phi,
#   D = theta <W Gp**2 Gm> / Phi**3   (Lifson-Jackson theta / I_0**2 at b = 0).
# The series alternate and cancel by up to about 0.87 / theta digits, so they are summed in mpmath
# with 30 + 1.8 / theta digits; the Fourier products are formed with exact integer (fixed-point)
# convolutions. Derivatives in i and theta are taken analytically from the series
# (dI_n/dz = (I_{n-1} + I_{n+1}) / 2). Step 6 (Kramers equation) is described at _b_kramers.


def _b_table(theta):
    # working digits and I_m(1/theta) for m = 0..M+1, with I_M below 10**-(dps+5) I_0
    dps = 30 + int(1.8 / theta)
    with mp.workdps(dps + 10):
        z = 1 / mp.mpf(theta)
        I = [mp.besseli(0, z)]
        tol = I[0] * mp.mpf(10) ** (-dps - 5)
        m = 1
        while True:
            I.append(mp.besseli(m, z))
            if m > 2 and I[-1] < tol:
                break
            m += 1
        I.append(mp.besseli(m + 1, z))
    return dps, I


def _b_phi(a, theta, tab):
    # Phi, dPhi/da and z dPhi/dz (z = 1/theta, total derivative at fixed a); v = a / Phi
    dps, I = tab
    M = len(I) - 2
    with mp.workdps(dps):
        z = 1 / mp.mpf(theta)
        b = mp.mpf(a) * z
        s = sb = sz = mp.mpf(0)
        for n in range(1, M + 1):
            sg = -1 if n % 2 else 1
            den = b * b + n * n
            In2 = I[n] ** 2
            s += sg * In2 / den
            sb += sg * In2 * n * n / den ** 2
            sz += sg * I[n] * (I[n - 1] + I[n + 1]) / den
        Phi = I[0] ** 2 + 2 * b * b * s
        dPhi_db = 4 * b * sb
        dPhi_dz = 2 * I[0] * I[1] + 2 * b * b * sz + mp.mpf(a) * dPhi_db
        return Phi, z * dPhi_db, z * dPhi_dz


def _b_velocity(i, theta, tab):
    a = abs(float(i))
    Phi, dPa, _ = _b_phi(a, theta, tab)
    with mp.workdps(tab[0]):
        v = a / Phi
        dv = (Phi - a * dPa) / Phi ** 2
    return float(v if i >= 0 else -v), float(dv)


def _b_fixed(seq, bits):
    # complex mp sequence -> (real ints, imaginary ints), fixed point with 'bits' fraction bits
    sc = mp.mpf(2) ** bits
    re = np.array([int(mp.nint(mp.re(x) * sc)) for x in seq], dtype=object)
    im = np.array([int(mp.nint(mp.im(x) * sc)) for x in seq], dtype=object)
    return re, im


def _b_conv(A, B):
    # exact complex convolution of integer sequences
    ar, ai = A
    br, bi = B
    return (np.convolve(ar, br) - np.convolve(ai, bi), np.convolve(ar, bi) + np.convolve(ai, br))


def _b_mean_product(P, R):
    # mean over x of the product of two real periodic functions from their Fourier coefficients
    # (indexed -K..K): <p r> = Re sum_k P_k R_{-k}
    return int(np.sum(P[0] * R[0][::-1]) - np.sum(P[1] * R[1][::-1]))


def _b_diffusion(i, theta, tab, deriv=False):
    # D = theta <W Gp**2 Gm> / Phi**3, and optionally d ln D / da
    dps, I = tab
    M = len(I) - 2
    a = abs(float(i))
    bits = int(3.33 * dps) + 30
    with mp.workdps(dps + 10):
        z = 1 / mp.mpf(theta)
        b = mp.mpf(a) * z
        J = mp.mpc(0, 1)
        Ih = [x / I[0] for x in I[:M + 1]]           # I_m / I_0, so all coefficients are <= 1
        ms = range(-M, M + 1)
        sg = [-1 if m % 2 else 1 for m in ms]
        w = [s * Ih[abs(m)] for s, m in zip(sg, ms)]
        gp = [Ih[abs(m)] * (b / (b + J * m) if m else 1) for m in ms]
        gm = [s * Ih[abs(m)] * (b / (b - J * m) if m else 1) for s, m in zip(sg, ms)]
        W, Gp, Gm = (_b_fixed(x, bits) for x in (w, gp, gm))
        P = _b_conv((W[0], W[0] * 0), Gp)              # coefficients of W Gp
        R = _b_conv(Gp, Gm)                             # coefficients of Gp Gm
        T = mp.mpf(_b_mean_product(P, R)) / mp.mpf(2) ** (4 * bits) * I[0] ** 4   # <W Gp**2 Gm>
        Phi, dPa, _ = _b_phi(a, theta, tab)
        D = mp.mpf(theta) * T / Phi ** 3
        if not deriv:
            return float(D)
        # d/db of the coefficients: b / (b + i m) -> i m / (b + i m)**2
        dgp = [Ih[abs(m)] * (J * m / (b + J * m) ** 2 if m else 0) for m in ms]
        dgm = [s * Ih[abs(m)] * (-J * m / (b - J * m) ** 2 if m else 0) for s, m in zip(sg, ms)]
        dGp, dGm = _b_fixed(dgp, bits), _b_fixed(dgm, bits)
        R1, R2 = _b_conv(dGp, Gm), _b_conv(Gp, dGm)
        dT = mp.mpf(_b_mean_product(P, (2 * R1[0] + R2[0], 2 * R1[1] + R2[1]))) / mp.mpf(2) ** (4 * bits) * I[0] ** 4
        return float(D), float(z * dT / T - 3 * dPa / Phi)


def _b_check(i, theta):
    i = float(i)
    theta = float(theta)
    if not (np.isfinite(i) and np.isfinite(theta)):
        raise ValueError("i and theta must be finite")
    if abs(i) > 10.0 or not (0.02 <= theta <= 50.0):
        raise ValueError("need |i| <= 10 and 0.02 <= theta <= 50")
    return i, theta


def mean_voltage(i, theta):
    '''Stationary dc voltage and differential resistance of a noisy overdamped junction.'''
    i, theta = _b_check(i, theta)
    result = _b_velocity(i, theta, _b_table(theta))
    return result


def effective_diffusion(i, theta):
    '''Effective diffusion coefficient of the junction phase.'''
    i, theta = _b_check(i, theta)
    D = _b_diffusion(i, theta, _b_table(theta))
    return D


def diffusion_peak(theta):
    '''Bias current of maximal phase diffusion (giant diffusion) and the enhancement there.'''
    theta = float(theta)
    if not (np.isfinite(theta) and 0.02 <= theta <= 1.0):
        raise ValueError("theta must be in [0.02, 1]")
    tab = _b_table(theta)

    def slope(a):
        return _b_diffusion(a, theta, tab, deriv=True)[1]

    # scan d ln D / di on a coarse grid for the sign change + -> -, then Illinois regula falsi
    grid = np.linspace(0.5, 3.0, 11)
    vals = [slope(a) for a in grid]
    k = next(j for j in range(10) if vals[j] > 0.0 >= vals[j + 1])
    lo, hi, flo, fhi = grid[k], grid[k + 1], vals[k], vals[k + 1]
    side = 0
    x = lo
    for _ in range(200):
        x = (lo * fhi - hi * flo) / (fhi - flo)
        fx = slope(x)
        if fx == 0.0 or hi - lo < 1e-13:
            break
        if fx > 0.0:
            lo, flo = x, fx
            if side == 1:
                fhi *= 0.5
            side = 1
        else:
            hi, fhi = x, fx
            if side == -1:
                flo *= 0.5
            side = -1
    i_peak = float(x)
    result = (i_peak, float(_b_diffusion(i_peak, theta, tab) / theta))
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
    # stated endpoint rule, relative to the voltages of mean_voltage at theta = 0.02 and 50
    v_lo, v_hi = mean_voltage(i, 0.02)[0], mean_voltage(i, 50.0)[0]
    if (v_lo - v) / v_lo > 1e-11 or (v - v_hi) / v_hi > 1e-11:
        raise ValueError("no theta in [0.02, 50] reproduces this voltage")

    def kappa_at(theta):
        Phi, _, zdPz = _b_phi(i, theta, _b_table(theta))
        return float(Phi / zdPz)               # d ln theta / d ln v = Phi / (z dPhi/dz)

    if v <= v_lo:
        theta = 0.02
    elif v >= v_hi:
        theta = 50.0
    else:
        # Illinois regula falsi on ln(v(theta) / v) in u = 1 / theta (v decreases with u)
        def f(u):
            return np.log(_b_velocity(i, 1.0 / u, _b_table(1.0 / u))[0] / v)

        lo, hi = 1.0 / 50.0, 1.0 / 0.02
        flo, fhi = np.log(v_hi / v), np.log(v_lo / v)
        side = 0
        u = lo
        for _ in range(300):
            u = (lo * fhi - hi * flo) / (fhi - flo)
            fu = f(u)
            if fu == 0.0 or hi - lo < 1e-14 * u:
                break
            if fu > 0.0:
                lo, flo = u, fu
                if side == 1:
                    fhi *= 0.5
                side = 1
            else:
                hi, fhi = u, fu
                if side == -1:
                    flo *= 0.5
                side = -1
        theta = float(1.0 / u)
    result = (theta, kappa_at(theta))
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
    tabs = [_b_table(t) for t in th]

    def V(j):
        tot, dtot = 0.0, 0.0
        for ck, rk, tk, tb in zip(c, r, th, tabs):
            v, dv = _b_velocity(j / ck, tk, tb)
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
    s_v = float(sum(2.0 * ck * rk * _b_diffusion(j_star / ck, tk, tb)
                    for ck, rk, tk, tb in zip(c, r, th, tabs)))
    result = (j_star, r_diff, s_v)
    return result


def _b_kramers(i, theta, beta_c, v0, N=130, P=64):
    # Kramers equation in Hermite functions of the velocity centred at v0 and a REAL Fourier basis
    # (1, cos p x, sin p x) of the phase, so that every matrix is real for real i. The matrix
    # continued fraction is then evaluated at the complex bias i + 1j * eps (complex-step
    # differentiation): Re gives v, Im / eps gives dv/di to rounding accuracy, with no
    # finite-difference error. The centre v0 is held fixed (the converged v does not depend on it).
    vth = np.sqrt(theta / beta_c)
    gam = 1.0 / beta_c
    K = 2 * P + 1
    Dx = np.zeros((K, K))
    Sx = np.zeros((K, K))
    Sx[P + 1, 0] = 1.0                                   # sin x * 1
    for p in range(1, P + 1):
        Dx[p, P + p] = p                                 # (sin p x)' = p cos p x
        Dx[P + p, p] = -p                                # (cos p x)' = -p sin p x
        if p + 1 <= P:                                   # sin x cos p x = [sin(p+1)x - sin(p-1)x] / 2
            Sx[P + p + 1, p] += 0.5
            Sx[p + 1, P + p] -= 0.5                      # sin x sin p x = [cos(p-1)x - cos(p+1)x] / 2
        if p - 1 >= 1:
            Sx[P + p - 1, p] -= 0.5
        Sx[p - 1, P + p] += 0.5
    eps = 1e-20
    ic = complex(i, eps)
    Dm = vth * Dx
    F = ((ic - v0) / beta_c) * np.eye(K) - Sx / beta_c
    Dh = Dm - F / vth
    V0 = v0 * Dx
    S = np.zeros((K, K), complex)
    for n in range(N, 0, -1):
        S = -np.linalg.solve(gam * n * np.eye(K) + V0 + np.sqrt(n + 1) * Dm @ S, np.sqrt(n) * Dh)
    Q = V0 + Dm @ S                                      # its constant-mode row is zero
    c0 = np.zeros(K, complex)
    c0[0] = 1.0
    c0[1:] = np.linalg.solve(Q[1:, 1:], -Q[1:, 0])
    w = v0 + vth * (S @ c0)[0]
    return float(w.real), float(w.imag / eps)


def rcsj_voltage(i, theta, beta_c):
    '''Stationary dc voltage and differential resistance of a noisy junction with capacitance (RCSJ).'''
    i = float(i)
    theta = float(theta)
    beta_c = float(beta_c)
    if not (np.isfinite(i) and np.isfinite(theta) and np.isfinite(beta_c)):
        raise ValueError("inputs must be finite")
    if abs(i) > 2.5 or not (0.1 <= theta <= 2.0) or not (beta_c == 0.0 or 0.1 <= beta_c <= 2.0):
        raise ValueError("need |i| <= 2.5, 0.1 <= theta <= 2 and beta_c = 0 or 0.1 <= beta_c <= 2")
    if beta_c == 0.0:
        return mean_voltage(i, theta)
    a = abs(i)
    v0 = mean_voltage(a, theta)[0]                       # overdamped mean velocity as the centre
    v, r_d = _b_kramers(a, theta, beta_c, v0)
    if a == 0.0:
        v = 0.0                                          # v is odd in i
    result = (v if i >= 0.0 else -v, r_d)
    return result
