# --- test case 0 ---
# Independent targets: no Stratonovich / Reimann double integral. The stationary
# Fokker-Planck density p(x) = sum_n c_n exp(i n x) of dphi/dtau = i - sin(phi) + xi obeys
# c_{n+1} + a_n c_n - c_{n-1} = 0 with a_n = 2 theta n + 2 j i (j = imaginary unit);
# its decaying solution is a continued fraction, and v = i + 2 pi Im c_1. The effective
# diffusion follows from homogenization:
# D = theta int (1 + chi')**2 p dx, with the periodic corrector chi solving
# (i - sin x) chi' + theta chi'' = v - (i - sin x); g_n = (chi')_n obeys
# g_{n+1} + b_n g_n - g_{n-1} = 0 (n >= 2), b_n = 2 j i - 2 theta n, g_1 (b_1 + g_2/g_1) = 1.
# All in extended precision, so exponentially small results keep their relative accuracy.
#
# Independent target for the junction with capacitance: the same Kramers equation expanded
# in Hermite functions CENTRED AT A DIFFERENT VELOCITY v0 = 0.6 i (another basis, so another
# truncated system), with a larger truncation than the reference (N = 120 Hermite functions,
# |p| <= 60 Fourier modes), solved by its own matrix continued fraction. dv/di from the
# fourth-order central difference [8 (f(i+h) - f(i-h)) - (f(i+2h) - f(i-2h))] / (12 h) with
# h = 1e-3 (no analytic derivative).
# Error budget of this target (all 14 points where it is used in step 6 and in the whole-task
# tests, against N = 220, |p| <= 110 with the same stencil): v within 5e-5 of its tolerance
# 1e-7 |v| + 1e-12, dv/di within 1.1e-12 relative except at the smallest tested voltage
# (v near 4.6e-9, where it is within 1.8e-7). Against the analytic dv/di of an expansion
# about v = 0 the stencil differs by below 1e-9 relative, except at that voltage, where the
# stencil truncation (about 2.6e-7) and the rounding in the shifted basis (v is about 1e-7 of v0)
# leave an error of about 3e-7 relative (about 3 per cent of the 1e-5 tolerance).
# Every test case below is self-contained (its own imports and helper functions).
# strong inertia and weak noise (beta_c = 2, theta = 0.1): locked, mixed and
# running regimes, where the noiseless junction would be hysteretic
import numpy as np


def _crown_case():
    import numpy as np

    def _t_kramers_v(i, theta, beta_c, N=120, P=60):
        v0 = 0.6 * i
        vth = np.sqrt(theta / beta_c)
        gam = 1.0 / beta_c
        ps = np.arange(-P, P + 1)
        M = ps.size
        D = np.diag(vth * 1j * ps)
        Fm = np.diag(np.full(M, (i - v0) / beta_c, dtype=complex))
        Fm += np.diag(np.full(M - 1, -1.0 / (2j * beta_c)), -1) + np.diag(np.full(M - 1, 1.0 / (2j * beta_c)), 1)
        Dh = D - Fm / vth
        V0 = np.diag(v0 * 1j * ps)
        S = np.zeros((M, M), complex)
        for n in range(N, 0, -1):
            S = -np.linalg.solve(gam * n * np.eye(M) + V0 + np.sqrt(n + 1) * D @ S, np.sqrt(n) * Dh)
        Q = V0 + D @ S
        rows = [k for k in range(M) if k != P]
        c0 = np.zeros(M, complex)
        c0[P] = 1 / (2 * np.pi)
        c0[rows] = np.linalg.solve(Q[np.ix_(rows, rows)], -Q[np.ix_(rows, [P])][:, 0] / (2 * np.pi))
        c1 = S @ c0
        return v0 + vth * 2 * np.pi * c1[P].real

    def _t_kramers(i, theta, beta_c):
        # v, and dv/di from the fourth-order central difference (five-point stencil, h = 1e-3)
        h = 1e-3
        f = lambda x: _t_kramers_v(x, theta, beta_c)
        d = (8.0 * (f(i + h) - f(i - h)) - (f(i + 2 * h) - f(i - 2 * h))) / (12.0 * h)
        return f(i), d

    def _t_check_rcsj(i, theta, beta_c):
        out = rcsj_voltage(i, theta, beta_c)
        assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
        v, dv = _t_kramers(i, theta, beta_c)
        assert abs(out[0] - v) <= 1e-7 * abs(v) + 1e-12, (i, theta, beta_c, out, v)
        assert abs(out[1] / dv - 1.0) < 1e-5, (i, theta, beta_c, out, dv)
        return out

    _t_o = _t_check_rcsj(0.7, 0.1, 2.0)
    _t_check_rcsj(1.0, 0.1, 2.0)
    _t_r = _t_check_rcsj(2.5, 0.1, 2.0)
    assert np.sqrt(2.5 ** 2 - 1.0) < _t_r[0] < 2.5


_crown_case()

# --- test case 1 ---
# weak inertia (beta_c = 0.1), reversed bias, and the overdamped limit beta_c = 0,
# which is the junction of step 1 (independent continued fraction of the overdamped equation)
import numpy as np


def _crown_case():
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

    def _t_kramers_v(i, theta, beta_c, N=120, P=60):
        v0 = 0.6 * i
        vth = np.sqrt(theta / beta_c)
        gam = 1.0 / beta_c
        ps = np.arange(-P, P + 1)
        M = ps.size
        D = np.diag(vth * 1j * ps)
        Fm = np.diag(np.full(M, (i - v0) / beta_c, dtype=complex))
        Fm += np.diag(np.full(M - 1, -1.0 / (2j * beta_c)), -1) + np.diag(np.full(M - 1, 1.0 / (2j * beta_c)), 1)
        Dh = D - Fm / vth
        V0 = np.diag(v0 * 1j * ps)
        S = np.zeros((M, M), complex)
        for n in range(N, 0, -1):
            S = -np.linalg.solve(gam * n * np.eye(M) + V0 + np.sqrt(n + 1) * D @ S, np.sqrt(n) * Dh)
        Q = V0 + D @ S
        rows = [k for k in range(M) if k != P]
        c0 = np.zeros(M, complex)
        c0[P] = 1 / (2 * np.pi)
        c0[rows] = np.linalg.solve(Q[np.ix_(rows, rows)], -Q[np.ix_(rows, [P])][:, 0] / (2 * np.pi))
        c1 = S @ c0
        return v0 + vth * 2 * np.pi * c1[P].real

    def _t_kramers(i, theta, beta_c):
        # v, and dv/di from the fourth-order central difference (five-point stencil, h = 1e-3)
        h = 1e-3
        f = lambda x: _t_kramers_v(x, theta, beta_c)
        d = (8.0 * (f(i + h) - f(i - h)) - (f(i + 2 * h) - f(i - 2 * h))) / (12.0 * h)
        return f(i), d

    def _t_check_rcsj(i, theta, beta_c):
        out = rcsj_voltage(i, theta, beta_c)
        v, dv = _t_kramers(i, theta, beta_c)
        assert abs(out[0] - v) <= 1e-7 * abs(v) + 1e-12, (i, theta, beta_c, out, v)
        assert abs(out[1] / dv - 1.0) < 1e-5, (i, theta, beta_c, out, dv)
        return out

    _t_check_rcsj(-1.3, 0.5, 0.1)
    _t_o = rcsj_voltage(0.8, 0.3, 0.0)
    _t_v, _t_dv = _t_cf(0.8, 0.3, want_D=False)
    assert abs(_t_o[0] - _t_v) <= 1e-7 * abs(_t_v) + 1e-12 and _t_rel(_t_o[1], _t_dv) < 1e-5, (_t_o, _t_v, _t_dv)
    # inertia changes the current-voltage curve: beta_c = 0.1 differs from the overdamped value
    assert abs(rcsj_voltage(0.8, 0.3, 0.1)[0] / _t_v - 1.0) > 1e-3


_crown_case()

# --- test case 2 ---
# thermally activated slips (exponentially small voltage), zero bias, strong noise
import numpy as np


def _crown_case():
    import numpy as np

    def _t_kramers_v(i, theta, beta_c, N=120, P=60):
        v0 = 0.6 * i
        vth = np.sqrt(theta / beta_c)
        gam = 1.0 / beta_c
        ps = np.arange(-P, P + 1)
        M = ps.size
        D = np.diag(vth * 1j * ps)
        Fm = np.diag(np.full(M, (i - v0) / beta_c, dtype=complex))
        Fm += np.diag(np.full(M - 1, -1.0 / (2j * beta_c)), -1) + np.diag(np.full(M - 1, 1.0 / (2j * beta_c)), 1)
        Dh = D - Fm / vth
        V0 = np.diag(v0 * 1j * ps)
        S = np.zeros((M, M), complex)
        for n in range(N, 0, -1):
            S = -np.linalg.solve(gam * n * np.eye(M) + V0 + np.sqrt(n + 1) * D @ S, np.sqrt(n) * Dh)
        Q = V0 + D @ S
        rows = [k for k in range(M) if k != P]
        c0 = np.zeros(M, complex)
        c0[P] = 1 / (2 * np.pi)
        c0[rows] = np.linalg.solve(Q[np.ix_(rows, rows)], -Q[np.ix_(rows, [P])][:, 0] / (2 * np.pi))
        c1 = S @ c0
        return v0 + vth * 2 * np.pi * c1[P].real

    def _t_kramers(i, theta, beta_c):
        # v, and dv/di from the fourth-order central difference (five-point stencil, h = 1e-3)
        h = 1e-3
        f = lambda x: _t_kramers_v(x, theta, beta_c)
        d = (8.0 * (f(i + h) - f(i - h)) - (f(i + 2 * h) - f(i - 2 * h))) / (12.0 * h)
        return f(i), d

    def _t_check_rcsj(i, theta, beta_c):
        out = rcsj_voltage(i, theta, beta_c)
        v, dv = _t_kramers(i, theta, beta_c)
        assert abs(out[0] - v) <= 1e-7 * abs(v) + 1e-12, (i, theta, beta_c, out, v)
        assert abs(out[1] / dv - 1.0) < 1e-5, (i, theta, beta_c, out, dv)
        return out

    _t_check_rcsj(0.05, 0.1, 2.0)
    _t_z = rcsj_voltage(0.0, 0.5, 1.0)
    assert abs(_t_z[0]) <= 1e-12, _t_z           # v = 0 at i = 0, stated tolerance 1e-7 |v| + 1e-12
    assert abs(_t_z[1] / _t_kramers(0.0, 0.5, 1.0)[1] - 1.0) < 1e-5
    _t_check_rcsj(-2.5, 2.0, 2.0)
    _t_check_rcsj(1.2, 2.0, 0.1)


_crown_case()

# --- test case 3 ---
# inputs outside the stated ranges raise ValueError
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((2.51, 0.3, 1.0), (0.5, 0.099, 1.0), (0.5, 2.01, 1.0), (0.5, 0.3, 0.05),
                   (0.5, 0.3, 2.01)):
        try:
            rcsj_voltage(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("rcsj_voltage%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 4 ---
# near the bistable range (beta_c = 1.8, theta = 0.15, i = 0.95) and the locked-to-running crossover
# at moderate inertia (beta_c = 0.7, theta = 0.25, i = 0.6), against the shifted-basis Kramers
# expansion (against N = 220, |p| <= 110 its v is within 3e-9 of the v tolerance and its dv/di
# within 6e-14 relative at these points); and the symmetry v(-i) = -v(i), r_d(-i) = r_d(i)
# between two computed outputs (twice the single-output tolerances)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_kramers_v(i, theta, beta_c, N=120, P=60):
        v0 = 0.6 * i
        vth = np.sqrt(theta / beta_c)
        gam = 1.0 / beta_c
        ps = np.arange(-P, P + 1)
        M = ps.size
        D = np.diag(vth * 1j * ps)
        Fm = np.diag(np.full(M, (i - v0) / beta_c, dtype=complex))
        Fm += np.diag(np.full(M - 1, -1.0 / (2j * beta_c)), -1) + np.diag(np.full(M - 1, 1.0 / (2j * beta_c)), 1)
        Dh = D - Fm / vth
        V0 = np.diag(v0 * 1j * ps)
        S = np.zeros((M, M), complex)
        for n in range(N, 0, -1):
            S = -np.linalg.solve(gam * n * np.eye(M) + V0 + np.sqrt(n + 1) * D @ S, np.sqrt(n) * Dh)
        Q = V0 + D @ S
        rows = [k for k in range(M) if k != P]
        c0 = np.zeros(M, complex)
        c0[P] = 1 / (2 * np.pi)
        c0[rows] = np.linalg.solve(Q[np.ix_(rows, rows)], -Q[np.ix_(rows, [P])][:, 0] / (2 * np.pi))
        c1 = S @ c0
        return v0 + vth * 2 * np.pi * c1[P].real

    def _t_kramers(i, theta, beta_c):
        # v, and dv/di from the fourth-order central difference (five-point stencil, h = 1e-3)
        h = 1e-3
        f = lambda x: _t_kramers_v(x, theta, beta_c)
        d = (8.0 * (f(i + h) - f(i - h)) - (f(i + 2 * h) - f(i - 2 * h))) / (12.0 * h)
        return f(i), d

    for _t_i, _t_th, _t_bc in ((0.95, 0.15, 1.8), (0.6, 0.25, 0.7)):
        _t_out = rcsj_voltage(_t_i, _t_th, _t_bc)
        _t_v, _t_dv = _t_kramers(_t_i, _t_th, _t_bc)
        assert abs(_t_out[0] - _t_v) <= 1e-7 * abs(_t_v) + 1e-12, (_t_i, _t_th, _t_bc, _t_out, _t_v)
        assert abs(_t_out[1] / _t_dv - 1.0) < 1e-5, (_t_i, _t_th, _t_bc, _t_out, _t_dv)
    _t_p, _t_m = rcsj_voltage(0.95, 0.15, 1.8), rcsj_voltage(-0.95, 0.15, 1.8)
    assert abs(_t_m[0] + _t_p[0]) <= 2e-7 * abs(_t_p[0]) + 2e-12, (_t_p, _t_m)
    assert abs(_t_m[1] / _t_p[1] - 1.0) < 2e-5, (_t_p, _t_m)


_crown_case()
