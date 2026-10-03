import numpy as np
import mpmath as _t_mp

# Independent targets for step 6: Brown's equation expanded in Legendre polynomials of z = cos(theta)
# (W = sum b_l P_l). The operator is pentadiagonal in l; it is built exactly from the recurrences for
# z P_l and (1 - z^2) P_l', and solved in extended precision (30 + sigma (1 + |h|)**2 / 2 digits):
# lam1 by inverse iteration on the block l >= 1 (the l = 0 row is mass conservation), the equilibrium
# density from the same block, and tau_int = int z Psi dz / int z rho0 dz with L Psi = -rho0,
# rho0 = (z - <z>) W_eq. No quadrature, no grid.


def _t_band_solve(rows, rhs, N):
    # Gaussian elimination with partial pivoting on a pentadiagonal system given as dict rows[i] = {j: a_ij}
    A = [dict(r) for r in rows]
    b = list(rhs)
    for k in range(N):
        cand = [i for i in range(k, min(k + 3, N)) if k in A[i]]
        p = max(cand, key=lambda i: abs(A[i][k]))
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
        piv = A[k][k]
        for i in range(k + 1, min(k + 3, N)):
            if k in A[i] and A[i][k] != 0:
                f = A[i][k] / piv
                for j, v in A[k].items():
                    A[i][j] = A[i].get(j, 0) - f * v
                del A[i][k]
                b[i] -= f * b[k]
    x = [_t_mp.mpf(0)] * N
    for k in range(N - 1, -1, -1):
        s = b[k] - _t_mp.fsum(v * x[j] for j, v in A[k].items() if j > k)
        x[k] = s / A[k][k]
    return x

def _t_brown(sigma, h, N=None, dps=None):
    if dps is None: dps = 30 + int(0.5 * sigma * (1 + abs(h)) ** 2)
    if N is None: N = int(60 + 3 * sigma)
    with _t_mp.workdps(dps):
        s = _t_mp.mpf(sigma); xi = 2 * s * _t_mp.mpf(h)
        def Mz(v):
            out = [_t_mp.mpf(0)] * (len(v) + 1)
            for l, c in enumerate(v):
                if c == 0: continue
                out[l + 1] += c * (l + 1) / _t_mp.mpf(2 * l + 1)
                if l >= 1: out[l - 1] += c * l / _t_mp.mpf(2 * l + 1)
            return out
        def Tz(v):
            out = [_t_mp.mpf(0)] * (len(v) + 1)
            for l, c in enumerate(v):
                if c == 0 or l == 0: continue
                f = c * l * (l + 1) / _t_mp.mpf(2 * l + 1)
                out[l - 1] += f; out[l + 1] -= f
            return out
        cols = {}
        for l in range(0, N + 1):
            e = [_t_mp.mpf(0)] * (l + 1); e[l] = _t_mp.mpf(1)
            g = [-2 * s * x for x in Mz(e)] + [_t_mp.mpf(0)]
            for k in range(len(e)): g[k] -= xi * e[k]
            Tg, Mg = Tz(g), Mz(g)
            col = {}
            col[l] = col.get(l, 0) - l * (l + 1)
            for k, v in enumerate(Tg): col[k] = col.get(k, 0) + v
            for k, v in enumerate(Mg): col[k] = col.get(k, 0) - 2 * v
            cols[l] = {k: v / 2 for k, v in col.items() if v != 0 and k <= N}
        # L' (rows/cols 1..N), stored by rows
        rows = [dict() for _ in range(N)]
        for l in range(1, N + 1):
            for k, v in cols[l].items():
                if k >= 1: rows[k - 1][l - 1] = v
        # equilibrium density: L' b + L[1:,0] b0 = 0, b0 = 1
        r0 = [-cols[0].get(k, 0) for k in range(1, N + 1)]
        beq = [_t_mp.mpf(1)] + _t_band_solve(rows, r0, N)
        # lam1: inverse iteration on L'
        x = [_t_mp.mpf(1)] * N
        lam_old = None
        for it in range(60):
            y = _t_band_solve(rows, x, N)
            nrm = _t_mp.sqrt(_t_mp.fsum(t * t for t in y)); y = [t / nrm for t in y]
            Ay = [_t_mp.fsum(v * y[j] for j, v in rows[i].items()) for i in range(N)]
            lam = _t_mp.fsum(a * b for a, b in zip(y, Ay))
            x = y
            if lam_old is not None and abs(lam - lam_old) < abs(lam) * _t_mp.mpf(10) ** (-(dps - 10)): break
            lam_old = lam
        lam1 = -lam
        # integral relaxation time of z: tau = -int z Psi dz / int z rho0 dz, L Psi = -rho0... (L^{-1} rho0)
        zW = Mz(beq)
        zmean = zW[0] / beq[0]
        rho0 = [zW[k] - zmean * (beq[k] if k < len(beq) else 0) for k in range(N + 1)]
        y = _t_band_solve(rows, rho0[1:], N)           # L' y = rho0  ->  Psi = -y solves L Psi = -rho0
        tau = -y[0] / rho0[1]                        # int z Psi = (2/3)(-y_1), C(0) = (2/3) rho0_1
        return lam1, tau


def _t_rel(x, y):
    return abs(x - y) / abs(y)


def _t_check6(sigma, h):
    out = brown_relaxation(sigma, h)
    assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
    lam, tau = (float(x) for x in _t_brown(sigma, h))
    assert _t_rel(out[0], lam) < 1e-8, (sigma, h, out, lam)
    assert _t_rel(out[1], tau) < 1e-8, (sigma, h, out, tau)
    return out



# --- test case 0: no anisotropy and no field energy (sigma = 0): free rotational diffusion,
# the slowest mode is P_1(z) with rate exactly 1/tau_N, and tau_int = tau_N ---
for _t_h in (0.0, 0.6):
    _t_o = _t_check6(0.0, _t_h)
    assert abs(_t_o[0] - 1.0) < 1e-12 and abs(_t_o[1] - 1.0) < 1e-12, _t_o

# --- test case 1: zero field, moderate to high barriers; at sigma = 60 Brown's high-barrier
# asymptote (2 / sqrt(pi)) sigma**1.5 exp(-sigma) is reached within a few percent ---
_t_check6(1.0, 0.0)
_t_check6(5.0, 0.0)
_t_o = _t_check6(60.0, 0.0)
_t_asym = 2.0 / np.sqrt(np.pi) * 60.0 ** 1.5 * np.exp(-60.0)
assert 0.9 < _t_o[0] / _t_asym < 1.0, (_t_o, _t_asym)

# --- test case 2: exponentially small rates in a weak field (relative accuracy still required) ---
_t_check6(40.0, 0.2)
_t_check6(55.0, 0.07)
_t_check6(25.0, -0.15)

# --- test case 3: strong fields: the shallow well is depleted and tau_int is far below 1/lam1;
# here the slow mode lives in a well whose Boltzmann weight is ~exp(-216) of the deep one ---
_t_o = _t_check6(60.0, 0.9)
assert _t_o[1] * _t_o[0] < 0.1, _t_o
_t_check6(30.0, 0.75)
_t_check6(20.0, -0.5)
_t_check6(5.0, 0.3)

# --- test case 4: h -> -h symmetry (z -> -z) ---
_t_p, _t_m = brown_relaxation(45.0, 0.6), brown_relaxation(45.0, -0.6)
assert _t_rel(_t_p[0], _t_m[0]) < 1e-10 and _t_rel(_t_p[1], _t_m[1]) < 1e-10

# --- test case 5: inputs outside the stated ranges raise ValueError ---
for _t_bad in ((-1.0, 0.1), (61.0, 0.1), (10.0, 0.95), (10.0, -0.91), (float("nan"), 0.1)):
    try:
        brown_relaxation(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("brown_relaxation%r must raise ValueError" % (_t_bad,))
