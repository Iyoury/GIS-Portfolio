import signal as _t_signal
import time as _t_btime

# Time budget of one call, as stated in the prompt: the call is interrupted once it exceeds the budget
# (by one second), so a solution that is too slow fails this check instead of holding up the tests.
_t_depth = [0]


def _t_budget(fn, seconds, name):
    def wrapped(*args, **kwargs):
        if _t_depth[0]:
            return fn(*args, **kwargs)

        def _alarm(signum, frame):
            raise TimeoutError("%s did not finish within its budget of %g s per call" % (name, seconds))
        try:
            old = _t_signal.signal(_t_signal.SIGALRM, _alarm)
            _t_signal.setitimer(_t_signal.ITIMER_REAL, seconds + 1.0)
            armed = True
        except (ValueError, AttributeError, OSError):      # no SIGALRM here: measure only
            armed = False
        _t_depth[0] += 1
        start = _t_btime.perf_counter()
        try:
            out = fn(*args, **kwargs)
        finally:
            _t_depth[0] -= 1
            if armed:
                _t_signal.setitimer(_t_signal.ITIMER_REAL, 0.0)
                _t_signal.signal(_t_signal.SIGALRM, old)
        elapsed = _t_btime.perf_counter() - start
        assert elapsed <= seconds, ("%s took %.1f s (budget %g s per call)" % (name, elapsed, seconds))
        return out
    return wrapped


branch_magnetization = _t_budget(branch_magnetization, 20.0, "branch_magnetization")
escape_barriers = _t_budget(escape_barriers, 20.0, "escape_barriers")
survival_probability = _t_budget(survival_probability, 20.0, "survival_probability")
switching_field_statistics = _t_budget(switching_field_statistics, 60.0, "switching_field_statistics")
ensemble_switching = _t_budget(ensemble_switching, 60.0, "ensemble_switching")
brown_relaxation = _t_budget(brown_relaxation, 60.0, "brown_relaxation")

import time as _t_time
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
    # the prompt requires every call to finish within 60 s on one CPU core
    start = _t_time.perf_counter()
    out = brown_relaxation(sigma, h)
    elapsed = _t_time.perf_counter() - start
    assert elapsed <= 60.0, ("brown_relaxation took %.1f s" % elapsed, sigma, h)
    assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
    lam, tau = (float(x) for x in _t_brown(sigma, h))
    assert _t_rel(out[0], lam) < 1e-5, (sigma, h, out, lam)
    assert _t_rel(out[1], tau) < 1e-5, (sigma, h, out, tau)
    return out



# --- test case 0: zero field, intermediate barrier ---
_t_check6(12.0, 0.0)

# --- test case 1: high barrier in a weak field (tiny relaxation rate) ---
_t_check6(48.0, 0.12)

# --- test case 2: strong field, depleted shallow well ---
_t_check6(50.0, -0.8)
