# Independent target for the junction with capacitance: the same Kramers equation expanded
# in Hermite functions CENTRED AT A DIFFERENT VELOCITY v0 = 0.6 i (another basis, so another
# truncated system), with a larger truncation (N = 150 Hermite, |p| <= 72 Fourier modes),
# solved by its own matrix continued fraction. dv/di from a central difference (h = 1e-4).
# Every test case below is self-contained (its own imports and helper functions).

# --- test case 0 ---
# intermediate inertia and noise, positive and negative bias
import numpy as np

def _t_kramers_v(i, theta, beta_c, N=150, P=72):
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
    h = 1e-4
    f = lambda x: _t_kramers_v(x, theta, beta_c)
    d = (f(i + h) - f(i - h)) / (2 * h)
    return f(i), d

def _t_check_rcsj(i, theta, beta_c):
    out = rcsj_voltage(i, theta, beta_c)
    assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
    v, dv = _t_kramers(i, theta, beta_c)
    assert abs(out[0] - v) <= 1e-7 * abs(v) + 1e-12, (i, theta, beta_c, out, v)
    assert abs(out[1] / dv - 1.0) < 1e-5, (i, theta, beta_c, out, dv)
    return out

_t_check_rcsj(0.9, 0.2, 1.0)
_t_check_rcsj(-0.4, 0.15, 1.5)

# --- test case 1 ---
# strong inertia near the critical current, and strong noise
import numpy as np

def _t_kramers_v(i, theta, beta_c, N=150, P=72):
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
    h = 1e-4
    f = lambda x: _t_kramers_v(x, theta, beta_c)
    d = (f(i + h) - f(i - h)) / (2 * h)
    return f(i), d

def _t_check_rcsj(i, theta, beta_c):
    out = rcsj_voltage(i, theta, beta_c)
    assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
    v, dv = _t_kramers(i, theta, beta_c)
    assert abs(out[0] - v) <= 1e-7 * abs(v) + 1e-12, (i, theta, beta_c, out, v)
    assert abs(out[1] / dv - 1.0) < 1e-5, (i, theta, beta_c, out, dv)
    return out

_t_check_rcsj(1.1, 0.12, 1.8)
_t_check_rcsj(1.6, 1.0, 0.5)
