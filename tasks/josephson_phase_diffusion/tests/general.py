# Independent target for the junction with capacitance: the same Kramers equation expanded
# in Hermite functions CENTRED AT A DIFFERENT VELOCITY v0 = 0.6 i (another basis, so another
# truncated system), with a larger truncation than the reference (N = 120 Hermite functions,
# |p| <= 60 Fourier modes), solved by its own matrix continued fraction. dv/di from the
# fourth-order central difference [8 (f(i+h) - f(i-h)) - (f(i+2h) - f(i-2h))] / (12 h) with
# h = 1e-3 (no analytic derivative).
# Error budget of this target (13 tested points, against N = 220, |p| <= 110 with the same
# stencil): v within 5e-5 of its tolerance 1e-7 |v| + 1e-12, dv/di within 1e-12 relative except
# at the smallest tested voltage (v near 4.6e-9). Against the analytic dv/di of an expansion
# about v = 0 the stencil differs by below 1e-9 relative, except at that voltage, where the
# stencil truncation (about 2.6e-7) and the rounding in the shifted basis (v is about 1e-7 of v0)
# leave an error of about 3e-7 relative (about 3 per cent of the 1e-5 tolerance).
# Every test case below is self-contained (its own imports and helper functions).

# --- test case 0 ---
# intermediate inertia and noise, positive and negative bias
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

_t_check_rcsj(0.9, 0.2, 1.0)
_t_check_rcsj(-0.4, 0.15, 1.5)

# --- test case 1 ---
# strong inertia near the critical current, and strong noise; at beta_c = 0 the junction of
# step 1, so rcsj_voltage must reproduce mean_voltage (two computed outputs: twice the step 6
# tolerances, 2e-7 |v| + 2e-12 for v and a relative 2e-5 for r_d)
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

_t_check_rcsj(1.1, 0.12, 1.8)
_t_check_rcsj(1.6, 1.0, 0.5)
_t_o = rcsj_voltage(-0.6, 0.4, 0.0)
_t_m = mean_voltage(-0.6, 0.4)
assert isinstance(_t_o, tuple) and len(_t_o) == 2 and all(type(x) is float for x in _t_o), _t_o
assert _t_m[0] < -1e-3 and abs(_t_o[0] - _t_m[0]) <= 2e-7 * abs(_t_m[0]) + 2e-12, (_t_o, _t_m)
assert abs(_t_o[1] / _t_m[1] - 1.0) < 2e-5, (_t_o, _t_m)
