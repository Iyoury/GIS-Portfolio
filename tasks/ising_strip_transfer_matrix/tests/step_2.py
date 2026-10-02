# --- test case 0 ---
import numpy as np

# Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
# With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
#   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
# ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
# even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
# odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1.
def _t_kaufman(L, T):
    K = 1.0 / T
    Kd = -0.5 * np.log(np.tanh(K))
    q = np.arange(2 * L)
    ch = np.cosh(2 * K) * np.cosh(2 * Kd) - np.sinh(2 * K) * np.sinh(2 * Kd) * np.cos(np.pi * q / L)
    g = np.arccosh(np.maximum(ch, 1.0))
    g[0] = 2.0 * (K - Kd)
    base = 0.5 * L * np.log(2.0 * np.sinh(2.0 * K))
    l0 = base + 0.5 * np.sum(g[1::2])
    l1 = base + 0.5 * np.sum(g[0::2])
    return -T * l0 / L, 1.0 / (l0 - l1), 1.0 / (2.0 * g[1])

out = strip_lengths(4, 3.0)
assert isinstance(out, tuple) and len(out) == 3
assert all(isinstance(v, float) for v in out)
f, xi_spin, xi_energy = out
_t_f, _t_xs, _t_xe = _t_kaufman(4, 3.0)
assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)

# --- test case 1 ---
import numpy as np

# Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
# With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
#   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
# ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
# even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
# odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1.
def _t_kaufman(L, T):
    K = 1.0 / T
    Kd = -0.5 * np.log(np.tanh(K))
    q = np.arange(2 * L)
    ch = np.cosh(2 * K) * np.cosh(2 * Kd) - np.sinh(2 * K) * np.sinh(2 * Kd) * np.cos(np.pi * q / L)
    g = np.arccosh(np.maximum(ch, 1.0))
    g[0] = 2.0 * (K - Kd)
    base = 0.5 * L * np.log(2.0 * np.sinh(2.0 * K))
    l0 = base + 0.5 * np.sum(g[1::2])
    l1 = base + 0.5 * np.sum(g[0::2])
    return -T * l0 / L, 1.0 / (l0 - l1), 1.0 / (2.0 * g[1])

out = strip_lengths(6, 5.0)
assert isinstance(out, tuple) and len(out) == 3
assert all(isinstance(v, float) for v in out)
f, xi_spin, xi_energy = out
_t_f, _t_xs, _t_xe = _t_kaufman(6, 5.0)
assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)

# --- test case 2 ---
import numpy as np

# Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
# With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
#   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
# ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
# even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
# odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1.
def _t_kaufman(L, T):
    K = 1.0 / T
    Kd = -0.5 * np.log(np.tanh(K))
    q = np.arange(2 * L)
    ch = np.cosh(2 * K) * np.cosh(2 * Kd) - np.sinh(2 * K) * np.sinh(2 * Kd) * np.cos(np.pi * q / L)
    g = np.arccosh(np.maximum(ch, 1.0))
    g[0] = 2.0 * (K - Kd)
    base = 0.5 * L * np.log(2.0 * np.sinh(2.0 * K))
    l0 = base + 0.5 * np.sum(g[1::2])
    l1 = base + 0.5 * np.sum(g[0::2])
    return -T * l0 / L, 1.0 / (l0 - l1), 1.0 / (2.0 * g[1])

out = strip_lengths(10, 1.0)
assert isinstance(out, tuple) and len(out) == 3
assert all(isinstance(v, float) for v in out)
f, xi_spin, xi_energy = out
_t_f, _t_xs, _t_xe = _t_kaufman(10, 1.0)
assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)

# --- test case 3 ---
import numpy as np

# Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
# With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
#   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
# ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
# even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
# odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1.
def _t_kaufman(L, T):
    K = 1.0 / T
    Kd = -0.5 * np.log(np.tanh(K))
    q = np.arange(2 * L)
    ch = np.cosh(2 * K) * np.cosh(2 * Kd) - np.sinh(2 * K) * np.sinh(2 * Kd) * np.cos(np.pi * q / L)
    g = np.arccosh(np.maximum(ch, 1.0))
    g[0] = 2.0 * (K - Kd)
    base = 0.5 * L * np.log(2.0 * np.sinh(2.0 * K))
    l0 = base + 0.5 * np.sum(g[1::2])
    l1 = base + 0.5 * np.sum(g[0::2])
    return -T * l0 / L, 1.0 / (l0 - l1), 1.0 / (2.0 * g[1])

out = strip_lengths(8, 2.0 / np.log(1.0 + np.sqrt(2.0)))
assert isinstance(out, tuple) and len(out) == 3
assert all(isinstance(v, float) for v in out)
f, xi_spin, xi_energy = out
_t_f, _t_xs, _t_xe = _t_kaufman(8, 2.0 / np.log(1.0 + np.sqrt(2.0)))
assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)

# --- test case 4 ---
import numpy as np

# Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
# With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
#   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
# ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
# even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
# odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1.
def _t_kaufman(L, T):
    K = 1.0 / T
    Kd = -0.5 * np.log(np.tanh(K))
    q = np.arange(2 * L)
    ch = np.cosh(2 * K) * np.cosh(2 * Kd) - np.sinh(2 * K) * np.sinh(2 * Kd) * np.cos(np.pi * q / L)
    g = np.arccosh(np.maximum(ch, 1.0))
    g[0] = 2.0 * (K - Kd)
    base = 0.5 * L * np.log(2.0 * np.sinh(2.0 * K))
    l0 = base + 0.5 * np.sum(g[1::2])
    l1 = base + 0.5 * np.sum(g[0::2])
    return -T * l0 / L, 1.0 / (l0 - l1), 1.0 / (2.0 * g[1])

out = strip_lengths(3, 10.0)
assert isinstance(out, tuple) and len(out) == 3
assert all(isinstance(v, float) for v in out)
f, xi_spin, xi_energy = out
_t_f, _t_xs, _t_xe = _t_kaufman(3, 10.0)
assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)
