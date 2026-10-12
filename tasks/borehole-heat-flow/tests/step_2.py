# --- test case 0 ---
# single layer closed form
# Targets: hand-computed layer sums and adaptive quadrature of 1/k and z/k; relative accuracy 1e-9.
import numpy as np


def _crown_case():
    import numpy as np
    _t2_R, _t2_S = layer_integrals([0.0], [2.5], 800.0)
    assert abs(_t2_R - 800.0 / 2.5) < 1e-9 * 320
    assert abs(_t2_S - 800.0**2 / (2 * 2.5)) < 1e-9 * 128000


_crown_case()

# --- test case 1 ---
# three layers hand computed
import numpy as np


def _crown_case():
    import numpy as np
    tops, k = [0.0, 100.0, 400.0], [2.0, 4.0, 3.0]
    z = np.array([0.0, 50.0, 100.0, 250.0, 400.0, 1000.0])
    _t2_R, _t2_S = (np.asarray(v) for v in layer_integrals(tops, k, z))
    _t2_R_exp = [0, 25, 50, 50 + 150 / 4, 50 + 300 / 4, 125 + 600 / 3]
    _t2_S_exp = [0, 2500 / 4, 10000 / 4,
             2500 + (250**2 - 100**2) / 8,
             2500 + (400**2 - 100**2) / 8,
             2500 + 150000 / 8 + (1000**2 - 400**2) / 6]
    np.testing.assert_allclose(_t2_R, _t2_R_exp, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_t2_S, _t2_S_exp, rtol=1e-9, atol=1e-12)


_crown_case()

# --- test case 2 ---
# resistance is harmonic not arithmetic
import numpy as np


def _crown_case():
    import numpy as np
    # equal thicknesses of k = 1 and k = 5: effective k = 2 / (1/1 + 1/5) = 1.667 (harmonic), not 3,
    # so R(1000) = 1000 / 1.667 = 600 m^2 K W^-1, to the stated relative accuracy 1e-9
    _t2_R, _ = layer_integrals([0.0, 500.0], [1.0, 5.0], 1000.0)
    _t2_R_harmonic = 1000.0 / (2 / (1 / 1.0 + 1 / 5.0))
    assert abs(_t2_R - _t2_R_harmonic) < 1e-9 * _t2_R_harmonic


_crown_case()

# --- test case 3 ---
# array depths match closed form
import numpy as np


def _crown_case():
    import numpy as np
    z = np.linspace(0, 900, 12).reshape(3, 4)
    out = layer_integrals([0.0, 300.0], [3.0, 2.0], z)
    one = layer_integrals([0.0, 300.0], [3.0, 2.0], np.array([450.0]))
    sc = layer_integrals([0.0, 300.0], [3.0, 2.0], 450.0)
    # output contract: a tuple of two arrays with the shape of z (a one-element z included), two floats for a scalar z
    assert (isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, np.ndarray) and v.shape == z.shape for v in out)
            and all(isinstance(v, np.ndarray) and v.shape == (1,) for v in one) and all(type(v) is float for v in sc))
    _t2_R, _t2_S = out
    zz = np.minimum(z, 300.0)
    _t2_R_exp = zz / 3.0 + np.maximum(z - 300.0, 0.0) / 2.0
    _t2_S_exp = zz ** 2 / 6.0 + (np.maximum(z, 300.0) ** 2 - 300.0 ** 2) / 4.0
    np.testing.assert_allclose(np.asarray(_t2_R), _t2_R_exp, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(_t2_S), _t2_S_exp, rtol=1e-9, atol=1e-12)
    # z = 450 m: R = 300/3 + 150/2 = 175, S = 300^2/6 + (450^2 - 300^2)/4 = 43125
    np.testing.assert_allclose([float(one[0][0]), float(one[1][0]), sc[0], sc[1]], [175.0, 43125.0, 175.0, 43125.0], rtol=1e-9)


_crown_case()

# --- test case 4 ---
# random columns against quadrature [seed = 11]
import numpy as np


def _crown_case():
    import numpy as np

    def _t2_quad_RS(tops, k, z):
        from scipy.integrate import quad
        tops = np.asarray(tops, float)
        def kz(x):
            return k[np.searchsorted(tops, x, side="right") - 1]
        brk = [t for t in tops[1:] if t < z]
        R = quad(lambda x: 1.0 / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        S = quad(lambda x: x / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        return R, S

    seed = 11
    # T(z) = T0 + q0 R - A S solves dT/dz = (q0 - A z) / k(z): R and S are the integrals of 1/k and z/k
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    tops = np.concatenate([[0.0], np.cumsum(rng.uniform(40.0, 600.0, n - 1))])
    k = rng.uniform(1.2, 6.0, n)
    z = np.sort(rng.uniform(0.0, tops[-1] + 800.0, 5))
    _t2_R, _t2_S = (np.asarray(v) for v in layer_integrals(tops, k, z))
    ref = np.array([_t2_quad_RS(tops, k, zi) for zi in z])
    np.testing.assert_allclose(_t2_R, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_t2_S, ref[:, 1], rtol=1e-9, atol=1e-12)


_crown_case()

# --- test case 5 ---
# random columns against quadrature [seed = 12]
import numpy as np


def _crown_case():
    import numpy as np

    def _t2_quad_RS(tops, k, z):
        from scipy.integrate import quad
        tops = np.asarray(tops, float)
        def kz(x):
            return k[np.searchsorted(tops, x, side="right") - 1]
        brk = [t for t in tops[1:] if t < z]
        R = quad(lambda x: 1.0 / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        S = quad(lambda x: x / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        return R, S

    seed = 12
    # T(z) = T0 + q0 R - A S solves dT/dz = (q0 - A z) / k(z): R and S are the integrals of 1/k and z/k
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    tops = np.concatenate([[0.0], np.cumsum(rng.uniform(40.0, 600.0, n - 1))])
    k = rng.uniform(1.2, 6.0, n)
    z = np.sort(rng.uniform(0.0, tops[-1] + 800.0, 5))
    _t2_R, _t2_S = (np.asarray(v) for v in layer_integrals(tops, k, z))
    ref = np.array([_t2_quad_RS(tops, k, zi) for zi in z])
    np.testing.assert_allclose(_t2_R, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_t2_S, ref[:, 1], rtol=1e-9, atol=1e-12)


_crown_case()

# --- test case 6 ---
# random columns against quadrature [seed = 13]
import numpy as np


def _crown_case():
    import numpy as np

    def _t2_quad_RS(tops, k, z):
        from scipy.integrate import quad
        tops = np.asarray(tops, float)
        def kz(x):
            return k[np.searchsorted(tops, x, side="right") - 1]
        brk = [t for t in tops[1:] if t < z]
        R = quad(lambda x: 1.0 / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        S = quad(lambda x: x / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        return R, S

    seed = 13
    # T(z) = T0 + q0 R - A S solves dT/dz = (q0 - A z) / k(z): R and S are the integrals of 1/k and z/k
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    tops = np.concatenate([[0.0], np.cumsum(rng.uniform(40.0, 600.0, n - 1))])
    k = rng.uniform(1.2, 6.0, n)
    z = np.sort(rng.uniform(0.0, tops[-1] + 800.0, 5))
    _t2_R, _t2_S = (np.asarray(v) for v in layer_integrals(tops, k, z))
    ref = np.array([_t2_quad_RS(tops, k, zi) for zi in z])
    np.testing.assert_allclose(_t2_R, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_t2_S, ref[:, 1], rtol=1e-9, atol=1e-12)


_crown_case()

# --- test case 7 ---
# random columns against quadrature [seed = 14]
import numpy as np


def _crown_case():
    import numpy as np

    def _t2_quad_RS(tops, k, z):
        from scipy.integrate import quad
        tops = np.asarray(tops, float)
        def kz(x):
            return k[np.searchsorted(tops, x, side="right") - 1]
        brk = [t for t in tops[1:] if t < z]
        R = quad(lambda x: 1.0 / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        S = quad(lambda x: x / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        return R, S

    seed = 14
    # T(z) = T0 + q0 R - A S solves dT/dz = (q0 - A z) / k(z): R and S are the integrals of 1/k and z/k
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    tops = np.concatenate([[0.0], np.cumsum(rng.uniform(40.0, 600.0, n - 1))])
    k = rng.uniform(1.2, 6.0, n)
    z = np.sort(rng.uniform(0.0, tops[-1] + 800.0, 5))
    _t2_R, _t2_S = (np.asarray(v) for v in layer_integrals(tops, k, z))
    ref = np.array([_t2_quad_RS(tops, k, zi) for zi in z])
    np.testing.assert_allclose(_t2_R, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_t2_S, ref[:, 1], rtol=1e-9, atol=1e-12)


_crown_case()

# --- test case 8 ---
# random columns against quadrature [seed = 15]
import numpy as np


def _crown_case():
    import numpy as np

    def _t2_quad_RS(tops, k, z):
        from scipy.integrate import quad
        tops = np.asarray(tops, float)
        def kz(x):
            return k[np.searchsorted(tops, x, side="right") - 1]
        brk = [t for t in tops[1:] if t < z]
        R = quad(lambda x: 1.0 / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        S = quad(lambda x: x / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        return R, S

    seed = 15
    # T(z) = T0 + q0 R - A S solves dT/dz = (q0 - A z) / k(z): R and S are the integrals of 1/k and z/k
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    tops = np.concatenate([[0.0], np.cumsum(rng.uniform(40.0, 600.0, n - 1))])
    k = rng.uniform(1.2, 6.0, n)
    z = np.sort(rng.uniform(0.0, tops[-1] + 800.0, 5))
    _t2_R, _t2_S = (np.asarray(v) for v in layer_integrals(tops, k, z))
    ref = np.array([_t2_quad_RS(tops, k, zi) for zi in z])
    np.testing.assert_allclose(_t2_R, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_t2_S, ref[:, 1], rtol=1e-9, atol=1e-12)


_crown_case()

# --- test case 9 ---
# random columns against quadrature [seed = 16]
import numpy as np


def _crown_case():
    import numpy as np

    def _t2_quad_RS(tops, k, z):
        from scipy.integrate import quad
        tops = np.asarray(tops, float)
        def kz(x):
            return k[np.searchsorted(tops, x, side="right") - 1]
        brk = [t for t in tops[1:] if t < z]
        R = quad(lambda x: 1.0 / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        S = quad(lambda x: x / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
        return R, S

    seed = 16
    # T(z) = T0 + q0 R - A S solves dT/dz = (q0 - A z) / k(z): R and S are the integrals of 1/k and z/k
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    tops = np.concatenate([[0.0], np.cumsum(rng.uniform(40.0, 600.0, n - 1))])
    k = rng.uniform(1.2, 6.0, n)
    z = np.sort(rng.uniform(0.0, tops[-1] + 800.0, 5))
    _t2_R, _t2_S = (np.asarray(v) for v in layer_integrals(tops, k, z))
    ref = np.array([_t2_quad_RS(tops, k, zi) for zi in z])
    np.testing.assert_allclose(_t2_R, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_t2_S, ref[:, 1], rtol=1e-9, atol=1e-12)


_crown_case()

# --- test case 10 ---
# a non-positive conductivity raises ValueError (stated error rule)
import numpy as np


def _crown_case():
    import numpy as np
    try:
        layer_integrals([0.0, 100.0], [2.0, 0.0], 50.0)
    except ValueError:
        pass
    else:
        raise AssertionError("layer_integrals must raise ValueError for a zero conductivity")


_crown_case()

# --- test case 11 ---
# shallow depths under a deep layer stack
import numpy as np


def _crown_case():
    import numpy as np
    # in the first layer R = z / k1 and S = z^2 / (2 k1) exactly, whatever the layers below: the layers that
    # z does not reach contribute nothing (no rounding residue of the squares of their tops), and
    # R = S = 0 at the surface
    tops, k = [0.0, 120.0, 995.3, 1358.1, 1946.2], [2.0, 3.0, 2.5, 3.1, 2.7]
    z = np.array([0.0, 0.01, 0.5, 2.0, 37.0])
    _t2_R, _t2_S = (np.asarray(v, float) for v in layer_integrals(tops, k, z))
    assert abs(_t2_R[0]) <= 1e-12 and abs(_t2_S[0]) <= 1e-12
    assert np.all(np.abs(_t2_R[1:] - z[1:] / 2.0) <= 1e-9 * z[1:] / 2.0)
    assert np.all(np.abs(_t2_S[1:] - z[1:] ** 2 / 4.0) <= 1e-9 * z[1:] ** 2 / 4.0)
    _t2_R0, _t2_S0 = layer_integrals(tops, k, 0.0)
    assert abs(_t2_R0) <= 1e-12 and abs(_t2_S0) <= 1e-12


_crown_case()
