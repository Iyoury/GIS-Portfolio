# Independent targets: no closed-form Gaussian integrals. Every integral is reduced to a
# one-dimensional radial integral with the shell theorem (averages over spheres) and
# done with composite Gauss-Legendre quadrature.
# Every test case below is self-contained: it repeats its own imports, helpers and setup.

# --- test case 0 ---
# H2, zeta = 1.24, R = 1.4 (Szabo and Ostlund, Section 3.5.2)
import numpy as np
from scipy.special import erf as _t_erf

_T_ALPHA = np.array([0.109818, 0.405771, 2.22766])

_T_D = np.array([0.444635, 0.535328, 0.154329])

_T_X16, _T_W16 = np.polynomial.legendre.leggauss(16)

def _t_grid(s_max, width):
    n = int(np.ceil(s_max / width))
    e = np.linspace(0.0, s_max, n + 1)
    h, m = 0.5 * (e[1:] - e[:-1]), 0.5 * (e[1:] + e[:-1])
    return (m[:, None] + h[:, None] * _T_X16[None, :]).ravel(), (h[:, None] * _T_W16[None, :]).ravel()

def _t_eri(za, zb, R):
    cen = np.array([0.0, float(R)])
    ex = [_T_ALPHA * za ** 2, _T_ALPHA * zb ** 2]
    idx = np.array(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), np.arange(3), indexing="ij")).reshape(4, -1)

    def one(i, j, k, l):
        a = ex[i][idx[0]][:, None]
        b = ex[j][idx[1]][:, None]
        c = ex[k][idx[2]][:, None]
        d = ex[l][idx[3]][:, None]
        w = (_T_D[idx[0]] * _T_D[idx[1]] * _T_D[idx[2]] * _T_D[idx[3]])[:, None]
        nrm = (2 * a / np.pi * 2 * b / np.pi * 2 * c / np.pi * 2 * d / np.pi) ** 0.75
        p, q = a + b, c + d
        P, Q = (a * cen[i] + b * cen[j]) / p, (c * cen[k] + d * cen[l]) / q
        kab = np.exp(-a * b / p * (cen[i] - cen[j]) ** 2)
        kcd = np.exp(-c * d / q * (cen[k] - cen[l]) ** 2)
        e = np.abs(P - Q)
        s, ws = _t_grid(float(np.max(e)) + 10.0 / np.sqrt(np.min(q)), min(1.0, 1.2 / np.sqrt(max(np.max(p), np.max(q)))))
        s, ws = s[None, :], ws[None, :]
        sp = np.sqrt(p)
        tiny = e < 1e-6
        es = np.where(tiny, 1.0, e)

        def G(u):
            return u * _t_erf(sp * u) + np.exp(-p * u * u) / np.sqrt(np.pi * p)

        # mean over a sphere around Q of the potential of the Gaussian cloud centred at P
        avg = np.where(tiny, _t_erf(sp * s) / s, (G(s + es) - G(s - es)) / (2 * s * es))
        rad = np.sum(4 * np.pi * s * s * np.exp(-q * s * s) * avg * ws, axis=1)[:, None]
        return float(np.sum(w * nrm * kab * kcd * (np.pi / p) ** 1.5 * rad))

    out = np.zeros((2, 2, 2, 2))
    memo = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in memo:
                        memo[key] = one(i, j, k, l)
                    out[i, j, k, l] = memo[key]
    return out

def _t_check(eri, args):
    assert isinstance(eri, np.ndarray) and eri.shape == (2, 2, 2, 2)
    target = _t_eri(*args)
    assert np.max(np.abs(np.asarray(eri) - np.asarray(target))) <= 1e-10, (eri, target)
    # permutation symmetry: comparisons between two computed entries, each within 1e-10 of the target
    assert np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) <= 2e-10
    assert np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) <= 2e-10
    return eri

eri = _t_check(sto3g_two_electron(1.24, 1.24, 1.4), (1.24, 1.24, 1.4))
assert abs(eri[0, 0, 0, 0] - 0.7746) < 1e-4 and abs(eri[0, 0, 1, 1] - 0.5697) < 1e-4
assert abs(eri[0, 1, 0, 1] - 0.2970) < 1e-4 and abs(eri[0, 0, 0, 1] - 0.4441) < 1e-4

# --- test case 1 ---
# HeH+, zeta(He) = 2.0925, zeta(H) = 1.24, R = 1.4632
import numpy as np
from scipy.special import erf as _t_erf

_T_ALPHA = np.array([0.109818, 0.405771, 2.22766])

_T_D = np.array([0.444635, 0.535328, 0.154329])

_T_X16, _T_W16 = np.polynomial.legendre.leggauss(16)

def _t_grid(s_max, width):
    n = int(np.ceil(s_max / width))
    e = np.linspace(0.0, s_max, n + 1)
    h, m = 0.5 * (e[1:] - e[:-1]), 0.5 * (e[1:] + e[:-1])
    return (m[:, None] + h[:, None] * _T_X16[None, :]).ravel(), (h[:, None] * _T_W16[None, :]).ravel()

def _t_eri(za, zb, R):
    cen = np.array([0.0, float(R)])
    ex = [_T_ALPHA * za ** 2, _T_ALPHA * zb ** 2]
    idx = np.array(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), np.arange(3), indexing="ij")).reshape(4, -1)

    def one(i, j, k, l):
        a = ex[i][idx[0]][:, None]
        b = ex[j][idx[1]][:, None]
        c = ex[k][idx[2]][:, None]
        d = ex[l][idx[3]][:, None]
        w = (_T_D[idx[0]] * _T_D[idx[1]] * _T_D[idx[2]] * _T_D[idx[3]])[:, None]
        nrm = (2 * a / np.pi * 2 * b / np.pi * 2 * c / np.pi * 2 * d / np.pi) ** 0.75
        p, q = a + b, c + d
        P, Q = (a * cen[i] + b * cen[j]) / p, (c * cen[k] + d * cen[l]) / q
        kab = np.exp(-a * b / p * (cen[i] - cen[j]) ** 2)
        kcd = np.exp(-c * d / q * (cen[k] - cen[l]) ** 2)
        e = np.abs(P - Q)
        s, ws = _t_grid(float(np.max(e)) + 10.0 / np.sqrt(np.min(q)), min(1.0, 1.2 / np.sqrt(max(np.max(p), np.max(q)))))
        s, ws = s[None, :], ws[None, :]
        sp = np.sqrt(p)
        tiny = e < 1e-6
        es = np.where(tiny, 1.0, e)

        def G(u):
            return u * _t_erf(sp * u) + np.exp(-p * u * u) / np.sqrt(np.pi * p)

        # mean over a sphere around Q of the potential of the Gaussian cloud centred at P
        avg = np.where(tiny, _t_erf(sp * s) / s, (G(s + es) - G(s - es)) / (2 * s * es))
        rad = np.sum(4 * np.pi * s * s * np.exp(-q * s * s) * avg * ws, axis=1)[:, None]
        return float(np.sum(w * nrm * kab * kcd * (np.pi / p) ** 1.5 * rad))

    out = np.zeros((2, 2, 2, 2))
    memo = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in memo:
                        memo[key] = one(i, j, k, l)
                    out[i, j, k, l] = memo[key]
    return out

def _t_check(eri, args):
    assert isinstance(eri, np.ndarray) and eri.shape == (2, 2, 2, 2)
    target = _t_eri(*args)
    assert np.max(np.abs(np.asarray(eri) - np.asarray(target))) <= 1e-10, (eri, target)
    # permutation symmetry: comparisons between two computed entries, each within 1e-10 of the target
    assert np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) <= 2e-10
    assert np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) <= 2e-10
    return eri

assert _t_check(sto3g_two_electron(2.0925, 1.24, 1.4632), (2.0925, 1.24, 1.4632)) is not None

# --- test case 2 ---
# unequal exponents 1.0 and 1.5 at R = 2.5
import numpy as np
from scipy.special import erf as _t_erf

_T_ALPHA = np.array([0.109818, 0.405771, 2.22766])

_T_D = np.array([0.444635, 0.535328, 0.154329])

_T_X16, _T_W16 = np.polynomial.legendre.leggauss(16)

def _t_grid(s_max, width):
    n = int(np.ceil(s_max / width))
    e = np.linspace(0.0, s_max, n + 1)
    h, m = 0.5 * (e[1:] - e[:-1]), 0.5 * (e[1:] + e[:-1])
    return (m[:, None] + h[:, None] * _T_X16[None, :]).ravel(), (h[:, None] * _T_W16[None, :]).ravel()

def _t_eri(za, zb, R):
    cen = np.array([0.0, float(R)])
    ex = [_T_ALPHA * za ** 2, _T_ALPHA * zb ** 2]
    idx = np.array(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), np.arange(3), indexing="ij")).reshape(4, -1)

    def one(i, j, k, l):
        a = ex[i][idx[0]][:, None]
        b = ex[j][idx[1]][:, None]
        c = ex[k][idx[2]][:, None]
        d = ex[l][idx[3]][:, None]
        w = (_T_D[idx[0]] * _T_D[idx[1]] * _T_D[idx[2]] * _T_D[idx[3]])[:, None]
        nrm = (2 * a / np.pi * 2 * b / np.pi * 2 * c / np.pi * 2 * d / np.pi) ** 0.75
        p, q = a + b, c + d
        P, Q = (a * cen[i] + b * cen[j]) / p, (c * cen[k] + d * cen[l]) / q
        kab = np.exp(-a * b / p * (cen[i] - cen[j]) ** 2)
        kcd = np.exp(-c * d / q * (cen[k] - cen[l]) ** 2)
        e = np.abs(P - Q)
        s, ws = _t_grid(float(np.max(e)) + 10.0 / np.sqrt(np.min(q)), min(1.0, 1.2 / np.sqrt(max(np.max(p), np.max(q)))))
        s, ws = s[None, :], ws[None, :]
        sp = np.sqrt(p)
        tiny = e < 1e-6
        es = np.where(tiny, 1.0, e)

        def G(u):
            return u * _t_erf(sp * u) + np.exp(-p * u * u) / np.sqrt(np.pi * p)

        # mean over a sphere around Q of the potential of the Gaussian cloud centred at P
        avg = np.where(tiny, _t_erf(sp * s) / s, (G(s + es) - G(s - es)) / (2 * s * es))
        rad = np.sum(4 * np.pi * s * s * np.exp(-q * s * s) * avg * ws, axis=1)[:, None]
        return float(np.sum(w * nrm * kab * kcd * (np.pi / p) ** 1.5 * rad))

    out = np.zeros((2, 2, 2, 2))
    memo = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in memo:
                        memo[key] = one(i, j, k, l)
                    out[i, j, k, l] = memo[key]
    return out

def _t_check(eri, args):
    assert isinstance(eri, np.ndarray) and eri.shape == (2, 2, 2, 2)
    target = _t_eri(*args)
    assert np.max(np.abs(np.asarray(eri) - np.asarray(target))) <= 1e-10, (eri, target)
    # permutation symmetry: comparisons between two computed entries, each within 1e-10 of the target
    assert np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) <= 2e-10
    assert np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) <= 2e-10
    return eri

assert _t_check(sto3g_two_electron(1.0, 1.5, 2.5), (1.0, 1.5, 2.5)) is not None

# --- test case 3 ---
# non-positive distance or exponent raises ValueError
import numpy as np
for _t_bad in ((1.24, 1.24, 0.0), (1.24, 1.24, -2.0), (0.0, 1.24, 1.4), (1.24, -1.0, 1.4)):
    _t_raised = False
    try:
        sto3g_two_electron(*_t_bad)
    except ValueError:
        _t_raised = True
    assert _t_raised, "sto3g_two_electron%r must raise ValueError" % (_t_bad,)

# --- test case 4 ---
# ends of the valid ranges: nearly coincident centres (R = 0.02 bohr) with the most diffuse and the
# tightest exponents, and centres 100 bohr apart, where the two charge clouds do not overlap and
# (00|11) is the point-charge value N**2 / R (shell theorem; N = norm of the contracted function)
import numpy as np
from scipy.special import erf as _t_erf

_T_ALPHA = np.array([0.109818, 0.405771, 2.22766])

_T_D = np.array([0.444635, 0.535328, 0.154329])

_T_X16, _T_W16 = np.polynomial.legendre.leggauss(16)

def _t_grid(s_max, width):
    n = int(np.ceil(s_max / width))
    e = np.linspace(0.0, s_max, n + 1)
    h, m = 0.5 * (e[1:] - e[:-1]), 0.5 * (e[1:] + e[:-1])
    return (m[:, None] + h[:, None] * _T_X16[None, :]).ravel(), (h[:, None] * _T_W16[None, :]).ravel()

def _t_eri(za, zb, R):
    cen = np.array([0.0, float(R)])
    ex = [_T_ALPHA * za ** 2, _T_ALPHA * zb ** 2]
    idx = np.array(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), np.arange(3), indexing="ij")).reshape(4, -1)

    def one(i, j, k, l):
        a = ex[i][idx[0]][:, None]
        b = ex[j][idx[1]][:, None]
        c = ex[k][idx[2]][:, None]
        d = ex[l][idx[3]][:, None]
        w = (_T_D[idx[0]] * _T_D[idx[1]] * _T_D[idx[2]] * _T_D[idx[3]])[:, None]
        nrm = (2 * a / np.pi * 2 * b / np.pi * 2 * c / np.pi * 2 * d / np.pi) ** 0.75
        p, q = a + b, c + d
        P, Q = (a * cen[i] + b * cen[j]) / p, (c * cen[k] + d * cen[l]) / q
        kab = np.exp(-a * b / p * (cen[i] - cen[j]) ** 2)
        kcd = np.exp(-c * d / q * (cen[k] - cen[l]) ** 2)
        e = np.abs(P - Q)
        s, ws = _t_grid(float(np.max(e)) + 10.0 / np.sqrt(np.min(q)), min(1.0, 1.2 / np.sqrt(max(np.max(p), np.max(q)))))
        s, ws = s[None, :], ws[None, :]
        sp = np.sqrt(p)
        tiny = e < 1e-6
        es = np.where(tiny, 1.0, e)

        def G(u):
            return u * _t_erf(sp * u) + np.exp(-p * u * u) / np.sqrt(np.pi * p)

        # mean over a sphere around Q of the potential of the Gaussian cloud centred at P
        avg = np.where(tiny, _t_erf(sp * s) / s, (G(s + es) - G(s - es)) / (2 * s * es))
        rad = np.sum(4 * np.pi * s * s * np.exp(-q * s * s) * avg * ws, axis=1)[:, None]
        return float(np.sum(w * nrm * kab * kcd * (np.pi / p) ** 1.5 * rad))

    out = np.zeros((2, 2, 2, 2))
    memo = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in memo:
                        memo[key] = one(i, j, k, l)
                    out[i, j, k, l] = memo[key]
    return out

def _t_check(eri, args):
    assert isinstance(eri, np.ndarray) and eri.shape == (2, 2, 2, 2)
    target = _t_eri(*args)
    assert np.max(np.abs(np.asarray(eri) - np.asarray(target))) <= 1e-10, (eri, target)
    # permutation symmetry: comparisons between two computed entries, each within 1e-10 of the target
    assert np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) <= 2e-10
    assert np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) <= 2e-10
    return eri

_t_N = float(np.sum(np.outer(_T_D, _T_D) * np.outer((2 * _T_ALPHA / np.pi) ** 0.75, (2 * _T_ALPHA / np.pi) ** 0.75)
                    * (np.pi / np.add.outer(_T_ALPHA, _T_ALPHA)) ** 1.5))
for _t_args in ((0.5, 0.5, 0.02), (3.0, 0.5, 0.02), (3.0, 3.0, 100.0), (0.5, 3.0, 100.0)):
    eri = _t_check(sto3g_two_electron(*_t_args), _t_args)
    if _t_args[2] > 50.0:
        assert abs(eri[0, 0, 1, 1] - _t_N ** 2 / 100.0) < 1e-10, (eri[0, 0, 1, 1], _t_N ** 2 / 100.0)
