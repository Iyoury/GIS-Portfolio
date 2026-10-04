import numpy as np
from scipy.special import erf as _t_erf

# Independent targets: no closed-form Gaussian integrals. Every integral is reduced to a
# one-dimensional radial integral with the shell theorem (averages over spheres) and
# done with composite Gauss-Legendre quadrature.
_T_ALPHA = np.array([0.109818, 0.405771, 2.22766])
_T_D = np.array([0.444635, 0.535328, 0.154329])
_T_X16, _T_W16 = np.polynomial.legendre.leggauss(16)
_T_TN, _T_TW = np.polynomial.legendre.leggauss(80)
_T_TN, _T_TW = 0.5 * (_T_TN + 1.0), 0.5 * _T_TW


def _t_grid(s_max, width):
    n = int(np.ceil(s_max / width))
    e = np.linspace(0.0, s_max, n + 1)
    h, m = 0.5 * (e[1:] - e[:-1]), 0.5 * (e[1:] + e[:-1])
    return (m[:, None] + h[:, None] * _T_X16[None, :]).ravel(), (h[:, None] * _T_W16[None, :]).ravel()


def _t_avg(beta, s, dist):
    # mean of exp(-beta |r - X|^2) over a sphere of radius s whose centre is dist from X
    if dist < 1e-12:
        return np.exp(-beta * s * s)
    return -np.exp(-beta * (s - dist) ** 2) * np.expm1(-4.0 * beta * s * dist) / (4.0 * beta * s * dist)


def _t_one(ZA, ZB, za, zb, R):
    cen = np.array([0.0, float(R)])
    chg = np.array([float(ZA), float(ZB)])
    ex = [_T_ALPHA * za ** 2, _T_ALPHA * zb ** 2]
    S = np.zeros((2, 2))
    H = np.zeros((2, 2))
    for m in range(2):
        for n in range(2):
            a = np.repeat(ex[m], 3)[:, None]
            b = np.tile(ex[n], 3)[:, None]
            w = np.repeat(_T_D, 3) * np.tile(_T_D, 3) * (2 * a[:, 0] / np.pi) ** 0.75 * (2 * b[:, 0] / np.pi) ** 0.75
            dist = abs(cen[n] - cen[m])
            s, ws = _t_grid(dist + 10.0 / np.sqrt(np.min(a + b)), min(0.5, 0.5 / np.sqrt(np.max(a + b))))
            s = s[None, :]
            ov = np.sum(4 * np.pi * s * s * np.exp(-a * s * s) * _t_avg(b, s, dist) * ws, axis=1)
            ke = np.sum(4 * np.pi * s * s * (3 * b - 2 * b * b * s * s) * np.exp(-b * s * s)
                        * _t_avg(a, s, dist) * ws, axis=1)
            a1, b1 = a[:, 0], b[:, 0]
            p = a1 + b1
            kab = np.exp(-a1 * b1 / p * dist ** 2)
            P = (a1 * cen[m] + b1 * cen[n]) / p
            pot = np.zeros(9)
            for C in range(2):
                e = np.abs(P - cen[C])
                # (1/e) int_0^e s^2 exp(-p s^2) ds, the integrand being negligible beyond s = 9 / sqrt(p)
                top = np.minimum(e, 9.0 / np.sqrt(p))
                inner = np.sum(_T_TW * _T_TN ** 2 * np.exp(-p[:, None] * (top[:, None] * _T_TN) ** 2),
                               axis=1) * top ** 3 / np.where(e > 0, e, 1.0)
                pot -= chg[C] * 4 * np.pi * (inner + np.exp(-p * e ** 2) / (2 * p))
            S[m, n] = np.sum(w * ov)
            H[m, n] = np.sum(w * (ke + kab * pot))
    return S, H


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
    assert np.allclose(eri, target, rtol=0.0, atol=1e-10), (eri, target)
    assert np.allclose(eri, eri.transpose(1, 0, 2, 3), rtol=0.0, atol=1e-10)
    assert np.allclose(eri, eri.transpose(2, 3, 0, 1), rtol=0.0, atol=1e-10)
    return eri


# --- test case 0: H2, zeta = 1.24, R = 1.4 (Szabo and Ostlund, Section 3.5.2) ---
eri = _t_check(sto3g_two_electron(1.24, 1.24, 1.4), (1.24, 1.24, 1.4))
assert abs(eri[0, 0, 0, 0] - 0.7746) < 1e-4 and abs(eri[0, 0, 1, 1] - 0.5697) < 1e-4
assert abs(eri[0, 1, 0, 1] - 0.2970) < 1e-4 and abs(eri[0, 0, 0, 1] - 0.4441) < 1e-4

# --- test case 1: HeH+, zeta(He) = 2.0925, zeta(H) = 1.24, R = 1.4632 ---
assert _t_check(sto3g_two_electron(2.0925, 1.24, 1.4632), (2.0925, 1.24, 1.4632)) is not None
# --- test case 2: unequal exponents 1.0 and 1.5 at R = 2.5 ---
assert _t_check(sto3g_two_electron(1.0, 1.5, 2.5), (1.0, 1.5, 2.5)) is not None
# --- test case 3: non-positive distance or exponent raises ValueError ---
for _t_bad in ((1.24, 1.24, 0.0), (1.24, 1.24, -2.0), (0.0, 1.24, 1.4), (1.24, -1.0, 1.4)):
    _t_raised = False
    try:
        sto3g_two_electron(*_t_bad)
    except ValueError:
        _t_raised = True
    assert _t_raised, "sto3g_two_electron%r must raise ValueError" % (_t_bad,)
