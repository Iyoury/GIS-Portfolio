# --- test case 0 ---
# Independent targets: no closed-form Gaussian integrals. Every integral is reduced to a
# one-dimensional radial integral with the shell theorem (averages over spheres) and
# done with composite Gauss-Legendre quadrature.
# Every test case below is self-contained: it repeats its own imports, helpers and setup.
# H2, zeta = 1.24, R = 1.4 (Szabo and Ostlund, Section 3.5.2)
import numpy as np


def _crown_case():
    import numpy as np

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

    def _t_check(out, args):
        # the output contract (checked once in this step): a tuple of two float arrays of shape (2, 2)
        assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating) and x.shape == (2, 2) for x in out), out
        S, H = out
        S_t, H_t = _t_one(*args)
        assert np.max(np.abs(np.asarray(S) - np.asarray(S_t))) <= 1e-10, (S, S_t)
        assert np.max(np.abs(np.asarray(H) - np.asarray(H_t))) <= 1e-10, (H, H_t)
        # symmetry: a comparison between two computed entries, each within 1e-10 of a symmetric target
        assert np.max(np.abs(S - S.T)) <= 2e-10 and np.max(np.abs(H - H.T)) <= 2e-10, (S, H)
        return S, H

    S, H = _t_check(sto3g_one_electron(1.0, 1.0, 1.24, 1.24, 1.4), (1.0, 1.0, 1.24, 1.24, 1.4))
    assert abs(S[0, 1] - 0.6593) < 1e-4 and abs(H[0, 0] + 1.1204) < 1e-4 and abs(H[0, 1] + 0.9584) < 1e-4


_crown_case()

# --- test case 1 ---
# HeH+, zeta(He) = 2.0925, zeta(H) = 1.24, R = 1.4632 (Szabo and Ostlund, Appendix B)
import numpy as np


def _crown_case():
    import numpy as np

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

    def _t_check(out, args):
        S, H = out
        S_t, H_t = _t_one(*args)
        assert np.max(np.abs(np.asarray(S) - np.asarray(S_t))) <= 1e-10, (S, S_t)
        assert np.max(np.abs(np.asarray(H) - np.asarray(H_t))) <= 1e-10, (H, H_t)
        # symmetry: a comparison between two computed entries, each within 1e-10 of a symmetric target
        assert np.max(np.abs(S - S.T)) <= 2e-10 and np.max(np.abs(H - H.T)) <= 2e-10, (S, H)
        return S, H

    S, H = _t_check(sto3g_one_electron(2.0, 1.0, 2.0925, 1.24, 1.4632), (2.0, 1.0, 2.0925, 1.24, 1.4632))
    assert abs(S[0, 1] - 0.4508) < 1e-4 and abs(H[0, 0] + 2.6527) < 1e-4
    assert abs(H[0, 1] + 1.3472) < 1e-4 and abs(H[1, 1] + 1.7318) < 1e-4


_crown_case()

# --- test case 2 ---
# H2, zeta = 1.0, R = 2.0
import numpy as np


def _crown_case():
    import numpy as np

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

    def _t_check(out, args):
        S, H = out
        S_t, H_t = _t_one(*args)
        assert np.max(np.abs(np.asarray(S) - np.asarray(S_t))) <= 1e-10, (S, S_t)
        assert np.max(np.abs(np.asarray(H) - np.asarray(H_t))) <= 1e-10, (H, H_t)
        # symmetry: a comparison between two computed entries, each within 1e-10 of a symmetric target
        assert np.max(np.abs(S - S.T)) <= 2e-10 and np.max(np.abs(H - H.T)) <= 2e-10, (S, H)
        return S, H

    assert _t_check(sto3g_one_electron(1.0, 1.0, 1.0, 1.0, 2.0), (1.0, 1.0, 1.0, 1.0, 2.0)) is not None


_crown_case()

# --- test case 3 ---
# HeH+ stretched to R = 3.0
import numpy as np


def _crown_case():
    import numpy as np

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

    def _t_check(out, args):
        S, H = out
        S_t, H_t = _t_one(*args)
        assert np.max(np.abs(np.asarray(S) - np.asarray(S_t))) <= 1e-10, (S, S_t)
        assert np.max(np.abs(np.asarray(H) - np.asarray(H_t))) <= 1e-10, (H, H_t)
        # symmetry: a comparison between two computed entries, each within 1e-10 of a symmetric target
        assert np.max(np.abs(S - S.T)) <= 2e-10 and np.max(np.abs(H - H.T)) <= 2e-10, (S, H)
        return S, H

    assert _t_check(sto3g_one_electron(2.0, 1.0, 2.0925, 1.24, 3.0), (2.0, 1.0, 2.0925, 1.24, 3.0)) is not None


_crown_case()

# --- test case 4 ---
# the stated error rule, once per named input at its limit: R, ZA, ZB, zetaA or zetaB equal to 0 raises ValueError
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((1.0, 1.0, 1.24, 1.24, 0.0), (0.0, 1.0, 1.24, 1.24, 1.4), (1.0, 0.0, 1.24, 1.24, 1.4),
                   (1.0, 1.0, 0.0, 1.24, 1.4), (1.0, 1.0, 1.24, 0.0, 1.4)):
        _t_raised = False
        try:
            sto3g_one_electron(*_t_bad)
        except ValueError:
            _t_raised = True
        assert _t_raised, "sto3g_one_electron%r must raise ValueError" % (_t_bad,)


_crown_case()

# --- test case 5 ---
# ends of the valid ranges: nearly coincident centres (R = 0.02 bohr) with the most diffuse and the
# tightest exponents and the largest charge, and atoms 100 bohr apart, where each function still
# feels the attraction -Z / R of the other nucleus
import numpy as np


def _crown_case():
    import numpy as np

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

    def _t_check(out, args):
        S, H = out
        S_t, H_t = _t_one(*args)
        assert np.max(np.abs(np.asarray(S) - np.asarray(S_t))) <= 1e-10, (S, S_t)
        assert np.max(np.abs(np.asarray(H) - np.asarray(H_t))) <= 1e-10, (H, H_t)
        # symmetry: a comparison between two computed entries, each within 1e-10 of a symmetric target
        assert np.max(np.abs(S - S.T)) <= 2e-10 and np.max(np.abs(H - H.T)) <= 2e-10, (S, H)
        return S, H

    for _t_args in ((3.0, 3.0, 0.5, 0.5, 0.02), (1.0, 3.0, 3.0, 0.5, 0.02),
                    (3.0, 1.0, 3.0, 0.5, 100.0), (1.0, 1.0, 0.5, 0.5, 100.0)):
        assert _t_check(sto3g_one_electron(*_t_args), _t_args) is not None


_crown_case()

# --- test case 6 ---
# Exact relations with independent targets (no quadrature, no reference values):
# (a) one-centre norm: S[0, 0] = S[1, 1] = sum_ij d_i d_j (2 sqrt(a_i a_j) / (a_i + a_j))**(3/2) for every
#     zeta and R (the zeta**2 scaling of the exponents cancels in the ratio);
# (b) separated atoms at R = 100 bohr (every cross term below 1e-50, the cloud beyond the other nucleus
#     below erfc(23)): S[0, 1] = H[0, 1] = 0 and H[0, 0] = T_A - ZA U_A - ZB S[0, 0] / R with the one-centre
#     closed forms T_A = sum d_i d_j n_i n_j 3 a_i a_j / (a_i + a_j) (pi / (a_i + a_j))**(3/2) and
#     U_A = sum d_i d_j n_i n_j 2 pi / (a_i + a_j), n_i = (2 a_i / pi)**(3/4), and the shell theorem for the
#     other nucleus (likewise H[1, 1]);
# (c) scaling: dividing every length by lam (zeta -> lam zeta, R -> R / lam) and multiplying the charges by
#     lam leaves S unchanged and multiplies H by lam**2 (kinetic energy ~ length**-2, attraction ~ charge /
#     length); two computed outputs, each within 1e-10: tolerances 2e-10 for S and (1 + lam**2) 1e-10 for H;
# (d) mirror symmetry: exchanging the atoms (ZA <-> ZB, zetaA <-> zetaB) exchanges the indices 0 and 1.
import numpy as np


def _crown_case():
    import numpy as np

    _T_ALPHA = np.array([0.109818, 0.405771, 2.22766])

    _T_D = np.array([0.444635, 0.535328, 0.154329])

    def _t_centre(zeta):
        a = _T_ALPHA * zeta ** 2
        w = np.outer(_T_D * (2 * a / np.pi) ** 0.75, _T_D * (2 * a / np.pi) ** 0.75)
        p, ab = np.add.outer(a, a), np.multiply.outer(a, a)
        return (float(np.sum(w * (np.pi / p) ** 1.5)), float(np.sum(w * 3 * ab / p * (np.pi / p) ** 1.5)),
                float(np.sum(w * 2 * np.pi / p)))

    _t_s0 = float(np.sum(np.outer(_T_D, _T_D) * (2 * np.sqrt(np.outer(_T_ALPHA, _T_ALPHA)) / np.add.outer(_T_ALPHA, _T_ALPHA)) ** 1.5))
    S, H = sto3g_one_electron(1.0, 1.5, 0.6, 1.3, 2.2)
    assert abs(S[0, 0] - _t_s0) < 1e-10 and abs(S[1, 1] - _t_s0) < 1e-10, (S, _t_s0)

    _t_args = (1.7, 2.6, 0.8, 2.2, 100.0)
    Sf, Hf = sto3g_one_electron(*_t_args)
    sA, tA, uA = _t_centre(0.8)
    sB, tB, uB = _t_centre(2.2)
    assert abs(Sf[0, 1]) < 1e-10 and abs(Sf[1, 0]) < 1e-10 and abs(Hf[0, 1]) < 1e-10 and abs(Hf[1, 0]) < 1e-10, (Sf, Hf)
    assert abs(Hf[0, 0] - (tA - 1.7 * uA - 2.6 * sA / 100.0)) < 1e-10, (Hf, tA - 1.7 * uA - 2.6 * sA / 100.0)
    assert abs(Hf[1, 1] - (tB - 2.6 * uB - 1.7 * sB / 100.0)) < 1e-10, (Hf, tB - 2.6 * uB - 1.7 * sB / 100.0)

    _t_lam = 2.0
    S2, H2 = sto3g_one_electron(2.0, 3.0, 1.2, 2.6, 1.1)
    assert np.max(np.abs(S2 - S)) <= 2e-10, (S2, S)
    assert np.max(np.abs(H2 - _t_lam ** 2 * H)) <= (1.0 + _t_lam ** 2) * 1e-10, (H2, H)

    Sm, Hm = sto3g_one_electron(1.5, 1.0, 1.3, 0.6, 2.2)
    assert np.max(np.abs(Sm - S[::-1, ::-1])) <= 2e-10 and np.max(np.abs(Hm - H[::-1, ::-1])) <= 2e-10, (Sm, Hm, S, H)


_crown_case()
