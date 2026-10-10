# Independent targets: no closed-form Gaussian integrals. Every integral is reduced to a
# one-dimensional radial integral with the shell theorem (averages over spheres) and
# done with composite Gauss-Legendre quadrature.
# Independent RHF target: direct minimisation over the single orbital-mixing angle
# (no Fock matrix, no self-consistent iteration).
# Independent full-CI target: lowest eigenvalue of the two-electron Hamiltonian in the
# symmetric (singlet) combinations of the non-orthogonal products phi_i(1) phi_j(2) (generalized eigenproblem with overlap
# S (x) S), with the quadrature integrals of each case (no orbitals, no configuration functions).
# Every test case below is self-contained: it repeats its own imports, helpers and setup.

# --- test case 0 ---
# H2 at R = 1.4, zeta = 1.24 (Szabo and Ostlund: RHF -1.1167, full CI -1.1373)
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

E_fci, E_rhf = _t_check((1.0, 1.0, 1.24, 1.24, 1.4))
assert abs(E_rhf + 1.1167) < 2e-4 and abs(E_fci + 1.1373) < 2e-4

# --- test case 1 ---
# HeH+ at R = 1.4632 (Szabo and Ostlund RHF: -2.8607) and stretched to 3.0
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

E_fci, E_rhf = _t_check((2.0, 1.0, 2.0925, 1.24, 1.4632))
assert abs(E_rhf + 2.8607) < 2e-4
assert _t_check((2.0, 1.0, 2.0925, 1.24, 3.0)) is not None

# --- test case 2 ---
# H2 stretched to 6 bohr: RHF keeps half ionic character and fails, full CI
# goes to two hydrogen atoms (2 x -0.466582 hartree in this basis)
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

E_fci, E_rhf = _t_check((1.0, 1.0, 1.24, 1.24, 6.0))
assert E_rhf - E_fci > 0.1
assert abs(E_fci + 0.933164) < 2e-3

# --- test case 3 ---
# unequal exponents, compressed bond and an ion-pair-like curve
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

assert _t_check((1.0, 1.0, 1.0, 1.5, 0.8)) is not None
assert _t_check((1.0, 1.0, 1.24, 2.69, 1.05)) is not None
assert _t_check((1.5, 1.5, 2.0, 1.24, 2.5)) is not None

# --- test case 4 ---
# H2 is bound in both methods and full CI is never above RHF
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

E1 = _t_check((1.0, 1.0, 1.24, 1.24, 1.4))
E5 = fci_energy(1.0, 1.0, 1.24, 1.24, 5.0)
assert E5[0] > E1[0] and E5[1] > E1[1] and E5[0] <= E5[1] + 2e-9   # two outputs, each within 1e-9

# --- test case 5 ---
# stretched, asymmetric molecules with two closed-shell branches (both electrons near A
# or both near B): RHF is the lower of them. A self-consistent-field run from the core-Hamiltonian guess
# settles here on the higher branch, 2.07 to 2.61 hartree above the global minimum
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

for _t_args in ((1.5, 1.5, 2.0, 3.0, 10.0), (2.0, 1.0, 3.0, 2.0, 6.0), (2.0, 1.0, 3.0, 2.0, 20.0)):
    E_fci, E_rhf = _t_check(_t_args)
    assert E_fci <= E_rhf + 2e-9, (_t_args, E_fci, E_rhf)   # two outputs, each within 1e-9

# --- test case 6 ---
# diffuse exponents at stretched R: in this basis the triplet lies several millihartree below the
# lowest singlet, so the lowest eigenvalue of the whole two-electron product space is not E_fci
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

def _t_triplet(ZA, ZB, za, zb, R):
    # the spatially antisymmetric (triplet) function phi_0(1) phi_1(2) - phi_1(1) phi_0(2)
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    b = np.array([0.0, 1.0, -1.0, 0.0])
    return float(b @ Hp @ b / (b @ Sp @ b)) + ZA * ZB / R

for _t_args in ((2.0, 2.0, 0.8, 0.8, 4.0), (1.5, 1.5, 0.8, 0.8, 5.0)):
    E_fci, E_rhf = _t_check(_t_args)
    # the case discriminates: the triplet is more than 1e-3 hartree below the singlet target
    assert _t_triplet(*_t_args) < _t_fci(*_t_args) - 1e-3, _t_args

# --- test case 7 ---
# ends of the valid ranges: the largest charges with the most diffuse exponents and a tight/diffuse
# pair at the shortest distance R = 0.1 bohr (nearly linearly dependent basis functions), and an
# asymmetric molecule with two closed-shell branches at R = 100 bohr
import numpy as np
from scipy.special import erf as _t_erf

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

def _t_energy(ZA, ZB, za, zb, R):
    from scipy.optimize import minimize_scalar
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)

    def e(th):
        v = np.array([np.cos(th), np.sin(th)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    # the energy can have several local minima (one closed-shell branch on each atom when the molecule
    # is stretched or asymmetric): every local minimum of a fine scan over the whole period is refined
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2001)
    vals = np.array([e(t) for t in grid])
    best = np.inf
    for i in range(len(grid)):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[(i + 1) % len(grid)]:
            res = minimize_scalar(e, bounds=(grid[i] - (grid[1] - grid[0]), grid[i] + (grid[1] - grid[0])),
                                  method="bounded", options={"xatol": 1e-13})
            best = min(best, float(res.fun))
    return best + ZA * ZB / R

def _t_fci(ZA, ZB, za, zb, R):
    from scipy.linalg import eigh as _t_eigh
    S, H = _t_one(ZA, ZB, za, zb, R)
    G = _t_eri(za, zb, R)
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H)
          + np.einsum("ikjl->ijkl", G)).reshape(4, 4)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    return float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0]) + ZA * ZB / R

def _t_check(args):
    out = fci_energy(*args)
    assert isinstance(out, tuple) and len(out) == 2, out
    assert type(out[0]) is float and type(out[1]) is float, out
    E_fci, E_rhf = out
    t_fci, t_rhf = _t_fci(*args), _t_energy(*args)
    assert abs(E_fci - t_fci) < 1e-9, (args, E_fci, t_fci)
    assert abs(E_rhf - t_rhf) < 1e-9, (args, E_rhf, t_rhf)
    return E_fci, E_rhf

for _t_args in ((3.0, 3.0, 0.5, 0.5, 0.1), (1.0, 3.0, 3.0, 0.5, 0.1), (3.0, 1.0, 3.0, 0.5, 100.0)):
    E_fci, E_rhf = _t_check(_t_args)
    assert E_fci <= E_rhf + 2e-9, (_t_args, E_fci, E_rhf)   # two outputs, each within 1e-9
