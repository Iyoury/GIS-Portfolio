# Independent integrals for s and p Cartesian Gaussians: Obara-Saika recursions (overlap,
# nuclear attraction, electron repulsion), kinetic energy as (1/2) sum_i <d_i a | d_i b>,
# and the Boys function from Kummer's function: F_m(T) = 1F1(m + 1/2; m + 3/2; -T) / (2m + 1).
# Every test case below is self-contained: it repeats its own imports, helpers and setup.

# --- test case 0 ---
# stretched H2 with polarization functions on both atoms
import numpy as np
from functools import lru_cache
from scipy.special import hyp1f1 as _t_hyp1f1

_T_AL = np.array([0.109818, 0.405771, 2.22766])

_T_DC = np.array([0.444635, 0.535328, 0.154329])

def _t_F(m, T):
    return _t_hyp1f1(m + 0.5, m + 1.5, -T) / (2 * m + 1)

def _t_basis(zA, zB, aA, aB, R):
    out = []
    for cen, z, ap in ((np.zeros(3), zA, aA), (np.array([0.0, 0.0, R]), zB, aB)):
        e = _T_AL * z * z
        out.append([(ee, cc, (0, 0, 0), cen) for ee, cc in zip(e, _T_DC * (2 * e / np.pi) ** 0.75)])
        for ang in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            out.append([(ap, (2 * ap / np.pi) ** 0.75 * 2 * np.sqrt(ap), ang, cen)])
    return out

def _t_ovl1d(a, la, Ax, b, lb, Bx):
    # 1-D Obara-Saika overlap of x^la e^{-a x^2} (centred Ax) and x^lb e^{-b x^2} (centred Bx), without (pi/p)^1/2
    p = a + b
    P = (a * Ax + b * Bx) / p

    @lru_cache(None)
    def S(i, j):
        if i < 0 or j < 0:
            return 0.0
        if i == 0 and j == 0:
            return np.exp(-a * b / p * (Ax - Bx) ** 2)
        if i > 0:
            return (P - Ax) * S(i - 1, j) + ((i - 1) * S(i - 2, j) + j * S(i - 1, j - 1)) / (2 * p)
        return (P - Bx) * S(i, j - 1) + (i * S(i - 1, j - 1) + (j - 1) * S(i, j - 2)) / (2 * p)
    return S(la, lb) * np.sqrt(np.pi / p)

def _t_S(a, la, A, b, lb, B):
    return np.prod([_t_ovl1d(a, la[k], A[k], b, lb[k], B[k]) for k in range(3)])

def _t_T(a, la, A, b, lb, B):
    # (1/2) sum_i <d_i a | d_i b>, d/dx x^l e^{-a x^2} = l x^(l-1) - 2 a x^(l+1)
    tot = 0.0
    for k in range(3):
        da = [(la[k], -2 * a, 1)] + ([(la[k], la[k], -1)] if la[k] > 0 else [])
        db = [(lb[k], -2 * b, 1)] + ([(lb[k], lb[k], -1)] if lb[k] > 0 else [])
        for _, ca, sa in da:
            for _, cb, sb in db:
                l1 = list(la); l1[k] += sa
                l2 = list(lb); l2[k] += sb
                tot += 0.5 * ca * cb * _t_S(a, l1, A, b, l2, B)
    return tot

def _t_V(a, la, A, b, lb, B, C):
    p = a + b
    P = (a * A + b * B) / p
    K = np.exp(-a * b / p * np.sum((A - B) ** 2))
    T = p * np.sum((P - C) ** 2)

    @lru_cache(None)
    def V(l1, l2, m):
        if min(l1) < 0 or min(l2) < 0:
            return 0.0
        if sum(l1) == 0 and sum(l2) == 0:
            return 2 * np.pi / p * K * _t_F(m, T)
        if sum(l1) > 0:
            i = next(k for k in range(3) if l1[k] > 0)
            d1 = tuple(x - (k == i) for k, x in enumerate(l1))
            d11 = tuple(x - (k == i) for k, x in enumerate(d1))
            d2 = tuple(x - (k == i) for k, x in enumerate(l2))
            r = (P[i] - A[i]) * V(d1, l2, m) - (P[i] - C[i]) * V(d1, l2, m + 1)
            r += d1[i] / (2 * p) * (V(d11, l2, m) - V(d11, l2, m + 1))
            r += l2[i] / (2 * p) * (V(d1, d2, m) - V(d1, d2, m + 1))
            return r
        i = next(k for k in range(3) if l2[k] > 0)
        d2 = tuple(x - (k == i) for k, x in enumerate(l2))
        d22 = tuple(x - (k == i) for k, x in enumerate(d2))
        d1 = tuple(x - (k == i) for k, x in enumerate(l1))
        r = (P[i] - B[i]) * V(l1, d2, m) - (P[i] - C[i]) * V(l1, d2, m + 1)
        r += d2[i] / (2 * p) * (V(l1, d22, m) - V(l1, d22, m + 1))
        r += l1[i] / (2 * p) * (V(d1, d2, m) - V(d1, d2, m + 1))
        return r
    return V(tuple(la), tuple(lb), 0)

def _t_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D):
    z, e = a + b, c + d
    P = (a * A + b * B) / z
    Q = (c * C + d * D) / e
    W = (z * P + e * Q) / (z + e)
    rho = z * e / (z + e)
    K = np.exp(-a * b / z * np.sum((A - B) ** 2) - c * d / e * np.sum((C - D) ** 2))
    T = rho * np.sum((P - Q) ** 2)
    pre = 2 * np.pi ** 2.5 / (z * e * np.sqrt(z + e)) * K
    cen = (A, B, C, D)

    def dec(l, i):
        return tuple(x - (k == i) for k, x in enumerate(l))

    @lru_cache(None)
    def I(l1, l2, l3, l4, m):
        ls = (l1, l2, l3, l4)
        if any(min(l) < 0 for l in ls):
            return 0.0
        if all(sum(l) == 0 for l in ls):
            return pre * _t_F(m, T)
        pos = next(q for q in range(4) if sum(ls[q]) > 0)
        i = next(k for k in range(3) if ls[pos][k] > 0)
        L = list(ls)
        L[pos] = dec(ls[pos], i)
        if pos < 2:      # electron 1, exponent z, centre P
            Xi, Ce, own, oth = z, P, pos, 1 - pos
            r = (P[i] - cen[pos][i]) * I(*L, m) + (W[i] - P[i]) * I(*L, m + 1)
            for q in (0, 1):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * z) * (I(*M, m) - rho / z * I(*M, m + 1))
            for q in (2, 3):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * (z + e)) * I(*M, m + 1)
        else:            # electron 2, exponent e, centre Q
            r = (Q[i] - cen[pos][i]) * I(*L, m) + (W[i] - Q[i]) * I(*L, m + 1)
            for q in (2, 3):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * e) * (I(*M, m) - rho / e * I(*M, m + 1))
            for q in (0, 1):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * (z + e)) * I(*M, m + 1)
        return r
    return I(tuple(la), tuple(lb), tuple(lc), tuple(ld), 0)

def _t_integrals(ZA, ZB, zA, zB, aA, aB, R):
    bs = _t_basis(zA, zB, aA, aB, R)
    n = len(bs)
    nuc = ((ZA, np.zeros(3)), (ZB, np.array([0.0, 0.0, R])))
    S = np.zeros((n, n)); H = np.zeros((n, n)); G = np.zeros((n,) * 4)
    for i in range(n):
        for j in range(n):
            for (a, ca, la, A) in bs[i]:
                for (b, cb, lb, B) in bs[j]:
                    S[i, j] += ca * cb * _t_S(a, la, A, b, lb, B)
                    H[i, j] += ca * cb * (_t_T(a, la, A, b, lb, B) - sum(Z * _t_V(a, la, A, b, lb, B, C) for Z, C in nuc))
    done = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in done:
                        done[key] = sum(ca * cb * cc * cd * _t_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D)
                                        for (a, ca, la, A) in bs[i] for (b, cb, lb, B) in bs[j]
                                        for (c, cc, lc, C) in bs[k] for (d, cd, ld, D) in bs[l])
                    G[i, j, k, l] = done[key]
    return S, H, G

def _t_sym(n):
    # columns: normalized symmetric combinations of the product functions phi_i(1) phi_j(2), i <= j
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    B = np.zeros((n * n, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        if i == j:
            B[i * n + i, col] = 1.0
        else:
            B[i * n + j, col] = B[j * n + i, col] = np.sqrt(0.5)
    return B

def _t_energies(ZA, ZB, zA, zB, aA, aB, R):
    # full CI: generalized eigenproblem in the NON-orthogonal symmetric product functions;
    # RHF: global minimum of 2 c.H.c + (cc|cc) over c.S.c = 1 by BFGS from 40 random starts
    from scipy.linalg import eigh as _t_eigh
    from scipy.optimize import minimize as _t_minimize
    S, H, G = _t_integrals(ZA, ZB, zA, zB, aA, aB, R)
    n = S.shape[0]
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H) + np.einsum("ikjl->ijkl", G)).reshape(n * n, n * n)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(n * n, n * n)
    B = _t_sym(n)                       # singlet: symmetric spatial functions only
    e_fci = float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0])

    def E(x):
        c = x / np.sqrt(x @ S @ x)
        return 2 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    rng = np.random.default_rng(2024)
    e_rhf = min(_t_minimize(E, x0, method="BFGS", options={"gtol": 1e-13, "maxiter": 5000}).fun
                for x0 in rng.normal(size=(40, n)))
    enuc = ZA * ZB / R
    return e_fci + enuc, float(e_rhf) + enuc

def _t_check_pol(args):
    out = polarized_energies(*args)
    assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
    t_fci, t_rhf = _t_energies(*args)
    assert abs(out[0] - t_fci) < 1e-9, (args, out, t_fci)
    assert abs(out[1] - t_rhf) < 1e-9, (args, out, t_rhf)
    return out

assert _t_check_pol((1.0, 1.0, 1.24, 1.24, 1.0, 1.0, 3.0)) is not None

# --- test case 1 ---
# HeH+ with a tight p shell on He and a diffuse one on H
import numpy as np
from functools import lru_cache
from scipy.special import hyp1f1 as _t_hyp1f1

_T_AL = np.array([0.109818, 0.405771, 2.22766])

_T_DC = np.array([0.444635, 0.535328, 0.154329])

def _t_F(m, T):
    return _t_hyp1f1(m + 0.5, m + 1.5, -T) / (2 * m + 1)

def _t_basis(zA, zB, aA, aB, R):
    out = []
    for cen, z, ap in ((np.zeros(3), zA, aA), (np.array([0.0, 0.0, R]), zB, aB)):
        e = _T_AL * z * z
        out.append([(ee, cc, (0, 0, 0), cen) for ee, cc in zip(e, _T_DC * (2 * e / np.pi) ** 0.75)])
        for ang in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            out.append([(ap, (2 * ap / np.pi) ** 0.75 * 2 * np.sqrt(ap), ang, cen)])
    return out

def _t_ovl1d(a, la, Ax, b, lb, Bx):
    # 1-D Obara-Saika overlap of x^la e^{-a x^2} (centred Ax) and x^lb e^{-b x^2} (centred Bx), without (pi/p)^1/2
    p = a + b
    P = (a * Ax + b * Bx) / p

    @lru_cache(None)
    def S(i, j):
        if i < 0 or j < 0:
            return 0.0
        if i == 0 and j == 0:
            return np.exp(-a * b / p * (Ax - Bx) ** 2)
        if i > 0:
            return (P - Ax) * S(i - 1, j) + ((i - 1) * S(i - 2, j) + j * S(i - 1, j - 1)) / (2 * p)
        return (P - Bx) * S(i, j - 1) + (i * S(i - 1, j - 1) + (j - 1) * S(i, j - 2)) / (2 * p)
    return S(la, lb) * np.sqrt(np.pi / p)

def _t_S(a, la, A, b, lb, B):
    return np.prod([_t_ovl1d(a, la[k], A[k], b, lb[k], B[k]) for k in range(3)])

def _t_T(a, la, A, b, lb, B):
    # (1/2) sum_i <d_i a | d_i b>, d/dx x^l e^{-a x^2} = l x^(l-1) - 2 a x^(l+1)
    tot = 0.0
    for k in range(3):
        da = [(la[k], -2 * a, 1)] + ([(la[k], la[k], -1)] if la[k] > 0 else [])
        db = [(lb[k], -2 * b, 1)] + ([(lb[k], lb[k], -1)] if lb[k] > 0 else [])
        for _, ca, sa in da:
            for _, cb, sb in db:
                l1 = list(la); l1[k] += sa
                l2 = list(lb); l2[k] += sb
                tot += 0.5 * ca * cb * _t_S(a, l1, A, b, l2, B)
    return tot

def _t_V(a, la, A, b, lb, B, C):
    p = a + b
    P = (a * A + b * B) / p
    K = np.exp(-a * b / p * np.sum((A - B) ** 2))
    T = p * np.sum((P - C) ** 2)

    @lru_cache(None)
    def V(l1, l2, m):
        if min(l1) < 0 or min(l2) < 0:
            return 0.0
        if sum(l1) == 0 and sum(l2) == 0:
            return 2 * np.pi / p * K * _t_F(m, T)
        if sum(l1) > 0:
            i = next(k for k in range(3) if l1[k] > 0)
            d1 = tuple(x - (k == i) for k, x in enumerate(l1))
            d11 = tuple(x - (k == i) for k, x in enumerate(d1))
            d2 = tuple(x - (k == i) for k, x in enumerate(l2))
            r = (P[i] - A[i]) * V(d1, l2, m) - (P[i] - C[i]) * V(d1, l2, m + 1)
            r += d1[i] / (2 * p) * (V(d11, l2, m) - V(d11, l2, m + 1))
            r += l2[i] / (2 * p) * (V(d1, d2, m) - V(d1, d2, m + 1))
            return r
        i = next(k for k in range(3) if l2[k] > 0)
        d2 = tuple(x - (k == i) for k, x in enumerate(l2))
        d22 = tuple(x - (k == i) for k, x in enumerate(d2))
        d1 = tuple(x - (k == i) for k, x in enumerate(l1))
        r = (P[i] - B[i]) * V(l1, d2, m) - (P[i] - C[i]) * V(l1, d2, m + 1)
        r += d2[i] / (2 * p) * (V(l1, d22, m) - V(l1, d22, m + 1))
        r += l1[i] / (2 * p) * (V(d1, d2, m) - V(d1, d2, m + 1))
        return r
    return V(tuple(la), tuple(lb), 0)

def _t_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D):
    z, e = a + b, c + d
    P = (a * A + b * B) / z
    Q = (c * C + d * D) / e
    W = (z * P + e * Q) / (z + e)
    rho = z * e / (z + e)
    K = np.exp(-a * b / z * np.sum((A - B) ** 2) - c * d / e * np.sum((C - D) ** 2))
    T = rho * np.sum((P - Q) ** 2)
    pre = 2 * np.pi ** 2.5 / (z * e * np.sqrt(z + e)) * K
    cen = (A, B, C, D)

    def dec(l, i):
        return tuple(x - (k == i) for k, x in enumerate(l))

    @lru_cache(None)
    def I(l1, l2, l3, l4, m):
        ls = (l1, l2, l3, l4)
        if any(min(l) < 0 for l in ls):
            return 0.0
        if all(sum(l) == 0 for l in ls):
            return pre * _t_F(m, T)
        pos = next(q for q in range(4) if sum(ls[q]) > 0)
        i = next(k for k in range(3) if ls[pos][k] > 0)
        L = list(ls)
        L[pos] = dec(ls[pos], i)
        if pos < 2:      # electron 1, exponent z, centre P
            Xi, Ce, own, oth = z, P, pos, 1 - pos
            r = (P[i] - cen[pos][i]) * I(*L, m) + (W[i] - P[i]) * I(*L, m + 1)
            for q in (0, 1):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * z) * (I(*M, m) - rho / z * I(*M, m + 1))
            for q in (2, 3):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * (z + e)) * I(*M, m + 1)
        else:            # electron 2, exponent e, centre Q
            r = (Q[i] - cen[pos][i]) * I(*L, m) + (W[i] - Q[i]) * I(*L, m + 1)
            for q in (2, 3):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * e) * (I(*M, m) - rho / e * I(*M, m + 1))
            for q in (0, 1):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * (z + e)) * I(*M, m + 1)
        return r
    return I(tuple(la), tuple(lb), tuple(lc), tuple(ld), 0)

def _t_integrals(ZA, ZB, zA, zB, aA, aB, R):
    bs = _t_basis(zA, zB, aA, aB, R)
    n = len(bs)
    nuc = ((ZA, np.zeros(3)), (ZB, np.array([0.0, 0.0, R])))
    S = np.zeros((n, n)); H = np.zeros((n, n)); G = np.zeros((n,) * 4)
    for i in range(n):
        for j in range(n):
            for (a, ca, la, A) in bs[i]:
                for (b, cb, lb, B) in bs[j]:
                    S[i, j] += ca * cb * _t_S(a, la, A, b, lb, B)
                    H[i, j] += ca * cb * (_t_T(a, la, A, b, lb, B) - sum(Z * _t_V(a, la, A, b, lb, B, C) for Z, C in nuc))
    done = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in done:
                        done[key] = sum(ca * cb * cc * cd * _t_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D)
                                        for (a, ca, la, A) in bs[i] for (b, cb, lb, B) in bs[j]
                                        for (c, cc, lc, C) in bs[k] for (d, cd, ld, D) in bs[l])
                    G[i, j, k, l] = done[key]
    return S, H, G

def _t_sym(n):
    # columns: normalized symmetric combinations of the product functions phi_i(1) phi_j(2), i <= j
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    B = np.zeros((n * n, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        if i == j:
            B[i * n + i, col] = 1.0
        else:
            B[i * n + j, col] = B[j * n + i, col] = np.sqrt(0.5)
    return B

def _t_energies(ZA, ZB, zA, zB, aA, aB, R):
    # full CI: generalized eigenproblem in the NON-orthogonal symmetric product functions;
    # RHF: global minimum of 2 c.H.c + (cc|cc) over c.S.c = 1 by BFGS from 40 random starts
    from scipy.linalg import eigh as _t_eigh
    from scipy.optimize import minimize as _t_minimize
    S, H, G = _t_integrals(ZA, ZB, zA, zB, aA, aB, R)
    n = S.shape[0]
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H) + np.einsum("ikjl->ijkl", G)).reshape(n * n, n * n)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(n * n, n * n)
    B = _t_sym(n)                       # singlet: symmetric spatial functions only
    e_fci = float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0])

    def E(x):
        c = x / np.sqrt(x @ S @ x)
        return 2 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    rng = np.random.default_rng(2024)
    e_rhf = min(_t_minimize(E, x0, method="BFGS", options={"gtol": 1e-13, "maxiter": 5000}).fun
                for x0 in rng.normal(size=(40, n)))
    enuc = ZA * ZB / R
    return e_fci + enuc, float(e_rhf) + enuc

def _t_check_pol(args):
    out = polarized_energies(*args)
    assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
    t_fci, t_rhf = _t_energies(*args)
    assert abs(out[0] - t_fci) < 1e-9, (args, out, t_fci)
    assert abs(out[1] - t_rhf) < 1e-9, (args, out, t_rhf)
    return out

assert _t_check_pol((2.0, 1.0, 2.0925, 1.24, 3.5, 0.15, 1.8)) is not None

# --- test case 2 ---
# unequal non-integer charges
import numpy as np
from functools import lru_cache
from scipy.special import hyp1f1 as _t_hyp1f1

_T_AL = np.array([0.109818, 0.405771, 2.22766])

_T_DC = np.array([0.444635, 0.535328, 0.154329])

def _t_F(m, T):
    return _t_hyp1f1(m + 0.5, m + 1.5, -T) / (2 * m + 1)

def _t_basis(zA, zB, aA, aB, R):
    out = []
    for cen, z, ap in ((np.zeros(3), zA, aA), (np.array([0.0, 0.0, R]), zB, aB)):
        e = _T_AL * z * z
        out.append([(ee, cc, (0, 0, 0), cen) for ee, cc in zip(e, _T_DC * (2 * e / np.pi) ** 0.75)])
        for ang in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            out.append([(ap, (2 * ap / np.pi) ** 0.75 * 2 * np.sqrt(ap), ang, cen)])
    return out

def _t_ovl1d(a, la, Ax, b, lb, Bx):
    # 1-D Obara-Saika overlap of x^la e^{-a x^2} (centred Ax) and x^lb e^{-b x^2} (centred Bx), without (pi/p)^1/2
    p = a + b
    P = (a * Ax + b * Bx) / p

    @lru_cache(None)
    def S(i, j):
        if i < 0 or j < 0:
            return 0.0
        if i == 0 and j == 0:
            return np.exp(-a * b / p * (Ax - Bx) ** 2)
        if i > 0:
            return (P - Ax) * S(i - 1, j) + ((i - 1) * S(i - 2, j) + j * S(i - 1, j - 1)) / (2 * p)
        return (P - Bx) * S(i, j - 1) + (i * S(i - 1, j - 1) + (j - 1) * S(i, j - 2)) / (2 * p)
    return S(la, lb) * np.sqrt(np.pi / p)

def _t_S(a, la, A, b, lb, B):
    return np.prod([_t_ovl1d(a, la[k], A[k], b, lb[k], B[k]) for k in range(3)])

def _t_T(a, la, A, b, lb, B):
    # (1/2) sum_i <d_i a | d_i b>, d/dx x^l e^{-a x^2} = l x^(l-1) - 2 a x^(l+1)
    tot = 0.0
    for k in range(3):
        da = [(la[k], -2 * a, 1)] + ([(la[k], la[k], -1)] if la[k] > 0 else [])
        db = [(lb[k], -2 * b, 1)] + ([(lb[k], lb[k], -1)] if lb[k] > 0 else [])
        for _, ca, sa in da:
            for _, cb, sb in db:
                l1 = list(la); l1[k] += sa
                l2 = list(lb); l2[k] += sb
                tot += 0.5 * ca * cb * _t_S(a, l1, A, b, l2, B)
    return tot

def _t_V(a, la, A, b, lb, B, C):
    p = a + b
    P = (a * A + b * B) / p
    K = np.exp(-a * b / p * np.sum((A - B) ** 2))
    T = p * np.sum((P - C) ** 2)

    @lru_cache(None)
    def V(l1, l2, m):
        if min(l1) < 0 or min(l2) < 0:
            return 0.0
        if sum(l1) == 0 and sum(l2) == 0:
            return 2 * np.pi / p * K * _t_F(m, T)
        if sum(l1) > 0:
            i = next(k for k in range(3) if l1[k] > 0)
            d1 = tuple(x - (k == i) for k, x in enumerate(l1))
            d11 = tuple(x - (k == i) for k, x in enumerate(d1))
            d2 = tuple(x - (k == i) for k, x in enumerate(l2))
            r = (P[i] - A[i]) * V(d1, l2, m) - (P[i] - C[i]) * V(d1, l2, m + 1)
            r += d1[i] / (2 * p) * (V(d11, l2, m) - V(d11, l2, m + 1))
            r += l2[i] / (2 * p) * (V(d1, d2, m) - V(d1, d2, m + 1))
            return r
        i = next(k for k in range(3) if l2[k] > 0)
        d2 = tuple(x - (k == i) for k, x in enumerate(l2))
        d22 = tuple(x - (k == i) for k, x in enumerate(d2))
        d1 = tuple(x - (k == i) for k, x in enumerate(l1))
        r = (P[i] - B[i]) * V(l1, d2, m) - (P[i] - C[i]) * V(l1, d2, m + 1)
        r += d2[i] / (2 * p) * (V(l1, d22, m) - V(l1, d22, m + 1))
        r += l1[i] / (2 * p) * (V(d1, d2, m) - V(d1, d2, m + 1))
        return r
    return V(tuple(la), tuple(lb), 0)

def _t_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D):
    z, e = a + b, c + d
    P = (a * A + b * B) / z
    Q = (c * C + d * D) / e
    W = (z * P + e * Q) / (z + e)
    rho = z * e / (z + e)
    K = np.exp(-a * b / z * np.sum((A - B) ** 2) - c * d / e * np.sum((C - D) ** 2))
    T = rho * np.sum((P - Q) ** 2)
    pre = 2 * np.pi ** 2.5 / (z * e * np.sqrt(z + e)) * K
    cen = (A, B, C, D)

    def dec(l, i):
        return tuple(x - (k == i) for k, x in enumerate(l))

    @lru_cache(None)
    def I(l1, l2, l3, l4, m):
        ls = (l1, l2, l3, l4)
        if any(min(l) < 0 for l in ls):
            return 0.0
        if all(sum(l) == 0 for l in ls):
            return pre * _t_F(m, T)
        pos = next(q for q in range(4) if sum(ls[q]) > 0)
        i = next(k for k in range(3) if ls[pos][k] > 0)
        L = list(ls)
        L[pos] = dec(ls[pos], i)
        if pos < 2:      # electron 1, exponent z, centre P
            Xi, Ce, own, oth = z, P, pos, 1 - pos
            r = (P[i] - cen[pos][i]) * I(*L, m) + (W[i] - P[i]) * I(*L, m + 1)
            for q in (0, 1):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * z) * (I(*M, m) - rho / z * I(*M, m + 1))
            for q in (2, 3):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * (z + e)) * I(*M, m + 1)
        else:            # electron 2, exponent e, centre Q
            r = (Q[i] - cen[pos][i]) * I(*L, m) + (W[i] - Q[i]) * I(*L, m + 1)
            for q in (2, 3):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * e) * (I(*M, m) - rho / e * I(*M, m + 1))
            for q in (0, 1):
                n = L[q][i]
                if n > 0:
                    M = list(L); M[q] = dec(L[q], i)
                    r += n / (2 * (z + e)) * I(*M, m + 1)
        return r
    return I(tuple(la), tuple(lb), tuple(lc), tuple(ld), 0)

def _t_integrals(ZA, ZB, zA, zB, aA, aB, R):
    bs = _t_basis(zA, zB, aA, aB, R)
    n = len(bs)
    nuc = ((ZA, np.zeros(3)), (ZB, np.array([0.0, 0.0, R])))
    S = np.zeros((n, n)); H = np.zeros((n, n)); G = np.zeros((n,) * 4)
    for i in range(n):
        for j in range(n):
            for (a, ca, la, A) in bs[i]:
                for (b, cb, lb, B) in bs[j]:
                    S[i, j] += ca * cb * _t_S(a, la, A, b, lb, B)
                    H[i, j] += ca * cb * (_t_T(a, la, A, b, lb, B) - sum(Z * _t_V(a, la, A, b, lb, B, C) for Z, C in nuc))
    done = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in done:
                        done[key] = sum(ca * cb * cc * cd * _t_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D)
                                        for (a, ca, la, A) in bs[i] for (b, cb, lb, B) in bs[j]
                                        for (c, cc, lc, C) in bs[k] for (d, cd, ld, D) in bs[l])
                    G[i, j, k, l] = done[key]
    return S, H, G

def _t_sym(n):
    # columns: normalized symmetric combinations of the product functions phi_i(1) phi_j(2), i <= j
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    B = np.zeros((n * n, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        if i == j:
            B[i * n + i, col] = 1.0
        else:
            B[i * n + j, col] = B[j * n + i, col] = np.sqrt(0.5)
    return B

def _t_energies(ZA, ZB, zA, zB, aA, aB, R):
    # full CI: generalized eigenproblem in the NON-orthogonal symmetric product functions;
    # RHF: global minimum of 2 c.H.c + (cc|cc) over c.S.c = 1 by BFGS from 40 random starts
    from scipy.linalg import eigh as _t_eigh
    from scipy.optimize import minimize as _t_minimize
    S, H, G = _t_integrals(ZA, ZB, zA, zB, aA, aB, R)
    n = S.shape[0]
    Hp = (np.einsum("ik,jl->ijkl", H, S) + np.einsum("ik,jl->ijkl", S, H) + np.einsum("ikjl->ijkl", G)).reshape(n * n, n * n)
    Sp = np.einsum("ik,jl->ijkl", S, S).reshape(n * n, n * n)
    B = _t_sym(n)                       # singlet: symmetric spatial functions only
    e_fci = float(_t_eigh(B.T @ Hp @ B, B.T @ Sp @ B, eigvals_only=True)[0])

    def E(x):
        c = x / np.sqrt(x @ S @ x)
        return 2 * (c @ H @ c) + np.einsum("i,j,k,l,ijkl->", c, c, c, c, G)

    rng = np.random.default_rng(2024)
    e_rhf = min(_t_minimize(E, x0, method="BFGS", options={"gtol": 1e-13, "maxiter": 5000}).fun
                for x0 in rng.normal(size=(40, n)))
    enuc = ZA * ZB / R
    return e_fci + enuc, float(e_rhf) + enuc

def _t_check_pol(args):
    out = polarized_energies(*args)
    assert isinstance(out, tuple) and len(out) == 2 and all(type(x) is float for x in out), out
    t_fci, t_rhf = _t_energies(*args)
    assert abs(out[0] - t_fci) < 1e-9, (args, out, t_fci)
    assert abs(out[1] - t_rhf) < 1e-9, (args, out, t_rhf)
    return out

assert _t_check_pol((2.4, 1.3, 1.9, 1.1, 0.6, 0.9, 2.2)) is not None
