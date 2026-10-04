import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def boys_function(n_max, t):
    '''Boys functions F_n(t) = integral from 0 to 1 of u**(2n) exp(-t u**2) du for n = 0..n_max.'''
    # Other method: the regularized lower incomplete gamma function,
    # F_n(t) = Gamma(n + 1/2) P(n + 1/2, t) / (2 t**(n + 1/2)), and a two-term series for tiny t.
    from scipy.special import gammainc, gammaln
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)) or not (0 <= n_max <= 16):
        raise ValueError("n_max must be an integer from 0 to 16")
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0) or np.any(t_arr > 1e6):
        raise ValueError("t must be finite with 0 <= t <= 1e6")
    F = np.empty(t_arr.shape + (int(n_max) + 1,))
    tiny = t_arr < 1e-9
    ts = np.where(tiny, 1.0, t_arr)
    for n in range(int(n_max) + 1):
        a = n + 0.5
        val = gammainc(a, ts) * np.exp(gammaln(a) - a * np.log(ts)) / 2.0
        F[..., n] = np.where(tiny, 1.0 / (2 * n + 1) - t_arr / (2 * n + 3), val)
    return F


def sto3g_one_electron(ZA, ZB, zetaA, zetaB, R):
    '''Overlap and core-Hamiltonian matrices for two atoms with one STO-3G 1s function each.

    Inputs:
      ZA, ZB: float, nuclear charges of atom A (at the origin) and atom B (at distance R on z), > 0.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, > 0.
      R: float, distance between the nuclei in bohr, R > 0.

    Output:
      (S, H): two float numpy arrays of shape (2, 2). Index 0 is the function on A and
      index 1 the function on B. S is the overlap matrix and H = T + V is the core
      Hamiltonian (kinetic energy plus attraction to both nuclei), in hartree.
      Absolute error below 1e-10 for every entry.

    Raises:
      ValueError if R, ZA, ZB, zetaA or zetaB is not positive.
    '''
    # Other method: no closed-form Gaussian integrals. Every integral is reduced to a
    # one-dimensional radial integral with the shell theorem (angular averages over
    # spheres) and done with composite Gauss-Legendre quadrature.
    if not (R > 0 and ZA > 0 and ZB > 0 and zetaA > 0 and zetaB > 0):
        raise ValueError("R, the nuclear charges and the Slater exponents must be positive")
    alpha_one = np.array([0.109818, 0.405771, 2.22766])
    coef = np.array([0.444635, 0.535328, 0.154329])
    centers = np.array([0.0, float(R)])
    charges = np.array([float(ZA), float(ZB)])
    expo = [alpha_one * zetaA ** 2, alpha_one * zetaB ** 2]
    x16, w16 = np.polynomial.legendre.leggauss(16)
    t_nodes, t_wts = np.polynomial.legendre.leggauss(80)
    t_nodes = 0.5 * (t_nodes + 1.0)
    t_wts = 0.5 * t_wts

    def radial_grid(s_max, width):
        n_pan = int(np.ceil(s_max / width))
        edges = np.linspace(0.0, s_max, n_pan + 1)
        half = 0.5 * (edges[1:] - edges[:-1])
        mid = 0.5 * (edges[1:] + edges[:-1])
        s = (mid[:, None] + half[:, None] * x16[None, :]).ravel()
        ws = (half[:, None] * w16[None, :]).ravel()
        return s, ws

    def sphere_avg(beta, s, dist):
        # average of exp(-beta |r - X|^2) over a sphere of radius s whose centre is at distance dist from X
        if dist < 1e-12:
            return np.exp(-beta * s * s)
        return -np.exp(-beta * (s - dist) ** 2) * np.expm1(-4.0 * beta * s * dist) / (4.0 * beta * s * dist)

    S = np.zeros((2, 2))
    H = np.zeros((2, 2))
    for m in range(2):
        for n in range(2):
            a = np.repeat(expo[m], 3)[:, None]
            b = np.tile(expo[n], 3)[:, None]
            w = (np.repeat(coef, 3) * np.tile(coef, 3)) * (2.0 * a[:, 0] / np.pi) ** 0.75 * (2.0 * b[:, 0] / np.pi) ** 0.75
            dist = abs(centers[n] - centers[m])
            s, ws = radial_grid(dist + 10.0 / np.sqrt(np.min(a + b)), min(0.5, 0.5 / np.sqrt(np.max(a + b))))
            s = s[None, :]
            over = np.sum(4.0 * np.pi * s * s * np.exp(-a * s * s) * sphere_avg(b, s, dist) * ws, axis=1)
            kin = np.sum(4.0 * np.pi * s * s * (3.0 * b - 2.0 * b * b * s * s) * np.exp(-b * s * s)
                         * sphere_avg(a, s, dist) * ws, axis=1)
            a = a[:, 0]
            b = b[:, 0]
            p = a + b
            k_ab = np.exp(-a * b / p * dist ** 2)
            center_p = (a * centers[m] + b * centers[n]) / p
            pot = np.zeros(9)
            for C in range(2):
                e_dist = np.abs(center_p - centers[C])
                # integral over space of exp(-p |r-P|^2) / |r - C|: inside the sphere through C the
                # shell average of 1/|r - C| is 1/e_dist, outside it is 1/s (that part is exact)
                # (1/e) int_0^e s^2 exp(-p s^2) ds over [0, min(e, 9 / sqrt(p))]: beyond that the integrand is
                # negligible, and a fixed rule on all of [0, e] misses the narrow peak of a tight function
                top = np.minimum(e_dist, 9.0 / np.sqrt(p))
                inner = np.sum(t_wts[None, :] * t_nodes[None, :] ** 2
                               * np.exp(-p[:, None] * (top[:, None] * t_nodes[None, :]) ** 2), axis=1) \
                    * top ** 3 / np.where(e_dist > 0, e_dist, 1.0)
                pot -= charges[C] * 4.0 * np.pi * (inner + np.exp(-p * e_dist ** 2) / (2.0 * p))
            S[m, n] = np.sum(w * over)
            H[m, n] = np.sum(w * (kin + k_ab * pot))
    return S, H


def sto3g_two_electron(zetaA, zetaB, R):
    '''Two-electron repulsion integrals over the two STO-3G 1s functions.

    Inputs:
      zetaA, zetaB: float, Slater exponents of the 1s functions on A (origin) and B, > 0.
      R: float, distance between the nuclei in bohr, R > 0.

    Output:
      eri: float numpy array of shape (2, 2, 2, 2), in hartree, with
           eri[i, j, k, l] = (ij|kl) = integral of phi_i(1) phi_j(1) (1/r12) phi_k(2) phi_l(2).
           Absolute error below 1e-10 for every entry.

    Raises:
      ValueError if R, zetaA or zetaB is not positive.
    '''
    # Other method: the potential of the charge cloud phi_i phi_j is averaged over spheres
    # around the centre of the cloud phi_k phi_l (shell theorem), then a 1D radial
    # integral is done with composite Gauss-Legendre quadrature. Only the 6 distinct
    # integrals are computed.
    if not (R > 0 and zetaA > 0 and zetaB > 0):
        raise ValueError("R and the Slater exponents must be positive")
    alpha_one = np.array([0.109818, 0.405771, 2.22766])
    coef = np.array([0.444635, 0.535328, 0.154329])
    centers = np.array([0.0, float(R)])
    expo = [alpha_one * zetaA ** 2, alpha_one * zetaB ** 2]
    idx = np.array(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), np.arange(3), indexing='ij')).reshape(4, -1)
    x16, w16 = np.polynomial.legendre.leggauss(16)

    def one_integral(i, j, k, l):
        a = expo[i][idx[0]][:, None]
        b = expo[j][idx[1]][:, None]
        c = expo[k][idx[2]][:, None]
        d = expo[l][idx[3]][:, None]
        w = (coef[idx[0]] * coef[idx[1]] * coef[idx[2]] * coef[idx[3]])[:, None]
        norm = (2.0 * a / np.pi * 2.0 * b / np.pi * 2.0 * c / np.pi * 2.0 * d / np.pi) ** 0.75
        p = a + b
        q = c + d
        center_p = (a * centers[i] + b * centers[j]) / p
        center_q = (c * centers[k] + d * centers[l]) / q
        k_ab = np.exp(-a * b / p * (centers[i] - centers[j]) ** 2)
        k_cd = np.exp(-c * d / q * (centers[k] - centers[l]) ** 2)
        e_dist = np.abs(center_p - center_q)
        s_max = float(np.max(e_dist)) + 10.0 / np.sqrt(np.min(q))
        width = min(1.0, 1.2 / np.sqrt(max(np.max(p), np.max(q))))
        n_pan = int(np.ceil(s_max / width))
        edges = np.linspace(0.0, s_max, n_pan + 1)
        half = 0.5 * (edges[1:] - edges[:-1])
        mid = 0.5 * (edges[1:] + edges[:-1])
        s = (mid[:, None] + half[:, None] * x16[None, :]).ravel()[None, :]
        ws = (half[:, None] * w16[None, :]).ravel()[None, :]
        sp = np.sqrt(p)
        tiny = e_dist < 1e-6
        e_safe = np.where(tiny, 1.0, e_dist)

        def G(u):
            return u * erf(sp * u) + np.exp(-p * u * u) / np.sqrt(np.pi * p)

        far = (G(s + e_safe) - G(s - e_safe)) / (2.0 * s * e_safe)
        near = erf(sp * s) / s
        avg = np.where(tiny, near, far)
        radial = np.sum(4.0 * np.pi * s * s * np.exp(-q * s * s) * avg * ws, axis=1)[:, None]
        vals = norm * k_ab * k_cd * (np.pi / p) ** 1.5 * radial
        return float(np.sum(w * vals))

    eri = np.zeros((2, 2, 2, 2))
    done = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in done:
                        done[key] = one_integral(i, j, k, l)
                    eri[i, j, k, l] = done[key]
    return eri


def fci_energy(ZA, ZB, zetaA, zetaB, R):
    '''Full-CI and closed-shell RHF ground-state energies of a two-electron diatomic in STO-3G.'''
    # Other methods: RHF by direct minimisation over the single orbital-mixing angle (no Fock
    # matrix, no SCF loop); full CI in the Loewdin-orthogonalized atomic orbitals instead of
    # the RHF molecular orbitals.
    S, H = sto3g_one_electron(ZA, ZB, zetaA, zetaB, R)
    eri = sto3g_two_electron(zetaA, zetaB, R)

    def energy(theta):
        v = np.array([np.cos(theta), np.sin(theta)])
        c = v / np.sqrt(v @ S @ v)
        return 2.0 * (c @ H @ c) + np.einsum('i,j,k,l,ijkl->', c, c, c, c, eri)

    # several local minima are possible (one closed-shell branch per atom): refine every local minimum
    # of a fine periodic scan and keep the lowest
    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 1441)[:-1]
    vals = np.array([energy(t) for t in grid])
    step = grid[1] - grid[0]
    best = np.inf
    for i in np.where((vals <= np.roll(vals, 1)) & (vals <= np.roll(vals, -1)))[0]:
        res = minimize_scalar(energy, bounds=(grid[i] - step, grid[i] + step), method='bounded',
                              options={'xatol': 1e-13})
        best = min(best, float(res.fun))
    E_rhf = float(best + ZA * ZB / R)
    s_val, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T          # Loewdin orbitals, still localized
    h = X.T @ H @ X
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', X, X, X, X, eri)
    # spatial two-electron functions symmetric in the electrons: chi_a chi_a, chi_b chi_b and
    # (chi_a chi_b + chi_b chi_a) / sqrt(2); matrix elements from the product-basis Hamiltonian
    # <ij|H|kl> = h_ik d_jl + d_ik h_jl + (ik|jl)
    I2 = np.eye(2)
    Hp = (np.einsum('ik,jl->ijkl', h, I2) + np.einsum('ik,jl->ijkl', I2, h)
          + np.einsum('ikjl->ijkl', g)).reshape(4, 4)
    B = np.zeros((4, 3))
    B[0, 0] = 1.0
    B[3, 1] = 1.0
    B[1, 2] = B[2, 2] = np.sqrt(0.5)
    E_fci = float(np.linalg.eigvalsh(B.T @ Hp @ B)[0] + ZA * ZB / R)
    result = (E_fci, E_rhf)
    return result


def vibrational_levels(ZA, ZB, zetaA, zetaB, massA, massB):
    '''Lowest five vibrational levels (J = 0) on the full-CI potential curve, from the dissociation limit.'''
    # Other methods: the dissociation limit from one-centre integrals evaluated primitive pair
    # by primitive pair, V(R) sampled on Chebyshev nodes in ln R and interpolated, and the
    # radial equation solved by Chebyshev spectral collocation (no DVR, no shooting).
    for m in (massA, massB):
        if not (np.isfinite(m) and 1.0 <= m <= 10.0):
            raise ValueError("nuclear masses must be finite and between 1 and 10 u")
    alpha = np.array([0.109818, 0.405771, 2.22766])
    coef = np.array([0.444635, 0.535328, 0.154329])

    def atom(Z, zeta):
        a = alpha * zeta ** 2
        nrm = coef * (2.0 * a / np.pi) ** 0.75
        s = h = j = 0.0
        for i in range(3):
            for k in range(3):
                p = a[i] + a[k]
                w = nrm[i] * nrm[k]
                s += w * (np.pi / p) ** 1.5
                h += w * (3.0 * a[i] * a[k] / p * (np.pi / p) ** 1.5 - 2.0 * np.pi * Z / p)
                for l in range(3):
                    for n in range(3):
                        q = a[l] + a[n]
                        j += w * nrm[l] * nrm[n] * 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
        return s, h, j

    sA, hA, jA = atom(ZA, zetaA)
    sB, hB, jB = atom(ZB, zetaB)
    # both electrons on A, both on B, or one on each: the lowest is the R -> infinity limit
    e_inf = min((2.0 * hA * sA + jA) / sA ** 2, (2.0 * hB * sB + jB) / sB ** 2, hA / sA + hB / sB)
    r_lo, r_hi = 0.35, 16.0
    # Chebyshev interpolant of V in x = ln R
    n_int = 110
    t = np.cos(np.pi * (np.arange(n_int) + 0.5) / n_int)
    x_lo, x_hi = np.log(r_lo), np.log(r_hi)
    xs = 0.5 * (x_hi + x_lo) + 0.5 * (x_hi - x_lo) * t
    Vs = np.array([fci_energy(ZA, ZB, zetaA, zetaB, np.exp(x))[0] for x in xs]) - e_inf
    cheb = np.polynomial.chebyshev.Chebyshev.fit(t, Vs, n_int - 1)

    def V(r):
        return cheb((2.0 * np.log(r) - x_hi - x_lo) / (x_hi - x_lo))

    # Chebyshev collocation on [r_lo, r_hi] with chi = 0 at both ends
    N = 420
    k = np.arange(N + 1)
    y = np.cos(np.pi * k / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    c = c * (-1.0) ** k
    dy = y[:, None] - y[None, :]
    D = np.outer(c, 1.0 / c) / (dy + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    L = 0.5 * (r_hi - r_lo)
    D2 = (D @ D)[1:-1, 1:-1] / L ** 2
    r = 0.5 * (r_hi + r_lo) + L * y[1:-1]
    mu = massA * massB / (massA + massB) * 1822.888486209
    A = -D2 / (2.0 * mu) + np.diag(V(r))
    E = np.sort(np.linalg.eigvals(A).real)
    bound = E[E < 0.0]
    if bound.size < 5:
        raise ValueError("fewer than five vibrational levels lie below the dissociation limit")
    levels = bound[:5] * 219474.6313632
    return levels


from functools import lru_cache

# Second solution, step 6: Obara-Saika recursions instead of McMurchie-Davidson Hermite
# expansions; kinetic energy as (1/2) sum_i <d_i a | d_i b>.
_T_AL = np.array([0.109818, 0.405771, 2.22766])
_T_DC = np.array([0.444635, 0.535328, 0.154329])


def _g6_F(m, T):
    return float(boys_function(m, T)[m])


def _g6_basis(zA, zB, aA, aB, R):
    out = []
    for cen, z, ap in ((np.zeros(3), zA, aA), (np.array([0.0, 0.0, R]), zB, aB)):
        e = _T_AL * z * z
        out.append([(ee, cc, (0, 0, 0), cen) for ee, cc in zip(e, _T_DC * (2 * e / np.pi) ** 0.75)])
        for ang in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            out.append([(ap, (2 * ap / np.pi) ** 0.75 * 2 * np.sqrt(ap), ang, cen)])
    return out


def _g6_ovl1d(a, la, Ax, b, lb, Bx):
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


def _g6_S(a, la, A, b, lb, B):
    return np.prod([_g6_ovl1d(a, la[k], A[k], b, lb[k], B[k]) for k in range(3)])


def _g6_T(a, la, A, b, lb, B):
    # (1/2) sum_i <d_i a | d_i b>, d/dx x^l e^{-a x^2} = l x^(l-1) - 2 a x^(l+1)
    tot = 0.0
    for k in range(3):
        da = [(la[k], -2 * a, 1)] + ([(la[k], la[k], -1)] if la[k] > 0 else [])
        db = [(lb[k], -2 * b, 1)] + ([(lb[k], lb[k], -1)] if lb[k] > 0 else [])
        for _, ca, sa in da:
            for _, cb, sb in db:
                l1 = list(la); l1[k] += sa
                l2 = list(lb); l2[k] += sb
                tot += 0.5 * ca * cb * _g6_S(a, l1, A, b, l2, B)
    return tot


def _g6_V(a, la, A, b, lb, B, C):
    p = a + b
    P = (a * A + b * B) / p
    K = np.exp(-a * b / p * np.sum((A - B) ** 2))
    T = p * np.sum((P - C) ** 2)

    @lru_cache(None)
    def V(l1, l2, m):
        if min(l1) < 0 or min(l2) < 0:
            return 0.0
        if sum(l1) == 0 and sum(l2) == 0:
            return 2 * np.pi / p * K * _g6_F(m, T)
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


def _g6_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D):
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
            return pre * _g6_F(m, T)
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


def _g6_integrals(ZA, ZB, zA, zB, aA, aB, R):
    bs = _g6_basis(zA, zB, aA, aB, R)
    n = len(bs)
    nuc = ((ZA, np.zeros(3)), (ZB, np.array([0.0, 0.0, R])))
    S = np.zeros((n, n)); H = np.zeros((n, n)); G = np.zeros((n,) * 4)
    for i in range(n):
        for j in range(n):
            for (a, ca, la, A) in bs[i]:
                for (b, cb, lb, B) in bs[j]:
                    S[i, j] += ca * cb * _g6_S(a, la, A, b, lb, B)
                    H[i, j] += ca * cb * (_g6_T(a, la, A, b, lb, B) - sum(Z * _g6_V(a, la, A, b, lb, B, C) for Z, C in nuc))
    done = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in done:
                        done[key] = sum(ca * cb * cc * cd * _g6_ERI(a, la, A, b, lb, B, c, lc, C, d, ld, D)
                                        for (a, ca, la, A) in bs[i] for (b, cb, lb, B) in bs[j]
                                        for (c, cc, lc, C) in bs[k] for (d, cd, ld, D) in bs[l])
                    G[i, j, k, l] = done[key]
    return S, H, G


def polarized_energies(ZA, ZB, zetaA, zetaB, alphaA, alphaB, R):
    '''Full-CI and RHF energies of a two-electron diatomic in the STO-3G 1s + p-shell basis.'''
    # Other routes: full CI in the singlet configuration functions (i <= j) of Loewdin
    # orbitals; RHF by steepest descent along great circles with an exact line search.
    S, H, G = _g6_integrals(ZA, ZB, zetaA, zetaB, alphaA, alphaB, R)
    n = S.shape[0]
    w, U = np.linalg.eigh(S)
    X = U @ np.diag(w ** -0.5) @ U.T
    h = X.T @ H @ X
    g = np.einsum("pi,qj,rk,sl,pqrs->ijkl", X, X, X, X, G)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]

    def norm(i, j):
        return 0.5 if i == j else np.sqrt(0.5)

    M = np.zeros((len(pairs), len(pairs)))
    I = np.eye(n)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            # singlet |ij> = N (chi_i chi_j + chi_j chi_i); matrix element from the product basis
            val = 0.0
            for (p, q) in ((i, j), (j, i)):
                for (r, s) in ((k, l), (l, k)):
                    val += h[p, r] * I[q, s] + I[p, r] * h[q, s] + g[p, r, q, s]
            M[a, b] = norm(i, j) * norm(k, l) * val
    E_fci = float(np.linalg.eigvalsh(M)[0]) + ZA * ZB / R

    def energy(x):
        return float(2.0 * (x @ h @ x) + np.einsum("i,j,k,l,ijkl->", x, x, x, x, g))

    def grad(x):
        r = 4.0 * (h + np.einsum("k,l,ijkl->ij", x, x, g)) @ x
        return r - (r @ x) * x

    rng = np.random.default_rng(77)
    best = np.inf
    for x in list(np.linalg.eigh(h)[1].T) + list(rng.normal(size=(40, n))):
        x = x / np.linalg.norm(x)
        for _ in range(3000):
            d = -grad(x)
            nd = np.linalg.norm(d)
            if nd < 1e-10:
                break
            d /= nd
            # great circle x cos(s) + d sin(s); exact line search with Brent's method
            res = minimize_scalar(lambda s_: energy(np.cos(s_) * x + np.sin(s_) * d), bounds=(0.0, 0.5),
                                  method="bounded", options={"xatol": 1e-14})
            y = np.cos(res.x) * x + np.sin(res.x) * d
            if energy(y) >= energy(x):
                break
            x = y / np.linalg.norm(y)
        best = min(best, energy(x))
    E_rhf = best + ZA * ZB / R
    result = (float(E_fci), float(E_rhf))
    return result
