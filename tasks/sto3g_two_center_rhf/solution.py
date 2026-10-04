import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def boys_function(n_max, t):
    '''Boys functions F_n(t) = integral from 0 to 1 of u**(2n) exp(-t u**2) du for n = 0..n_max.

    Inputs:
      n_max: int, highest order, 0 <= n_max <= 16.
      t: float or numpy array of floats (any shape), 0 <= t <= 1e6 (t = 0 is allowed).

    Output:
      F: float numpy array of shape np.shape(t) + (n_max + 1,), F[..., n] = F_n(t).
         Relative error below 1e-11 for every entry; F_n(0) = 1 / (2n + 1).

    Raises:
      ValueError if n_max is not an integer in [0, 16], or if any t is negative, not finite
      or larger than 1e6.
    '''
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)) or not (0 <= n_max <= 16):
        raise ValueError("n_max must be an integer from 0 to 16")
    n_max = int(n_max)
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0) or np.any(t_arr > 1e6):
        raise ValueError("t must be finite with 0 <= t <= 1e6")
    flat = t_arr.ravel()
    F = np.empty((flat.size, n_max + 1))
    split = n_max + 25.0
    small = flat < split
    # small t: series for the highest order, F_n(t) = exp(-t) sum_k (2t)^k / ((2n+1)(2n+3)...(2n+2k+1)),
    # then the downward recursion F_(m-1) = (2 t F_m + exp(-t)) / (2m - 1), which is stable
    ts = flat[small]
    if ts.size:
        term = 1.0 / (2 * n_max + 1) * np.ones_like(ts)
        total = term.copy()
        for k in range(1, 400):
            term = term * 2.0 * ts / (2 * n_max + 2 * k + 1)
            total += term
            if np.all(term <= 1e-17 * total):
                break
        e = np.exp(-ts)
        F[small, n_max] = e * total
        for m in range(n_max, 0, -1):
            F[small, m - 1] = (2.0 * ts * F[small, m] + e) / (2 * m - 1)
    # large t: F_0 from erf, then the upward recursion F_(m+1) = ((2m+1) F_m - exp(-t)) / (2t),
    # stable because t > n_max + 25
    tl = flat[~small]
    if tl.size:
        r = np.sqrt(tl)
        F[~small, 0] = 0.5 * np.sqrt(np.pi) * erf(r) / r
        e = np.exp(-tl)
        for m in range(n_max):
            F[~small, m + 1] = ((2 * m + 1) * F[~small, m] - e) / (2.0 * tl)
    F = F.reshape(t_arr.shape + (n_max + 1,))
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
    if not (R > 0 and ZA > 0 and ZB > 0 and zetaA > 0 and zetaB > 0):
        raise ValueError("R, the nuclear charges and the Slater exponents must be positive")
    alpha_one = np.array([0.109818, 0.405771, 2.22766])
    coef = np.array([0.444635, 0.535328, 0.154329])
    centers = np.array([0.0, float(R)])
    charges = np.array([float(ZA), float(ZB)])
    expo = np.array([alpha_one * zetaA ** 2, alpha_one * zetaB ** 2])          # (function, primitive)
    # all primitive pairs at once, axes (m, n, i, j): primitive i of function m, j of function n
    a = expo[:, None, :, None]
    b = expo[None, :, None, :]
    xa = centers[:, None, None, None]
    xb = centers[None, :, None, None]
    p = a + b
    mu = a * b / p
    dist2 = (xa - xb) ** 2
    norm = (2.0 * a / np.pi) ** 0.75 * (2.0 * b / np.pi) ** 0.75
    weight = coef[None, None, :, None] * coef[None, None, None, :] * norm * np.exp(-mu * dist2)
    center_p = (a * xa + b * xb) / p
    overlap = (np.pi / p) ** 1.5
    kinetic = mu * (3.0 - 2.0 * mu * dist2) * (np.pi / p) ** 1.5
    # attraction to both nuclei, one Boys evaluation for every primitive pair and nucleus
    F0 = boys_function(0, p[..., None] * (center_p[..., None] - centers) ** 2)[..., 0]
    attract = -2.0 * np.pi / p * np.sum(charges * F0, axis=-1)
    S = np.sum(weight * overlap, axis=(2, 3))
    H = np.sum(weight * (kinetic + attract), axis=(2, 3))
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
    if not (R > 0 and zetaA > 0 and zetaB > 0):
        raise ValueError("R and the Slater exponents must be positive")
    alpha_one = np.array([0.109818, 0.405771, 2.22766])
    coef = np.array([0.444635, 0.535328, 0.154329])
    centers = np.array([0.0, float(R)])
    expo = np.array([alpha_one * zetaA ** 2, alpha_one * zetaB ** 2])          # (function, primitive)
    # axes (i, j, k, l, pi, pj, pk, pl): functions i..l and their primitives, all at once
    a = expo[:, None, None, None, :, None, None, None]
    b = expo[None, :, None, None, None, :, None, None]
    c = expo[None, None, :, None, None, None, :, None]
    d = expo[None, None, None, :, None, None, None, :]
    xa = centers[:, None, None, None, None, None, None, None]
    xb = centers[None, :, None, None, None, None, None, None]
    xc = centers[None, None, :, None, None, None, None, None]
    xd = centers[None, None, None, :, None, None, None, None]
    w = (coef[:, None, None, None] * coef[None, :, None, None]
         * coef[None, None, :, None] * coef[None, None, None, :])
    p = a + b
    q = c + d
    center_p = (a * xa + b * xb) / p
    center_q = (c * xc + d * xd) / q
    norm = (2.0 * a / np.pi * 2.0 * b / np.pi * 2.0 * c / np.pi * 2.0 * d / np.pi) ** 0.75
    gauss = np.exp(-a * b / p * (xa - xb) ** 2 - c * d / q * (xc - xd) ** 2)
    pre = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    val = pre * gauss * boys_function(0, p * q / (p + q) * (center_p - center_q) ** 2)[..., 0]
    eri = np.sum(w * norm * val, axis=(4, 5, 6, 7))
    return eri


def fci_energy(ZA, ZB, zetaA, zetaB, R):
    '''Full-CI and closed-shell RHF ground-state energies of a two-electron diatomic in STO-3G.

    Inputs:
      ZA, ZB: float, nuclear charges (for example 1, 1 for H2 and 2, 1 for HeH+), > 0.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, > 0.
      R: float, distance between the nuclei in bohr, R > 0.

    Output:
      (E_fci, E_rhf): tuple of two Python floats, total energies in hartree (electronic
      energy plus the nuclear repulsion ZA * ZB / R), absolute errors below 1e-9.
        E_fci: lowest eigenvalue of the electronic Hamiltonian in the space of all
               two-electron states built from the two basis functions (full CI).
        E_rhf: closed-shell restricted Hartree-Fock ground-state energy.
    '''
    S, H = sto3g_one_electron(ZA, ZB, zetaA, zetaB, R)
    eri = sto3g_two_electron(zetaA, zetaB, R)
    enuc = ZA * ZB / R
    # RHF: with two basis functions every normalized closed-shell orbital is c(theta) = X (cos theta,
    # sin theta) in the symmetric orthogonalization X = S^(-1/2), theta in [0, pi). The energy
    # E(theta) = 2 c.H.c + (cc|cc) can have several local minima (an asymmetric or stretched molecule has
    # one closed-shell branch on each atom), and a self-consistent-field iteration started from one guess
    # can settle on the higher one. The global minimum is found from a scan of the whole period, every
    # local minimum of the scan being refined, and the lowest refined value kept.
    s_val, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T

    def orbital(theta):
        return X @ np.array([np.cos(theta), np.sin(theta)])

    def energy(theta):
        c = orbital(theta)
        return 2.0 * (c @ H @ c) + np.einsum('i,j,k,l,ijkl->', c, c, c, c, eri)

    grid = np.linspace(0.0, np.pi, 721)[:-1]
    vals = np.array([energy(t) for t in grid])
    step = grid[1] - grid[0]
    best_e, best_t = np.inf, 0.0
    for i in np.where((vals <= np.roll(vals, 1)) & (vals <= np.roll(vals, -1)))[0]:
        res = minimize_scalar(energy, bounds=(grid[i] - step, grid[i] + step), method='bounded',
                              options={'xatol': 1e-13})
        if res.fun < best_e:
            best_e, best_t = float(res.fun), float(res.x)
    E_rhf = float(best_e + enuc)
    C = np.column_stack([orbital(best_t), orbital(best_t + 0.5 * np.pi)])
    # Full CI: singlet configuration state functions in the orthonormal RHF orbitals
    # (1 = occupied, 2 = virtual): |1 1|, |2 2| and the open-shell singlet (1 2).
    h = C.T @ H @ C
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C, eri, optimize=True)
    M = np.empty((3, 3))
    M[0, 0] = 2.0 * h[0, 0] + g[0, 0, 0, 0]
    M[1, 1] = 2.0 * h[1, 1] + g[1, 1, 1, 1]
    M[2, 2] = h[0, 0] + h[1, 1] + g[0, 0, 1, 1] + g[0, 1, 0, 1]
    M[0, 1] = M[1, 0] = g[0, 1, 0, 1]
    M[0, 2] = M[2, 0] = np.sqrt(2.0) * (h[0, 1] + g[0, 0, 0, 1])
    M[1, 2] = M[2, 1] = np.sqrt(2.0) * (h[0, 1] + g[1, 1, 0, 1])
    E_fci = float(np.linalg.eigvalsh(M)[0] + enuc)
    result = (E_fci, E_rhf)
    return result


def _s5_atom(Z, zeta):
    # one-centre integrals of the (unrenormalized) STO-3G 1s function on an isolated nucleus Z:
    # norm <phi|phi>, core energy <phi|-1/2 Laplacian - Z/r|phi> and (phi phi|phi phi)
    a = np.array([0.109818, 0.405771, 2.22766]) * zeta ** 2
    d = np.array([0.444635, 0.535328, 0.154329]) * (2.0 * a / np.pi) ** 0.75
    p = a[:, None] + a[None, :]
    dd = d[:, None] * d[None, :]
    s = np.sum(dd * (np.pi / p) ** 1.5)
    h = np.sum(dd * (3.0 * a[:, None] * a[None, :] / p * (np.pi / p) ** 1.5 - 2.0 * np.pi * Z / p))
    P = p[:, :, None, None]
    Q = p[None, None, :, :]
    j = np.sum(dd[:, :, None, None] * dd[None, None, :, :] * 2.0 * np.pi ** 2.5 / (P * Q * np.sqrt(P + Q)))
    return s, h, j


def vibrational_levels(ZA, ZB, zetaA, zetaB, massA, massB):
    '''Lowest five vibrational levels (J = 0) on the full-CI potential curve, from the dissociation limit.

    Inputs:
      ZA, ZB: float, nuclear charges, 1 <= Z <= 3.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, 0.8 <= zeta <= 3.
      massA, massB: float, nuclear masses in unified atomic mass units (u), 1 <= mass <= 10.

    Output:
      levels: float numpy array of shape (5,), the energies of the vibrational states
              v = 0..4 in cm^-1, measured from the exact R -> infinity limit of the
              full-CI energy (so all are negative), increasing. Absolute error below
              0.01 cm^-1 for each level.

    Raises:
      ValueError if a mass is not finite or not in [1, 10] u, or if fewer than five
      bound vibrational levels lie below the dissociation limit.
    '''
    for m in (massA, massB):
        if not (np.isfinite(m) and 1.0 <= m <= 10.0):
            raise ValueError("nuclear masses must be finite and between 1 and 10 u")
    # Dissociation limit in this basis: as R -> infinity all cross integrals vanish and the
    # lowest singlet is the lowest of the three fragment arrangements (both electrons on A,
    # both on B, one on each). The leftover Coulomb energy of the fragment charges
    # q_A q_B / R vanishes only in the limit, so the limit is not E(R) at any finite R.
    sA, hA, jA = _s5_atom(ZA, zetaA)
    sB, hB, jB = _s5_atom(ZB, zetaB)
    e_inf = min((2.0 * hA * sA + jA) / sA ** 2,
                (2.0 * hB * sB + jB) / sB ** 2,
                hA / sA + hB / sB)
    # Radial nuclear equation -(1/2 mu) chi'' + V chi = E chi, V(R) = E_fci(R) - e_inf,
    # solved with the sinc discrete-variable representation (Colbert-Miller) on a uniform
    # grid R = k h. The domain is widened (inner wall moved in, outer wall moved out) until
    # the five lowest levels no longer change, so diffuse states are not cut off.
    mu = massA * massB / (massA + massB) * 1822.888486209
    h = 0.03
    cache = {}

    def V(k):
        if k not in cache:
            cache[k] = fci_energy(ZA, ZB, zetaA, zetaB, k * h)[0] - e_inf
        return cache[k]

    def solve(k_lo, k_hi):
        n = np.arange(k_hi - k_lo + 1)
        diff = n[:, None] - n[None, :]
        T = np.where(diff == 0, np.pi ** 2 / 3.0, 2.0 * (-1.0) ** diff / np.where(diff == 0, 1, diff) ** 2)
        T = T / (2.0 * mu * h * h)
        Vd = np.array([V(k) for k in range(k_lo, k_hi + 1)])
        if Vd.min() >= 0.0:
            return Vd, np.array([])           # no state can lie below the limit
        E = np.linalg.eigvalsh(T + np.diag(Vd))
        return Vd, E[E < 0.0]

    domains = [(12, 533), (9, 667), (6, 900), (4, 1200)]     # [0.36, 16], [0.27, 20], [0.18, 27], [0.12, 36]
    bound = None
    for k_lo, k_hi in domains:
        Vd, cur = solve(k_lo, k_hi)
        if Vd.min() >= 0.0:
            break
        if (bound is not None and bound.size >= 5 and cur.size >= 5
                and np.max(np.abs(cur[:5] - bound[:5])) * 219474.6313632 < 1e-4):
            bound = cur
            break
        bound = cur
    if bound is None or bound.size < 5:
        raise ValueError("fewer than five vibrational levels lie below the dissociation limit")
    levels = bound[:5] * 219474.6313632
    return levels


def _s6_E(i, j, t, X, a, b):
    # McMurchie-Davidson Hermite expansion coefficients E_t^{ij} for one Cartesian direction
    p = a + b
    q = a * b / p
    if t < 0 or t > i + j:
        return 0.0
    if i == 0 and j == 0 and t == 0:
        return np.exp(-q * X * X)
    if j == 0:
        return (_s6_E(i - 1, j, t - 1, X, a, b) / (2 * p) - q * X / a * _s6_E(i - 1, j, t, X, a, b)
                + (t + 1) * _s6_E(i - 1, j, t + 1, X, a, b))
    return (_s6_E(i, j - 1, t - 1, X, a, b) / (2 * p) + q * X / b * _s6_E(i, j - 1, t, X, a, b)
            + (t + 1) * _s6_E(i, j - 1, t + 1, X, a, b))


def _s6_R(t, u, v, n, p, PC, F):
    # Hermite Coulomb integrals R_tuv^n, F[n] = F_n(p |PC|^2)
    if t < 0 or u < 0 or v < 0:
        return 0.0
    if t == 0 and u == 0 and v == 0:
        return (-2.0 * p) ** n * F[n]
    if t > 0:
        return (t - 1) * _s6_R(t - 2, u, v, n + 1, p, PC, F) + PC[0] * _s6_R(t - 1, u, v, n + 1, p, PC, F)
    if u > 0:
        return (u - 1) * _s6_R(t, u - 2, v, n + 1, p, PC, F) + PC[1] * _s6_R(t, u - 1, v, n + 1, p, PC, F)
    return (v - 1) * _s6_R(t, u, v - 2, n + 1, p, PC, F) + PC[2] * _s6_R(t, u, v - 1, n + 1, p, PC, F)


def _s6_overlap(a, la, A, b, lb, B):
    p = a + b
    return np.prod([_s6_E(la[k], lb[k], 0, A[k] - B[k], a, b) for k in range(3)]) * (np.pi / p) ** 1.5


def _s6_kinetic(a, la, A, b, lb, B):
    lb = list(lb)
    val = b * (2 * sum(lb) + 3) * _s6_overlap(a, la, A, b, lb, B)
    for k in range(3):
        up = list(lb)
        up[k] += 2
        val -= 2.0 * b * b * _s6_overlap(a, la, A, b, up, B)
        if lb[k] >= 2:
            dn = list(lb)
            dn[k] -= 2
            val -= 0.5 * lb[k] * (lb[k] - 1) * _s6_overlap(a, la, A, b, dn, B)
    return val


def _s6_attraction(a, la, A, b, lb, B, C):
    p = a + b
    P = (a * A + b * B) / p
    PC = P - C
    F = boys_function(sum(la) + sum(lb), p * float(PC @ PC))
    val = 0.0
    for t in range(la[0] + lb[0] + 1):
        for u in range(la[1] + lb[1] + 1):
            for v in range(la[2] + lb[2] + 1):
                val += (_s6_E(la[0], lb[0], t, A[0] - B[0], a, b) * _s6_E(la[1], lb[1], u, A[1] - B[1], a, b)
                        * _s6_E(la[2], lb[2], v, A[2] - B[2], a, b) * _s6_R(t, u, v, 0, p, PC, F))
    return 2.0 * np.pi / p * val


def _s6_repulsion(a, la, A, b, lb, B, c, lc, C, d, ld, D):
    p, q = a + b, c + d
    al = p * q / (p + q)
    P, Q = (a * A + b * B) / p, (c * C + d * D) / q
    PQ = P - Q
    F = boys_function(sum(la) + sum(lb) + sum(lc) + sum(ld), al * float(PQ @ PQ))
    E1 = [[_s6_E(la[k], lb[k], t, A[k] - B[k], a, b) for t in range(la[k] + lb[k] + 1)] for k in range(3)]
    E2 = [[_s6_E(lc[k], ld[k], t, C[k] - D[k], c, d) for t in range(lc[k] + ld[k] + 1)] for k in range(3)]
    val = 0.0
    for t, ex in enumerate(E1[0]):
        for u, ey in enumerate(E1[1]):
            for v, ez in enumerate(E1[2]):
                for tau, fx in enumerate(E2[0]):
                    for nu, fy in enumerate(E2[1]):
                        for phi, fz in enumerate(E2[2]):
                            val += (ex * ey * ez * fx * fy * fz * (-1) ** (tau + nu + phi)
                                    * _s6_R(t + tau, u + nu, v + phi, 0, al, PQ, F))
    return 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q)) * val


def polarized_energies(ZA, ZB, zetaA, zetaB, alphaA, alphaB, R):
    '''Full-CI and RHF energies of a two-electron diatomic in the STO-3G 1s + p-shell basis.

    Inputs:
      ZA, ZB: float, nuclear charges, 1 <= Z <= 3 (A at the origin, B at (0, 0, R)).
      zetaA, zetaB: float, Slater exponents of the STO-3G 1s functions, 0.5 <= zeta <= 3.
      alphaA, alphaB: float, exponents of the p shells on A and B, 0.1 <= alpha <= 5.
      R: float, distance between the nuclei in bohr, 0.3 <= R <= 10.

    Output:
      (E_fci, E_rhf): tuple of two Python floats, total energies in hartree (electronic energy
      plus ZA * ZB / R) of the singlet ground state in the 8-function basis, full CI and
      closed-shell RHF, absolute errors below 1e-9.
    '''
    # Basis: on each atom the STO-3G 1s function of steps 2-3 and a normalized primitive
    # Cartesian p shell (x - X) exp(-alpha |r - X|^2), (y - Y) ..., (z - Z) ...
    alpha_1s = np.array([0.109818, 0.405771, 2.22766])
    d_1s = np.array([0.444635, 0.535328, 0.154329])
    basis = []
    for cen, z, ap in ((np.zeros(3), zetaA, alphaA), (np.array([0.0, 0.0, float(R)]), zetaB, alphaB)):
        e = alpha_1s * z * z
        basis.append([(ee, cc, (0, 0, 0), cen) for ee, cc in zip(e, d_1s * (2 * e / np.pi) ** 0.75)])
        for ang in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            basis.append([(ap, (2 * ap / np.pi) ** 0.75 * 2.0 * np.sqrt(ap), ang, cen)])
    nb = len(basis)
    nuclei = ((float(ZA), np.zeros(3)), (float(ZB), np.array([0.0, 0.0, float(R)])))
    S = np.zeros((nb, nb))
    H = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(i + 1):
            s = h = 0.0
            for (a, ca, la, A) in basis[i]:
                for (b, cb, lb, B) in basis[j]:
                    s += ca * cb * _s6_overlap(a, la, A, b, lb, B)
                    h += ca * cb * _s6_kinetic(a, la, A, b, lb, B)
                    for Z, C in nuclei:
                        h -= Z * ca * cb * _s6_attraction(a, la, A, b, lb, B, C)
            S[i, j] = S[j, i] = s
            H[i, j] = H[j, i] = h
    G = np.zeros((nb,) * 4)
    for i in range(nb):
        for j in range(i + 1):
            for k in range(nb):
                for l in range(k + 1):
                    if i * (i + 1) // 2 + j < k * (k + 1) // 2 + l:
                        continue
                    val = 0.0
                    for (a, ca, la, A) in basis[i]:
                        for (b, cb, lb, B) in basis[j]:
                            for (c, cc, lc, C) in basis[k]:
                                for (d, cd, ld, D) in basis[l]:
                                    val += ca * cb * cc * cd * _s6_repulsion(a, la, A, b, lb, B, c, lc, C, d, ld, D)
                    for w, x, y, zz in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                                        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                        G[w, x, y, zz] = val
    enuc = ZA * ZB / R
    s_val, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T
    # RHF: the closed-shell determinant of lowest total energy, i.e. the global minimum of
    # E(x) = 2 x.h.x + (xx|xx) over normalized orbitals x in the orthonormal basis. The orbital
    # with the lowest Fock eigenvalue (aufbau) is not always the one of lowest total energy here:
    # a diffuse p shell can give a lower orbital energy but a higher total energy. So E is
    # minimized directly on the unit sphere (projected gradient with backtracking) from many
    # starting orbitals, and the best minima are polished by a maximum-overlap SCF.
    h_o = X.T @ H @ X
    g_o = np.einsum('pi,qj,rk,sl,pqrs->ijkl', X, X, X, X, G)

    def e_of(x):
        return float(2.0 * (x @ h_o @ x) + np.einsum('i,j,k,l,ijkl->', x, x, x, x, g_o))

    def fock(x):
        return h_o + np.einsum('k,l,ijkl->ij', x, x, g_o)     # closed shell: J - K/2 with P = 2 x x

    rng = np.random.default_rng(12345)
    starts = list(np.linalg.eigh(h_o)[1].T) + list(rng.normal(size=(40, nb)))
    found = []
    for x in starts:
        x = x / np.linalg.norm(x)
        e = e_of(x)
        step = 0.1
        for _ in range(400):
            grad = 4.0 * fock(x) @ x
            grad -= (grad @ x) * x
            if np.linalg.norm(grad) < 1e-9:
                break
            while step > 1e-12:
                y = x - step * grad
                y /= np.linalg.norm(y)
                ey = e_of(y)
                if ey < e - 1e-4 * step * (grad @ grad):
                    x, e = y, ey
                    step *= 2.0
                    break
                step *= 0.5
        found.append((e, x))
    found.sort(key=lambda z: z[0])
    E_rhf = np.inf
    for e, x in found[:4]:
        for _ in range(500):            # maximum-overlap SCF: follow the eigenvector closest to x
            w, V = np.linalg.eigh(fock(x))
            j = int(np.argmax(np.abs(V.T @ x)))
            y = V[:, j] * np.sign(V[:, j] @ x)
            if np.linalg.norm(y - x) < 1e-13:
                x = y
                break
            x = y
        E_rhf = min(E_rhf, e, e_of(x))
    E_rhf += enuc
    # Full CI: two-electron Hamiltonian in the orthonormal product basis chi_i(1) chi_j(2)
    h = X.T @ H @ X
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', X, X, X, X, G)
    I = np.eye(nb)
    Hp = (np.einsum('ik,jl->ijkl', h, I) + np.einsum('ik,jl->ijkl', I, h)
          + np.einsum('ikjl->ijkl', g)).reshape(nb * nb, nb * nb)
    # restrict to spatial functions symmetric under exchange of the electrons (the singlet):
    # in a finite basis the lowest antisymmetric (triplet) state can lie slightly lower
    pairs = [(i, j) for i in range(nb) for j in range(i, nb)]
    B = np.zeros((nb * nb, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        if i == j:
            B[i * nb + i, col] = 1.0
        else:
            B[i * nb + j, col] = B[j * nb + i, col] = np.sqrt(0.5)
    E_fci = float(np.linalg.eigvalsh(B.T @ Hp @ B)[0]) + enuc
    result = (float(E_fci), float(E_rhf))
    return result
