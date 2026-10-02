import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def boys_f0(t):
    '''Boys function of order zero, F0(t) = integral from 0 to 1 of exp(-t u**2) du.

    Input:
      t: float or numpy array of floats (any shape), t >= 0 (t = 0 is allowed).

    Output:
      F: float numpy array with the same shape as np.asarray(t); F0(0) = 1.
         Relative error below 1e-11 for every t >= 0.

    Raises:
      ValueError if any t is negative or not finite.
    '''
    # Other method: adaptive numerical quadrature of the defining integral.
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0):
        raise ValueError("t must be finite and >= 0")
    flat = t_arr.ravel()
    out = np.empty_like(flat)
    for i, tv in enumerate(flat):
        upper = 1.0 if tv <= 100.0 else 10.0 / np.sqrt(tv)
        out[i] = quad(lambda u: np.exp(-tv * u * u), 0.0, upper,
                      epsabs=1e-16, epsrel=1e-13, limit=200)[0]
    F = out.reshape(t_arr.shape)
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
                inner = np.sum(t_wts[None, :] * t_nodes[None, :] ** 2
                               * np.exp(-p[:, None] * (e_dist[:, None] * t_nodes[None, :]) ** 2), axis=1) * e_dist ** 2
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

    grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 721)
    vals = np.array([energy(t) for t in grid])
    i0 = int(np.argmin(vals))
    lo = grid[max(i0 - 1, 0)]
    hi = grid[min(i0 + 1, len(grid) - 1)]
    res = minimize_scalar(energy, bounds=(lo, hi), method='bounded', options={'xatol': 1e-12})
    E_rhf = float(res.fun + ZA * ZB / R)
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
        if not (np.isfinite(m) and m > 0.0):
            raise ValueError("nuclear masses must be positive and finite")
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
