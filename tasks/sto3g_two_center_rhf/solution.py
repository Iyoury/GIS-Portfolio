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
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0):
        raise ValueError("t must be finite and >= 0")
    F = np.ones_like(t_arr)
    small = t_arr < 1e-10
    big = ~small
    root = np.sqrt(t_arr[big])
    F[big] = 0.5 * np.sqrt(np.pi) * erf(root) / root
    F[small] = 1.0 - t_arr[small] / 3.0
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
    expo = [alpha_one * zetaA ** 2, alpha_one * zetaB ** 2]
    S = np.zeros((2, 2))
    H = np.zeros((2, 2))
    for m in range(2):
        for n in range(2):
            a = expo[m][:, None]
            b = expo[n][None, :]
            p = a + b
            mu = a * b / p
            dist2 = (centers[m] - centers[n]) ** 2
            norm = (2.0 * a / np.pi) ** 0.75 * (2.0 * b / np.pi) ** 0.75
            weight = coef[:, None] * coef[None, :] * norm * np.exp(-mu * dist2)
            center_p = (a * centers[m] + b * centers[n]) / p
            overlap = (np.pi / p) ** 1.5
            kinetic = mu * (3.0 - 2.0 * mu * dist2) * (np.pi / p) ** 1.5
            attract = np.zeros_like(p)
            for C in range(2):
                attract -= 2.0 * np.pi / p * charges[C] * boys_f0(p * (center_p - centers[C]) ** 2)
            S[m, n] = np.sum(weight * overlap)
            H[m, n] = np.sum(weight * (kinetic + attract))
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
    expo = [alpha_one * zetaA ** 2, alpha_one * zetaB ** 2]
    w = (coef[:, None, None, None] * coef[None, :, None, None]
         * coef[None, None, :, None] * coef[None, None, None, :])
    eri = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    a = expo[i][:, None, None, None]
                    b = expo[j][None, :, None, None]
                    c = expo[k][None, None, :, None]
                    d = expo[l][None, None, None, :]
                    p = a + b
                    q = c + d
                    center_p = (a * centers[i] + b * centers[j]) / p
                    center_q = (c * centers[k] + d * centers[l]) / q
                    norm = (2.0 * a / np.pi * 2.0 * b / np.pi * 2.0 * c / np.pi * 2.0 * d / np.pi) ** 0.75
                    gauss = np.exp(-a * b / p * (centers[i] - centers[j]) ** 2
                                   - c * d / q * (centers[k] - centers[l]) ** 2)
                    pre = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
                    val = pre * gauss * boys_f0(p * q / (p + q) * (center_p - center_q) ** 2)
                    eri[i, j, k, l] = np.sum(w * norm * val)
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
    # RHF: Roothaan-Hall self-consistent field in the symmetric orthogonalization
    s_val, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T
    P = np.zeros((2, 2))
    for _ in range(500):
        F = H + np.einsum('ls,mnsl->mn', P, eri) - 0.5 * np.einsum('ls,mlsn->mn', P, eri)
        _, C_ortho = np.linalg.eigh(X.T @ F @ X)
        C = X @ C_ortho
        P_new = 2.0 * np.outer(C[:, 0], C[:, 0])
        change = np.max(np.abs(P_new - P))
        P = P_new
        if change < 1e-14:
            break
    F = H + np.einsum('ls,mnsl->mn', P, eri) - 0.5 * np.einsum('ls,mlsn->mn', P, eri)
    E_rhf = float(0.5 * np.sum(P * (H + F)) + enuc)
    # Full CI: singlet configuration state functions in the orthonormal RHF orbitals
    # (1 = occupied, 2 = virtual): |1 1|, |2 2| and the open-shell singlet (1 2).
    h = C.T @ H @ C
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C, eri)
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
      massA, massB: float, nuclear masses in unified atomic mass units (u), > 0.

    Output:
      levels: float numpy array of shape (5,), the energies of the vibrational states
              v = 0..4 in cm^-1, measured from the exact R -> infinity limit of the
              full-CI energy (so all are negative), increasing. Absolute error below
              0.01 cm^-1 for each level.

    Raises:
      ValueError if a mass is not positive and finite, or if fewer than five bound
      vibrational levels lie below the dissociation limit.
    '''
    for m in (massA, massB):
        if not (np.isfinite(m) and m > 0.0):
            raise ValueError("nuclear masses must be positive and finite")
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
    # grid; the five lowest states are negligible outside [0.35, 16] bohr.
    mu = massA * massB / (massA + massB) * 1822.888486209
    h = 0.02
    R = np.arange(0.35, 16.0 + 0.5 * h, h)
    V = np.array([fci_energy(ZA, ZB, zetaA, zetaB, r)[0] for r in R]) - e_inf
    n = np.arange(R.size)
    diff = n[:, None] - n[None, :]
    with np.errstate(divide="ignore"):
        T = np.where(diff == 0, np.pi ** 2 / 3.0, 2.0 * (-1.0) ** diff / np.where(diff == 0, 1, diff) ** 2)
    T = T / (2.0 * mu * h * h)
    E = np.linalg.eigvalsh(T + np.diag(V))
    bound = E[E < 0.0]
    if bound.size < 5:
        raise ValueError("fewer than five vibrational levels lie below the dissociation limit")
    levels = bound[:5] * 219474.6313632
    return levels
