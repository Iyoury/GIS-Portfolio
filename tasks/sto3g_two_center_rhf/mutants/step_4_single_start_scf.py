import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.integrate import solve_ivp


def fci_energy(ZA, ZB, zetaA, zetaB, R):
    '''Singlet ground-state full-CI and closed-shell RHF energies of a two-electron diatomic in STO-3G.

    Inputs:
      ZA, ZB: float, nuclear charges (for example 1, 1 for H2 and 2, 1 for HeH+), 1 <= Z <= 3.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, 0.5 <= zeta <= 3.
      R: float, distance between the nuclei in bohr, 0.1 <= R <= 100.

    Output:
      (E_fci, E_rhf): tuple of two Python floats, total energies in hartree (electronic
      energy plus the nuclear repulsion ZA * ZB / R), absolute errors below 1e-9.
        E_fci: lowest singlet eigenvalue of the electronic Hamiltonian (spatial two-electron
               function symmetric under exchange of the electrons) in the space of all
               two-electron states built from the two basis functions (full CI).
        E_rhf: closed-shell restricted Hartree-Fock ground-state energy.
    '''
    S, H = sto3g_one_electron(ZA, ZB, zetaA, zetaB, R)
    eri = sto3g_two_electron(zetaA, zetaB, R)
    enuc = ZA * ZB / R
    # MUTANT: one Roothaan-Hall self-consistent-field run from the core-Hamiltonian guess, which can settle
    # on a higher closed-shell branch instead of the global minimum
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
        if change < 1e-11:          # the energy error is quadratic in the density error
            break
    F = H + np.einsum('ls,mnsl->mn', P, eri) - 0.5 * np.einsum('ls,mlsn->mn', P, eri)
    E_rhf = float(0.5 * np.sum(P * (H + F)) + enuc)
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
