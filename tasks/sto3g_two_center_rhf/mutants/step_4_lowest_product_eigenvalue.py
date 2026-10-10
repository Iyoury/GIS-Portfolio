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
    # MUTANT: E_fci is the lowest eigenvalue of the whole two-electron product space psi_p(1) psi_q(2)
    # (4 x 4, <pq|H|rs> = h_pr d_qs + d_pr h_qs + (pr|qs)), which also contains the spatially
    # antisymmetric (triplet) state, instead of the lowest singlet
    I2 = np.eye(2)
    Hp = (np.einsum('pr,qs->pqrs', h, I2) + np.einsum('pr,qs->pqrs', I2, h)
          + np.einsum('prqs->pqrs', g)).reshape(4, 4)
    E_fci = float(np.linalg.eigvalsh(Hp)[0] + enuc)
    result = (E_fci, E_rhf)
    return result
