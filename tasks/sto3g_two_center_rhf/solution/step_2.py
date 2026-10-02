import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


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
