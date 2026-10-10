import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.integrate import solve_ivp


def sto3g_one_electron(ZA, ZB, zetaA, zetaB, R):
    '''Overlap and core-Hamiltonian matrices for two atoms with one STO-3G 1s function each.

    Inputs:
      ZA, ZB: float, nuclear charges of atom A (at the origin) and atom B (at distance R on z),
              1 <= Z <= 3.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, 0.5 <= zeta <= 3.
      R: float, distance between the nuclei in bohr, 0.02 <= R <= 100.

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
    # MUTANT: Gaussian exponents scaled by zeta instead of zeta**2
    expo = np.array([alpha_one * zetaA, alpha_one * zetaB])
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
