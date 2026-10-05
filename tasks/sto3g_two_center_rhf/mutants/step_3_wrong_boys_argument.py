import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.integrate import solve_ivp


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
    # MUTANT: Boys argument (p + q)|P - Q|^2 instead of p q / (p + q) |P - Q|^2
    val = pre * gauss * boys_function(0, (p + q) * (center_p - center_q) ** 2)[..., 0]
    eri = np.sum(w * norm * val, axis=(4, 5, 6, 7))
    return eri
