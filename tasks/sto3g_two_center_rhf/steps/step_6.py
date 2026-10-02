import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


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
    raise NotImplementedError
