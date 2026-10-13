import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import brentq


def electron_positron_eos(rho, T, Ye):
    '''Equation of state and adiabatic exponents of the electron-positron gas.

    Inputs:
      rho: float, mass density in g cm^-3.
      T: float, temperature in K, 1e7 <= T <= 1e11.
      Ye: float, electrons per baryon, 0 < Ye <= 1; 1e-10 <= rho Ye <= 1e13.

    Output:
      dict with exactly the nine keys "psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1",
        "nabla_ad", whose values are Python floats: psi, densities (cm^-3), P (erg cm^-3), u (erg cm^-3),
        s (erg K^-1 cm^-3) and cv (erg K^-1 cm^-3) as in steps 1-4; gamma1 = (d ln P / d ln rho) and
        nabla_ad = (d ln T / d ln P), both at constant s / rho and Ye for the electron-positron gas alone,
        pairs in equilibrium. Accuracy with respect to the exact values for the given (rho, T, Ye),
        including the error propagated from psi: psi, n_minus, P, u, s relative 1e-10; n_plus relative
        1e-10 when at least 1e-250 cm^-3, otherwise within 1e-250 cm^-3; cv, gamma1, nabla_ad relative
        1e-8.

    Raises:
      ValueError if Ye is not in (0, 1], or if T or rho Ye is outside its range.
    '''
    raise NotImplementedError
