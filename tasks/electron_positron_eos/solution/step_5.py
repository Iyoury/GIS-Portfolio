import numpy as np
import math

def electron_positron_eos(rho, T, Ye):
    '''Equation of state of the electron-positron gas for a mass density, temperature and electron fraction.

    Inputs:
      rho: float, mass density in g cm^-3.
      T: float, temperature in K, 1e7 <= T <= 1e11.
      Ye: float, electrons per baryon, 0 < Ye <= 1; 1e-10 <= rho Ye <= 1e13.

    Output:
      dict with the Python floats "psi", "n_minus", "n_plus", "P", "u", "s", "cv" (units as in steps
        1-4). Accuracy with respect to the exact values for the given (rho, T, Ye): psi, n_minus, P, u, s
        relative 1e-10; n_plus relative 1e-10 when at least 1e-250 cm^-3, otherwise within 1e-250 cm^-3;
        cv relative 1e-8. This includes the error propagated from psi, which n_minus, n_plus, P, u and s
        amplify up to about 600 times, so psi must be found to about 1e-13 relative.

    Raises:
      ValueError if rho, T or Ye is not finite, if Ye is not in (0, 1], or if T or rho Ye is outside
      its range.
    '''
    rho, T, Ye = float(rho), float(T), float(Ye)
    if not (math.isfinite(rho) and math.isfinite(T) and math.isfinite(Ye)):
        raise ValueError("rho, T and Ye must be finite")
    if not 0.0 < Ye <= 1.0:
        raise ValueError("need 0 < Ye <= 1")
    psi = degeneracy_parameter(rho * Ye, T)
    n_minus, n_plus, n_net = pair_densities(T, psi)
    P, u, s = pair_thermodynamics(T, psi)
    cv = specific_heat(rho * Ye, T)
    result = {"psi": psi, "n_minus": n_minus, "n_plus": n_plus, "P": P, "u": u, "s": s, "cv": cv}
    return result
