import numpy as np
from scipy.optimize import brentq


def critical_exponents(L):
    '''Finite-size estimates of critical exponents and of the central charge at the exact Tc.

    Input:
      L: int, strip width, 3 <= L <= 7.

    Output:
      dict with exactly the keys "x_sigma", "x_energy", "beta_over_nu", "gamma_over_nu", "y_h"
      and "c", each a float within 1e-4 of the value defined in the prompt. Every quantity is
      taken at T = Tc = 2 / ln(1 + sqrt(2)):
        x_sigma, x_energy: scaling dimensions from Cardy's relation (correlation length
          W / (2 pi x) on a periodic strip of width W) at the single width L;
        beta_over_nu, gamma_over_nu, y_h: two-width effective exponents of the power laws
          m_W ~ W**(-beta/nu), chi_W ~ W**(gamma/nu), H_edge(W) ~ W**(-y_h) between W = L, L+1;
        c: central charge from f_W = f_inf - pi c Tc / (6 W**2) + d / W**4 held exactly for
          W = L, L+1, L+2.
    '''
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    r = np.log((L + 1.0) / L)
    lengths = [strip_lengths(W, Tc) for W in (L, L + 1, L + 2)]
    widths = np.array([L, L + 1, L + 2], dtype=float)
    A = np.column_stack([np.ones(3), -np.pi * Tc / (6.0 * widths ** 2), widths ** -4.0])
    c = float(np.linalg.solve(A, np.array([x[0] for x in lengths]))[1])
    result = {
        "x_sigma": float(L / (2.0 * np.pi * lengths[0][1])),
        "x_energy": float(L / (2.0 * np.pi * lengths[0][2])),
        "beta_over_nu": float(np.log(strip_magnetization(L, Tc) / strip_magnetization(L + 1, Tc)) / r),
        "gamma_over_nu": float(np.log(strip_susceptibility(L + 1, Tc) / strip_susceptibility(L, Tc)) / r),
        "y_h": float(np.log(lee_yang_edge(L, Tc) / lee_yang_edge(L + 1, Tc)) / r),
        "c": c,
    }
    return result
