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
    raise NotImplementedError
