import numpy as np
from scipy.optimize import brentq


def critical_exponents(L):
    '''Finite-size estimates of critical exponents and of the central charge at the exact Tc.

    Input:
      L: int, strip width, 3 <= L <= 7.

    Output:
      dict with exactly the keys "x_sigma", "x_energy", "beta_over_nu", "gamma_over_nu", "y_h"
      and "c", each a float within 1e-4 of its formula. With Tc = 2 / ln(1 + sqrt(2)),
      r = ln((L+1) / L) and every quantity taken at T = Tc:
        x_sigma = L / (2 pi xi_spin(L)), x_energy = L / (2 pi xi_energy(L));
        beta_over_nu = ln(m_L / m_(L+1)) / r;
        gamma_over_nu = ln(chi_(L+1) / chi_L) / r;
        y_h = ln(H_edge(L) / H_edge(L+1)) / r;
        c from f_W = f_inf - pi c Tc / (6 W**2) + d / W**4, solved exactly for W = L, L+1, L+2.
    '''
    raise NotImplementedError
