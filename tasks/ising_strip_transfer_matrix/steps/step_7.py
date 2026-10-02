import numpy as np
from scipy.optimize import brentq


def bulk_thermodynamics(T, h):
    '''Free energy, magnetization, susceptibility and specific heat per spin of the infinite lattice in a field.

    Inputs:
      T: float, temperature, 2 <= T <= 3 (J = 1, k_B = 1).
      h: float, uniform field, 0.005 <= h <= 0.2; the energy is E = - sum_<ij> s_i s_j - h sum_i s_i.

    Output:
      (f, m, chi, c): tuple of four Python floats for the infinite square lattice (the limit of
      the strip quantities as the width goes to infinity):
        f: free energy per spin, f = -T lim ln(Z) / N (units of J), absolute accuracy 1e-10;
        m: magnetization per spin <s_i> = -df/dh, absolute accuracy 1e-9;
        chi: susceptibility per spin, chi = dm/dh at fixed T (units 1/J), relative accuracy 1e-5;
        c: specific heat per spin at fixed h, c = -T d2f/dT2 (units k_B), relative accuracy 1e-5.
    '''
    raise NotImplementedError
