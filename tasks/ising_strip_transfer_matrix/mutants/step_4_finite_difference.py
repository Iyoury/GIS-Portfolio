import numpy as np
from scipy.optimize import brentq


def strip_susceptibility(L, T):
    '''Zero-field magnetic susceptibility per spin of an infinitely long periodic strip.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10.
      T: float, temperature, 1 <= T <= 10 (J = 1, k_B = 1).

    Output:
      chi: float, chi = - d^2 f / d h^2 at h = 0, where f(h) is the free energy per spin of
           the infinitely long strip in the uniform field h of transfer_matrix (units 1/J).
           Relative accuracy 1e-6.
    '''
    def free_energy(h):
        return -T * np.log(np.linalg.eigvalsh(transfer_matrix(L, T, h))[-1]) / L

    d = 1e-4                                           # fixed finite-difference step in h
    chi = float(-(free_energy(d) - 2.0 * free_energy(0.0) + free_energy(-d)) / d ** 2)
    return chi
