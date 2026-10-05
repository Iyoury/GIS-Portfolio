import numpy as np
import math


def pair_densities(T, psi):
    '''Number densities of electrons and positrons in an ideal Fermi gas with pairs.

    Inputs:
      T: float, temperature in K, 1e7 <= T <= 1e11.
      psi: float, (mu + m_e c^2) / (k T) with mu the electron chemical potential without the rest
           mass, 0 < psi <= 1e6; electrons occupy 1 / (exp(eps/theta - psi) + 1), positrons
           1 / (exp(eps/theta + psi) + 1), eps = sqrt(1 + p^2), theta = k T / (m_e c^2).

    Output:
      (n_minus, n_plus, n_net): Python floats in cm^-3, n = (1 / (pi^2 lambda^3)) int p^2 f dp with
        lambda = hbar / (m_e c); n_net = n_minus - n_plus. n_minus and n_net with a relative error
        below 1e-10; n_plus with a relative error below 1e-10 when at least 1e-250 cm^-3, otherwise
        within 1e-250 cm^-3. Each call within 10 s.

    Raises:
      ValueError if T or psi is not finite, if T is outside [1e7, 1e11] or psi is not in (0, 1e6].
    '''
    raise NotImplementedError
