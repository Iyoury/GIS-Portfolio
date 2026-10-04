import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def activation_inventory(lam, branching, sigma, capture_to, n0, history):
    '''Atoms of every nuclide after an irradiation history with neutron capture, decay and cooling.

    Inputs:
      lam: 1-D array of n decay constants in 1/s (1 <= n <= 30), 0 <= lam[i] <= 1e10.
      branching: (n, n) array of decay branching fractions as in network_inventory (step 2).
      sigma: 1-D array of n radiative-capture cross sections in barns (1 b = 1e-24 cm^2), sigma[i] = 0 or 1e-6 <= sigma[i] <= 1e7.
      capture_to: 1-D integer array of n entries; capture_to[i] = index of the nuclide made by a capture
                  on nuclide i, or -1 if that product is not followed.
      n0: 1-D array of n initial numbers of atoms, nonnegative, sum(n0) <= 1e100.
      history: list or tuple of (duration, flux) pairs applied in order; duration in s (0 <= duration <= 1e12),
               flux in neutrons / (cm^2 s) (0 <= flux <= 1e18; flux 0 is a cooling period).

    Output:
      x: numpy array of shape (n,), the numbers of atoms at the end of the history. With S = sum(n0):
         every entry whose exact value is positive and at least 1e-250 * S has a relative error below
         1e-10; every other entry (exact zeros included) lies within max(1e-250 * S, 1e-300) of the exact
         value.

    Raises:
      ValueError for inputs outside these ranges or of the wrong shape, for a capture_to entry that is not
      an integer in [-1, n - 1] or equals its own index, for branching as in step 2, or if the combined
      decay and capture network (edges where branching[j, i] > 0 or sigma[i] > 0 and capture_to[i] = j)
      has a cycle, or if lam[i] + sigma[i] * 1e-24 * flux exceeds 1e10 per s in some period.
    '''
    raise NotImplementedError
