import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def chain_inventory(lam, n0, t):
    '''Atoms of every member of a linear radioactive decay chain after a time t (Bateman problem).

    Inputs:
      lam: 1-D array of n decay constants (1 <= n <= 30), member i decays into member i + 1 at the
           rate lam[i]; the last member decays out of the chain. 0 <= lam[i] <= 1e10 (per unit of time).
      n0: 1-D array of n initial numbers of atoms, nonnegative.
      t: elapsed time, 0 <= t <= 1e20 (same unit of time).

    Output:
      x: numpy array of shape (n,), the numbers of atoms at time t. With S = sum(n0): every entry whose
         exact value is positive and at least 1e-250 * S has a relative error below 1e-10; every other
         entry (exact zeros included) lies within max(1e-250 * S, 1e-300) of the exact value.

    Raises:
      ValueError if lam and n0 are not 1-D arrays of the same length between 1 and 30, if an entry of
      lam is negative, above 1e10 or not finite, if an entry of n0 is negative or not finite, or if t
      is not finite or outside [0, 1e20].
    '''
    raise NotImplementedError
