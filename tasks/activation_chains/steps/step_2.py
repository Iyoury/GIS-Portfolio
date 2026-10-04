import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def network_inventory(lam, branching, source, n0, t):
    '''Atoms of every nuclide of a branching decay network with constant production after a time t.

    Inputs:
      lam: 1-D array of n decay constants (1 <= n <= 30), 0 <= lam[i] <= 1e10 (per unit of time).
      branching: (n, n) array, branching[j, i] = fraction of the decays of nuclide i that give nuclide j;
                 nonnegative, zero diagonal, column sums at most 1 (the rest leaves the network). The
                 directed graph with an edge i -> j wherever branching[j, i] > 0 must be acyclic.
      source: 1-D array of n constant production rates (atoms per unit of time), nonnegative.
      n0: 1-D array of n initial numbers of atoms, nonnegative.
      t: elapsed time, 0 <= t <= 1e20.

    Output:
      x: numpy array of shape (n,), the numbers of atoms at time t. Relative error below 1e-10 for every
         entry not smaller than 1e-250 * (sum(n0) + t * sum(source)); smaller entries within that bound
         of the exact value.

    Raises:
      ValueError if the arrays do not have these shapes (1 <= n <= 30), if an entry is negative or not
      finite, if a decay constant exceeds 1e10, if a diagonal entry of branching is nonzero or a column
      sum exceeds 1 + 1e-12, if the network has a cycle, or if t is not finite or outside [0, 1e20].
    '''
    raise NotImplementedError
