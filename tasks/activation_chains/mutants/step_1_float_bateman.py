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
      x: numpy array of shape (n,), the numbers of atoms at time t. Relative error below 1e-10 for every
         entry not smaller than 1e-250 * sum(n0); smaller entries within 1e-250 * sum(n0) of the exact value.

    Raises:
      ValueError if lam and n0 are not 1-D arrays of the same length between 1 and 30, if an entry of
      lam is negative, above 1e10 or not finite, if an entry of n0 is negative or not finite, or if t
      is not finite or outside [0, 1e20].
    '''
    lam = np.asarray(lam, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    if lam.ndim != 1 or n0.shape != lam.shape or not 1 <= lam.size <= 30:
        raise ValueError("lam and n0 must be 1-D arrays of the same length between 1 and 30")
    if not (np.all(np.isfinite(lam)) and np.all((lam >= 0.0) & (lam <= 1e10))):
        raise ValueError("decay constants must lie in [0, 1e10]")
    if not (np.all(np.isfinite(n0)) and np.all(n0 >= 0.0)):
        raise ValueError("initial amounts must be finite and nonnegative")
    t = float(t)
    if not (np.isfinite(t) and 0.0 <= t <= 1e20):
        raise ValueError("need 0 <= t <= 1e20")
    n = lam.size
    if t == 0.0 or lam.max() == 0.0:
        x = n0.copy()
        return x
    # MUTANT: the Bateman sum in double precision (equal constants split by a relative 1e-9)
    lam = lam * (1.0 + 1e-9 * np.arange(n))
    x = np.zeros(n)
    for j in range(n):
        if n0[j] == 0.0:
            continue
        for k in range(j, n):
            pre = np.prod(lam[j:k])
            tot = 0.0
            for i in range(j, k + 1):
                others = [lam[l] - lam[i] for l in range(j, k + 1) if l != i]
                tot += np.exp(-lam[i] * t) / np.prod(others)
            x[k] += n0[j] * pre * tot
    return x
