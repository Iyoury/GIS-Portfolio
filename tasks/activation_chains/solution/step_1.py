import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def chain_inventory(lam, n0, t):
    '''Atoms of every member of a linear radioactive decay chain after a time t (Bateman problem).

    Inputs:
      lam: 1-D array of n decay constants (1 <= n <= 30), member i decays into member i + 1 at the
           rate lam[i]; the last member decays out of the chain. 0 <= lam[i] <= 1e10 (per unit of time).
      n0: 1-D array of n initial numbers of atoms, nonnegative, sum(n0) <= 1e100.
      t: elapsed time, 0 <= t <= 1e20 (same unit of time).

    Output:
      x: numpy array of shape (n,), the numbers of atoms at time t. With S = sum(n0): every entry whose
         exact value is positive and at least 1e-250 * S has a relative error below 1e-10; every other
         entry (exact zeros included) lies within max(1e-250 * S, 1e-300) of the exact value.

    Raises:
      ValueError if lam and n0 are not 1-D arrays of the same length between 1 and 30, if an entry of
      lam is negative, above 1e10 or not finite, if an entry of n0 is negative or not finite, if
      sum(n0) > 1e100, or if t is not finite or outside [0, 1e20].
    '''
    lam = np.asarray(lam, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    if lam.ndim != 1 or n0.shape != lam.shape or not 1 <= lam.size <= 30:
        raise ValueError("lam and n0 must be 1-D arrays of the same length between 1 and 30")
    if not (np.all(np.isfinite(lam)) and np.all((lam >= 0.0) & (lam <= 1e10))):
        raise ValueError("decay constants must lie in [0, 1e10]")
    if not (np.all(np.isfinite(n0)) and np.all(n0 >= 0.0)):
        raise ValueError("initial amounts must be finite and nonnegative")
    if n0.sum() > 1e100:
        raise ValueError("sum(n0) must not exceed 1e100")
    t = float(t)
    if not (np.isfinite(t) and 0.0 <= t <= 1e20):
        raise ValueError("need 0 <= t <= 1e20")
    n = lam.size
    if t == 0.0 or lam.max() == 0.0:
        x = n0.copy()
        return x
    # The chain matrix M (M[i, i] = -lam[i], M[i + 1, i] = lam[i]) has nonnegative off-diagonal entries.
    # exp(t M) is built by scaling and squaring in nonnegative arithmetic only:
    #  - tau = t / 2**s with tau * max(lam) <= 1/2; exp(tau M) = exp(-c) exp(tau M + c I) with c = tau max(lam),
    #    and tau M + c I >= 0, so its Taylor series has no cancellation;
    #  - each squaring E -> E @ E only adds products of nonnegative numbers. The diagonal of exp(t M) is
    #    exp(-lam t) exactly (M is triangular), and it is set from the exponential at every level: a
    #    squared diagonal would double its relative error at each of the s squarings and spoil every
    #    entry that depends on it.
    # The Bateman sum over exp(-lam_i t) / prod(lam_l - lam_i) cancels catastrophically for close decay
    # constants and is undefined for equal ones; a dense expm is accurate only relative to the largest entry.
    rmax = lam.max()
    s = max(0, int(np.ceil(np.log2(t * rmax / 0.5))))
    tau = t / 2.0 ** s
    c = tau * rmax
    B = np.diag(c - tau * lam)
    B[np.arange(1, n), np.arange(n - 1)] = tau * lam[:-1]
    E = np.eye(n)
    term = np.eye(n)
    for k in range(1, 400):
        term = term @ B / k
        E = E + term
        if np.all(term <= 1e-18 * E):
            break
    E = E * np.exp(-c)
    np.fill_diagonal(E, np.exp(-lam * tau))
    for level in range(1, s + 1):
        E = E @ E
        np.fill_diagonal(E, np.exp(-lam * (tau * 2.0 ** level)))
    x = E @ n0
    return x
