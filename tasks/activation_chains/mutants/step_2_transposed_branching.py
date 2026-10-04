import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def network_inventory(lam, branching, source, n0, t):
    '''Atoms of every nuclide of a branching decay network with constant production after a time t.

    Inputs:
      lam: 1-D array of n decay constants (1 <= n <= 30), 0 <= lam[i] <= 1e10 (per unit of time).
      branching: (n, n) array, branching[j, i] = fraction of the decays of nuclide i that give nuclide j;
                 nonnegative, zero diagonal, column sums at most 1 + 1e-12 (1 up to rounding; the rest
                 leaves the network). The
                 directed graph with an edge i -> j wherever branching[j, i] > 0 must be acyclic.
      source: 1-D array of n constant production rates (atoms per unit of time), nonnegative.
      n0: 1-D array of n initial numbers of atoms, nonnegative.
      t: elapsed time, 0 <= t <= 1e20, with sum(n0) + t * sum(source) <= 1e100.

    Output:
      x: numpy array of shape (n,), the numbers of atoms at time t. With S = sum(n0) + t * sum(source):
         every entry whose exact value is positive and at least 1e-250 * S has a relative error below
         1e-10; every other entry (exact zeros included) lies within max(1e-250 * S, 1e-300) of the exact
         value.

    Raises:
      ValueError if the arrays do not have these shapes (1 <= n <= 30), if an entry is negative or not
      finite, if a decay constant exceeds 1e10, if a diagonal entry of branching is nonzero or a column
      sum exceeds 1 + 1e-12, if the network has a cycle, if t is not finite or outside [0, 1e20], or if
      sum(n0) + t * sum(source) > 1e100.
    '''
    lam = np.asarray(lam, dtype=float)
    branching = np.asarray(branching, dtype=float)
    source = np.asarray(source, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    n = lam.size
    if (lam.ndim != 1 or not 1 <= n <= 30 or branching.shape != (n, n) or source.shape != (n,)
            or n0.shape != (n,)):
        raise ValueError("need lam, source, n0 of shape (n,) and branching of shape (n, n), 1 <= n <= 30")
    for name, arr in (("lam", lam), ("branching", branching), ("source", source), ("n0", n0)):
        if not (np.all(np.isfinite(arr)) and np.all(arr >= 0.0)):
            raise ValueError("%s must be finite and nonnegative" % name)
    if np.any(lam > 1e10):
        raise ValueError("decay constants must not exceed 1e10")
    if np.any(np.diag(branching) != 0.0) or np.any(branching.sum(axis=0) > 1.0 + 1e-12):
        raise ValueError("branching needs a zero diagonal and column sums at most 1 + 1e-12")
    t = float(t)
    if not (np.isfinite(t) and 0.0 <= t <= 1e20):
        raise ValueError("need 0 <= t <= 1e20")
    if n0.sum() + t * source.sum() > 1e100:
        raise ValueError("sum(n0) + t * sum(source) must not exceed 1e100")
    # acyclic: repeatedly remove nuclides without incoming edges (Kahn)
    edges = branching > 0.0
    indeg = edges.sum(axis=1)
    removed = np.zeros(n, dtype=bool)
    for _ in range(n):
        free = np.where(~removed & (indeg == 0))[0]
        if free.size == 0:
            break
        removed[free] = True
        indeg = indeg - edges[:, free].sum(axis=1)
    if not removed.all():
        raise ValueError("the decay network has a cycle")
    if t == 0.0:
        x = n0.copy()
        return x
    # rate matrix augmented by one constant "source" state: d/dt [x, 1] = [[M, q], [0, 0]] [x, 1]
    m = n + 1
    rates = np.zeros((m, m))
    rates[:n, :n] = branching.T * lam[None, :]        # MUTANT: branching read as branching[i, j]
    rates[:n, n] = source
    removal = np.r_[lam, 0.0]
    # exp(t A) of A = rates - diag(removal) by scaling and squaring in nonnegative arithmetic only (the
    # off-diagonal entries of A are >= 0). On an acyclic network exp(t A) has the diagonal exp(-removal t)
    # exactly; it is reset at every squaring level so that its rounding is never doubled s times.
    rmax = removal.max()
    if rmax == 0.0:
        s = 0
    else:
        s = max(0, int(np.ceil(np.log2(t * rmax / 0.5))))
    tau = t / 2.0 ** s
    c = tau * rmax
    B = tau * rates + np.diag(c - tau * removal)
    E = np.eye(m)
    term = np.eye(m)
    for k in range(1, 400):
        term = term @ B / k
        E = E + term
        if np.all(term <= 1e-18 * E):
            break
    E = E * np.exp(-c)
    np.fill_diagonal(E, np.exp(-removal * tau))
    for level in range(1, s + 1):
        E = E @ E
        np.fill_diagonal(E, np.exp(-removal * (tau * 2.0 ** level)))
    x = E @ np.r_[n0, 1.0]
    x = x[:n]
    return x
