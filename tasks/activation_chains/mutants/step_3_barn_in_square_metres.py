import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def activation_inventory(lam, branching, sigma, capture_to, n0, history):
    '''Atoms of every nuclide after an irradiation history with neutron capture, decay and cooling.

    Inputs:
      lam: 1-D array of n decay constants in 1/s (1 <= n <= 30), 0 <= lam[i] <= 1e10.
      branching: (n, n) array of decay branching fractions as in network_inventory (step 2).
      sigma: 1-D array of n radiative-capture cross sections in barns (1 b = 1e-24 cm^2), 0 <= sigma[i] <= 1e7.
      capture_to: 1-D integer array of n entries; capture_to[i] = index of the nuclide made by a capture
                  on nuclide i, or -1 if that product is not followed.
      n0: 1-D array of n initial numbers of atoms, nonnegative.
      history: sequence of (duration, flux) pairs applied in order; duration in s (0 <= duration <= 1e12),
               flux in neutrons / (cm^2 s) (0 <= flux <= 1e18; flux 0 is a cooling period).

    Output:
      x: numpy array of shape (n,), the numbers of atoms at the end of the history. Relative error below
         1e-10 for every entry not smaller than 1e-250 * sum(n0); smaller entries within that bound of the
         exact value.

    Raises:
      ValueError for inputs outside these ranges or of the wrong shape, for a capture_to entry that is not
      an integer in [-1, n - 1] or equals its own index, for branching as in step 2, or if the combined
      decay and capture network (edges where branching[j, i] > 0 or sigma[i] > 0 and capture_to[i] = j)
      has a cycle, or if lam[i] + sigma[i] * 1e-24 * flux exceeds 1e10 per s in some period.
    '''
    lam = np.asarray(lam, dtype=float)
    branching = np.asarray(branching, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    n = lam.size
    if lam.ndim != 1 or not 1 <= n <= 30 or sigma.shape != (n,) or n0.shape != (n,) or branching.shape != (n, n):
        raise ValueError("need lam, sigma, n0 of shape (n,) and branching of shape (n, n), 1 <= n <= 30")
    cap = np.asarray(capture_to)
    if cap.shape != (n,) or not (np.issubdtype(cap.dtype, np.integer) and not np.issubdtype(cap.dtype, np.bool_)):
        raise ValueError("capture_to must be an integer array of shape (n,)")
    cap = cap.astype(int)
    if np.any((cap < -1) | (cap >= n)) or np.any(cap == np.arange(n)):
        raise ValueError("capture_to entries must be -1 or another nuclide index")
    for name, arr in (("lam", lam), ("sigma", sigma), ("n0", n0), ("branching", branching)):
        if not (np.all(np.isfinite(arr)) and np.all(arr >= 0.0)):
            raise ValueError("%s must be finite and nonnegative" % name)
    if np.any(lam > 1e10) or np.any(sigma > 1e7):
        raise ValueError("need lam <= 1e10 and sigma <= 1e7 b")
    steps = []
    for item in history:
        try:
            duration, flux = (float(v) for v in item)
        except (TypeError, ValueError):
            raise ValueError("history must be a sequence of (duration, flux) pairs")
        if not (np.isfinite(duration) and np.isfinite(flux) and 0.0 <= duration <= 1e12 and 0.0 <= flux <= 1e18):
            raise ValueError("need 0 <= duration <= 1e12 s and 0 <= flux <= 1e18 per cm^2 s")
        steps.append((duration, flux))
    # combined network: decay edges and capture edges (whatever the flux, so that the check does not
    # depend on the history)
    capture = np.zeros((n, n))
    for i in range(n):
        if cap[i] >= 0:
            capture[cap[i], i] = 1.0
    edges = (branching > 0.0) | ((capture > 0.0) & (sigma[None, :] > 0.0))
    indeg = edges.sum(axis=1)
    removed = np.zeros(n, dtype=bool)
    for _ in range(n):
        free = np.where(~removed & (indeg == 0))[0]
        if free.size == 0:
            break
        removed[free] = True
        indeg = indeg - edges[:, free].sum(axis=1)
    if not removed.all():
        raise ValueError("the decay and capture network has a cycle")
    if any(np.any(lam + sigma * 1e-24 * flux > 1e10) for _, flux in steps):
        raise ValueError("lam + sigma * flux must not exceed 1e10 per s")
    x = n0.copy()
    for duration, flux in steps:
        # under a flux phi nuclide i is removed at lam_i + sigma_i phi (sigma in cm^2); the decays go out
        # with the branching fractions and the captures to capture_to[i]. As a branching network:
        # total rate r_i and fractions (lam_i b_ji + sigma_i phi [j = capture_to i]) / r_i.
        capture_rate = sigma * 1e-28 * flux           # MUTANT: barns converted to m^2 although the flux is per cm^2
        total = lam + capture_rate
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total > 0.0, (branching * lam[None, :] + capture * capture_rate[None, :]) / total, 0.0)
        x = network_inventory(total, frac, np.zeros(n), x, duration)
    return x
