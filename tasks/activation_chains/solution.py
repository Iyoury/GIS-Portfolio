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
      t: elapsed time, 0 <= t <= 1e20.

    Output:
      x: numpy array of shape (n,), the numbers of atoms at time t. With S = sum(n0) + t * sum(source):
         every entry whose exact value is positive and at least 1e-250 * S has a relative error below
         1e-10; every other entry (exact zeros included) lies within max(1e-250 * S, 1e-300) of the exact
         value.

    Raises:
      ValueError if the arrays do not have these shapes (1 <= n <= 30), if an entry is negative or not
      finite, if a decay constant exceeds 1e10, if a diagonal entry of branching is nonzero or a column
      sum exceeds 1 + 1e-12, if the network has a cycle, or if t is not finite or outside [0, 1e20].
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
    rates[:n, :n] = branching * lam[None, :]
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
    if np.any(np.diag(branching) != 0.0) or np.any(branching.sum(axis=0) > 1.0 + 1e-12):
        raise ValueError("branching needs a zero diagonal and column sums at most 1 + 1e-12")
    steps = []
    try:
        items = list(history)
    except TypeError:
        raise ValueError("history must be a sequence of (duration, flux) pairs")
    for item in items:
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
        capture_rate = sigma * 1e-24 * flux
        total = lam + capture_rate
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total > 0.0, (branching * lam[None, :] + capture * capture_rate[None, :]) / total, 0.0)
        # a column sum of the data at the bound 1 + 1e-12 must not be pushed above it by rounding
        frac = frac / np.maximum(1.0, frac.sum(axis=0) / (1.0 + 1e-12))[None, :]
        x = network_inventory(total, frac, np.zeros(n), x, duration)
    return x


def monitor_flux(lam, branching, sigma, capture_to, n0, t_irr, t_cool, k, activity, flux_lo, flux_hi):
    '''Neutron flux from the measured activity of an activation product (flux-monitor analysis).

    Inputs:
      lam: 1-D array of n decay constants in 1/s (1 <= n <= 30), 0 <= lam[i] <= 1e10.
      branching: (n, n) array, branching[j, i] = fraction of the decays of nuclide i that give nuclide j;
                 nonnegative, zero diagonal, column sums at most 1 + 1e-12.
      sigma: 1-D array of n radiative-capture cross sections in barns, 0 <= sigma[i] <= 1e7.
      capture_to: 1-D integer array of n entries, the nuclide made by a capture on nuclide i, or -1 if
                  it is not followed; never i itself.
      n0: 1-D array of n initial numbers of atoms, nonnegative.
      All values finite; the combined decay and capture network must be acyclic, exactly as in
      activation_inventory (step 3).
      t_irr: irradiation time in s at a constant unknown flux, 0 < t_irr <= 1e9.
      t_cool: cooling time in s without flux after the irradiation, 0 <= t_cool <= 1e9.
      k: int, index of the measured nuclide, with lam[k] > 0.
      activity: measured activity lam[k] * x[k] of nuclide k at the end of the cooling, in Bq, > 0.
      flux_lo, flux_hi: the flux lies in [flux_lo, flux_hi], 1e-2 <= flux_lo < flux_hi <= 1e18, and the
           caller guarantees that the activity A(flux) is strictly increasing on this interval.

    Output:
      flux: Python float, the flux in neutrons / (cm^2 s) with A(flux) = activity; relative error below
            1e-8 wherever d ln A / d ln flux >= 0.5 at the solution.

    Raises:
      ValueError for nuclide data as in step 3, for times, k, activity or flux bounds outside these
      ranges, or if activity lies outside [A(flux_lo), A(flux_hi)] by more than a relative 1e-9.
      An activity with |ln(activity / A(end))| <= 1e-9 for an end of the interval, on either side of
      it, gives that end.
    '''
    t_irr, t_cool, activity = float(t_irr), float(t_cool), float(activity)
    flux_lo, flux_hi = float(flux_lo), float(flux_hi)
    if not (np.isfinite(t_irr) and 0.0 < t_irr <= 1e9 and np.isfinite(t_cool) and 0.0 <= t_cool <= 1e9):
        raise ValueError("need 0 < t_irr <= 1e9 s and 0 <= t_cool <= 1e9 s")
    if not (np.isfinite(activity) and activity > 0.0):
        raise ValueError("activity must be positive and finite")
    if not (np.isfinite(flux_lo) and np.isfinite(flux_hi) and 1e-2 <= flux_lo < flux_hi <= 1e18):
        raise ValueError("need 1e-2 <= flux_lo < flux_hi <= 1e18")
    lam_arr = np.asarray(lam, dtype=float)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    k = int(k)
    if lam_arr.ndim != 1 or not 0 <= k < lam_arr.size or not lam_arr[k] > 0.0:
        raise ValueError("k must index a radioactive nuclide")

    def log_activity(log_flux):
        flux = np.exp(log_flux)
        x = activation_inventory(lam, branching, sigma, capture_to, n0, [(t_irr, flux), (t_cool, 0.0)])
        return np.log(lam_arr[k] * x[k]) if x[k] > 0.0 else -np.inf

    # The activity of a product made by m successive captures grows like flux**m at low flux, and burnup
    # of the target and of the product bends it over at high flux; on the interval it is increasing, so
    # ln A(ln flux) - ln activity has one root. Working in logarithms keeps the relative accuracy of tiny
    # activities and gives a well-scaled bracket over many decades.
    lo, hi = np.log(flux_lo), np.log(flux_hi)
    target = np.log(activity)
    f_lo, f_hi = log_activity(lo) - target, log_activity(hi) - target
    # the interval is closed: an activity within a relative 1e-9 of an end value, on either side, gives
    # that end; otherwise it must lie strictly between the two end values
    if abs(f_lo) <= 1e-9:
        return float(flux_lo)
    if abs(f_hi) <= 1e-9:
        return float(flux_hi)
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError("the activity is not reached for a flux in [flux_lo, flux_hi]")
    root = brentq(lambda y: log_activity(y) - target, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
    flux = float(np.exp(root))
    return flux
