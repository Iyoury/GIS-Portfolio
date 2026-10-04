import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def _mp_expm_apply(rates, removal, x0, t):
    # exp(t A) x0 for A = rates - diag(removal) by mpmath's expm, the precision doubled until two runs
    # agree to 1e-13 in every entry above 1e-250 of the scale sum(x0)
    m = len(removal)
    scale = float(sum(x0))
    old = None
    dps = 40
    while True:
        with mp.workdps(dps):
            A = mp.matrix(m, m)
            for i in range(m):
                for j in range(m):
                    A[i, j] = mp.mpf(rates[i][j]) - (mp.mpf(removal[i]) if i == j else 0)
            E = mp.expm(A * mp.mpf(t))
            v = E * mp.matrix([mp.mpf(a) for a in x0])
            new = [v[i] for i in range(m)]
        if old is not None:
            floor = mp.mpf(1e-250) * scale
            if all(abs(a - b) <= mp.mpf("1e-13") * abs(b) or (abs(a) < floor and abs(b) < floor)
                   for a, b in zip(old, new)):
                return np.array([float(a) for a in new])
        old = new
        dps *= 2
        if dps > 5000:
            raise RuntimeError("no convergence in extended precision")


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
    # Other method: the chain matrix exponential by mpmath (Pade / Taylor with scaling and squaring) in
    # adaptively increased precision.
    n = lam.size
    if t == 0.0:
        x = n0.copy()
        return x
    rates = np.zeros((n, n))
    rates[np.arange(1, n), np.arange(n - 1)] = lam[:-1]
    x = _mp_expm_apply(rates, lam, n0, t)
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
    # Other method: the augmented rate matrix exponentiated by mpmath in adaptively increased precision.
    if t == 0.0:
        x = n0.copy()
        return x
    rates = np.zeros((n + 1, n + 1))
    rates[:n, :n] = branching * lam[None, :]
    rates[:n, n] = source
    y0 = np.r_[n0, 1.0]
    x = _mp_expm_apply(rates, np.r_[lam, 0.0], y0, t)[:n]
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
    # Other method: each period is exponentiated directly from the decay and capture rates in extended
    # precision (no branching fractions).
    x = n0.copy()
    for duration, flux in steps:
        if duration == 0.0:
            continue
        capture_rate = sigma * 1e-24 * flux
        rates = branching * lam[None, :] + capture * capture_rate[None, :]
        x = _mp_expm_apply(rates, lam + capture_rate, x, duration)
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

    # Other method: bisection on ln(flux) down to a bracket of 1e-6, then secant steps on ln A.
    def log_activity(y):
        x = activation_inventory(lam, branching, sigma, capture_to, n0, [(t_irr, float(np.exp(y))), (t_cool, 0.0)])
        return float(np.log(lam_arr[k] * x[k])) if x[k] > 0.0 else -np.inf

    target = np.log(activity)
    a, b = np.log(flux_lo), np.log(flux_hi)
    fa, fb = log_activity(a) - target, log_activity(b) - target
    if abs(fa) <= 1e-9:
        return float(flux_lo)
    if abs(fb) <= 1e-9:
        return float(flux_hi)
    if fa > 0.0 or fb < 0.0:
        raise ValueError("the activity is not reached for a flux in [flux_lo, flux_hi]")
    while b - a > 1e-6:
        mid = 0.5 * (a + b)
        fm = log_activity(mid) - target
        if fm <= 0.0:
            a, fa = mid, fm
        else:
            b, fb = mid, fm
    y0, y1, f0, f1 = a, b, fa, fb
    for _ in range(30):
        if f1 == f0:
            break
        y2 = y1 - f1 * (y1 - y0) / (f1 - f0)
        y0, f0 = y1, f1
        y1, f1 = y2, log_activity(y2) - target
        if abs(y1 - y0) < 1e-14:
            break
    flux = float(np.exp(y1))
    return flux
