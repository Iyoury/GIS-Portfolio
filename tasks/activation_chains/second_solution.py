import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def _topological_order(rates):
    # Kahn's algorithm on the edges i -> j with rates[j][i] > 0 (the caller has checked acyclicity)
    n = len(rates)
    indeg = [sum(1 for i in range(n) if rates[j][i] > 0.0) for j in range(n)]
    order, ready = [], [j for j in range(n) if indeg[j] == 0]
    while ready:
        i = ready.pop()
        order.append(i)
        for j in range(n):
            if rates[j][i] > 0.0:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
    return order


def _exp_poly_amounts(order, rates, removal, x0, src, t, dps):
    # Generalized Bateman solution: every amount is a finite sum of terms c t**p exp(-a t), where a runs
    # over the removal rates of the nuclide and of its ancestors (0 for the constant source). Going down
    # the network in topological order, x_j = x0_j exp(-r_j t) + int_0^t exp(-r_j (t - s)) g_j(s) ds with
    # the feeding g_j = sum_i rates[j, i] x_i + src_j, and each term of g_j integrates in closed form:
    #   a == r_j: t**p exp(-a t)  ->  t**(p + 1) / (p + 1) exp(-r_j t)
    #   a != r_j: t**p exp(-a t)  ->  p! / d**(p + 1) [exp(-r_j t) - exp(-a t) sum_m (d t)**m / m!], d = a - r_j
    # (equal rates exactly, nearly equal ones, or a short time, through large terms that cancel in the sum).
    # Next to every coefficient c a bound m >= |c| on the magnitudes it was summed from is carried along
    # (the same operations on absolute values), so that the rounding error of every amount is bounded by
    # a modest multiple of eps * sum_a exp(-a t) sum_p m t**p, however strongly the terms cancel.
    n = len(removal)
    with mp.workdps(dps):
        T = mp.mpf(t)
        terms = [None] * n
        out = [mp.mpf(0)] * n
        mag = [mp.mpf(0)] * n
        expo = {}
        for j in order:
            g = {}
            for i in range(n):
                w = rates[j][i]
                if w > 0.0 and terms[i]:
                    W = mp.mpf(w)
                    for a, (cs, ms) in terms[i].items():
                        acc, macc = g.setdefault(a, ([], []))
                        while len(acc) < len(cs):
                            acc.append(mp.mpf(0))
                            macc.append(mp.mpf(0))
                        for p, (c, m) in enumerate(zip(cs, ms)):
                            acc[p] += W * c
                            macc[p] += W * m
            if src[j] > 0.0:
                acc, macc = g.setdefault(0.0, ([mp.mpf(0)], [mp.mpf(0)]))
                acc[0] += mp.mpf(src[j])
                macc[0] += mp.mpf(src[j])
            rj = removal[j]
            xj = {rj: ([mp.mpf(x0[j])], [mp.mpf(x0[j])])}
            for a, (cs, ms) in g.items():
                if a == rj:
                    acc, macc = xj[rj]
                    while len(acc) < len(cs) + 1:
                        acc.append(mp.mpf(0))
                        macc.append(mp.mpf(0))
                    for p, (c, m) in enumerate(zip(cs, ms)):
                        acc[p + 1] += c / (p + 1)
                        macc[p + 1] += m / (p + 1)
                else:
                    d = mp.mpf(a) - mp.mpf(rj)
                    acc_a, macc_a = xj.setdefault(a, ([], []))
                    while len(acc_a) < len(cs):
                        acc_a.append(mp.mpf(0))
                        macc_a.append(mp.mpf(0))
                    for p, (c, m) in enumerate(zip(cs, ms)):
                        if m == 0:
                            continue
                        scale = mp.factorial(p) / d ** (p + 1)
                        f, mf = c * scale, m * abs(scale)
                        xj[rj][0][0] += f
                        xj[rj][1][0] += mf
                        dm = mp.mpf(1)
                        for q in range(p + 1):
                            acc_a[q] -= f * dm / mp.factorial(q)
                            macc_a[q] += mf * abs(dm) / mp.factorial(q)
                            dm *= d
            terms[j] = xj
            val, size = mp.mpf(0), mp.mpf(0)
            for a, (cs, ms) in xj.items():
                if a not in expo:
                    expo[a] = mp.exp(-mp.mpf(a) * T)
                poly, mpoly, tp = mp.mpf(0), mp.mpf(0), mp.mpf(1)
                for c, m in zip(cs, ms):
                    poly += c * tp
                    mpoly += m * tp
                    tp *= T
                val += expo[a] * poly
                size += expo[a] * mpoly
            out[j] = val
            mag[j] = size
        return out, mag, mp.eps


def _unit_columns(branching):
    # columns of the branching matrix whose sum exceeds 1 (by at most 1e-12, as the callers check) are
    # rescaled to sum to 1, one entry at a time
    out = np.array(branching, dtype=float)
    for i in range(out.shape[1]):
        s = sum(float(v) for v in out[:, i])
        if s > 1.0:
            out[:, i] = [v / s for v in out[:, i]]
    return out


def _exp_poly_apply(rates, removal, x0, src, t):
    # amounts at time t of dx/dt = rates x - diag(removal) x + src on an acyclic network. Running error
    # bound: every coefficient comes out of at most a few thousand correctly rounded operations (at most
    # 30 levels of convolution, each with sums of at most about 30 * 30 terms), so the error of an amount
    # is taken as 1e5 eps times its magnitude sum; the precision is raised until that bound is below
    # 1e-14 of the amount or of the absolute floor max(1e-250 S, 1e-300) (S = sum(x0) + t sum(src)).
    rates = [[float(v) for v in row] for row in np.asarray(rates, dtype=float)]
    removal = [float(v) for v in removal]
    x0 = [float(v) for v in x0]
    src = [float(v) for v in src]
    order = _topological_order(rates)
    scale = sum(x0) + t * sum(src)
    dps = 30
    while True:
        vals, mags, eps = _exp_poly_amounts(order, rates, removal, x0, src, t, dps)
        with mp.workdps(dps):
            floor = max(mp.mpf(1e-250) * mp.mpf(scale), mp.mpf(1e-300))
            worst = mp.mpf(0)
            for v, m in zip(vals, mags):
                worst = max(worst, mp.mpf(1e5) * eps * m / max(abs(v), floor))
        if worst <= mp.mpf("1e-14"):
            return np.array([float(v) for v in vals])
        # raise the precision by the digits still missing, plus a margin of 10 digits
        dps = int(dps + max(mp.log10(worst / mp.mpf("1e-14")), 0) + 10)
        if dps > 40000:
            raise RuntimeError("no convergence in extended precision")


def chain_inventory(lam, n0, t):
    '''Atoms of every member of a linear radioactive decay chain after a time t (Bateman problem).

    Inputs:
      lam: 1-D array of n decay constants (1 <= n <= 30), member i decays into member i + 1 at the
           rate lam[i]; the last member decays out of the chain. 0 <= lam[i] <= 1e10 (per unit of time).
      n0: 1-D array of n initial numbers of atoms, nonnegative, sum(n0) <= 1e100.
      t: elapsed time, 0 <= t <= 1e20 (same unit of time).

    Output:
      x: numpy array of shape (n,), the numbers of atoms at time t. With S = sum(n0) and
         F = max(1e-250 * S, 1e-300): every entry whose exact value is at least F has a relative error
         below 1e-10; every other entry (exact zeros included) lies within F of the exact value.

    Raises:
      ValueError if lam and n0 are not 1-D arrays of the same length between 1 and 30, if an entry of
      lam is negative or above 1e10, if an entry of n0 is negative, if sum(n0) > 1e100, or if t is
      outside [0, 1e20].
    '''
    lam = np.asarray(lam, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    if lam.ndim != 1 or n0.shape != lam.shape or not 1 <= lam.size <= 30:
        raise ValueError("lam and n0 must be 1-D arrays of the same length between 1 and 30")
    if not (np.all(np.isfinite(lam)) and np.all((lam >= 0.0) & (lam <= 1e10))):
        raise ValueError("decay constants must lie in [0, 1e10]")
    if not (np.all(np.isfinite(n0)) and np.all(n0 >= 0.0)):
        raise ValueError("initial amounts must be nonnegative")
    if n0.sum() > 1e100:
        raise ValueError("sum(n0) must not exceed 1e100")
    t = float(t)
    if not (np.isfinite(t) and 0.0 <= t <= 1e20):
        raise ValueError("need 0 <= t <= 1e20")
    # Other method: the Bateman solution in its general (exponential-polynomial) form, built member by
    # member and evaluated in adaptively increased precision; equal decay constants are handled exactly.
    n = lam.size
    if t == 0.0:
        x = n0.copy()
        return x
    rates = np.zeros((n, n))
    rates[np.arange(1, n), np.arange(n - 1)] = lam[:-1]
    x = _exp_poly_apply(rates, lam, n0, np.zeros(n), t)
    return x


def network_inventory(lam, branching, source, n0, t):
    '''Atoms of every nuclide of a branching decay network with constant production after a time t.

    Inputs:
      lam: 1-D array of n decay constants (1 <= n <= 30), 0 <= lam[i] <= 1e10 (per unit of time).
      branching: (n, n) array, branching[j, i] = fraction of the decays of nuclide i that give nuclide j;
                 nonnegative, zero diagonal, column sums at most 1 + 1e-12 (the rest leaves the network;
                 a column whose sum lies in (1, 1 + 1e-12] is divided by its sum before use). The
                 directed graph with an edge i -> j wherever branching[j, i] > 0 must be acyclic.
      source: 1-D array of n constant production rates (atoms per unit of time), nonnegative.
      n0: 1-D array of n initial numbers of atoms, nonnegative.
      t: elapsed time, 0 <= t <= 1e20, with sum(n0) + t * sum(source) <= 1e100.

    Output:
      x: numpy array of shape (n,), the numbers of atoms at time t. With S = sum(n0) + t * sum(source)
         and F = max(1e-250 * S, 1e-300): every entry whose exact value is at least F has a relative error
         below 1e-10; every other entry (exact zeros included) lies within F of the exact value.

    Raises:
      ValueError if the arrays do not have these shapes (1 <= n <= 30), if an entry is negative, if a
      decay constant exceeds 1e10, if a diagonal entry of branching is nonzero or a column sum exceeds
      1 + 1e-12, if the network has a cycle, if t is outside [0, 1e20], or if
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
            raise ValueError("%s must be nonnegative" % name)
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
    # Other method: the generalized Bateman solution (sums of t**p exp(-a t) terms) built in topological
    # order, the constant production entering as a term with a = 0, in adaptively increased precision.
    if t == 0.0:
        x = n0.copy()
        return x
    # a column sum in (1, 1 + 1e-12] counts as 1: the transfer rates of that nuclide are rescaled to add up
    # to its decay constant
    x = _exp_poly_apply(_unit_columns(branching) * lam[None, :], lam, n0, source, t)
    return x


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
      x: numpy array of shape (n,), the numbers of atoms at the end of the history. With S = sum(n0) and
         F = max(1e-250 * S, 1e-300): every entry whose exact value is at least F has a relative error
         below 1e-10; every other entry (exact zeros included) lies within F of the exact value.

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
            raise ValueError("%s must be nonnegative" % name)
    if np.any(lam > 1e10) or np.any(sigma > 1e7) or np.any((sigma > 0.0) & (sigma < 1e-6)):
        raise ValueError("need lam <= 1e10 and sigma = 0 or 1e-6 <= sigma <= 1e7 b")
    if n0.sum() > 1e100:
        raise ValueError("sum(n0) must not exceed 1e100")
    if np.any(np.diag(branching) != 0.0) or np.any(branching.sum(axis=0) > 1.0 + 1e-12):
        raise ValueError("branching needs a zero diagonal and column sums at most 1 + 1e-12")
    steps = []
    # an ordered container is required: periods are applied in their given order
    if not isinstance(history, (list, tuple)):
        raise ValueError("history must be a list or tuple of (duration, flux) pairs")
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
    # Other method: each period is solved directly from the decay and capture rates (no branching
    # fractions) by the generalized Bateman solution in extended precision.
    x = n0.copy()
    for duration, flux in steps:
        if duration == 0.0:
            continue
        capture_rate = sigma * 1e-24 * flux
        rates = _unit_columns(branching) * lam[None, :] + capture * capture_rate[None, :]
        x = _exp_poly_apply(rates, lam + capture_rate, x, np.zeros(n), duration)
    return x


def monitor_flux(lam, branching, sigma, capture_to, n0, t_irr, t_cool, k, activity, flux_lo, flux_hi):
    '''Neutron flux from the measured activity of an activation product (flux-monitor analysis).

    Inputs:
      lam: 1-D array of n decay constants in 1/s (1 <= n <= 30), 0 <= lam[i] <= 1e10.
      branching: (n, n) array, branching[j, i] = fraction of the decays of nuclide i that give nuclide j;
                 nonnegative, zero diagonal, column sums at most 1 + 1e-12 (normalized as in step 2).
      sigma: 1-D array of n radiative-capture cross sections in barns, sigma[i] = 0 or 1e-6 <= sigma[i] <= 1e7.
      capture_to: 1-D integer array of n entries, the nuclide made by a capture on nuclide i, or -1 if
                  it is not followed; never i itself.
      n0: 1-D array of n initial numbers of atoms, nonnegative, sum(n0) <= 1e100.
      The combined decay and capture network must be acyclic, exactly as in activation_inventory
      (step 3).
      t_irr: irradiation time in s at a constant unknown flux, 0 < t_irr <= 1e9.
      t_cool: cooling time in s without flux after the irradiation, 0 <= t_cool <= 1e9.
      k: int, index of the measured nuclide, with lam[k] > 0.
      activity: measured activity lam[k] * x[k] of nuclide k at the end of the cooling, in Bq, > 0.
      flux_lo, flux_hi: the flux lies in [flux_lo, flux_hi], 1e-2 <= flux_lo < flux_hi <= 1e18, and the
           caller guarantees that the activity A(flux) is strictly increasing on this interval.

    Output:
      flux: Python float, the flux in neutrons / (cm^2 s) with A(flux) = activity, A computed with
            activation_inventory (step 3); relative error below 1e-8 wherever d ln A / d ln flux >= 0.5
            at the solution, for activities of any size within the domain.

    Raises:
      ValueError for nuclide data as in step 3, for times, k, activity or flux bounds outside these
      ranges, if A(flux_lo) < lam[k] * max(1e-250 * sum(n0), 1e-300)
      (at flux_lo the measured nuclide must hold at least 1e-250 of the initial atoms and at least 1e-300
      atoms; for x[k](flux_lo) within a relative 1e-10 of this bound, the accuracy of step 3,
      ValueError and a result are both correct), or if activity lies outside [A(flux_lo), A(flux_hi)]
      by more than a relative 1e-9.
      An activity with |ln(activity / A(end))| <= 1e-9 for an end of the interval, on either side of
      it, gives that end (flux_lo is checked first); the end is returned as the given bound itself, equal
      to it to a relative 1e-11. A is known to a relative 1e-10 (the accuracy of step 3), so the
      tolerance holds up to 1e-10 in ln: for 1e-9 - 1e-10 < |ln| <= 1e-9 + 1e-10 the end, the root and
      (outside the interval) ValueError are all correct.
    '''
    t_irr, t_cool, activity = float(t_irr), float(t_cool), float(activity)
    flux_lo, flux_hi = float(flux_lo), float(flux_hi)
    if not (np.isfinite(t_irr) and 0.0 < t_irr <= 1e9 and np.isfinite(t_cool) and 0.0 <= t_cool <= 1e9):
        raise ValueError("need 0 < t_irr <= 1e9 s and 0 <= t_cool <= 1e9 s")
    if not (np.isfinite(activity) and activity > 0.0):      # 0 is rejected too
        raise ValueError("activity must be positive")
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
        return float(np.log(lam_arr[k]) + np.log(x[k])) if x[k] > 0.0 else -np.inf

    target = np.log(activity)
    a, b = np.log(flux_lo), np.log(flux_hi)
    fa, fb = log_activity(a) - target, log_activity(b) - target
    x_lo = activation_inventory(lam, branching, sigma, capture_to, n0, [(t_irr, flux_lo), (t_cool, 0.0)])[k]
    s0 = float(np.sum(n0))
    # domain: x[k](flux_lo) >= max(1e-250 sum(n0), 1e-300), in logarithms (lam[k] times it may underflow)
    if not (s0 > 0.0 and x_lo > 0.0 and np.log(x_lo) >= max(np.log(s0) - 250.0 * np.log(10.0), -300.0 * np.log(10.0))):
        raise ValueError("A(flux_lo) is below lam[k] * max(1e-250 * sum(n0), 1e-300)")
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
