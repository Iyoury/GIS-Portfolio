import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def monitor_flux(lam, branching, sigma, capture_to, n0, t_irr, t_cool, k, activity, flux_lo, flux_hi):
    '''Neutron flux from the measured activity of an activation product (flux-monitor analysis).

    Inputs:
      lam: 1-D array of n decay constants in 1/s (1 <= n <= 30), 0 <= lam[i] <= 1e10.
      branching: (n, n) array, branching[j, i] = fraction of the decays of nuclide i that give nuclide j;
                 nonnegative, zero diagonal, column sums at most 1 + 1e-12.
      sigma: 1-D array of n radiative-capture cross sections in barns, 0 <= sigma[i] <= 1e7.
      capture_to: 1-D integer array of n entries, the nuclide made by a capture on nuclide i, or -1 if
                  it is not followed; never i itself.
      n0: 1-D array of n initial numbers of atoms, nonnegative, sum(n0) <= 1e100.
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
