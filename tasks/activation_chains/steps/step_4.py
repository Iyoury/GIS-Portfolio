import numpy as np
from scipy.optimize import brentq
import mpmath as mp


def monitor_flux(lam, branching, sigma, capture_to, n0, t_irr, t_cool, k, activity, flux_lo, flux_hi):
    '''Neutron flux from the measured activity of an activation product (flux-monitor analysis).

    Inputs:
      lam, branching, sigma, capture_to, n0: the nuclide data and initial atoms, as in
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
      ranges, or if activity is not between A(flux_lo) and A(flux_hi).
    '''
    raise NotImplementedError
