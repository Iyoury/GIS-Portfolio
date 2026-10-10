"""Step 3 - present-day subsurface signature of a past surface-temperature history (half-space)."""
import numpy as np
import math

SECONDS_PER_YEAR = 365.25 * 86400.0   # Julian year


def paleoclimate_perturbation(z, t_years, dT, kappa):
    """Present-day temperature perturbation (K) at depth z.

    History convention: interval 0 runs from the present back to t_years[0],
    interval i runs from t_years[i-1] back to t_years[i]. During interval i
    the ground-surface temperature differed from the long-term reference
    surface temperature by dT[i] (K). Before t_years[-1] the surface stayed
    at the reference temperature long enough for the ground to be in steady
    state. The subsurface is a homogeneous conductive half-space with thermal
    diffusivity kappa. Times in years are converted with SECONDS_PER_YEAR.

    Parameters
    ----------
    z : float or array_like, depth (m) >= 0.
    t_years : 1-D array_like, positive, strictly increasing ends of the
        history intervals (years before present).
    dT : 1-D array_like, same length, departure during each interval (K).
    kappa : float, thermal diffusivity (m^2 s^-1), > 0.

    Returns
    -------
    Python float for scalar z, else a numpy float array with the shape of z.
    Absolute accuracy 1e-6 K per kelvin of the largest |dT|.

    Raises
    ------
    ValueError if t_years and dT are not 1-D of equal length, if they
    contain a non-finite value, if t_years[0] <= 0 or t_years does not
    increase strictly, if kappa is non-finite or <= 0, or if any depth is
    non-finite or negative.
    """
    raise NotImplementedError
