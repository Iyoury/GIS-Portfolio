"""Step 5 - present-day signature of a past surface-temperature history in a layered column."""
import numpy as np
import math

SECONDS_PER_YEAR = 365.25 * 86400.0   # Julian year


def layered_paleoclimate_perturbation(z, layer_tops, layer_k, rho_c, t_years, dT):
    """Present-day temperature perturbation (K) at depth z in a layered column.

    History convention as in paleoclimate_perturbation (step 3). The ground
    is a column of horizontal layers (tops layer_tops, the first at 0, the
    last layer extending to infinite depth) with conductivities layer_k and a
    uniform volumetric heat capacity rho_c; temperature and heat flux are
    continuous at the layer tops.

    Parameters
    ----------
    z : float or array_like, depth (m) >= 0.
    layer_tops : 1-D array_like, layer tops (m), first 0, strictly increasing.
    layer_k : 1-D array_like, same length, conductivities (W m^-1 K^-1) > 0.
    rho_c : float, volumetric heat capacity (J m^-3 K^-1) > 0.
    t_years : 1-D array_like, positive, strictly increasing ends of the
        history intervals (years before present).
    dT : 1-D array_like, same length, departure during each interval (K),
        not all zero.

    Returns
    -------
    Python float for scalar z, else a numpy float array with the shape of z.
    Absolute accuracy 1e-8 K per kelvin of the largest |dT|; up to 1000
    depths, 50 layers and 20 history intervals per call.

    Raises
    ------
    ValueError if any conductivity is <= 0 or rho_c <= 0.
    """
    raise NotImplementedError
