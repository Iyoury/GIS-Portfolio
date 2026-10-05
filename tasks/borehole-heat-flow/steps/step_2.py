"""Step 2 - Bullard thermal resistance and heat-production integral."""
import numpy as np


def layer_integrals(layer_top_tvd, layer_k, z):
    """Depth functions R(z), S(z) of the steady layered geotherm T = T0 + q0 R - A S.

    Parameters
    ----------
    layer_top_tvd : 1-D array_like, depth (m) of the top of each layer; the
        first is 0, strictly increasing. The last layer extends to infinity.
    layer_k : 1-D array_like, thermal conductivity of each layer (W m^-1 K^-1), > 0.
    z : float or array_like, depths (m) >= 0.

    Returns
    -------
    (R, S) : R in m^2 K W^-1, S in m^3 K W^-1; floats for scalar z, otherwise
    arrays with the shape of z. Relative accuracy 1e-9.

    Raises
    ------
    ValueError if the layer arrays are not 1-D of equal length, if the first
    top is not 0 or tops do not increase strictly, if any conductivity is
    non-finite or <= 0, or if any depth is non-finite or negative.
    """
    raise NotImplementedError
