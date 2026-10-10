import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import least_squares


def invert_basin_depths(x_edges, y_edges, gz_obs, drho0, lam):
    """Depths of a basin of columns (density contrast drho0 exp(-lam z)) from its gravity anomaly.

    Inputs:
      x_edges, y_edges: 1-D float arrays (nx + 1 and ny + 1 entries), strictly increasing, nx ny <= 100.
      gz_obs: float array (nx, ny), the vertical attraction (m s^-2) of all columns at the cell centres on
        the surface z = 0; it is that of a basin whose every depth satisfies 0 < depth <= 3 w_min (w_min
        the smallest cell width, in x or y), depth <= 5e4 m and lam depth <= 2.
      drho0: float, nonzero density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      depths: numpy array (nx, ny) in m, each with a relative error below 1e-8.

    Raises:
      ValueError for invalid edges, nx ny > 100, gz_obs of the wrong shape or non-finite, drho0 zero or
      non-finite, lam outside [0, 1e-2], an observation without the sign of drho0, or (lam > 0) an
      observation of at least 2 pi G |drho0| / lam.
    """
    raise NotImplementedError
