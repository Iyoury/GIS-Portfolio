import numpy as np
import math


def invert_basin_depths(x_edges, y_edges, gz_obs, drho0, lam):
    """Depths of a basin of columns (density contrast drho0 exp(-lam z)) from its gravity anomaly.

    Inputs:
      x_edges, y_edges: 1-D float arrays (nx + 1 and ny + 1 entries), strictly increasing, nx ny <= 100.
      gz_obs: float array (nx, ny), the vertical attraction (m s^-2) of all columns at the cell centres on
        the surface z = 0; it is that of a basin with every depth in (0, 5e4] m.
      drho0: float, nonzero density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      depths: numpy array (nx, ny) in m, each with a relative error below 1e-8. Each call within 30 s.

    Raises:
      ValueError for invalid edges, nx ny > 100, gz_obs of the wrong shape or non-finite, drho0 zero or
      non-finite, lam outside [0, 1e-2], an observation without the sign of drho0, or (lam > 0) an
      observation of at least 2 pi G |drho0| / lam.
    """
    raise NotImplementedError
