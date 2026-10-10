import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import least_squares


def column_gz(stations, x1, x2, y1, y2, depth, drho0, lam):
    """Vertical attraction of a column with density contrast drho0 exp(-lam z), 0 <= z <= depth.

    Inputs:
      stations: float array of shape (3,) or (n, 3), up to 1000 stations (x, y, z) in m with z <= 0.
      x1, x2, y1, y2: floats, the column x1 <= x <= x2, y1 <= y <= y2 (m), x1 < x2, y1 < y2.
      depth: float, 0 < depth <= 5e4 (m).
      drho0: float, density contrast at the surface (kg m^-3).
      lam: float, 0 <= lam <= 1e-2 (per m).

    Output:
      gz: z component of the attraction (m s^-2, positive down for drho0 > 0); a Python float for one
        station, a numpy array (n,) otherwise; relative error below 1e-10 at every station up to 1e7 m
        from the column.

    Raises:
      ValueError for stations of the wrong shape, non-finite or with z > 0, a non-finite parameter,
      x1 >= x2 or y1 >= y2, depth outside (0, 5e4] or lam outside [0, 1e-2].
    """
    raise NotImplementedError
