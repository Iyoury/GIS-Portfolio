import numpy as np
import math


def prism_gravity(points, bounds, rho):
    """Potential and attraction of a right rectangular prism of uniform density.

    Inputs:
      points: float array of shape (3,) or (n, 3), observation points (x, y, z) in m, z positive down.
      bounds: (x1, x2, y1, y2, z1, z2), the prism x1 <= x <= x2, y1 <= y <= y2, z1 <= z <= z2 (m).
      rho: float, density (kg m^-3).

    Output:
      (U, g): U = G rho int dV / |Q - P| (m^2 s^-2) and g = grad U (m s^-2, toward the mass for rho > 0,
        z component positive down). Python float and array (3,) for one point, arrays (n,) and (n, 3)
        otherwise. U with a relative error below 1e-10; each component of g within
        1e-10 |g| + 1e-14 G |rho| L (L the longest edge), at every point, inside, on the surface or
        outside at any distance. Each call within 10 s for up to 1000 points.

    Raises:
      ValueError for invalid bounds (not six finite numbers with x1 < x2, y1 < y2, z1 < z2), a
      non-finite rho, or points of the wrong shape or with a non-finite value.
    """
    raise NotImplementedError
