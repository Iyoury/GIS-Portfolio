import numpy as np
import math


def prism_gradients(points, bounds, rho):
    """Gravity-gradient tensor T_ij = d g_i / d x_j of a right rectangular prism of uniform density.

    Inputs:
      points: float array of shape (3,) or (n, 3), observation points (m), not on the prism surface.
      bounds: (x1, x2, y1, y2, z1, z2) (m).
      rho: float, density (kg m^-3).

    Output:
      T: numpy array of shape (3, 3) (one point) or (n, 3, 3), in s^-2; each component within
        1e-9 ||T|| (Frobenius norm of the exact tensor), inside or outside at any distance. Each call within
        10 s for up to 1000 points.

    Raises:
      ValueError as in prism_gravity, and for a point on the surface of the prism.
    """
    raise NotImplementedError
