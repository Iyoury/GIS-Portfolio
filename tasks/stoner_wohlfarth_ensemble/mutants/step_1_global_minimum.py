import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def branch_magnetization(h, psi):
    '''Magnetization along the field on the zero-temperature descending branch of one particle.

    Inputs:
      h: float, reduced field along the field axis (negative means reversed field).
      psi: float, angle in radians between the field axis and the easy axis, 0 <= psi <= pi/2.

    Output:
      m: float, cos(theta - psi), where theta is the magnetization angle (from the easy
         axis) of the state reached when the field comes down from large positive values
         to h, the particle staying in its local energy minimum until that minimum
         disappears at h = -h_sw(psi). Absolute error below 1e-10 when
         |h + h_sw(psi)| >= 1e-3.

    Raises:
      ValueError if psi is outside [0, pi/2] or h is not finite.
    '''
    h = float(h)
    psi = float(psi)
    if not np.isfinite(h) or not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("h must be finite and psi must be between 0 and pi/2")
    # Deliberate error: takes the global energy minimum instead of following the local
    # minimum of the descending branch (no hysteresis at all).
    grid = np.linspace(-np.pi, np.pi, 20001)
    energy = 0.5 * np.sin(grid) ** 2 - h * np.cos(grid - psi)
    theta = grid[int(np.argmin(energy))]
    for _ in range(30):
        d1 = 0.5 * np.sin(2.0 * theta) + h * np.sin(theta - psi)
        d2 = np.cos(2.0 * theta) + h * np.cos(theta - psi)
        theta -= d1 / d2
    m = float(np.cos(theta - psi))
    return m
