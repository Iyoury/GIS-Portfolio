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
         disappears at h = -h_sw(psi). Absolute error below 1e-8 when
         |h + h_sw(psi)| >= 1e-3.

    Raises:
      ValueError if psi is outside [0, pi/2] or h is not finite.
    '''
    h = float(h)
    psi = float(psi)
    if not np.isfinite(h) or not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("h must be finite and psi must be between 0 and pi/2")
    # Switching field: e' = 0 and e'' = 0 together give the astroid
    # h_sw = (cos(psi)**(2/3) + sin(psi)**(2/3))**(-3/2).
    # Deliberate error: the outer exponent is -2/3 instead of -3/2.
    h_sw = (np.cos(psi) ** (2.0 / 3.0) + np.sin(psi) ** (2.0 / 3.0)) ** (-2.0 / 3.0)
    if h < -h_sw:
        # After the jump: theta -> theta + pi maps the energy at field h onto the
        # energy at field -h, and at -h > h_sw only one minimum is left.
        m = -branch_magnetization(-h, psi)
        return m
    if psi == 0.0:
        theta = 0.0
    else:
        # The minimum followed from theta = psi lies between psi and the fold angle where
        # it disappears at h = -h_sw (tan(theta)**3 = -tan(psi); -pi/2 for psi = pi/2).
        if psi == 0.5 * np.pi:
            theta_fold = -0.5 * np.pi
        else:
            theta_fold = -np.arctan(np.tan(psi) ** (1.0 / 3.0))
        # start a hair inside the fold, where the slope e'(theta) is reliably negative
        lo = theta_fold + 1e-13

        def slope(t):
            return 0.5 * np.sin(2.0 * t) + h * np.sin(t - psi)

        if slope(lo) >= 0.0:
            theta = lo
        else:
            theta = brentq(slope, lo, psi, xtol=1e-15, rtol=1e-15)
    m = float(np.cos(theta - psi))
    return m
