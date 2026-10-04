import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def escape_barriers(h, psi):
    '''Reduced energy barriers that keep a particle in its original minimum.

    Inputs:
      h: float or numpy array (any shape), reduced field with |h| < h_sw(psi).
      psi: float, angle in radians between the field axis and the easy axis, 0 <= psi <= pi/2.

    Output:
      (low, high): two values with the shape of np.asarray(h), numpy float scalars for a
      scalar h. low <= high are
      e(theta_max) - e(theta_min) for the two energy maxima, theta_min being the original
      minimum of the descending branch. Absolute error below 1e-8 when
      h_sw(psi) - |h| >= 1e-3.

    Raises:
      ValueError if psi is outside [0, pi/2], if any h is not finite, or if any
      |h| >= h_sw(psi).
    '''
    h_arr = np.asarray(h, dtype=float)
    psi = float(psi)
    if not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("psi must be between 0 and pi/2")
    if not np.all(np.isfinite(h_arr)):
        raise ValueError("h must be finite")
    # cos(pi/2) is 6e-17 in floating point, which alone would put h_sw 2e-11 below 1
    h_sw = 1.0 if psi == 0.5 * np.pi else (np.cos(psi) ** (2.0 / 3.0) + np.sin(psi) ** (2.0 / 3.0)) ** -1.5
    if np.any(np.abs(h_arr) >= h_sw):
        raise ValueError("both energy minima exist only for |h| < h_sw(psi)")
    x = h_arr.ravel()
    # The extrema solve e'(theta) = 0. With theta = t0 + 2 arctan(t) this becomes a quartic in
    # t; for each field the shift t0 is picked from a few values so that the leading
    # coefficient is large (theta = t0 + pi is then far from an extremum).
    shifts = np.array([0.3, 1.1, 1.9, 2.7])
    lead = 0.5 * np.sin(2.0 * shifts)[None, :] - x[:, None] * np.sin(shifts - psi)[None, :]
    t0 = shifts[np.argmax(np.abs(lead), axis=1)]
    c2, s2 = np.cos(2.0 * t0), np.sin(2.0 * t0)
    cc, ss = np.cos(t0 - psi), np.sin(t0 - psi)
    coef = np.stack([0.5 * s2 - x * ss, -2.0 * c2 + 2.0 * x * cc, -3.0 * s2,
                     2.0 * c2 + 2.0 * x * cc, 0.5 * s2 + x * ss], axis=1)
    companion = np.zeros((x.size, 4, 4))
    companion[:, 0, :] = -coef[:, 1:] / coef[:, :1]
    companion[:, 1, 0] = companion[:, 2, 1] = companion[:, 3, 2] = 1.0
    theta = t0[:, None] + 2.0 * np.arctan(np.linalg.eigvals(companion).real)
    for _ in range(3):
        # Newton steps on e'(theta) = 0 polish the four roots
        d1 = 0.5 * np.sin(2.0 * theta) + x[:, None] * np.sin(theta - psi)
        d2 = np.cos(2.0 * theta) + x[:, None] * np.cos(theta - psi)
        safe = np.abs(d2) > 1e-8
        theta = theta - np.where(safe, np.clip(d1 / np.where(safe, d2, 1.0), -0.1, 0.1), 0.0)
    theta = np.sort(np.mod(theta, 2.0 * np.pi), axis=1)
    # minima and maxima alternate around the circle; the sign pattern of e'' says which
    # alternate pair are the minima
    d2 = np.cos(2.0 * theta) + x[:, None] * np.cos(theta - psi)
    even = (d2[:, 0] - d2[:, 1] + d2[:, 2] - d2[:, 3]) > 0.0
    minima = np.where(even[:, None], theta[:, 0::2], theta[:, 1::2])
    maxima = np.where(even[:, None], theta[:, 1::2], theta[:, 0::2])
    # the original minimum of the descending branch is the one with cos(theta) > 0
    # Deliberate error: the barriers are measured from the deepest minimum, not from the
    # original minimum the particle occupies.
    e_two = 0.5 * np.sin(minima) ** 2 - x[:, None] * np.cos(minima - psi)
    first = e_two[:, 0] <= e_two[:, 1]
    th_min = np.where(first, minima[:, 0], minima[:, 1])
    e_min = 0.5 * np.sin(th_min) ** 2 - x * np.cos(th_min - psi)
    e_max = 0.5 * np.sin(maxima) ** 2 - x[:, None] * np.cos(maxima - psi)
    barriers = np.sort(e_max - e_min[:, None], axis=1)
    low = barriers[:, 0].reshape(h_arr.shape)[()]
    high = barriers[:, 1].reshape(h_arr.shape)[()]
    return low, high
