import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def ensemble_switching(psis, weights, a, f0, rate):
    '''Dynamic coercive field and half-switching field of an ensemble during the sweep.

    Inputs:
      psis: 1-D array of easy-axis angles in radians, each in [0, pi/2].
      weights: 1-D array of the same length, nonnegative, not all zero (normalized by
               their sum).
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      (h_c, h_half): tuple of two floats, absolute errors below 1e-6.
        h_c: the ensemble magnetization is zero at h = -h_c during the sweep.
        h_half: half of the total weight has left its original minimum at h = -h_half.

    Raises:
      ValueError if psis and weights are not 1-D arrays of the same nonzero length, if
      a psi is outside [0, pi/2], if a weight is negative or not finite, if the weights
      add up to zero, if a, f0 or rate is not a positive finite number, if a is outside
      [40, 1000] or if f0 / rate is outside [1e5, 1e13].
    '''
    psis = np.asarray(psis, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if psis.ndim != 1 or weights.shape != psis.shape or psis.size == 0:
        raise ValueError("psis and weights must be 1-D arrays of the same nonzero length")
    if not np.all((psis >= 0.0) & (psis <= 0.5 * np.pi)):
        raise ValueError("every psi must be between 0 and pi/2")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("weights must be finite, nonnegative and not all zero")
    for name, value in (("a", a), ("f0", f0), ("rate", rate)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("%s must be a positive finite number" % name)
    if not (40.0 <= a <= 1000.0 and 1e5 <= f0 / rate <= 1e13):
        raise ValueError("need 40 <= a <= 1000 and 1e5 <= f0 / rate <= 1e13")
    w = weights / weights.sum()

    def ensemble_state(x):
        # expected ensemble magnetization and weight that has left, at field x
        magnetization, left = 0.0, 0.0
        for psi, wi in zip(psis, w):
            # Deliberate error: zero-temperature particles (no thermal escape before -h_sw).
            # cos(pi/2) is 6e-17 in floating point, which alone would put h_sw 2e-11 below 1
            h_sw = 1.0 if psi == 0.5 * np.pi else (np.cos(psi) ** (2.0 / 3.0) + np.sin(psi) ** (2.0 / 3.0)) ** -1.5
            P = 1.0 if x > -h_sw else 0.0
            # the other minimum is the original minimum of the field -x turned by pi
            m_other = -branch_magnetization(-x, psi)
            m_orig = branch_magnetization(x, psi) if P > 0.0 else m_other
            magnetization += wi * (P * m_orig + (1.0 - P) * m_other)
            left += wi * (1.0 - P)
        return magnetization, left

    # at h = 1 every particle is still in its original minimum (h_sw <= 1), at h = -1 none is
    h_c = -brentq(lambda x: ensemble_state(x)[0], -1.0, 1.0, xtol=1e-13, rtol=1e-15)
    h_half = -brentq(lambda x: ensemble_state(x)[1] - 0.5, -1.0, 1.0, xtol=1e-13, rtol=1e-15)
    result = (float(h_c), float(h_half))
    return result
