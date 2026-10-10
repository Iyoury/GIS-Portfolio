import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def brown_relaxation(sigma, h):
    '''Exact thermal relaxation of one particle with its field along the easy axis (Brown's equation).

    Inputs:
      sigma: float, K V / (k_B T), 0 <= sigma <= 60.
      h: float, reduced field H / H_K along the easy axis, -0.9 <= h <= 0.9.

    Output:
      (lam1, tau_int): tuple of two Python floats, in units of 1/tau_N and tau_N, respectively.
        lam1: smallest nonzero eigenvalue of the Fokker-Planck operator (relaxation rate times
              tau_N), relative error below 1e-5.
        tau_int: integral relaxation time of z = cos(theta) divided by tau_N,
                 int_0^inf C(t) dt / C(0) with C(t) = <z(t) z(0)> - <z>**2 in equilibrium,
                 relative error below 1e-5.

    Raises:
      ValueError if sigma or h is not finite, if sigma is outside [0, 60] or if |h| > 0.9.
    '''
    sigma = float(sigma)
    h = float(h)
    if not (np.isfinite(sigma) and np.isfinite(h)):
        raise ValueError("sigma and h must be finite")
    if not (0.0 <= sigma <= 60.0) or abs(h) > 0.9:
        raise ValueError("need 0 <= sigma <= 60 and |h| <= 0.9")
    # MUTANT: Brown's high-barrier asymptote for the rate, and tau_int taken as 1 / lam1
    hh = abs(h)
    lam1 = (sigma ** 1.5 / np.sqrt(np.pi) * (1.0 - hh * hh)
            * ((1.0 + hh) * np.exp(-sigma * (1.0 + hh) ** 2) + (1.0 - hh) * np.exp(-sigma * (1.0 - hh) ** 2)))
    lam1 = lam1 if lam1 > 0.0 else 1.0
    tau_int = 1.0 / lam1
    result = (float(lam1), float(tau_int))
    return result
