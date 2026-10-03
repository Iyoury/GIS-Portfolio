import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def survival_probability(h, psi, a, f0, rate):
    '''Probability that a particle has not left its original minimum when the sweep reaches h.

    Inputs:
      h: float, reduced field reached by the descending sweep.
      psi: float, easy-axis angle in radians, 0 <= psi <= pi/2.
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      P: float in [0, 1], P = 1 for h >= h_sw(psi) and P = 0 for h <= -h_sw(psi).
         Absolute error below 1e-10.

    Raises:
      ValueError if psi is outside [0, pi/2], if h is not finite, if a, f0 or rate is
      not a positive finite number, if a is outside [40, 1000] or if f0 / rate is
      outside [1e5, 1e13].
    '''
    h = float(h)
    psi = float(psi)
    if not np.isfinite(h) or not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("h must be finite and psi must be between 0 and pi/2")
    for name, value in (("a", a), ("f0", f0), ("rate", rate)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("%s must be a positive finite number" % name)
    if not (40.0 <= a <= 1000.0 and 1e5 <= f0 / rate <= 1e13):
        raise ValueError("need 40 <= a <= 1000 and 1e5 <= f0 / rate <= 1e13")
    # cos(pi/2) is 6e-17 in floating point, which alone would put h_sw 2e-11 below 1
    h_sw = 1.0 if psi == 0.5 * np.pi else (np.cos(psi) ** (2.0 / 3.0) + np.sin(psi) ** (2.0 / 3.0)) ** -1.5
    if h >= h_sw:
        P = 1.0           # only the original minimum exists: no escape so far
        return P
    if h <= -h_sw:
        P = 0.0           # the original minimum has disappeared
        return P

    def rate_over_f0(x):
        # both escape routes; dE / (k_B T) = 2 K V Delta_e / (k_B T) = 2 a Delta_e
        # quadrature nodes can round onto +-h_sw, where the barriers take their limits
        x = float(np.clip(x, -np.nextafter(h_sw, 0.0), np.nextafter(h_sw, 0.0)))
        low, high = escape_barriers(x, psi)
        # Deliberate error: only the route over the lower barrier is counted.
        return float(np.exp(-2.0 * a * low))

    # P = exp(-int Gamma dt) and dt = dh' / rate over the part of the sweep with two minima.
    # Where P matters the integral is tiny (of order rate / f0), so it needs a purely
    # relative tolerance.
    integral = quad(rate_over_f0, h, h_sw, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    P = float(np.exp(-f0 / rate * integral))
    return P
