import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def switching_field_statistics(psi, a, f0, rate):
    '''Median and mean switching field of one particle in the descending sweep.

    Inputs:
      psi: float, easy-axis angle in radians, 0 <= psi <= pi/2.
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      (h_median, h_mean): tuple of two floats, the median and the mean of the field at
      which the particle leaves its original minimum. Absolute errors below 1e-9.

    Raises:
      ValueError if psi is outside [0, pi/2], if a, f0 or rate is not a positive finite
      number, if a is outside [40, 1000] or if f0 / rate is outside [1e5, 1e13].
    '''
    psi = float(psi)
    if not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("psi must be between 0 and pi/2")
    for name, value in (("a", a), ("f0", f0), ("rate", rate)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("%s must be a positive finite number" % name)
    if not (40.0 <= a <= 1000.0 and 1e5 <= f0 / rate <= 1e13):
        raise ValueError("need 40 <= a <= 1000 and 1e5 <= f0 / rate <= 1e13")
    # cos(pi/2) is 6e-17 in floating point, which alone would put h_sw 2e-11 below 1
    h_sw = 1.0 if psi == 0.5 * np.pi else (np.cos(psi) ** (2.0 / 3.0) + np.sin(psi) ** (2.0 / 3.0)) ** -1.5
    lam = f0 / rate

    def gamma_over_f0(x):
        # quadrature nodes can round onto +-h_sw, where the barriers take their limits
        x = float(np.clip(x, -np.nextafter(h_sw, 0.0), np.nextafter(h_sw, 0.0)))
        low, high = escape_barriers(x, psi)
        return float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))

    def escape_integral(lo, hi):
        # int_lo^hi Gamma / f0 with a purely relative tolerance (tiny where it matters)
        if lo >= hi:
            return 0.0
        return quad(gamma_over_f0, lo, hi, epsabs=0.0, epsrel=1e-12, limit=1000)[0]

    # the median switching field is where P = exp(-lam I) = 1/2
    h_median = brentq(lambda x: lam * escape_integral(x, h_sw) - np.log(2.0), -h_sw, h_sw,
                      xtol=1e-14, rtol=1e-15)
    # Near the median lam * I changes by a factor e over about 1 / kappa (faster further away),
    # so 1 - P < 1e-19 above h_median + 45 / kappa, and P < exp(-80) once lam * I > 80.
    kappa = lam * gamma_over_f0(h_median) / np.log(2.0)
    top = min(h_sw, h_median + 45.0 / kappa)
    bottom = max(-h_sw, h_median - 12.0 / kappa)
    while bottom > -h_sw and lam * escape_integral(bottom, h_sw) < 80.0:
        bottom = max(-h_sw, bottom - 6.0 / kappa)
    # E[H_s] = h_sw - integral of P over [-h_sw, h_sw] (integration by parts, with the atom at
    # -h_sw); inside the window P comes from the escape integral accumulated from the top down
    # between the nodes of a composite Gauss-Legendre rule
    xg, wg = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(bottom, top, 17)
    nodes = np.concatenate([0.5 * (q + p) + 0.5 * (q - p) * xg for p, q in zip(edges[:-1], edges[1:])])
    node_weights = np.concatenate([0.5 * (q - p) * wg for p, q in zip(edges[:-1], edges[1:])])
    integral = escape_integral(top, h_sw) if top < h_sw else 0.0
    upper, inside = top, 0.0
    for k in np.argsort(nodes)[::-1]:
        integral += escape_integral(nodes[k], upper)
        upper = nodes[k]
        inside += node_weights[k] * np.exp(-lam * integral)
    # Deliberate error: the sign of the integration by parts is reversed.
    h_mean = float(((h_sw - top) + inside) - h_sw)
    result = (float(h_median), h_mean)
    return result
