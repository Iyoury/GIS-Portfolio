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
      which the particle leaves its original minimum. Absolute errors below 1e-7.

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
    inside = np.nextafter(h_sw, 0.0)
    xg, wg = np.polynomial.legendre.leggauss(16)

    def gamma_over_f0(x):
        # escape rate over f0 at the field(s) x, both routes; nodes that round onto +-h_sw are moved
        # one ulp inside, where the barriers take their limits
        low, high = escape_barriers(np.clip(x, -inside, inside), psi)
        return np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high)

    def escape_integral(lo, hi):
        # int_lo^hi Gamma / f0 with a purely relative tolerance (tiny where it matters); an interval
        # shorter than 1e-9 (next to +-h_sw) by one 16-point Gauss-Legendre rule
        if lo >= hi:
            return 0.0
        if hi - lo < 1e-9:
            return float(0.5 * (hi - lo) * np.dot(wg, gamma_over_f0(0.5 * (lo + hi) + 0.5 * (hi - lo) * xg)))
        return quad(lambda x: float(gamma_over_f0(x)), lo, hi, epsabs=0.0, epsrel=1e-12, limit=1000)[0]

    # the median switching field is where P = exp(-lam I) = 1/2
    h_median = brentq(lambda x: lam * escape_integral(x, h_sw) - np.log(2.0), -h_sw, h_sw,
                      xtol=1e-14, rtol=1e-15)

    # E[H_s] = h_sw - integral of P over [-h_sw, h_sw] (integration by parts, with the atom at -h_sw).
    # P need not fall in one window: for psi above about pi/4 the lower barrier has a second local
    # minimum at a positive field (at +h_sw itself close to the hard axis, where the other minimum
    # forms next to the original one), so the escape rate peaks there as well as towards -h_sw and the
    # switching field can be bimodal. The integral is therefore taken by adaptive composite
    # Gauss-Legendre quadrature in h. U = lam I is accumulated from h_sw downwards with quad at the
    # panel edges: h = 0 (both barriers 1/2), a window around the median, edges graded towards both
    # ends +-h_sw and a coarse uniform set. Inside a panel, U at the 16 nodes comes from its upper edge
    # by a Gauss-Legendre rule on each gap between nodes. A panel is halved until halving changes its
    # integral of P by at most 1e-12 and the gap rules reproduce the quad values at the edges (in units
    # of P); a panel on which U changes by at most 1e-13 has P constant, and nothing is added below
    # the field where U > 100 (P < 4e-44).
    def panel(p, q, u_q):
        # Gauss-Legendre integral of P over [p, q] from U at q, and U at p from the gap rules
        nodes = 0.5 * (p + q) + 0.5 * (q - p) * xg
        pts = np.concatenate([[p], nodes, [q]])
        lo, hi = pts[:-1], pts[1:]
        t = 0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * xg[None, :]
        gaps = 0.5 * (hi - lo) * (gamma_over_f0(t) @ wg)
        u = u_q + lam * np.cumsum(gaps[::-1])[::-1]
        return 0.5 * (q - p) * np.dot(wg, np.exp(-u[1:])), u[0]

    kappa = lam * float(gamma_over_f0(h_median)) / np.log(2.0)
    graded = h_sw * (1.0 - 10.0 ** -np.arange(1.0, 16.0))
    edges = np.concatenate([[h_sw, 0.0], graded, -graded, np.linspace(-h_sw, h_sw, 9)[1:-1],
                            np.clip(h_median + np.linspace(-12.0, 45.0, 17) / kappa, -h_sw, h_sw)])
    edges = np.unique(edges)[::-1]
    u_edges = [0.0]
    for p, q in zip(edges[1:], edges[:-1]):
        if u_edges[-1] > 100.0:
            break
        u_edges.append(u_edges[-1] + lam * escape_integral(p, q))
    integral = 0.0
    for j in range(1, len(u_edges)):
        p, q, u_p, u_q = edges[j], edges[j - 1], u_edges[j], u_edges[j - 1]
        if u_q > 100.0:
            break
        if u_p - u_q <= 1e-13:
            integral += (q - p) * np.exp(-0.5 * (u_p + u_q))
            continue
        stack = [(p, q, u_p, u_q, panel(p, q, u_q)[0], 0)]
        while stack:
            p, q, u_p, u_q, whole, depth = stack.pop()
            m = 0.5 * (p + q)
            u_m = u_q + lam * escape_integral(m, q)
            left, u_p_gauss = panel(p, m, u_m)
            right, u_m_gauss = panel(m, q, u_q)
            if depth >= 40 or (abs(left + right - whole) <= 1e-12
                               and abs(u_p_gauss - u_p) * np.exp(-u_m) <= 1e-12
                               and abs(u_m_gauss - u_m) * np.exp(-u_q) <= 1e-12):
                integral += left + right
            else:
                stack.append((p, m, u_p, u_m, left, depth + 1))
                stack.append((m, q, u_m, u_q, right, depth + 1))
    h_mean = float(h_sw - integral)
    result = (float(h_median), h_mean)
    return result
