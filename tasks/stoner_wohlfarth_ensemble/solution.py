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
    # cos(pi/2) is 6e-17 in floating point, which alone would put h_sw 2e-11 below 1
    h_sw = 1.0 if psi == 0.5 * np.pi else (np.cos(psi) ** (2.0 / 3.0) + np.sin(psi) ** (2.0 / 3.0)) ** -1.5
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


def escape_barriers(h, psi):
    '''Reduced energy barriers that keep a particle in its original minimum.

    Inputs:
      h: float or numpy array (any shape), reduced field with |h| < h_sw(psi).
      psi: float, angle in radians between the field axis and the easy axis, 0 <= psi <= pi/2.

    Output:
      (low, high): two values with the shape of np.asarray(h): float numpy arrays, or float
      scalars (Python float or numpy floating) for a scalar h. low <= high are
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
    first = np.cos(minima[:, 0]) >= np.cos(minima[:, 1])
    th_min = np.where(first, minima[:, 0], minima[:, 1])
    e_min = 0.5 * np.sin(th_min) ** 2 - x * np.cos(th_min - psi)
    e_max = 0.5 * np.sin(maxima) ** 2 - x[:, None] * np.cos(maxima - psi)
    barriers = np.sort(e_max - e_min[:, None], axis=1)
    low = barriers[:, 0].reshape(h_arr.shape)[()]
    high = barriers[:, 1].reshape(h_arr.shape)[()]
    return low, high


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
         Absolute error below 1e-8; for 0 < psi < pi/2 not required at fields less than
         1e-6 below h_sw(psi).

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
        return float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))

    # P = exp(-int Gamma dt) and dt = dh' / rate over the part of the sweep with two minima.
    # Where P matters the integral is tiny (of order rate / f0), so it needs a purely
    # relative tolerance.
    integral = quad(rate_over_f0, h, h_sw, epsabs=0.0, epsrel=1e-12, limit=1000)[0]
    P = float(np.exp(-f0 / rate * integral))
    return P


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
    # piecewise Chebyshev representation in theta (z = cos theta); every panel keeps its own
    # relative accuracy, so the exponentially different well weights are resolved
    C = np.polynomial.chebyshev
    h = abs(h)                     # the problem is symmetric under z -> -z, h -> -h
    K, d, iters = int(30 + 1.5 * sigma), 24, 300
    edges = np.linspace(0.0, np.pi, K + 1)
    xk = np.cos(np.pi * (np.arange(d + 1) + 0.5) / (d + 1))[::-1]      # Chebyshev points in [-1, 1]
    a, b = edges[:-1, None], edges[1:, None]
    TH = 0.5 * (a + b) + 0.5 * (b - a) * xk[None, :]                   # (K, d+1)
    half = 0.5 * (b - a)[:, 0]
    V = C.chebvander(xk, d)
    Vinv = np.linalg.inv(V)
    shift = sigma + 2 * sigma * abs(h)
    Z = np.cos(TH)
    lw = sigma * Z * Z + 2 * sigma * h * Z - shift          # log w
    W = np.exp(lw)
    S = np.sin(TH)

    def cum_from_right(vals):
        # int_theta^{pi} f dtheta at every node
        coef = vals @ Vinv.T                                  # (K, d+1)
        out = np.empty_like(vals)
        acc = 0.0
        for k in range(K - 1, -1, -1):
            ci = C.chebint(coef[k], lbnd=1) * half[k]        # = -int_x^1 on the panel
            out[k] = acc - C.chebval(xk, ci)
            acc = acc + (-C.chebval(-1.0, ci))
        return out, acc

    def cum_from_left(vals):
        coef = vals @ Vinv.T
        out = np.empty_like(vals)
        acc = 0.0
        for k in range(K):
            ci = C.chebint(coef[k], lbnd=-1) * half[k]
            out[k] = acc + C.chebval(xk, ci)
            acc = acc + C.chebval(1.0, ci)
        return out, acc

    def total(vals):
        return cum_from_left(vals)[1]

    # barrier at z = -h, theta_b = arccos(-h): the flux int_{-1}^{z} w f dz of a function f with
    # int w f dz = 0 is taken from the nearer end on each side of the barrier (from z = -1 on the
    # side of the shallow well, as minus the integral from z = +1 on the side of the deep one),
    # which avoids the cancellation of the two well masses
    shallow = TH > np.arccos(-h)

    def flux(vals):
        right, _ = cum_from_right(vals)
        left, _ = cum_from_left(vals)
        return np.where(shallow, right, -left)

    def project(g):
        return g - total(W * g * S) / total(W * S)

    # start from a step at the barrier (z = -h): the slow mode of an asymmetric double well lives
    # mostly in the shallow well, whose Boltzmann weight can be far below rounding relative to the
    # deep one, so a smooth start such as g = z has no usable component along it
    g = project(np.where(Z < -h, 1.0, 0.0) + 1e-3 * Z)
    # inverse iteration on the Sturm-Liouville form: solve d/dz[(1 - z^2) w dg_new/dz] = -2 w g
    # by two cumulative integrals, with the Rayleigh quotient of the new iterate as lam1
    lam_old = None
    for it in range(iters):
        F = flux(W * g * S)                                  # int_{-1}^{z} w g dz
        q = 2.0 * F / (S * W)
        gn, _ = cum_from_left(q)
        sc = np.max(np.abs(gn))
        gn = project(gn / sc)
        num = 2.0 * total(F * F / (S * W)) / sc ** 2
        den = total(W * gn * gn * S)
        lam = num / den
        g = gn
        if lam_old is not None and abs(lam - lam_old) <= 1e-15 * abs(lam):
            break
        lam_old = lam
    # integral relaxation time (Garanin): with Q(z) = int_{-1}^{z} (z' - <z>) w dz',
    # tau = 2 int Q^2 / ((1 - z^2) w) dz / int (z - <z>)^2 w dz
    zbar = total(W * Z * S) / total(W * S)
    # Q is divided by w, which is smallest at the barrier and in the shallow well: there the
    # integral from z = -1 is a sum over the shallow side only and is accurate
    Q, _ = cum_from_right(W * (Z - zbar) * S)
    tau = 2.0 * total(Q * Q / (S * W)) / total(W * (Z - zbar) ** 2 * S)
    result = (float(lam), float(tau))
    return result


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
            P = survival_probability(x, psi, a, f0, rate)
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
