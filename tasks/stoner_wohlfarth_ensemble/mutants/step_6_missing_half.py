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
              tau_N), relative error below 1e-6.
        tau_int: integral relaxation time of z = cos(theta) divided by tau_N,
                 int_0^inf C(t) dt / C(0) with C(t) = <z(t) z(0)> - <z>**2 in equilibrium,
                 relative error below 1e-6.

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
    # MUTANT: time unit taken from dW/dt = d/dz[...] (factor 1/2 of Brown's equation dropped)
    lam, tau = 2.0 * lam, 0.5 * tau
    result = (float(lam), float(tau))
    return result
