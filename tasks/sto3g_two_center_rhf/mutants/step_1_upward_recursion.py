import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def boys_function(n_max, t):
    '''Boys functions F_n(t) = integral from 0 to 1 of u**(2n) exp(-t u**2) du for n = 0..n_max.

    Inputs:
      n_max: int, highest order, 0 <= n_max <= 16.
      t: float or numpy array of floats (any shape), 0 <= t <= 1e6 (t = 0 is allowed).

    Output:
      F: float numpy array of shape np.shape(t) + (n_max + 1,), F[..., n] = F_n(t).
         Relative error below 1e-12 for every entry; F_n(0) = 1 / (2n + 1).

    Raises:
      ValueError if n_max is not an integer in [0, 16], or if any t is negative, not finite
      or larger than 1e6.
    '''
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)) or not (0 <= n_max <= 16):
        raise ValueError("n_max must be an integer from 0 to 16")
    n_max = int(n_max)
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0) or np.any(t_arr > 1e6):
        raise ValueError("t must be finite with 0 <= t <= 1e6")
    flat = t_arr.ravel()
    F = np.empty((flat.size, n_max + 1))
    # MUTANT: F_0 from erf and the upward recursion for every t (unstable for small t)
    split = -1.0
    small = flat < split
    # small t: series for the highest order, F_n(t) = exp(-t) sum_k (2t)^k / ((2n+1)(2n+3)...(2n+2k+1)),
    # then the downward recursion F_(m-1) = (2 t F_m + exp(-t)) / (2m - 1), which is stable
    ts = flat[small]
    if ts.size:
        term = 1.0 / (2 * n_max + 1) * np.ones_like(ts)
        total = term.copy()
        for k in range(1, 400):
            term = term * 2.0 * ts / (2 * n_max + 2 * k + 1)
            total += term
            if np.all(term <= 1e-17 * total):
                break
        e = np.exp(-ts)
        F[small, n_max] = e * total
        for m in range(n_max, 0, -1):
            F[small, m - 1] = (2.0 * ts * F[small, m] + e) / (2 * m - 1)
    # large t: F_0 from erf, then the upward recursion F_(m+1) = ((2m+1) F_m - exp(-t)) / (2t),
    # stable because t > n_max + 25
    tl = flat[~small]
    if tl.size:
        r = np.sqrt(tl)
        F[~small, 0] = 0.5 * np.sqrt(np.pi) * erf(r) / r
        e = np.exp(-tl)
        for m in range(n_max):
            F[~small, m + 1] = ((2 * m + 1) * F[~small, m] - e) / (2.0 * tl)
    F = F.reshape(t_arr.shape + (n_max + 1,))
    return F
