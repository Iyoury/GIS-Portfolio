import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def array_criterion(ics, rs, theta0, v_crit):
    '''Criterion current, differential resistance and voltage noise of a series junction array.

    Inputs:
      ics: 1-D array of critical currents c_k in units of I_0, finite, > 0,
           max(ics) / min(ics) <= 5.
      rs: 1-D array of the same length, normal resistances r_k in units of R_0, finite, > 0.
      theta0: float, k_B T / E_J0 with E_J0 = hbar I_0 / (2 e); every theta0 / c_k must
              lie in [0.02, 50].
      v_crit: float, voltage criterion in units of I_0 R_0,
              0 < v_crit <= 0.5 * sum(c_k * r_k).

    Output:
      (j_star, r_diff, s_v): tuple of three Python floats.
        j_star: bias current (units of I_0) at which the mean array voltage equals v_crit.
        r_diff: differential resistance dV/dJ of the array at j_star (units of R_0).
        s_v: lim_{t->inf} Var(int_0^t V dt') / t at j_star, the zero-frequency (two-sided)
             voltage noise, in units of (hbar / 2e) I_0 R_0.
        Relative errors below 1e-9 (j_star) and 1e-8 (r_diff, s_v).

    Raises:
      ValueError if ics and rs are not 1-D arrays of the same nonzero length, if an entry
      is not finite and positive, if max(ics) / min(ics) > 5, if theta0 is not finite and
      positive or a ratio theta0 / c_k is outside [0.02, 50], or if v_crit is not in
      (0, 0.5 * sum(c_k * r_k)].
    '''
    c = np.asarray(ics, dtype=float)
    r = np.asarray(rs, dtype=float)
    if c.ndim != 1 or r.shape != c.shape or c.size == 0:
        raise ValueError("ics and rs must be 1-D arrays of the same nonzero length")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r)) and np.all(c > 0.0) and np.all(r > 0.0)):
        raise ValueError("critical currents and resistances must be finite and positive")
    if c.max() / c.min() > 5.0:
        raise ValueError("max(ics) / min(ics) must not exceed 5")
    theta0 = float(theta0)
    if not (np.isfinite(theta0) and theta0 > 0.0):
        raise ValueError("theta0 must be finite and positive")
    th = theta0 / c                                   # E_J is proportional to I_c
    if np.any(th < 0.02) or np.any(th > 50.0):
        raise ValueError("every theta0 / c_k must lie in [0.02, 50]")
    v_crit = float(v_crit)
    cr = c * r
    if not (np.isfinite(v_crit) and 0.0 < v_crit <= 0.5 * cr.sum()):
        raise ValueError("v_crit must be in (0, 0.5 sum(c_k r_k)]")

    # Junction k: i_k = J / c_k, V_k = I_ck R_k v(i_k, theta_k). The array voltage rises
    # monotonically from 0 at J = 0; at J = 2 max(c) every i_k >= 2 and V >= sqrt(3) sum(c r).
    def excess(j):
        return sum(ck * rk * mean_voltage(j / ck, tk)[0] for ck, rk, tk in zip(c, r, th)) - v_crit

    j_star = brentq(excess, 0.0, 2.0 * c.max(), xtol=1e-15, rtol=1e-14)
    r_diff = 0.0
    s_v = 0.0
    for ck, rk, tk in zip(c, r, th):
        r_diff += rk * mean_voltage(j_star / ck, tk)[1]
        # independent phases; reduced time of junction k runs at omega_k = 2 e I_ck R_k / hbar
        s_v += 2.0 * ck * rk * effective_diffusion(j_star / ck, tk)
    result = (float(j_star), float(r_diff), float(s_v))
    return result
