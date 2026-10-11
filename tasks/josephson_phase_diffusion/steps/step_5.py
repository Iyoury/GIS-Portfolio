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
        j_star is graded against the bias at which the exact array voltage equals v_crit,
        relative error below 1e-9 + 2e-9 / s with s = d ln(V) / d ln(J) of the exact array
        voltage at j_star (the relative error of mean_voltage, below 1e-9, enters j_star
        divided by s); r_diff and s_v, at the returned j_star, relative errors below 1e-8.

    Raises:
      ValueError if ics and rs are not 1-D arrays of the same nonzero length, if an entry
      is not finite and positive, if max(ics) / min(ics) > 5, if theta0 is not finite and
      positive or a ratio theta0 / c_k is outside [0.02, 50], or if v_crit is not in
      (0, 0.5 * sum(c_k * r_k)].
    '''
    raise NotImplementedError
