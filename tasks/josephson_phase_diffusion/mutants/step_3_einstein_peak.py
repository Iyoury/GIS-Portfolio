import numpy as np
import mpmath as mp
from scipy.optimize import brentq


def diffusion_peak(theta):
    '''Bias current of maximal phase diffusion (giant diffusion) and the enhancement there.

    Inputs:
      theta: float, noise strength k_B T / E_J, 0.02 <= theta <= 1.

    Output:
      (i_peak, d_peak): tuple of two Python floats.
        i_peak: the bias i > 0 at which the exact D_eff(i, theta) is largest, absolute
                error below 1e-6 (the maximum is flat: this needs D_eff, or
                d ln(D_eff) / di, far more accurately than the 1e-9 guaranteed by
                effective_diffusion).
        d_peak: D_eff(i_peak, theta) / theta at the returned i_peak (relative error below
                1e-9).

    Raises:
      ValueError if theta is not finite or is outside [0.02, 1].
    '''
    theta = float(theta)
    if not (np.isfinite(theta) and 0.02 <= theta <= 1.0):
        raise ValueError("theta must be in [0.02, 1]")
    # MUTANT: takes the phase diffusion as theta * dv/di (Einstein relation), so it returns the
    # maximum of the differential resistance instead of the maximum of D_eff
    def slope(x):
        h = 1e-5
        return (mean_voltage(x + h, theta)[1] - mean_voltage(x - h, theta)[1]) / (2.0 * h)

    grid = np.linspace(0.5, 3.0, 101)
    vals = [mean_voltage(x, theta)[1] for x in grid]
    k = int(np.argmax(vals))
    i_peak = brentq(slope, grid[max(k - 1, 0)], grid[min(k + 1, 100)], xtol=1e-12)
    result = (float(i_peak), float(mean_voltage(i_peak, theta)[1]))
    return result
