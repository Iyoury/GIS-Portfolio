"""Step 4 - joint estimate of surface temperature, heat flow and glacial amplitude."""
import numpy as np


def fit_heat_flow(T, R, S, P, A):
    """Least-squares fit of  T = T0 + q0 R - A S + g P.

    Parameters
    ----------
    T : (n,) observed temperatures (deg C), n >= 4
    R : (n,) thermal resistance at the observation depths (m^2 K W^-1)
    S : (n,) heat-production integral at the observation depths (m^3 K W^-1)
    P : (n,) paleoclimate perturbation for a unit-amplitude history (K per K)
    A : float, uniform radiogenic heat production (W m^-3), >= 0 (known)

    Returns
    -------
    dict with keys T0 (deg C), q0 (W m^-2), amplitude (g, K), sigma_T0,
    sigma_q0, sigma_amplitude (1-sigma standard errors of ordinary least
    squares, noise variance estimated as RSS/(n-3)) and rms_residual (K,
    sqrt(RSS/n)); every value a Python float. Accuracy: T0 and g within
    1e-9 K, q0 within 1e-12 W m^-2 of the exact least-squares solution,
    the standard errors within a relative 1e-6, rms_residual within 1e-9 K.

    Raises
    ------
    ValueError for unequal lengths, n < 4, non-finite values, A < 0 or
    non-finite, or an unresolvable design: one of the columns 1, R, P is all
    zero, or the smallest singular value of the n x 3 matrix [1, R, P] with
    each column scaled to unit length is below 1e-8 (e.g. P proportional to R).
    """
    raise NotImplementedError
