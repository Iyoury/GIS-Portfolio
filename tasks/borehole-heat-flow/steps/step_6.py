"""Step 6 - joint inversion of one borehole log for q0, T0, the glacial amplitude and the end of the glaciation."""
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

# You may call true_vertical_depth (step 1), layer_integrals (step 2),
# fit_heat_flow (step 4) and layered_paleoclimate_perturbation (step 5).


def deglaciation_heat_flow_inversion(survey_md, survey_inc, survey_azi, log_md, log_temp,
                                     layer_top_md, layer_k, heat_production, rho_c,
                                     hist_t_years, hist_dT_shape, t_end_bounds, z_min):
    """Joint estimate of q0, T0, the glacial amplitude g and the end of the glaciation t_end.

    Parameters
    ----------
    survey_md, survey_inc, survey_azi : deviation survey as in step 1:
        station MD (m, first 0 at the collar, strictly increasing),
        inclination from vertical (degrees, 0 = straight down) and azimuth
        (degrees clockwise from north).
    log_md, log_temp : 1-D arrays of equal length, MD of each reading (m,
        within the survey) and its equilibrium temperature (deg C).
    layer_top_md, layer_k : MD of the top of each horizontal layer (m, first
        0, strictly increasing, within the survey; the last layer extends
        below the hole) and its conductivity (W m^-1 K^-1, > 0).
    heat_production : uniform radiogenic heat production A (W m^-3), >= 0.
    rho_c : uniform volumetric heat capacity (J m^-3 K^-1), > 0.
    hist_t_years : the m - 1 known interval ends (years before present,
        positive, strictly increasing).
    hist_dT_shape : the m >= 2 shape values s_0 ... s_(m-1) (K per unit
        amplitude). Interval 0 runs from the present back to t_end,
        interval 1 (the glaciation) from t_end back to hist_t_years[0],
        interval i >= 2 from hist_t_years[i-2] back to hist_t_years[i-1];
        the surface departure is g * s_i during interval i and zero before
        hist_t_years[-1]. P(z; t_end) = layered_paleoclimate_perturbation(
        z, layer tops in TVD, layer_k, rho_c, [t_end, *hist_t_years],
        hist_dT_shape) (step 5).
    t_end_bounds : (t_lo, t_hi) in years, 0 < t_lo < t_hi < hist_t_years[0].
    z_min : readings with TVD >= z_min (m) are used, shallower ones excluded.

    Model: T = T0 + q0 R(z) - A S(z) + g P(z; t_end) + noise at the TVD z of
    each used reading. t_end is the global minimizer over [t_lo, t_hi] of the
    profile RSS(t) (least squares in T0, q0, g at fixed t); T0, q0, g are the
    least-squares values at t_end and RSS_min = RSS(t_end).

    Returns
    -------
    dict: "T0" (deg C), "q0" (W m^-2), "amplitude" (g, K), "t_end" (years),
    "sigma_T0", "sigma_q0", "sigma_amplitude", "sigma_t_end" (Wald standard
    errors: s^2 (J^T J)^-1, s^2 = RSS_min / (n_used - 4), J the Jacobian of
    the model in (T0, q0, g, t_end) at the estimate, t_end in years), "t_end_low",
    "t_end_high" (years: crossings of RSS(t) with RSS_min (1 + F /
    (n_used - 4)), F = scipy.stats.f.ppf(0.95, 1, n_used - 4), nearest to
    t_end below and above it), "rms_residual" (K, sqrt(RSS_min / n_used)) as
    Python floats, and "n_used" (int, number of readings used).
    Accuracy (logs with rms_residual >= 1e-4 K, q0 > 0): T0 within 2e-3 K;
    q0, g and t_end within a relative 5e-4; the standard errors within a
    relative 5e-3; t_end_low and t_end_high within a relative 5e-4;
    rms_residual within a relative 1e-3; n_used exact. These limits include
    the errors propagated from the stated accuracies of steps 1, 2 and 5
    for step 5 errors that vary smoothly with t_end.

    Raises
    ------
    ValueError for fewer than 5 readings with TVD >= z_min,
    hist_dT_shape[0] == hist_dT_shape[1], a minimum of RSS(t) over
    [t_lo, t_hi] attained at t_lo or t_hi, or a nearest threshold crossing
    that does not exist inside [t_lo, t_hi].
    """
    raise NotImplementedError
