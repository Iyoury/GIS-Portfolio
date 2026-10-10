"""Step 6 - full workflow: paleoclimate-corrected surface heat flow of a borehole."""
import numpy as np

# You may call true_vertical_depth (step 1), layer_integrals (step 2),
# paleoclimate_perturbation (step 3) and fit_heat_flow (step 4).


def surface_heat_flow(survey_md, survey_inc, survey_azi, log_md, log_temp,
                      layer_top_md, layer_k, heat_production, kappa,
                      hist_t_years, hist_dT_shape, z_min):
    """Paleoclimate-corrected surface heat flow from an inclined borehole.

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
    kappa : thermal diffusivity of the ground (m^2 s^-1), > 0.
    hist_t_years, hist_dT_shape : ends of the history intervals (years
        before present, positive, strictly increasing) and the surface
        departure of the unit-amplitude history during each interval (K per
        unit amplitude; the departure is g * hist_dT_shape[i]).
    z_min : readings with TVD >= z_min (m) are used, shallower ones excluded.

    Workflow
    --------
    1. TVD of every reading and every layer top (step 1).
    2. Keep readings with TVD >= z_min.
    3. R and S at the kept depths (step 2); P of the unit-amplitude history (step 3).
    4. Joint least-squares estimate of T0, q0 and g (step 4).

    Returns
    -------
    dict: "T0" (deg C), "q0" (W m^-2), "amplitude" (g, K), "sigma_T0",
    "sigma_q0", "sigma_amplitude" (1-sigma standard errors, same units),
    "rms_residual" (K, sqrt(RSS/n_used)) as Python floats, and "n_used"
    (int, number of readings used).
    Accuracy: for temperatures that follow the model exactly, T0, q0 and g
    within 1e-5 K, 1e-8 W m^-2 and 1e-4 K of the generating values and
    rms_residual below 1e-6 K; for noisy temperatures, T0, q0 and g within
    5e-4 K, 2e-6 W m^-2 and 2e-3 K of the exact least-squares solution, the
    three standard errors within 2 % and rms_residual within 1 % (relative);
    n_used exact.

    Raises
    ------
    ValueError if log_md and log_temp are not 1-D of equal length, if any
    earlier step raises (invalid survey, layer tops or log depths beyond the
    last survey station, invalid conductivities, history or kappa,
    heat_production negative or non-finite), if fewer than 4 readings remain
    after the z_min cut, or if T0, q0 and g are not separately resolvable
    (criterion of step 4, e.g. an all-zero hist_dT_shape).
    """
    raise NotImplementedError
