"""Step 5 - full workflow: paleoclimate-corrected surface heat flow of a borehole."""
import numpy as np

SECONDS_PER_YEAR = 365.25 * 86400.0   # Julian year

# You may call true_vertical_depth (step 1), layer_integrals (step 2),
# paleoclimate_perturbation (step 3) and fit_heat_flow (step 4).


def surface_heat_flow(survey_md, survey_inc, survey_azi, log_md, log_temp,
                      layer_top_md, layer_k, heat_production, kappa,
                      hist_t_years, hist_dT_shape, z_min):
    """Paleoclimate-corrected surface heat flow from an inclined borehole.

    Parameters
    ----------
    survey_md, survey_inc, survey_azi : deviation survey (m, degrees, degrees)
    log_md, log_temp : 1-D arrays of equal length, reading MD (m) and temperature (deg C)
    layer_top_md, layer_k : layer tops along the hole (m, first 0) and conductivities (W/m/K)
    heat_production : uniform A (W m^-3), >= 0
    kappa : thermal diffusivity (m^2 s^-1)
    hist_t_years, hist_dT_shape : unit-amplitude surface-temperature history
    z_min : readings with TVD < z_min (m) are excluded

    Workflow
    --------
    1. TVD of every reading and every layer top (step 1).
    2. Keep readings with TVD >= z_min.
    3. R and S at the kept depths (step 2); P of the unit-amplitude history (step 3).
    4. Joint least-squares estimate of T0, q0 and g (step 4).

    Returns
    -------
    dict: "T0", "q0", "amplitude", "sigma_T0", "sigma_q0", "sigma_amplitude",
    "rms_residual" (Python floats) and "n_used" (int).

    Raises
    ------
    ValueError if log_md and log_temp are not 1-D of equal length, if any
    earlier step raises (invalid survey, layers, history, heat production),
    or if fewer than 4 readings remain after the z_min cut.
    """
    raise NotImplementedError
