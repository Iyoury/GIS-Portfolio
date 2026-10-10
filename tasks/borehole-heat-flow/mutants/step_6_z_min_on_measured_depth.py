"""Mutant (step 6): Shallow-reading cut applied to measured depth instead of true vertical depth."""
import numpy as np


def surface_heat_flow(survey_md, survey_inc, survey_azi, log_md, log_temp,
                      layer_top_md, layer_k, heat_production, kappa,
                      hist_t_years, hist_dT_shape, z_min):
    """Paleoclimate-corrected surface heat flow from an inclined borehole."""
    log_md = np.asarray(log_md, float)
    log_temp = np.asarray(log_temp, float)
    if log_md.shape != log_temp.shape or log_md.ndim != 1:
        raise ValueError("log arrays must be 1-D with equal length")
    z = true_vertical_depth(survey_md, survey_inc, survey_azi, log_md)
    tops = true_vertical_depth(survey_md, survey_inc, survey_azi, np.asarray(layer_top_md, float))
    keep = log_md >= z_min                               # MUTANT: cut on MD
    z, T = z[keep], log_temp[keep]
    R, S = layer_integrals(tops, layer_k, z)
    P = paleoclimate_perturbation(z, hist_t_years, hist_dT_shape, kappa)
    fit = fit_heat_flow(T, R, S, P, heat_production)
    fit["n_used"] = int(z.size)
    return fit
