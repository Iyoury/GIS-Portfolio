"""Step 7 - regional maximum-likelihood inversion of several borehole logs sharing one glacial history, with
serially correlated (AR(1)) noise in every log."""
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

# You may call true_vertical_depth (step 1), layer_integrals (step 2) and
# layered_paleoclimate_perturbation (step 5).


def regional_deglaciation_inversion(holes, hist_t_years, hist_dT_shape, t_end_bounds):
    """Regional maximum-likelihood estimate of the glacial amplitude g and the end of the glaciation t_end
    with serially correlated (AR(1)) noise in every log.

    Parameters
    ----------
    holes : list of H >= 2 dicts, one per borehole, each with exactly the keys
        "survey_md", "survey_inc", "survey_azi", "log_md", "log_temp",
        "layer_top_md", "layer_k", "heat_production", "rho_c" and "z_min",
        with the meanings, units and valid ranges of the arguments of the same
        names of deglaciation_heat_flow_inversion (step 6); log_md need not be
        sorted, but its values are distinct within a hole.
    hist_t_years, hist_dT_shape, t_end_bounds : the glacial history of step 6,
        shared by every hole: the m - 1 known interval ends (years before
        present, positive, strictly increasing), the m >= 2 shape values
        s_0 ... s_(m-1) (K per unit amplitude, s_0 != s_1) and (t_lo, t_hi) in
        years with 0 < t_lo < t_hi < hist_t_years[0].

    Model: in hole h the readings with TVD z >= z_min_h, numbered i = 1 ... n_h
    by increasing MD, obey
    T = T0_h + q0_h R_h(z) - A_h S_h(z) + g P_h(z; t_end)
    + e_i, with P_h(z; t_end) = layered_paleoclimate_perturbation(z, the hole's
    layer tops in TVD, its layer_k, its rho_c, [t_end, *hist_t_years],
    hist_dT_shape) (step 5). The errors are a stationary Gaussian AR(1)
    sequence in i: e_1 ~ N(0, s_h^2 / (1 - phi_h^2)), e_i = phi_h e_(i-1) + w_i
    with independent w_i ~ N(0, s_h^2); phi_h (|phi_h| < 1) and s_h > 0 are
    unknown, the holes independent. Every parameter is estimated by maximum
    likelihood (exact joint density of all used readings, the first reading of
    every hole included); t_end is the global maximizer over [t_lo, t_hi] of
    the profile log-likelihood l_p(t) (maximum over all other parameters at
    t_end = t, g over the whole real line) and l_max = l_p(t_end).

    Returns
    -------
    dict: "g" (K), "t_end" (years), "sigma_g", "sigma_t_end" (standard errors:
    square roots of the diagonal of the inverse of the expected Fisher
    information of the regression parameters T0_h, q0_h, g, t_end at the
    estimates, phi_h and s_h held at their estimates; t_end in years),
    "t_end_low", "t_end_high" (years: points nearest to t_end below and above
    it where 2 (l_max - l_p(t)) = scipy.stats.chi2.ppf(0.95, 1)) and
    "log_likelihood" (l_max, natural logarithm of the joint normal density of
    all used readings) as Python floats, and "holes": a list, in the input
    order, of dicts with "T0" (deg C), "q0" (W m^-2), "sigma_T0", "sigma_q0"
    (standard errors), "phi" (maximum-likelihood phi_h), "innovation_sd"
    (maximum-likelihood s_h, K) as Python floats and "n_used" (int, readings
    used in the hole).
    Domain: at least 8 used readings per hole; maximum-likelihood s_h from
    1e-3 K to 0.03 K and phi_h from -0.3 to 0.9, and q0_h > 0, in every hole.
    Accuracy: g and t_end within a relative 2e-3 (g != 0 in a returned
    result); every q0 within a relative 5e-4; every T0 within 2e-3 K; the
    standard errors within a relative 1e-2; t_end_low and t_end_high within a
    relative 5e-4; every phi within 2e-3;
    every innovation_sd within a relative 3e-3; log_likelihood within 0.1;
    n_used exact. These limits include the errors propagated from the stated
    accuracies of steps 1, 2 and 5.

    Raises
    ------
    ValueError if a hole has fewer than 8 readings with TVD >= its z_min, if
    the maximum of l_p over [t_lo, t_hi] is attained at t_lo or t_hi, or if a
    nearest crossing does not exist inside [t_lo, t_hi].
    """
    raise NotImplementedError
