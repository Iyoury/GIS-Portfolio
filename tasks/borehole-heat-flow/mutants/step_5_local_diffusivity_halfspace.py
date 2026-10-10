"""Mutant (step 5): erfc half-space response with the diffusivity of the layer that contains each depth."""
import numpy as np
import math

SECONDS_PER_YEAR = 365.25 * 86400.0

def layered_paleoclimate_perturbation(z, layer_tops, layer_k, rho_c, t_years, dT):
    """Present-day departure (K) from the steady state at depth z in a layered column, for a
    piecewise-constant surface-temperature history."""
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    tops = np.asarray(layer_tops, float)
    k = np.asarray(layer_k, float)
    t = np.asarray(t_years, float)
    d = np.asarray(dT, float)
    if tops.ndim != 1 or k.ndim != 1 or tops.shape != k.shape or tops.size < 1:
        raise ValueError("layer arrays must be 1-D with equal length")
    if not np.all(np.isfinite(tops)) or tops[0] != 0.0 or np.any(np.diff(tops) <= 0):
        raise ValueError("layer tops must start at 0 and increase strictly")
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("conductivities must be finite and positive")
    if not (np.isfinite(rho_c) and rho_c > 0):
        raise ValueError("volumetric heat capacity must be finite and positive")
    if t.ndim != 1 or t.shape != d.shape or t.size < 1:
        raise ValueError("history arrays must be 1-D with equal length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(d))):
        raise ValueError("history values must be finite")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("history times must be positive and increasing")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("depths must be >= 0")
    # erfc with the diffusivity of the layer that contains each depth
    lay = np.searchsorted(tops, zz, side="right") - 1
    kap_z = k[lay] / rho_c

    ef = np.frompyfunc(math.erfc, 1, 1)
    erfc = lambda x: np.asarray(ef(np.asarray(x, float)), dtype=float)
    ts = t * SECONDS_PER_YEAR
    edges = np.concatenate(([0.0], ts))
    out = np.zeros(zz.size)
    for i in range(t.size):
        e_old = erfc(zz / (2 * np.sqrt(kap_z * edges[i + 1])))
        e_new = np.zeros_like(zz) if edges[i] == 0.0 else erfc(zz / (2 * np.sqrt(kap_z * edges[i])))
        out += d[i] * (e_old - e_new)
    dT_z = float(out[0]) if z_in.ndim == 0 else out.reshape(z_in.shape)
    return dT_z
