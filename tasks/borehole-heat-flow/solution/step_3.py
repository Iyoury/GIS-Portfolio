"""Reference for step 3: present-day half-space perturbation of a piecewise-constant surface history."""
import numpy as np
import math


def _erfc_array(x):
    """Element-wise complementary error function (numpy + math only)."""
    f = np.frompyfunc(math.erfc, 1, 1)
    return np.asarray(f(np.asarray(x, float)), dtype=float)


SECONDS_PER_YEAR = 365.25 * 86400.0


def paleoclimate_perturbation(z, t_years, dT, kappa):
    """Present-day temperature perturbation (K) at depth z (m) from a
    piecewise-constant surface-temperature history."""
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    t = np.asarray(t_years, float)
    d = np.asarray(dT, float)
    if t.ndim != 1 or t.shape != d.shape or t.size < 1:
        raise ValueError("history arrays must be 1-D with equal length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(d))):
        raise ValueError("history values must be finite")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("history times must be positive and increasing")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("diffusivity must be positive")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("depths must be >= 0")
    ts = t * SECONDS_PER_YEAR
    edges = np.concatenate(([0.0], ts))
    out = np.zeros(zz.size)
    for i in range(t.size):
        t_new, t_old = edges[i], edges[i + 1]
        e_old = _erfc_array(zz / (2 * np.sqrt(kappa * t_old)))
        if t_new == 0.0:
            e_new = np.zeros_like(zz)       # limit z / sqrt(kappa * 0) -> inf
        else:
            e_new = _erfc_array(zz / (2 * np.sqrt(kappa * t_new)))
        out += d[i] * (e_old - e_new)
    return float(out[0]) if z_in.ndim == 0 else out.reshape(z_in.shape)
