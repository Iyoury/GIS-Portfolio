"""Mutant (step 2): thermal resistance computed with the arithmetic mean conductivity of all layers instead of summing the layer resistances in series."""
import numpy as np


def layer_integrals(layer_top_tvd, layer_k, z):
    """Thermal resistance R(z) = int_0^z dz'/k and S(z) = int_0^z z'/k dz'."""
    tops = np.asarray(layer_top_tvd, float)
    k = np.asarray(layer_k, float)
    z_in = np.asarray(z, float)
    zz = np.atleast_1d(z_in).ravel()
    if tops.ndim != 1 or tops.shape != k.shape or tops.size < 1:
        raise ValueError("layer arrays must be 1-D with equal length")
    if not np.all(np.isfinite(tops)):
        raise ValueError("layer tops must be finite")
    if tops[0] != 0.0 or np.any(np.diff(tops) <= 0):
        raise ValueError("layer tops must start at 0 and increase strictly")
    if np.any(~np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("conductivities must be positive")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("depths must be >= 0")
    bottoms = np.append(tops[1:], np.inf)
    R = np.zeros(zz.size)
    S = np.zeros(zz.size)
    for top, bot, kk in zip(tops, bottoms, k):
        lo = np.clip(zz, top, bot) - top
        a = top
        b = top + lo
        R += (b - a) / np.mean(k)          # MUTANT: arithmetic mean conductivity
        S += (b**2 - a**2) / (2 * kk)
    if z_in.ndim == 0:
        return float(R[0]), float(S[0])
    return R.reshape(z_in.shape), S.reshape(z_in.shape)
