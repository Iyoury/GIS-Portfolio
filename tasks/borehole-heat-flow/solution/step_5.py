"""Reference for step 5: paleoclimate perturbation in a layered conductivity column."""
import numpy as np

SECONDS_PER_YEAR = 365.25 * 86400.0


def _log_surface_response(z, lay, s, tops, k, rho_c):
    """log F(z, s): Laplace transform (in time) of the departure at depth z for a unit departure applied
    at the surface, F(0, s) = 1, in the layered column (z: depths, lay: their layer indices, s: complex)."""
    nl = tops.size
    kap = k / rho_c
    q = np.sqrt(s[None, :] / kap[:, None])                  # (layers, nodes), Re q > 0
    kq = k[:, None] * q
    h = np.diff(tops)
    # admittance Y = k F' / F at the top of each layer, from the bottom up (continuity of F and k F'):
    # the bottom half-space keeps only the decaying solution, Y = -k q
    Y = np.empty((nl, s.size), complex)
    Y[-1] = -kq[-1]
    for i in range(nl - 2, -1, -1):
        th = np.tanh(q[i] * h[i])
        Y[i] = (Y[i + 1] - kq[i] * th) / (1.0 - Y[i + 1] * th / kq[i])
    # F at the layer tops: F_{i+1} / F_i = 1 / (cosh(q h) - r sinh(q h)) with r = Y_{i+1} / (k_i q_i),
    # written as 2 e^{-q h} / ((1 + e^{-2qh}) - r (1 - e^{-2qh})): no cancellation, no overflow
    lFtop = np.zeros((nl, s.size), complex)
    for i in range(nl - 1):
        e2 = np.exp(-2.0 * q[i] * h[i])
        r = Y[i + 1] / kq[i]
        lFtop[i + 1] = lFtop[i] - q[i] * h[i] + np.log(2.0 / ((1.0 + e2) - r * (1.0 - e2)))
    out = np.empty((z.size, s.size), complex)
    for i in np.unique(lay):
        sel = np.where(lay == i)[0]
        d = (z[sel] - tops[i])[:, None]
        if i == nl - 1:
            out[sel] = lFtop[i][None, :] - q[i][None, :] * d
        else:
            # inside the layer: F(d) / F_top = (cosh(q(h-d)) - r sinh(q(h-d))) / (cosh(qh) - r sinh(qh))
            r = (Y[i + 1] / kq[i])[None, :]
            e2a = np.exp(-2.0 * q[i][None, :] * (h[i] - d))
            e2b = np.exp(-2.0 * q[i] * h[i])[None, :]
            num = (1.0 + e2a) - r * (1.0 - e2a)
            den = (1.0 + e2b) - r * (1.0 - e2b)
            out[sel] = lFtop[i][None, :] - q[i][None, :] * d + np.log(num / den)
    return out


def _step_response(z, lay, tau, tops, k, rho_c, M=32):
    """Departure now at depth z after the surface was raised by 1 K a time tau (s) ago: inverse Laplace
    transform of F / s at tau on the fixed Talbot contour (Abate and Valko 2004)."""
    r = 2.0 * M / (5.0 * tau)
    th = np.arange(1, M) * np.pi / M
    cot = np.cos(th) / np.sin(th)
    sig = r * th * (cot + 1j)
    beta = th + (th * cot - 1.0) * cot
    s = np.concatenate(([r + 0j], sig))
    terms = np.exp(_log_surface_response(z, lay, s, tops, k, rho_c) - np.log(s)[None, :] + s[None, :] * tau)
    val = 0.5 * terms[:, 0].real + np.sum((terms[:, 1:] * (1.0 + 1j * beta)[None, :]).real, axis=1)
    return r / M * val


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
    # The departure obeys rho_c dT/dt = d/dz (k dT/dz) with T and k dT/dz continuous at the interfaces
    # and the surface history as boundary value; the erfc half-space formula holds only for a uniform
    # column. By superposition it is a sum of step responses U(z, tau) (surface raised by 1 K a time tau
    # ago), each obtained from its Laplace transform F(z, s) / s, which the layers give in closed form.
    lay = np.searchsorted(tops, zz, side="right") - 1
    ts = t * SECONDS_PER_YEAR
    edges = np.concatenate(([0.0], ts))
    inside = zz > 0
    out = np.zeros(zz.size)
    if np.any(inside):
        zi, li = zz[inside], lay[inside]
        U = {}
        for tau in edges[1:]:
            U[tau] = _step_response(zi, li, tau, tops, k, float(rho_c))
        acc = np.zeros(zi.size)
        for i in range(t.size):
            u_old = U[edges[i + 1]]
            u_new = U[edges[i]] if edges[i] > 0 else 0.0     # U(z > 0, 0) = 0
            acc += d[i] * (u_old - u_new)
        out[inside] = acc
    out[~inside] = d[0]                                       # at the surface: the present departure
    dT_z = float(out[0]) if z_in.ndim == 0 else out.reshape(z_in.shape)
    return dT_z
