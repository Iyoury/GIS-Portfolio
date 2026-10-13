import numpy as np
import math

_EPEOS5_MEC2 = 8.1871057769e-7          # erg, m_e c^2
_EPEOS5_KB = 1.380649e-16               # erg / K
_EPEOS5_LAMBDA_C = 3.8615926796e-11     # cm, hbar / (m_e c)
_EPEOS5_NA = 6.02214076e23              # 1 / mol
_EPEOS5_PREF = 1.0 / (math.pi ** 2 * _EPEOS5_LAMBDA_C ** 3)

_EPEOS5_GX, _EPEOS5_GW = np.polynomial.legendre.leggauss(24)


def _epeos5_nodes(theta, psi):
    """The quadrature nodes of pair_densities (step 1), rebuilt here so that this step does not depend on
    the private helpers of another step: kinetic energies E (units m_e c^2), weights for int_0^inf p^2 dp
    and the electron occupation argument x_- = (E - E_F) / theta, E_F = psi theta - 1. Gauss-Legendre in
    t = sqrt(E) below the Fermi edge and on panels of width theta in y = x_- over the edge E_F +- 60 theta,
    so that x_- near the edge is y itself, not a small difference of two large energies."""
    EF = psi * theta - 1.0
    c = max(EF, 0.0)
    lo, hi = max(c - 60.0 * theta, 0.0), c + 60.0 * theta
    Es, ws, xs = [], [], []
    if lo > 0.0:
        tlo = math.sqrt(lo)
        tb = [0.0]
        t = max(tlo / 2.0 ** 30, 1e-6)
        while t < tlo:
            tb.append(t)
            t *= 2.0
        tb.append(tlo)
        tb = np.array(tb)
        a, b = tb[:-1], tb[1:]
        t = ((0.5 * (b - a))[:, None] * _EPEOS5_GX[None, :] + (0.5 * (a + b))[:, None]).ravel()
        wt = ((0.5 * (b - a))[:, None] * _EPEOS5_GW[None, :]).ravel()
        E = t * t
        Es.append(E)
        ws.append(wt * 2.0 * t * t * (E + 1.0) * np.sqrt(E + 2.0))   # p^2 dp = 2 t^2 (E+1) sqrt(E+2) dt
        xs.append((E - EF) / theta)
        ylo, yhi = (lo - EF) / theta, (hi - EF) / theta
        nY = max(int(math.ceil(yhi - ylo)), 1)
        yb = np.linspace(ylo, yhi, nY + 1)
        a, b = yb[:-1], yb[1:]
        y = ((0.5 * (b - a))[:, None] * _EPEOS5_GX[None, :] + (0.5 * (a + b))[:, None]).ravel()
        wy = ((0.5 * (b - a))[:, None] * _EPEOS5_GW[None, :]).ravel()
        E = EF + theta * y
        p = np.sqrt(E * (E + 2.0))
        Es.append(E)
        ws.append(wy * theta * p * (E + 1.0))                          # p^2 dp = p (E+1) dE
        xs.append(y)
    else:
        nE = max(int(math.ceil((hi - lo) / theta)), 1)
        tb = np.unique(np.concatenate([[0.0], np.sqrt(np.linspace(lo, hi, nE + 1)[1:])]))
        a, b = tb[:-1], tb[1:]
        t = ((0.5 * (b - a))[:, None] * _EPEOS5_GX[None, :] + (0.5 * (a + b))[:, None]).ravel()
        wt = ((0.5 * (b - a))[:, None] * _EPEOS5_GW[None, :]).ravel()
        E = t * t
        Es.append(E)
        ws.append(wt * 2.0 * t * t * (E + 1.0) * np.sqrt(E + 2.0))
        xs.append((E - EF) / theta)
    return np.concatenate(Es), np.concatenate(ws), np.concatenate(xs)


def _epeos5_occ(x):
    # f = 1 / (e^x + 1) and 1 - f, without overflow or cancellation
    ex = np.exp(-np.abs(x))
    f = np.where(x > 0, ex / (1.0 + ex), 1.0 / (1.0 + ex))
    g = np.where(x > 0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
    return f, g


def _epeos5_response(T, psi, n_net, P, s, cv):
    """gamma1 and nabla_ad of the electron-positron gas at (T, psi), with n_net, P, s and c_V of that state.

    Gibbs-Duhem with only n_net conserved (mu_+ = -mu_-) gives dP = s dT + n_net dmu, so
      (dP/dn)_T = n_net / (dn/dmu)_T,   (dP/dT)_n = s - n_net (dn/dT)_mu / (dn/dmu)_T,
    with the grand-canonical response moments (weights f (1 - f) of electrons and positrons)
      k T (dn/dmu)_T = n_psi = PREF sum w (g_- + g_+),   T (dn/dT)_mu = X = PREF sum w (x_- g_- - x_+ g_+).
    On the edge-centred nodes x_- is the distance from the Fermi edge in units of k T, so the antisymmetric
    edge moment X, about k T / E_F of its scale in a degenerate gas, keeps its relative accuracy. The first
    law per baryon at constant s / n_net gives Gamma3 - 1 = (dP/dT)_n / c_V; then
      gamma1 = chi_rho + chi_T (Gamma3 - 1),  nabla_ad = (Gamma3 - 1) / gamma1,
    chi_rho = n_net (dP/dn)_T / P, chi_T = T (dP/dT)_n / P. Pair creation enters every moment."""
    theta = _EPEOS5_KB * T / _EPEOS5_MEC2
    E, w, xm = _epeos5_nodes(theta, psi)
    xp = (E + 1.0) / theta + psi
    fm, gm = _epeos5_occ(xm)
    fp, gp = _epeos5_occ(xp)
    wa, wb = w * fm * gm, w * fp * gp
    n_psi = _EPEOS5_PREF * float(np.sum(wa + wb))
    X = _EPEOS5_PREF * float(np.sum(wa * xm - wb * xp))
    P_T = s - n_net * _EPEOS5_KB * X / n_psi
    P_n = n_net * _EPEOS5_KB * T / n_psi
    chi_rho = n_net * P_n / P
    chi_T = T * P_T / P
    g3m1 = P_T / cv
    gamma1 = chi_rho + chi_T * g3m1
    nabla_ad = g3m1 / gamma1
    return float(gamma1), float(nabla_ad)


def electron_positron_eos(rho, T, Ye):
    '''Equation of state and adiabatic exponents of the electron-positron gas.

    Inputs:
      rho: float, mass density in g cm^-3.
      T: float, temperature in K, 1e7 <= T <= 1e11.
      Ye: float, electrons per baryon, 0 < Ye <= 1; 1e-10 <= rho Ye <= 1e13.

    Output:
      dict with exactly the nine keys "psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1",
        "nabla_ad", whose values are Python floats: psi, densities (cm^-3), P (erg cm^-3), u (erg cm^-3),
        s (erg K^-1 cm^-3) and cv (erg K^-1 cm^-3) as in steps 1-4; gamma1 = (d ln P / d ln rho) and
        nabla_ad = (d ln T / d ln P), both at constant s / rho and Ye for the electron-positron gas alone,
        pairs in equilibrium. Accuracy with respect to the exact values for the given (rho, T, Ye),
        including the error propagated from psi: psi, n_minus, P, u, s relative 1e-10; n_plus relative
        1e-10 when at least 1e-250 cm^-3, otherwise within 1e-250 cm^-3; cv, gamma1, nabla_ad relative
        1e-8.

    Raises:
      ValueError if Ye is not in (0, 1], or if T or rho Ye is outside its range.
    '''
    rho, T, Ye = float(rho), float(T), float(Ye)
    if not (math.isfinite(rho) and math.isfinite(T) and math.isfinite(Ye)):
        raise ValueError("rho, T and Ye must be finite")
    if not 0.0 < Ye <= 1.0:
        raise ValueError("need 0 < Ye <= 1")
    rho_Ye = rho * Ye
    psi = degeneracy_parameter(rho_Ye, T)
    n_minus, n_plus, n_net = pair_densities(T, psi)
    P, u, s = pair_thermodynamics(T, psi)
    cv = specific_heat(rho_Ye, T)
    gamma1, nabla_ad = _epeos5_response(T, psi, rho_Ye * _EPEOS5_NA, P, s, cv)
    result = {"psi": psi, "n_minus": n_minus, "n_plus": n_plus, "P": P, "u": u, "s": s, "cv": cv,
              "gamma1": gamma1, "nabla_ad": nabla_ad}
    return result
