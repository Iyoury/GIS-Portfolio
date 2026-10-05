import numpy as np
import math

MEC2 = 8.1871057769e-7          # erg, m_e c^2
KB = 1.380649e-16               # erg / K
LAMBDA_C = 3.8615926796e-11     # cm, hbar / (m_e c)
NA = 6.02214076e23              # 1 / mol
PREF = 1.0 / (math.pi ** 2 * LAMBDA_C ** 3)

_GX, _GW = np.polynomial.legendre.leggauss(24)


def _check_T_psi(T, psi):
    T, psi = float(T), float(psi)
    if not (math.isfinite(T) and math.isfinite(psi)):
        raise ValueError("T and psi must be finite")
    if not 1e7 <= T <= 1e11:
        raise ValueError("need 1e7 K <= T <= 1e11 K")
    if not 0.0 < psi <= 1e6:
        raise ValueError("need 0 < psi <= 1e6")
    return T, psi, KB * T / MEC2


def _nodes(theta, psi):
    """Quadrature nodes for int_0^inf p^2 dp (...): kinetic energies E (units m_e c^2), weights, and the
    electron occupation argument x_- = (E - E_F) / theta with E_F = psi theta - 1.
    Below the Fermi edge (and when it is close to E = 0): composite Gauss-Legendre in t = sqrt(E), smooth
    where p ~ sqrt(E). Over the edge E_F +- 60 theta (when it lies above 0): panels of width theta in
    y = (E - E_F) / theta, so x_- = y exactly instead of a small difference of two large energies."""
    EF = psi * theta - 1.0
    c = max(EF, 0.0)
    lo, hi = max(c - 60.0 * theta, 0.0), c + 60.0 * theta
    Es, ws, xs = [], [], []
    if lo > 0.0:
        # t-panels on [0, sqrt(lo)], doubling, then y-panels on [lo, hi] (E = EF + theta y)
        tlo = math.sqrt(lo)
        tb = [0.0]
        t = max(tlo / 2.0 ** 30, 1e-6)
        while t < tlo:
            tb.append(t)
            t *= 2.0
        tb.append(tlo)
        tb = np.array(tb)
        a, b = tb[:-1], tb[1:]
        t = ((0.5 * (b - a))[:, None] * _GX[None, :] + (0.5 * (a + b))[:, None]).ravel()
        wt = ((0.5 * (b - a))[:, None] * _GW[None, :]).ravel()
        E = t * t
        Es.append(E)
        ws.append(wt * 2.0 * t * t * (E + 1.0) * np.sqrt(E + 2.0))   # p^2 dp = 2 t^2 (E+1) sqrt(E+2) dt
        xs.append((E - EF) / theta)
        ylo, yhi = (lo - EF) / theta, (hi - EF) / theta
        nY = max(int(math.ceil(yhi - ylo)), 1)
        yb = np.linspace(ylo, yhi, nY + 1)
        a, b = yb[:-1], yb[1:]
        y = ((0.5 * (b - a))[:, None] * _GX[None, :] + (0.5 * (a + b))[:, None]).ravel()
        wy = ((0.5 * (b - a))[:, None] * _GW[None, :]).ravel()
        E = EF + theta * y
        p = np.sqrt(E * (E + 2.0))
        Es.append(E)
        ws.append(wy * theta * p * (E + 1.0))                          # p^2 dp = p (E+1) dE
        xs.append(y)
    else:
        nE = max(int(math.ceil((hi - lo) / theta)), 1)
        tb = np.unique(np.concatenate([[0.0], np.sqrt(np.linspace(lo, hi, nE + 1)[1:])]))
        a, b = tb[:-1], tb[1:]
        t = ((0.5 * (b - a))[:, None] * _GX[None, :] + (0.5 * (a + b))[:, None]).ravel()
        wt = ((0.5 * (b - a))[:, None] * _GW[None, :]).ravel()
        E = t * t
        Es.append(E)
        ws.append(wt * 2.0 * t * t * (E + 1.0) * np.sqrt(E + 2.0))
        xs.append((E - EF) / theta)
    return np.concatenate(Es), np.concatenate(ws), np.concatenate(xs)


def _occ(x):
    # 1 / (e^x + 1) and 1 minus it, without overflow or cancellation
    ex = np.exp(-np.abs(x))
    f = np.where(x > 0, ex / (1.0 + ex), 1.0 / (1.0 + ex))
    g = np.where(x > 0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
    return f, g


def _xp(E, theta, psi):
    # positron occupation argument eps / theta + psi, eps = E + 1 (always above 1 / theta)
    return (E + 1.0) / theta + psi


def specific_heat(rho_Ye, T):
    '''Specific heat per volume at constant net electron density of the electron-positron gas.

    Inputs:
      rho_Ye: float, rho Y_e in g cm^-3, 1e-10 <= rho_Ye <= 1e13 (n_net = rho_Ye N_A).
      T: float, temperature in K, 1e7 <= T <= 1e11.

    Output:
      cv: Python float, (d u_tot / d T) at constant n_net in erg K^-1 cm^-3, u_tot the total energy
        density including the rest energies; relative error below 1e-8. Each call within 10 s.

    Raises:
      ValueError if rho_Ye or T is not finite or is outside its range.
    '''
    psi = degeneracy_parameter(rho_Ye, T)
    T, psi, theta = _check_T_psi(T, psi)
    E, w, xm = _nodes(theta, psi)
    xp = _xp(E, theta, psi)
    fm, gm = _occ(xm)
    fp, gp = _occ(xp)
    eps = E + 1.0
    wa, wb = w * fm * gm, w * fp * gp            # fluctuation weights f (1 - f) of electrons, positrons
    WA, WB = wa.sum(), wb.sum()
    # c_V = (1 / k T^2) (<dE^2> - <dE dN>^2 / <dN^2>). The numerator <dE^2><dN^2> - <dE dN>^2 equals
    # 1/2 sum_ij w_i w_j (e_i n_j - e_j n_i)^2 (n = +1 for electrons, -1 for positrons): pairs of the same
    # species give centred second moments, mixed pairs (e_i + e_j)^2 > 0. In the degenerate limit the
    # Schur form cancels to (kT / E_F)^2; this one has no cancellation.
    ea = (wa * eps).sum() / WA
    M2a = (wa * (eps - ea) ** 2).sum()
    S1a, S2a = (wa * eps).sum(), (wa * eps * eps).sum()
    if WB > 0:
        eb = (wb * eps).sum() / WB
        M2b = (wb * (eps - eb) ** 2).sum()
        S1b, S2b = (wb * eps).sum(), (wb * eps * eps).sum()
    else:
        M2b = S1b = S2b = 0.0
    det = WA * M2a                                  # electrons only
    cv = KB * PREF / theta ** 2 * det / WA
    cv = float(cv)
    return cv
