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


def _log_net(theta, psi, E, w):
    # ln n_net. f(b - psi) - f(b + psi) = sinh(psi) / (cosh(b) + cosh(psi)), b = (E + 1)/theta: electrons
    # minus positrons without cancellation, also when pairs outnumber the net electrons by many decades.
    # With M = max(b, psi): ln term = ln(1 - e^{-2 psi}) + psi - M - ln(e^{b-M} + e^{-b-M} + e^{psi-M} + e^{-psi-M})
    b = (E + 1.0) / theta
    M = np.maximum(b, psi)
    lt = psi - M - np.log(np.exp(b - M) + np.exp(-b - M) + np.exp(psi - M) + np.exp(-psi - M))
    lw = np.log(w) + lt
    m = lw.max()
    return math.log(PREF) + math.log(-math.expm1(-2.0 * psi)) + m + math.log(np.sum(np.exp(lw - m)))


def _net(theta, psi, E, w):
    return math.exp(_log_net(theta, psi, E, w))


def degeneracy_parameter(rho_Ye, T):
    '''psi = (mu + m_e c^2) / (k T) at which the net electron density equals rho_Ye N_A.

    Inputs:
      rho_Ye: float, rho Y_e in g cm^-3, 1e-10 <= rho_Ye <= 1e13.
      T: float, temperature in K, 1e7 <= T <= 1e11.

    Output:
      psi: Python float, psi > 0 with n_net(T, psi) = rho_Ye N_A (n_net of pair_densities), with a
        relative error below 1e-10.

    Raises:
      ValueError if rho_Ye or T is not finite or is outside its range.
    '''
    rho_Ye, T = float(rho_Ye), float(T)
    if not (math.isfinite(rho_Ye) and math.isfinite(T)):
        raise ValueError("rho_Ye and T must be finite")
    if not 1e7 <= T <= 1e11:
        raise ValueError("need 1e7 K <= T <= 1e11 K")
    if not 1e-10 <= rho_Ye <= 1e13:
        raise ValueError("need 1e-10 <= rho_Ye <= 1e13 g / cm^3")
    theta = KB * T / MEC2
    lt = math.log(rho_Ye * NA)

    def f(lpsi):
        psi = math.exp(lpsi)
        E, w, _ = _nodes(theta, psi)
        return _log_net(theta, psi, E, w) - lt

    # n_net increases with psi; bracket in ln psi, then bisection safeguarded false position with the
    # Illinois modification (the weight glo or ghi of an end kept twice in a row is halved), so that the
    # bracket shrinks from both sides; plain false position can stall with one end fixed
    llo, lhi = math.log(1e-300), 0.0
    flo, fhi = f(llo), f(lhi)
    while fhi < 0:
        llo, flo = lhi, fhi
        lhi += 1.0
        fhi = f(lhi)
    glo, ghi, kept = flo, fhi, 0
    for _ in range(300):
        lm = llo - glo * (lhi - llo) / (ghi - glo)
        if not (llo < lm < lhi) or lhi - llo > 0.5:
            lm = 0.5 * (llo + lhi)
        fm = f(lm)
        if fm > 0:
            lhi, fhi, ghi = lm, fm, fm
            if kept == -1:
                glo *= 0.5
            kept = -1
        else:
            llo, flo, glo = lm, fm, fm
            if kept == 1:
                ghi *= 0.5
            kept = 1
        if fm == 0 or lhi - llo < 1e-14 or min(abs(flo), abs(fhi)) < 1e-15:
            break
    psi = math.exp(lhi if abs(fhi) < abs(flo) else llo)
    return psi
