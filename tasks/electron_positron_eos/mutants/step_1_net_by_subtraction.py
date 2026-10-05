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


def pair_densities(T, psi):
    '''Number densities of electrons and positrons in an ideal Fermi gas with pairs.

    Inputs:
      T: float, temperature in K, 1e7 <= T <= 1e11.
      psi: float, (mu + m_e c^2) / (k T) with mu the electron chemical potential without the rest
           mass, 0 < psi <= 1e6; electrons occupy 1 / (exp(eps/theta - psi) + 1), positrons
           1 / (exp(eps/theta + psi) + 1), eps = sqrt(1 + p^2), theta = k T / (m_e c^2).

    Output:
      (n_minus, n_plus, n_net): Python floats in cm^-3, n = (1 / (pi^2 lambda^3)) int p^2 f dp with
        lambda = hbar / (m_e c); n_net = n_minus - n_plus. n_minus and n_net with a relative error
        below 1e-10; n_plus with a relative error below 1e-10 when at least 1e-250 cm^-3, otherwise
        within 1e-250 cm^-3. Each call within 10 s.

    Raises:
      ValueError if T or psi is not finite, if T is outside [1e7, 1e11] or psi is not in (0, 1e6].
    '''
    T, psi, theta = _check_T_psi(T, psi)
    E, w, xm = _nodes(theta, psi)
    xp = _xp(E, theta, psi)
    n_minus = PREF * float(np.sum(w * _occ(xm)[0]))
    n_plus = PREF * float(np.sum(w * _occ(xp)[0]))
    n_net = n_minus - n_plus                       # direct difference
    result = (n_minus, n_plus, n_net)
    return result
