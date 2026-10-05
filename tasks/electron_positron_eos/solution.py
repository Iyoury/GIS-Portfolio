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
    n_net = _net(theta, psi, E, w)
    result = (n_minus, n_plus, n_net)
    return result


def _sigma(x):
    # -[f ln f + (1 - f) ln(1 - f)] for f = 1/(e^x + 1) = log1p(e^{-|x|}) + |x| / (e^{|x|} + 1) > 0
    ax = np.abs(x)
    e = np.exp(-ax)
    return np.log1p(e) + ax * e / (1.0 + e)


def pair_thermodynamics(T, psi):
    '''Pressure, internal energy and entropy of an ideal electron-positron gas.

    Inputs:
      T: float, temperature in K, 1e7 <= T <= 1e11.
      psi: float, (mu + m_e c^2) / (k T), 0 < psi <= 1e6 (as in pair_densities).

    Output:
      (P, u, s): Python floats. P the pressure (erg cm^-3); u the energy density (erg cm^-3): kinetic
        energy of the electrons plus kinetic energy and 2 m_e c^2 per positron; s the entropy density
        (erg K^-1 cm^-3). Each with a relative error below 1e-10. Each call within 10 s.

    Raises:
      ValueError if T or psi is not finite, if T is outside [1e7, 1e11] or psi is not in (0, 1e6].
    '''
    T, psi, theta = _check_T_psi(T, psi)
    E, w, xm = _nodes(theta, psi)
    xp = _xp(E, theta, psi)
    fm, fp = _occ(xm)[0], _occ(xp)[0]
    p2 = E * (E + 2.0)
    P = MEC2 * PREF / 3.0 * float(np.sum(w * p2 / (E + 1.0) * (fm + fp)))
    u = MEC2 * PREF * float(np.sum(w * (E * fm + (E + 2.0) * fp)))
    s = KB * PREF * float(np.sum(w * (_sigma(xm) + _sigma(xp))))
    result = (P, u, s)
    return result


def degeneracy_parameter(rho_Ye, T):
    '''psi = (mu + m_e c^2) / (k T) at which the net electron density equals rho_Ye N_A.

    Inputs:
      rho_Ye: float, rho Y_e in g cm^-3, 1e-10 <= rho_Ye <= 1e13.
      T: float, temperature in K, 1e7 <= T <= 1e11.

    Output:
      psi: Python float, psi > 0 with n_net(T, psi) = rho_Ye N_A (n_net of pair_densities), with a
        relative error below 1e-10. Each call within 10 s.

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

    # n_net increases with psi; bracket in ln psi, then bisection safeguarded secant
    llo, lhi = math.log(1e-300), 0.0
    flo, fhi = f(llo), f(lhi)
    while fhi < 0:
        llo, flo = lhi, fhi
        lhi += 1.0
        fhi = f(lhi)
    for _ in range(300):
        lm = llo - flo * (lhi - llo) / (fhi - flo)
        if not (llo < lm < lhi) or lhi - llo > 0.5:
            lm = 0.5 * (llo + lhi)
        fm = f(lm)
        if fm > 0:
            lhi, fhi = lm, fm
        else:
            llo, flo = lm, fm
        if fm == 0 or lhi - llo < 1e-14 or min(abs(flo), abs(fhi)) < 1e-15:
            break
    psi = math.exp(lhi if abs(fhi) < abs(flo) else llo)
    return psi


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
    det = WA * M2a + WB * M2b + WB * S2a + 2.0 * S1a * S1b + WA * S2b
    cv = KB * PREF / theta ** 2 * det / (WA + WB)
    cv = float(cv)
    return cv


def electron_positron_eos(rho, T, Ye):
    '''Equation of state of the electron-positron gas for a mass density, temperature and electron fraction.

    Inputs:
      rho: float, mass density in g cm^-3.
      T: float, temperature in K, 1e7 <= T <= 1e11.
      Ye: float, electrons per baryon, 0 < Ye <= 1; 1e-10 <= rho Ye <= 1e13.

    Output:
      dict with the Python floats "psi", "n_minus", "n_plus", "P", "u", "s", "cv" (units and accuracy
        as in steps 1-4). Each call within 30 s.

    Raises:
      ValueError if rho, T or Ye is not finite, if Ye is not in (0, 1], or if T or rho Ye is outside
      its range.
    '''
    rho, T, Ye = float(rho), float(T), float(Ye)
    if not (math.isfinite(rho) and math.isfinite(T) and math.isfinite(Ye)):
        raise ValueError("rho, T and Ye must be finite")
    if not 0.0 < Ye <= 1.0:
        raise ValueError("need 0 < Ye <= 1")
    psi = degeneracy_parameter(rho * Ye, T)
    n_minus, n_plus, n_net = pair_densities(T, psi)
    P, u, s = pair_thermodynamics(T, psi)
    cv = specific_heat(rho * Ye, T)
    result = {"psi": psi, "n_minus": n_minus, "n_plus": n_plus, "P": P, "u": u, "s": s, "cv": cv}
    return result
