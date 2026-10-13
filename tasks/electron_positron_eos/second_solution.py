import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import brentq

# Second solution, built on routes different from the reference:
#   - quadrature: adaptive QUADPACK (scipy.integrate.quad) instead of fixed composite Gauss-Legendre panels,
#     on panels of the kinetic energy E (units m_e c^2) split at max(E_F, 0) = c, at c +- 40 k T (where
#     positive) and ending at c + 80 k T; the first panel (from E = 0) and every panel of a non-degenerate gas
#     (E_F <= 0) are integrated in the momentum p, the other panels in y = (E - E_F) / k T;
#   - net electron density: for psi >= 1 the plain difference n_minus - n_plus (n_plus / n_minus >= e^{-2 psi},
#     but below 0.163 over the domain, measured; pointwise f_+ / f_- <= 0.36), so the difference amplifies the
#     relative quadrature error by at most 1.39 (rigorously at most 2.1); for psi < 1 the charge-susceptibility
#     integral n_net(psi) = int_0^psi (d n_net / d psi') dpsi', whose integrand
#     PREF int p^2 [f_-(1 - f_-) + f_+(1 - f_+)]
#     is positive and analytic in psi' for |Im psi'| < pi (n_net(0) = 0 by charge-conjugation symmetry);
#     Gauss-Legendre with 16 nodes in psi' on [0, psi];
#   - pressure from the grand potential, P = k T PREF int p^2 [ln(1 + e^{-x_-}) + ln(1 + e^{-x_+})] dp
#     (x = occupation argument), not from the p^4 / eps moment;
#   - psi by Brent's method on ln n_net(ln psi);
#   - c_V from the thermodynamic Jacobian in the variables (T, mu): c_V = u_T - u_mu n_T / n_mu, with the
#     partial derivatives at fixed mu (resp. fixed T) integrated from the analytic derivatives of the
#     occupation numbers (d f / d T at fixed mu = f (1 - f) x / T, d f_-+ / d psi = +- f (1 - f)), written with
#     the occupation arguments x (eps = theta (x_- + psi) = theta (x_+ - psi)) so that the mu n_T term is
#     carried by one integral B in both u_T and u_mu n_T / n_mu;
#   - gamma1 and nabla_ad (version 6) from their definitions, without fluctuation moments, Maxwell relations
#     or c_V: Richardson-extrapolated five-point difference quotients (steps 4e-3 and 2e-3) of ln P and of
#     the entropy per net electron sigma = s / n_net in ln n_net and ln T, each stencil state with psi solved
#     afresh (Brent, then two Newton steps in psi) and P, s from the integrals above, without the range
#     checks of the public functions (the stencils may step just outside T = 1e7 K or rho Ye = 1e13 g cm^-3);
#     Gamma3 - 1 = -(d sigma / d ln n)_T / (d sigma / d ln T)_n, gamma1 = (d ln P / d ln n)_T
#     + (d ln P / d ln T)_n (Gamma3 - 1), nabla_ad = (Gamma3 - 1) / gamma1.
# The first momentum panel (from p = 0 up to the Fermi window) is split at p = 1, 2, 4, ... (version 6): a
# single 21-point Gauss-Kronrod rule over it could report convergence with an actual error of up to 5e-11.
# quad is called with full_output=1, which returns QUADPACK's round-off notices instead of raising warnings
# (epsrel 1e-13 is at the round-off level), so the process-wide warning filters are left untouched.

_SS_MEC2 = 8.1871057769e-7
_SS_KB = 1.380649e-16
_SS_LAMBDA_C = 3.8615926796e-11
_SS_NA = 6.02214076e23
_SS_PREF = 1.0 / (math.pi ** 2 * _SS_LAMBDA_C ** 3)
_SS_GX, _SS_GW = np.polynomial.legendre.leggauss(16)


def _ss_check(T, psi):
    T, psi = float(T), float(psi)
    if not (math.isfinite(T) and math.isfinite(psi)):
        raise ValueError("T and psi must be finite")
    if not 1e7 <= T <= 1e11:
        raise ValueError("need 1e7 K <= T <= 1e11 K")
    if not 0.0 < psi <= 1e6:
        raise ValueError("need 0 < psi <= 1e6")
    return T, psi, _SS_KB * T / _SS_MEC2


def _ss_fermi(x):
    if x > 0:
        e = math.exp(-x)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(x))


def _ss_fq(x):
    # f (1 - f) = e^{-|x|} / (1 + e^{-|x|})^2
    e = math.exp(-abs(x))
    return e / (1.0 + e) ** 2


def _ss_lnp(x):
    # ln(1 + e^{-x})
    if x > 0:
        return math.log1p(math.exp(-x))
    return -x + math.log1p(math.exp(x))


def _ss_sig(x):
    # -[f ln f + (1 - f) ln(1 - f)], symmetric in x
    a = abs(x)
    e = math.exp(-a)
    return math.log1p(e) + a * e / (1.0 + e)


def _ss_quad(f, a, b, points=None):
    return quad(f, a, b, epsabs=0.0, epsrel=1e-13, limit=400, points=points, full_output=1)[0]


def _ss_integrate(g, theta, psi):
    # int_0^inf p^2 dp g(E, y), E = sqrt(1 + p^2) - 1 the kinetic energy and y = (E - E_F) / theta = x_- passed
    # separately. Below the edge window (and for a non-degenerate gas) in the momentum p itself; over the
    # window E_F +- 40 theta in y with p^2 dp = p (E + 1) dE, so that the electron occupation near the edge is
    # evaluated without cancellation; beyond E_F + 80 theta the occupations are below e^{-80}
    EF = psi * theta - 1.0
    c = max(EF, 0.0)
    pts = sorted({max(c - 40.0 * theta, 0.0), c, c + 40.0 * theta, c + 80.0 * theta})
    pts = [0.0] + [q for q in pts if q > 0.0]
    tot = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        if a == 0.0 or EF <= 0.0 or b <= c - 40.0 * theta:
            def fp(p):
                E = p * p / (1.0 + math.sqrt(1.0 + p * p))
                return p * p * g(E, (E - EF) / theta)
            pa, pb = math.sqrt(a * (a + 2.0)), math.sqrt(b * (b + 2.0))
            # breakpoints at p = 1, 2, 4, ...: a single 21-point Gauss-Kronrod rule over a long first panel
            # (p up to the Fermi momentum) can report convergence while missing the curvature of
            # eps = sqrt(1 + p^2) near p = 1 (error estimate 1e-13, actual error up to 5e-11, measured)
            br = [q for q in (2.0 ** k for k in range(64)) if pa < q < pb]
            tot += _ss_quad(fp, pa, pb, points=br or None)
        else:
            ya, yb = (a - EF) / theta, (b - EF) / theta

            def fy(y):
                E = EF + theta * y
                return theta * math.sqrt(E * (E + 2.0)) * (E + 1.0) * g(E, y)
            tot += _ss_quad(fy, ya, yb, points=[0.0] if ya < 0 < yb else None)
    return tot


def _ss_densities(theta, psi):
    xp = lambda E: (E + 1.0) / theta + psi
    n_minus = _ss_PREF_int(lambda E, y: _ss_fermi(y), theta, psi)
    n_plus = _ss_PREF_int(lambda E, y: _ss_fermi(xp(E)), theta, psi)
    return n_minus, n_plus


def _ss_PREF_int(g, theta, psi):
    return _SS_PREF * _ss_integrate(g, theta, psi)


def _ss_chi(theta, psi):
    # d n_net / d psi at fixed T: PREF int p^2 [f_-(1 - f_-) + f_+(1 - f_+)] dp > 0
    return _ss_PREF_int(lambda E, y: _ss_fq(y) + _ss_fq((E + 1.0) / theta + psi), theta, psi)


def _ss_log_net(theta, psi):
    if psi >= 1.0:
        n_minus, n_plus = _ss_densities(theta, psi)
        return math.log(n_minus - n_plus)
    # n_net = int_0^psi chi(psi') dpsi' (Gauss-Legendre in psi'), kept in logarithms
    chis = [_ss_chi(theta, 0.5 * psi * (1.0 + t)) for t in _SS_GX]
    return math.log(0.5 * psi) + math.log(sum(w * c for w, c in zip(_SS_GW, chis)))


def pair_densities(T, psi):
    '''Number densities of electrons and positrons in an ideal Fermi gas with pairs.

    Inputs:
      T: float, temperature in K, 1e7 <= T <= 1e11.
      psi: float, (mu + m_e c^2) / (k T) with mu the electron chemical potential without the rest
           energy, 0 < psi <= 1e6; a state of total energy eps (units m_e c^2) is occupied by an
           electron with probability 1 / (exp(eps/theta - psi) + 1) and by a positron with
           probability 1 / (exp(eps/theta + psi) + 1), theta = k T / (m_e c^2).

    Output:
      (n_minus, n_plus, n_net): Python floats in cm^-3, the electron and positron number densities
        (two spin states per (2 pi hbar)^3 of phase space) and n_net = n_minus - n_plus. n_minus with a
        relative error below 1e-10; n_plus and n_net each with a relative error below 1e-10 when at
        least 1e-250 cm^-3, otherwise within 1e-250 cm^-3.

    Raises:
      ValueError if T is outside [1e7, 1e11] or psi is not in (0, 1e6].
    '''
    T, psi, theta = _ss_check(T, psi)
    n_minus, n_plus = _ss_densities(theta, psi)
    if psi >= 1.0:
        n_net = n_minus - n_plus
    else:
        n_net = math.exp(_ss_log_net(theta, psi))
    return float(n_minus), float(n_plus), float(n_net)


def pair_thermodynamics(T, psi):
    '''Pressure, internal energy and entropy of an ideal electron-positron gas.

    Inputs:
      T: float, temperature in K, 1e7 <= T <= 1e11.
      psi: float, (mu + m_e c^2) / (k T) as in the conventions, 0 < psi <= 1e6.

    Output:
      (P, u, s): Python floats. P the pressure (erg cm^-3); u the energy density (erg cm^-3): kinetic
        energy of the electrons plus kinetic energy and 2 m_e c^2 per positron; s the entropy density
        (erg K^-1 cm^-3). Each with a relative error below 1e-10.

    Raises:
      ValueError if T is outside [1e7, 1e11] or psi is not in (0, 1e6].
    '''
    T, psi, theta = _ss_check(T, psi)
    xp = lambda E: (E + 1.0) / theta + psi
    P = _SS_KB * T * _ss_PREF_int(lambda E, y: _ss_lnp(y) + _ss_lnp(xp(E)), theta, psi)
    u = _SS_MEC2 * _ss_PREF_int(lambda E, y: E * _ss_fermi(y) + (E + 2.0) * _ss_fermi(xp(E)), theta, psi)
    s = _SS_KB * _ss_PREF_int(lambda E, y: _ss_sig(y) + _ss_sig(xp(E)), theta, psi)
    return float(P), float(u), float(s)


def degeneracy_parameter(rho_Ye, T):
    '''psi = (mu + m_e c^2) / (k T) at which the net electron density equals rho_Ye N_A.

    Inputs:
      rho_Ye: float, rho Y_e in g cm^-3, 1e-10 <= rho_Ye <= 1e13.
      T: float, temperature in K, 1e7 <= T <= 1e11.

    Output:
      psi: Python float, psi > 0 with n_net(T, psi) = rho_Ye N_A (n_net of pair_densities), with a
        relative error below 1e-10.

    Raises:
      ValueError if rho_Ye or T is outside its range.
    '''
    rho_Ye, T = float(rho_Ye), float(T)
    if not (math.isfinite(rho_Ye) and math.isfinite(T)):
        raise ValueError("rho_Ye and T must be finite")
    if not 1e7 <= T <= 1e11:
        raise ValueError("need 1e7 K <= T <= 1e11 K")
    if not 1e-10 <= rho_Ye <= 1e13:
        raise ValueError("need 1e-10 <= rho_Ye <= 1e13 g / cm^3")
    theta = _SS_KB * T / _SS_MEC2
    lt = math.log(rho_Ye * _SS_NA)
    f = lambda lp: _ss_log_net(theta, math.exp(lp)) - lt
    lo, hi = math.log(1e-300), 0.0
    while f(hi) < 0:
        lo, hi = hi, hi + 2.0
    return float(math.exp(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)))


def specific_heat(rho_Ye, T):
    '''Specific heat per volume at constant net electron density of the electron-positron gas.

    Inputs:
      rho_Ye: float, rho Y_e in g cm^-3, 1e-10 <= rho_Ye <= 1e13 (n_net = rho_Ye N_A).
      T: float, temperature in K, 1e7 <= T <= 1e11.

    Output:
      cv: Python float, (d u_tot / d T) at constant n_net in erg K^-1 cm^-3, u_tot the total energy
        density including the rest energies; relative error below 1e-8.

    Raises:
      ValueError if rho_Ye or T is outside its range.
    '''
    psi = degeneracy_parameter(rho_Ye, T)
    T, psi, theta = _ss_check(T, psi)
    xp = lambda E: (E + 1.0) / theta + psi
    # moments of the fluctuation weights f (1 - f) in the occupation arguments x_- = y, x_+ = eps/theta + psi
    A = _ss_PREF_int(lambda E, y: y * y * _ss_fq(y) + xp(E) ** 2 * _ss_fq(xp(E)), theta, psi)
    B = _ss_PREF_int(lambda E, y: y * _ss_fq(y) - xp(E) * _ss_fq(xp(E)), theta, psi)
    W = _ss_PREF_int(lambda E, y: _ss_fq(y) + _ss_fq(xp(E)), theta, psi)
    # partial derivatives of u_tot = u + n_net m_e c^2 and n_net: in T at fixed mu = psi k T, in psi at fixed T
    u_T = _SS_KB * (A + psi * B)
    u_psi = _SS_KB * T * (B + psi * W)
    n_T = B / T
    n_psi = W
    # constant n_net: dpsi = -(n_T / n_psi) dT along the path, d mu / d psi = k T at fixed T
    cv = u_T - u_psi * n_T / n_psi
    return float(cv)


def _ss_psi_of_n(n, theta):
    # psi > 0 with n_net(theta, psi) = n, without range checks (the stencils of the response may step just
    # outside the input domain of the public functions)
    lt = math.log(n)
    f = lambda lp: _ss_log_net(theta, math.exp(lp)) - lt
    lo, hi = math.log(1e-300), 0.0
    while f(hi) < 0:
        lo, hi = hi, hi + 2.0
    psi = math.exp(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500))
    # brentq stops at a tolerance in ln psi, which leaves psi uncertain by about 1e-15 |ln psi| psi; in a
    # cool classical gas (psi up to about 600) ln P and s / n_net move by that absolute error of psi, which
    # the difference quotients below would divide by 1e-2. Two Newton steps in psi itself
    # (d ln n_net / d psi = chi / n_net) bring it to rounding level.
    for _ in range(2):
        psi -= (_ss_log_net(theta, psi) - lt) * n / _ss_chi(theta, psi)
    return psi


def _ss_lnP_sigma(n, T):
    # ln P and the entropy per net electron sigma = s / n_net of the state (n_net = n, T), from the grand
    # potential and the Gibbs entropy integrand, psi solved afresh
    theta = _SS_KB * T / _SS_MEC2
    psi = _ss_psi_of_n(n, theta)
    xp = lambda E: (E + 1.0) / theta + psi
    P = _SS_KB * T * _ss_PREF_int(lambda E, y: _ss_lnp(y) + _ss_lnp(xp(E)), theta, psi)
    s = _SS_KB * _ss_PREF_int(lambda E, y: _ss_sig(y) + _ss_sig(xp(E)), theta, psi)
    return math.log(P), s / n


def _ss_exponents(n, T):
    # gamma1 and nabla_ad from the definitions, without fluctuation moments, Maxwell relations or c_V:
    # Richardson-extrapolated five-point differences (steps h = 4e-3 and 2e-3 in ln n and in ln T) of
    # ln P and sigma = s / n_net, then along constant sigma
    #   Gamma3 - 1 = (d ln T / d ln n)_sigma = -(d sigma / d ln n)_T / (d sigma / d ln T)_n,
    #   gamma1 = (d ln P / d ln n)_T + (d ln P / d ln T)_n (Gamma3 - 1),  nabla_ad = (Gamma3 - 1) / gamma1.
    # In a degenerate gas (d ln P / d ln T)_n is below 1e-9 and its difference quotient keeps only a few
    # digits, but it enters gamma1 multiplied by Gamma3 - 1 and nabla_ad not at all.
    cache = {}

    def at(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = _ss_lnP_sigma(n * math.exp(a), T * math.exp(b))
        return cache[(a, b)]

    def d5(g, h):
        return [(-g(2 * h)[i] + 8.0 * g(h)[i] - 8.0 * g(-h)[i] + g(-2 * h)[i]) / (12.0 * h) for i in (0, 1)]

    def est(h):
        dn = d5(lambda e: at(e, 0.0), h)
        dT = d5(lambda e: at(0.0, e), h)
        g3m1 = -dn[1] / dT[1]
        g1 = dn[0] + dT[0] * g3m1
        return g1, g3m1 / g1

    a, b = est(4e-3), est(2e-3)
    return b[0] + (b[0] - a[0]) / 15.0, b[1] + (b[1] - a[1]) / 15.0


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
    psi = degeneracy_parameter(rho * Ye, T)
    n_minus, n_plus, n_net = pair_densities(T, psi)
    P, u, s = pair_thermodynamics(T, psi)
    cv = specific_heat(rho * Ye, T)
    gamma1, nabla_ad = _ss_exponents(rho * Ye * _SS_NA, T)
    return {"psi": psi, "n_minus": n_minus, "n_plus": n_plus, "P": P, "u": u, "s": s, "cv": cv,
            "gamma1": float(gamma1), "nabla_ad": float(nabla_ad)}
