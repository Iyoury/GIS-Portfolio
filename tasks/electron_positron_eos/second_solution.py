import numpy as np
import math
import warnings
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq

# Second solution: adaptive QUADPACK integration in the kinetic energy E (units m_e c^2), split at the
# Fermi edge and at +- 40 k T around it, instead of fixed Gauss-Legendre panels; the chemical potential
# by Brent's method on ln n_net(ln psi); the specific heat from two-pass centred moments evaluated with
# the same adaptive quadrature.

warnings.simplefilter("ignore", IntegrationWarning)   # QUADPACK reports round-off at 1e-13; checked against the targets

MEC2 = 8.1871057769e-7
KB = 1.380649e-16
LAMBDA_C = 3.8615926796e-11
NA = 6.02214076e23
PREF = 1.0 / (math.pi ** 2 * LAMBDA_C ** 3)


def _check(T, psi):
    T, psi = float(T), float(psi)
    if not (math.isfinite(T) and math.isfinite(psi)):
        raise ValueError("T and psi must be finite")
    if not 1e7 <= T <= 1e11:
        raise ValueError("need 1e7 K <= T <= 1e11 K")
    if not 0.0 < psi <= 1e6:
        raise ValueError("need 0 < psi <= 1e6")
    return T, psi, KB * T / MEC2


def _fermi(x):
    if x > 0:
        e = math.exp(-x)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(x))


def _sig(x):
    a = abs(x)
    e = math.exp(-a)
    return math.log1p(e) + a * e / (1.0 + e)


def _integrate(g, theta, psi):
    # int_0^inf p^2 dp g(E, y) with p^2 dp = p (E + 1) dE, y = (E - E_F) / theta given separately so
    # that the occupation near the edge is evaluated without cancellation
    EF = psi * theta - 1.0
    c = max(EF, 0.0)
    pts = sorted({max(c - 40.0 * theta, 0.0), c, c + 40.0 * theta, c + 80.0 * theta})
    pts = [0.0] + [q for q in pts if q > 0.0]
    tot = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        if a == 0.0 or EF <= 0.0 or b <= c - 40.0 * theta:
            f = lambda E: math.sqrt(E * (E + 2.0)) * (E + 1.0) * g(E, (E - EF) / theta)
            tot += quad(f, a, b, epsabs=0.0, epsrel=1e-13, limit=400)[0]
        else:
            # integrate in y over the edge window
            ya, yb = (a - EF) / theta, (b - EF) / theta
            def fy(y):
                E = EF + theta * y
                return theta * math.sqrt(E * (E + 2.0)) * (E + 1.0) * g(E, y)
            tot += quad(fy, ya, yb, epsabs=0.0, epsrel=1e-13, limit=400, points=[0.0] if ya < 0 < yb else None)[0]
    return tot


def pair_densities(T, psi):
    T, psi, theta = _check(T, psi)
    n_minus = PREF * _integrate(lambda E, y: _fermi(y), theta, psi)
    n_plus = PREF * _integrate(lambda E, y: _fermi((E + 1.0) / theta + psi), theta, psi)
    n_net = math.exp(_log_net(theta, psi))
    return n_minus, n_plus, n_net


def _log_net(theta, psi):
    # electrons minus positrons as one positive integrand: sinh(psi) / (cosh(b) + cosh(psi)), b = eps/theta,
    # scaled by e^{-(b0 - psi)} at the lowest energy b0 = 1/theta so that nothing underflows
    b0 = 1.0 / theta
    M0 = max(b0, psi)

    def g(E, y):
        b = (E + 1.0) / theta
        M = max(b, psi)
        return math.exp(psi - M - math.log(math.exp(b - M) + math.exp(-b - M) + math.exp(psi - M) + math.exp(-psi - M))
                        - (psi - M0))
    return math.log(PREF) + math.log(-math.expm1(-2.0 * psi)) + (psi - M0) + math.log(_integrate(g, theta, psi))


def pair_thermodynamics(T, psi):
    T, psi, theta = _check(T, psi)
    xp = lambda E: (E + 1.0) / theta + psi
    P = MEC2 * PREF / 3.0 * _integrate(lambda E, y: E * (E + 2.0) / (E + 1.0) * (_fermi(y) + _fermi(xp(E))), theta, psi)
    u = MEC2 * PREF * _integrate(lambda E, y: E * _fermi(y) + (E + 2.0) * _fermi(xp(E)), theta, psi)
    s = KB * PREF * _integrate(lambda E, y: _sig(y) + _sig(xp(E)), theta, psi)
    return P, u, s


def degeneracy_parameter(rho_Ye, T):
    rho_Ye, T = float(rho_Ye), float(T)
    if not (math.isfinite(rho_Ye) and math.isfinite(T)):
        raise ValueError("rho_Ye and T must be finite")
    if not 1e7 <= T <= 1e11:
        raise ValueError("need 1e7 K <= T <= 1e11 K")
    if not 1e-10 <= rho_Ye <= 1e13:
        raise ValueError("need 1e-10 <= rho_Ye <= 1e13 g / cm^3")
    theta = KB * T / MEC2
    lt = math.log(rho_Ye * NA)
    f = lambda lp: _log_net(theta, math.exp(lp)) - lt
    lo, hi = math.log(1e-300), 0.0
    while f(hi) < 0:
        lo, hi = hi, hi + 2.0
    return math.exp(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500))


def specific_heat(rho_Ye, T):
    psi = degeneracy_parameter(rho_Ye, T)
    T, psi, theta = _check(T, psi)
    fq = lambda x: _fermi(x) * _fermi(-x)                 # f (1 - f)
    xp = lambda E: (E + 1.0) / theta + psi
    I = lambda h: _integrate(h, theta, psi)
    WA = I(lambda E, y: fq(y))
    WB = I(lambda E, y: fq(xp(E)))
    S1a = I(lambda E, y: (E + 1.0) * fq(y))
    S1b = I(lambda E, y: (E + 1.0) * fq(xp(E)))
    S2a = I(lambda E, y: (E + 1.0) ** 2 * fq(y))
    S2b = I(lambda E, y: (E + 1.0) ** 2 * fq(xp(E)))
    ea = S1a / WA
    M2a = I(lambda E, y: (E + 1.0 - ea) ** 2 * fq(y))
    M2b = I(lambda E, y: (E + 1.0 - S1b / WB) ** 2 * fq(xp(E))) if WB > 0 else 0.0
    det = WA * M2a + WB * M2b + WB * S2a + 2.0 * S1a * S1b + WA * S2b
    return float(KB * PREF / theta ** 2 * det / (WA + WB))


def electron_positron_eos(rho, T, Ye):
    rho, T, Ye = float(rho), float(T), float(Ye)
    if not (math.isfinite(rho) and math.isfinite(T) and math.isfinite(Ye)):
        raise ValueError("rho, T and Ye must be finite")
    if not 0.0 < Ye <= 1.0:
        raise ValueError("need 0 < Ye <= 1")
    psi = degeneracy_parameter(rho * Ye, T)
    n_minus, n_plus, n_net = pair_densities(T, psi)
    P, u, s = pair_thermodynamics(T, psi)
    cv = specific_heat(rho * Ye, T)
    return {"psi": psi, "n_minus": n_minus, "n_plus": n_plus, "P": P, "u": u, "s": s, "cv": cv}
