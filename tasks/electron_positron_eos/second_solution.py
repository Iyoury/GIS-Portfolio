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
#     carried by one integral B in both u_T and u_mu n_T / n_mu.
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
            tot += _ss_quad(fp, math.sqrt(a * (a + 2.0)), math.sqrt(b * (b + 2.0)))
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
    T, psi, theta = _ss_check(T, psi)
    n_minus, n_plus = _ss_densities(theta, psi)
    if psi >= 1.0:
        n_net = n_minus - n_plus
    else:
        n_net = math.exp(_ss_log_net(theta, psi))
    return float(n_minus), float(n_plus), float(n_net)


def pair_thermodynamics(T, psi):
    T, psi, theta = _ss_check(T, psi)
    xp = lambda E: (E + 1.0) / theta + psi
    P = _SS_KB * T * _ss_PREF_int(lambda E, y: _ss_lnp(y) + _ss_lnp(xp(E)), theta, psi)
    u = _SS_MEC2 * _ss_PREF_int(lambda E, y: E * _ss_fermi(y) + (E + 2.0) * _ss_fermi(xp(E)), theta, psi)
    s = _SS_KB * _ss_PREF_int(lambda E, y: _ss_sig(y) + _ss_sig(xp(E)), theta, psi)
    return float(P), float(u), float(s)


def degeneracy_parameter(rho_Ye, T):
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
