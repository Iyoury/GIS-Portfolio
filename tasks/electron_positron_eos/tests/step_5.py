# --- test case 0 ---
# The complete equation of state of a carbon-oxygen white-dwarf interior (rho = 2e6 g cm^-3, T = 1e9 K,
# Ye = 0.5, psi about 7.8) and of a pair plasma (rho = 1e-3 g cm^-3, T = 1e10 K, Ye = 0.5, about 5e10 pairs
# per net electron), all nine values, and the stated form of the result: a dict with exactly the nine keys,
# whose values are Python floats.
# psi, n_minus, n_plus, P, u, s and c_V of these states are the version-3 40-digit targets (tanh-sinh
# quadrature of the defining integrals with breakpoints every k T over the Fermi edge, psi by an mpmath root
# search, c_V by a centred difference of u at constant n_net, T +- 1e-7 T); they agree with the version-6
# routes named below within 1e-14 (the c_V differences are the 1e-14 truncation of that centred difference).
# target evidence, gamma1 and nabla_ad (independent_recomputation, none from a solution):
#   white dwarf: route Q, 50-digit tanh-sinh quadrature of n_net, P and s (Gibbs entropy integrand), psi
#   re-solved, gamma1 = (n / P) d(P, sigma) / d(n, sigma) and nabla_ad = (P / T) d(T, sigma) / d(P, sigma)
#   (sigma = s / n_net) from five-point differences in (ln T, ln psi) at h = 1e-10 (h and 2h agree within
#   3e-38); cross-checked within 9.3e-19 by route G, 34-digit Gauss-Legendre quadrature of the response
#   moments with the analytic (T, mu) Jacobian (no differences). Limits: exact definitions; quadrature error
#   below 1e-17.
#   pair plasma: route P0, the charge-conjugation limit (P, s, u_tot even and n_net odd in psi): gamma1 =
#   T (dP0/dT)^2 / (P0 c_V0), nabla_ad = P0 / (T dP0/dT) with 1-D integrals of the psi = 0 gas in 50 digits;
#   cross-checked by route S, the Bessel-function series of the Fermi-Dirac integrals (200 terms, exact for
#   psi < m_e c^2 / k T), within 1.2e-24. Limits: error O(psi^2) = 1.3e-22.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (2000000.0, 1000000000.0, 0.5): (7.798772316595553735539734, 6.02214298916745876143701e+29, 222916745876143700987606.6, 114478573268634348539048.3, 208291874619394925772455.9, 167383473004818.0744619993, 130656532160103.4906555759, 1.480345605824271699195458, 0.3152906029200376139301397),
        (0.001, 10000000000.0, 0.5): (1.143531551258349070810209e-11, 1.42657113518383029437691e+31, 1.42657113515371959057691e+31, 4.109758750359864921587657e+25, 1.288730256177001349765261e+26, 16997061312154530.37094254, 52269535406676546.90725823, 1.344877454920806575882875, 0.2417923119110591744603644),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V, gamma1, nabla_ad

    for _t_args, _t_target in _T_EOS.items():
        _t_out = electron_positron_eos(*_t_args)
        assert (isinstance(_t_out, dict) and set(_t_out) == set(_T_KEYS)
                and all(type(_t_v) is float for _t_v in _t_out.values())), _t_out
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 1 ---
# A helium-burning state (rho = 1e8 g cm^-3, T = 3e8 K, Ye = 0.43, psi about 73, n_plus about 2e-12
# cm^-3) and a degenerate stellar core (rho = 3.5e12 g cm^-3, T = 2e7 K, Ye = 0.46, psi about 3.5e4), all
# nine values.
# psi, n_minus, n_plus, P, u, s and c_V of these states are the version-3 40-digit targets (tanh-sinh
# quadrature of the defining integrals with breakpoints every k T over the Fermi edge, psi by an mpmath root
# search, c_V by a centred difference of u at constant n_net, T +- 1e-7 T); they agree with the version-6
# routes named below within 1e-14 (the c_V differences are the 1e-14 truncation of that centred difference).
# target evidence, gamma1 and nabla_ad (independent_recomputation):
#   helium-burning state: route Q (as in case 0), cross-checked by route M, the Sommerfeld expansion of the
#   degenerate gas (closed-form zero-temperature integrals plus terms to theta^24, 60 digits, derivatives in
#   (T, mu) term by term), within 1.5e-18 (the truncation of the asymptotic series at E_F = 53 k T).
#   degenerate core: route M (kmax 10 and 12 agree to all digits), cross-checked by route Q within 1.3e-21 and
#   by the zero-temperature closed forms of the Chandrasekhar gas within 2.9e-13 (their thermal correction).
#   Limits: M neglects positrons and the lower end of the momentum integral, below e^-1000 here.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (100000000.0, 300000000.0, 0.43): (72.55703066526062842798715, 2.5895205268e+31, 2.2091457640632335647917e-12, 1.751430670766828173376139e+25, 3.92645989995075988402764e+25, 524678003752616.3780682717, 524212451387668.4186415838, 1.374344363018786789787206, 0.2605123701296794384451733),
        (3500000000000.0, 20000000.0, 0.46): (35058.90723666841769594433, 9.6956466236e+35, 9.597418536528029851685467e-15329, 2.346290944286472048010346e+31, 6.960500198217961422680609e+31, 37687080202983614.54510263, 37687080081917452.09285463, 1.333380973844477759374144, 0.2500089474956492274936809),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V, gamma1, nabla_ad

    for _t_args, _t_target in _T_EOS.items():
        _t_out = electron_positron_eos(*_t_args)
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 2 ---
# classical gases with pairs, all nine quantities in closed form: both species follow Maxwell-Juettner
# statistics (electron occupation exponent above 27, so Fermi corrections below 1e-12), with a = 1 / theta,
# N = K2(a) / a, E = K1(a) / a + 3 K2(a) / a^2, F = K0(a) / a + 5 K1(a) / a^2 + 12 K2(a) / a^3 (the moments
# int p^2 eps^k e^{-a eps} dp) and B = N / (pi^2 lambda^3): psi = asinh(rho Ye N_A / (2 B)),
# n_minus, n_plus = B e^{+-psi}, P = (n_minus + n_plus) k T, <eps> = E / N,
# u = m_e c^2 [(n_minus + n_plus) <eps> - rho Ye N_A], s = k [(n_minus + n_plus) (<eps> / theta + 1)
# - psi rho Ye N_A] and c_V = (2 k / (pi^2 lambda^3 theta^2)) [F cosh psi - (E^2 / N) tanh psi sinh psi];
# the exponents from the first law at constant s / rho with the pair number in equilibrium:
# N_tot = n_minus + n_plus = sqrt((rho Ye N_A)^2 + 4 B^2), (dP/dT)_n = k N_tot + 4 k T B (dB/dT) / N_tot with
# dB/dT = (k / (m_e c^2 pi^2 lambda^3)) (3 K2(a) + a K1(a)), (dP/dn)_T = k T rho Ye N_A / N_tot,
# gamma1 = (rho Ye N_A / P) (dP/dn)_T + T (dP/dT)_n^2 / (P c_V), nabla_ad = (dP/dT)_n / (c_V gamma1);
# rho = 2e-10 g cm^-3, T = 1.4e8 K, Ye = 0.5 (psi about 9.8; pair creation adds 1.5e-5 to c_V),
# rho = 1e-8 g cm^-3, T = 1.2e8 K, Ye = 1 (psi about 21.7) and a pair plasma at rho = 2e-10 g cm^-3,
# T = 2e8 K, Ye = 0.5 (psi about 0.0156, 0.97 positrons per electron, gamma1 = 1.063: the heat goes into
# pair creation). target evidence (special_case): these closed forms agree with the exact Bessel-function
# series of the Fermi-Dirac integrals (route S, definition-level exponents) within 3.2e-13 for the seven
# equation-of-state values and 2.2e-14 for gamma1 and nabla_ad; limits: occupation exponent above 27.
import numpy as np


def _crown_case():
    import numpy as np

    import mpmath as _t_mp

    _T_MEC2 = 8.1871057769e-7
    _T_KB = 1.380649e-16
    _T_LC = 3.8615926796e-11
    _T_NA = 6.02214076e23

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")

    for _t_rho, _t_T, _t_Ye in ((2e-10, 1.4e8, 0.5), (1e-8, 1.2e8, 1.0), (2e-10, 2e8, 0.5)):
        with _t_mp.workdps(40):
            _t_n = _t_mp.mpf(_t_rho) * _t_mp.mpf(_t_Ye) * _t_mp.mpf(_T_NA)
            _t_kb = _t_mp.mpf(_T_KB)
            _t_th = _t_kb * _t_T / _t_mp.mpf(_T_MEC2)
            _t_a = 1 / _t_th
            _t_pref = 1 / (_t_mp.pi ** 2 * _t_mp.mpf(_T_LC) ** 3)
            _t_K0, _t_K1, _t_K2 = _t_mp.besselk(0, _t_a), _t_mp.besselk(1, _t_a), _t_mp.besselk(2, _t_a)
            _t_N = _t_K2 / _t_a
            _t_E = _t_K1 / _t_a + 3 * _t_K2 / _t_a ** 2
            _t_F = _t_K0 / _t_a + 5 * _t_K1 / _t_a ** 2 + 12 * _t_K2 / _t_a ** 3
            _t_B = _t_pref * _t_N
            _t_psi = _t_mp.asinh(_t_n / (2 * _t_B))
            _t_nm, _t_np = _t_B * _t_mp.exp(_t_psi), _t_B * _t_mp.exp(-_t_psi)
            _t_g = _t_E / _t_N
            _t_P = (_t_nm + _t_np) * _t_kb * _t_T
            _t_cv = (2 * _t_kb * _t_pref / _t_th ** 2
                     * (_t_F * _t_mp.cosh(_t_psi) - _t_E ** 2 / _t_N * _t_mp.tanh(_t_psi) * _t_mp.sinh(_t_psi)))
            _t_Nt = _t_mp.sqrt(_t_n ** 2 + 4 * _t_B ** 2)
            _t_dB = _t_pref * _t_kb / _t_mp.mpf(_T_MEC2) * (3 * _t_K2 + _t_a * _t_K1)
            _t_PT = _t_kb * _t_Nt + 4 * _t_kb * _t_T * _t_B * _t_dB / _t_Nt
            _t_Pn = _t_kb * _t_T * _t_n / _t_Nt
            _t_g1 = _t_n * _t_Pn / _t_P + _t_T * _t_PT ** 2 / (_t_P * _t_cv)
            _t_want = [
                _t_psi, _t_nm, _t_np, _t_P,
                _t_mp.mpf(_T_MEC2) * ((_t_nm + _t_np) * _t_g - _t_n),
                _t_kb * ((_t_nm + _t_np) * (_t_g / _t_th + 1) - _t_psi * _t_n),
                _t_cv, _t_g1, _t_PT / (_t_cv * _t_g1),
            ]
            _t_want = [float(_t_x) for _t_x in _t_want]
        _t_out = electron_positron_eos(_t_rho, _t_T, _t_Ye)
        for _t_k, _t_w in zip(_T_KEYS, _t_want):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            assert _t_rel(_t_out[_t_k], _t_w) < _t_tol, (_t_rho, _t_T, _t_Ye, _t_k, _t_out[_t_k], _t_w)


_crown_case()

# --- test case 3 ---
# strongly degenerate gases: an ultra-relativistic gas (rho = 2e13 g cm^-3, T = 1e9 K, Ye = 0.5, E_F about
# 1300 k T) and a mildly relativistic one (rho = 4e6 g cm^-3, T = 1e7 K, Ye = 0.5, E_F about 370 k T); here
# the
# adiabat is far from the one at constant entropy density (holding s fixed would make nabla_ad 3 to 4 times
# larger).
# psi, n_minus, n_plus, P, u, s and c_V are the version-5 Sommerfeld targets (route M below, terms to k = 8;
# n_plus the Maxwell-Juettner value theta K2(1/theta) e^{-psi} / (pi^2 lambda^3), below 1e-500 cm^-3).
# target evidence, gamma1 and nabla_ad (independent_recomputation): route M, the Sommerfeld expansion with
# mu = psi theta (rest energy included): each density is (1 / (pi^2 lambda^3)) [int_1^mu H deps + sum_k
# 2 (1 - 2^{1-2k}) zeta(2k) theta^{2k} H^{(2k-1)}(mu)], H = eps p for n, p^3 / 3 for P / (m_e c^2), terms to
# k = 12 in 60-digit mpmath (k = 10 agrees within 1e-33), s = (dP/dT) at constant mu, and the exponents from
# the (theta, mu) derivatives term by term through the definitions (Jacobian chain rule at constant s /
# n_net);
# cross-checked by route Q within 3.8e-22 and 4.2e-18. Limits: positrons and the lower end of the integral
# below e^-190.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (20000000000000.0, 1000000000.0, 0.5): (1288.86654822270575456093, 6.02214076e+36, 9.715760937524490143851642e-534, 2.678989045519733471283266e+32, 7.988003455906072934178029e+32, 6366984542011694531.149651, 6366969410111229921.828398, 1.333347441045153807425956, 0.2500026467347764942018664),
        (4000000.0, 10000000.0, 0.5): (959.0270475026477711036308, 1.204428152e+30, 1.42223500707363258703769e-648, 219531284305144376444627.0, 389177260304632978889267.2, 2770546806919.348960127573, 2770509852894.45173713809, 1.495730731345984772986431, 0.3080588198366754455387633),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V, gamma1, nabla_ad

    for _t_args, _t_target in _T_EOS.items():
        _t_out = electron_positron_eos(*_t_args)
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 4 ---
# the endpoint Ye = 1 (pure hydrogen) is in the domain: rho = 1e6, T = 1e9 K, Ye = 1 has rho Ye = 1e6
# g cm^-3 like the white-dwarf state (2e6, 1e9, 0.5), so the equation of state, which depends on rho and
# Ye only through rho Ye (and gamma1, nabla_ad, defined at fixed Ye, only through rho Ye and T), must
# reproduce the same targets (case 0: 40-digit version-3 quadrature; gamma1, nabla_ad from route Q)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_WANT = (7.798772316595553735539734, 6.02214298916745876143701e+29, 222916745876143700987606.6, 114478573268634348539048.3, 208291874619394925772455.9, 167383473004818.0744619993, 130656532160103.4906555759, 1.480345605824271699195458, 0.3152906029200376139301397)      # psi, n_minus, n_plus, P, u, s, c_V, gamma1, nabla_ad at rho Ye = 1e6, T = 1e9 K

    _t_out = electron_positron_eos(1e6, 1e9, 1.0)
    for _t_k, _t_want in zip(_T_KEYS, _T_WANT):
        _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
        assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 5 ---
# the corners of the domain, with T and rho Ye at their inclusive endpoints: a cold, strongly degenerate
# gas (rho = 2e13, T = 1e7 K, Ye = 0.5, so rho Ye = 1e13 g cm^-3 exactly; E_F about 1.3e5 k T, where
# (d ln P / d ln T) at constant rho is 7.9e-10) and a pair plasma (rho = 2e-10, T = 1e11 K, Ye = 0.5, so
# rho Ye = 1e-10 g cm^-3 exactly; psi about 2e-21). psi and c_V are the version-3 40-digit (rho_Ye, T)
# targets; P, u, s and the pair-plasma densities come from 40-digit mpmath quadrature (tanh-sinh and
# Gauss-Legendre, agreeing to 25 digits) at that psi; in the degenerate gas n_minus = rho Ye N_A + n_plus
# with n_plus about 6e-56207 cm^-3 (within 1e-250 of 0); these agree with the version-6 routes within 1e-14.
# target evidence, gamma1 and nabla_ad (independent_recomputation): degenerate corner, route M (case 3),
# cross-checked by route Q within 3.8e-22 and by the zero-temperature closed forms of the Chandrasekhar gas
# (Gamma1 = 8 x^5 / (3 sqrt(1 + x^2) [x (2 x^2 - 3) sqrt(1 + x^2) + 3 asinh x]), Gamma3 - 1 = (2 + x^2) /
# (3 (1 + x^2)), x = p_F / (m_e c) = 217.3) within 6.3e-15, their thermal correction; pair corner, route P0
# (case 0), cross-checked by route Q within 4e-20. Limits: as in cases 0 and 3.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (20000000000000.0, 10000000.0, 0.5): (128886.9100517552405035025, 6.02214076e36, 6.015906150270230151335683e-56207, 2.67897843472169752993636e+32, 7.987971624185773846332954e+32, 63669921072302993.04776527, 63669921057170992.70326489, 1.333347441128975670624385, 0.2500026467211563819912430),
        (2e-10, 100000000000.0, 0.5): (2.170425950671224465122001e-21, 1.520486632169166839841366e+34, 1.520486632169166839835344e+34, 4.40998243289230441302235e+29, 1.323665810506425580785849e+30, 17646640537956560220.88133, 52953392105589435431.00965, 1.333501269354263228825673, 0.2499049279893684559662543),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V, gamma1, nabla_ad

    for _t_args, _t_target in _T_EOS.items():
        _t_out = electron_positron_eos(*_t_args)
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 6 ---
# Ye outside (0, 1], rho Ye outside [1e-10, 1e13] g cm^-3 or T outside [1e7, 1e11] K
import numpy as np


def _crown_case():
    import numpy as np

    for _t_bad in ((1e6, 1e9, 0.0), (1e6, 1e9, 1.2), (1e-11, 1e9, 0.5), (3e13, 1e9, 0.5), (1e6, 2e11, 0.5)):
        try:
            electron_positron_eos(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("electron_positron_eos%r must raise ValueError" % (_t_bad,))


_crown_case()

# --- test case 7 ---
# cold, completely degenerate gases near the dense, cool edge of the domain: rho = 1e13 g cm^-3, T = 1.2e7 K,
# Ye = 0.5 (rho Ye = 5e12 g cm^-3, x = p_F / (m_e c) = 172.5, E_F about 8.5e4 k T); rho = 2e12 g cm^-3,
# T = 1e7 K, Ye = 0.5 (x = 100.9, E_F about 5.9e4 k T); rho = 2e9 g cm^-3, T = 1.2e7 K, Ye = 0.5 (x = 10.1,
# E_F about 4.5e3 k T), all nine values. Here (d ln P / d ln T) at constant rho is of order (k T / E_F)^2
# (1.8e-9, 3.7e-9 and 5.4e-7), and gamma1, nabla_ad approach 4/3 and 1/4 (ultra-relativistic) up to that
# order.
# target evidence (independent_recomputation): every value from route M (case 3; positrons below e^-5000,
# so n_plus is 0 within 1e-250 cm^-3), cross-checked by route Q within 2e-19 and by the zero-temperature
# closed forms (case 5) within 2.3e-14, 1.4e-13 and 1.9e-9 (their thermal correction, (k T / E_F)^2 times
# a number of order one). Limits: as in case 3.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (10000000000000.0, 12000000.0, 0.5): (85248.53688119610373416772, 3.011070379999999935119360e+36, 0.0, 1.063140082636891018625853e+32, 3.164982610341944400791784e+32, 48131743507712112.23058938, 48131743481563473.76582521, 1.333355724483034272384698, 0.2500042019486261495150064),
        (2000000000000.0, 10000000.0, 0.5): (59826.30991611336868202790, 6.022140759999999870238720e+35, 0.0, 1.243375864526248506335782e+31, 3.681556419581798793460370e+31, 13717797345812305.70838126, 13717797330678361.17456363, 1.333398756379680322038698, 0.2500122937886535476341015),
        (2000000000.0, 12000000.0, 0.5): (5009.711713597293419964022, 6.022140759999999870238720e+32, 0.0, 1.231717082916497306016177e+27, 3.273613160425826363859985e+27, 165412153377004.7707423970, 165412126798414.3099071280, 1.339521870281569108493869, 0.2512662347374253974075928),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V, gamma1, nabla_ad

    for _t_args, _t_target in _T_EOS.items():
        _t_out = electron_positron_eos(*_t_args)
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 8 ---
# pair plasmas whose pair creation softens the gas: rho = 2e-10 g cm^-3, T = 1e9 K, Ye = 0.5 (gamma1 about
# 1.235, far below 4/3, since heat goes into the rest energy of new pairs; psi about 5.6e-14) and rho = 2e-6
# g cm^-3, T = 1.5e9 K, Ye = 0.5 (gamma1 about 1.292; psi about 3.7e-11), all nine values.
# target evidence (independent_recomputation): route P0 (case 0) for all nine values, cross-checked by
# route S (Bessel-function series, 20 and 30 terms, exact for psi < m_e c^2 / k T, definition-level
# exponents from 5-point differences at h = 1e-10) within 6e-22. Limits: P0 error O(psi^2) below 1.4e-21.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (2e-10, 1000000000.0, 0.5): (5.550389008533468532415132e-14, 5.429401447875313921095587e+26, 5.429401447874711707019587e+26, 149983167725161293647.3845, 1154261298403621148477.112, 1304244466128.831746027901, 9180238928315.666598808444, 1.235439546580357527946329, 0.1149962078584323710693016),
        (2e-06, 1500000000.0, 0.5): (3.713137429582492009010039e-11, 8.154021755611578081689622e+27, 8.154021755009364005689622e+27, 3386687527938383732660.219, 19696114303033261477646.27, 15388534554309.78949624100, 81160513672968.48408548038, 1.292307520617508863781116, 0.1467190834399450633241680),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V, gamma1, nabla_ad

    for _t_args, _t_target in _T_EOS.items():
        _t_out = electron_positron_eos(*_t_args)
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 9 ---
# pair plasmas at T = 5e9 K that differ only in their small net electron density: rho = 2e-10 and 2e2
# g cm^-3, Ye = 0.5 (psi about 2.1e-17 and 2.1e-5). P, s and the total energy are even in psi and n_net is
# odd (charge conjugation), so c_V, gamma1 and nabla_ad of the two states differ only at order psi^2: the
# two outputs must agree within 2e-8 plus the difference of the two stored targets (1.3e-10), and each must
# meet its target. target evidence (independent_recomputation): rho = 2e-10, route P0 (case 0), cross-checked
# by route S (100 terms) to all 25 digits; rho = 2e2, route S (100 terms; h and 2h agree within 1.5e-38),
# which agrees with the psi -> 0 limit within 8.5e-13 (the order psi^2 terms).
# Limits: S exact for psi < m_e c^2 / k T = 1.19.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (2e-10, 5000000000.0, 0.5): (6256259625424889.940812264, 1.357356045927199719113353, 0.2240324401332989766768334),
        (200.0, 5000000000.0, 0.5): (6256259624639907.343007368, 1.357356045928359885929531, 0.2240324401362354976576267),
    }      # (rho, T, Ye): c_V, gamma1, nabla_ad

    _t_outs = []
    for _t_args, _t_target in _T_EOS.items():
        _t_out = electron_positron_eos(*_t_args)
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            assert _t_rel(_t_out[_t_k], _t_want) < 1e-8, (_t_args, _t_k, _t_out[_t_k], _t_want)
        _t_outs.append(_t_out)
    _t_tg = list(_T_EOS.values())
    for _t_i, _t_k in enumerate(_T_KEYS):
        _t_d = _t_rel(float(_t_tg[0][_t_i]), float(_t_tg[1][_t_i]))
        assert _t_rel(_t_outs[0][_t_k], _t_outs[1][_t_k]) < 2e-8 + _t_d, (_t_k, _t_outs[0][_t_k], _t_outs[1][_t_k])


_crown_case()
