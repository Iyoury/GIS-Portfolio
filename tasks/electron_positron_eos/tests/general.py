# --- test case 0 ---
# the complete equation of state, with its adiabatic exponents, of a carbon-oxygen white-dwarf interior
# (rho = 2e6 g cm^-3, T = 1e9 K, Ye = 0.5) and of a pair plasma (rho = 1e-3 g cm^-3, T = 1e10 K, Ye = 0.5).
# psi, n_minus, n_plus, P, u, s and c_V of these states are the version-3 40-digit targets (tanh-sinh
# quadrature of the defining integrals with breakpoints every k T over the Fermi edge, psi by an mpmath root
# search, c_V by a centred difference of u at constant n_net, T +- 1e-7 T); they agree with the version-6
# routes named below within 1e-14 (the c_V differences are the 1e-14 truncation of that centred difference).
# target evidence, gamma1 and nabla_ad (independent_recomputation): route Q (white dwarf; 50-digit
# quadrature, exponents from the definitions by five-point differences at h = 1e-10, cross-checked by the
# 34-digit Gauss-Legendre route G with analytic response moments) and route P0 (pair plasma; the
# charge-conjugation limit, cross-checked by the 200-term Bessel series within 1.2e-24). Limits: quadrature
# error below 1e-17; P0 error O(psi^2) = 1.3e-22.
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
        for _t_k, _t_want in zip(_T_KEYS, [float(_t_x) for _t_x in _t_target]):
            _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 1 ---
# thermodynamic consistency of the complete equation of state of a hot, partly pair-populated gas
# (rho = 1e6 g cm^-3, T = 3e9 K, Ye = 0.5: psi about 0.65, n_plus / n_minus about 0.29), all from
# electron_positron_eos. Each bound below is the first-order error budget when every returned quantity has
# its stated accuracy (psi, n_minus, n_plus, P, u, s relative 1e-10, cv, gamma1, nabla_ad relative 1e-8),
# computed from the returned values:
#  (a) charge neutrality n_minus - n_plus = rho Ye N_A, within 1e-10 (n_minus + n_plus) / (rho Ye N_A)
#      (about 1.8e-10 here);
#  (b) the Euler relation u_tot + P - T s = mu n_net, with u_tot = u + n_net m_e c^2, n_net = rho Ye N_A and
#      mu = psi k T (rest energy included), within 1e-10 (|u| + |P| + T |s| + |mu n_net|) / |mu n_net|
#      (n_net m_e c^2 is exact; about 2.6e-9 here);
#  (c) c_V = (d u / d T) and c_V = T (d s / d T) at constant rho and Ye (constant n_net), by the five-point
#      centred difference with h = 5e-3 T of the returned u and s, within 1e-8 (c_V itself)
#      + c 1e-10 (|X(T+2h)| + 8 |X(T+h)| + 8 |X(T-h)| + |X(T-2h)|) / (12 h c_V) (X = u with c = 1, or s with
#      c = T; about 6.5e-9 and 1.0e-8 here) + 1.2e-8 for the truncation error of the stencil (its h^4 term,
#      4.8e-9 for u and 6.0e-9 for s at this state, measured from the difference formula at h and 2 h,
#      whose ratio is 16);
#  (d) gamma1 and nabla_ad as the exponents of the adiabat at constant s / rho: with five-point centred
#      differences (h = 5e-3 rho and 5e-3 T) of the returned ln P and sigma = s / rho,
#      chi_rho = rho dlnP/drho, chi_T = T dlnP/dT, Gamma3 - 1 = -(rho dsigma/drho) / (T dsigma/dT),
#      gamma1_FD = chi_rho + chi_T (Gamma3 - 1) and nabla_FD = (Gamma3 - 1) / gamma1_FD, within 1e-8 (the
#      returned exponent itself) + the first-order propagation of 1e-10 relative errors of every returned P
#      and s through the differences and these formulas (6.4e-8 for gamma1 and 1.1e-7 for nabla_ad here)
#      + 2e-8 for the truncation error of the stencils (5.5e-9 for gamma1 and 8.8e-9 for nabla_ad, measured
#      on the exact Bessel-series equation of state at h and 2 h, ratio 16);
#  (e) every returned value against the independent targets of this state: route S, the Bessel-function
#      series of the Fermi-Dirac integrals (90 terms, exact for psi < m_e c^2 / k T = 1.98) with
#      definition-level exponents, cross-checked by route Q (quadrature) within 4.3e-18 for gamma1 and
#      nabla_ad (target evidence: independent_recomputation; limits: S exact here).
import numpy as np


def _crown_case():
    import numpy as np

    _T_MEC2 = 8.1871057769e-7
    _T_KB = 1.380649e-16
    _T_NA = 6.02214076e23

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _t_rho, _t_T, _t_Ye = 1e6, 3e9, 0.5
    _t_n = _t_rho * _t_Ye * _T_NA
    _t_o = electron_positron_eos(_t_rho, _t_T, _t_Ye)
    # (a) charge neutrality
    _t_tol = 1e-10 * (abs(_t_o["n_minus"]) + abs(_t_o["n_plus"])) / _t_n
    assert abs((_t_o["n_minus"] - _t_o["n_plus"]) / _t_n - 1.0) < _t_tol, (_t_o, _t_tol)
    # (b) Euler relation
    _t_utot = _t_o["u"] + _t_n * _T_MEC2
    _t_lhs = _t_utot + _t_o["P"] - _t_T * _t_o["s"]
    _t_rhs = _t_o["psi"] * _T_KB * _t_T * _t_n
    _t_tol = 1e-10 * (abs(_t_o["u"]) + abs(_t_o["P"]) + _t_T * abs(_t_o["s"]) + abs(_t_rhs)) / abs(_t_rhs)
    assert abs(_t_lhs / _t_rhs - 1.0) < _t_tol, (_t_lhs, _t_rhs, _t_tol)
    # (c) c_V as the temperature derivative of u and T times that of s, at constant rho Ye
    _t_h = 5e-3 * _t_T
    _t_v = {_t_k: electron_positron_eos(_t_rho, _t_T + _t_k * _t_h, _t_Ye) for _t_k in (-2, -1, 1, 2)}
    for _t_key, _t_scale in (("u", 1.0), ("s", _t_T)):
        _t_x = {_t_k: _t_v[_t_k][_t_key] for _t_k in _t_v}
        _t_d = _t_scale * (-_t_x[2] + 8.0 * _t_x[1] - 8.0 * _t_x[-1] + _t_x[-2]) / (12.0 * _t_h)
        _t_noise = _t_scale * 1e-10 * (abs(_t_x[2]) + 8.0 * abs(_t_x[1]) + 8.0 * abs(_t_x[-1])
                                       + abs(_t_x[-2])) / (12.0 * _t_h)
        _t_tol = 1e-8 + _t_noise / abs(_t_o["cv"]) + 1.2e-8
        assert abs(_t_d / _t_o["cv"] - 1.0) < _t_tol, (_t_key, _t_d, _t_o["cv"], _t_tol)
    # (d) gamma1 and nabla_ad along the adiabat at constant s / rho (fixed Ye)
    _t_hr = 5e-3 * _t_rho
    _t_w = {_t_k: electron_positron_eos(_t_rho + _t_k * _t_hr, _t_T, _t_Ye) for _t_k in (-2, -1, 1, 2)}
    _t_c = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    _t_lp = lambda _t_d, _t_k: np.log(_t_d[_t_k]["P"])
    _t_rT = {_t_k: _t_v[_t_k] for _t_k in _t_c}
    _t_rR = {_t_k: _t_w[_t_k] for _t_k in _t_c}
    _t_sgT = {_t_k: _t_v[_t_k]["s"] / _t_rho for _t_k in _t_c}
    _t_sgR = {_t_k: _t_w[_t_k]["s"] / (_t_rho + _t_k * _t_hr) for _t_k in _t_c}
    _t_chiR = _t_rho * sum(_t_c[_t_k] * _t_lp(_t_rR, _t_k) for _t_k in _t_c) / (12.0 * _t_hr)
    _t_chiT = _t_T * sum(_t_c[_t_k] * _t_lp(_t_rT, _t_k) for _t_k in _t_c) / (12.0 * _t_h)
    _t_A = _t_rho * sum(_t_c[_t_k] * _t_sgR[_t_k] for _t_k in _t_c) / (12.0 * _t_hr)
    _t_B = _t_T * sum(_t_c[_t_k] * _t_sgT[_t_k] for _t_k in _t_c) / (12.0 * _t_h)
    _t_g3 = -_t_A / _t_B
    _t_g1 = _t_chiR + _t_chiT * _t_g3
    _t_nab = _t_g3 / _t_g1
    # first-order propagation of relative errors 1e-10 of every returned P and s
    _t_eP = 1e-10 * 18.0 / (12.0 * 5e-3)
    _t_eA = _t_rho * 1e-10 * sum(abs(_t_c[_t_k] * _t_sgR[_t_k]) for _t_k in _t_c) / (12.0 * _t_hr)
    _t_eB = _t_T * 1e-10 * sum(abs(_t_c[_t_k] * _t_sgT[_t_k]) for _t_k in _t_c) / (12.0 * _t_h)
    _t_eg3 = abs(_t_g3) * (_t_eA / abs(_t_A) + _t_eB / abs(_t_B))
    _t_eg1 = _t_eP + abs(_t_g3) * _t_eP + abs(_t_chiT) * _t_eg3
    _t_tol_g1 = 1e-8 + _t_eg1 / abs(_t_g1) + 2e-8
    _t_tol_nab = 1e-8 + _t_eg3 / abs(_t_g3) + _t_eg1 / abs(_t_g1) + 2e-8
    assert abs(_t_o["gamma1"] / _t_g1 - 1.0) < _t_tol_g1, (_t_o["gamma1"], _t_g1, _t_tol_g1)
    assert abs(_t_o["nabla_ad"] / _t_nab - 1.0) < _t_tol_nab, (_t_o["nabla_ad"], _t_nab, _t_tol_nab)
    # (e) independent targets of this state (route S)
    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_WANT = (0.6486513228130821646768135, 4.213963436506592180276093e+29, 1.202893056506592245156733e+29, 230129716760110528128621.4, 683195960316092445100529.5, 359649128822510.8047512153, 1057700648681477.994867297, 1.356763462745362799443262, 0.2024551470114841150948779)
    for _t_k, _t_want in zip(_T_KEYS, _T_WANT):
        _t_tol = 1e-8 if _t_k in ("cv", "gamma1", "nabla_ad") else 1e-10
        assert _t_rel(_t_o[_t_k], _t_want) < _t_tol, (_t_k, _t_o[_t_k], _t_want)


_crown_case()

# --- test case 2 ---
# cold, completely degenerate gases (E_F from 6e4 to 1.3e5 k T): rho = 1e13 g cm^-3, T = 1.2e7 K;
# rho = 2e13 g cm^-3, T = 1e7 K (the corner of the domain, rho Ye = 1e13 g cm^-3); rho = 2e12 g cm^-3,
# T = 1e7 K; all with Ye = 0.5 and all nine values. (d ln P / d ln T) at constant rho is 8e-10 to 4e-9 here.
# target evidence (independent_recomputation): the corner's psi, n_minus, n_plus, P, u, s, c_V are the
# version-3 40-digit targets (quadrature, within 1e-16 of route M); every other value is from route M (the
# Sommerfeld expansion, terms to theta^24, 60 digits, exponents from the definitions term by term),
# cross-checked by route Q within 1.7e-21 and by the zero-temperature closed forms within 1.4e-13 (their
# thermal correction). Limits: positrons and the lower end of the integral below e^-5000.
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (10000000000000.0, 12000000.0, 0.5): (85248.53688119610373416772, 3.011070379999999935119360e+36, 0.0, 1.063140082636891018625853e+32, 3.164982610341944400791784e+32, 48131743507712112.23058938, 48131743481563473.76582521, 1.333355724483034272384698, 0.2500042019486261495150064),
        (20000000000000.0, 10000000.0, 0.5): (128886.9100517552405035025, 6.02214076e36, 6.015906150270230151335683e-56207, 2.67897843472169752993636e+32, 7.987971624185773846332954e+32, 63669921072302993.04776527, 63669921057170992.70326489, 1.333347441128975670624385, 0.2500026467211563819912430),
        (2000000000000.0, 10000000.0, 0.5): (59826.30991611336868202790, 6.022140759999999870238720e+35, 0.0, 1.243375864526248506335782e+31, 3.681556419581798793460370e+31, 13717797345812305.70838126, 13717797330678361.17456363, 1.333398756379680322038698, 0.2500122937886535476341015),
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

# --- test case 3 ---
# a pair plasma softened by pair creation (rho = 2e-10 g cm^-3, T = 1e9 K, Ye = 0.5: about 9e12 pairs per
# net electron, psi about 5.6e-14, gamma1 about 1.235), all nine values. target evidence
# (independent_recomputation): route P0, the charge-conjugation limit (1-D integrals of the psi = 0 gas in
# 50 digits; error O(psi^2) = 3e-27), cross-checked by route S (Bessel-function series, 20 terms) to all 25
# digits, and by an earlier 40-digit Bessel-series adiabat (gamma1 = 1.23543954658035752, nabla_ad =
# 0.114996207858432367).
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv", "gamma1", "nabla_ad")
    _T_EOS = {
        (2e-10, 1000000000.0, 0.5): (5.550389008533468532415132e-14, 5.429401447875313921095587e+26, 5.429401447874711707019587e+26, 149983167725161293647.3845, 1154261298403621148477.112, 1304244466128.831746027901, 9180238928315.666598808444, 1.235439546580357527946329, 0.1149962078584323710693016),
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
