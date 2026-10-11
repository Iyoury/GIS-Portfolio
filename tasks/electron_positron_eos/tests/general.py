# Independent targets: closed forms of the classical (Maxwell-Juettner) gas with modified Bessel
# functions, and values computed once with mpmath at 40 digits by tanh-sinh quadrature of the defining
# integrals in the momentum (breakpoints every k T over the Fermi edge), the chemical potential by a
# root search on those integrals and the specific heat by a centred difference of the energy at constant
# net electron density (T +- 1e-7 T), all in mpmath; positron densities of a classical positron gas
# (eps/kT + psi >= 40) from the Maxwell-Juettner formula.

# --- test case 0 ---
# the complete equation of state of a carbon-oxygen white-dwarf interior (rho = 2e6, T = 1e9,
# Ye = 0.5) and of a pair plasma (rho = 1e-3, T = 1e10), against 40-digit targets
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_EOS = {
        (2000000.0, 1000000000.0, 0.5): (7.798772316595553735539734, 6.02214298916745876143701e+29, 222916745876143700987606.6, 114478573268634348539048.3, 208291874619394925772455.9, 167383473004818.0744619993, 130656532160103.4906555759),
        (0.001, 10000000000.0, 0.5): (1.143531551258349070810209e-11, 1.42657113518383029437691e+31, 1.42657113515371959057691e+31, 4.109758750359864921587657e+25, 1.288730256177001349765261e+26, 16997061312154530.37094254, 52269535406676546.90725823),
        (100000000.0, 300000000.0, 0.43): (72.55703066526062842798715, 2.5895205268e+31, 2.2091457640632335647917e-12, 1.751430670766828173376139e+25, 3.92645989995075988402764e+25, 524678003752616.3780682717, 524212451387668.4186415838),
        (3500000000000.0, 20000000.0, 0.46): (35058.90723666841769594433, 9.6956466236e+35, 9.597418536528029851685467e-15329, 2.346290944286472048010346e+31, 6.960500198217961422680609e+31, 37687080202983614.54510263, 37687080081917452.09285463),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")
    for _t_args in ((2e6, 1e9, 0.5), (1e-3, 1e10, 0.5)):
        _t_out = electron_positron_eos(*_t_args)
        for _t_k, _t_want in zip(_T_KEYS, [float(x) for x in _T_EOS[_t_args]]):
            _t_tol = 1e-8 if _t_k == "cv" else 1e-10
            assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 1 ---
# thermodynamic consistency of the complete equation of state of a hot, partly pair-populated gas
# (rho = 1e6 g cm^-3, T = 3e9 K, Ye = 0.5: psi about 0.65, n_plus / n_minus about 0.29), all from
# electron_positron_eos. Each bound below is the first-order error budget when every returned quantity has
# its stated accuracy (psi, n_minus, n_plus, P, u, s relative 1e-10, cv relative 1e-8), computed from the
# returned values:
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
#      whose ratio is 16)
import numpy as np


def _crown_case():
    import numpy as np

    _T_MEC2 = 8.1871057769e-7
    _T_KB = 1.380649e-16
    _T_NA = 6.02214076e23

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


_crown_case()
