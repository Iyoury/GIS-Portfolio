# Independent targets: closed forms of the classical (Maxwell-Juettner) gas with modified Bessel
# functions, and values computed once with mpmath at 40 digits by tanh-sinh quadrature of the defining
# integrals in the momentum (breakpoints every k T over the Fermi edge), the chemical potential by a
# root search on those integrals and the specific heat by a centred difference of the energy at constant
# net electron density (T +- 1e-7 T), all in mpmath; positron densities of a classical positron gas
# (eps/kT + psi >= 40) from the Maxwell-Juettner formula.

# --- test case 0 ---
# a white-dwarf-like, a hot pair plasma, a helium-burning and a degenerate core state,
# against 40-digit targets
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

def _t_check5(args, target):
    out = electron_positron_eos(*args)
    assert isinstance(out, dict) and set(out) == set(_T_KEYS), out
    assert all(type(out[k]) is float for k in _T_KEYS), out
    for k, want in zip(_T_KEYS, target):
        tol = 1e-8 if k == "cv" else 1e-10
        if k == "n_plus" and want < 1e-250:
            assert abs(out[k] - want) <= 1e-250, (args, k, out[k], want)
        else:
            assert _t_rel(out[k], want) < tol, (args, k, out[k], want)

for _t_args, _t_v in _T_EOS.items():
    _t_check5(_t_args, [float(x) for x in _t_v])

# --- test case 1 ---
# rho, T or Ye not finite or out of range, or rho Ye outside [1e-10, 1e13]
import numpy as np
for _t_bad in ((float("nan"), 1e9, 0.5), (1e6, float("inf"), 0.5), (1e6, 1e9, 0.0), (1e6, 1e9, 1.2),
               (1e6, 1e9, float("nan")), (1e-11, 1e9, 0.5), (3e13, 1e9, 0.5), (1e6, 5e6, 0.5),
               (1e6, 2e11, 0.5)):
    try:
        electron_positron_eos(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("electron_positron_eos%r must raise ValueError" % (_t_bad,))

# --- test case 2 ---
# the endpoint Ye = 1 (pure hydrogen) is in the domain: rho = 1e6, T = 1e9 K, Ye = 1 has rho Ye = 1e6
# g cm^-3 like the white-dwarf state (2e6, 1e9, 0.5), so the equation of state, which depends on rho and
# Ye only through rho Ye, must reproduce the same 40-digit targets
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

_T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")
_T_WANT = (7.798772316595553735539734, 6.02214298916745876143701e+29, 222916745876143700987606.6,
           114478573268634348539048.3, 208291874619394925772455.9, 167383473004818.0744619993,
           130656532160103.4906555759)      # psi, n_minus, n_plus, P, u, s, c_V at rho Ye = 1e6, T = 1e9 K

_t_out = electron_positron_eos(1e6, 1e9, 1.0)
assert isinstance(_t_out, dict) and set(_t_out) == set(_T_KEYS), _t_out
assert all(type(_t_out[_t_k]) is float for _t_k in _T_KEYS), _t_out
for _t_k, _t_want in zip(_T_KEYS, _T_WANT):
    _t_tol = 1e-8 if _t_k == "cv" else 1e-10
    assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_k, _t_out[_t_k], _t_want)

# --- test case 3 ---
# the corners of the domain, with T and rho Ye at their inclusive endpoints: a cold, strongly degenerate
# gas (rho = 2e13, T = 1e7 K, Ye = 0.5, so rho Ye = 1e13 g cm^-3 exactly) and a pair plasma
# (rho = 2e-10, T = 1e11 K, Ye = 0.5, so rho Ye = 1e-10 g cm^-3 exactly). psi and c_V are the 40-digit
# (rho_Ye, T) targets; P, u, s and the pair-plasma densities come from 40-digit mpmath quadrature
# (tanh-sinh and Gauss-Legendre, agreeing to 25 digits) at that psi; in the degenerate gas
# n_minus = rho Ye N_A + n_plus with n_plus about 6e-56207 cm^-3 (within 1e-250 of 0)
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

_T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")
_T_CORNERS = {
    (2e13, 1e7, 0.5): (128886.9100517552405035025, 6.02214076e36, 6.015906150270230151335683e-56207, 2.67897843472169752993636e+32, 7.987971624185773846332954e+32, 63669921072302993.04776527, 63669921057170992.70326489),
    (2e-10, 1e11, 0.5): (2.170425950671224465122001e-21, 1.520486632169166839841366e+34, 1.520486632169166839835344e+34, 4.40998243289230441302235e+29, 1.323665810506425580785849e+30, 17646640537956560220.88133, 52953392105589435431.00965),
}      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V

for _t_args, _t_target in _T_CORNERS.items():
    _t_out = electron_positron_eos(*_t_args)
    assert isinstance(_t_out, dict) and set(_t_out) == set(_T_KEYS), _t_out
    assert all(type(_t_out[_t_k]) is float for _t_k in _T_KEYS), _t_out
    for _t_k, _t_want in zip(_T_KEYS, _t_target):
        _t_tol = 1e-8 if _t_k == "cv" else 1e-10
        if _t_k == "n_plus" and _t_want < 1e-250:
            assert abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
        else:
            assert _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)
