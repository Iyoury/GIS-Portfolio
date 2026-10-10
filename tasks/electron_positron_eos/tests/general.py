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

# --- test case 1 ---
# consistency between the steps. At the psi returned by degeneracy_parameter for rho Ye = 5e5 g cm^-3,
# T = 1e9 K, pair_densities gives n_net = rho Ye N_A: d ln n_net / d ln psi is about 4.84 there, so psi
# within its relative 1e-10 and n_net within its relative 1e-10 put n_net within 5.9e-10 of rho Ye N_A.
# In a classical gas without pairs (rho Ye = 1e-6, T = 1e8 K) the Euler relation
# u_tot + P - T s = mu n_net holds (u_tot = u + n_net m_e c^2, mu = psi k T including the rest energy);
# with every quantity within its relative 1e-10 the two sides agree to about 3.4e-10 (the largest terms,
# n_net m_e c^2 and T s, are 1.6 and 0.7 times the result), well within the 1e-8 checked here
import numpy as np

_T_MEC2 = 8.1871057769e-7
_T_KB = 1.380649e-16
_T_NA = 6.02214076e23

def _t_rel(a, b):
    return abs(a - b) / abs(b)

_t_psi = degeneracy_parameter(1e6 * 0.5, 1e9)
_t_nm, _t_npl, _t_net = pair_densities(1e9, _t_psi)
assert _t_rel(_t_net, 1e6 * 0.5 * _T_NA) < 6e-10, (_t_net, _t_psi)
_t_psi = degeneracy_parameter(1e-6, 1e8)
_t_nm, _t_npl, _t_net = pair_densities(1e8, _t_psi)
_t_P, _t_u, _t_s = pair_thermodynamics(1e8, _t_psi)
_t_lhs = _t_u + _t_net * _T_MEC2 + _t_P - 1e8 * _t_s
_t_rhs = _t_psi * _T_KB * 1e8 * _t_net
assert _t_rel(_t_lhs, _t_rhs) < 1e-8, (_t_lhs, _t_rhs)
