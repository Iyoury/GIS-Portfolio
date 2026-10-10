# Independent targets: closed forms of the classical (Maxwell-Juettner) gas with modified Bessel
# functions, and values computed once with mpmath at 40 digits by tanh-sinh quadrature of the defining
# integrals in the momentum (breakpoints every k T over the Fermi edge), the chemical potential by a
# root search on those integrals and the specific heat by a centred difference of the energy at constant
# net electron density (T +- 1e-7 T), all in mpmath; positron densities of a classical positron gas
# (eps/kT + psi >= 40) from the Maxwell-Juettner formula.

# --- test case 0 ---
# from strongly degenerate gases (c_V about 8e-10 of u / T at rho_Ye = 1e13, T = 1e7 K) to
# pair plasmas, against centred differences of the 40-digit energy at constant net density; includes
# two degenerate, non-relativistic gases near T = 1e7 K (rho_Ye = 1e5 g cm^-3 at 1e7 K and 4e5 g cm^-3
# at 1.2e7 K), where the psi solving n_net = rho_Ye N_A is hard to converge (d ln c_V / d ln psi is
# about 7 and 4 there, so an error of psi enters c_V amplified)
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

_T_RT = {
    (10000000000000.0, 10000000.0): (128886.9100517552405035025, 63669921057170992.70326489),
    (100000000000.0, 10000000.0): (27773.88007446850020519232, 2955938453488459.352956206),
    (1000000000.0, 100000000.0): (601.1599854541224417860008, 1378411651089536.226956685),
    (1000000.0, 1000000000.0): (7.798772316595553735539734, 130656532160103.4906555759),
    (100000.0, 5000000000.0): (0.02130730935998780178981344, 6255474883540448.363773655),
    (1.0, 10000000000.0): (2.287063102516697966668524e-8, 52269535406676540.4997098),
    (1e-10, 100000000000.0): (2.170425950671224465122001e-21, 52953392105589435431.00965),
    (1e-10, 10000000.0): (564.4249121875995906962405, 0.01252414082046084989431409),
    (1e-10, 3000000000.0): (1.372907152528198255239101e-16, 1201091785062658.571302717),
    (10000000000000.0, 100000000000.0): (12.63346997607270908696668, 615272897283901411681.9991),
    (100000.0, 10000000.0): (654.7662355324754837308754, 696370732683.6245810121454),
    (400000.0, 12000000.0): (615.7131023969530689569447, 1497668503860.948085152025),
}      # (rho_Ye, T): psi, c_V

for (_t_rY, _t_T), _t_v in _T_RT.items():
    _t_got = specific_heat(_t_rY, _t_T)
    assert type(_t_got) is float and _t_rel(_t_got, float(_t_v[1])) < 1e-8, (_t_rY, _t_T, _t_got, _t_v[1])

# --- test case 1 ---
# rho_Ye or T not finite or outside its range
import numpy as np
for _t_bad in ((float("nan"), 1e9), (1.0, float("nan")), (5e-11, 1e9), (2e13, 1e9), (1.0, 9e6), (1.0, 2e11)):
    try:
        specific_heat(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("specific_heat%r must raise ValueError" % (_t_bad,))
