# --- test case 0 ---
# Independent targets, none from the solution code: closed forms of the classical (Maxwell-Juettner) gas
# with modified Bessel functions, evaluated here in 40-digit mpmath (case 0); the Sommerfeld expansion of
# the degenerate gas and the Bessel-function series of the Fermi-Dirac integrals, evaluated once in 50- to
# 60-digit mpmath (cases 1 and 2); and values computed once with mpmath at 40 digits by tanh-sinh
# quadrature of the defining integrals in the momentum (breakpoints every k T over the Fermi edge), with
# the chemical potential from a root search on those integrals (cases 3 to 5).
# classical gas without pairs: n_net = n_- = theta K2(1/theta) e^psi / (pi^2 lambda^3)
# (positrons below 2e-19 of the electrons, shifting psi by less than 1e-20 relative; Fermi corrections
# below 1e-11), so psi follows in closed form
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

    for _t_rY, _t_T in ((1e-8, 1.2e8), (1e-10, 1e8)):
        with _t_mp.workdps(40):
            _t_th = _t_mp.mpf(_T_KB) * _t_T / _t_mp.mpf(_T_MEC2)
            _t_base = _t_th * _t_mp.besselk(2, 1 / _t_th) / (_t_mp.pi ** 2 * _t_mp.mpf(_T_LC) ** 3)
            _t_want = float(_t_mp.log(_t_mp.mpf(_t_rY) * _T_NA / _t_base))
        _t_got = degeneracy_parameter(_t_rY, _t_T)
        assert type(_t_got) is float and _t_rel(_t_got, _t_want) < 1e-10, (_t_rY, _t_T, _t_got, _t_want)


_crown_case()

# --- test case 1 ---
# strongly degenerate gases from the Sommerfeld expansion (positrons and the lower end of the momentum
# integral contribute below e^{-300}): with mu = psi theta the electron chemical energy (rest energy
# included, units m_e c^2), n = int_1^mu eps p deps + sum_k 2 (1 - 2^{1-2k}) zeta(2k) theta^{2k}
# d^{2k-1}(eps p)/deps^{2k-1} at mu, terms to k = 8, solved for n = rho_Ye N_A in 60-digit mpmath (the
# terms to k = 6 and to k = 8 agree within 1e-28): an ultra-relativistic gas at rho_Ye = 1e13 g cm^-3,
# T = 1e9 K (psi about 1289), a mildly relativistic one at 2e6 g cm^-3, 1e7 K (psi about 959) and
# 1e11 g cm^-3 at 5e7 K (psi about 5555)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_RT = {
        (1e13, 1e9): (1288.86654822270575456093, 6366969410111229921.828398),
        (2e6, 1e7): (959.0270475026477711036308, 2770509852894.45173713809),
        (1e11, 5e7): (5554.775446195058934056089, 14779689541708190.63932463),
    }      # (rho_Ye, T): psi, c_V

    for (_t_rY, _t_T), _t_v in _T_RT.items():
        _t_got = degeneracy_parameter(_t_rY, _t_T)
        assert type(_t_got) is float and _t_rel(_t_got, float(_t_v[0])) < 1e-10, (_t_rY, _t_T, _t_got, _t_v[0])


_crown_case()

# --- test case 2 ---
# pair plasmas with Fermi-Dirac corrections, from the series of the Fermi-Dirac integrals in modified
# Bessel functions, exact for psi < 1 / theta: n_net = (2 / (pi^2 lambda^3)) sum_j (-1)^{j+1} sinh(j psi)
# K2(j / theta) theta / j (from 1 / (e^x + 1) = sum_j (-1)^{j+1} e^{-j x}, x > 0), solved for
# n_net = rho_Ye N_A in 50-digit mpmath with terms down to 1e-45: rho_Ye = 1e-10 g cm^-3 at T = 1e9 K
# (psi = 5.6e-14, n_net / n_minus = 1.1e-13) and 1e4 g cm^-3 at 2e9 K (psi = 0.081)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_RT = {
        (1e-10, 1e9): (5.550389008533471215010165e-14, 9180238928315.663093607304),
        (1e4, 2e9): (0.08088769411333025563206732, 275377233856826.6345115606),
    }      # (rho_Ye, T): psi, c_V

    for (_t_rY, _t_T), _t_v in _T_RT.items():
        _t_got = degeneracy_parameter(_t_rY, _t_T)
        assert type(_t_got) is float and _t_rel(_t_got, float(_t_v[0])) < 1e-10, (_t_rY, _t_T, _t_got, _t_v[0])


_crown_case()

# --- test case 3 ---
# degenerate gases against the 40-digit targets, from psi about 1.3e5 (rho_Ye = 1e13 g cm^-3, T = 1e7 K);
# includes two degenerate, non-relativistic gases near T = 1e7 K (rho_Ye = 1e5 g cm^-3 at 1e7 K and 4e5
# g cm^-3 at 1.2e7 K, E_F about 60 and 120 k T, psi about 650 and 620), where the slope of ln n_net against
# ln psi falls from several hundred on the classical side to 17 and 8 at the root, so a root search that
# stalls on one side of its bracket misses the 1e-10 accuracy
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_RT = {
        (10000000000000.0, 10000000.0): (128886.9100517552405035025, 63669921057170992.70326489),
        (100000000000.0, 10000000.0): (27773.88007446850020519232, 2955938453488459.352956206),
        (1000000000.0, 100000000.0): (601.1599854541224417860008, 1378411651089536.226956685),
        (100000.0, 10000000.0): (654.7662355324754837308754, 696370732683.6245810121454),
        (400000.0, 12000000.0): (615.7131023969530689569447, 1497668503860.948085152025),
    }      # (rho_Ye, T): psi, c_V

    for (_t_rY, _t_T), _t_v in _T_RT.items():
        _t_got = degeneracy_parameter(_t_rY, _t_T)
        assert type(_t_got) is float and _t_rel(_t_got, float(_t_v[0])) < 1e-10, (_t_rY, _t_T, _t_got, _t_v[0])


_crown_case()

# --- test case 4 ---
# pair plasmas against the 40-digit targets, down to psi about 2e-21 at the hottest, most dilute corner
# (rho_Ye = 1e-10 g cm^-3, T = 1e11 K), where the positrons and electrons outnumber their difference by
# 20 decades, and a dense gas at 1e11 K (rho_Ye = 1e13 g cm^-3)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_RT = {
        (100000.0, 5000000000.0): (0.02130730935998780178981344, 6255474883540448.363773655),
        (1.0, 10000000000.0): (2.287063102516697966668524e-8, 52269535406676540.4997098),
        (1e-10, 100000000000.0): (2.170425950671224465122001e-21, 52953392105589435431.00965),
        (1e-10, 3000000000.0): (1.372907152528198255239101e-16, 1201091785062658.571302717),
        (10000000000000.0, 100000000000.0): (12.63346997607270908696668, 615272897283901411681.9991),
    }      # (rho_Ye, T): psi, c_V

    for (_t_rY, _t_T), _t_v in _T_RT.items():
        _t_got = degeneracy_parameter(_t_rY, _t_T)
        assert type(_t_got) is float and _t_rel(_t_got, float(_t_v[0])) < 1e-10, (_t_rY, _t_T, _t_got, _t_v[0])


_crown_case()

# --- test case 5 ---
# a partly degenerate gas with pairs (rho_Ye = 1e6 g cm^-3, T = 1e9 K, psi about 7.8) and the coolest,
# most dilute corner (rho_Ye = 1e-10 g cm^-3, T = 1e7 K, psi about 564), against the 40-digit targets
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_RT = {
        (1000000.0, 1000000000.0): (7.798772316595553735539734, 130656532160103.4906555759),
        (1e-10, 10000000.0): (564.4249121875995906962405, 0.01252414082046084989431409),
    }      # (rho_Ye, T): psi, c_V

    for (_t_rY, _t_T), _t_v in _T_RT.items():
        _t_got = degeneracy_parameter(_t_rY, _t_T)
        assert type(_t_got) is float and _t_rel(_t_got, float(_t_v[0])) < 1e-10, (_t_rY, _t_T, _t_got, _t_v[0])


_crown_case()

# --- test case 6 ---
# rho_Ye or T outside its range
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((5e-11, 1e9), (2e13, 1e9), (1.0, 9e6), (1.0, 2e11)):
        try:
            degeneracy_parameter(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("degeneracy_parameter%r must raise ValueError" % (_t_bad,))


_crown_case()
