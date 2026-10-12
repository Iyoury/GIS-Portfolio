# --- test case 0 ---
# Independent targets, none from the solution code: closed forms of the classical (Maxwell-Juettner) gas
# with modified Bessel functions, evaluated here in 40-digit mpmath (cases 0 to 2); the Sommerfeld
# expansion of the degenerate gas and the Bessel-function series of the Fermi-Dirac integrals, evaluated
# once in 50- to 60-digit mpmath (cases 3 and 4); and values computed once with mpmath at 40 digits by
# tanh-sinh quadrature of the defining integrals in the momentum (breakpoints every k T over the Fermi
# edge), the chemical potential by a root search on those integrals and the specific heat by a centred
# difference of the energy at constant net electron density (T +- 1e-7 T) (cases 5 to 7).
# Classical gas without pairs, from cool and non-relativistic (T = 1e7 K, theta = 1.7e-3) to mildly
# relativistic (1.2e8 K, theta = 0.020): at constant n_net = n_minus = rho_Ye N_A the energy per electron
# is m_e c^2 (K1(1/theta) / K2(1/theta) + 3 theta - 1), so c_V = n k d/dtheta [K1(1/theta) / K2(1/theta)
# + 3 theta] (derivative in closed form with K0' = -K1, K1' = -K0 - K1 / x, K2' = -K1 - 2 K2 / x).
# The electron occupation exponent stays above 27 and the positrons are below 1e-15 of the electrons, so
# the Fermi and pair corrections to c_V are below 1e-11 (measured against the exact Bessel series: at
# most 8e-14).
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    _T_MEC2 = 8.1871057769e-7
    _T_KB = 1.380649e-16
    _T_NA = 6.02214076e23

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    for _t_rY, _t_T in ((1e-10, 1e7), (1e-10, 1e8), (1e-8, 1.2e8)):
        with _t_mp.workdps(40):
            _t_n = _t_mp.mpf(_t_rY) * _t_mp.mpf(_T_NA)
            _t_th = _t_mp.mpf(_T_KB) * _t_T / _t_mp.mpf(_T_MEC2)
            _t_x = 1 / _t_th
            _t_K0, _t_K1, _t_K2 = _t_mp.besselk(0, _t_x), _t_mp.besselk(1, _t_x), _t_mp.besselk(2, _t_x)
            _t_dK1 = -_t_K0 - _t_K1 / _t_x
            _t_dK2 = -_t_K1 - 2 * _t_K2 / _t_x
            _t_dratio = (_t_dK1 * _t_K2 - _t_K1 * _t_dK2) / _t_K2 ** 2       # d(K1/K2)/dx
            _t_want = float(_t_n * _t_mp.mpf(_T_KB) * (3 - _t_dratio / _t_th ** 2))
        _t_got = specific_heat(_t_rY, _t_T)
        assert type(_t_got) is float, type(_t_got)
        assert _t_rel(_t_got, _t_want) < 1e-8, (_t_rY, _t_T, _t_got, _t_want)


_crown_case()

# --- test case 1 ---
# Classical gas with thermal pairs at T = 1.4e8 K (1 / theta = 42.4) and rho_Ye = 1e-10 and 1e-9 g cm^-3:
# both species follow Maxwell-Juettner statistics (electron occupation exponent above 30, so Fermi
# corrections below 1e-13), n_minus, n_plus = B e^{+-psi} with B = theta K2(1/theta) / (pi^2 lambda^3),
# so psi = asinh(rho_Ye N_A / (2 B)) is about 9.8 and 12.1 and
# u_tot = 2 m_e c^2 cosh(psi) B <eps>. At constant n_net,
#   c_V = (2 k / (pi^2 lambda^3 theta^2)) [F cosh psi - (E^2 / N) tanh psi sinh psi],
# with N = K2(a) / a, E = K1(a) / a + 3 K2(a) / a^2, F = K0(a) / a + 5 K1(a) / a^2 + 12 K2(a) / a^3
# (a = 1 / theta: the moments int p^2 eps^k e^{-a eps} dp, k = 0, 1, 2). Pair creation raises c_V above
# the electron-only value by 1.5e-5 and 1.5e-7, far more than the 1e-8 accuracy.
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

    for _t_rY in (1e-10, 1e-9):
        with _t_mp.workdps(40):
            _t_n = _t_mp.mpf(_t_rY) * _t_mp.mpf(_T_NA)
            _t_th = _t_mp.mpf(_T_KB) * 1.4e8 / _t_mp.mpf(_T_MEC2)
            _t_a = 1 / _t_th
            _t_pref = 1 / (_t_mp.pi ** 2 * _t_mp.mpf(_T_LC) ** 3)
            _t_K0, _t_K1, _t_K2 = _t_mp.besselk(0, _t_a), _t_mp.besselk(1, _t_a), _t_mp.besselk(2, _t_a)
            _t_N = _t_K2 / _t_a
            _t_E = _t_K1 / _t_a + 3 * _t_K2 / _t_a ** 2
            _t_F = _t_K0 / _t_a + 5 * _t_K1 / _t_a ** 2 + 12 * _t_K2 / _t_a ** 3
            _t_psi = _t_mp.asinh(_t_n / (2 * _t_pref * _t_N))
            _t_want = float(2 * _t_mp.mpf(_T_KB) * _t_pref / _t_th ** 2
                            * (_t_F * _t_mp.cosh(_t_psi) - _t_E ** 2 / _t_N * _t_mp.tanh(_t_psi) * _t_mp.sinh(_t_psi)))
        _t_got = specific_heat(_t_rY, 1.4e8)
        assert _t_rel(_t_got, _t_want) < 1e-8, (_t_rY, _t_got, _t_want)


_crown_case()

# --- test case 2 ---
# a classical pair plasma at T = 2e8 K (1 / theta = 29.65), rho_Ye = 1e-10, 1e-9 and 1e-8 g cm^-3
# (psi about 0.0156, 0.155 and 1.23): the positrons number 0.97, 0.73 and 0.086 of the electrons, and pair
# creation makes c_V about 3.9e4, 3.8e3 and 210 times the value of the electrons alone. Same closed form as
# case 1: with a = 1 / theta, N = K2(a) / a, E = K1(a) / a + 3 K2(a) / a^2, F = K0(a) / a + 5 K1(a) / a^2
# + 12 K2(a) / a^3 and psi = asinh(rho_Ye N_A / (2 N / (pi^2 lambda^3))),
#   c_V = (2 k / (pi^2 lambda^3 theta^2)) [F cosh psi - (E^2 / N) tanh psi sinh psi].
# The occupation exponents stay above 28.4, so the Fermi corrections are below 2e-13 of c_V (measured
# against the exact Bessel-function series of the Fermi-Dirac integrals)
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

    for _t_rY in (1e-10, 1e-9, 1e-8):
        with _t_mp.workdps(40):
            _t_n = _t_mp.mpf(_t_rY) * _t_mp.mpf(_T_NA)
            _t_th = _t_mp.mpf(_T_KB) * 2e8 / _t_mp.mpf(_T_MEC2)
            _t_a = 1 / _t_th
            _t_pref = 1 / (_t_mp.pi ** 2 * _t_mp.mpf(_T_LC) ** 3)
            _t_K0, _t_K1, _t_K2 = _t_mp.besselk(0, _t_a), _t_mp.besselk(1, _t_a), _t_mp.besselk(2, _t_a)
            _t_N = _t_K2 / _t_a
            _t_E = _t_K1 / _t_a + 3 * _t_K2 / _t_a ** 2
            _t_F = _t_K0 / _t_a + 5 * _t_K1 / _t_a ** 2 + 12 * _t_K2 / _t_a ** 3
            _t_psi = _t_mp.asinh(_t_n / (2 * _t_pref * _t_N))
            _t_want = float(2 * _t_mp.mpf(_T_KB) * _t_pref / _t_th ** 2
                            * (_t_F * _t_mp.cosh(_t_psi) - _t_E ** 2 / _t_N * _t_mp.tanh(_t_psi) * _t_mp.sinh(_t_psi)))
        _t_got = specific_heat(_t_rY, 2e8)
        assert _t_rel(_t_got, _t_want) < 1e-8, (_t_rY, _t_got, _t_want)


_crown_case()

# --- test case 3 ---
# strongly degenerate gases from the Sommerfeld expansion (positrons and the lower end of the momentum
# integral contribute below e^{-300}): with mu = psi theta the electron chemical energy (rest energy
# included, units m_e c^2), n and u are int_1^mu H deps + sum_k 2 (1 - 2^{1-2k}) zeta(2k) theta^{2k}
# H^{(2k-1)}(mu) with H = eps p and (eps - 1) eps p, the zero-temperature integrals in closed form, terms to
# k = 8; psi from n = rho_Ye N_A and c_V = u_theta - u_mu n_theta / n_mu (times k / (pi^2 lambda^3)), in
# 60-digit mpmath (the terms to k = 6 and to k = 8 agree within 1e-22). An ultra-relativistic gas at
# rho_Ye = 1e13 g cm^-3, T = 1e9 K (E_F about 1300 k T), a mildly relativistic one at 2e6 g cm^-3, 1e7 K
# (E_F about 370 k T) and 1e11 g cm^-3 at 5e7 K (E_F about 5400 k T, c_V about 4e-7 of u / T)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_SOMM = {
        (1e13, 1e9): (1288.86654822270575456093, 6366969410111229921.828398),
        (2e6, 1e7): (959.0270475026477711036308, 2770509852894.45173713809),
        (1e11, 5e7): (5554.775446195058934056089, 14779689541708190.63932463),
    }      # (rho_Ye, T): psi, c_V

    for (_t_rY, _t_T), _t_v in _T_SOMM.items():
        _t_got = specific_heat(_t_rY, _t_T)
        assert _t_rel(_t_got, float(_t_v[1])) < 1e-8, (_t_rY, _t_T, _t_got, _t_v[1])


_crown_case()

# --- test case 4 ---
# pair plasmas with Fermi-Dirac corrections, from the series of the Fermi-Dirac integrals in modified
# Bessel functions, exact for psi < 1 / theta: 1 / (e^x + 1) = sum_j (-1)^{j+1} e^{-j x} for x > 0 and
# int p^2 eps^k e^{-a eps} dp in K0, K1, K2 (a = j / theta), so n_net = (2 / (pi^2 lambda^3)) sum_j
# (-1)^{j+1} sinh(j psi) K2(j / theta) theta / j; psi from n_net = rho_Ye N_A and c_V = u_T - u_psi n_T / n_psi
# with the T and psi derivatives of the series taken term by term, in 50-digit mpmath until the terms fall
# below 1e-45. rho_Ye = 1e-10 g cm^-3 at T = 1e9 K (psi = 5.6e-14, n_net / n_minus = 1.1e-13) and 1e4 g cm^-3
# at 2e9 K (psi = 0.081); the terms j >= 2 (the Fermi-Dirac corrections) change c_V by 1.2e-3 and 1.6e-2
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_SER = {
        (1e-10, 1e9): (5.550389008533471215010165e-14, 9180238928315.663093607304),
        (1e4, 2e9): (0.08088769411333025563206732, 275377233856826.6345115606),
    }      # (rho_Ye, T): psi, c_V

    for (_t_rY, _t_T), _t_v in _T_SER.items():
        _t_got = specific_heat(_t_rY, _t_T)
        assert _t_rel(_t_got, float(_t_v[1])) < 1e-8, (_t_rY, _t_T, _t_got, _t_v[1])


_crown_case()

# --- test case 5 ---
# degenerate gases against the 40-digit targets: c_V about 8e-10 of u / T at rho_Ye = 1e13 g cm^-3,
# T = 1e7 K, where the Schur form <dE^2> - <dE dN>^2 / <dN^2> loses about 2e-7, and two degenerate,
# non-relativistic gases near T = 1e7 K (rho_Ye = 1e5 g cm^-3 at 1e7 K and 4e5 g cm^-3 at 1.2e7 K), where
# the psi solving n_net = rho_Ye N_A is hard to converge (d ln c_V / d ln psi is about 7 and 4 there, so an
# error of psi enters c_V amplified); the Sommerfeld expansion of case 3 reproduces these five c_V within
# 4e-16
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
        _t_got = specific_heat(_t_rY, _t_T)
        assert _t_rel(_t_got, float(_t_v[1])) < 1e-8, (_t_rY, _t_T, _t_got, _t_v[1])


_crown_case()

# --- test case 6 ---
# pair plasmas against the 40-digit targets, from rho_Ye = 1e5 g cm^-3 at 5e9 K to the hottest, most dilute
# corner (rho_Ye = 1e-10 g cm^-3, T = 1e11 K, psi about 2e-21), where c_V is almost entirely that of the
# thermal pairs, and a dense gas at 1e11 K (rho_Ye = 1e13 g cm^-3)
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
        _t_got = specific_heat(_t_rY, _t_T)
        assert _t_rel(_t_got, float(_t_v[1])) < 1e-8, (_t_rY, _t_T, _t_got, _t_v[1])


_crown_case()

# --- test case 7 ---
# partly degenerate and classical gases against the 40-digit targets: a white-dwarf-like state
# (rho_Ye = 1e6 g cm^-3, T = 1e9 K, psi about 7.8) and the coolest, most dilute corner
# (rho_Ye = 1e-10 g cm^-3, T = 1e7 K, psi about 564, where d ln c_V / d ln psi is about 564)
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
        _t_got = specific_heat(_t_rY, _t_T)
        assert _t_rel(_t_got, float(_t_v[1])) < 1e-8, (_t_rY, _t_T, _t_got, _t_v[1])


_crown_case()

# --- test case 8 ---
# rho_Ye or T outside its range
import numpy as np


def _crown_case():
    import numpy as np
    for _t_bad in ((5e-11, 1e9), (2e13, 1e9), (1.0, 9e6), (1.0, 2e11)):
        try:
            specific_heat(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("specific_heat%r must raise ValueError" % (_t_bad,))


_crown_case()
