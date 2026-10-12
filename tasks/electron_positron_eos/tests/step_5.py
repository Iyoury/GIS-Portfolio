# --- test case 0 ---
# Independent targets, none from the solution code: values computed once with mpmath at 40 digits by
# tanh-sinh quadrature of the defining integrals in the momentum (breakpoints every k T over the Fermi
# edge), the chemical potential by a root search on those integrals and the specific heat by a centred
# difference of the energy at constant net electron density (T +- 1e-7 T) (cases 0, 1, 4 and 5); closed
# forms of the classical (Maxwell-Juettner) gas with modified Bessel functions, evaluated here in 40-digit
# mpmath (case 2); and the Sommerfeld expansion of the degenerate gas in 60-digit mpmath (case 3).
# The complete equation of state of a carbon-oxygen white-dwarf interior (rho = 2e6 g cm^-3, T = 1e9 K,
# Ye = 0.5) and of a pair plasma (rho = 1e-3 g cm^-3, T = 1e10 K, Ye = 0.5), against 40-digit targets;
# the result is a dict with exactly the seven keys
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")
    _T_EOS = {
        (2000000.0, 1000000000.0, 0.5): (7.798772316595553735539734, 6.02214298916745876143701e+29, 222916745876143700987606.6, 114478573268634348539048.3, 208291874619394925772455.9, 167383473004818.0744619993, 130656532160103.4906555759),
        (0.001, 10000000000.0, 0.5): (1.143531551258349070810209e-11, 1.42657113518383029437691e+31, 1.42657113515371959057691e+31, 4.109758750359864921587657e+25, 1.288730256177001349765261e+26, 16997061312154530.37094254, 52269535406676546.90725823),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V

    def _t_check5(args, target):
        out = electron_positron_eos(*args)
        assert isinstance(out, dict) and set(out) == set(_T_KEYS), out
        for k, want in zip(_T_KEYS, target):
            tol = 1e-8 if k == "cv" else 1e-10
            if k == "n_plus" and want < 1e-250:
                assert type(out[k]) is float and abs(out[k] - want) <= 1e-250, (args, k, out[k], want)
            else:
                assert type(out[k]) is float and _t_rel(out[k], want) < tol, (args, k, out[k], want)

    for _t_args, _t_v in _T_EOS.items():
        _t_check5(_t_args, [float(x) for x in _t_v])


_crown_case()

# --- test case 1 ---
# a helium-burning state (rho = 1e8 g cm^-3, T = 3e8 K, Ye = 0.43, psi about 73, n_plus about 2e-12
# cm^-3) and a degenerate stellar core (rho = 3.5e12 g cm^-3, T = 2e7 K, Ye = 0.46, psi about 3.5e4),
# against 40-digit targets
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")
    _T_EOS = {
        (100000000.0, 300000000.0, 0.43): (72.55703066526062842798715, 2.5895205268e+31, 2.2091457640632335647917e-12, 1.751430670766828173376139e+25, 3.92645989995075988402764e+25, 524678003752616.3780682717, 524212451387668.4186415838),
        (3500000000000.0, 20000000.0, 0.46): (35058.90723666841769594433, 9.6956466236e+35, 9.597418536528029851685467e-15329, 2.346290944286472048010346e+31, 6.960500198217961422680609e+31, 37687080202983614.54510263, 37687080081917452.09285463),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V

    def _t_check5(args, target):
        out = electron_positron_eos(*args)
        for k, want in zip(_T_KEYS, target):
            tol = 1e-8 if k == "cv" else 1e-10
            if k == "n_plus" and want < 1e-250:
                assert type(out[k]) is float and abs(out[k] - want) <= 1e-250, (args, k, out[k], want)
            else:
                assert type(out[k]) is float and _t_rel(out[k], want) < tol, (args, k, out[k], want)

    for _t_args, _t_v in _T_EOS.items():
        _t_check5(_t_args, [float(x) for x in _t_v])


_crown_case()

# --- test case 2 ---
# classical gases with pairs, all seven quantities in closed form: both species follow Maxwell-Juettner
# statistics (electron occupation exponent above 27, so Fermi corrections below 1e-12), with a = 1 / theta,
# N = K2(a) / a, E = K1(a) / a + 3 K2(a) / a^2, F = K0(a) / a + 5 K1(a) / a^2 + 12 K2(a) / a^3 (the moments
# int p^2 eps^k e^{-a eps} dp) and B = N / (pi^2 lambda^3): psi = asinh(rho Ye N_A / (2 B)),
# n_minus, n_plus = B e^{+-psi}, P = (n_minus + n_plus) k T, <eps> = E / N,
# u = m_e c^2 [(n_minus + n_plus) <eps> - rho Ye N_A], s = k [(n_minus + n_plus) (<eps> / theta + 1)
# - psi rho Ye N_A] and c_V = (2 k / (pi^2 lambda^3 theta^2)) [F cosh psi - (E^2 / N) tanh psi sinh psi];
# rho = 2e-10 g cm^-3, T = 1.4e8 K, Ye = 0.5 (psi about 9.8; pair creation adds 1.5e-5 to c_V) and
# rho = 1e-8 g cm^-3, T = 1.2e8 K, Ye = 1 (psi about 21.7). The closed forms agree with the exact
# Bessel-function series of the Fermi-Dirac integrals within 4e-13.
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

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")

    for _t_rho, _t_T, _t_Ye in ((2e-10, 1.4e8, 0.5), (1e-8, 1.2e8, 1.0)):
        with _t_mp.workdps(40):
            _t_n = _t_mp.mpf(_t_rho) * _t_mp.mpf(_t_Ye) * _t_mp.mpf(_T_NA)
            _t_th = _t_mp.mpf(_T_KB) * _t_T / _t_mp.mpf(_T_MEC2)
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
            _t_want = [
                _t_psi, _t_nm, _t_np,
                (_t_nm + _t_np) * _t_mp.mpf(_T_KB) * _t_T,
                _t_mp.mpf(_T_MEC2) * ((_t_nm + _t_np) * _t_g - _t_n),
                _t_mp.mpf(_T_KB) * ((_t_nm + _t_np) * (_t_g / _t_th + 1) - _t_psi * _t_n),
                2 * _t_mp.mpf(_T_KB) * _t_pref / _t_th ** 2
                * (_t_F * _t_mp.cosh(_t_psi) - _t_E ** 2 / _t_N * _t_mp.tanh(_t_psi) * _t_mp.sinh(_t_psi)),
            ]
            _t_want = [float(_t_x) for _t_x in _t_want]
        _t_out = electron_positron_eos(_t_rho, _t_T, _t_Ye)
        for _t_k, _t_w in zip(_T_KEYS, _t_want):
            _t_tol = 1e-8 if _t_k == "cv" else 1e-10
            assert type(_t_out[_t_k]) is float and _t_rel(_t_out[_t_k], _t_w) < _t_tol, (_t_rho, _t_T, _t_Ye, _t_k, _t_out[_t_k], _t_w)


_crown_case()

# --- test case 3 ---
# strongly degenerate gases from the Sommerfeld expansion (positrons and the lower end of the momentum
# integral contribute below e^{-300}): with mu = psi theta (rest energy included, units m_e c^2) each
# density is (1 / (pi^2 lambda^3)) [int_1^mu H deps + sum_k 2 (1 - 2^{1-2k}) zeta(2k) theta^{2k}
# H^{(2k-1)}(mu)], H = eps p for n, p^3 / 3 for P / (m_e c^2), (eps - 1) eps p for u / (m_e c^2), the
# zero-temperature integrals in closed form, terms to k = 8, in 60-digit mpmath; psi from n_minus =
# rho Ye N_A, s = (dP/dT) at constant mu, c_V = u_theta - u_mu n_theta / n_mu (times k / (pi^2 lambda^3)),
# n_plus the Maxwell-Juettner value theta K2(1/theta) e^{-psi} / (pi^2 lambda^3) (below 1e-500 cm^-3).
# An ultra-relativistic gas (rho = 2e13 g cm^-3, T = 1e9 K, Ye = 0.5, E_F about 1300 k T) and a mildly
# relativistic one (rho = 4e6 g cm^-3, T = 1e7 K, Ye = 0.5, E_F about 370 k T)
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")
    _T_EOS = {
        (2e13, 1e9, 0.5): (1288.86654822270575456093, 6.02214076e+36, 9.715760937524490143851642e-534, 2.678989045519733471283266e+32, 7.988003455906072934178029e+32, 6366984542011694531.149651, 6366969410111229921.828398),
        (4e6, 1e7, 0.5): (959.0270475026477711036308, 1.204428152e+30, 1.42223500707363258703769e-648, 219531284305144376444627.0, 389177260304632978889267.2, 2770546806919.348960127573, 2770509852894.45173713809),
    }      # (rho, T, Ye): psi, n_minus, n_plus, P, u, s, c_V

    def _t_check5(args, target):
        out = electron_positron_eos(*args)
        for k, want in zip(_T_KEYS, target):
            tol = 1e-8 if k == "cv" else 1e-10
            if k == "n_plus" and want < 1e-250:
                assert type(out[k]) is float and abs(out[k] - want) <= 1e-250, (args, k, out[k], want)
            else:
                assert type(out[k]) is float and _t_rel(out[k], want) < tol, (args, k, out[k], want)

    for _t_args, _t_v in _T_EOS.items():
        _t_check5(_t_args, [float(x) for x in _t_v])


_crown_case()

# --- test case 4 ---
# the endpoint Ye = 1 (pure hydrogen) is in the domain: rho = 1e6, T = 1e9 K, Ye = 1 has rho Ye = 1e6
# g cm^-3 like the white-dwarf state (2e6, 1e9, 0.5), so the equation of state, which depends on rho and
# Ye only through rho Ye, must reproduce the same 40-digit targets
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _T_KEYS = ("psi", "n_minus", "n_plus", "P", "u", "s", "cv")
    _T_WANT = (7.798772316595553735539734, 6.02214298916745876143701e+29, 222916745876143700987606.6,
               114478573268634348539048.3, 208291874619394925772455.9, 167383473004818.0744619993,
               130656532160103.4906555759)      # psi, n_minus, n_plus, P, u, s, c_V at rho Ye = 1e6, T = 1e9 K

    _t_out = electron_positron_eos(1e6, 1e9, 1.0)
    for _t_k, _t_want in zip(_T_KEYS, _T_WANT):
        _t_tol = 1e-8 if _t_k == "cv" else 1e-10
        assert type(_t_out[_t_k]) is float and _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_k, _t_out[_t_k], _t_want)


_crown_case()

# --- test case 5 ---
# the corners of the domain, with T and rho Ye at their inclusive endpoints: a cold, strongly degenerate
# gas (rho = 2e13, T = 1e7 K, Ye = 0.5, so rho Ye = 1e13 g cm^-3 exactly) and a pair plasma
# (rho = 2e-10, T = 1e11 K, Ye = 0.5, so rho Ye = 1e-10 g cm^-3 exactly). psi and c_V are the 40-digit
# (rho_Ye, T) targets; P, u, s and the pair-plasma densities come from 40-digit mpmath quadrature
# (tanh-sinh and Gauss-Legendre, agreeing to 25 digits) at that psi; in the degenerate gas
# n_minus = rho Ye N_A + n_plus with n_plus about 6e-56207 cm^-3 (within 1e-250 of 0)
import numpy as np


def _crown_case():
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
        for _t_k, _t_want in zip(_T_KEYS, _t_target):
            _t_tol = 1e-8 if _t_k == "cv" else 1e-10
            if _t_k == "n_plus" and _t_want < 1e-250:
                assert type(_t_out[_t_k]) is float and abs(_t_out[_t_k] - _t_want) <= 1e-250, (_t_args, _t_k, _t_out[_t_k], _t_want)
            else:
                assert type(_t_out[_t_k]) is float and _t_rel(_t_out[_t_k], _t_want) < _t_tol, (_t_args, _t_k, _t_out[_t_k], _t_want)


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
