# Independent targets: closed forms of the classical (Maxwell-Juettner) gas with modified Bessel
# functions, and values computed once with mpmath at 40 digits by tanh-sinh quadrature of the defining
# integrals in the momentum (breakpoints every k T over the Fermi edge), the chemical potential by a
# root search on those integrals and the specific heat by a centred difference of the energy at constant
# net electron density (T +- 1e-7 T), all in mpmath; positron densities of a classical positron gas
# (eps/kT + psi >= 40) from the Maxwell-Juettner formula.

# --- test case 0 ---
# classical gas: P = (n_- + n_+) k T, energy and entropy of the Maxwell-Juettner gas
import numpy as np
import mpmath as _t_mp

_T_MEC2 = 8.1871057769e-7
_T_KB = 1.380649e-16
_T_LC = 3.8615926796e-11

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_classical(T, psi):
    # Maxwell-Juettner gas (Boltzmann occupation): n = theta K2(1/theta) e^{+-psi} / (pi^2 lambda^3),
    # mean total energy K1/K2 + 3 theta, P = (n_- + n_+) k T, entropy per particle k (<eps>/theta -+ psi + 1)
    with _t_mp.workdps(40):
        th = _t_mp.mpf(_T_KB) * T / _t_mp.mpf(_T_MEC2)
        pref = 1 / (_t_mp.pi ** 2 * _t_mp.mpf(_T_LC) ** 3)
        base = pref * th * _t_mp.besselk(2, 1 / th)
        nm, npl = base * _t_mp.exp(psi), base * _t_mp.exp(-psi)
        eps = _t_mp.besselk(1, 1 / th) / _t_mp.besselk(2, 1 / th) + 3 * th
        P = (nm + npl) * _t_mp.mpf(_T_KB) * T
        u = _t_mp.mpf(_T_MEC2) * (nm * (eps - 1) + npl * (eps + 1))
        s = _t_mp.mpf(_T_KB) * (nm * (eps / th - psi + 1) + npl * (eps / th + psi + 1))
        return [float(x) for x in (nm, npl, nm - npl, P, u, s)]

def _t_check2(T, psi, target):
    out = pair_thermodynamics(T, psi)
    assert isinstance(out, tuple) and len(out) == 3 and all(type(x) is float for x in out), out
    for got, want, name in zip(out, target, ("P", "u", "s")):
        assert _t_rel(got, want) < 1e-10, (T, psi, name, got, want)
    return out

for _t_T, _t_psi in ((1e8, 20.0), (2e7, 5.0), (5e7, 30.0)):
    _t_check2(_t_T, _t_psi, _t_classical(_t_T, _t_psi)[3:])

# --- test case 1 ---
# strongly degenerate gases (the entropy is about 6e-10 of (u + P) / T at T = 1e7 K,
# psi = 1.3e5), classical gases and pair plasmas, against 40-digit targets
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

_T_TP = {
    (10000000.0, 130000.0): (6.179520231754482940999412e+36, 2.346813986462897923301478e-56690, 6.179520231754482940999412e+36, 2.772731709361358577554444e+32, 8.267948868315288590509692e+32, 64774410687625976.98471617),
    (10000000.0, 28000.0): (6.170496462484169203098532e+34, 2.556423490417399431908767e-12392, 6.170496462484169203098532e+34, 5.95949721157503759905127e+29, 1.738932937371963944986056e+30, 3004276725739380.803125785),
    (100000000.0, 600.0): (5.987006519260052776955614e+32, 2.331333906118185771750287e-259, 5.987006519260052776955614e+32, 1.22212344294881691186725e+27, 3.24742404739105466548459e+27, 1373086556674055.173670741),
    (100000000.0, 20.0): (42675956126.08147776853184, 0.0000001813025798068760043550264, 42675956126.08147758722926, 589.2051614951826653777979, 902.1269438709880449639985, 0.0002464648556891977851192496),
    (1000000000.0, 7.8): (6.026503558362130101088441e+29, 222643242700327863383998.4, 6.026501331929703097809808e+29, 114580685499195711885415.6, 208481457489471018636693.1, 167460504760700.7249915957),
    (3000000000.0, 1.37e-16): (2.265548259712415287309019e+29, 2.265548259712414686370141e+29, 60093887820506.10281849091, 190797379209953350043218.2, 773678058629460653932668.2, 321491812613138.0177251293),
    (5000000000.0, 0.0213): (1.528135218089610167956547e+30, 1.467934471355435427873768e+30, 6.020074673417474008277946e+28, 2131525594815557722593699.0, 7333518332588401682110976.0, 1902689145813356.301256761),
    (10000000000.0, 2.29e-08): (1.426571165318145243719095e+31, 1.426571105019405234208083e+31, 602987400095110113678719.5, 4.109758750359865874815893e+25, 1.288730251242745214163508e+26, 16997061312154532.3771599),
    (100000000000.0, 2.2e-21): (1.520486632169166839841407e+34, 1.520486632169166839835303e+34, 61041979653361.19312931601, 4.40998243289230441302235e+29, 1.323665810506425580785848e+30, 17646640537956560220.88133),
    (100000000000.0, 50.0): (3.529868337264115685100486e+38, 3252255814620.44422883235, 3.529868337264115685100486e+38, 6.115873067840409031700818e+34, 1.831877107397111367022785e+35, 9599759739041385840372.273),
    (100000000000.0, 1000000.0): (2.812796130304916134373229e+51, 5.559113513738466642968216e-434261, 2.812796130304916134373229e+51, 9.708710411369150311930866e+51, 2.912612893124171100601138e+52, 3.832845240140071471603274e+30),
    (20000000.0, 1000000.0): (2.250236607519217944970892e+40, 2.455871423753166156237421e-434397, 2.250236607519217944970892e+40, 1.553393256146546323000493e+37, 4.658338295271499426436939e+37, 30662760573351414102.7831),
}      # (T, psi): n_minus, n_plus, n_net, P, u, s

def _t_check2(T, psi, target):
    out = pair_thermodynamics(T, psi)
    assert isinstance(out, tuple) and len(out) == 3 and all(type(x) is float for x in out), out
    for got, want, name in zip(out, target, ("P", "u", "s")):
        assert _t_rel(got, want) < 1e-10, (T, psi, name, got, want)
    return out

for (_t_T, _t_psi), _t_v in _T_TP.items():
    _t_check2(_t_T, _t_psi, [float(x) for x in _t_v[3:]])

# --- test case 2 ---
# T or psi not finite or outside its range
import numpy as np
for _t_bad in ((float("nan"), 1.0), (1e9, float("nan")), (9e6, 1.0), (1.1e11, 1.0), (1e9, 0.0), (1e9, 2e6)):
    try:
        pair_thermodynamics(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("pair_thermodynamics%r must raise ValueError" % (_t_bad,))
