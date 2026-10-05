import math
import numpy as np
import mpmath as _t_mp
import signal as _t_signal
import time as _t_btime

# Independent targets: closed forms of the classical (Maxwell-Juettner) gas with modified Bessel
# functions, and values computed once with mpmath at 40 digits by tanh-sinh quadrature of the defining
# integrals in the momentum (breakpoints every k T over the Fermi edge), the chemical potential by a
# root search on those integrals and the specific heat by a centred difference of the energy at constant
# net electron density (T +- 1e-7 T), all in mpmath; positron densities of a classical positron gas
# (eps/kT + psi >= 40) from the Maxwell-Juettner formula.

_T_MEC2 = 8.1871057769e-7
_T_KB = 1.380649e-16
_T_LC = 3.8615926796e-11
_T_NA = 6.02214076e23

# Time budget of one call, as stated in the prompt: the call is interrupted once it exceeds the budget
# (by one second), so a solution that is too slow fails this check instead of holding up the tests.
_t_depth = [0]


def _t_budget(fn, seconds, name):
    def wrapped(*args, **kwargs):
        if _t_depth[0]:
            return fn(*args, **kwargs)

        def _alarm(signum, frame):
            raise TimeoutError("%s did not finish within its budget of %g s per call" % (name, seconds))
        try:
            old = _t_signal.signal(_t_signal.SIGALRM, _alarm)
            _t_signal.setitimer(_t_signal.ITIMER_REAL, seconds + 1.0)
            armed = True
        except (ValueError, AttributeError, OSError):      # no SIGALRM here: measure only
            armed = False
        _t_depth[0] += 1
        start = _t_btime.perf_counter()
        try:
            out = fn(*args, **kwargs)
        finally:
            _t_depth[0] -= 1
            if armed:
                _t_signal.setitimer(_t_signal.ITIMER_REAL, 0.0)
                _t_signal.signal(_t_signal.SIGALRM, old)
        elapsed = _t_btime.perf_counter() - start
        assert elapsed <= seconds, ("%s took %.1f s (budget %g s per call)" % (name, elapsed, seconds))
        return out
    return wrapped


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


degeneracy_parameter = _t_budget(degeneracy_parameter, 10.0, "degeneracy_parameter")

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
}      # (rho_Ye, T): psi, c_V


# --- test case 0: classical gas without pairs: n_net = n_- = theta K2(1/theta) e^psi / (pi^2 lambda^3)
# (positrons below 1e-30 of the electrons, Fermi corrections below 1e-11), so psi follows in closed form ---
for _t_rY, _t_T in ((1e-8, 1.2e8), (1e-10, 1e8)):
    with _t_mp.workdps(40):
        th = _t_mp.mpf(_T_KB) * _t_T / _t_mp.mpf(_T_MEC2)
        base = th * _t_mp.besselk(2, 1 / th) / (_t_mp.pi ** 2 * _t_mp.mpf(_T_LC) ** 3)
        _t_want = float(_t_mp.log(_t_mp.mpf(_t_rY) * _T_NA / base))
    _t_got = degeneracy_parameter(_t_rY, _t_T)
    assert type(_t_got) is float and _t_rel(_t_got, _t_want) < 1e-10, (_t_rY, _t_T, _t_got, _t_want)

# --- test case 1: strongly degenerate (psi about 1.3e5 at rho_Ye = 1e13, T = 1e7 K) to pair plasmas
# (psi about 2e-21 at rho_Ye = 1e-10, T = 1e11 K), against 40-digit targets ---
for (_t_rY, _t_T), _t_v in _T_RT.items():
    _t_got = degeneracy_parameter(_t_rY, _t_T)
    assert type(_t_got) is float and _t_rel(_t_got, float(_t_v[0])) < 1e-10, (_t_rY, _t_T, _t_got, _t_v[0])

# --- test case 2: rho_Ye or T not finite or outside its range ---
for _t_bad in ((float("nan"), 1e9), (1.0, float("inf")), (5e-11, 1e9), (2e13, 1e9), (1.0, 9e6), (1.0, 2e11)):
    try:
        degeneracy_parameter(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("degeneracy_parameter%r must raise ValueError" % (_t_bad,))
