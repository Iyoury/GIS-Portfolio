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


electron_positron_eos = _t_budget(electron_positron_eos, 30.0, "electron_positron_eos")

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


# --- test case 0: a white-dwarf-like, a hot pair plasma, a helium-burning and a degenerate core state,
# against 40-digit targets ---
for _t_args, _t_v in _T_EOS.items():
    _t_check5(_t_args, [float(x) for x in _t_v])

# --- test case 1: rho, T or Ye not finite or out of range, or rho Ye outside [1e-10, 1e13] ---
for _t_bad in ((float("nan"), 1e9, 0.5), (1e6, float("inf"), 0.5), (1e6, 1e9, 0.0), (1e6, 1e9, 1.2),
               (1e6, 1e9, float("nan")), (1e-11, 1e9, 0.5), (3e13, 1e9, 0.5), (1e6, 5e6, 0.5)):
    try:
        electron_positron_eos(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("electron_positron_eos%r must raise ValueError" % (_t_bad,))
