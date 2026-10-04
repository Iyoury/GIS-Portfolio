import numpy as np
import mpmath as _t_mp

# Independent targets: closed forms, the Bateman sum evaluated in 600-digit arithmetic (it only
# cancels in floating point), and matrix exponentials by mpmath at 300 digits.


def _t_rel(a, b):
    return abs(a - b) / abs(b)


def _t_bateman(lam, n0, t, dps=600):
    # linear chain with distinct decay constants: superposition over the initially populated members
    with _t_mp.workdps(dps):
        lam = [_t_mp.mpf(float(x)) for x in lam]
        t = _t_mp.mpf(float(t))
        n = len(lam)
        out = [_t_mp.mpf(0)] * n
        for j in range(n):
            if n0[j] == 0:
                continue
            for k in range(j, n):
                pre = _t_mp.fprod(lam[j:k]) if k > j else _t_mp.mpf(1)
                tot = _t_mp.mpf(0)
                for i in range(j, k + 1):
                    tot += _t_mp.exp(-lam[i] * t) / _t_mp.fprod([lam[l] - lam[i] for l in range(j, k + 1) if l != i])
                out[k] += _t_mp.mpf(float(n0[j])) * pre * tot
        return [float(x) for x in out]


def _t_expm_apply(rates, removal, x0, t, dps=300):
    # exp(t (rates - diag(removal))) x0 with mpmath
    with _t_mp.workdps(dps):
        m = len(removal)
        A = _t_mp.matrix(m, m)
        for i in range(m):
            for j in range(m):
                A[i, j] = _t_mp.mpf(float(rates[i][j])) - (_t_mp.mpf(float(removal[i])) if i == j else 0)
        v = _t_mp.expm(A * _t_mp.mpf(float(t))) * _t_mp.matrix([_t_mp.mpf(float(a)) for a in x0])
        return [v[i] for i in range(m)]


def _t_check(x, target, scale):
    x = np.asarray(x)
    assert x.shape == (len(target),), x.shape
    for a, b in zip(x, target):
        if b > 0.0 and b >= 1e-250 * scale:
            assert _t_rel(a, b) < 1e-10, (a, b)
        else:
            assert abs(a - b) <= max(1e-250 * scale, 1e-300), (a, b)


def _t_activation(lam, branching, sigma, capture_to, n0, history):
    # each period exponentiated from the decay and capture rates at 300 digits
    n = len(lam)
    x = [float(a) for a in n0]
    for duration, flux in history:
        rates = np.zeros((n, n))
        removal = np.zeros(n)
        for i in range(n):
            removal[i] = lam[i] + sigma[i] * 1e-24 * flux
            for j in range(n):
                rates[j, i] = branching[j][i] * lam[i]
            if capture_to[i] >= 0:
                rates[capture_to[i], i] += sigma[i] * 1e-24 * flux
        x = [float(a) for a in _t_expm_apply(rates, removal, x, duration)]
    return x


# gold flux monitor: 197Au (stable, 98.65 b) -> 198Au (2.6941 d, 25100 b) -> 199Au (3.139 d, 30 b);
# 198Au and 199Au beta-decay to mercury, which is not followed
_T_DAY = 86400.0
_T_LAM = np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)])
_T_SIG = np.array([98.65, 25100.0, 30.0])
_T_CAP = np.array([1, 2, -1])
_T_B = np.zeros((3, 3))
_T_N0 = np.array([1e18, 0.0, 0.0])


# gold flux monitor: 197Au (stable, 98.65 b) -> 198Au (2.6941 d, 25100 b) -> 199Au (3.139 d, 30 b)
_T_DAY = 86400.0
_T_DATA = (np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)]), np.zeros((3, 3)),
           np.array([98.65, 25100.0, 30.0]), np.array([1, 2, -1]), np.array([1e18, 0.0, 0.0]))


def _t_activity(k, flux, t_irr, t_cool, lam, branching, sigma, capture_to, n0):
    x = _t_activation(lam, branching, sigma, capture_to, n0, [(t_irr, flux), (t_cool, 0.0)])
    return lam[k] * x[k]


def _t_roundtrip(k, flux, t_irr, t_cool, lo, hi, data):
    act = _t_activity(k, flux, t_irr, t_cool, *data)
    got = monitor_flux(*data, t_irr, t_cool, k, act, lo, hi)
    assert isinstance(got, float), type(got)
    assert _t_rel(got, flux) < 1e-8, (k, flux, got)


# --- test case 0: gold monitor, activity of 198Au after 5 days of irradiation and 1 day of cooling,
# for fluxes from 1e8 to 5e13 n / (cm^2 s) (the activity increases up to about 1.6e15) ---
for _t_flux in (1e8, 3.3e12, 5e13):
    _t_roundtrip(1, _t_flux, 5 * _T_DAY, _T_DAY, 1e6, 1e15, _T_DATA)

# --- test case 1: the two-capture product 199Au at low flux: its activity grows like flux**2 and is
# about 2e-11 Bq at flux 1e4 ---
for _t_flux in (1e4, 7.7e9):
    _t_roundtrip(2, _t_flux, 5 * _T_DAY, _T_DAY, 1e-2, 1e15, _T_DATA)

# --- test case 2: burnup of 198Au (25100 b) bends the activity over (maximum near 1.6e15): at 1e14 the
# slope d ln A / d ln flux is about 0.65 instead of 1 ---
_t_roundtrip(1, 1e14, 5 * _T_DAY, _T_DAY, 1e13, 1.5e15, _T_DATA)
# near the edge of the accuracy statement: at 1.547e14 the slope d ln A / d ln flux is about 0.52
_t_roundtrip(1, 1.547e14, 5 * _T_DAY, _T_DAY, 1e13, 1.5e15, _T_DATA)

# --- test case 3: the ends of the domain: no cooling (t_cool = 0); t_irr = 1e9 s and t_cool = 1e9 s on
# a cobalt monitor (60Co, 5.27 y); flux_hi = 1e18 on a monitor whose product (1e-3 per s, no capture)
# keeps rising up to that flux; activities equal to A(flux_lo) and A(flux_hi) ---
_t_roundtrip(1, 2e11, 5 * _T_DAY, 0.0, 1e6, 1e15, _T_DATA)
_t_co = (np.array([0.0, np.log(2.0) / (10.467 * 60.0), np.log(2.0) / (5.2714 * 3.15576e7), np.log(2.0) / (1.65 * 3600.0)]),
         np.array([[0.0] * 4, [0.0] * 4, [0.0, 0.9975, 0.0, 0.0], [0.0] * 4]),
         np.array([20.7, 0.0, 2.0, 0.0]), np.array([1, -1, 3, -1]), np.array([5e20, 0.0, 0.0, 0.0]))
_t_roundtrip(2, 1e11, 1e9, 1e9, 1e6, 1e13, _t_co)
_t_simple = (np.array([0.0, 1e-3]), np.zeros((2, 2)), np.array([1.0, 0.0]), np.array([1, -1]), np.array([1e20, 0.0]))
_t_roundtrip(1, 1e17, 1e3, 10.0, 1e14, 1e18, _t_simple)
# the smallest positive cross section, 1e-6 b, on the target
_t_faint = (np.array([0.0, 1e-3]), np.zeros((2, 2)), np.array([1e-6, 0.0]), np.array([1, -1]), np.array([1e20, 0.0]))
_t_roundtrip(1, 1e12, 1e3, 10.0, 1e6, 1e18, _t_faint)
# the interval is closed with a two-sided tolerance: activities equal to A(flux_lo) or A(flux_hi), or
# 5e-10 inside them (whose exact roots are about 5e-10 away from the ends), or 5e-10 outside them, give
# back those ends
for _t_end, _t_in in ((1e8, 1.0 + 5e-10), (1e13, 1.0 - 5e-10)):
    for _t_f in (1.0, _t_in, 2.0 - _t_in):
        act = _t_f * _t_activity(1, _t_end, 5 * _T_DAY, _T_DAY, *_T_DATA)
        got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, act, 1e8, 1e13)
        assert _t_rel(got, _t_end) < 1e-11, (got, _t_end, _t_f)
    # at the bound on both sides (|ln| = 0.999e-9) the end is returned
    _t_a = _t_activity(1, _t_end, 5 * _T_DAY, _T_DAY, *_T_DATA)
    for _t_f in (np.exp(0.999e-9), np.exp(-0.999e-9)):
        got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_f * _t_a, 1e8, 1e13)
        assert _t_rel(got, _t_end) < 1e-11, (got, _t_end, _t_f)
    # just beyond the tolerance, outside the interval (|ln| = 1.01e-9 > 1e-9): no flux in the interval
    _t_raised = False
    try:
        monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, (np.exp(-1.01e-9) if _t_end == 1e8 else np.exp(1.01e-9)) * _t_a, 1e8, 1e13)
    except ValueError:
        _t_raised = True
    assert _t_raised, ("activity 1.01e-9 outside the end at %g must raise ValueError" % _t_end)
    # and 2e-9 outside
    _t_out = np.exp(-2e-9) if _t_end == 1e8 else np.exp(2e-9)
    _t_raised = False
    try:
        monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_out * _t_activity(1, _t_end, 5 * _T_DAY, _T_DAY, *_T_DATA), 1e8, 1e13)
    except ValueError:
        _t_raised = True
    assert _t_raised, ("activity 2e-9 outside the end at %g must raise ValueError" % _t_end)

# --- test case 4: an activity outside [A(flux_lo), A(flux_hi)] raises ValueError ---
for _t_act in (_t_activity(1, 1e7, 5 * _T_DAY, _T_DAY, *_T_DATA), _t_activity(1, 1e14, 5 * _T_DAY, _T_DAY, *_T_DATA)):
    try:
        monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_act, 1e8, 1e13)
    except ValueError:
        pass
    else:
        raise AssertionError("monitor_flux must raise ValueError for activity %r" % _t_act)

# --- test case 5: bad times, nuclide index, activity or flux bounds raise ValueError ---
_t_act = _t_activity(1, 1e10, 5 * _T_DAY, _T_DAY, *_T_DATA)
for _t_args in ((0.0, _T_DAY, 1, _t_act, 1e8, 1e13), (5 * _T_DAY, -1.0, 1, _t_act, 1e8, 1e13),
                (5 * _T_DAY, _T_DAY, 0, _t_act, 1e8, 1e13), (5 * _T_DAY, _T_DAY, 1.0, _t_act, 1e8, 1e13), (5 * _T_DAY, _T_DAY, True, _t_act, 1e8, 1e13),
                (5 * _T_DAY, _T_DAY, 1, -5.0, 1e8, 1e13), (5 * _T_DAY, _T_DAY, 1, _t_act, 1e13, 1e8),
                (5 * _T_DAY, _T_DAY, 1, _t_act, 1e-3, 1e13), (2e9, _T_DAY, 1, _t_act, 1e8, 1e13),
                (5 * _T_DAY, 2e9, 1, _t_act, 1e8, 1e13), (5 * _T_DAY, _T_DAY, -1, _t_act, 1e8, 1e13),
                (5 * _T_DAY, _T_DAY, 3, _t_act, 1e8, 1e13), (5 * _T_DAY, _T_DAY, 1, float("nan"), 1e8, 1e13),
                (5 * _T_DAY, _T_DAY, 1, _t_act, 1e8, 2e18), (5 * _T_DAY, _T_DAY, 1, 0.0, 1e8, 1e13),
                (5 * _T_DAY, _T_DAY, 1, _t_act, 1e10, 1e10)):
    try:
        monitor_flux(*_T_DATA, *_t_args)
    except ValueError:
        pass
    else:
        raise AssertionError("monitor_flux must raise ValueError for %r" % (_t_args,))
# precedence of the ends: with bounds 1e-10 apart both end activities lie within the tolerance of
# A(flux_lo); flux_lo is checked first and returned
_t_lo = 1e10
_t_hi = 1e10 * (1.0 + 1e-10)
got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_activity(1, _t_lo, 5 * _T_DAY, _T_DAY, *_T_DATA), _t_lo, _t_hi)
assert _t_rel(got, _t_lo) < 1e-11, (got, _t_lo)
# outside the domain: an irradiation of 1e-300 s leaves about 1e-330 of the target atoms in the product,
# below 1e-250 sum(n0), so A(flux_lo) < 1e-250 lam[k] sum(n0)
_t_raised = False
try:
    monitor_flux(np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([1e-6, 0.0]), np.array([1, -1]),
                 np.array([1e100, 0.0]), 1e-300, 0.0, 1, 1e-230, 1e-2, 1e18)
except ValueError:
    _t_raised = True
assert _t_raised, "A(flux_lo) below 1e-250 lam[k] sum(n0) must raise ValueError"
# nuclide data rejected by step 3: a positive cross section below 1e-6 b (its rate would underflow)
_t_raised = False
try:
    monitor_flux(np.array([0.0, 1e-3]), np.zeros((2, 2)), np.array([1e-300, 0.0]), np.array([1, -1]),
                 np.array([1e100, 0.0]), 1e3, 0.0, 1, 1.0, 1e6, 1e18)
except ValueError:
    _t_raised = True
assert _t_raised, "a cross section of 1e-300 b must raise ValueError"
