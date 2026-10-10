# Independent targets: closed forms, the Bateman sum evaluated in 600-digit arithmetic (it only
# cancels in floating point), and matrix exponentials by mpmath at 300 digits.

# --- test case 0 ---
# burnup of a stable target: n0 exp(-sigma phi t), down to exp(-550), about 1e-239, exp(-575),
# about 8e-250, exp(-575.64), 1.01e-250 just above 1e-250 S (relative rule), and exp(-600), about 3e-261 (below 1e-250 S: absolute rule)
import numpy as np
import mpmath as _t_mp

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_check(x, target, scale):
    x = np.asarray(x)
    assert x.shape == (len(target),), x.shape
    for a, b in zip(x, target):
        if b > 0.0 and b >= 1e-250 * scale:
            assert _t_rel(a, b) < 1e-10, (a, b)
        else:
            assert abs(a - b) <= max(1e-250 * scale, 1e-300), (a, b)

for _t_flux, _t_t in ((1e13, 1e5), (1e17, 2e7), (2e17, 1e10), (1e18, 5.5e6), (1e18, 5.75e6), (1e18, 5756363.229176583), (1e18, 6e6)):
    x = activation_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([100.0]), np.array([-1]), np.array([3.0]),
                             [(_t_t, _t_flux)])
    with _t_mp.workdps(40):
        target = float(3 * _t_mp.exp(-_t_mp.mpf(100) * _t_mp.mpf("1e-24") * _t_flux * _t_t))
    _t_check(x, [target], 3.0)

# --- test case 1 ---
# gold monitor through irradiation and cooling, from 1e2 to 1e17 n / (cm^2 s): 199Au is
# made by two captures and is down to about 1e-9 atoms at the lowest flux
import numpy as np
import mpmath as _t_mp

def _t_rel(a, b):
    return abs(a - b) / abs(b)

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

for _t_flux in (1e2, 1e9, 1e14, 1e17):
    hist = [(5 * _T_DAY, _t_flux), (_T_DAY, 0.0)]
    _t_check(activation_inventory(_T_LAM, _T_B, _T_SIG, _T_CAP, _T_N0, hist),
             _t_activation(_T_LAM, _T_B, _T_SIG, _T_CAP, _T_N0, hist), 1e18)

# --- test case 2 ---
# a reactor-style history (cycles of different flux with cooling between them) on a
# network with decay branching: 59Co captures (20.7 b) into 60mCo (10.467 min), which decays to 60Co
# (5.2714 y) in 99.75 % of its decays; 60Co captures (2 b) into 61Co (1.65 h)
import numpy as np
import mpmath as _t_mp

def _t_rel(a, b):
    return abs(a - b) / abs(b)

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

lam = np.array([0.0, np.log(2.0) / (10.467 * 60.0), np.log(2.0) / (5.2714 * 3.15576e7), np.log(2.0) / (1.65 * 3600.0)])
B = np.zeros((4, 4))
B[2, 1] = 0.9975
sig = np.array([20.7, 0.0, 2.0, 0.0])
cap = np.array([1, -1, 3, -1])
n0 = np.array([5e20, 0.0, 0.0, 0.0])
hist = [(3e6, 2e13), (5e5, 0.0), (3e6, 5e13), (1e3, 0.0), (10.0, 1e14), (4e2, 0.0)]
_t_check(activation_inventory(lam, B, sig, cap, n0, hist), _t_activation(lam, B, sig, cap, n0, hist), 5e20)

# --- test case 3 ---
# splitting a period into two equal periods, or inserting a period of zero length,
# changes nothing
import numpy as np

def _t_rel(a, b):
    return abs(a - b) / abs(b)

def _t_check_pair(x, y, scale):
    # two computed outputs, each within the stated tolerance of the exact amounts: twice that tolerance
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape, (x.shape, y.shape)
    for a, b in zip(x, y):
        if b > 0.0 and b >= 1e-250 * scale:
            assert _t_rel(a, b) < 2e-10, (a, b)
        else:
            assert abs(a - b) <= 2.0 * max(1e-250 * scale, 1e-300), (a, b)

# gold flux monitor: 197Au (stable, 98.65 b) -> 198Au (2.6941 d, 25100 b) -> 199Au (3.139 d, 30 b);
# 198Au and 199Au beta-decay to mercury, which is not followed
_T_DAY = 86400.0
_T_LAM = np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)])
_T_SIG = np.array([98.65, 25100.0, 30.0])
_T_CAP = np.array([1, 2, -1])
_T_B = np.zeros((3, 3))
_T_N0 = np.array([1e18, 0.0, 0.0])

a = activation_inventory(_T_LAM, _T_B, _T_SIG, _T_CAP, _T_N0, [(4e5, 3e12), (1e5, 0.0)])
b = activation_inventory(_T_LAM, _T_B, _T_SIG, _T_CAP, _T_N0, [(2e5, 3e12), (0.0, 1e15), (2e5, 3e12), (1e5, 0.0)])
assert np.asarray(a).shape == (3,), np.asarray(a).shape
_t_check_pair(a, b, 1e18)

# --- test case 4 ---
# the ends of the domain: sigma = 1e7 b, flux = 1e18 (zero cross sections), a period of
# 1e12 s, an empty history (returns n0), an empty inventory (S = 0, all exact zeros), lam = 1e10, a combined
# rate of exactly 1e10 per s, and 30 nuclides
import numpy as np
import mpmath as _t_mp

def _t_rel(a, b):
    return abs(a - b) / abs(b)

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

x = activation_inventory(np.array([0.0, 0.0]), np.zeros((2, 2)), np.array([1e7, 0.0]), np.array([1, -1]),
                         np.array([2.0, 0.0]), [(1e6, 1e10)])
e = np.exp(-0.1)
_t_check(x, [2.0 * e, 2.0 * (1.0 - e)], 2.0)
x = activation_inventory(np.array([1e-3, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2), np.array([-1, -1]),
                         np.array([1.0, 0.0]), [(2e3, 1e18)])
_t_check(x, [np.exp(-2.0), -np.expm1(-2.0)], 1.0)
x = activation_inventory(np.array([1e-12]), np.zeros((1, 1)), np.zeros(1), np.array([-1]), np.array([5.0]), [(1e12, 0.0)])
_t_check(x, [5.0 * np.exp(-1.0)], 5.0)
_t_check(activation_inventory(_T_LAM, _T_B, _T_SIG, _T_CAP, _T_N0, []), _T_N0, 1e18)
_t_check(activation_inventory(_T_LAM, _T_B, _T_SIG, _T_CAP, np.zeros(3), [(1e5, 1e14)]), [0.0, 0.0, 0.0], 0.0)
# lam = 1e10 under no flux: n0 exp(-1e10 t)
x = activation_inventory(np.array([1e10]), np.zeros((1, 1)), np.zeros(1), np.array([-1]), np.array([2.0]), [(1e-9, 0.0)])
_t_check(x, [2.0 * np.exp(-10.0)], 2.0)
# a combined rate of exactly 1e10 per s: lam = 1e10 - 10, sigma = 1e7 b at flux 1e18 (capture rate 10 per s)
# into a stable product: target n0 exp(-1), product n0 (10 / 1e10) (1 - exp(-1))
x = activation_inventory(np.array([1e10 - 10.0, 0.0]), np.zeros((2, 2)), np.array([1e7, 0.0]), np.array([1, -1]),
                         np.array([1.0, 0.0]), [(1e-10, 1e18)])
_t_check(x, [np.exp(-1.0), 1e-9 * -np.expm1(-1.0)], 1.0)
# an empty history with a branching column sum exactly at the bound 1 + 1e-12, and capture links of
# nuclides without cross section (sigma = 0, so no edge even though capture_to forms a loop), give n0 back
_t_check(activation_inventory(np.array([1.0, 0.0]), np.array([[0.0, 0.0], [1.0 + 1e-12, 0.0]]), np.zeros(2),
                              np.array([-1, -1]), np.array([1.0, 2.0]), []), [1.0, 2.0], 3.0)
_t_check(activation_inventory(np.array([1e-3, 0.0]), np.zeros((2, 2)), np.zeros(2), np.array([1, 0]),
                              np.array([1.0, 2.0]), []), [1.0, 2.0], 3.0)
# a valid history given as a tuple, and initial amounts summing to 1e100
hist = ((1e5, 1e14), (1e4, 0.0))
_t_check(activation_inventory(_T_LAM, _T_B, _T_SIG, _T_CAP, _T_N0, hist),
         _t_activation(_T_LAM, _T_B, _T_SIG, _T_CAP, _T_N0, list(hist)), 1e18)
_t_check(activation_inventory(_T_LAM, _T_B, _T_SIG, _T_CAP, np.array([1e100, 0.0, 0.0]), [(1e5, 1e14)]),
         _t_activation(_T_LAM, _T_B, _T_SIG, _T_CAP, np.array([1e100, 0.0, 0.0]), [(1e5, 1e14)]), 1e100)
# the smallest positive cross section, 1e-6 b: sigma phi t = 1e-6 at 1e18 n / (cm^2 s) for 1e6 s
x = activation_inventory(np.array([0.0, 0.0]), np.zeros((2, 2)), np.array([1e-6, 0.0]), np.array([1, -1]),
                         np.array([1.0, 0.0]), [(1e6, 1e18)])
_t_check(x, [np.exp(-1e-6), -np.expm1(-1e-6)], 1.0)
# 30 nuclides with an empty history give n0 back
n0 = np.linspace(1.0, 3.0, 30)
_t_check(activation_inventory(np.full(30, 1e-3), np.diag(np.ones(29), -1), np.zeros(30), np.full(30, -1), n0, []), n0, n0.sum())

# --- test case 5 ---
# a capture loop, capture onto itself or out of range, sigma above 1e7 b or positive below 1e-6 b, branching with a column
# sum of 1.2 or a nonzero diagonal (even with an empty history), a malformed shape (scalar sigma, branching not n x n, n0 too long, capture_to too short), a history that is not a list or tuple (None, a set, a generator) or has a non-numeric entry,
# a capture loop with positive cross sections (even with an empty history), sum(n0) above 1e100, lam above 1e10, 31 nuclides, a NaN cross section, a bad history entry,
# an infinite or too long duration, a too large flux, or a too large combined rate raise ValueError
import numpy as np
_t_loop = np.zeros((2, 2))
_t_loop[0, 1] = 1.0
for _t_bad in ((np.array([0.0, 1.0]), _t_loop, np.array([5.0, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, 1e10)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([0, -1]), np.ones(2), [(1.0, 1e10)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([1, -1]), np.ones(2), [(1.0,)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, -1e10)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([1.0, -1.0]), np.ones(2), [(1.0, 1e10)]),
               (np.array([1e10, 1.0]), np.zeros((2, 2)), np.array([1e7, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, 1e18)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([2e7, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, 1.0)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([-2, -1]), np.ones(2), [(1.0, 1e10)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([2, -1]), np.ones(2), [(1.0, 1e10)]),
               (np.array([1.0, 1.0]), np.array([[0.0, 0.0], [1.2, 0.0]]), np.zeros(2), np.array([-1, -1]), np.ones(2), []),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), [(2e12, 0.0)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), [(1.0, 2e18)]),
               (np.array([2e10, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), []),
               (np.full(31, 1e-3), np.zeros((31, 31)), np.zeros(31), np.full(31, -1), np.ones(31), []),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([np.nan, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, 1e10)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), [(np.inf, 0.0)]),
               (np.array([1.0, 1.0]), np.array([[0.5, 0.0], [0.5, 0.0]]), np.zeros(2), np.array([-1, -1]), np.ones(2), []),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), None),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array(5.0), np.array([-1, -1]), np.ones(2), []),
               (np.array([0.0, 1.0]), np.zeros((3, 3)), np.zeros(2), np.array([-1, -1]), np.ones(2), []),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(3), []),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), [(1.0, "bad")]),
               (np.array([0.0, 0.0]), np.zeros((2, 2)), np.array([5.0, 2.0]), np.array([1, 0]), np.ones(2), []),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([1e-7, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, 1e10)]),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.array([1e100, 1e99]), []),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), {(1.0, 0.0), (2.0, 0.0)}),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), ((1.0, 0.0) for _ in range(2))),
               (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([1]), np.ones(2), [(1.0, 1e10)])):
    try:
        activation_inventory(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("activation_inventory must raise ValueError for %r" % (_t_bad,))
