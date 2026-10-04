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
        if b >= 1e-250 * scale:
            assert _t_rel(a, b) < 1e-10, (a, b)
        else:
            assert abs(a - b) <= 1e-250 * scale, (a, b)


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


_T_YEAR, _T_DAY, _T_MIN = 3.15576e7, 86400.0, 60.0

# --- test case 0: the 238U series as a linear chain and as a branching network (with 0.02 % of 214Bi
# decaying to 210Tl, 1.30 min, which feeds 210Pb) after 3e5 years, against the Bateman sum ---
lam = list(np.log(2.0) / np.array([4.468e9 * _T_YEAR, 24.10 * _T_DAY, 1.17 * _T_MIN, 2.455e5 * _T_YEAR,
                                   7.538e4 * _T_YEAR, 1600.0 * _T_YEAR, 3.8235 * _T_DAY, 3.098 * _T_MIN,
                                   26.8 * _T_MIN, 19.9 * _T_MIN, 164.3e-6, 22.2 * _T_YEAR, 5.012 * _T_DAY,
                                   138.376 * _T_DAY])) + [0.0]
n = len(lam)
n0 = np.r_[1e20, np.zeros(n - 1)]
t = 3e5 * _T_YEAR
x = chain_inventory(np.array(lam), n0, t)
target = _t_bateman(lam[:-1], n0[:-1], t)
_t_check(x[:-1], target, 1e20)
# the same chain with a 210Tl side branch (index n): 214Bi (index 9) -> 210Tl with 2e-4, 210Tl -> 210Pb (index 11)
lam2 = np.r_[lam, np.log(2.0) / (1.30 * _T_MIN)]
B = np.zeros((n + 1, n + 1))
for i in range(n - 1):
    B[i + 1, i] = 1.0
B[10, 9], B[n, 9], B[11, n] = 1.0 - 2e-4, 2e-4, 1.0
y = network_inventory(lam2, B, np.zeros(n + 1), np.r_[n0, 0.0], t)
rates = B * lam2[None, :]
target2 = [float(a) for a in _t_expm_apply(rates, lam2, np.r_[n0, 0.0], t)]
_t_check(y, target2, 1e20)
assert 0.0 < y[n] < 1e-3 * y[9], (y[n], y[9])

# --- test case 1: flux-monitor analysis on a cobalt network after a two-cycle history: the 60Co
# activity measured after cooling gives back the flux of the second cycle ---
lam = np.array([0.0, np.log(2.0) / (10.467 * 60.0), np.log(2.0) / (5.2714 * _T_YEAR), np.log(2.0) / (1.65 * 3600.0)])
B = np.zeros((4, 4))
B[2, 1] = 0.9975
sig = np.array([20.7, 0.0, 2.0, 0.0])
cap = np.array([1, -1, 3, -1])
n0 = np.array([5e20, 0.0, 0.0, 0.0])
for flux in (3e9, 2e13):
    act = lam[2] * _t_activation(lam, B, sig, cap, n0, [(3e6, flux), (2e5, 0.0)])[2]
    got = monitor_flux(lam, B, sig, cap, n0, 3e6, 2e5, 2, act, 1e6, 1e15)
    assert _t_rel(got, flux) < 1e-8, (got, flux)
    _t_check(activation_inventory(lam, B, sig, cap, n0, [(3e6, flux), (2e5, 0.0)]),
             _t_activation(lam, B, sig, cap, n0, [(3e6, flux), (2e5, 0.0)]), 5e20)
