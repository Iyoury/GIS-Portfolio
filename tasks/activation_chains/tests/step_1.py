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


_T_YEAR, _T_DAY, _T_MIN = 3.15576e7, 86400.0, 60.0
# uranium series from 238U down to stable 206Pb (half-lives in s): decay constants from 4.9e-18 to 4.2e3 per s
_T_U238 = list(np.log(2.0) / np.array([4.468e9 * _T_YEAR, 24.10 * _T_DAY, 1.17 * _T_MIN, 2.455e5 * _T_YEAR,
                                        7.538e4 * _T_YEAR, 1600.0 * _T_YEAR, 3.8235 * _T_DAY, 3.098 * _T_MIN,
                                        26.8 * _T_MIN, 19.9 * _T_MIN, 164.3e-6, 22.2 * _T_YEAR, 5.012 * _T_DAY,
                                        138.376 * _T_DAY])) + [0.0]

# --- test case 0: one nuclide: n0 exp(-lam t), down to 1e-217; a stable nuclide stays ---
for _t_lam, _t_t in ((0.3, 2.0), (1e-9, 1e3), (5.0, 100.0), (0.0, 1e20)):
    x = chain_inventory(np.array([_t_lam]), np.array([2.5]), _t_t)
    _t_check(x, [2.5 * np.exp(-_t_lam * _t_t)], 2.5)

# --- test case 1: equal decay constants (the Bateman formula is undefined): from member 0 alone,
# x_k = (lam t)**k / k! exp(-lam t) for k < n - 1, here down to 1e-79 ---
from math import factorial as _t_fact
for _t_lt in (1e-3, 0.7, 30.0):
    n = 20
    x = chain_inventory(np.full(n, 2.0), np.r_[1.0, np.zeros(n - 1)], _t_lt / 2.0)
    _t_check(x[:-1], [_t_lt ** k / _t_fact(k) * np.exp(-_t_lt) for k in range(n - 1)], 1.0)

# --- test case 2: nearly equal decay constants (relative differences 1e-13 to 1e-6), where the Bateman
# sum cancels to all digits in double precision ---
for _t_d, _t_t in ((1e-13, 0.5), (1e-9, 3.0), (1e-6, 40.0)):
    lam = 0.8 * (1.0 + _t_d * np.arange(1, 8))
    n0 = np.array([1.0, 0.0, 0.3, 0.0, 0.0, 1e-3, 0.0])
    _t_check(chain_inventory(lam, n0, _t_t), _t_bateman(lam, n0, _t_t), n0.sum())

# --- test case 3: the 238U series from pure 238U (decay constants spread over 21 decades) after one
# year, 1e4 years and 1e9 years, against the Bateman sum at 600 digits ---
n0 = np.r_[1.0, np.zeros(14)]
for _t_t in (1.0 * _T_YEAR, 1e4 * _T_YEAR, 1e9 * _T_YEAR):
    x = chain_inventory(np.array(_T_U238), n0, _t_t)
    target = _t_bateman(_T_U238[:-1], n0[:-1], _t_t)
    _t_check(x[:-1], target, 1.0)
    if _t_t > 1e5 * _T_YEAR:
        # atoms are conserved: the stable end member holds the rest
        assert _t_rel(x[-1], 1.0 - sum(target)) < 1e-10, (x[-1], 1.0 - sum(target))

# --- test case 4: every member populated initially, mixed fast and slow members ---
lam = np.array([3e-7, 2.0, 1e-12, 50.0, 4e-3, 7e-5])
n0 = np.array([1.0, 2e-3, 0.5, 0.0, 1e-8, 3.0])
for _t_t in (0.01, 1e3, 1e6):
    _t_check(chain_inventory(lam, n0, _t_t), _t_bateman(lam, n0, _t_t), n0.sum())

# --- test case 5: the ends of the domain: t = 0 returns n0, lam = 1e10, a 30-member chain (with equal
# and distinct decay constants), an empty inventory (S = 0, all exact zeros) ---
lam = np.array([5.0, 1e-3, 7e2])
n0 = np.array([1.0, 2.0, 0.5])
_t_check(chain_inventory(lam, n0, 0.0), n0, n0.sum())
_t_check(chain_inventory(np.array([1e10]), np.array([4.0]), 1e-9), [4.0 * np.exp(-10.0)], 4.0)
x = chain_inventory(np.full(30, 1.0), np.r_[1.0, np.zeros(29)], 2.0)
_t_check(x[:-1], [2.0 ** k / _t_fact(k) * np.exp(-2.0) for k in range(29)], 1.0)
lam = np.geomspace(1e-6, 1e6, 30)
n0 = np.r_[1.0, np.zeros(29)]
_t_check(chain_inventory(lam, n0, 50.0), _t_bateman(lam, n0, 50.0), 1.0)
_t_check(chain_inventory(np.array([0.3, 2.0]), np.zeros(2), 10.0), [0.0, 0.0], 0.0)

# --- test case 6: bad shapes, decay constants, amounts or times raise ValueError ---
for _t_bad in ((np.array([0.1, 0.2]), np.array([1.0]), 1.0), (np.zeros(31), np.ones(31), 1.0),
               (np.zeros((2, 2)), np.ones((2, 2)), 1.0), (np.array([]), np.array([]), 1.0),
               (np.array([-0.1]), np.array([1.0]), 1.0), (np.array([2e10]), np.array([1.0]), 1.0),
               (np.array([0.1]), np.array([-1.0]), 1.0), (np.array([0.1]), np.array([1.0]), -1.0),
               (np.array([0.1]), np.array([1.0]), 2e20), (np.array([np.nan]), np.array([1.0]), 1.0)):
    try:
        chain_inventory(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("chain_inventory%r must raise ValueError" % (_t_bad,))
