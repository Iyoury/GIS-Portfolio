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


def _t_network(lam, branching, source, n0, t):
    n = len(lam)
    rates = np.zeros((n + 1, n + 1))
    rates[:n, :n] = np.asarray(branching) * np.asarray(lam)[None, :]
    rates[:n, n] = source
    v = _t_expm_apply(rates, list(lam) + [0.0], list(n0) + [1.0], t)
    return [float(a) for a in v[:n]]


# --- test case 0: one nuclide with constant production: n0 exp(-lam t) + q (1 - exp(-lam t)) / lam,
# also when lam t is far below rounding (then about n0 + q t), when production dominates, and pure decay
# down to exp(-550), about 1e-239 ---
for _t_lam, _t_q, _t_n0, _t_t in ((0.5, 3.0, 1.0, 2.0), (1e-10, 7.0, 0.0, 1e-5), (2.0, 1e6, 1e-3, 1e3), (0.0, 2.0, 1.0, 10.0), (1.0, 0.0, 1.0, 550.0)):
    x = network_inventory(np.array([_t_lam]), np.zeros((1, 1)), np.array([_t_q]), np.array([_t_n0]), _t_t)
    with _t_mp.workdps(50):
        L = _t_mp.mpf(_t_lam)
        grow = _t_mp.mpf(_t_t) if _t_lam == 0.0 else -_t_mp.expm1(-L * _t_t) / L
        target = float(_t_n0 * _t_mp.exp(-L * _t_t) + _t_q * grow)
    _t_check(x, [target], _t_n0 + _t_t * _t_q)

# --- test case 1: one parent branching into two stable daughters (fractions 0.3 and 0.6, 10 % leaves the
# network): daughter_j = b_j n0 (1 - exp(-lam t)), down to lam t = 1e-12 ---
for _t_t in (1e-12, 0.4, 50.0):
    B = np.zeros((3, 3))
    B[1, 0], B[2, 0] = 0.3, 0.6
    x = network_inventory(np.array([1.0, 0.0, 0.0]), B, np.zeros(3), np.array([2.0, 0.0, 0.0]), _t_t)
    e = -np.expm1(-_t_t)
    _t_check(x, [2.0 * np.exp(-_t_t), 0.6 * e, 1.2 * e], 2.0)

# --- test case 2: branching networks with production, stiff decay constants and nuclides given in a
# non-topological order, against mpmath's matrix exponential at 300 digits ---
_t_rng = np.random.default_rng(7)
for case in range(5):
    n = 6
    lam = 10.0 ** _t_rng.uniform(-9, 4, n)
    B = np.tril(_t_rng.uniform(0.0, 1.0, (n, n)) * (_t_rng.uniform(0.0, 1.0, (n, n)) < 0.6), -1)
    B = B / np.maximum(B.sum(axis=0), 1.0)
    perm = _t_rng.permutation(n)
    B, lam = B[np.ix_(perm, perm)], lam[perm]
    q = _t_rng.uniform(0.0, 1.0, n) * (_t_rng.uniform(0.0, 1.0, n) < 0.4)
    n0 = _t_rng.uniform(0.0, 1.0, n) * (_t_rng.uniform(0.0, 1.0, n) < 0.5)
    n0[perm[0]] += 1.0
    t = 10.0 ** _t_rng.uniform(-3, 7)
    _t_check(network_inventory(lam, B, q, n0, t), _t_network(lam, B, q, n0, t), n0.sum() + t * q.sum())

# --- test case 3: a long, strongly branching network: the last nuclides are reached only through many
# low-probability branches (amounts down to about 1e-45); and a very stiff branching series ---
n = 12
lam = np.geomspace(1e-3, 1e3, n)
B = np.zeros((n, n))
for i in range(n - 1):
    B[i + 1, i] = 1e-9
    if i + 2 < n:
        B[i + 2, i] = 1e-6
x = network_inventory(lam, B, np.r_[1e-3, np.zeros(n - 1)], np.r_[1.0, np.zeros(n - 1)], 20.0)
target = _t_network(lam, B, np.r_[1e-3, np.zeros(n - 1)], np.r_[1.0, np.zeros(n - 1)], 20.0)
assert min(target) < 1e-40, min(target)
_t_check(x, target, 1.0 + 20.0 * 1e-3)

# the 238U series with the 0.02 % branch of 214Bi through 210Tl (1.30 min) to 210Pb, after 3e5 years:
# decay constants from 4.9e-18 to 4.2e3 per s, so exp(t A) needs about 55 halvings of t
_t_y, _t_d, _t_m = 3.15576e7, 86400.0, 60.0
lam = np.log(2.0) / np.array([4.468e9 * _t_y, 24.10 * _t_d, 1.17 * _t_m, 2.455e5 * _t_y, 7.538e4 * _t_y, 1600.0 * _t_y,
                              3.8235 * _t_d, 3.098 * _t_m, 26.8 * _t_m, 19.9 * _t_m, 164.3e-6, 22.2 * _t_y, 5.012 * _t_d,
                              138.376 * _t_d, 1.30 * _t_m])
n = lam.size
B = np.zeros((n, n))
for i in range(n - 2):
    B[i + 1, i] = 1.0
B[10, 9], B[n - 1, 9], B[11, n - 1] = 1.0 - 2e-4, 2e-4, 1.0
n0 = np.r_[1.0, np.zeros(n - 1)]
_t_check(network_inventory(lam, B, np.zeros(n), n0, 3e5 * _t_y), _t_network(lam, B, np.zeros(n), n0, 3e5 * _t_y), 1.0)

# --- test case 4: the ends of the domain: t = 0 returns n0, lam = 1e10, t = 1e20 with a stable nuclide
# and production (n0 + q t), a 30-nuclide network, column sums up to 1 + 1e-12 accepted, and an
# empty inventory without production (S = 0, all exact zeros) ---
B = np.zeros((3, 3))
B[1, 0], B[2, 1] = 1.0, 0.4
lam, q, n0 = np.array([2.0, 1e-4, 0.0]), np.array([0.5, 0.0, 1.0]), np.array([1.0, 3.0, 0.0])
_t_check(network_inventory(lam, B, q, n0, 0.0), n0, n0.sum())
_t_check(network_inventory(np.array([1e10]), np.zeros((1, 1)), np.zeros(1), np.array([2.0]), 1e-9), [2.0 * np.exp(-10.0)], 2.0)
_t_check(network_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([3.0]), np.array([1.0]), 1e20), [1.0 + 3e20], 1.0 + 3e20)
_t_check(network_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([1e80]), np.zeros(1), 1e20), [1e100], 1e100)
lam = np.geomspace(1e-6, 1e6, 30)
B = np.diag(np.ones(29), -1)
n0 = np.r_[1.0, np.zeros(29)]
_t_check(network_inventory(lam, B, np.zeros(30), n0, 50.0), _t_bateman(lam, n0, 50.0), 1.0)
B = np.zeros((3, 3))
B[1, 0], B[2, 0] = 0.5, 0.5 + 5e-13
x = network_inventory(np.array([1.0, 0.0, 0.0]), B, np.zeros(3), np.array([1.0, 0.0, 0.0]), 2.0)
_t_check(x, [np.exp(-2.0), 0.5 * -np.expm1(-2.0), (0.5 + 5e-13) * -np.expm1(-2.0)], 1.0)
B = np.zeros((2, 2))
B[1, 0] = 1.0 + 1e-12                       # exactly at the accepted bound
x = network_inventory(np.array([1.0, 0.0]), B, np.zeros(2), np.array([1.0, 0.0]), 2.0)
_t_check(x, [np.exp(-2.0), (1.0 + 1e-12) * -np.expm1(-2.0)], 1.0)
_t_check(network_inventory(np.array([1.0, 0.5]), np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2), np.zeros(2), 3.0), [0.0, 0.0], 0.0)

# --- test case 5: a cycle, a nonzero diagonal, a column sum above 1 + 1e-12, 31 nuclides, sum(n0) + t sum(source) above 1e100, negative production or a bad time
# raise ValueError ---
_t_cyc = np.array([[0.0, 1.0], [1.0, 0.0]])
_t_diag = np.array([[0.5, 0.0], [0.5, 0.0]])
_t_over = np.array([[0.0, 0.0], [1.2, 0.0]])
_t_ok = np.array([[0.0, 0.0], [1.0, 0.0]])
for _t_bad in ((np.ones(2), _t_cyc, np.zeros(2), np.ones(2), 1.0), (np.ones(2), _t_diag, np.zeros(2), np.ones(2), 1.0),
               (np.ones(2), _t_over, np.zeros(2), np.ones(2), 1.0), (np.ones(2), _t_ok, np.array([0.0, -1.0]), np.ones(2), 1.0),
               (np.ones(2), _t_ok, np.zeros(2), np.ones(2), float("nan")), (np.array([2e10, 1.0]), _t_ok, np.zeros(2), np.ones(2), 1.0), (np.ones(2), _t_ok, np.zeros(2), np.ones(2), 2e20), (np.ones(2), np.array([[0.0, 0.0], [1.0 + 1e-9, 0.0]]), np.zeros(2), np.ones(2), 1.0), (np.ones(31), np.zeros((31, 31)), np.zeros(31), np.ones(31), 1.0), (np.zeros(1), np.zeros((1, 1)), np.array([1e90]), np.zeros(1), 1e20), (np.zeros(1), np.zeros((1, 1)), np.array([1e81]), np.zeros(1), 1e20), (np.ones(2), np.zeros((3, 3)), np.zeros(2), np.ones(2), 1.0)):
    try:
        network_inventory(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("network_inventory must raise ValueError for %r" % (_t_bad,))
