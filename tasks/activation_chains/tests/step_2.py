# --- test case 0 ---
# Independent targets: closed forms, the Bateman sum evaluated in 600-digit arithmetic (it only
# cancels in floating point), and matrix exponentials by mpmath at 300 digits.
# one nuclide with constant production: n0 exp(-lam t) + q (1 - exp(-lam t)) / lam,
# also when lam t is far below rounding (then about n0 + q t), when production dominates, and pure decay
# down to exp(-550), about 1e-239, exp(-575), about 8e-250, exp(-575.64), 1.01e-250 just above 1e-250 S (relative rule), and exp(-600), about 3e-261 (below 1e-250 S: absolute rule);
# tiny inventories, where F = max(1e-250 S, 1e-300) is 1e-300: n0 = 1e-70 decaying to exp(-500) n0, about 7e-288
# (relative rule), and to exp(-570) n0, about 3e-318 (absolute rule), and a production of 1e-200 per unit time alone
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(x, target, scale):
        x = np.asarray(x, dtype=float).reshape(len(target))
        for a, b in zip(x, target):
            if b >= max(1e-250 * scale, 1e-300):
                assert _t_rel(a, b) < 1e-10, (a, b)
            else:
                assert abs(a - b) <= max(1e-250 * scale, 1e-300), (a, b)

    # output contract: one nuclide gives a numpy array of shape (1,), not a scalar
    x = network_inventory(np.array([0.5]), np.zeros((1, 1)), np.array([3.0]), np.array([1.0]), 2.0)
    assert isinstance(x, np.ndarray) and x.shape == (1,), (type(x), np.shape(x))
    for _t_lam, _t_q, _t_n0, _t_t in ((0.5, 3.0, 1.0, 2.0), (1e-10, 7.0, 0.0, 1e-5), (2.0, 1e6, 1e-3, 1e3), (0.0, 2.0, 1.0, 10.0), (1.0, 0.0, 1.0, 550.0), (1.0, 0.0, 1.0, 575.0), (1.0, 0.0, 1.0, 575.6363229176583), (1.0, 0.0, 1.0, 600.0),
                                     (1.0, 0.0, 1e-70, 500.0), (1.0, 0.0, 1e-70, 570.0), (0.5, 1e-200, 0.0, 2.0)):
        x = network_inventory(np.array([_t_lam]), np.zeros((1, 1)), np.array([_t_q]), np.array([_t_n0]), _t_t)
        with _t_mp.workdps(50):
            L = _t_mp.mpf(_t_lam)
            grow = -_t_mp.expm1(-L * _t_t) / L if _t_lam > 0.0 else _t_mp.mpf(_t_t)
            target = float(_t_n0 * _t_mp.exp(-L * _t_t) + _t_q * grow)
        _t_check(x, [target], _t_n0 + _t_t * _t_q)


_crown_case()

# --- test case 1 ---
# one parent branching into two stable daughters (fractions 0.3 and 0.6, 10 % leaves the
# network): daughter_j = b_j n0 (1 - exp(-lam t)), down to lam t = 1e-12
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(x, target, scale):
        x = np.asarray(x, dtype=float).reshape(len(target))
        for a, b in zip(x, target):
            if b >= max(1e-250 * scale, 1e-300):
                assert _t_rel(a, b) < 1e-10, (a, b)
            else:
                assert abs(a - b) <= max(1e-250 * scale, 1e-300), (a, b)

    for _t_t in (1e-12, 0.4, 50.0):
        B = np.zeros((3, 3))
        B[1, 0], B[2, 0] = 0.3, 0.6
        x = network_inventory(np.array([1.0, 0.0, 0.0]), B, np.zeros(3), np.array([2.0, 0.0, 0.0]), _t_t)
        e = -np.expm1(-_t_t)
        _t_check(x, [2.0 * np.exp(-_t_t), 0.6 * e, 1.2 * e], 2.0)


_crown_case()

# --- test case 2 ---
# branching networks with production, stiff decay constants and nuclides given in a
# non-topological order, against mpmath's matrix exponential at 300 digits
import numpy as np


def _crown_case():
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
        x = np.asarray(x, dtype=float).reshape(len(target))
        for a, b in zip(x, target):
            if b >= max(1e-250 * scale, 1e-300):
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


_crown_case()

# --- test case 3 ---
# a long, strongly branching network: the last nuclides are reached only through many
# low-probability branches (amounts down to about 1e-45); and a very stiff branching series
import numpy as np


def _crown_case():
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
        x = np.asarray(x, dtype=float).reshape(len(target))
        for a, b in zip(x, target):
            if b >= max(1e-250 * scale, 1e-300):
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


_crown_case()

# --- test case 4 ---
# the ends of the domain: t = 0 returns n0, lam = 1e10, t = 1e20 with a stable nuclide
# and production (n0 + q t), a 30-nuclide network, column sums up to 1 + 1e-12 accepted and normalized, and an
# empty inventory without production (S = 0, all exact zeros)
import numpy as np


def _crown_case():
    import numpy as np
    import mpmath as _t_mp

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

    def _t_check(x, target, scale):
        x = np.asarray(x, dtype=float).reshape(len(target))
        for a, b in zip(x, target):
            if b >= max(1e-250 * scale, 1e-300):
                assert _t_rel(a, b) < 1e-10, (a, b)
            else:
                assert abs(a - b) <= max(1e-250 * scale, 1e-300), (a, b)

    B = np.zeros((3, 3))
    B[1, 0], B[2, 1] = 1.0, 0.4
    lam, q, n0 = np.array([2.0, 1e-4, 0.0]), np.array([0.5, 0.0, 1.0]), np.array([1.0, 3.0, 0.0])
    _t_check(network_inventory(lam, B, q, n0, 0.0), n0, n0.sum())
    _t_check(network_inventory(np.array([1e10]), np.zeros((1, 1)), np.zeros(1), np.array([2.0]), 1e-9), [2.0 * np.exp(-10.0)], 2.0)
    _t_check(network_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([3.0]), np.array([1.0]), 1e20), [1.0 + 3e20], 1.0 + 3e20)
    _t_check(network_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([1e80]), np.zeros(1), 1e20), [1e100], 1e100)
    # tiny positive rate and time whose product underflows to 0
    _t_check(network_inventory(np.array([1e-300]), np.zeros((1, 1)), np.zeros(1), np.array([1.0]), 1e-300), [1.0], 1.0)
    lam = np.geomspace(1e-6, 1e6, 30)
    B = np.diag(np.ones(29), -1)
    n0 = np.r_[1.0, np.zeros(29)]
    _t_check(network_inventory(lam, B, np.zeros(30), n0, 50.0), _t_bateman(lam, n0, 50.0), 1.0)
    # column sums in (1, 1 + 1e-12] are rounding of 1: the column is divided by its sum, so the stable daughters
    # share exactly what the parent lost and the network keeps its one atom
    B = np.zeros((3, 3))
    B[1, 0], B[2, 0] = 0.5, 0.5 + 5e-13
    x = network_inventory(np.array([1.0, 0.0, 0.0]), B, np.zeros(3), np.array([1.0, 0.0, 0.0]), 2.0)
    _t_s = 0.5 + (0.5 + 5e-13)
    _t_check(x, [np.exp(-2.0), 0.5 / _t_s * -np.expm1(-2.0), (0.5 + 5e-13) / _t_s * -np.expm1(-2.0)], 1.0)
    assert _t_rel(float(np.sum(np.asarray(x, dtype=float))), 1.0) < 1e-10, x
    B = np.zeros((2, 2))
    B[1, 0] = 1.0 + 1e-12                       # exactly at the accepted bound: normalized to 1
    x = network_inventory(np.array([1.0, 0.0]), B, np.zeros(2), np.array([1.0, 0.0]), 2.0)
    _t_check(x, [np.exp(-2.0), -np.expm1(-2.0)], 1.0)
    _t_check(network_inventory(np.array([1.0, 0.5]), np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2), np.zeros(2), 3.0), [0.0, 0.0], 0.0)


_crown_case()

# --- test case 5 ---
# each stated error rule once, at its boundary where it has one: a cycle, a nonzero diagonal, a column sum of
# 1 + 1e-9, a negative branching entry, a decay constant just above 1e10, a time just above 1e20,
# sum(n0) + t sum(source) = 1.01e101 (just above 1e100), 31 nuclides and a branching matrix of the wrong
# shape raise ValueError
import numpy as np


def _crown_case():
    import numpy as np
    _t_cyc = np.array([[0.0, 1.0], [1.0, 0.0]])
    _t_diag = np.array([[0.5, 0.0], [0.5, 0.0]])
    _t_ok = np.array([[0.0, 0.0], [1.0, 0.0]])
    for _t_bad in ((np.ones(2), _t_cyc, np.zeros(2), np.ones(2), 1.0), (np.ones(2), _t_diag, np.zeros(2), np.ones(2), 1.0),
                   (np.ones(2), np.array([[0.0, 0.0], [1.0 + 1e-9, 0.0]]), np.zeros(2), np.ones(2), 1.0),
                   (np.ones(2), np.array([[0.0, -0.1], [1.0, 0.0]]), np.zeros(2), np.ones(2), 1.0),
                   (np.array([1.0001e10, 1.0]), _t_ok, np.zeros(2), np.ones(2), 1.0),
                   (np.ones(2), _t_ok, np.zeros(2), np.ones(2), 1.0001e20),
                   (np.zeros(1), np.zeros((1, 1)), np.array([1.01e81]), np.zeros(1), 1e20),
                   (np.ones(31), np.zeros((31, 31)), np.zeros(31), np.ones(31), 1.0),
                   (np.ones(2), np.zeros((3, 3)), np.zeros(2), np.ones(2), 1.0)):
        try:
            network_inventory(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("network_inventory must raise ValueError for %r" % (_t_bad,))


_crown_case()

# --- test case 6 ---
# a diamond network given out of topological order: 0 decays to 1 (40 %) and 2 (60 %), both decay to 3
# (all of 1, 70 % of 2; 30 % of the decays of 2 leave the network), and 3 decays to stable 4; from 0 alone,
# every amount is the sum over the decay paths that reach it of the product of the fractions times the
# Bateman amount of the linear chain along the path (600 digits); the atoms lost are 30 % of the decays of 2
import numpy as np


def _crown_case():
    import mpmath as _t_mp

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_bateman_last(lam, t, dps=600):
        # last member of a linear chain with distinct decay constants, one atom initially in the first member
        with _t_mp.workdps(dps):
            lam = [_t_mp.mpf(float(x)) for x in lam]
            t = _t_mp.mpf(float(t))
            k = len(lam) - 1
            tot = _t_mp.mpf(0)
            for i in range(k + 1):
                tot += _t_mp.exp(-lam[i] * t) / _t_mp.fprod([lam[l] - lam[i] for l in range(k + 1) if l != i])
            return _t_mp.fprod(lam[:k]) * tot

    def _t_check(x, target, scale):
        x = np.asarray(x, dtype=float).reshape(len(target))
        for a, b in zip(x, target):
            if b >= max(1e-250 * scale, 1e-300):
                assert _t_rel(a, b) < 1e-10, (a, b)
            else:
                assert abs(a - b) <= max(1e-250 * scale, 1e-300), (a, b)

    # physical nuclides 0..4 (decay constants per s) stored at the positions pos[0..4]
    rate = [1e-3, 0.7, 2.5e-2, 3.0, 0.0]
    edges = {(0, 1): 0.4, (0, 2): 0.6, (1, 3): 1.0, (2, 3): 0.7, (3, 4): 1.0}
    paths = {0: [[0]], 1: [[0, 1]], 2: [[0, 2]], 3: [[0, 1, 3], [0, 2, 3]], 4: [[0, 1, 3, 4], [0, 2, 3, 4]]}
    pos = [3, 0, 4, 1, 2]
    lam = np.zeros(5)
    B = np.zeros((5, 5))
    for p in range(5):
        lam[pos[p]] = rate[p]
    for (a, b), f in edges.items():
        B[pos[b], pos[a]] = f
    n0 = np.zeros(5)
    n0[pos[0]] = 2.0
    for _t_t in (10.0, 1e3, 1e5):
        with _t_mp.workdps(600):
            amounts = []
            for p in range(5):
                tot = _t_mp.mpf(0)
                for path in paths[p]:
                    frac = _t_mp.fprod([_t_mp.mpf(edges[(path[m], path[m + 1])]) for m in range(len(path) - 1)])
                    tot += frac * _t_bateman_last([rate[q] for q in path], _t_t)
                amounts.append(2 * tot)
            lost = _t_mp.mpf("0.3") * (_t_mp.mpf("0.6") * (2 - amounts[0]) - amounts[2])
            target = [0.0] * 5
            for p in range(5):
                target[pos[p]] = float(amounts[p])
            total = float(2 - lost)
        x = network_inventory(lam, B, np.zeros(5), n0, _t_t)
        _t_check(x, target, 2.0)
        # balance: the atoms still in the network are the initial ones minus 30 % of the decays of nuclide 2
        assert _t_rel(float(np.sum(x)), total) < 1e-10, (float(np.sum(x)), total)


_crown_case()

# --- test case 7 ---
# a chain 0 -> 1 -> 2 -> 3 (3 decays out of the network) with one initial atom in member 0 and a constant
# production q = 5e-3 per unit time into member 0: the production acts as a parent of decay constant 0
# feeding member 0 at the rate q, so each amount is a Bateman sum that includes the rate 0 (600 digits),
# plus the Bateman amount of the initial atom; at long times (lam t >= 200 for every member) each member
# reaches the saturation amount q / lam, where its decay rate equals the production rate
import numpy as np


def _crown_case():
    import mpmath as _t_mp

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check(x, target, scale):
        x = np.asarray(x, dtype=float).reshape(len(target))
        for a, b in zip(x, target):
            if b >= max(1e-250 * scale, 1e-300):
                assert _t_rel(a, b) < 1e-10, (a, b)
            else:
                assert abs(a - b) <= max(1e-250 * scale, 1e-300), (a, b)

    def _t_bateman_sum(rates, k, t):
        # sum_i exp(-r_i t) / prod_{l != i} (r_l - r_i) over the first k + 1 rates (distinct), 600 digits
        tot = _t_mp.mpf(0)
        for i in range(k + 1):
            tot += _t_mp.exp(-rates[i] * t) / _t_mp.fprod([rates[l] - rates[i] for l in range(k + 1) if l != i])
        return tot

    lam = np.array([2e-3, 5e-2, 1e-1, 3e-2])
    B = np.diag(np.ones(3), -1)
    q = np.array([5e-3, 0.0, 0.0, 0.0])
    n0 = np.array([1.0, 0.0, 0.0, 0.0])
    for _t_t in (10.0, 300.0, 1e5):
        with _t_mp.workdps(600):
            r = [_t_mp.mpf(float(v)) for v in lam]
            T = _t_mp.mpf(_t_t)
            Q = _t_mp.mpf(float(q[0]))
            target = []
            for k in range(4):
                pre = _t_mp.fprod(r[:k]) if k > 0 else _t_mp.mpf(1)
                from_atom = pre * _t_bateman_sum(r, k, T)
                from_source = Q * pre * _t_bateman_sum([_t_mp.mpf(0)] + r, k + 1, T)
                target.append(float(from_atom + from_source))
        x = network_inventory(lam, B, q, n0, _t_t)
        _t_check(x, target, 1.0 + _t_t * 5e-3)
    # saturation: lam[k] x[k] = q for every member once lam t >= 200
    x = network_inventory(lam, B, q, n0, 1e5)
    for k in range(4):
        assert _t_rel(lam[k] * float(np.asarray(x, dtype=float).reshape(4)[k]), 5e-3) < 1e-10, (k, x)


_crown_case()
