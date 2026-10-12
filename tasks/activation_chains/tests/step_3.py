# --- test case 0 ---
# Independent targets: closed forms, the Bateman sum of a capture chain evaluated in 600-digit
# arithmetic (it only cancels in floating point), and matrix exponentials by mpmath at 300 digits.
# burnup of a stable target: n0 exp(-sigma phi t), down to exp(-550), about 1e-239, exp(-575),
# about 8e-250, exp(-575.64), 1.01e-250 just above 1e-250 S (relative rule), and exp(-600), about 3e-261 (below 1e-250 S: absolute rule);
# and a tiny inventory, where the threshold F = max(1e-250 S, 1e-300) is set by 1e-300
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
    x = activation_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([100.0]), np.array([-1]), np.array([3.0]),
                             [(1e5, 1e13)])
    assert isinstance(x, np.ndarray) and x.shape == (1,), (type(x), np.shape(x))
    for _t_flux, _t_t in ((1e13, 1e5), (1e17, 2e7), (2e17, 1e10), (1e18, 5.5e6), (1e18, 5.75e6), (1e18, 5756363.229176583), (1e18, 6e6)):
        x = activation_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([100.0]), np.array([-1]), np.array([3.0]),
                                 [(_t_t, _t_flux)])
        with _t_mp.workdps(40):
            target = float(3 * _t_mp.exp(-_t_mp.mpf(100) * _t_mp.mpf("1e-24") * _t_flux * _t_t))
        _t_check(x, [target], 3.0)
    # S = 1e-70, F = 1e-300: sigma phi t = 500 leaves about 7e-288 (relative rule), 570 about 3e-318 (a subnormal
    # double above 1e-250 S but below F: absolute rule)
    for _t_t in (5e6, 5.7e6):
        x = activation_inventory(np.array([0.0]), np.zeros((1, 1)), np.array([100.0]), np.array([-1]), np.array([1e-70]),
                                 [(_t_t, 1e18)])
        with _t_mp.workdps(40):
            target = float(_t_mp.mpf(1e-70) * _t_mp.exp(-_t_mp.mpf(100) * _t_mp.mpf("1e-24") * _t_mp.mpf(1e18) * _t_mp.mpf(_t_t)))
        _t_check(x, [target], 1e-70)


_crown_case()

# --- test case 1 ---
# gold monitor through irradiation and cooling, from 1e2 to 1e17 n / (cm^2 s): 199Au is
# made by two captures and is down to about 1e-9 atoms at the lowest flux
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


_crown_case()

# --- test case 2 ---
# a reactor-style history (cycles of different flux with cooling between them) on a
# network with decay branching: 59Co captures (20.7 b) into 60mCo (10.467 min), which decays to 60Co
# (5.2714 y) in 99.75 % of its decays; 60Co captures (2 b) into 61Co (1.65 h)
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


_crown_case()

# --- test case 3 ---
# splitting a period into two equal periods, or inserting a period of zero length,
# changes nothing
import numpy as np


def _crown_case():
    import numpy as np

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    def _t_check_pair(x, y, scale):
        # two computed outputs, each within the stated tolerance of the exact amounts: twice that tolerance
        x, y = np.asarray(x, dtype=float).reshape(3), np.asarray(y, dtype=float).reshape(3)
        for a, b in zip(x, y):
            if b >= max(1e-250 * scale, 1e-300):
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
    _t_check_pair(a, b, 1e18)


_crown_case()

# --- test case 4 ---
# the ends of the domain: sigma = 1e7 b, flux = 1e18 (zero cross sections), a period of
# 1e12 s, an empty history (returns n0), an empty inventory (S = 0, all exact zeros), lam = 1e10, a combined
# rate of exactly 1e10 per s, and 30 nuclides (an empty history, and a chain of 29 successive captures)
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

    def _t_capture_chain(lam, sigma, flux, n0_first, t_irr, t_cool, dps=600):
        # nuclide i captures into i + 1 (rate c_i = sigma_i 1e-24 flux) and decays out of the network (lam_i):
        # during the irradiation a linear chain with transfer rates c_i and removal rates r_i = lam_i + c_i,
        # x_k = n0 prod_{i<k} c_i sum_{i<=k} exp(-r_i t) / prod_{l<=k, l!=i} (r_l - r_i) (distinct r_i); during
        # the cooling every nuclide only decays
        with _t_mp.workdps(dps):
            c = [_t_mp.mpf(float(s)) * _t_mp.mpf("1e-24") * _t_mp.mpf(float(flux)) for s in sigma]
            r = [_t_mp.mpf(float(l)) + ci for l, ci in zip(lam, c)]
            T, Tc = _t_mp.mpf(float(t_irr)), _t_mp.mpf(float(t_cool))
            out = []
            for k in range(len(lam)):
                tot = _t_mp.mpf(0)
                for i in range(k + 1):
                    tot += _t_mp.exp(-r[i] * T) / _t_mp.fprod([r[l] - r[i] for l in range(k + 1) if l != i])
                xk = _t_mp.mpf(float(n0_first)) * _t_mp.fprod(c[:k]) * tot
                out.append(float(xk * _t_mp.exp(-_t_mp.mpf(float(lam[k])) * Tc)))
            return out

    # 30 nuclides through an irradiation and a cooling: a stable target followed by 29 radioactive products, each
    # made by a capture on the previous one (up to 29 successive captures, amounts down to about 1e-73 of n0)
    lam = np.r_[0.0, np.geomspace(1e-6, 1e-3, 29)]
    sig = np.r_[np.geomspace(1e2, 1e4, 29), 0.0]
    cap = np.r_[np.arange(1, 30), -1]
    n0 = np.r_[1e20, np.zeros(29)]
    _t_check(activation_inventory(lam, np.zeros((30, 30)), sig, cap, n0, [(1e5, 1e15), (2e4, 0.0)]),
             _t_capture_chain(lam, sig, 1e15, 1e20, 1e5, 2e4), 1e20)


_crown_case()

# --- test case 5 ---
# each stated error rule once: a cycle through a decay and a capture edge (with an empty history), capture
# onto itself, out of range or from a non-integer array, a positive cross section below 1e-6 b, a combined
# rate above 1e10 per s, a branching column sum of 1.2 (with an empty history), a history given as a set or
# with an entry that is not a pair, a duration above 1e12 s, 31 nuclides, sum(n0) above 1e100 and a
# negative decay constant raise ValueError
import numpy as np


def _crown_case():
    import numpy as np
    _t_loop = np.zeros((2, 2))
    _t_loop[0, 1] = 1.0
    for _t_bad in ((np.array([0.0, 1.0]), _t_loop, np.array([5.0, 0.0]), np.array([1, -1]), np.ones(2), []),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([0, -1]), np.ones(2), [(1.0, 1e10)]),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([2, -1]), np.ones(2), [(1.0, 1e10)]),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([1.0, -1.0]), np.ones(2), [(1.0, 1e10)]),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([1e-7, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, 1e10)]),
                   (np.array([1e10, 1.0]), np.zeros((2, 2)), np.array([1e7, 0.0]), np.array([1, -1]), np.ones(2), [(1.0, 1e18)]),
                   (np.array([1.0, 1.0]), np.array([[0.0, 0.0], [1.2, 0.0]]), np.zeros(2), np.array([-1, -1]), np.ones(2), []),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), {(1.0, 0.0), (2.0, 0.0)}),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([5.0, 0.0]), np.array([1, -1]), np.ones(2), [(1.0,)]),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), [(1.0001e12, 0.0)]),
                   (np.full(31, 1e-3), np.zeros((31, 31)), np.zeros(31), np.full(31, -1), np.ones(31), []),
                   (np.array([0.0, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.array([1e100, 1e99]), []),
                   (np.array([-1e-3, 1.0]), np.zeros((2, 2)), np.zeros(2), np.array([-1, -1]), np.ones(2), [(1.0, 0.0)])):
        try:
            activation_inventory(*_t_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("activation_inventory must raise ValueError for %r" % (_t_bad,))


_crown_case()

# --- test case 6 ---
# single-capture activation with burnup of target and product, in closed form: a stable target (removal
# rate c = sigma_t 1e-24 phi) feeds a product removed at r = lam_p + sigma_p 1e-24 phi, so after an
# irradiation of length t the product holds n0 c (exp(-c t) - exp(-r t)) / (r - c) (n0 c t exp(-c t) when
# r = c), then it decays during the cooling; 50-digit arithmetic. Cobalt (59Co 37.18 b -> 60Co 5.2714 y,
# 2.0 b) from a negligible to a dominant burnup (sigma phi t = 111 at 1e17), a product removed at the same
# rate as the target, and the saturation activity of 198Au, which equals the production rate n0 c
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

    def _t_two_member(n0, sig_t, lam_p, sig_p, flux, t_irr, t_cool):
        with _t_mp.workdps(50):
            c = _t_mp.mpf(float(sig_t)) * _t_mp.mpf("1e-24") * _t_mp.mpf(float(flux))
            r = _t_mp.mpf(float(lam_p)) + _t_mp.mpf(float(sig_p)) * _t_mp.mpf("1e-24") * _t_mp.mpf(float(flux))
            T, N = _t_mp.mpf(float(t_irr)), _t_mp.mpf(float(n0))
            if r == c:
                xp = N * c * T * _t_mp.exp(-c * T)
            else:
                xp = N * c * (_t_mp.exp(-c * T) - _t_mp.exp(-r * T)) / (r - c)
            xp = xp * _t_mp.exp(-_t_mp.mpf(float(lam_p)) * _t_mp.mpf(float(t_cool)))
            return [float(N * _t_mp.exp(-c * T)), float(xp)]

    _t_lco = np.log(2.0) / (5.2714 * 3.15576e7)
    for _t_flux in (1e9, 1e13, 1e15, 1e17):
        x = activation_inventory(np.array([0.0, _t_lco]), np.zeros((2, 2)), np.array([37.18, 2.0]), np.array([1, -1]),
                                 np.array([4e22, 0.0]), [(3e7, _t_flux), (1e7, 0.0)])
        _t_check(x, _t_two_member(4e22, 37.18, _t_lco, 2.0, _t_flux, 3e7, 1e7), 4e22)
    # product removed at the target's rate: lam_p = 1e-9 per s with sigma_t = 100 b at 1e13 (c = 1e-9 per s up to
    # rounding), where the textbook two-member formula divides by zero
    x = activation_inventory(np.array([0.0, 1e-9]), np.zeros((2, 2)), np.array([100.0, 0.0]), np.array([1, -1]),
                             np.array([1e20, 0.0]), [(2e9, 1e13)])
    _t_check(x, _t_two_member(1e20, 100.0, 1e-9, 0.0, 1e13, 2e9, 0.0), 1e20)
    # saturation: 197Au (98.65 b) at 100 n / (cm^2 s) for 1e8 s, so that lam t = 298 for 198Au (2.6941 d, 25100 b)
    # while sigma phi t = 1e-12: the 198Au activity equals the production rate n0 sigma 1e-24 phi to about 1e-12
    _t_lau = np.log(2.0) / (2.6941 * 86400.0)
    x = activation_inventory(np.array([0.0, _t_lau]), np.zeros((2, 2)), np.array([98.65, 25100.0]), np.array([1, -1]),
                             np.array([1e18, 0.0]), [(1e8, 100.0)])
    assert _t_rel(_t_lau * float(np.asarray(x, dtype=float).reshape(2)[1]), 1e18 * 98.65e-24 * 100.0) < 1e-10, x


_crown_case()

# --- test case 7 ---
# decay competing with capture, with decay branching: nuclide A (stored at index 2, lam = 1e-4 per s,
# sigma = 5000 b) decays to B (index 0, 30 %) and C (index 3, 70 %) and captures into D (index 1); B, C, D
# are stable. At flux phi A is removed at r = lam + c (c = sigma 1e-24 phi) and its losses are shared in the
# ratios 0.3 lam : 0.7 lam : c, so the daughters hold n0 (1 - exp(-r t)) times these shares over r; in the
# cooling that follows A decays at lam alone. Closed forms in 50 digits, at 1e14 and 1e16 n / (cm^2 s), and
# with the column of A summing to 1 + 1e-12 (divided by its sum); no atom leaves, so the total is conserved
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

    lam = np.array([0.0, 0.0, 1e-4, 0.0])
    sig = np.array([0.0, 0.0, 5000.0, 0.0])
    cap = np.array([-1, -1, 1, -1])
    n0 = np.array([0.0, 0.0, 3e15, 0.0])
    for _t_fb, _t_fc in ((0.3, 0.7), (0.3, 0.7 + 1e-12)):
        B = np.zeros((4, 4))
        B[0, 2], B[3, 2] = _t_fb, _t_fc
        for _t_flux in (1e14, 1e16):
            x = activation_inventory(lam, B, sig, cap, n0, [(2e4, _t_flux), (5e3, 0.0)])
            with _t_mp.workdps(50):
                L, N = _t_mp.mpf("1e-4"), _t_mp.mpf("3e15")
                s = _t_mp.mpf(_t_fb) + _t_mp.mpf(_t_fc)
                fb, fc = _t_mp.mpf(_t_fb) / max(s, 1), _t_mp.mpf(_t_fc) / max(s, 1)
                c = _t_mp.mpf(5000) * _t_mp.mpf("1e-24") * _t_mp.mpf(_t_flux)
                r = L + c
                a1 = N * _t_mp.exp(-r * _t_mp.mpf(2e4))
                gone = N - a1
                a2 = a1 * _t_mp.exp(-L * _t_mp.mpf(5e3))
                dec2 = a1 - a2
                target = [float(fb * (L / r * gone + dec2)), float(c / r * gone), float(a2), float(fc * (L / r * gone + dec2))]
            _t_check(x, target, 3e15)
            # no atom leaves the network: the four nuclides hold the initial 3e15 atoms
            assert _t_rel(float(np.sum(x)), 3e15) < 1e-10, float(np.sum(x))


_crown_case()

# --- test case 8 ---
# the history is applied in its order: a stable target (sigma = 50 b) and its product (lam = 2e-6 per s, no
# capture), irradiated at 1e14 for 4e5 s and cooled for 3e5 s, against the reverse order (cooling first,
# nothing to decay yet); the product then holds the two-member closed form times exp(-lam 3e5), or the
# closed form itself; and a history of two irradiations at different fluxes with a cooling between them,
# propagated period by period with the closed-form two-member solution in 50 digits
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

    def _t_periods(n0, history, sig=50.0, lam_p=2e-6):
        # target T (stable, capture into P) and product P (decay out of the network): per period with
        # c = sig 1e-24 flux, T -> T exp(-c t), P -> P exp(-lam t) + T c (exp(-c t) - exp(-lam t)) / (lam - c)
        with _t_mp.workdps(50):
            T, P = _t_mp.mpf(float(n0[0])), _t_mp.mpf(float(n0[1]))
            L = _t_mp.mpf(float(lam_p))
            for dur, flux in history:
                c = _t_mp.mpf(float(sig)) * _t_mp.mpf("1e-24") * _t_mp.mpf(float(flux))
                d = _t_mp.mpf(float(dur))
                P = P * _t_mp.exp(-L * d) + T * c * (_t_mp.exp(-c * d) - _t_mp.exp(-L * d)) / (L - c)
                T = T * _t_mp.exp(-c * d)
            return [float(T), float(P)]

    data = (np.array([0.0, 2e-6]), np.zeros((2, 2)), np.array([50.0, 0.0]), np.array([1, -1]))
    n0 = np.array([2e21, 0.0])
    h1 = [(4e5, 1e14), (3e5, 0.0)]
    h2 = [(3e5, 0.0), (4e5, 1e14)]
    x1 = activation_inventory(*data, n0, h1)
    x2 = activation_inventory(*data, n0, h2)
    _t_check(x1, _t_periods(n0, h1), 2e21)
    _t_check(x2, _t_periods(n0, h2), 2e21)
    # the cooling after the irradiation multiplies the product by exp(-lam 3e5) (two computed outputs: 2e-10)
    assert _t_rel(float(np.asarray(x1, dtype=float).reshape(2)[1]) / float(np.asarray(x2, dtype=float).reshape(2)[1]),
                  float(_t_mp.exp(-_t_mp.mpf("0.6")))) < 2e-10, (x1, x2)
    h3 = ((1e5, 3e15), (2e5, 0.0), (5e4, 1e13), (1e4, 0.0))
    _t_check(activation_inventory(*data, n0, h3), _t_periods(n0, h3), 2e21)


_crown_case()
