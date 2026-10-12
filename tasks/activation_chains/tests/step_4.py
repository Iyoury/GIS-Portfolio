# --- test case 0 ---
# Independent targets: the activity from mpmath's matrix exponential of every period at 300 digits (the
# flux is the input of the round trip), the end tolerance located with 60-digit logarithms, and closed forms
# for the domain rule.
# gold monitor, activity of 198Au after 5 days of irradiation and 1 day of cooling,
# for fluxes from 1e8 to 5e13 n / (cm^2 s) (the activity increases up to about 1.6e15)
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
    _T_DATA = (np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)]), np.zeros((3, 3)),
               np.array([98.65, 25100.0, 30.0]), np.array([1, 2, -1]), np.array([1e18, 0.0, 0.0]))

    def _t_activity(k, flux, t_irr, t_cool, lam, branching, sigma, capture_to, n0):
        x = _t_activation(lam, branching, sigma, capture_to, n0, [(t_irr, flux), (t_cool, 0.0)])
        return lam[k] * x[k]

    def _t_roundtrip(k, flux, t_irr, t_cool, lo, hi, data):
        act = _t_activity(k, flux, t_irr, t_cool, *data)
        got = monitor_flux(*data, t_irr, t_cool, k, act, lo, hi)
        assert _t_rel(got, flux) < 1e-8, (k, flux, got)

    # output contract: the flux is a Python float
    got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_activity(1, 1e10, 5 * _T_DAY, _T_DAY, *_T_DATA), 1e6, 1e15)
    assert isinstance(got, float), type(got)
    for _t_flux in (1e8, 3.3e12, 5e13):
        _t_roundtrip(1, _t_flux, 5 * _T_DAY, _T_DAY, 1e6, 1e15, _T_DATA)


_crown_case()

# --- test case 1 ---
# the two-capture product 199Au at low flux: its activity grows like flux**2 and is
# about 2e-11 Bq at flux 1e4; a two-capture product whose activity is near the bottom of the double range;
# and a tiny inventory whose product stays above the floor of 1e-300 atoms of the domain
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
    _T_DATA = (np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)]), np.zeros((3, 3)),
               np.array([98.65, 25100.0, 30.0]), np.array([1, 2, -1]), np.array([1e18, 0.0, 0.0]))

    def _t_activity(k, flux, t_irr, t_cool, lam, branching, sigma, capture_to, n0):
        x = _t_activation(lam, branching, sigma, capture_to, n0, [(t_irr, flux), (t_cool, 0.0)])
        return lam[k] * x[k]

    def _t_roundtrip(k, flux, t_irr, t_cool, lo, hi, data):
        act = _t_activity(k, flux, t_irr, t_cool, *data)
        got = monitor_flux(*data, t_irr, t_cool, k, act, lo, hi)
        assert _t_rel(got, flux) < 1e-8, (k, flux, got)

    for _t_flux in (1e4, 7.7e9):
        _t_roundtrip(2, _t_flux, 5 * _T_DAY, _T_DAY, 1e-2, 1e15, _T_DATA)
    # a far smaller activity: a two-capture product with lam[k] = 1e-295 per s, about 5e-302 Bq at the root;
    # at flux_lo its activity (about 5e-330 Bq) is below the smallest positive double, while it holds 5e-35
    # of the target atoms (inside the domain)
    _t_tiny = (np.array([0.0, 0.0, 1e-295]), np.zeros((3, 3)), np.array([1e7, 1e7, 0.0]), np.array([1, 2, -1]),
               np.array([1.0, 0.0, 0.0]))
    _t_roundtrip(2, 1e12, 100.0, 10.0, 1e-2, 1e15, _t_tiny)
    # 1e-70 target atoms irradiated for 1e-200 s: at flux_lo = 1 the product holds about 1e-294 atoms (above
    # max(1e-250 sum(n0), 1e-300) = 1e-300: inside the domain), about 1e-295 Bq at the root
    _t_small = (np.array([0.0, 1e-3]), np.zeros((2, 2)), np.array([1.0, 0.0]), np.array([1, -1]), np.array([1e-70, 0.0]))
    _t_roundtrip(1, 100.0, 1e-200, 10.0, 1.0, 1e4, _t_small)


_crown_case()

# --- test case 2 ---
# burnup of 198Au (25100 b) bends the activity over (maximum near 1.6e15): at 1e14 the
# slope d ln A / d ln flux is about 0.65 instead of 1
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
    _T_DATA = (np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)]), np.zeros((3, 3)),
               np.array([98.65, 25100.0, 30.0]), np.array([1, 2, -1]), np.array([1e18, 0.0, 0.0]))

    def _t_activity(k, flux, t_irr, t_cool, lam, branching, sigma, capture_to, n0):
        x = _t_activation(lam, branching, sigma, capture_to, n0, [(t_irr, flux), (t_cool, 0.0)])
        return lam[k] * x[k]

    def _t_roundtrip(k, flux, t_irr, t_cool, lo, hi, data):
        act = _t_activity(k, flux, t_irr, t_cool, *data)
        got = monitor_flux(*data, t_irr, t_cool, k, act, lo, hi)
        assert _t_rel(got, flux) < 1e-8, (k, flux, got)

    _t_roundtrip(1, 1e14, 5 * _T_DAY, _T_DAY, 1e13, 1.5e15, _T_DATA)
    # near the edge of the accuracy statement: at 1.547e14 the slope d ln A / d ln flux is about 0.52
    _t_roundtrip(1, 1.547e14, 5 * _T_DAY, _T_DAY, 1e13, 1.5e15, _T_DATA)
    # at the edge itself: at 1.645e14 the slope is just above 0.5 (0.5 is reached at about 1.6454e14)
    _t_h = 1e-5
    _t_slope = (np.log(_t_activity(1, 1.645e14 * np.exp(_t_h), 5 * _T_DAY, _T_DAY, *_T_DATA))
                - np.log(_t_activity(1, 1.645e14 * np.exp(-_t_h), 5 * _T_DAY, _T_DAY, *_T_DATA))) / (2 * _t_h)
    assert 0.5 < _t_slope < 0.5005, _t_slope
    _t_roundtrip(1, 1.645e14, 5 * _T_DAY, _T_DAY, 1e13, 1.5e15, _T_DATA)


_crown_case()

# --- test case 3 ---
# the ends of the domain: no cooling (t_cool = 0); t_irr = 1e9 s and t_cool = 1e9 s on
# a cobalt monitor (60Co, 5.27 y); flux_hi = 1e18 on a monitor whose product (1e-3 per s, no capture)
# keeps rising up to that flux; activities equal to A(flux_lo) and A(flux_hi)
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
    _T_DATA = (np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)]), np.zeros((3, 3)),
               np.array([98.65, 25100.0, 30.0]), np.array([1, 2, -1]), np.array([1e18, 0.0, 0.0]))

    def _t_activity(k, flux, t_irr, t_cool, lam, branching, sigma, capture_to, n0):
        x = _t_activation(lam, branching, sigma, capture_to, n0, [(t_irr, flux), (t_cool, 0.0)])
        return lam[k] * x[k]

    def _t_roundtrip(k, flux, t_irr, t_cool, lo, hi, data):
        act = _t_activity(k, flux, t_irr, t_cool, *data)
        got = monitor_flux(*data, t_irr, t_cool, k, act, lo, hi)
        assert _t_rel(got, flux) < 1e-8, (k, flux, got)

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
    # back those ends: the given bound itself, to the stated relative 1e-11
    for _t_end, _t_in in ((1e8, 1.0 + 5e-10), (1e13, 1.0 - 5e-10)):
        for _t_f in (1.0, _t_in, 2.0 - _t_in):
            act = _t_f * _t_activity(1, _t_end, 5 * _T_DAY, _T_DAY, *_T_DATA)
            got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, act, 1e8, 1e13)
            assert _t_rel(got, _t_end) < 1e-11, (got, _t_end, _t_f)
        # the tolerance 1e-9 holds up to the accuracy of the end values, which activation_inventory gives to a
        # relative 1e-10 (so 1e-10 in ln): at both ends and on both sides, the representable activity closest to
        # |ln(activity / A(end))| = 1e-9 - 1e-10 from inside gives the end; outside the interval the
        # representable activity closest to |ln| = 1e-9 + 1e-10 from beyond raises ValueError, and inside the
        # interval it gives a flux within 1e-8 of the end (the exact root is about 1.1e-9 away, the slope
        # d ln A / d ln flux being 1.00 at 1e8 and 0.96 at 1e13). A(end) is the extended-precision target; the
        # logarithms are taken in mpmath.
        _t_a = _t_activity(1, _t_end, 5 * _T_DAY, _T_DAY, *_T_DATA)
        for _t_sgn in (1, -1):
            _t_outside = (_t_sgn < 0) == (_t_end == 1e8)
            with _t_mp.workdps(60):
                _t_A = _t_mp.mpf(_t_a)
                _t_lr = lambda x: abs(_t_mp.log(_t_mp.mpf(x) / _t_A))
                _t_lo_b, _t_hi_b = _t_mp.mpf("1e-9") - _t_mp.mpf("1e-10"), _t_mp.mpf("1e-9") + _t_mp.mpf("1e-10")
                _t_in = float(_t_A * _t_mp.exp(_t_sgn * _t_lo_b))
                while _t_lr(_t_in) > _t_lo_b:
                    _t_in = float(np.nextafter(_t_in, _t_a))
                while _t_lr(np.nextafter(_t_in, _t_sgn * 1e308)) <= _t_lo_b:
                    _t_in = float(np.nextafter(_t_in, _t_sgn * 1e308))
                _t_beyond = float(_t_A * _t_mp.exp(_t_sgn * _t_hi_b))
                while _t_lr(_t_beyond) <= _t_hi_b:
                    _t_beyond = float(np.nextafter(_t_beyond, _t_sgn * 1e308))
                while _t_lr(np.nextafter(_t_beyond, _t_a)) > _t_hi_b:
                    _t_beyond = float(np.nextafter(_t_beyond, _t_a))
            got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_in, 1e8, 1e13)
            assert _t_rel(got, _t_end) < 1e-11, (got, _t_end, _t_sgn)
            if _t_outside:
                _t_raised = False
                try:
                    monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_beyond, 1e8, 1e13)
                except ValueError:
                    _t_raised = True
                assert _t_raised, ("activity just beyond 1e-9 + 1e-10 outside the end at %g must raise ValueError" % _t_end)
            else:
                got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_beyond, 1e8, 1e13)
                assert _t_rel(got, _t_end) < 1e-8, (got, _t_end, _t_sgn)
        # and 2e-9 outside
        _t_out = np.exp(-2e-9) if _t_end == 1e8 else np.exp(2e-9)
        _t_raised = False
        try:
            monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, _t_out * _t_a, 1e8, 1e13)
        except ValueError:
            _t_raised = True
        assert _t_raised, ("activity 2e-9 outside the end at %g must raise ValueError" % _t_end)


_crown_case()

# --- test case 4 ---
# each stated error rule once: t_irr = 0, k of a stable nuclide, k a float or a bool, activity 0, flux_lo
# equal to flux_hi or below 1e-2 raise ValueError; flux_lo is checked first when both ends are within the
# tolerance; and a measured nuclide below the domain floor at flux_lo (1e-250 of the initial atoms, also
# with a tiny lam[k], or 1e-300 atoms) or nuclide data rejected by step 3 raise ValueError
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
    _T_DATA = (np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)]), np.zeros((3, 3)),
               np.array([98.65, 25100.0, 30.0]), np.array([1, 2, -1]), np.array([1e18, 0.0, 0.0]))

    def _t_activity(k, flux, t_irr, t_cool, lam, branching, sigma, capture_to, n0):
        x = _t_activation(lam, branching, sigma, capture_to, n0, [(t_irr, flux), (t_cool, 0.0)])
        return lam[k] * x[k]

    _t_act = _t_activity(1, 1e10, 5 * _T_DAY, _T_DAY, *_T_DATA)
    for _t_args in ((0.0, _T_DAY, 1, _t_act, 1e8, 1e13), (5 * _T_DAY, _T_DAY, 0, _t_act, 1e8, 1e13),
                    (5 * _T_DAY, _T_DAY, 1.0, _t_act, 1e8, 1e13), (5 * _T_DAY, _T_DAY, True, _t_act, 1e8, 1e13),
                    (5 * _T_DAY, _T_DAY, 1, 0.0, 1e8, 1e13), (5 * _T_DAY, _T_DAY, 1, _t_act, 1e10, 1e10),
                    (5 * _T_DAY, _T_DAY, 1, _t_act, 9.99e-3, 1e13)):
        try:
            monitor_flux(*_T_DATA, *_t_args)
        except ValueError:
            pass
        else:
            raise AssertionError("monitor_flux must raise ValueError for %r" % (_t_args,))
    # precedence of the ends: with bounds 1e-10 apart both end activities lie within the tolerance of
    # A(flux_lo); flux_lo is checked first and returned (to the stated relative 1e-11, which separates it from
    # flux_hi)
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
    # the same with a tiny lam[k] = 1e-75, for which 1e-250 * lam[k] * sum(n0) underflows to 0: two successive
    # captures give x[2] = (sigma phi t)**2 / 2 of the target atoms, about 5e-267 at flux_lo = 1e-2 (below
    # 1e-250) and 5e-227 at flux_hi = 1e18; the activity is the one at flux_hi
    _t_raised = False
    try:
        monitor_flux(np.array([0.0, 0.0, 1e-75]), np.zeros((3, 3)), np.array([1e7, 1e7, 0.0]), np.array([1, 2, -1]),
                     np.array([1.0, 0.0, 0.0]), 1e-114, 0.0, 2, 1e-75 * 0.5 * (1e-17 * 1e18 * 1e-114) ** 2, 1e-2, 1e18)
    except ValueError:
        _t_raised = True
    assert _t_raised, "A(flux_lo) below 1e-250 lam[k] sum(n0) with a tiny lam[k] must raise ValueError"
    # outside the domain through the floor of 1e-300 atoms: 1e-70 target atoms irradiated for 1e-225 s leave about
    # 1e-319 atoms of product at flux_lo = 1, 1e-249 of the initial atoms but fewer than 1e-300 atoms, so
    # A(flux_lo) < lam[k] max(1e-250 sum(n0), 1e-300); the activity 1e-320 Bq lies between A(flux_lo) (about 1e-322)
    # and A(flux_hi) (about 1e-318)
    _t_raised = False
    try:
        monitor_flux(np.array([0.0, 1e-3]), np.zeros((2, 2)), np.array([1.0, 0.0]), np.array([1, -1]),
                     np.array([1e-70, 0.0]), 1e-225, 0.0, 1, 1e-320, 1.0, 1e4)
    except ValueError:
        _t_raised = True
    assert _t_raised, "fewer than 1e-300 atoms of the measured nuclide at flux_lo must raise ValueError"
    # nuclide data rejected by step 3: a positive cross section below 1e-6 b (its rate would underflow)
    _t_raised = False
    try:
        monitor_flux(np.array([0.0, 1e-3]), np.zeros((2, 2)), np.array([1e-300, 0.0]), np.array([1, -1]),
                     np.array([1e100, 0.0]), 1e3, 0.0, 1, 1.0, 1e6, 1e18)
    except ValueError:
        _t_raised = True
    assert _t_raised, "a cross section of 1e-300 b must raise ValueError"


_crown_case()

# --- test case 5 ---
# a cobalt monitor in closed form: 59Co (37.18 b, 4e22 atoms) captures into 60Co (5.2714 y), which is also
# burnt up by its own capture (2.0 b); after 3e7 s of irradiation and 1e7 s of cooling the 60Co activity is
# lam n0 c (exp(-c t) - exp(-r t)) / (r - c) exp(-lam t_cool), with c = 37.18e-24 phi and
# r = lam + 2.0e-24 phi (50-digit arithmetic). A increases up to about 2.7e15; the flux is recovered at 1e9
# (d ln A / d ln phi = 1.00), 1e13 (0.99) and 3e14 (0.83), the interval reaching 2e15 (slope 0.15)
import numpy as np


def _crown_case():
    import mpmath as _t_mp

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    _t_lam = np.log(2.0) / (5.2714 * 3.15576e7)
    _t_data = (np.array([0.0, _t_lam]), np.zeros((2, 2)), np.array([37.18, 2.0]), np.array([1, -1]), np.array([4e22, 0.0]))

    def _t_activity(flux, t_irr=3e7, t_cool=1e7):
        with _t_mp.workdps(50):
            L = _t_mp.mpf(_t_lam)
            c = _t_mp.mpf("37.18") * _t_mp.mpf("1e-24") * _t_mp.mpf(flux)
            r = L + _t_mp.mpf(2) * _t_mp.mpf("1e-24") * _t_mp.mpf(flux)
            T = _t_mp.mpf(t_irr)
            return float(L * _t_mp.mpf("4e22") * c * (_t_mp.exp(-c * T) - _t_mp.exp(-r * T)) / (r - c)
                         * _t_mp.exp(-L * _t_mp.mpf(t_cool)))

    got = monitor_flux(*_t_data, 3e7, 1e7, 1, _t_activity(1e9), 1e8, 2e15)
    assert _t_rel(got, 1e9) < 1e-8, got
    got = monitor_flux(*_t_data, 3e7, 1e7, 1, _t_activity(1e13), 1e8, 2e15)
    assert _t_rel(got, 1e13) < 1e-8, got
    got = monitor_flux(*_t_data, 3e7, 1e7, 1, _t_activity(3e14), 1e8, 2e15)
    assert _t_rel(got, 3e14) < 1e-8, got


_crown_case()

# --- test case 6 ---
# power laws at low flux: on the gold monitor at 10 n / (cm^2 s) the burnup terms are below 1e-14, so the
# 198Au activity (one capture) is proportional to the flux and the 199Au activity (two captures) to its
# square: 10 times the 198Au activity gives 10 times the flux, 4 times the 199Au activity twice the flux.
# The activities at 10 come from mpmath's matrix exponential of each period at 300 digits
import numpy as np


def _crown_case():
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

    # gold flux monitor: 197Au (stable, 98.65 b) -> 198Au (2.6941 d, 25100 b) -> 199Au (3.139 d, 30 b)
    _T_DAY = 86400.0
    _T_DATA = (np.array([0.0, np.log(2.0) / (2.6941 * _T_DAY), np.log(2.0) / (3.139 * _T_DAY)]), np.zeros((3, 3)),
               np.array([98.65, 25100.0, 30.0]), np.array([1, 2, -1]), np.array([1e18, 0.0, 0.0]))
    x = _t_activation(*_T_DATA, [(5 * _T_DAY, 10.0), (_T_DAY, 0.0)])
    a1, a2 = _T_DATA[0][1] * x[1], _T_DATA[0][2] * x[2]
    got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, a1, 1e-2, 1e6)
    assert _t_rel(got, 10.0) < 1e-8, got
    got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 1, 10.0 * a1, 1e-2, 1e6)
    assert _t_rel(got, 100.0) < 1e-8, got
    got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 2, a2, 1e-2, 1e6)
    assert _t_rel(got, 10.0) < 1e-8, got
    got = monitor_flux(*_T_DATA, 5 * _T_DAY, _T_DAY, 2, 4.0 * a2, 1e-2, 1e6)
    assert _t_rel(got, 20.0) < 1e-8, got


_crown_case()

# --- test case 7 ---
# 30 nuclides and the largest inventory, sum(n0) = 1e100: a stable target followed by 29 products, each
# made by a capture on the previous one; the activity of the three-capture product (index 3) after 1e5 s
# of irradiation and 2e4 s of cooling grows like phi**3 at low flux (slope 2.6 at 3e16); target from the
# Bateman sum of the capture chain at 600 digits (distinct removal rates)
import numpy as np


def _crown_case():
    import mpmath as _t_mp

    def _t_rel(a, b):
        return abs(a - b) / abs(b)

    lam = np.r_[0.0, np.geomspace(1e-6, 1e-3, 29)]
    sig = np.r_[np.geomspace(1e2, 1e4, 29), 0.0]
    cap = np.r_[np.arange(1, 30), -1]
    n0 = np.r_[1e100, np.zeros(29)]

    def _t_capture_activity(k, flux, t_irr, t_cool, dps=600):
        # nuclide i captures into i + 1 (rate c_i = sig_i 1e-24 flux) and decays out of the network: during the
        # irradiation a linear chain with transfer rates c_i and removal rates r_i = lam_i + c_i,
        # x_k = n0 prod_{i<k} c_i sum_{i<=k} exp(-r_i t) / prod_{l<=k, l!=i} (r_l - r_i); then decay alone
        with _t_mp.workdps(dps):
            c = [_t_mp.mpf(float(s)) * _t_mp.mpf("1e-24") * _t_mp.mpf(float(flux)) for s in sig[:k + 1]]
            r = [_t_mp.mpf(float(l)) + ci for l, ci in zip(lam[:k + 1], c)]
            T = _t_mp.mpf(float(t_irr))
            tot = _t_mp.mpf(0)
            for i in range(k + 1):
                tot += _t_mp.exp(-r[i] * T) / _t_mp.fprod([r[l] - r[i] for l in range(k + 1) if l != i])
            xk = _t_mp.mpf("1e100") * _t_mp.fprod(c[:k]) * tot
            L = _t_mp.mpf(float(lam[k]))
            return float(L * xk * _t_mp.exp(-L * _t_mp.mpf(float(t_cool))))

    got = monitor_flux(lam, np.zeros((30, 30)), sig, cap, n0, 1e5, 2e4, 3, _t_capture_activity(3, 1e10, 1e5, 2e4), 1e8, 1e17)
    assert _t_rel(got, 1e10) < 1e-8, got
    got = monitor_flux(lam, np.zeros((30, 30)), sig, cap, n0, 1e5, 2e4, 3, _t_capture_activity(3, 3e16, 1e5, 2e4), 1e8, 1e17)
    assert _t_rel(got, 3e16) < 1e-8, got


_crown_case()
