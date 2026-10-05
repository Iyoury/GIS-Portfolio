"""Tests for step 1: true_vertical_depth."""
import importlib
import math

import numpy as np
# pytest is not installed in the grading image: a small stand-in provides
# pytest.mark.parametrize and pytest.raises, and the tests run on import.
import contextlib as _contextlib

class _Mark:
    @staticmethod
    def parametrize(names, values, ids=None):
        def deco(fn):
            fn._crown_params = (names, list(values))
            return fn
        return deco

class _PytestShim:
    mark = _Mark()

    @staticmethod
    @_contextlib.contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"{getattr(exc, '__name__', exc)} was not raised")

pytest = _PytestShim()
_PYTEST_SHIM = True


def _resolve(name, modules):
    """Find the function under test: same namespace first, then modules."""
    g = globals()
    if name in g and callable(g[name]):
        return g[name]
    for m in modules:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ImportError(f"function {name!r} not found")


def _f():
    return _resolve("true_vertical_depth", ("step_1", "steps.step_1", "solution.step_1", "solution"))


def test_vertical_hole():
    out = np.asarray(_f()([0, 100, 250, 600], [0, 0, 0, 0], [0, 45, 90, 10], [0, 33.3, 600]))
    np.testing.assert_allclose(out, [0, 33.3, 600], atol=1e-6)


@pytest.mark.parametrize("inc", [15.0, 35.0, 60.0])
def test_straight_inclined_hole(inc):
    md = [0, 150, 300, 450]
    got = _f()(md, [inc] * 4, [200.0] * 4, 321.7)
    assert abs(got - 321.7 * math.cos(math.radians(inc))) < 1e-6


def test_constant_build_arc_is_exact_between_stations():
    # inclination builds uniformly from 0 to 90 deg over 900 m: a circular arc
    # of radius R = 900 / (pi/2); TVD(m) = R sin(m / R) at ANY measured depth
    L = 900.0
    Rad = L / (math.pi / 2)
    md = np.array([0.0, 300.0, 600.0, 900.0])
    inc = md / L * 90.0
    q = np.array([37.5, 137.5, 300.0, 451.2, 777.7, 900.0])
    got = np.asarray(_f()(md, inc, [45.0] * 4, q))
    np.testing.assert_allclose(got, Rad * np.sin(q / Rad), atol=1e-6)


def test_three_dimensional_segment():
    # build and turn in one segment; values checked by numerical integration
    # of the unit tangent along the great-circle arc
    got = np.asarray(_f()([0, 400], [20, 50], [0, 90], [137.0, 250.0, 400.0]))
    np.testing.assert_allclose(got, [128.63223148, 228.74044021, 341.01702731], atol=1e-6)


def test_azimuth_turn_at_constant_inclination():
    got = np.asarray(_f()([0, 100, 200, 300], [40] * 4, [0, 60, 120, 180], [50.0, 175.0, 300.0]))
    np.testing.assert_allclose(got, [39.73029462, 139.32487515, 238.38176775], atol=1e-6)


def test_scalar_and_zero_query():
    got = _f()([0, 100], [30, 30], [10, 10], 0.0)
    assert type(got) is float and abs(got) < 1e-6


def test_array_query_values_match_scalar_queries():
    q = np.array([[10.0, 20.0], [30.0, 40.0]])
    md, inc, azi = [0, 100], [10, 12], [0, 0]
    out = _f()(md, inc, azi, q)
    assert isinstance(out, np.ndarray) and out.shape == q.shape, (type(out), np.shape(out))
    ref = np.array([[_f()(md, inc, azi, float(v)) for v in row] for row in q])
    np.testing.assert_allclose(out, ref, atol=1e-9)
    # constant azimuth, inclination linear in MD: TVD = (sin(I(m)) - sin(I1)) / k
    k = math.radians(2.0) / 100.0
    np.testing.assert_allclose(out, (np.sin(math.radians(10.0) + k * q) - math.sin(math.radians(10.0))) / k,
                               atol=1e-6)


def _mp_build_tvd(inc1_deg, dinc_rad, seg_len, m):
    """Exact TVD on a constant-azimuth segment whose inclination changes by dinc_rad over seg_len
    (inclination linear in MD): (sin(I1 + k m) - sin(I1)) / k written without cancellation as
    m cos(I1 + k m / 2) sinc(k m / 2)."""
    k = dinc_rad / seg_len
    h = 0.5 * k * m
    sinc = 1.0 - h * h / 6.0 if h < 1e-6 else math.sin(h) / h
    return m * math.cos(math.radians(inc1_deg) + h) * sinc


@pytest.mark.parametrize("dinc_rad", [8e-10, 3e-8, 2e-6])
def test_tiny_dogleg_on_a_long_segment(dinc_rad):
    # a dogleg of 8e-10 rad over 20 km already bends the path by ~4e-6 m of TVD:
    # treating a small non-zero dogleg as straight fails the 1e-6 m accuracy
    L, inc1 = 20000.0, 30.0
    inc2 = inc1 + math.degrees(dinc_rad)
    q = np.array([5000.0, 13000.0, 20000.0])
    got = np.asarray(_f()([0.0, L], [inc1, inc2], [75.0, 75.0], q))
    ref = np.array([_mp_build_tvd(inc1, dinc_rad, L, m) for m in q])
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)


def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
    from scipy.integrate import quad
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
        axis = np.cross(t1, t2)
        beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
        if beta == 0.0:
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
        total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                      a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
    return total


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_random_three_dimensional_surveys(seed):
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = _f()(md, inc, azi, q)
    assert isinstance(got, np.ndarray) and got.shape == q.shape
    ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)


def test_dogleg_just_outside_the_band_is_accepted():
    # dogleg = 180 deg - 2e-6 rad: outside the 1e-6 rad rejection band, so the arc is defined
    inc, azi = [10.0, 170.0 - math.degrees(2e-6)], [0.0, 180.0]
    v = _f()([0.0, 100.0], inc, azi, [40.0, 100.0])
    v = np.asarray(v, float)
    # the path is almost a half circle (radius ~ 100/pi m): it turns back up, so the TVD at the
    # end of the segment is smaller than at MD 40, and both stay within [0, MD]
    assert np.all(np.isfinite(v)) and np.all(v >= -1e-9) and np.all(v <= 100.0 + 1e-9) and v[1] < v[0]


@pytest.mark.parametrize("case", ["beyond", "negative", "nan", "start", "order", "lengths",
                                  "one_station", "two_d", "inc_neg", "inc_180", "dogleg_180",
                                  "dogleg_near_180", "md_nan", "inc_nan", "azi_nan"])
def test_invalid_raises(case):
    md, inc, azi, q = [0.0, 100.0, 200.0], [10.0, 12.0, 14.0], [0.0, 0.0, 0.0], 50.0
    if case == "beyond":
        q = 250.0
    elif case == "negative":
        q = -1.0
    elif case == "nan":
        q = float("nan")
    elif case == "start":
        md = [5.0, 100.0, 200.0]
    elif case == "order":
        md = [0.0, 200.0, 100.0]
    elif case == "lengths":
        inc = [10.0, 12.0]
    elif case == "one_station":
        md, inc, azi, q = [0.0], [10.0], [0.0], 0.0
    elif case == "two_d":
        md, inc, azi = [[0.0, 100.0, 200.0]], [[10.0, 12.0, 14.0]], [[0.0, 0.0, 0.0]]
    elif case == "inc_neg":
        inc = [-5.0, 12.0, 14.0]
    elif case == "inc_180":
        inc = [10.0, 180.0, 14.0]
    elif case == "dogleg_180":
        inc, azi = [10.0, 170.0, 20.0], [0.0, 180.0, 0.0]
    elif case == "dogleg_near_180":
        # opposite azimuths: dogleg = I1 + I2 = 180 deg - 5e-7 rad, inside the 1e-6 rad band
        inc, azi = [10.0, 170.0 - math.degrees(5e-7), 20.0], [0.0, 180.0, 0.0]
    elif case == "md_nan":
        md = [0.0, float("nan"), 200.0]
    elif case == "inc_nan":
        inc = [10.0, float("nan"), 14.0]
    else:
        azi = [0.0, float("nan"), 0.0]
    with pytest.raises(ValueError):
        _f()(md, inc, azi, q)


def _crown_run_all():
    failures = []
    for _name, _fn in list(globals().items()):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        params = getattr(_fn, "_crown_params", None)
        if params is None:
            calls = [((), _name)]
        else:
            names, values = params
            n_args = len([s for s in names.split(",") if s.strip()])
            calls = [((v,) if n_args == 1 else tuple(v), f"{_name}[{k}]") for k, v in enumerate(values)]
        for args, label in calls:
            try:
                _fn(*args)
            except Exception as err:  # noqa: BLE001
                failures.append(f"{label}: {type(err).__name__}: {err}")
    if failures:
        raise AssertionError("failed tests:\n" + "\n".join(failures))


if _PYTEST_SHIM:
    _crown_run_all()
