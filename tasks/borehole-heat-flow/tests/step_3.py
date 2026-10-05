"""Tests for step 3: paleoclimate_perturbation."""
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

YEAR = 365.25 * 86400.0


def _f():
    return _resolve("paleoclimate_perturbation", ("step_3", "steps.step_3", "solution.step_3", "solution"))


@pytest.mark.parametrize("z", [0.0, 50.0, 300.0, 1000.0, 3000.0])
def test_single_step_matches_erfc(z):
    # surface warmed by 1.5 K 10 kyr ago and stayed there
    kappa = 1.2e-6
    exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * YEAR)))
    got = _f()(z, [1.0e4], [1.5], kappa)
    assert type(got) is float
    assert abs(got - exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|


def test_equal_steps_collapse_to_one():
    z = np.array([100.0, 700.0, 1500.0])
    a = np.asarray(_f()(z, [2e3, 3e4, 8e4], [-2.0, -2.0, -2.0], 1e-6))
    b = np.asarray(_f()(z, [8e4], [-2.0], 1e-6))
    np.testing.assert_allclose(a, b, atol=4e-6)


def test_past_pulse():
    # cold interval 10-100 ka only: difference of two erfc responses
    kappa, z = 1.2e-6, 900.0
    e = lambda t: math.erfc(z / (2 * math.sqrt(kappa * t * YEAR)))
    got = _f()(z, [1.0e4, 1.0e5], [0.0, -6.0], kappa)
    assert abs(got - (-6.0) * (e(1.0e5) - e(1.0e4))) < 6e-6


def test_multistep_history():
    # checked against Duhamel convolution with the half-space impulse response
    got = np.asarray(_f()([80.0, 400.0, 1200.0], [1e3, 1.1e4, 9e4, 1.3e5],
                          [0.8, 0.0, -1.0, 0.3], 1.15e-6))
    np.testing.assert_allclose(got, [0.56809482, -0.10491562, -0.44221609], atol=1e-6)


def test_linearity_and_decay():
    z = np.array([10.0, 200.0, 20000.0])
    a = np.asarray(_f()(z, [5e3, 5e4], [0.4, -1.0], 1e-6))
    b = np.asarray(_f()(z, [5e3, 5e4], [0.8, -2.0], 1e-6))
    np.testing.assert_allclose(b, 2 * a, atol=4e-6)
    assert abs(a[2]) < 1e-6


def test_two_dimensional_depth_array():
    # the output keeps the full shape of z; z = 0 gives the present surface departure dT[0]
    hist_t, hist_d, kappa = [2e3, 2.5e4, 1.1e5], [0.6, -2.5, 0.4], 1.1e-6
    z = np.array([[0.0, 150.0, 400.0], [900.0, 1600.0, 3000.0]])
    got = _f()(z, hist_t, hist_d, kappa)
    assert isinstance(got, np.ndarray) and got.shape == (2, 3)
    flat = np.array([_f()(float(v), hist_t, hist_d, kappa) for v in z.ravel()])
    np.testing.assert_allclose(got.ravel(), flat, atol=2.5e-6)
    assert abs(got[0, 0] - 0.6) < 2.5e-6
    YEAR_S = 365.25 * 86400.0
    edges = [0.0] + [t * YEAR_S for t in hist_t]
    exp = 0.0
    for i, d in enumerate(hist_d):
        e_old = math.erfc(900.0 / (2 * math.sqrt(kappa * edges[i + 1])))
        e_new = 0.0 if i == 0 else math.erfc(900.0 / (2 * math.sqrt(kappa * edges[i])))
        exp += d * (e_old - e_new)
    assert abs(got[1, 0] - exp) < 2.5e-6


@pytest.mark.parametrize("case", ["t0", "order", "t_nan", "dT_nan", "kappa", "kappa_nan", "len", "two_d", "zneg", "znan"])
def test_invalid_raises(case):
    z, t, d, k = 100.0, [1e3, 1e4], [0.5, -1.0], 1e-6
    if case == "t0":
        t = [0.0, 1e4]
    elif case == "order":
        t = [1e4, 1e3]
    elif case == "t_nan":
        t = [1e3, float("nan")]
    elif case == "dT_nan":
        d = [0.5, float("nan")]
    elif case == "kappa":
        k = 0.0
    elif case == "kappa_nan":
        k = float("nan")
    elif case == "len":
        d = [0.5]
    elif case == "two_d":
        t, d = [[1e3, 1e4]], [[0.5, -1.0]]
    elif case == "zneg":
        z = -1.0
    else:
        z = float("nan")
    with pytest.raises(ValueError):
        _f()(z, t, d, k)


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
