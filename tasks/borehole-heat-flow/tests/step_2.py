"""Tests for step 2: layer_integrals."""
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
    return _resolve("layer_integrals", ("step_2", "steps.step_2", "solution.step_2", "solution"))


def test_single_layer_closed_form():
    out = _f()([0.0], [2.5], 800.0)
    assert isinstance(out, tuple) and len(out) == 2
    R, S = out
    assert type(R) is float and type(S) is float, (type(R), type(S))
    assert abs(R - 800.0 / 2.5) < 1e-9 * 320
    assert abs(S - 800.0**2 / (2 * 2.5)) < 1e-9 * 128000


def test_three_layers_hand_computed():
    tops, k = [0.0, 100.0, 400.0], [2.0, 4.0, 3.0]
    z = np.array([0.0, 50.0, 100.0, 250.0, 400.0, 1000.0])
    R, S = (np.asarray(v) for v in _f()(tops, k, z))
    R_exp = [0, 25, 50, 50 + 150 / 4, 50 + 300 / 4, 125 + 600 / 3]
    S_exp = [0, 2500 / 4, 10000 / 4,
             2500 + (250**2 - 100**2) / 8,
             2500 + (400**2 - 100**2) / 8,
             2500 + 150000 / 8 + (1000**2 - 400**2) / 6]
    np.testing.assert_allclose(R, R_exp, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(S, S_exp, rtol=1e-9, atol=1e-12)


def test_resistance_is_harmonic_not_arithmetic():
    # equal thicknesses of k = 1 and k = 5: effective k = 1.667, not 3
    R, _ = _f()([0.0, 500.0], [1.0, 5.0], 1000.0)
    assert abs(1000.0 / R - 2 / (1 / 1.0 + 1 / 5.0)) < 1e-9


def test_array_depths_match_closed_form():
    z = np.linspace(0, 900, 12).reshape(3, 4)
    out = _f()([0.0, 300.0], [3.0, 2.0], z)
    assert isinstance(out, tuple) and len(out) == 2
    R, S = out
    assert isinstance(R, np.ndarray) and isinstance(S, np.ndarray) and R.shape == z.shape and S.shape == z.shape
    zz = np.minimum(z, 300.0)
    R_exp = zz / 3.0 + np.maximum(z - 300.0, 0.0) / 2.0
    S_exp = zz ** 2 / 6.0 + (np.maximum(z, 300.0) ** 2 - 300.0 ** 2) / 4.0
    np.testing.assert_allclose(np.asarray(R), R_exp, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(S), S_exp, rtol=1e-9, atol=1e-12)


def _quad_RS(tops, k, z):
    """R = integral of 1/k(z') and S = integral of z'/k(z') from 0 to z, by adaptive quadrature."""
    from scipy.integrate import quad
    tops = np.asarray(tops, float)
    def kz(x):
        return k[np.searchsorted(tops, x, side="right") - 1]
    brk = [t for t in tops[1:] if t < z]
    R = quad(lambda x: 1.0 / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
    S = quad(lambda x: x / kz(x), 0.0, z, points=brk or None, epsabs=0, epsrel=1e-13, limit=200)[0] if z > 0 else 0.0
    return R, S


@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15, 16])
def test_random_columns_against_quadrature(seed):
    # T(z) = T0 + q0 R - A S solves dT/dz = (q0 - A z) / k(z): R and S are the integrals of 1/k and z/k
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    tops = np.concatenate([[0.0], np.cumsum(rng.uniform(40.0, 600.0, n - 1))])
    k = rng.uniform(1.2, 6.0, n)
    z = np.sort(rng.uniform(0.0, tops[-1] + 800.0, 5))
    R, S = (np.asarray(v) for v in _f()(tops, k, z))
    ref = np.array([_quad_RS(tops, k, zi) for zi in z])
    np.testing.assert_allclose(R, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(S, ref[:, 1], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("case", ["k0", "knan", "tops", "tops_nan", "first", "neg", "znan", "len", "two_d"])
def test_invalid_raises(case):
    tops, k, z = [0.0, 100.0], [2.0, 3.0], 50.0
    if case == "k0":
        k = [2.0, 0.0]
    elif case == "tops":
        tops = [0.0, 0.0]
    elif case == "tops_nan":
        tops, k = [0.0, float("nan"), 500.0], [2.0, 3.0, 2.5]
    elif case == "first":
        tops = [10.0, 100.0]
    elif case == "knan":
        k = [2.0, float("nan")]
    elif case == "neg":
        z = -5.0
    elif case == "znan":
        z = float("nan")
    elif case == "len":
        k = [2.0]
    else:
        tops, k = [[0.0, 100.0]], [[2.0, 3.0]]
    with pytest.raises(ValueError):
        _f()(tops, k, z)


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
