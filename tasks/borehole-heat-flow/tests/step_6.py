"""Tests for step 6: layered_paleoclimate_perturbation."""
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

# Independent targets: the erfc half-space solution for a uniform column, the image series of Carslaw
# and Jaeger for a layer over a half-space (closed form, math.erfc only), and, for multilayer columns,
# values computed once with mpmath at 90 digits (Laplace transform from the transfer matrices of the
# layers, mpmath's own Talbot inversion), stored below.


def _f():
    return _resolve("layered_paleoclimate_perturbation", ("step_6", "steps.step_6", "solution.step_6", "solution"))


def _timed(*args):
    return _f()(*args)


def _halfspace(z, kappa, t_years, dT):
    edges = [0.0] + [t * YEAR for t in t_years]
    if z == 0.0:
        return dT[0]
    tot = 0.0
    for i, d in enumerate(dT):
        e_old = math.erfc(z / (2 * math.sqrt(kappa * edges[i + 1])))
        e_new = 0.0 if i == 0 else math.erfc(z / (2 * math.sqrt(kappa * edges[i])))
        tot += d * (e_old - e_new)
    return tot


def _two_layer_step(x, t, l, k1, k2, rho_c):
    # surface raised by 1 K a time t (s) ago over a layer 0 < z < l (k1) on a half-space (k2):
    # alpha = (e1 - e2) / (e1 + e2), e = sqrt(k rho_c) the thermal effusivity
    a1, a2 = k1 / rho_c, k2 / rho_c
    al = (math.sqrt(k1 * rho_c) - math.sqrt(k2 * rho_c)) / (math.sqrt(k1 * rho_c) + math.sqrt(k2 * rho_c))
    tot, n = 0.0, 0
    while True:
        if x <= l:
            term = (-al) ** n * (math.erfc((2 * n * l + x) / (2 * math.sqrt(a1 * t)))
                                 + al * math.erfc((2 * (n + 1) * l - x) / (2 * math.sqrt(a1 * t))))
        else:
            term = (1 + al) * (-al) ** n * math.erfc(((2 * n + 1) * l / math.sqrt(a1) + (x - l) / math.sqrt(a2))
                                                     / (2 * math.sqrt(t)))
        tot += term
        n += 1
        if abs(al) ** n < 1e-18 or n > 2000:
            return tot


def _two_layer(x, l, k1, k2, rho_c, t_years, dT):
    if x == 0.0:
        return dT[0]
    edges = [0.0] + [t * YEAR for t in t_years]
    tot = 0.0
    for i, d in enumerate(dT):
        u_old = _two_layer_step(x, edges[i + 1], l, k1, k2, rho_c)
        u_new = 0.0 if i == 0 else _two_layer_step(x, edges[i], l, k1, k2, rho_c)
        tot += d * (u_old - u_new)
    return tot


def test_uniform_column_is_the_half_space():
    # equal conductivities in every layer: kappa = k / rho_c everywhere, the erfc solution of step 3
    tops, k, rho_c = [0.0, 300.0, 800.0], [3.0, 3.0, 3.0], 2.5e6
    t_y, d = [1e3, 1.1e4, 9e4, 1.3e5], [0.8, 0.0, -1.0, 0.3]
    z = [0.0, 1.0, 50.0, 299.9, 300.0, 300.1, 700.0, 1500.0, 3000.0]
    got = np.asarray(_timed(np.array(z), tops, k, rho_c, t_y, d))
    exp = [_halfspace(v, 3.0 / 2.5e6, t_y, d) for v in z]
    assert np.max(np.abs(got - np.array(exp))) < 1e-8      # 1e-8 K per K of the largest |dT|


@pytest.mark.parametrize("case", [(2.0, 5.0, 400.0), (6.0, 0.8, 400.0), (1.2, 4.4, 35.0)])
def test_layer_over_half_space(case):
    # a poorly and a highly conducting cover, and a thin cover: image series of the step response
    k1, k2, l = case
    rho_c, t_y, d = 2.4e6, [300.0, 1.2e4, 1.0e5, 1.25e5], [0.5, 0.0, -6.0, 1.5]
    z = [0.0, 3.0, 0.5 * l, 0.999 * l, l, 1.001 * l, 2.0 * l, l + 900.0, l + 2500.0]
    got = np.asarray(_timed(np.array(z), [0.0, l], [k1, k2], rho_c, t_y, d))
    exp = [_two_layer(v, l, k1, k2, rho_c, t_y, d) for v in z]
    assert np.max(np.abs(got - np.array(exp))) < 6e-8      # 1e-8 K per K, largest |dT| = 6


_T_ABITIBI = {1.0: -0.0008520484521674185183899386, 75.0: -0.06374574375906197936180743, 150.0: -0.1265498095117345146312706, 300.0: -0.1864289701210462478056366, 420.0: -0.230553633845341526486891, 650.0: -0.3377790202469069022517831, 900.0: -0.4209209384432630549218601, 1100.0: -0.4649965444213023133078132, 1500.0: -0.4847821038676746412810152, 2500.0: -0.3593394742270351026293501}
_T_CONTRAST = {5.0: 1.074603418068411517440318, 30.0: 0.4517810019483242366355346, 60.0: -0.2713805757456721570727339, 61.0: -0.2742609672749097256974367, 130.0: -0.4567504980928152345608845, 199.0: -0.6059285087183417791936461, 200.0: -0.6078422224580352027394069, 400.0: -1.475117606748449669874537, 900.0: -2.388730544290318546369798}


def test_multilayer_columns():
    # a five-layer Shield column with the glacial history of the task, and a strongly contrasted column
    # (k from 0.8 to 6.5) with a recent warming: values against 90-digit targets
    got = _timed(np.array([float(v) for v in _T_ABITIBI]), [0, 150, 420, 900, 1300], [2.3, 4.6, 3.1, 2.8, 3.5],
                 2.5e6, [1e4, 1e5, 1.2e5], [0.0, -1.0, 0.25])
    assert np.max(np.abs(np.asarray(got) - np.array(list(_T_ABITIBI.values())))) < 1e-8
    got = _timed(np.array([float(v) for v in _T_CONTRAST]), [0, 60, 200], [0.8, 6.5, 2.0], 2.1e6,
                 [500.0, 1.2e4, 9e4], [1.2, -0.4, -5.5])
    assert np.max(np.abs(np.asarray(got) - np.array(list(_T_CONTRAST.values())))) < 5.5e-8


def test_shape_scalar_and_surface():
    # the output keeps the full shape of z; a scalar depth gives a float; z = 0 gives dT[0]
    tops, k, rc, t_y, d = [0, 150, 420], [2.3, 4.6, 3.1], 2.5e6, [2e3, 2.5e4, 1.1e5], [0.6, -2.5, 0.4]
    z = np.array([[0.0, 150.0, 400.0], [900.0, 1600.0, 3000.0]])
    got = _timed(z, tops, k, rc, t_y, d)
    assert isinstance(got, np.ndarray) and got.shape == (2, 3)
    one = _timed(400.0, tops, k, rc, t_y, d)
    assert type(one) is float
    assert abs(got[0, 2] - one) < 5e-8          # two computed outputs, each within 2.5e-8
    assert abs(got[0, 0] - 0.6) < 2.5e-8


def test_many_depths_layers_and_intervals():
    # 1000 depths, 50 layers and 20 history intervals in one call. With one conductivity in every layer
    # the column is a half-space (erfc solution); with varying conductivities every depth agrees with a
    # single-depth call
    z = np.linspace(0.0, 3000.0, 1000)
    t_y, d = np.logspace(2.0, 5.5, 20), np.sin(np.arange(20.0))
    tops = np.linspace(0.0, 2450.0, 50)
    out = np.asarray(_timed(z, tops, np.full(50, 2.9), 2.4e6, t_y, d))
    assert out.shape == (1000,)
    want = np.array([_halfspace(float(x), 2.9 / 2.4e6, list(t_y), list(d)) for x in z])
    assert np.max(np.abs(out - want)) < 1e-8 * np.max(np.abs(d))
    k = np.linspace(1.5, 5.0, 50)
    out = np.asarray(_timed(z, tops, k, 2.4e6, t_y, d))
    assert out.shape == (1000,) and np.all(np.isfinite(out))
    for j in (0, 137, 500, 999):
        one = _timed(float(z[j]), tops, k, 2.4e6, t_y, d)
        assert abs(out[j] - one) < 2e-8 * np.max(np.abs(d)), (j, out[j], one)


@pytest.mark.parametrize("case", ["top0", "order", "k0", "knan", "len", "rc", "rcnan", "t0", "t_order", "dT_nan",
                                  "hist_len", "zneg", "znan", "two_d", "empty"])
def test_invalid_raises(case):
    z, tops, k, rc, t, d = 100.0, [0.0, 200.0], [2.0, 3.0], 2.5e6, [1e3, 1e4], [0.5, -1.0]
    if case == "top0":
        tops = [10.0, 200.0]
    elif case == "order":
        tops = [0.0, 200.0, 150.0]
        k = [2.0, 3.0, 2.5]
    elif case == "k0":
        k = [2.0, 0.0]
    elif case == "knan":
        k = [2.0, float("nan")]
    elif case == "len":
        k = [2.0]
    elif case == "rc":
        rc = 0.0
    elif case == "rcnan":
        rc = float("nan")
    elif case == "t0":
        t = [0.0, 1e4]
    elif case == "t_order":
        t = [1e4, 1e3]
    elif case == "dT_nan":
        d = [0.5, float("nan")]
    elif case == "hist_len":
        d = [0.5]
    elif case == "zneg":
        z = -1.0
    elif case == "two_d":
        tops, k = [[0.0, 200.0]], [[2.0, 3.0]]          # equal shapes, but not 1-D
    elif case == "empty":
        tops, k = [], []                                 # no layer at all
    else:
        z = float("nan")
    with pytest.raises(ValueError):
        _f()(z, tops, k, rc, t, d)


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
