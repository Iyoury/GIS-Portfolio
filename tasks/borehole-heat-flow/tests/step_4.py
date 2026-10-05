"""Tests for step 4: fit_heat_flow."""
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

KEYS = {"T0", "q0", "amplitude", "sigma_T0", "sigma_q0", "sigma_amplitude", "rms_residual"}


def _f():
    return _resolve("fit_heat_flow", ("step_4", "steps.step_4", "solution.step_4", "solution"))


def _synthetic(noise, seed=5):
    rng = np.random.default_rng(seed)
    z = np.linspace(150, 1400, 120)
    R = z / 3.1 + 0.00002 * z**1.5
    S = z**2 / (2 * 3.1)
    P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
    T = 4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + noise * rng.standard_normal(z.size)
    return T, R, S, P


def test_exact_recovery():
    T, R, S, P = _synthetic(0.0)
    r = _f()(T, R, S, P, 1.1e-6)
    assert KEYS <= set(r)
    assert abs(r["T0"] - 4.7) < 1e-8
    assert abs(r["q0"] - 0.044) < 1e-11
    assert abs(r["amplitude"] - 7.0) < 1e-8
    assert r["rms_residual"] < 1e-9


def test_noisy_estimates_and_covariance():
    T, R, S, P = _synthetic(0.01)
    r = _f()(T, R, S, P, 1.1e-6)
    G = np.column_stack([np.ones_like(R), R, P])
    y = T + 1.1e-6 * S
    m, rss, *_ = np.linalg.lstsq(G, y, rcond=None)
    C = rss[0] / (len(T) - 3) * np.linalg.inv(G.T @ G)
    assert abs(r["T0"] - m[0]) < 1e-9
    assert abs(r["q0"] - m[1]) < 1e-12
    assert abs(r["amplitude"] - m[2]) < 1e-9
    for key, i in (("sigma_T0", 0), ("sigma_q0", 1), ("sigma_amplitude", 2)):
        assert abs(r[key] / math.sqrt(C[i, i]) - 1) < 1e-6
    assert abs(r["rms_residual"] - math.sqrt(rss[0] / len(T))) < 1e-9


def test_heat_production_enters_with_negative_sign():
    T, R, S, P = _synthetic(0.0)
    r = _f()(T, R, S, P, 1.1e-6)
    r0 = _f()(T, R, S, P, 0.0)
    assert abs(r["q0"] - 0.044) < 1e-11
    assert r0["q0"] < 0.044 - 1e-4


def test_output_types():
    T, R, S, P = _synthetic(0.01)
    r = _f()(T, R, S, P, 1.1e-6)
    for k in KEYS:
        assert type(r[k]) is float
    assert r["sigma_q0"] > 0 and r["sigma_T0"] > 0 and r["sigma_amplitude"] > 0


def _t_scaled_min_sv(R, P):
    """Smallest singular value of [1, R, P] after scaling each column to unit length (test's own)."""
    G = np.column_stack([np.ones(R.size), R, P])
    return float(np.linalg.svd(G / np.linalg.norm(G, axis=0), compute_uv=False)[-1])


def _t_nearly_proportional_P(R, target):
    """P = 0.01 R plus a component outside span{1, R}, scaled so that the smallest scaled singular
    value of [1, R, P] equals `target` (secant iteration on the test's own singular values)."""
    Q, _ = np.linalg.qr(np.column_stack([np.ones(R.size), R]))
    u = R ** 2 - Q @ (Q.T @ (R ** 2))
    u /= np.linalg.norm(u)
    alpha = target * np.linalg.norm(0.01 * R)
    for _ in range(30):
        P = 0.01 * R + alpha * u
        s = _t_scaled_min_sv(R, P)
        if abs(s / target - 1.0) < 1e-3:
            break
        alpha *= target / s
    return P, s


def test_resolvability_threshold_is_1e_minus_8():
    T, R, S, P0 = _synthetic(0.0)
    # just above the threshold: the fit must run and return finite values
    P, s = _t_nearly_proportional_P(R, 3e-8)
    assert 1e-8 < s < 1e-7
    r = _f()(T, R, S, P, 1e-6)
    assert all(np.isfinite(r[k]) for k in KEYS)
    # just below the threshold: rejected
    P, s = _t_nearly_proportional_P(R, 3e-9)
    assert 0.0 < s < 1e-8
    with pytest.raises(ValueError):
        _f()(T, R, S, P, 1e-6)


@pytest.mark.parametrize("case", ["short", "len", "zeroP", "P_prop_R", "P_nearly_prop_R", "negA", "nanA", "nan"])
def test_invalid_raises(case):
    T, R, S, P = _synthetic(0.0)
    A = 1e-6
    if case == "short":
        T, R, S, P = T[:3], R[:3], S[:3], P[:3]
    elif case == "len":
        R = R[:-1]
    elif case == "zeroP":
        P = np.zeros_like(P)
    elif case == "P_prop_R":
        P = 0.01 * R
    elif case == "P_nearly_prop_R":
        P, s = _t_nearly_proportional_P(R, 3e-9)
        assert 0.0 < s < 1e-8          # premise checked by the test's own singular values
    elif case == "negA":
        A = -1e-6
    elif case == "nanA":
        A = float("nan")
    else:
        T = T.copy(); T[4] = np.nan
    with pytest.raises(ValueError):
        _f()(T, R, S, P, A)


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
