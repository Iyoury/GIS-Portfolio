"""Whole-task tests: surface_heat_flow on three synthetic inclined boreholes
regenerated in this file with fixed seeds (no external data file)."""
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


ARGS = ["survey_md", "survey_inc", "survey_azi", "log_md", "log_temp", "layer_top_md",
        "layer_k", "heat_production", "kappa", "hist_t_years", "hist_dT_shape", "z_min"]
OUT = {"T0", "q0", "amplitude", "sigma_T0", "sigma_q0", "sigma_amplitude",
       "rms_residual", "n_used"}


def _f():
    return _resolve("surface_heat_flow", ("solution", "main"))


# ---------------------------------------------------------------------------
# Synthetic boreholes, regenerated here with fixed seeds (numpy + math only).
# This is the generator documented in source.md with the adaptive quadrature
# for R and S replaced by the exact layer sums; both agree to < 1e-12.
# ---------------------------------------------------------------------------
_YEAR = 365.25 * 86400.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_HIST_T = [1.0e4, 1.0e5, 1.2e5]
_HIST_SHAPE = [0.0, -1.0, 0.25]
_Z_MIN = 150.0


def _g_tangent(inc, azi):
    return np.array([np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])


def _g_tvd(md, inc_deg, azi_deg, m):
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    total = 0.0
    for i in range(len(md) - 1):
        a, b = md[i], min(md[i + 1], m)
        if b <= a:
            break
        t1, t2 = _g_tangent(inc[i], azi[i]), _g_tangent(inc[i + 1], azi[i + 1])
        beta = np.arccos(np.clip(t1 @ t2, -1, 1))
        L = md[i + 1] - md[i]
        s = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        f = (s - a) / L
        if beta < 1e-12:
            tz = np.full_like(s, t1[2])
        else:
            tz = (np.sin((1 - f) * beta) * t1[2] + np.sin(f * beta) * t2[2]) / np.sin(beta)
        total += 0.5 * (b - a) * np.sum(_GL_W * tz)
    return total


def _g_RS(z, tops, k):
    R = S = 0.0
    bots = list(tops[1:]) + [math.inf]
    for top, bot, kk in zip(tops, bots, k):
        b = min(max(z, top), bot)
        R += (b - top) / kk
        S += (b * b - top * top) / (2 * kk)
    return R, S


def _g_P(z, shape, kappa):
    edges = [0.0] + [t * _YEAR for t in _HIST_T]
    out = 0.0
    for i, d in enumerate(shape):
        e_old = math.erfc(z / (2 * math.sqrt(kappa * edges[i + 1])))
        e_new = 0.0 if edges[i] == 0 else math.erfc(z / (2 * math.sqrt(kappa * edges[i])))
        out += d * (e_old - e_new)
    return out


def _g_survey(total, step, inc0, inc1, azi0, azi1, wobble, seed):
    rng = np.random.default_rng(seed)
    md = np.arange(0, total + 1e-9, step)
    f = md / total
    inc = inc0 + (inc1 - inc0) * f + wobble * np.sin(6 * f) * rng.uniform(0.5, 1)
    azi = azi0 + (azi1 - azi0) * f**1.3
    return md, inc, azi


_CASES = {
    "noiseless": dict(seed=1, survey=(1500, 30, 35, 44, 180, 196, 2.0, 1),
                      layers=([0, 180, 420, 610, 900, 1150, 1330], [2.9, 3.4, 2.6, 4.6, 3.1, 2.4, 3.8]),
                      A=0.6e-6, kappa=1.2e-6, q0=0.041, T0=5.2, g=6.5, log_step=5.0, log_top=20.0, noise=0.0),
    "noisy": dict(seed=20260929, survey=(1800, 25, 28, 40, 330, 350, 1.5, 2),
                  layers=([0, 120, 350, 700, 820, 1210, 1500], [3.3, 2.7, 3.9, 2.5, 4.4, 3.0, 2.8]),
                  A=0.9e-6, kappa=1.1e-6, q0=0.037, T0=4.1, g=7.8, log_step=5.0, log_top=15.0, noise=0.005),
    "deep": dict(seed=4242, survey=(2400, 50, 8, 15, 90, 110, 1.0, 3),
                 layers=([0, 260, 900, 1400, 2000], [3.6, 2.9, 3.2, 2.6, 3.5]),
                 A=1.8e-6, kappa=1.3e-6, q0=0.052, T0=3.6, g=5.0, log_step=10.0, log_top=30.0, noise=0.010),
}

# Target provenance. No expected value is taken from the reference solution.
#  * noiseless hole: targets are the generator inputs q0, T0, g (analytic truth).
#  * noisy / deep holes: targets are recomputed below, at test time, by ordinary least
#    squares (numpy.linalg.lstsq, covariance from the pseudo-inverse) on the generator's OWN
#    model terms: TVD from Gauss-Legendre integration of the tangent, R and S from exact layer
#    sums, P from math.erfc. Valid for the stated 1-D conductive model with horizontal layers
#    and uniform diffusivity; the least-squares covariance is the usual s^2 (G^T G)^-1 with
#    s^2 = RSS/(n-3).
#  * n_used: count of readings whose generator TVD is >= z_min (definition).
_TARGET_PROVENANCE = {
    "q0, T0, amplitude (noiseless)": ("generator inputs", "analytic truth",
                                      "noise-free data following the stated model"),
    "q0, T0, amplitude, sigmas, rms (noisy, deep)": (
        "independent OLS on generator-side model terms (lstsq + pinv)",
        "agrees with the reference to 1e-10 W m^-2 and 1e-7 K",
        "1-D conduction, horizontal layers, uniform kappa; Gaussian noise"),
    "n_used": ("count of generator TVD >= z_min", "definition", "always"),
}


def _independent_fit(z, T, R, S, P, A, z_min):
    keep = z >= z_min
    G = np.column_stack([np.ones(int(keep.sum())), R[keep], P[keep]])
    y = T[keep] + A * S[keep]
    m, *_ = np.linalg.lstsq(G, y, rcond=None)
    res = y - G @ m
    n = y.size
    Gp = np.linalg.pinv(G)
    C = (res @ res) / (n - 3) * (Gp @ Gp.T)
    return {"T0": m[0], "q0": m[1], "amplitude": m[2], "sigma_T0": math.sqrt(C[0, 0]),
            "sigma_q0": math.sqrt(C[1, 1]), "sigma_amplitude": math.sqrt(C[2, 2]),
            "rms_residual": math.sqrt(res @ res / n), "n_used": int(n)}


_CACHE = {}


def _load(name):
    """Return (argument list for surface_heat_flow, attributes dict)."""
    if name not in _CACHE:
        c = _CASES[name]
        rng = np.random.default_rng(c["seed"])
        md, inc, azi = _g_survey(*c["survey"])
        tops_md, k = (np.asarray(v, float) for v in c["layers"])
        log_md = np.arange(c["log_top"], md[-1] + 1e-9, c["log_step"])
        tops = [_g_tvd(md, inc, azi, m) for m in tops_md]
        T = np.empty(log_md.size)
        zs, Rs, Ss, Ps = (np.empty(log_md.size) for _ in range(4))
        for j, m in enumerate(log_md):
            z = _g_tvd(md, inc, azi, m)
            R, S = _g_RS(z, tops, k)
            P = _g_P(z, _HIST_SHAPE, c["kappa"])
            zs[j], Rs[j], Ss[j], Ps[j] = z, R, S, P
            T[j] = c["T0"] + c["q0"] * R - c["A"] * S + c["g"] * P
        if c["noise"] > 0:
            T = T + c["noise"] * rng.standard_normal(T.size)
        args = [md, inc, azi, log_md, T, tops_md, k, c["A"], c["kappa"],
                np.array(_HIST_T), np.array(_HIST_SHAPE), _Z_MIN]
        attrs = {"true_q0": c["q0"], "true_T0": c["T0"], "true_amplitude": c["g"], "z_min": _Z_MIN}
        ref = _independent_fit(zs, T, Rs, Ss, Ps, c["A"], _Z_MIN)
        attrs.update({"ref_" + key: v for key, v in ref.items()})
        _CACHE[name] = (args, attrs)
    args, attrs = _CACHE[name]
    return [a.copy() if isinstance(a, np.ndarray) else a for a in args], dict(attrs)


def test_output_contract():
    d, a = _load("noisy")
    r = _f()(*d)
    assert isinstance(r, dict) and OUT <= set(r)
    for k in OUT - {"n_used"}:
        assert type(r[k]) is float, k      # plain Python floats as stated
    assert isinstance(r["n_used"], int)
    assert int(round(float(r["n_used"]))) == int(a["ref_n_used"])
    for k in OUT - {"n_used"}:
        assert np.isfinite(float(r[k]))


def test_noiseless_hole_returns_generator_parameters():
    # analytic truth: temperatures generated from q0 = 0.041 W/m2, T0 = 5.2 C,
    # glacial amplitude 6.5 K on an independently integrated trajectory
    d, a = _load("noiseless")
    r = _f()(*d)
    assert abs(r["q0"] - a["true_q0"]) < 1e-8
    assert abs(r["T0"] - a["true_T0"]) < 1e-5
    assert abs(r["amplitude"] - a["true_amplitude"]) < 1e-4
    assert r["rms_residual"] < 1e-6
    assert int(round(float(r["n_used"]))) == int(a["ref_n_used"])


@pytest.mark.parametrize("name", ["noisy", "deep"])
def test_noisy_holes_match_least_squares_solution(name):
    d, a = _load(name)
    r = _f()(*d)
    assert int(round(float(r["n_used"]))) == int(a["ref_n_used"])
    assert abs(r["q0"] - a["ref_q0"]) < 2e-6
    assert abs(r["T0"] - a["ref_T0"]) < 5e-4
    assert abs(r["amplitude"] - a["ref_amplitude"]) < 2e-3
    for k in ("sigma_q0", "sigma_T0", "sigma_amplitude"):
        assert abs(r[k] / a["ref_" + k] - 1) < 0.02
    assert abs(r["rms_residual"] / a["ref_rms_residual"] - 1) < 0.01
    # the generator truth is consistent with the reported uncertainty
    assert abs(r["q0"] - a["true_q0"]) < 4 * r["sigma_q0"]
    assert abs(r["amplitude"] - a["true_amplitude"]) < 4 * r["sigma_amplitude"]


@pytest.mark.parametrize("case", ["log_len", "few_after_cut", "zero_shape", "top_beyond_hole",
                                  "negative_A", "bad_kappa", "log_beyond_hole", "bad_survey_start",
                                  "bad_survey_order", "hist_not_increasing", "hist_nan"])
def test_invalid_inputs_raise(case):
    d, _ = _load("noisy")
    i = {a: j for j, a in enumerate(ARGS)}
    if case == "log_len":
        d[i["log_temp"]] = d[i["log_temp"]][:-1]
    elif case == "few_after_cut":
        d[i["z_min"]] = 1e5
    elif case == "zero_shape":
        d[i["hist_dT_shape"]] = np.zeros_like(d[i["hist_dT_shape"]])
    elif case == "top_beyond_hole":
        tops = np.array(d[i["layer_top_md"]], float)
        tops[-1] = d[i["survey_md"]][-1] + 100.0
        d[i["layer_top_md"]] = tops
    elif case == "negative_A":
        d[i["heat_production"]] = -1e-6
    elif case == "bad_kappa":
        d[i["kappa"]] = 0.0
    elif case == "bad_survey_start":
        md = np.array(d[i["survey_md"]], float); md[0] = 5.0
        d[i["survey_md"]] = md
    elif case == "bad_survey_order":
        md = np.array(d[i["survey_md"]], float); md[1], md[2] = md[2], md[1]
        d[i["survey_md"]] = md
    elif case == "hist_not_increasing":
        t = np.array(d[i["hist_t_years"]], float); t[1] = t[0]
        d[i["hist_t_years"]] = t
    elif case == "hist_nan":
        t = np.array(d[i["hist_t_years"]], float); t[1] = float("nan")
        d[i["hist_t_years"]] = t
    else:
        md = np.array(d[i["log_md"]], float)
        md[-1] = d[i["survey_md"]][-1] + 5.0
        d[i["log_md"]] = md
    with pytest.raises(ValueError):
        _f()(*d)


def test_z_min_is_applied_in_vertical_depth():
    # raising z_min by 200 m of TVD removes exactly the readings between the two cuts
    d, a = _load("noisy")
    i = {x: j for j, x in enumerate(ARGS)}
    r1 = _f()(*d)
    d[i["z_min"]] = float(a["z_min"]) + 200.0
    r2 = _f()(*d)
    # 46 log readings of this hole have 150 m <= TVD < 350 m (40 would be removed by an MD cut)
    assert int(round(float(r1["n_used"]))) - int(round(float(r2["n_used"]))) == 46
    assert abs(r2["q0"] - a["true_q0"]) < 4 * r2["sigma_q0"]


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
