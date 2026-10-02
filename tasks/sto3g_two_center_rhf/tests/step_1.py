import numpy as np

# Independent target: Gauss-Legendre quadrature of the defining integral (no erf).
_T_XG, _T_WG = np.polynomial.legendre.leggauss(400)


def _t_F0(t):
    out = []
    for tv in np.ravel(np.asarray(t, dtype=float)):
        top = 1.0 if tv <= 100.0 else 10.0 / np.sqrt(tv)
        u = 0.5 * top * (_T_XG + 1.0)
        out.append(0.5 * top * np.sum(_T_WG * np.exp(-tv * u * u)))
    return np.array(out).reshape(np.shape(t))

# --- test case 0: t = 0 is allowed and F0(0) = 1 ---
F = boys_f0(0.0)
assert np.shape(F) == ()
assert isinstance(F, (np.ndarray, np.floating)) and np.ndim(F) == 0 and np.isfinite(F) and abs(float(F) - 1.0) < 1e-11

# --- test case 1: values against the quadrature target, relative 1e-11 ---
t = np.array([0.0, 1e-12, 1e-6, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0])
F = boys_f0(t)
assert isinstance(F, np.ndarray) and F.shape == (9,) and F.dtype.kind == "f"
assert np.allclose(F, _t_F0(t), rtol=1e-11, atol=0.0)

# --- test case 2: a 2-D input keeps its shape ---
t = np.array([[0.0, 0.25, 2.0], [7.5, 0.04, 55.0]])
F = boys_f0(t)
assert F.shape == (2, 3)
assert np.allclose(F, _t_F0(t), rtol=1e-11, atol=0.0)

# --- test case 3: limits, tiny t (Taylor series) and huge t (asymptotic form) ---
assert abs(float(boys_f0(1e-14)) / (1.0 - 1e-14 / 3.0) - 1.0) < 1e-11
assert abs(float(boys_f0(1e4)) / (0.5 * np.sqrt(np.pi / 1e4)) - 1.0) < 1e-11

# --- test case 4: negative or non-finite t raise ValueError ---
for _t_bad in (-1.0, float("nan"), np.array([1.0, -0.5]), float("inf")):
    try:
        boys_f0(_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("boys_f0(%r) must raise ValueError" % (_t_bad,))
