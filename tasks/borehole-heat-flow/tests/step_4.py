# Tests for step 4: fit_heat_flow (joint OLS estimate of T0, q0 and g).
# Targets: the generating parameters of noise-free synthetic profiles, and numpy.linalg.lstsq with the OLS
# covariance computed inside the test for a noisy profile.

# --- test case 0 ---
# exact recovery
import numpy as np


def _crown_case():
    import numpy as np

    _t4_KEYS = {"T0", "q0", "amplitude", "sigma_T0", "sigma_q0", "sigma_amplitude", "rms_residual"}

    def _t4_synthetic(noise, seed=5):
        rng = np.random.default_rng(seed)
        z = np.linspace(150, 1400, 120)
        R = z / 3.1 + 0.00002 * z**1.5
        S = z**2 / (2 * 3.1)
        P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
        T = 4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + noise * rng.standard_normal(z.size)
        return T, R, S, P

    _t4_T, _t4_R, _t4_S, _t4_P = _t4_synthetic(0.0)
    r = fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 1.1e-6)
    assert _t4_KEYS <= set(r)
    # noise-free data: the exact least-squares solution is the generating (T0, q0, g) = (4.7, 0.044, 7.0)
    assert abs(r["T0"] - 4.7) < 1e-9
    assert abs(r["q0"] - 0.044) < 1e-12
    assert abs(r["amplitude"] - 7.0) < 1e-9
    assert r["rms_residual"] < 1e-9


_crown_case()

# --- test case 1 ---
# four observations are enough
import numpy as np


def _crown_case():
    import numpy as np
    # n = 4 is the smallest valid sample: a resolvable four-reading design is fitted exactly
    z = np.array([200.0, 500.0, 900.0, 1400.0])
    _t4_R = z / 3.1 + 0.00002 * z**1.5
    _t4_S = z**2 / (2 * 3.1)
    _t4_P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
    _t4_T = 4.7 + 0.044 * _t4_R - 1.1e-6 * _t4_S + 7.0 * _t4_P
    r = fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 1.1e-6)
    # noise-free data: the exact least-squares solution is the generating (T0, q0, g) = (4.7, 0.044, 7.0)
    assert abs(r["T0"] - 4.7) < 1e-9
    assert abs(r["q0"] - 0.044) < 1e-12
    assert abs(r["amplitude"] - 7.0) < 1e-9
    assert r["rms_residual"] < 1e-9


_crown_case()

# --- test case 2 ---
# noisy estimates and covariance
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t4_synthetic(noise, seed=5):
        rng = np.random.default_rng(seed)
        z = np.linspace(150, 1400, 120)
        R = z / 3.1 + 0.00002 * z**1.5
        S = z**2 / (2 * 3.1)
        P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
        T = 4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + noise * rng.standard_normal(z.size)
        return T, R, S, P

    _t4_T, _t4_R, _t4_S, _t4_P = _t4_synthetic(0.01)
    r = fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 1.1e-6)
    _t4_G = np.column_stack([np.ones_like(_t4_R), _t4_R, _t4_P])
    y = _t4_T + 1.1e-6 * _t4_S
    # independent target: numpy.linalg.lstsq (SVD) and the OLS covariance s^2 (G^T G)^-1, s^2 = RSS / (n - 3)
    m, rss, *_ = np.linalg.lstsq(_t4_G, y, rcond=None)
    _t4_C = rss[0] / (len(_t4_T) - 3) * np.linalg.inv(_t4_G.T @ _t4_G)
    assert abs(r["T0"] - m[0]) < 1e-9
    assert abs(r["q0"] - m[1]) < 1e-12
    assert abs(r["amplitude"] - m[2]) < 1e-9
    for key, i in (("sigma_T0", 0), ("sigma_q0", 1), ("sigma_amplitude", 2)):
        assert abs(r[key] / math.sqrt(_t4_C[i, i]) - 1) < 1e-6
    assert abs(r["rms_residual"] - math.sqrt(rss[0] / len(_t4_T))) < 1e-9


_crown_case()

# --- test case 3 ---
# heat production enters with negative sign
import numpy as np


def _crown_case():
    import numpy as np

    def _t4_synthetic(noise, seed=5):
        rng = np.random.default_rng(seed)
        z = np.linspace(150, 1400, 120)
        R = z / 3.1 + 0.00002 * z**1.5
        S = z**2 / (2 * 3.1)
        P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
        T = 4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + noise * rng.standard_normal(z.size)
        return T, R, S, P

    _t4_T, _t4_R, _t4_S, _t4_P = _t4_synthetic(0.0)
    r = fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 1.1e-6)
    r0 = fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 0.0)
    assert abs(r["q0"] - 0.044) < 1e-12
    # ignoring the known heat production (A = 0) biases q0 low by more than 1e-4 W m^-2
    assert r0["q0"] < 0.044 - 1e-4


_crown_case()

# --- test case 4 ---
# output types
import numpy as np


def _crown_case():
    import numpy as np

    _t4_KEYS = {"T0", "q0", "amplitude", "sigma_T0", "sigma_q0", "sigma_amplitude", "rms_residual"}

    def _t4_synthetic(noise, seed=5):
        rng = np.random.default_rng(seed)
        z = np.linspace(150, 1400, 120)
        R = z / 3.1 + 0.00002 * z**1.5
        S = z**2 / (2 * 3.1)
        P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
        T = 4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + noise * rng.standard_normal(z.size)
        return T, R, S, P

    _t4_T, _t4_R, _t4_S, _t4_P = _t4_synthetic(0.01)
    r = fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 1.1e-6)
    for k in _t4_KEYS:
        assert type(r[k]) is float
    assert r["sigma_q0"] > 0 and r["sigma_T0"] > 0 and r["sigma_amplitude"] > 0


_crown_case()

# --- test case 5 ---
# resolvability threshold is 1e minus 8
import numpy as np


def _crown_case():
    import numpy as np

    _t4_KEYS = {"T0", "q0", "amplitude", "sigma_T0", "sigma_q0", "sigma_amplitude", "rms_residual"}

    def _t4_synthetic(noise, seed=5):
        rng = np.random.default_rng(seed)
        z = np.linspace(150, 1400, 120)
        R = z / 3.1 + 0.00002 * z**1.5
        S = z**2 / (2 * 3.1)
        P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
        T = 4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + noise * rng.standard_normal(z.size)
        return T, R, S, P

    def _t4_t_scaled_min_sv(R, P):
        """Smallest singular value of [1, R, P] after scaling each column to unit length (test's own)."""
        G = np.column_stack([np.ones(R.size), R, P])
        return float(np.linalg.svd(G / np.linalg.norm(G, axis=0), compute_uv=False)[-1])

    def _t4_t_nearly_proportional_P(R, target):
        """P = 0.01 R plus a component outside span{1, R}, scaled so that the smallest scaled singular
        value of [1, R, P] equals `target` (secant iteration on the test's own singular values)."""
        Q, _ = np.linalg.qr(np.column_stack([np.ones(R.size), R]))
        u = R ** 2 - Q @ (Q.T @ (R ** 2))
        u /= np.linalg.norm(u)
        alpha = target * np.linalg.norm(0.01 * R)
        for _ in range(30):
            P = 0.01 * R + alpha * u
            s = _t4_t_scaled_min_sv(R, P)
            if abs(s / target - 1.0) < 1e-3:
                break
            alpha *= target / s
        return P, s

    _t4_t_msg = ""
    _t4_T, _t4_R, _t4_S, _t4_P0 = _t4_synthetic(0.0)
    # just above the threshold: the fit must run and return finite values
    _t4_P, s = _t4_t_nearly_proportional_P(_t4_R, 3e-8)
    assert 1e-8 < s < 1e-7
    r = fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 1e-6)
    assert all(np.isfinite(r[k]) for k in _t4_KEYS)
    # just below the threshold: rejected
    _t4_P, s = _t4_t_nearly_proportional_P(_t4_R, 3e-9)
    assert 0.0 < s < 1e-8
    try:
        fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, 1e-6)
    except ValueError:
        pass
    else:
        raise AssertionError("fit_heat_flow must raise ValueError" + _t4_t_msg)


_crown_case()

# --- test case 6 ---
# invalid raises
import numpy as np


def _crown_case():
    import numpy as np

    def _t4_synthetic(noise, seed=5):
        rng = np.random.default_rng(seed)
        z = np.linspace(150, 1400, 120)
        R = z / 3.1 + 0.00002 * z**1.5
        S = z**2 / (2 * 3.1)
        P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
        T = 4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + noise * rng.standard_normal(z.size)
        return T, R, S, P

    def _t4_t_scaled_min_sv(R, P):
        """Smallest singular value of [1, R, P] after scaling each column to unit length (test's own)."""
        G = np.column_stack([np.ones(R.size), R, P])
        return float(np.linalg.svd(G / np.linalg.norm(G, axis=0), compute_uv=False)[-1])

    def _t4_t_nearly_proportional_P(R, target):
        """P = 0.01 R plus a component outside span{1, R}, scaled so that the smallest scaled singular
        value of [1, R, P] equals `target` (secant iteration on the test's own singular values)."""
        Q, _ = np.linalg.qr(np.column_stack([np.ones(R.size), R]))
        u = R ** 2 - Q @ (Q.T @ (R ** 2))
        u /= np.linalg.norm(u)
        alpha = target * np.linalg.norm(0.01 * R)
        for _ in range(30):
            P = 0.01 * R + alpha * u
            s = _t4_t_scaled_min_sv(R, P)
            if abs(s / target - 1.0) < 1e-3:
                break
            alpha *= target / s
        return P, s

    for case in ["short", "len", "zeroP", "P_prop_R", "P_nearly_prop_R", "negA", "nanA", "nan"]:
        _t4_t_msg = " (case %r)" % (case,)
        _t4_T, _t4_R, _t4_S, _t4_P = _t4_synthetic(0.0)
        A = 1e-6
        if case == "short":
            _t4_T, _t4_R, _t4_S, _t4_P = _t4_T[:3], _t4_R[:3], _t4_S[:3], _t4_P[:3]
        elif case == "len":
            _t4_R = _t4_R[:-1]
        elif case == "zeroP":
            _t4_P = np.zeros_like(_t4_P)
        elif case == "P_prop_R":
            _t4_P = 0.01 * _t4_R
        elif case == "P_nearly_prop_R":
            _t4_P, s = _t4_t_nearly_proportional_P(_t4_R, 3e-9)
            assert 0.0 < s < 1e-8          # premise checked by the test's own singular values
        elif case == "negA":
            A = -1e-6
        elif case == "nanA":
            A = float("nan")
        else:
            _t4_T = _t4_T.copy(); _t4_T[4] = np.nan
        try:
            fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, A)
        except ValueError:
            pass
        else:
            raise AssertionError("fit_heat_flow must raise ValueError" + _t4_t_msg)


_crown_case()
