# --- test case 0 ---
# exact recovery
# Targets: the generating parameters of noise-free synthetic profiles, and numpy.linalg.lstsq with the OLS covariance
# computed inside the test for a noisy profile.
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
    # noise-free data: the exact least-squares solution is the generating (T0, q0, g) = (4.7, 0.044, 7.0); stated
    # accuracy (scaled smallest singular value 0.039 >= 1e-3): every fitted term T0, q0 R, g P within 1e-9 K
    assert abs(r["T0"] - 4.7) <= 1e-9
    assert abs(r["q0"] - 0.044) * np.max(np.abs(_t4_R)) <= 1e-9
    assert abs(r["amplitude"] - 7.0) * np.max(np.abs(_t4_P)) <= 1e-9
    assert r["rms_residual"] <= 1e-9
    # the same profile with P in units of 1e-4 (g = 7e4 K): the limit on g scales with 1 / max|P|
    r = fit_heat_flow(4.7 + 0.044 * _t4_R - 1.1e-6 * _t4_S + 7e4 * (1e-4 * _t4_P), _t4_R, _t4_S, 1e-4 * _t4_P, 1.1e-6)
    assert abs(r["amplitude"] - 7e4) * np.max(np.abs(1e-4 * _t4_P)) <= 1e-9
    assert abs(r["T0"] - 4.7) <= 1e-9 and abs(r["q0"] - 0.044) * np.max(np.abs(_t4_R)) <= 1e-9


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
    # stated accuracy (scaled smallest singular value 0.050): T0, q0 R and g P within 1e-9 K
    assert abs(r["T0"] - 4.7) <= 1e-9
    assert abs(r["q0"] - 0.044) * np.max(np.abs(_t4_R)) <= 1e-9
    assert abs(r["amplitude"] - 7.0) * np.max(np.abs(_t4_P)) <= 1e-9
    assert r["rms_residual"] <= 1e-9


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
    assert abs(r["T0"] - m[0]) <= 1e-9
    assert abs(r["q0"] - m[1]) * np.max(np.abs(_t4_R)) <= 1e-9
    assert abs(r["amplitude"] - m[2]) * np.max(np.abs(_t4_P)) <= 1e-9
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
    assert abs(r["q0"] - 0.044) * np.max(np.abs(_t4_R)) <= 1e-9
    # ignoring the known heat production (A = 0) biases q0 low by more than 1e-4 W m^-2
    assert r0["q0"] < 0.044 - 1e-4


_crown_case()

# --- test case 4 ---
# output types, and the standard errors at the floor rms_residual = 1e-6 K of their stated accuracy
import numpy as np


def _crown_case():
    import math
    import numpy as np
    # a residual r orthogonal to the columns 1, R, P with rms exactly 1e-6 K: the exact least-squares solution is the
    # generating (T0, q0, g) = (4.7, 0.044, 7.0), RSS = |r|^2, and the standard errors are
    # sqrt(RSS / (n - 3) [(G^T G)^-1]_ii) with (G^T G)^-1 = Rm^-1 Rm^-T from a QR factorization G = Q Rm
    z = np.linspace(150, 1400, 120)
    R = z / 3.1 + 0.00002 * z**1.5
    S = z**2 / (2 * 3.1)
    P = -(np.exp(-z / 900.0) - np.exp(-z / 250.0)) - 0.3 * np.exp(-z / 2000.0)
    Q, Rm = np.linalg.qr(np.column_stack([np.ones(z.size), R, P]))
    r = np.random.default_rng(5).standard_normal(z.size)
    for _ in range(2):
        r = r - Q @ (Q.T @ r)
    r *= 1e-6 / math.sqrt(np.mean(r * r))
    out = fit_heat_flow(4.7 + 0.044 * R - 1.1e-6 * S + 7.0 * P + r, R, S, P, 1.1e-6)
    keys = ["T0", "q0", "amplitude", "sigma_T0", "sigma_q0", "sigma_amplitude", "rms_residual"]
    # output contract: a dict of Python floats
    assert isinstance(out, dict) and all(type(out[k]) is float for k in keys)
    assert abs(out["T0"] - 4.7) <= 1e-9 and abs(out["q0"] - 0.044) * np.max(R) <= 1e-9
    assert abs(out["amplitude"] - 7.0) * np.max(np.abs(P)) <= 1e-9
    Ri = np.linalg.inv(Rm)
    sig = np.sqrt(float(r @ r) / (z.size - 3) * np.diag(Ri @ Ri.T))
    for key, v in zip(("sigma_T0", "sigma_q0", "sigma_amplitude"), sig):
        assert abs(out[key] / v - 1.0) < 1e-6, (key, out[key], v)
    assert abs(out["rms_residual"] - 1e-6) < 1e-9


_crown_case()

# --- test case 5 ---
# resolvability threshold 1e-8: a nearly collinear design just above it is fitted with an accurate rms residual
# Targets: a residual orthogonal to [1, R, P] (its rms is the exact least-squares rms residual) and the stated
# rejection below the threshold.
import numpy as np


def _crown_case():
    import math
    import numpy as np
    z = np.linspace(150, 1400, 120)
    R = z / 3.1 + 0.00002 * z**1.5
    S = z**2 / (2 * 3.1)
    one = np.ones(z.size)
    Q2, _ = np.linalg.qr(np.column_stack([one, R]))
    u = R**2 - Q2 @ (Q2.T @ R**2)
    u /= np.linalg.norm(u)
    for alpha, accepted in ((1.286e-6, True), (1.286e-7, False)):
        # P = 0.01 R + alpha u (u: unit part of R^2 orthogonal to 1 and R) is nearly proportional to R
        P = 0.01 * R + alpha * u
        G = np.column_stack([one, R, P])
        s_min = np.linalg.svd(G / np.linalg.norm(G, axis=0), compute_uv=False)[-1]
        if not accepted:
            # scaled smallest singular value 3.0e-9 < 1e-8: unresolvable, ValueError
            assert 0.0 < s_min < 1e-8
            try:
                fit_heat_flow(4.7 + 0.044 * R + 7.0 * P, R, S, P, 1.1e-6)
            except ValueError:
                continue
            raise AssertionError("fit_heat_flow must raise ValueError below the threshold")
        assert 2.9e-8 < s_min < 3.1e-8
        # r orthogonal to 1, R, P with rms 1e-3 K; the exact least-squares solution is (T0, q0, g) = (4.7, -1.956, 207):
        # q0 R and g P reach 885 and 937 K and nearly cancel; the rms residual is 1e-3 K (60-digit check: 4e-15 K)
        Q, _ = np.linalg.qr(G)
        r = np.random.default_rng(7).standard_normal(z.size)
        for _ in range(2):
            r = r - Q @ (Q.T @ r)
        r *= 1e-3 / math.sqrt(np.mean(r * r))
        out = fit_heat_flow(4.7 - 1.956 * R - 1.1e-6 * S + 207.0 * P + r, R, S, P, 1.1e-6)
        # stated accuracy for every resolvable design (fitted terms up to 1e3 K): rms_residual within 1e-9 K
        assert abs(out["rms_residual"] - 1e-3) <= 1e-9, out["rms_residual"]


_crown_case()

# --- test case 6 ---
# stated error rules: n < 4, A < 0, a zero R or P column, P proportional to R
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

    for case in ["short", "zeroR", "zeroP", "P_prop_R", "negA"]:
        _t4_t_msg = " (case %r)" % (case,)
        _t4_T, _t4_R, _t4_S, _t4_P = _t4_synthetic(0.0)
        A = 1e-6
        if case == "short":
            _t4_T, _t4_R, _t4_S, _t4_P = _t4_T[:3], _t4_R[:3], _t4_S[:3], _t4_P[:3]
        elif case == "zeroR":
            _t4_R = np.zeros_like(_t4_R)
        elif case == "zeroP":
            _t4_P = np.zeros_like(_t4_P)
        elif case == "P_prop_R":
            _t4_P = 0.01 * _t4_R
        else:
            A = -1e-6
        try:
            fit_heat_flow(_t4_T, _t4_R, _t4_S, _t4_P, A)
        except ValueError:
            pass
        else:
            raise AssertionError("fit_heat_flow must raise ValueError" + _t4_t_msg)


_crown_case()

# --- test case 7 ---
# nearly collinear but resolved design (scaled smallest singular value 1.05e-3): coefficients and standard errors
# Targets: the generating values (residual orthogonal to [1, R, P]) and the Frisch-Waugh identity
# [(G^T G)^-1]_jj = 1 / |G_j - projection of G_j on the other two columns|^2.
import numpy as np


def _crown_case():
    import math
    import numpy as np
    z = np.linspace(150, 1400, 120)
    R = z / 3.1 + 0.00002 * z**1.5
    S = z**2 / (2 * 3.1)
    one = np.ones(z.size)
    Q2, _ = np.linalg.qr(np.column_stack([one, R]))
    u = R**2 - Q2 @ (Q2.T @ R**2)
    P = 0.01 * R + 0.045 * u / np.linalg.norm(u)
    G = np.column_stack([one, R, P])
    s_min = np.linalg.svd(G / np.linalg.norm(G, axis=0), compute_uv=False)[-1]
    assert 1e-3 <= s_min <= 1.1e-3
    Q, _ = np.linalg.qr(G)
    r = np.random.default_rng(7).standard_normal(z.size)
    for _ in range(2):
        r = r - Q @ (Q.T @ r)
    r *= 1e-3 / math.sqrt(np.mean(r * r))
    sig = []
    for j in range(3):
        others = np.delete(G, j, axis=1)
        e = G[:, j] - others @ np.linalg.lstsq(others, G[:, j], rcond=None)[0]
        sig.append(math.sqrt(float(r @ r) / (z.size - 3) / float(e @ e)))
    # (T0, q0, g) = (4.7, 0.044, 7.0), and (4.7, -1.956, 207.0) whose terms q0 R and g P reach 885 and 939 K
    for T0, q0, g in ((4.7, 0.044, 7.0), (4.7, -1.956, 207.0)):
        out = fit_heat_flow(T0 + q0 * R - 1.1e-6 * S + g * P + r, R, S, P, 1.1e-6)
        assert abs(out["T0"] - T0) <= 1e-9, (g, out["T0"])
        assert abs(out["q0"] - q0) * np.max(np.abs(R)) <= 1e-9, (g, out["q0"])
        assert abs(out["amplitude"] - g) * np.max(np.abs(P)) <= 1e-9, (g, out["amplitude"])
        assert abs(out["rms_residual"] - 1e-3) <= 1e-9
        for key, want in zip(("sigma_T0", "sigma_q0", "sigma_amplitude"), sig):
            assert abs(out[key] / want - 1.0) <= 1e-6, (g, key, out[key], want)


_crown_case()
