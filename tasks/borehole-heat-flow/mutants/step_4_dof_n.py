"""Mutant (step 4): residual variance divided by n instead of n - 3 degrees of freedom."""
import numpy as np


def fit_heat_flow(T, R, S, P, A):
    """Least-squares T = T0 + q0 R - A S + g P  ->  (T0, q0, g) + errors."""
    T, R, S, P = (np.asarray(v, float).ravel() for v in (T, R, S, P))
    n = T.size
    if not (R.size == S.size == P.size == n):
        raise ValueError("arrays must have equal length")
    if n < 4:
        raise ValueError("need at least 4 temperatures")
    if not np.all(np.isfinite(np.concatenate([T, R, S, P]))) or not np.isfinite(A) or A < 0:
        raise ValueError("invalid input")
    y = T + A * S
    G = np.column_stack([np.ones(n), R, P])
    # stated criterion: a zero column, or smallest singular value < 1e-8 after
    # scaling each column to unit Euclidean length
    norms = np.linalg.norm(G, axis=0)
    if np.any(norms == 0.0) or np.linalg.svd(G / norms, compute_uv=False)[-1] < 1e-8:
        raise ValueError("T0, q0 and amplitude are not separately resolvable (rank-deficient design)")
    GtG = G.T @ G
    m = np.linalg.solve(GtG, G.T @ y)
    res = y - G @ m
    s2 = float(res @ res) / n               # MUTANT: n instead of n - 3
    C = s2 * np.linalg.inv(GtG)
    return {"T0": float(m[0]), "q0": float(m[1]), "amplitude": float(m[2]),
            "sigma_T0": float(np.sqrt(C[0, 0])), "sigma_q0": float(np.sqrt(C[1, 1])),
            "sigma_amplitude": float(np.sqrt(C[2, 2])),
            "rms_residual": float(np.sqrt(np.mean(res**2)))}
