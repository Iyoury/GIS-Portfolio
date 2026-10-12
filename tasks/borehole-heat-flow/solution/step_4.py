"""Reference for step 4: joint least-squares estimate of T0, q0 and the glacial amplitude."""
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
    if np.any(norms == 0.0):
        raise ValueError("T0, q0 and amplitude are not separately resolvable (zero column)")
    U, sv, Vt = np.linalg.svd(G / norms, full_matrices=False)
    if sv[-1] < 1e-8:
        raise ValueError("T0, q0 and amplitude are not separately resolvable (rank-deficient design)")
    # least squares through the thin SVD of the column-scaled design (the normal equations would square its
    # condition number): m = D^-1 V S^-1 U^T y and (G^T G)^-1 = D^-1 V S^-2 V^T D^-1, D = diag(norms)
    W = Vt.T / sv
    m = (W @ (U.T @ y)) / norms
    res = y - G @ m
    s2 = float(res @ res) / (n - 3)
    C = s2 * (W @ W.T) / np.outer(norms, norms)
    return {"T0": float(m[0]), "q0": float(m[1]), "amplitude": float(m[2]),
            "sigma_T0": float(np.sqrt(C[0, 0])), "sigma_q0": float(np.sqrt(C[1, 1])),
            "sigma_amplitude": float(np.sqrt(C[2, 2])),
            "rms_residual": float(np.sqrt(np.mean(res**2)))}
