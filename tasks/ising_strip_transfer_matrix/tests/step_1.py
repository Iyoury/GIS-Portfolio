# --- test case 0 ---
import numpy as np

def _t_brute(L, T, h):
    # entry by entry from the energy; the in-row bonds and the field of each row get half weight
    K = 1.0 / T
    n = 2 ** L
    M = np.empty((n, n), dtype=complex)
    for a in range(n):
        sa = [1 - 2 * ((a >> i) & 1) for i in range(L)]
        for b in range(n):
            sb = [1 - 2 * ((b >> i) & 1) for i in range(L)]
            e = sum(0.5 * K * sa[i] * sa[(i + 1) % L] + K * sa[i] * sb[i]
                    + 0.5 * K * sb[i] * sb[(i + 1) % L] for i in range(L))
            M[a, b] = np.exp(e + 0.5 * (h / T) * (sum(sa) + sum(sb)))
    return M


A = np.asarray(transfer_matrix(3, 2.0, 0.3))
assert A.shape == (8, 8)
# K = 0.5 and h / T = 0.15: all up to all up, all up to all down, all down to all down
assert abs(A[0, 0] / np.exp(3.45) - 1.0) < 1e-10
assert abs(A[0, 7] - 1.0) < 1e-10
assert abs(A[7, 7] / np.exp(2.55) - 1.0) < 1e-10
assert np.allclose(A, A.T, rtol=1e-10, atol=0.0)
assert np.allclose(A, _t_brute(3, 2.0, 0.3), rtol=1e-10, atol=0.0)

# --- test case 1 ---
import numpy as np

def _t_brute(L, T, h):
    # entry by entry from the energy; the in-row bonds and the field of each row get half weight
    K = 1.0 / T
    n = 2 ** L
    M = np.empty((n, n), dtype=complex)
    for a in range(n):
        sa = [1 - 2 * ((a >> i) & 1) for i in range(L)]
        for b in range(n):
            sb = [1 - 2 * ((b >> i) & 1) for i in range(L)]
            e = sum(0.5 * K * sa[i] * sa[(i + 1) % L] + K * sa[i] * sb[i]
                    + 0.5 * K * sb[i] * sb[(i + 1) % L] for i in range(L))
            M[a, b] = np.exp(e + 0.5 * (h / T) * (sum(sa) + sum(sb)))
    return M


# purely imaginary field: K = 0.4, h / T = 0.08j
A = np.asarray(transfer_matrix(4, 2.5, 0.2j))
assert A.shape == (16, 16)
assert abs(A[0, 0] / np.exp(3.2 + 0.32j) - 1.0) < 1e-10
assert abs(A[5, 0] - np.exp(0.16j)) < 1e-10          # row 5 = spins (-, +, -, +)
assert abs(A[0, 15] - 1.0) < 1e-10
assert np.allclose(A, A.T, rtol=1e-10, atol=0.0)
assert np.allclose(A, _t_brute(4, 2.5, 0.2j), rtol=1e-10, atol=0.0)

# --- test case 2 ---
import numpy as np

def _t_brute(L, T, h):
    # entry by entry from the energy; the in-row bonds and the field of each row get half weight
    K = 1.0 / T
    n = 2 ** L
    M = np.empty((n, n), dtype=complex)
    for a in range(n):
        sa = [1 - 2 * ((a >> i) & 1) for i in range(L)]
        for b in range(n):
            sb = [1 - 2 * ((b >> i) & 1) for i in range(L)]
            e = sum(0.5 * K * sa[i] * sa[(i + 1) % L] + K * sa[i] * sb[i]
                    + 0.5 * K * sb[i] * sb[(i + 1) % L] for i in range(L))
            M[a, b] = np.exp(e + 0.5 * (h / T) * (sum(sa) + sum(sb)))
    return M


A = np.asarray(transfer_matrix(6, 1.7))
assert A.shape == (64, 64)
assert abs(A[63, 21] - 1.0) < 1e-10                   # all down against (-, +, -, +, -, +)
assert np.allclose(A, _t_brute(6, 1.7, 0.0), rtol=1e-10, atol=0.0)
