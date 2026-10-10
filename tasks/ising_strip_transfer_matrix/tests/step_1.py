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
# symmetry compares two computed entries, each with relative accuracy 1e-10
assert np.max(np.abs(A - A.T) / np.abs(A.T)) <= 2e-10
_t_B = _t_brute(3, 2.0, 0.3)
assert np.max(np.abs(A - _t_B) / np.abs(_t_B)) <= 1e-10

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
# symmetry compares two computed entries, each with relative accuracy 1e-10
assert np.max(np.abs(A - A.T) / np.abs(A.T)) <= 2e-10
_t_B = _t_brute(4, 2.5, 0.2j)
assert np.max(np.abs(A - _t_B) / np.abs(_t_B)) <= 1e-10

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
_t_B = _t_brute(6, 1.7, 0.0)
assert np.max(np.abs(A - _t_B) / np.abs(_t_B)) <= 1e-10

# --- test case 3 ---
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


# lowest temperature of the range: K = 2 and h / T = 0.2
A = np.asarray(transfer_matrix(3, 0.5, 0.1))
assert A.shape == (8, 8)
assert abs(A[0, 0] / np.exp(12.6) - 1.0) < 1e-10
assert abs(A[7, 7] / np.exp(11.4) - 1.0) < 1e-10
assert abs(A[0, 7] - 1.0) < 1e-10
_t_B = _t_brute(3, 0.5, 0.1)
assert np.max(np.abs(A - _t_B) / np.abs(_t_B)) <= 1e-10
# highest temperature and largest imaginary field of the range: K = 0.1 and h / T = 2j
A = np.asarray(transfer_matrix(3, 10.0, 20j))
assert A.shape == (8, 8)
assert abs(A[0, 0] / np.exp(0.6 + 6j) - 1.0) < 1e-10
assert abs(A[7, 7] / np.exp(0.6 - 6j) - 1.0) < 1e-10
assert abs(A[0, 7] - 1.0) < 1e-10
_t_B = _t_brute(3, 10.0, 20j)
assert np.max(np.abs(A - _t_B) / np.abs(_t_B)) <= 1e-10

# --- test case 4 ---
import numpy as np

def _t_energy_all(L, T, h):
    # every entry from the energy, summed column by column of the strip over all pairs of rows
    K = 1.0 / T
    lab = np.arange(2 ** L)
    s = [1 - 2 * ((lab >> i) & 1) for i in range(L)]
    e = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for i in range(L):
        own = 0.5 * K * s[i] * s[(i + 1) % L]
        e += own[:, None] + own[None, :] + K * np.outer(s[i], s[i])
        e += 0.5 * (h / T) * (s[i][:, None] + s[i][None, :])
    return np.exp(e)


# widest strip and lowest temperature of the range: K = 2, 1024 x 1024 matrix
A = np.asarray(transfer_matrix(10, 0.5))
assert A.shape == (1024, 1024)
assert abs(A[0, 0] / np.exp(40.0) - 1.0) < 1e-10        # all up to all up: 10 + 10 + 20
assert abs(A[0, 1] / np.exp(32.0) - 1.0) < 1e-10        # row 1 has spin 0 down: 10 + 6 + 16
assert abs(A[0, 1023] - 1.0) < 1e-10                     # all up to all down
_t_B = _t_energy_all(10, 0.5, 0.0)
assert np.max(np.abs(A - _t_B) / np.abs(_t_B)) <= 1e-10
# largest real field of the range, h = -20: the field term of a row is (h / T) / 2 times its sum of spins
A = np.asarray(transfer_matrix(10, 0.5, -20.0))
assert A.shape == (1024, 1024)
assert abs(A[1023, 1023] / np.exp(440.0) - 1.0) < 1e-10  # all down to all down: 40 + 400
assert abs(A[0, 0] / np.exp(-360.0) - 1.0) < 1e-10      # all up to all up: 40 - 400
assert abs(A[0, 1023] - 1.0) < 1e-10
_t_B = _t_energy_all(10, 0.5, -20.0)
assert np.max(np.abs(A - _t_B) / np.abs(_t_B)) <= 1e-10
# symmetry compares two computed entries, each with relative accuracy 1e-10
assert np.max(np.abs(A - A.T) / np.abs(A.T)) <= 2e-10
