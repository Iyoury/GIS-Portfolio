import numpy as np
from scipy.optimize import brentq


def transfer_matrix(L, T, h=0.0):
    '''Symmetric row-to-row transfer matrix of the 2D Ising model on a periodic strip in a field h.

    Inputs:
      L: int, width of the strip in spins, L >= 3. Spin L is the same as spin 0.
      T: float, temperature, T > 0 (J = 1, k_B = 1).
      h: float or complex, uniform field; the energy is E = - sum_<ij> s_i s_j - h sum_i s_i.
         A complex h, such as 1j * H, is allowed.

    Output:
      M: numpy array of shape (2**L, 2**L), complex when h is complex. Row label
         n = sum_i b_i * 2**i, with b_i = 0 for spin i = +1 and b_i = 1 for spin i = -1.
         M[n, m] is the Boltzmann weight that joins row n to row m, with the bonds inside a
         row and the field on a row shared equally between the two factors that touch that
         row, so that M == M.T. Relative accuracy 1e-10 for every entry.
    '''
    # Bonds between rows: Kronecker product of one 2x2 bond matrix per column. A row's own weight
    # (in-row bonds from the domain walls found by a bit rotation, field from the number of down
    # spins) enters as a diagonal factor on each side.
    K = 1.0 / T
    bond = np.array([[np.exp(K), np.exp(-K)], [np.exp(-K), np.exp(K)]])
    between = np.ones((1, 1))
    for _ in range(L):
        between = np.kron(between, bond)
    labels = np.arange(2 ** L)
    rotated = (labels >> 1) | ((labels & 1) << (L - 1))
    walls = np.array([bin(int(x)).count("1") for x in labels ^ rotated])
    down = np.array([bin(int(x)).count("1") for x in labels])
    own = K * (L - 2 * walls) + (h / T) * (L - 2 * down)
    half = np.exp(0.5 * own)
    M = half[:, None] * between * half[None, :]
    return M


def strip_lengths(L, T):
    '''Free energy per spin and the spin and energy correlation lengths of an infinitely long periodic strip.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10.
      T: float, temperature, 1 <= T <= 10 (J = 1, k_B = 1, zero field).

    Output:
      (f, xi_spin, xi_energy): tuple of three floats.
        f: free energy per spin (units of J), relative accuracy 1e-10.
        xi_spin: decay length of <s_0(row 0) s_0(row r)> along the strip (lattice spacings).
        xi_energy: decay length of the connected correlation of e = s_0 s_1 along the strip.
        Both lengths with relative accuracy 1e-6.
    '''
    # Exact free-fermion (Kaufman) spectrum of the periodic strip; no matrix at all.
    K = 1.0 / T
    K_dual = -0.5 * np.log(np.tanh(K))
    q = np.arange(2 * L)
    ch = (np.cosh(2 * K) * np.cosh(2 * K_dual)
          - np.sinh(2 * K) * np.sinh(2 * K_dual) * np.cos(np.pi * q / L))
    gam = np.arccosh(np.maximum(ch, 1.0))
    gam[0] = 2.0 * (K - K_dual)
    log_lam0 = 0.5 * L * np.log(2.0 * np.sinh(2.0 * K)) + 0.5 * np.sum(gam[1::2])
    f = float(-T * log_lam0 / L)
    # top odd state: modes of even q; the gap is half the alternating sum of the mode energies
    xi_spin = float(1.0 / (0.5 * np.sum(gam[1::2] - gam[0::2])))
    # second even state: the two lowest modes q = 1 and q = 2L - 1 are flipped together
    xi_energy = float(1.0 / (2.0 * gam[1]))
    return f, xi_spin, xi_energy


def strip_magnetization(L, T):
    '''Amplitude m_L of the slowest-decaying term of the spin correlation along a periodic strip.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10.
      T: float, temperature, 0.5 <= T <= 10 (J = 1, k_B = 1, zero field).

    Output:
      m_L: float >= 0, defined for an infinitely long strip by
           <s_0(row 0) s_0(row r)> = m_L**2 * exp(-r / xi_spin) + (terms that decay faster).
           Absolute accuracy 1e-9.
    '''
    # Top two eigenvectors of the whole matrix, then rotated inside their two-dimensional span into
    # states of definite spin-flip parity (below Tc the pair is degenerate to machine precision).
    M = transfer_matrix(L, T)
    n = 2 ** L
    w, v = np.linalg.eigh(M)
    V = v[:, -2:]
    flip = np.arange(n) ^ (n - 1)
    P = V.T @ V[flip, :]
    pw, pv = np.linalg.eigh(0.5 * (P + P.T))
    even = V @ pv[:, np.argmax(pw)]
    odd = V @ pv[:, np.argmin(pw)]
    s0 = 1 - 2 * (np.arange(n) & 1)
    m_L = abs(float(np.sum(even * s0 * odd)))
    return m_L


def strip_susceptibility(L, T):
    '''Zero-field magnetic susceptibility per spin of an infinitely long periodic strip.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 10.
      T: float, temperature, 1 <= T <= 10 (J = 1, k_B = 1).

    Output:
      chi: float, chi = - d^2 f / d h^2 at h = 0, where f(h) is the free energy per spin of
           the infinitely long strip in the uniform field h of transfer_matrix (units 1/J).
           Relative accuracy 1e-6.
    '''
    # Sum over r done by a linear solve in the odd sector:
    # chi = (1 / (T L)) a . (lam0 + M_odd) (lam0 - M_odd)^(-1) a, with a = S |0> projected on it.
    M = transfer_matrix(L, T)
    n = 2 ** L
    labels = np.arange(n)
    flip = labels ^ (n - 1)
    rep = labels[labels < flip]
    cols = np.arange(rep.size)
    P_even = np.zeros((n, rep.size))
    P_odd = np.zeros((n, rep.size))
    P_even[rep, cols] = P_even[flip[rep], cols] = 1.0 / np.sqrt(2.0)
    P_odd[rep, cols] = 1.0 / np.sqrt(2.0)
    P_odd[flip[rep], cols] = -1.0 / np.sqrt(2.0)
    w, v = np.linalg.eigh(P_even.T @ M @ P_even)
    lam0 = w[-1]
    psi0 = P_even @ v[:, -1]
    M_odd = P_odd.T @ M @ P_odd
    S = np.sum(1 - 2 * ((labels[:, None] >> np.arange(L)[None, :]) & 1), axis=1)
    a = P_odd.T @ (S * psi0)
    x = np.linalg.solve(lam0 * np.eye(rep.size) - M_odd, a)
    chi = float(a @ (lam0 * x + M_odd @ x) / (T * L))
    return chi


def lee_yang_edge(L, T):
    '''Yang-Lee edge of an infinitely long periodic strip in a purely imaginary field.

    Inputs:
      L: int, width of the strip in spins, 3 <= L <= 8.
      T: float, temperature, 1.5 <= T <= 10 (J = 1, k_B = 1).

    Output:
      H_edge: float, the smallest H > 0 at which the two eigenvalues of largest modulus of
              transfer_matrix(L, T, 1j * H) are equal (units of J). Relative accuracy 1e-8.
    '''
    # Bisection on whether the two leading eigenvalues have left the real axis.
    def split(H):
        w = np.linalg.eigvals(transfer_matrix(L, T, 1j * H))
        top = w[np.argsort(-np.abs(w))[:2]]
        return bool(np.max(np.abs(top.imag)) > 1e-9 * np.abs(top[0]))

    lo = hi = 1e-12 * T
    while not split(hi):
        lo, hi = hi, 2.0 * hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if split(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-15 * hi:
            break
    H_edge = float(0.5 * (lo + hi))
    return H_edge


def critical_exponents(L):
    '''Finite-size estimates of critical exponents and of the central charge at the exact Tc.

    Input:
      L: int, strip width, 3 <= L <= 7.

    Output:
      dict with exactly the keys "x_sigma", "x_energy", "beta_over_nu", "gamma_over_nu", "y_h"
      and "c", each a float within 1e-4 of the value defined in the prompt. Every quantity is
      taken at T = Tc = 2 / ln(1 + sqrt(2)):
        x_sigma, x_energy: scaling dimensions from Cardy's relation (correlation length
          W / (2 pi x) on a periodic strip of width W) at the single width L;
        beta_over_nu, gamma_over_nu, y_h: two-width effective exponents of the power laws
          m_W ~ W**(-beta/nu), chi_W ~ W**(gamma/nu), H_edge(W) ~ W**(-y_h) between W = L, L+1;
        c: central charge from f_W = f_inf - pi c Tc / (6 W**2) + d / W**4 held exactly for
          W = L, L+1, L+2.
    '''
    # Same estimators; c by eliminating f_inf and d with divided differences in u = 1 / W**2.
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    r = np.log((L + 1.0) / L)
    f1, xs, xe = strip_lengths(L, Tc)
    f2 = strip_lengths(L + 1, Tc)[0]
    f3 = strip_lengths(L + 2, Tc)[0]
    u1, u2, u3 = 1.0 / L ** 2, 1.0 / (L + 1) ** 2, 1.0 / (L + 2) ** 2
    d12 = (f2 - f1) / (u2 - u1)
    d23 = (f3 - f2) / (u3 - u2)
    curv = (d23 - d12) / (u3 - u1)
    slope = d12 - curv * (u1 + u2)
    m = [strip_magnetization(W, Tc) for W in (L, L + 1)]
    chi = [strip_susceptibility(W, Tc) for W in (L, L + 1)]
    edge = [lee_yang_edge(W, Tc) for W in (L, L + 1)]
    result = {
        "x_sigma": float(L / (2.0 * np.pi * xs)),
        "x_energy": float(L / (2.0 * np.pi * xe)),
        "beta_over_nu": float((np.log(m[0]) - np.log(m[1])) / r),
        "gamma_over_nu": float((np.log(chi[1]) - np.log(chi[0])) / r),
        "y_h": float((np.log(edge[0]) - np.log(edge[1])) / r),
        "c": float(-6.0 * slope / (np.pi * Tc)),
    }
    return result


def _v7_vumps(T, h, chi=16, start=None):
    # Variational uniform MPS (VUMPS) for the fixed point of the row-to-row transfer matrix of the
    # infinitely wide lattice. Same bond splitting exp(K s s') = sum_k W[s, k] W[s', k] as a
    # four-leg site tensor O[left, right, up, down]. Returns f, m and the state for warm starts.
    K = 1.0 / T
    W = np.array([[np.sqrt(np.cosh(K)), np.sqrt(np.sinh(K))],
                  [np.sqrt(np.cosh(K)), -np.sqrt(np.sinh(K))]])
    s = np.array([1.0, -1.0])
    weight = np.exp((h / T) * s)
    O = np.einsum("s,sa,sb,su,sd->abud", weight, W, W, W, W)
    O_spin = np.einsum("s,s,sa,sb,su,sd->abud", s, weight, W, W, W, W)
    n = 2 * chi * chi

    def polar(X):
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        return U @ Vt

    def fixed_point(Mat, x):
        # power iteration from the previous environment (the map is positive)
        for _ in range(5000):
            y = Mat @ x
            y /= np.linalg.norm(y)
            if np.linalg.norm(y - x) < 1e-14:
                return y
            x = y
        return x

    def top_symmetric(Mat):
        w, v = np.linalg.eigh(0.5 * (Mat + Mat.T))
        x = v[:, -1]
        return x / (np.sign(np.sum(x)) * np.linalg.norm(x))

    if start is None:
        rng = np.random.default_rng(7)
        AC = rng.random((chi, 2, chi)) + 0.1
        C = np.diag(np.linspace(1.0, 0.1, chi))
        AL = polar(AC.reshape(2 * chi, chi)).reshape(chi, 2, chi)
        AR = polar(AC.reshape(chi, 2 * chi)).reshape(chi, 2, chi)
        FL = np.ones(n) / np.sqrt(n)
        FR = np.ones(n) / np.sqrt(n)
    else:
        AL, AR, FL, FR = start
    for _ in range(500):
        FL = fixed_point(np.einsum("xsp,abst,ytq->pbqxay", AL, O, AL).reshape(n, n), FL)
        FR = fixed_point(np.einsum("psx,abst,qty->paqxby", AR, O, AR).reshape(n, n), FR)
        L3, R3 = FL.reshape(chi, 2, chi), FR.reshape(chi, 2, chi)
        AC = top_symmetric(np.einsum("xaq,abst,ybp->qtpxsy", L3, O, R3).reshape(n, n)).reshape(chi, 2, chi)
        C = top_symmetric(np.einsum("xaq,yap->qpxy", L3, R3).reshape(chi * chi, chi * chi)).reshape(chi, chi)
        AL = (polar(AC.reshape(2 * chi, chi)) @ polar(C).T).reshape(chi, 2, chi)
        AR = (polar(C).T @ polar(AC.reshape(chi, 2 * chi))).reshape(chi, 2, chi)
        if np.linalg.norm(AC.reshape(2 * chi, chi) - AL.reshape(2 * chi, chi) @ C) < 1e-13:
            break
    L3, R3 = FL.reshape(chi, 2, chi), FR.reshape(chi, 2, chi)
    num = np.einsum("xaq,xsy,abst,qtp,ybp->", L3, AC, O_spin, AC, R3)
    den = np.einsum("xaq,xsy,abst,qtp,ybp->", L3, AC, O, AC, R3)
    m = float(num / den)
    # per-site eigenvalue of the row transfer matrix = partition function per spin
    norm = np.einsum("xaq,xy,qp,yap->", L3, C, C, R3)
    AC_n = AC / np.sqrt(np.einsum("xsy,xsy->", AC, AC))
    C_n = C / np.sqrt(np.sum(C * C))
    lam = np.einsum("xaq,xsy,abst,qtp,ybp->", L3, AC_n, O, AC_n, R3) / np.einsum("xaq,xy,qp,yap->", L3, C_n, C_n, R3)
    f = float(-T * np.log(lam))
    return f, m, (AL, AR, FL, FR)


def bulk_thermodynamics(T, h):
    '''Free energy, magnetization, susceptibility and specific heat per spin of the infinite lattice in a field.'''
    # Other route: VUMPS instead of corner transfer matrices, the free energy from the per-site
    # eigenvalue of the row transfer matrix, and both response functions as second derivatives
    # of f (fourth-order five-point stencils), chi = -d2f/dh2 instead of dm/dh.
    f0, m0, st = _v7_vumps(T, h)

    def f_at(TT, hh):
        return _v7_vumps(TT, hh, start=st)[0]

    def second(g, x, d):
        return (-g(x + 2 * d) + 16.0 * g(x + d) - 30.0 * g(x) + 16.0 * g(x - d) - g(x - 2 * d)) / (12.0 * d * d)

    vals_h = {0: f0}
    vals_T = {0: f0}
    dh, dT = 2e-4, 2e-3
    for k in (-2, -1, 1, 2):
        vals_h[k] = f_at(T, h + k * dh)
        vals_T[k] = f_at(T + k * dT, h)
    chi = -(-vals_h[2] + 16 * vals_h[1] - 30 * f0 + 16 * vals_h[-1] - vals_h[-2]) / (12 * dh * dh)
    c = -T * (-vals_T[2] + 16 * vals_T[1] - 30 * f0 + 16 * vals_T[-1] - vals_T[-2]) / (12 * dT * dT)
    result = (float(f0), float(m0), float(chi), float(c))
    return result
