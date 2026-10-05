import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.integrate import solve_ivp


def _s6_E(i, j, t, X, a, b):
    # McMurchie-Davidson Hermite expansion coefficients E_t^{ij} for one Cartesian direction
    p = a + b
    q = a * b / p
    if t < 0 or t > i + j:
        return 0.0
    if i == 0 and j == 0 and t == 0:
        return np.exp(-q * X * X)
    if j == 0:
        return (_s6_E(i - 1, j, t - 1, X, a, b) / (2 * p) - q * X / a * _s6_E(i - 1, j, t, X, a, b)
                + (t + 1) * _s6_E(i - 1, j, t + 1, X, a, b))
    return (_s6_E(i, j - 1, t - 1, X, a, b) / (2 * p) + q * X / b * _s6_E(i, j - 1, t, X, a, b)
            + (t + 1) * _s6_E(i, j - 1, t + 1, X, a, b))


def _s6_R(t, u, v, n, p, PC, F):
    # Hermite Coulomb integrals R_tuv^n, F[n] = F_n(p |PC|^2)
    if t < 0 or u < 0 or v < 0:
        return 0.0
    if t == 0 and u == 0 and v == 0:
        return (-2.0 * p) ** n * F[n]
    if t > 0:
        return (t - 1) * _s6_R(t - 2, u, v, n + 1, p, PC, F) + PC[0] * _s6_R(t - 1, u, v, n + 1, p, PC, F)
    if u > 0:
        return (u - 1) * _s6_R(t, u - 2, v, n + 1, p, PC, F) + PC[1] * _s6_R(t, u - 1, v, n + 1, p, PC, F)
    return (v - 1) * _s6_R(t, u, v - 2, n + 1, p, PC, F) + PC[2] * _s6_R(t, u, v - 1, n + 1, p, PC, F)


def _s6_overlap(a, la, A, b, lb, B):
    p = a + b
    return np.prod([_s6_E(la[k], lb[k], 0, A[k] - B[k], a, b) for k in range(3)]) * (np.pi / p) ** 1.5


def _s6_kinetic(a, la, A, b, lb, B):
    lb = list(lb)
    # MUTANT: kinetic energy formula of s functions used for every angular momentum
    val = b * 3 * _s6_overlap(a, la, A, b, lb, B)
    for k in range(3):
        up = list(lb)
        up[k] += 2
        val -= 2.0 * b * b * _s6_overlap(a, la, A, b, up, B)
    return val


def _s6_attraction(a, la, A, b, lb, B, C):
    p = a + b
    P = (a * A + b * B) / p
    PC = P - C
    F = boys_function(sum(la) + sum(lb), p * float(PC @ PC))
    val = 0.0
    for t in range(la[0] + lb[0] + 1):
        for u in range(la[1] + lb[1] + 1):
            for v in range(la[2] + lb[2] + 1):
                val += (_s6_E(la[0], lb[0], t, A[0] - B[0], a, b) * _s6_E(la[1], lb[1], u, A[1] - B[1], a, b)
                        * _s6_E(la[2], lb[2], v, A[2] - B[2], a, b) * _s6_R(t, u, v, 0, p, PC, F))
    return 2.0 * np.pi / p * val


def _s6_repulsion(a, la, A, b, lb, B, c, lc, C, d, ld, D):
    p, q = a + b, c + d
    al = p * q / (p + q)
    P, Q = (a * A + b * B) / p, (c * C + d * D) / q
    PQ = P - Q
    F = boys_function(sum(la) + sum(lb) + sum(lc) + sum(ld), al * float(PQ @ PQ))
    E1 = [[_s6_E(la[k], lb[k], t, A[k] - B[k], a, b) for t in range(la[k] + lb[k] + 1)] for k in range(3)]
    E2 = [[_s6_E(lc[k], ld[k], t, C[k] - D[k], c, d) for t in range(lc[k] + ld[k] + 1)] for k in range(3)]
    val = 0.0
    for t, ex in enumerate(E1[0]):
        for u, ey in enumerate(E1[1]):
            for v, ez in enumerate(E1[2]):
                for tau, fx in enumerate(E2[0]):
                    for nu, fy in enumerate(E2[1]):
                        for phi, fz in enumerate(E2[2]):
                            val += (ex * ey * ez * fx * fy * fz * (-1) ** (tau + nu + phi)
                                    * _s6_R(t + tau, u + nu, v + phi, 0, al, PQ, F))
    return 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q)) * val


def polarized_energies(ZA, ZB, zetaA, zetaB, alphaA, alphaB, R):
    '''Full-CI and RHF energies of a two-electron diatomic in the STO-3G 1s + p-shell basis.

    Inputs:
      ZA, ZB: float, nuclear charges, 1 <= Z <= 3 (A at the origin, B at (0, 0, R)).
      zetaA, zetaB: float, Slater exponents of the STO-3G 1s functions, 0.5 <= zeta <= 3.
      alphaA, alphaB: float, exponents of the p shells on A and B, 0.1 <= alpha <= 5.
      R: float, distance between the nuclei in bohr, 0.3 <= R <= 10.

    Output:
      (E_fci, E_rhf): tuple of two Python floats, total energies in hartree (electronic energy
      plus ZA * ZB / R) of the singlet ground state in the 8-function basis, full CI and
      closed-shell RHF, absolute errors below 1e-9.
    '''
    # Basis: on each atom the STO-3G 1s function of steps 2-3 and a normalized primitive
    # Cartesian p shell (x - X) exp(-alpha |r - X|^2), (y - Y) ..., (z - Z) ...
    alpha_1s = np.array([0.109818, 0.405771, 2.22766])
    d_1s = np.array([0.444635, 0.535328, 0.154329])
    basis = []
    for cen, z, ap in ((np.zeros(3), zetaA, alphaA), (np.array([0.0, 0.0, float(R)]), zetaB, alphaB)):
        e = alpha_1s * z * z
        basis.append([(ee, cc, (0, 0, 0), cen) for ee, cc in zip(e, d_1s * (2 * e / np.pi) ** 0.75)])
        for ang in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            basis.append([(ap, (2 * ap / np.pi) ** 0.75 * 2.0 * np.sqrt(ap), ang, cen)])
    nb = len(basis)
    nuclei = ((float(ZA), np.zeros(3)), (float(ZB), np.array([0.0, 0.0, float(R)])))
    S = np.zeros((nb, nb))
    H = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(i + 1):
            s = h = 0.0
            for (a, ca, la, A) in basis[i]:
                for (b, cb, lb, B) in basis[j]:
                    s += ca * cb * _s6_overlap(a, la, A, b, lb, B)
                    h += ca * cb * _s6_kinetic(a, la, A, b, lb, B)
                    for Z, C in nuclei:
                        h -= Z * ca * cb * _s6_attraction(a, la, A, b, lb, B, C)
            S[i, j] = S[j, i] = s
            H[i, j] = H[j, i] = h
    G = np.zeros((nb,) * 4)
    for i in range(nb):
        for j in range(i + 1):
            for k in range(nb):
                for l in range(k + 1):
                    if i * (i + 1) // 2 + j < k * (k + 1) // 2 + l:
                        continue
                    val = 0.0
                    for (a, ca, la, A) in basis[i]:
                        for (b, cb, lb, B) in basis[j]:
                            for (c, cc, lc, C) in basis[k]:
                                for (d, cd, ld, D) in basis[l]:
                                    val += ca * cb * cc * cd * _s6_repulsion(a, la, A, b, lb, B, c, lc, C, d, ld, D)
                    for w, x, y, zz in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                                        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                        G[w, x, y, zz] = val
    enuc = ZA * ZB / R
    s_val, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T
    # RHF: the closed-shell determinant of lowest total energy, i.e. the global minimum of
    # E(x) = 2 x.h.x + (xx|xx) over normalized orbitals x in the orthonormal basis. The orbital
    # with the lowest Fock eigenvalue (aufbau) is not always the one of lowest total energy here:
    # a diffuse p shell can give a lower orbital energy but a higher total energy. So E is
    # minimized directly on the unit sphere (projected gradient with backtracking) from many
    # starting orbitals, and the best minima are polished by a maximum-overlap SCF.
    h_o = X.T @ H @ X
    g_o = np.einsum('pi,qj,rk,sl,pqrs->ijkl', X, X, X, X, G)

    def e_of(x):
        return float(2.0 * (x @ h_o @ x) + np.einsum('i,j,k,l,ijkl->', x, x, x, x, g_o))

    def fock(x):
        return h_o + np.einsum('k,l,ijkl->ij', x, x, g_o)     # closed shell: J - K/2 with P = 2 x x

    rng = np.random.default_rng(12345)
    starts = list(np.linalg.eigh(h_o)[1].T) + list(rng.normal(size=(40, nb)))
    found = []
    for x in starts:
        x = x / np.linalg.norm(x)
        e = e_of(x)
        step = 0.1
        for _ in range(400):
            grad = 4.0 * fock(x) @ x
            grad -= (grad @ x) * x
            if np.linalg.norm(grad) < 1e-9:
                break
            while step > 1e-12:
                y = x - step * grad
                y /= np.linalg.norm(y)
                ey = e_of(y)
                if ey < e - 1e-4 * step * (grad @ grad):
                    x, e = y, ey
                    step *= 2.0
                    break
                step *= 0.5
        found.append((e, x))
    found.sort(key=lambda z: z[0])
    E_rhf = np.inf
    for e, x in found[:4]:
        for _ in range(500):            # maximum-overlap SCF: follow the eigenvector closest to x
            w, V = np.linalg.eigh(fock(x))
            j = int(np.argmax(np.abs(V.T @ x)))
            y = V[:, j] * np.sign(V[:, j] @ x)
            if np.linalg.norm(y - x) < 1e-13:
                x = y
                break
            x = y
        E_rhf = min(E_rhf, e, e_of(x))
    E_rhf += enuc
    # Full CI: two-electron Hamiltonian in the orthonormal product basis chi_i(1) chi_j(2)
    h = X.T @ H @ X
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', X, X, X, X, G)
    I = np.eye(nb)
    Hp = (np.einsum('ik,jl->ijkl', h, I) + np.einsum('ik,jl->ijkl', I, h)
          + np.einsum('ikjl->ijkl', g)).reshape(nb * nb, nb * nb)
    # restrict to spatial functions symmetric under exchange of the electrons (the singlet):
    # in a finite basis the lowest antisymmetric (triplet) state can lie slightly lower
    pairs = [(i, j) for i in range(nb) for j in range(i, nb)]
    B = np.zeros((nb * nb, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        if i == j:
            B[i * nb + i, col] = 1.0
        else:
            B[i * nb + j, col] = B[j * nb + i, col] = np.sqrt(0.5)
    E_fci = float(np.linalg.eigvalsh(B.T @ Hp @ B)[0]) + enuc
    result = (float(E_fci), float(E_rhf))
    return result
