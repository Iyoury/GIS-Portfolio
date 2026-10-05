import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def _s5_atom(Z, zeta):
    # one-centre integrals of the (unrenormalized) STO-3G 1s function on an isolated nucleus Z:
    # norm <phi|phi>, core energy <phi|-1/2 Laplacian - Z/r|phi> and (phi phi|phi phi)
    a = np.array([0.109818, 0.405771, 2.22766]) * zeta ** 2
    d = np.array([0.444635, 0.535328, 0.154329]) * (2.0 * a / np.pi) ** 0.75
    p = a[:, None] + a[None, :]
    dd = d[:, None] * d[None, :]
    s = np.sum(dd * (np.pi / p) ** 1.5)
    h = np.sum(dd * (3.0 * a[:, None] * a[None, :] / p * (np.pi / p) ** 1.5 - 2.0 * np.pi * Z / p))
    P = p[:, :, None, None]
    Q = p[None, None, :, :]
    j = np.sum(dd[:, :, None, None] * dd[None, None, :, :] * 2.0 * np.pi ** 2.5 / (P * Q * np.sqrt(P + Q)))
    return s, h, j


def _s5_fci(ZA, ZB, zetaA, zetaB, R):
    # E_fci(R) of fci_energy (step 4): the lowest singlet of the full-CI matrix. Full CI does not depend
    # on the orthonormal orbitals used, so the symmetric orthogonalization S^(-1/2) replaces the RHF
    # orbitals here (the RHF scan of fci_energy is not needed for the curve).
    S, H = sto3g_one_electron(ZA, ZB, zetaA, zetaB, R)
    eri = sto3g_two_electron(zetaA, zetaB, R)
    s_val, s_vec = np.linalg.eigh(S)
    C = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T
    h = C.T @ H @ C
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C, eri, optimize=True)
    M = np.empty((3, 3))
    M[0, 0] = 2.0 * h[0, 0] + g[0, 0, 0, 0]
    M[1, 1] = 2.0 * h[1, 1] + g[1, 1, 1, 1]
    M[2, 2] = h[0, 0] + h[1, 1] + g[0, 0, 1, 1] + g[0, 1, 0, 1]
    M[0, 1] = M[1, 0] = g[0, 1, 0, 1]
    M[0, 2] = M[2, 0] = np.sqrt(2.0) * (h[0, 1] + g[0, 0, 0, 1])
    M[1, 2] = M[2, 1] = np.sqrt(2.0) * (h[0, 1] + g[1, 1, 0, 1])
    return float(np.linalg.eigvalsh(M)[0] + ZA * ZB / R)


def _s5_inner(V, mu, r0, a, N):
    # Lagrange mesh on [r0, a]: basis phi_i(r) = (s / s_i) l_i(s), s = r - r0, l_i the Lagrange polynomials
    # on the N Gauss-Legendre nodes; phi_i(r0) = 0 and phi_i(a) free. Kinetic matrix int phi_i' phi_j' dr
    # exact (Gauss quadrature of a polynomial of degree 2N - 2), overlap and potential at the Gauss
    # approximation. Eigenpairs (E_n, u_n) of this problem with a free end at a (no Bloch term needed in
    # the weak form) give the R-matrix u(a) / u'(a) = (1 / 2 mu) sum_n u_n(a)**2 / (E_n - E).
    x, w = np.polynomial.legendre.leggauss(N)
    L = a - r0
    s = 0.5 * L * (x + 1.0)
    ws = 0.5 * L * w
    bw = (-1.0) ** np.arange(N) * np.sqrt((1.0 - x * x) * w)     # barycentric weights of the nodes
    diff = s[:, None] - s[None, :]
    np.fill_diagonal(diff, 1.0)
    Dl = (bw[None, :] / bw[:, None]) / diff
    np.fill_diagonal(Dl, 0.0)
    np.fill_diagonal(Dl, -Dl.sum(axis=1))
    D = np.diag(1.0 / s) + (s[:, None] / s[None, :]) * Dl        # D[k, i] = phi_i'(s_k)
    K = D.T @ (ws[:, None] * D)
    sq = np.sqrt(ws)
    Vm = np.array([V(r0 + si) for si in s])
    En, Y = np.linalg.eigh(K / (sq[:, None] * sq[None, :]) / (2.0 * mu) + np.diag(Vm))
    la = (bw / (L - s)) / np.sum(bw / (L - s))
    ua = (Y / sq[:, None]).T @ ((L / s) * la)
    return En, ua


def _s5_outer_nodes(u, up, E, mu, a, Vext, Rb):
    # zeros beyond a of the solution continued outward from (u(a), u'(a)) in the analytic potential; beyond
    # the last classical turning point Rb the integration goes on until int sqrt(2 mu (V - E)) dR = 35,
    # after which the growing solution dominates and no further zero can occur
    from scipy.integrate import solve_ivp

    def f(R, y):
        q = 2.0 * mu * (Vext(R) - E)
        return [y[1], q * y[0], np.sqrt(max(q, 0.0)) * y[3], 0.0]

    def stop(R, y):
        return y[2] - 35.0
    stop.terminal = True

    def zero(R, y):
        return y[0]

    tol = dict(rtol=1e-10, atol=[1e-14, 1e-14, 1e-8, 1.0], method='DOP853')
    y = np.array([u, up, 0.0, 0.0]) / np.hypot(u, up)
    n, R = 0, a
    if Rb > a:
        sol = solve_ivp(f, (a, Rb), y, events=(zero,), **tol)
        if sol.status != 0:
            raise RuntimeError(sol.message)
        n += len(sol.t_events[0])
        R, y = Rb, sol.y[:, -1].copy()
    y[2], y[3] = 0.0, 1.0
    span = max(Rb, 1.0)
    while True:
        y[:2] /= np.hypot(y[0], y[1])
        sol = solve_ivp(f, (R, R + span), y, events=(stop, zero), **tol)
        if sol.status == -1:
            raise RuntimeError(sol.message)
        n += len(sol.t_events[1])
        if sol.status == 1:
            return n
        R, y = sol.t[-1], sol.y[:, -1].copy()
        span *= 2.0


def _s5_levels(En, ua, mu, a, tail, attractive, v_floor):
    # Sturm oscillation theorem: the number of levels below E is the number of zeros on (r0, infinity) of
    # the solution regular at r0. Inside the mesh it follows from the R-matrix: with n_N mesh eigenvalues
    # below E the Pruefer angle at a lies in (n_N pi - pi/2, n_N pi + pi/2], so the inner zeros are
    # n_N - 1 + [u(a) / u'(a) >= 0]; the outer zeros come from the integration above.
    def Vext(R):
        return min(d + c / R for d, c in tail)

    def Rb(E):
        out = a
        for d, c in tail:
            if c < 0.0:
                out = max(out, c / (E - d))
        return out

    def count(E):
        Rm = np.sum(ua ** 2 / (En - E)) / (2.0 * mu)
        nN = int(np.sum(En < E))
        zin = nN - 1 + (1 if Rm >= 0.0 else 0) if nN >= 1 else 0
        return zin + _s5_outer_nodes(Rm, 1.0, E, mu, a, Vext, Rb(E))

    lo = v_floor
    pts = [(lo, 0)]
    top = None
    E = v_floor
    while True:
        E = min(E / 4.0, -1e-12)
        c = count(E)
        pts.append((E, c))
        if c >= 5:
            top = E
            break
        if E == -1e-12:
            break
    levels = []
    for v in range(5):
        below = max((p for p in pts if p[1] <= v), key=lambda p: p[0])
        above = [p for p in pts if p[1] > v]
        if not above:
            # fewer than v + 1 levels below -1e-12 hartree: with an attractive Coulomb tail there are
            # infinitely many, and the missing ones lie in (-1e-12, 0) hartree
            if not attractive:
                return None
            levels.append(-0.5e-12)
            continue
        lo_e, hi_e = below[0], min(above, key=lambda p: p[0])[0]
        while hi_e - lo_e > 1e-11:
            mid = 0.5 * (lo_e + hi_e)
            c = count(mid)
            pts.append((mid, c))
            if c > v:
                hi_e = mid
            else:
                lo_e = mid
        levels.append(0.5 * (lo_e + hi_e))
    return np.array(levels)


def vibrational_levels(ZA, ZB, zetaA, zetaB, massA, massB):
    '''Lowest five vibrational levels (J = 0) on the full-CI potential curve, from the dissociation limit.

    Inputs:
      ZA, ZB: float, nuclear charges, 1 <= Z <= 2.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, 0.8 <= zeta <= 3.
      massA, massB: float, nuclear masses in unified atomic mass units (u), 1 <= mass <= 10.

    Output:
      levels: float numpy array of shape (5,), the energies of the vibrational states
              v = 0..4 in cm^-1, measured from the exact R -> infinity limit of the
              full-CI energy (so all are negative), increasing. Absolute error below
              0.01 cm^-1 for each level.

    Raises:
      ValueError if a charge is not finite or not in [1, 2], if a mass is not finite or not in
      [1, 10] u, or if fewer than five bound vibrational levels lie below the dissociation limit.
    '''
    for z in (ZA, ZB):
        if not (np.isfinite(z) and 1.0 <= z <= 2.0):
            raise ValueError("nuclear charges must be finite and between 1 and 2")
    for m in (massA, massB):
        if not (np.isfinite(m) and 1.0 <= m <= 10.0):
            raise ValueError("nuclear masses must be finite and between 1 and 10 u")
    # Dissociation limit in this basis: as R -> infinity all cross integrals vanish and the lowest singlet
    # is the lowest of the three fragment arrangements (both electrons on A, both on B, one on each), whose
    # energies at large R are E_j + q_A q_B / R. Once the cross integrals have died out (Gaussian decay)
    # the curve is exactly V(R) = min_j (E_j - E_inf + c_j / R), so the radial equation is solved with the
    # computed curve on [r0, a] and this analytic tail on [a, infinity): no outer wall, whatever the extent
    # of the states (near-threshold levels and Rydberg-like levels of a weak -c/R tail reach far out).
    sA, hA, jA = _s5_atom(ZA, zetaA)
    sB, hB, jB = _s5_atom(ZB, zetaB)
    arr = [((2.0 * hA * sA + jA) / sA ** 2, (ZA - 2.0) * ZB), ((2.0 * hB * sB + jB) / sB ** 2, (ZB - 2.0) * ZA),
           (hA / sA + hB / sB, (ZA - 1.0) * (ZB - 1.0))]
    e_inf = min(e for e, c in arr)
    tail = [(e - e_inf, c) for e, c in arr]
    attractive = min(c for d, c in tail if d == 0.0) < 0.0
    mu = massA * massB / (massA + massB) * 1822.888486209
    cache = {}

    def V(R):
        if R not in cache:
            cache[R] = _s5_fci(ZA, ZB, zetaA, zetaB, R) - e_inf
        return cache[R]

    def Vtail(R):
        return min(d + c / R for d, c in tail)

    # start of the analytic tail: the computed curve equals it to 1e-12 hartree at a, a + 1 and a + 2
    a = 8.0
    while max(abs(V(a + k) - Vtail(a + k)) for k in (0.0, 1.0, 2.0)) > 1e-12:
        a += 1.0
    # inner wall: below the first point where V <= 0 the barrier integral int sqrt(2 mu V) dR reaches 15
    # (a level then changes by about exp(-30) of its kinetic energy when the wall moves)
    grid = np.arange(0.1, a, 0.05)
    vals = np.array([V(r) for r in grid])
    neg = np.where(vals <= 0.0)[0]
    r_in = grid[neg[0]] if neg.size else a
    r0, acc = r_in, 0.0
    while acc < 15.0 and r0 > 0.02:
        r0 = max(r0 - 0.01, 0.02)
        acc += 0.01 * np.sqrt(2.0 * mu * max(V(r0), 0.0))
    v_min = min(float(vals.min()), Vtail(a))
    kmax = np.sqrt(2.0 * mu * max(-v_min, 1e-4))
    N = int(np.ceil(kmax * (a - r0))) + 40
    # convergence in the representation: wall moved in, tail start moved out and the mesh refined until the
    # five levels no longer change (no fixed limit on the number of refinements)
    prev = None
    while True:
        En, ua = _s5_inner(V, mu, r0, a, N)
        cur = _s5_levels(En, ua, mu, a, tail, attractive, v_min - 1e-6)
        if cur is None:
            raise ValueError("fewer than five vibrational levels lie below the dissociation limit")
        if prev is not None and np.max(np.abs(cur - prev)) * 219474.6313632 < 1e-4:
            break
        prev = cur
        r0, a, N = max(0.85 * r0, 0.02), a + 3.0, int(np.ceil(1.25 * N))
    levels = cur * 219474.6313632
    return levels
