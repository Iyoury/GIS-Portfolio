import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp


# 20-point Gauss-Legendre rule on [0, 1], used by the composite quadrature of boys_function
_G1_X, _G1_W = np.polynomial.legendre.leggauss(20)
_G1_X, _G1_W = 0.5 * (_G1_X + 1.0), 0.5 * _G1_W


def boys_function(n_max, t):
    '''Boys functions F_n(t) = integral from 0 to 1 of u**(2n) exp(-t u**2) du for n = 0..n_max.'''
    # Other method: direct quadrature of the defining integral (no series, no recursion, no erf,
    # no incomplete gamma function). The integrand u**(2n) exp(-t u**2) is negligible beyond
    # u = 13 / sqrt(t) (below 1e-60 of the integral for every n <= 16), so the integral is taken over
    # [0, min(1, 13 / sqrt(t))], split into 8 equal panels with the 20-point Gauss-Legendre rule each:
    # in v = sqrt(t) u every panel spans at most 1.7 units of a smooth integrand, and for t -> 0 the
    # rule is exact for the polynomial u**(2n).
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)) or not (0 <= n_max <= 16):
        raise ValueError("n_max must be an integer from 0 to 16")
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)) or np.any(t_arr < 0.0) or np.any(t_arr > 1e6):
        raise ValueError("t must be finite with 0 <= t <= 1e6")
    flat = t_arr.ravel()
    top = np.where(flat > 169.0, 13.0 / np.sqrt(np.maximum(flat, 169.0)), 1.0)
    panels = 8
    nodes = ((np.arange(panels)[:, None] + _G1_X[None, :]) / panels).ravel()
    wts = np.tile(_G1_W, panels) / panels
    u = top[:, None] * nodes[None, :]
    u2 = u * u
    term = (top[:, None] * wts[None, :]) * np.exp(-flat[:, None] * u2)
    F = np.empty((flat.size, int(n_max) + 1))
    for n in range(int(n_max) + 1):
        F[:, n] = np.sum(term, axis=1)
        term = term * u2
    return F.reshape(t_arr.shape + (int(n_max) + 1,))


# Second solution, steps 2-3 (and the one-centre integrals of step 5): momentum space instead of the
# closed-form Gaussian integrals of the reference and the real-space shell-theorem quadrature of the
# tests. With g~(k) = (pi / a)**1.5 exp(-k**2 / 4a) exp(-i k.A) the Fourier transform of exp(-a |r - A|**2),
# Parseval's theorem, the Coulomb kernel 4 pi / k**2 and the angular average j0(k d) = sin(k d) / (k d)
# of exp(i k.d), every integral becomes a radial integral over |k| done by quadrature (no error function,
# no Boys function):
#   <a|b>                = N_a N_b (pi**2 / ab)**1.5 / (2 pi**2) int k**2 exp(-k**2 (a + b) / 4ab) j0(k |A - B|) dk
#   <a|-Laplacian / 2|b> = the same with k**4 / 2 in place of k**2
#   <a|1 / |r - C||b>    = N_a N_b K_ab (pi / p)**1.5 (2 / pi) int exp(-k**2 / 4p) j0(k |P - C|) dk
#   (ab|cd)              = N_a N_b N_c N_d K_ab K_cd (pi**2 / pq)**1.5 (2 / pi)
#                          int exp(-k**2 (1 / 4p + 1 / 4q)) j0(k |P - Q|) dk
# (K_ab, p and P of the Gaussian product of a and b; only the charge clouds use the product theorem).

# 16-point Gauss-Legendre rule on [0, 1], the panel rule of the momentum-space radial integrals
_K_X, _K_W = np.polynomial.legendre.leggauss(16)
_K_X, _K_W = 0.5 * (_K_X + 1.0), 0.5 * _K_W
_K_TOP = np.sqrt(60.0)        # x**4 exp(-x**2) is below 1e-23 of its maximum beyond x = sqrt(60)


def _k_radial(power, gamma, d):
    # J = int_0^infinity k**power exp(-gamma k**2) j0(k d) dk for arrays gamma > 0 and d >= 0 of the same
    # shape. With k = x / sqrt(gamma), J = gamma**(-(power + 1) / 2) int_0^sqrt(60) x**power exp(-x**2)
    # j0(x delta) dx, delta = d / sqrt(gamma), done with composite 16-point Gauss-Legendre on panels no
    # wider than 0.5 and than half an oscillation period pi / delta.
    gamma = np.asarray(gamma, float).ravel()
    d = np.asarray(d, float).ravel()
    delta = d / np.sqrt(gamma)
    n_pan = np.maximum(16, np.ceil(_K_TOP * delta / np.pi)).astype(int)
    out = np.empty(gamma.size)
    for n in np.unique(n_pan):
        sel = np.where(n_pan == n)[0]
        x = ((np.arange(n)[:, None] + _K_X[None, :]) / n).ravel() * _K_TOP
        w = np.tile(_K_W, n) * (_K_TOP / n)
        out[sel] = np.sinc(np.outer(delta[sel], x) / np.pi) @ (w * x ** power * np.exp(-x * x))
    return out * gamma ** (-0.5 * (power + 1))


def _k_basis(zeta):
    # exponents and normalized contraction coefficients of the STO-3G 1s function
    a = np.array([0.109818, 0.405771, 2.22766]) * zeta ** 2
    return a, np.array([0.444635, 0.535328, 0.154329]) * (2.0 * a / np.pi) ** 0.75


def _k_atom(Z, zeta):
    # one-centre norm <phi|phi>, core energy <phi|-Laplacian/2 - Z/r|phi> and (phi phi|phi phi) of the
    # STO-3G 1s function on a nucleus Z, from the same momentum-space radial integrals at zero separation
    a, cf = _k_basis(zeta)
    A = a[:, None] * np.ones(3)[None, :]
    B = a[None, :] * np.ones(3)[:, None]
    w = cf[:, None] * cf[None, :]
    zero = np.zeros((3, 3))
    beta = (A + B) / (4.0 * A * B)
    pre = (np.pi ** 2 / (A * B)) ** 1.5 / (2.0 * np.pi ** 2)
    s = np.sum(w * pre * _k_radial(2, beta, zero).reshape(3, 3))
    p = A + B
    h = np.sum(w * (0.5 * pre * _k_radial(4, beta, zero).reshape(3, 3)
                    - Z * (np.pi / p) ** 1.5 * (2.0 / np.pi) * _k_radial(0, 1.0 / (4.0 * p), zero).reshape(3, 3)))
    P = p.ravel()[:, None] * np.ones(9)[None, :]
    Q = p.ravel()[None, :] * np.ones(9)[:, None]
    ww = w.ravel()[:, None] * w.ravel()[None, :]
    j = np.sum(ww * (np.pi ** 2 / (P * Q)) ** 1.5 * (2.0 / np.pi)
               * _k_radial(0, 0.25 / P + 0.25 / Q, np.zeros((9, 9))).reshape(9, 9))
    return s, h, j


def sto3g_one_electron(ZA, ZB, zetaA, zetaB, R):
    '''Overlap and core-Hamiltonian matrices for two atoms with one STO-3G 1s function each.

    Inputs:
      ZA, ZB: float, nuclear charges of atom A (at the origin) and atom B (at distance R on z),
              1 <= Z <= 3.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, 0.5 <= zeta <= 3.
      R: float, distance between the nuclei in bohr, 0.02 <= R <= 100.

    Output:
      (S, H): two float numpy arrays of shape (2, 2). Index 0 is the function on A and
      index 1 the function on B. S is the overlap matrix and H = T + V is the core
      Hamiltonian (kinetic energy plus attraction to both nuclei), in hartree.
      Absolute error below 1e-10 for every entry.

    Raises:
      ValueError if R, ZA, ZB, zetaA or zetaB is not positive.
    '''
    # Other method: momentum space (see _k_radial above), no closed-form Gaussian integrals and no
    # real-space quadrature.
    if not (R > 0 and ZA > 0 and ZB > 0 and zetaA > 0 and zetaB > 0):
        raise ValueError("R, the nuclear charges and the Slater exponents must be positive")
    centers = np.array([0.0, float(R)])
    charges = np.array([float(ZA), float(ZB)])
    ex, cf = zip(_k_basis(zetaA), _k_basis(zetaB))
    S = np.zeros((2, 2))
    H = np.zeros((2, 2))
    for m in range(2):
        for n in range(2):
            a = ex[m][:, None] * np.ones(3)[None, :]
            b = ex[n][None, :] * np.ones(3)[:, None]
            w = cf[m][:, None] * cf[n][None, :]
            dist = abs(centers[m] - centers[n]) * np.ones((3, 3))
            beta = (a + b) / (4.0 * a * b)
            pre = (np.pi ** 2 / (a * b)) ** 1.5 / (2.0 * np.pi ** 2)
            ovl = pre * _k_radial(2, beta, dist).reshape(3, 3)
            kin = 0.5 * pre * _k_radial(4, beta, dist).reshape(3, 3)
            p = a + b
            P = (a * centers[m] + b * centers[n]) / p
            att = np.zeros((3, 3))
            for C in range(2):
                att -= charges[C] * _k_radial(0, 1.0 / (4.0 * p), np.abs(P - centers[C])).reshape(3, 3)
            att *= np.exp(-a * b / p * dist ** 2) * (np.pi / p) ** 1.5 * (2.0 / np.pi)
            S[m, n] = np.sum(w * ovl)
            H[m, n] = np.sum(w * (kin + att))
    return S, H


def sto3g_two_electron(zetaA, zetaB, R):
    '''Two-electron repulsion integrals over the two STO-3G 1s functions.

    Inputs:
      zetaA, zetaB: float, Slater exponents of the 1s functions on A (origin) and B, 0.5 <= zeta <= 3.
      R: float, distance between the nuclei in bohr, 0.02 <= R <= 100.

    Output:
      eri: float numpy array of shape (2, 2, 2, 2), in hartree, with
           eri[i, j, k, l] = (ij|kl) = integral of phi_i(1) phi_j(1) (1/r12) phi_k(2) phi_l(2).
           Absolute error below 1e-10 for every entry.

    Raises:
      ValueError if R, zetaA or zetaB is not positive.
    '''
    # Other method: momentum space (see _k_radial above). Only the 6 distinct integrals are computed.
    if not (R > 0 and zetaA > 0 and zetaB > 0):
        raise ValueError("R and the Slater exponents must be positive")
    centers = np.array([0.0, float(R)])
    ex, cf = zip(_k_basis(zetaA), _k_basis(zetaB))
    eri = np.zeros((2, 2, 2, 2))
    done = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    key = tuple(sorted([tuple(sorted((i, j))), tuple(sorted((k, l)))]))
                    if key not in done:
                        a = ex[i][:, None, None, None]
                        b = ex[j][None, :, None, None]
                        c = ex[k][None, None, :, None]
                        d = ex[l][None, None, None, :]
                        w = (cf[i][:, None, None, None] * cf[j][None, :, None, None]
                             * cf[k][None, None, :, None] * cf[l][None, None, None, :])
                        p = a + b
                        q = c + d
                        P = (a * centers[i] + b * centers[j]) / p
                        Q = (c * centers[k] + d * centers[l]) / q
                        kk = (np.exp(-a * b / p * (centers[i] - centers[j]) ** 2)
                              * np.exp(-c * d / q * (centers[k] - centers[l]) ** 2))
                        shape = (3, 3, 3, 3)
                        rad = _k_radial(0, np.broadcast_to(0.25 / p + 0.25 / q, shape),
                                        np.broadcast_to(np.abs(P - Q), shape)).reshape(shape)
                        done[key] = float(np.sum(w * kk * (np.pi ** 2 / (p * q)) ** 1.5 * (2.0 / np.pi) * rad))
                    eri[i, j, k, l] = done[key]
    return eri


def fci_energy(ZA, ZB, zetaA, zetaB, R):
    '''Singlet ground-state full-CI and closed-shell RHF energies of a two-electron diatomic in STO-3G.'''
    # Other methods. RHF without a scan or an iteration: every normalized orbital is X (cos theta,
    # sin theta) with X = S^(-1/2), and E(theta) = 2 c.H.c + (cc|cc) is then a trigonometric polynomial
    # with harmonics 0, 2 and 4 only: E = a0 + a1 cos phi + b1 sin phi + a2 cos 2 phi + b2 sin 2 phi in
    # phi = 2 theta. Its coefficients follow exactly from 8 samples (discrete Fourier transform), and its
    # stationary points are the arguments of the roots of the quartic polynomial 2 z**2 dE/dphi in
    # z = exp(i phi); the global minimum is the lowest value of E at these points (each polished by Newton
    # steps on dE/dphi). Full CI: the whole 4-dimensional two-electron product space of the RHF orbitals
    # (singlets and the triplet) with a spin penalty instead of a restriction to singlet functions.
    S, H = sto3g_one_electron(ZA, ZB, zetaA, zetaB, R)
    eri = sto3g_two_electron(zetaA, zetaB, R)
    s_val, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T

    def orbital(theta):
        return X @ np.array([np.cos(theta), np.sin(theta)])

    def energy(theta):
        c = orbital(theta)
        return float(2.0 * (c @ H @ c) + np.einsum('i,j,k,l,ijkl->', c, c, c, c, eri))

    phis = 2.0 * np.pi * np.arange(8) / 8.0
    samp = np.array([energy(0.5 * f) for f in phis])
    a1, b1 = 2.0 * np.mean(samp * np.cos(phis)), 2.0 * np.mean(samp * np.sin(phis))
    a2, b2 = 2.0 * np.mean(samp * np.cos(2.0 * phis)), 2.0 * np.mean(samp * np.sin(2.0 * phis))
    poly = np.array([2.0 * b2 + 2.0j * a2, b1 + 1.0j * a1, 0.0, b1 - 1.0j * a1, 2.0 * b2 - 2.0j * a2])
    cands = list(phis)
    if np.max(np.abs(poly)) > 0.0:
        cands += list(np.angle(np.roots(poly)))
    best_e, best_t = np.inf, 0.0
    for f in cands:
        for _ in range(3):
            d1 = -a1 * np.sin(f) + b1 * np.cos(f) - 2.0 * a2 * np.sin(2.0 * f) + 2.0 * b2 * np.cos(2.0 * f)
            d2 = -a1 * np.cos(f) - b1 * np.sin(f) - 4.0 * a2 * np.cos(2.0 * f) - 4.0 * b2 * np.sin(2.0 * f)
            if d2 <= 0.0:
                break
            f = f - d1 / d2
        e = energy(0.5 * f)
        if e < best_e:
            best_e, best_t = e, 0.5 * f
    E_rhf = float(best_e + ZA * ZB / R)
    # full CI: product functions chi_i(1) chi_j(2) of the RHF orbital and its S-orthogonal partner, with
    # <ij|H|kl> = h_ik d_jl + d_ik h_jl + (ik|jl); the penalty lambda (1 - P12) / 2, P12 the exchange of the
    # two electrons, lifts the spatially antisymmetric (triplet) state by lambda = 20 hartree, so the lowest
    # eigenvalue is the lowest singlet
    C = np.column_stack([orbital(best_t), orbital(best_t + 0.5 * np.pi)])
    h = C.T @ H @ C
    g = np.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C, eri)
    I2 = np.eye(2)
    Hp = (np.einsum('ik,jl->ijkl', h, I2) + np.einsum('ik,jl->ijkl', I2, h)
          + np.einsum('ikjl->ijkl', g)).reshape(4, 4)
    P12 = np.einsum('il,jk->ijkl', I2, I2).reshape(4, 4)
    E_fci = float(np.linalg.eigvalsh(Hp + 10.0 * (np.eye(4) - P12))[0] + ZA * ZB / R)
    result = (E_fci, E_rhf)
    return result


def vibrational_levels(ZA, ZB, zetaA, zetaB, massA, massB):
    '''Lowest five vibrational levels (J = 0) on the full-CI potential curve, from the dissociation limit.'''
    # Other methods: the dissociation limit from one-centre integrals in momentum space (_k_atom),
    # V(R) sampled on Chebyshev nodes in ln R and interpolated up to the distance
    # where the cross integrals have died out, the exact Coulomb tail of the separated fragments
    # beyond it, and the radial equation by the sinc DVR on uniform grids whose length is doubled and
    # whose step is refined until the five levels stop changing (no shooting, no fixed outer wall).
    for z in (ZA, ZB):
        if not (np.isfinite(z) and 1.0 <= z <= 2.0):
            raise ValueError("nuclear charges must be finite and between 1 and 2")
    for m in (massA, massB):
        if not (np.isfinite(m) and 1.0 <= m <= 10.0):
            raise ValueError("nuclear masses must be finite and between 1 and 10 u")
    sA, hA, jA = _k_atom(ZA, zetaA)
    sB, hB, jB = _k_atom(ZB, zetaB)
    # both electrons on A, both on B, or one on each: energies E_j + c_j / R at large R
    arr = [((2.0 * hA * sA + jA) / sA ** 2, (ZA - 2.0) * ZB), ((2.0 * hB * sB + jB) / sB ** 2, (ZB - 2.0) * ZA),
           (hA / sA + hB / sB, (ZA - 1.0) * (ZB - 1.0))]
    e_inf = min(e for e, c in arr)

    def tail(r):
        return np.min([e - e_inf + c / r for e, c in arr], axis=0)

    # where the computed curve has become the tail (to 1e-12 hartree over 2 bohr)
    r_c = 6.0
    while max(abs(fci_energy(ZA, ZB, zetaA, zetaB, r_c + d)[0] - e_inf - tail(r_c + d)) for d in (0.0, 1.0, 2.0)) > 1e-12:
        r_c += 2.0
    r_lo = 0.3
    # Chebyshev interpolant of V in x = ln R on [r_lo, r_c]
    n_int = 200
    t = np.cos(np.pi * (np.arange(n_int) + 0.5) / n_int)
    x_lo, x_hi = np.log(r_lo), np.log(r_c)
    xs = 0.5 * (x_hi + x_lo) + 0.5 * (x_hi - x_lo) * t
    Vs = np.array([fci_energy(ZA, ZB, zetaA, zetaB, np.exp(x))[0] for x in xs]) - e_inf
    cheb = np.polynomial.chebyshev.Chebyshev.fit(t, Vs, n_int - 1)

    def V(r):
        r = np.asarray(r, float)
        inside = r <= r_c
        out = np.empty_like(r)
        out[inside] = cheb((2.0 * np.log(r[inside]) - x_hi - x_lo) / (x_hi - x_lo))
        out[~inside] = tail(r[~inside])
        return out

    mu = massA * massB / (massA + massB) * 1822.888486209
    v_min = float(np.min(V(np.linspace(r_lo, r_c, 4000))))
    kmax = np.sqrt(2.0 * mu * max(-v_min, 1e-4))

    def dvr(h, length):
        # Colbert-Miller sinc DVR on R = r_lo + k h, k = 1 .. n (chi = 0 at r_lo and at r_lo + (n + 1) h)
        n = int(length / h)
        k = np.arange(1, n + 1)
        d = k[:, None] - k[None, :]
        T = np.where(d == 0, np.pi ** 2 / 3.0, 2.0 * (-1.0) ** d / np.where(d == 0, 1, d) ** 2) / (2.0 * mu * h * h)
        E = np.linalg.eigvalsh(T + np.diag(V(r_lo + k * h)))
        return E[:5]

    def converged_in_length(h):
        length = r_c - r_lo
        cur = dvr(h, length)
        while True:
            length *= 2.0
            nxt = dvr(h, length)
            if np.max(np.abs(nxt - cur)) * 219474.6313632 < 1e-4:
                return nxt, length
            cur = nxt

    h = 0.6 * np.pi / kmax
    cur, length = converged_in_length(h)
    while True:
        h *= 0.8
        nxt, length = converged_in_length(h)
        if np.max(np.abs(nxt - cur)) * 219474.6313632 < 1e-4:
            break
        cur = nxt
    bound = nxt[nxt < 0.0]
    if bound.size < 5:
        raise ValueError("fewer than five vibrational levels lie below the dissociation limit")
    levels = bound[:5] * 219474.6313632
    return levels


# Second solution, step 6: Gaussian-transform (Rys-type) integrals evaluated by quadrature, no Boys
# function, no Hermite expansion and no recursion. 1 / r = (2 / sqrt(pi)) int_0^inf exp(-s**2 r**2) ds
# makes every Coulomb integral over Cartesian Gaussians a product of x, y and z factors under one
# integral over s; with u**2 = s**2 / (rho + s**2) that integral runs over u in [0, 1]:
#   attraction  <a|1/|r - C||b> = (2 pi / p) K_ab int_0^1 exp(-p |PC|**2 u**2) prod_x E_u[(x - Ax)^la (x - Bx)^lb] du
#   repulsion   (ab|cd) = 2 pi**2.5 / (p q sqrt(p + q)) K_ab K_cd int_0^1 exp(-rho |PQ|**2 u**2)
#                         prod_x E_u[(x1 - Ax)^la (x1 - Bx)^lb (x2 - Cx)^lc (x2 - Dx)^ld] du,
# where E_u is the mean under a Gaussian (one-dimensional for the attraction, mean Px - u**2 (Px - Cx),
# variance (1 - u**2) / 2p; two-dimensional for the repulsion, means Px - u**2 q (Px - Qx) / (p + q) and
# Qx + u**2 p (Px - Qx) / (p + q), covariance [[q (1 - u**2) + rho u**2, rho u**2], [rho u**2,
# p (1 - u**2) + rho u**2]] / 2pq). These means of polynomials of degree <= 4 are exact with the 3-point
# Gauss-Hermite rule in each direction, and the u integral is done with Gauss-Legendre on
# [0, min(1, sqrt(46 / T))], T the exponent factor (the integrand is a polynomial times exp(-T u**2)).
# Overlap and kinetic integrals are one-dimensional Gauss-Hermite quadratures of the same kind.

_Q_Z, _Q_W = np.polynomial.hermite_e.hermegauss(3)
_Q_W = _Q_W / np.sqrt(2.0 * np.pi)                     # probabilists' rule, weights summing to 1
_Q_U, _Q_UW = np.polynomial.legendre.leggauss(40)
_Q_U, _Q_UW = 0.5 * (_Q_U + 1.0), 0.5 * _Q_UW


def _q_prims(zA, zB, aA, aB, R):
    # primitive Cartesian Gaussians: exponent, coefficient (contraction times normalization), angular
    # exponents (lx, ly, lz), centre, index of the basis function they belong to
    al = np.array([0.109818, 0.405771, 2.22766])
    dc = np.array([0.444635, 0.535328, 0.154329])
    ex, cf, ang, cen, fun = [], [], [], [], []
    f = 0
    for c, z, ap in ((np.zeros(3), zA, aA), (np.array([0.0, 0.0, R]), zB, aB)):
        e = al * z * z
        for ee, cc in zip(e, dc * (2.0 * e / np.pi) ** 0.75):
            ex.append(ee); cf.append(cc); ang.append((0, 0, 0)); cen.append(c); fun.append(f)
        f += 1
        for l in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            ex.append(ap); cf.append((2.0 * ap / np.pi) ** 0.75 * 2.0 * np.sqrt(ap)); ang.append(l)
            cen.append(c); fun.append(f)
            f += 1
    return (np.array(ex), np.array(cf), np.array(ang), np.array(cen), np.array(fun), f)


def _q_mean1(m, var, A, la, B, lb):
    # E[(x - A)^la (x - B)^lb] for x ~ N(m, var), 3-point Gauss-Hermite (exact up to degree 5);
    # arrays broadcast, the quadrature node on the last axis
    x = m[..., None] + np.sqrt(var)[..., None] * _Q_Z
    return np.sum(_Q_W * (x - A[..., None]) ** la[..., None] * (x - B[..., None]) ** lb[..., None], axis=-1)


def _q_integrals(ZA, ZB, zA, zB, aA, aB, R):
    ex, cf, ang, cen, fun, nb = _q_prims(zA, zB, aA, aB, R)
    n = ex.size
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    i, j = i.ravel(), j.ravel()                       # all ordered primitive pairs
    a, b = ex[i], ex[j]
    A, B = cen[i], cen[j]
    la, lb = ang[i], ang[j]
    p = a + b
    P = (a[:, None] * A + b[:, None] * B) / p[:, None]
    Kx = np.exp(-(a * b / p)[:, None] * (A - B) ** 2)          # per direction
    K = np.prod(Kx, axis=1)
    # map from primitive pairs to basis-function pairs (mu nb + nu) with the coefficients
    M = np.zeros((nb * nb, n * n))
    M[fun[i] * nb + fun[j], np.arange(n * n)] = cf[i] * cf[j]
    # overlap and kinetic: one-dimensional means under N(Px, 1 / 2p)
    var = 1.0 / (2.0 * p)
    s1 = np.empty((n * n, 3))
    t1 = np.empty((n * n, 3))
    for x in range(3):
        m = P[:, x]
        s1[:, x] = Kx[:, x] * np.sqrt(np.pi / p) * _q_mean1(m, var, A[:, x], la[:, x], B[:, x], lb[:, x])
        # <d/dx a | d/dx b> with d/dx (x - A)^l e^{-a (x - A)^2} = l (x - A)^(l - 1) - 2a (x - A)^(l + 1)
        xx = m[:, None] + np.sqrt(var)[:, None] * _Q_Z
        ya = xx - A[:, x, None]
        yb = xx - B[:, x, None]
        da = la[:, x, None] * ya ** np.maximum(la[:, x, None] - 1, 0) - 2.0 * a[:, None] * ya ** (la[:, x, None] + 1)
        db = lb[:, x, None] * yb ** np.maximum(lb[:, x, None] - 1, 0) - 2.0 * b[:, None] * yb ** (lb[:, x, None] + 1)
        t1[:, x] = Kx[:, x] * np.sqrt(np.pi / p) * np.sum(_Q_W * da * db, axis=1)
    ovl = np.prod(s1, axis=1)
    kin = 0.5 * (t1[:, 0] * s1[:, 1] * s1[:, 2] + s1[:, 0] * t1[:, 1] * s1[:, 2] + s1[:, 0] * s1[:, 1] * t1[:, 2])
    # nuclear attraction
    att = np.zeros(n * n)
    for Z, C in ((ZA, np.zeros(3)), (ZB, np.array([0.0, 0.0, R]))):
        T = p * np.sum((P - C) ** 2, axis=1)
        top = np.minimum(1.0, np.sqrt(46.0 / np.maximum(T, 1e-300)))
        u = top[:, None] * _Q_U[None, :]
        wu = top[:, None] * _Q_UW[None, :]
        u2 = u * u
        prod = np.exp(-T[:, None] * u2)
        varu = (1.0 - u2) / (2.0 * p[:, None])
        for x in range(3):
            mu = P[:, x, None] - u2 * (P[:, x] - C[x])[:, None]
            prod = prod * _q_mean1(mu, varu, A[:, x, None] * np.ones_like(u), la[:, x, None] * np.ones_like(u, int),
                                   B[:, x, None] * np.ones_like(u), lb[:, x, None] * np.ones_like(u, int))
        att -= Z * 2.0 * np.pi / p * K * np.sum(wu * prod, axis=1)
    S = (M @ ovl).reshape(nb, nb)
    H = (M @ (kin + att)).reshape(nb, nb)
    # electron repulsion over all primitive quartets (pair 1, pair 2), in blocks of pair 1
    V = np.empty((n * n, n * n))
    zz1 = np.repeat(_Q_Z, 3)
    zz2 = np.tile(_Q_Z, 3)
    ww = np.repeat(_Q_W, 3) * np.tile(_Q_W, 3)
    for lo in range(0, n * n, 12):
        s = slice(lo, min(lo + 12, n * n))
        p1, q1 = p[s, None], p[None, :]
        rho = p1 * q1 / (p1 + q1)
        PQ = P[s, None, :] - P[None, :, :]
        T = rho * np.sum(PQ ** 2, axis=2)
        top = np.minimum(1.0, np.sqrt(46.0 / np.maximum(T, 1e-300)))
        u = top[..., None] * _Q_U
        wu = top[..., None] * _Q_UW
        u2 = u * u
        prod = np.exp(-T[..., None] * u2)
        p3, q3, r3 = p1[..., None], q1[..., None], rho[..., None]
        c11 = (q3 * (1.0 - u2) + r3 * u2) / (2.0 * p3 * q3)
        c12 = r3 * u2 / (2.0 * p3 * q3)
        c22 = (p3 * (1.0 - u2) + r3 * u2) / (2.0 * p3 * q3)
        L11 = np.sqrt(c11)
        L21 = c12 / L11
        L22 = np.sqrt(np.maximum(c22 - L21 * L21, 0.0))
        for x in range(3):
            d = PQ[..., x][..., None]
            m1 = P[s, None, x][..., None] - u2 * q3 / (p3 + q3) * d
            m2 = P[None, :, x][..., None] + u2 * p3 / (p3 + q3) * d
            x1 = m1[..., None] + L11[..., None] * zz1
            x2 = m2[..., None] + L21[..., None] * zz1 + L22[..., None] * zz2
            f1 = ((x1 - A[s, None, x, None, None]) ** la[s, None, x, None, None]
                  * (x1 - B[s, None, x, None, None]) ** lb[s, None, x, None, None])
            f2 = ((x2 - A[None, :, x, None, None]) ** la[None, :, x, None, None]
                  * (x2 - B[None, :, x, None, None]) ** lb[None, :, x, None, None])
            prod = prod * np.sum(ww * f1 * f2, axis=-1)
        V[s] = (2.0 * np.pi ** 2.5 / (p1 * q1 * np.sqrt(p1 + q1)) * K[s, None] * K[None, :]
                * np.sum(wu * prod, axis=-1))
    G = (M @ V @ M.T).reshape(nb, nb, nb, nb)
    return S, H, G


def polarized_energies(ZA, ZB, zetaA, zetaB, alphaA, alphaB, R):
    '''Full-CI and RHF energies of a two-electron diatomic in the STO-3G 1s + p-shell basis.'''
    # Other routes: RHF by steepest descent along great circles with an exact line search from many
    # starting orbitals; full CI in the whole 64-dimensional product space of the RHF orbitals with a
    # spin penalty on the spatially antisymmetric (triplet) states.
    S, H, G = _q_integrals(ZA, ZB, zetaA, zetaB, alphaA, alphaB, R)
    n = S.shape[0]
    w, U = np.linalg.eigh(S)
    X = U @ np.diag(w ** -0.5) @ U.T
    h = X.T @ H @ X
    g = np.einsum("pi,qj,rk,sl,pqrs->ijkl", X, X, X, X, G)

    def energy(x):
        return float(2.0 * (x @ h @ x) + np.einsum("i,j,k,l,ijkl->", x, x, x, x, g))

    def grad(x):
        r = 4.0 * (h + np.einsum("k,l,ijkl->ij", x, x, g)) @ x
        return r - (r @ x) * x

    rng = np.random.default_rng(77)
    best, best_x = np.inf, None
    for x in list(np.linalg.eigh(h)[1].T) + list(rng.normal(size=(40, n))):
        x = x / np.linalg.norm(x)
        for _ in range(3000):
            d = -grad(x)
            nd = np.linalg.norm(d)
            if nd < 1e-10:
                break
            d /= nd
            # great circle x cos(s) + d sin(s); exact line search with Brent's method
            res = minimize_scalar(lambda s_: energy(np.cos(s_) * x + np.sin(s_) * d), bounds=(0.0, 0.5),
                                  method="bounded", options={"xatol": 1e-14})
            y = np.cos(res.x) * x + np.sin(res.x) * d
            if energy(y) >= energy(x):
                break
            x = y / np.linalg.norm(y)
        if energy(x) < best:
            best, best_x = energy(x), x
    E_rhf = best + ZA * ZB / R
    # full CI: orthonormal orbitals = the RHF orbital and an orthonormal complement (QR), product
    # functions chi_i(1) chi_j(2) with <ij|H|kl> = h_ik d_jl + d_ik h_jl + (ik|jl), and the penalty
    # lambda (1 - P12) / 2 (P12 exchanges the electrons, lambda = 20 hartree) that lifts every triplet
    # above the singlets, so the lowest eigenvalue is the lowest singlet
    Q = np.linalg.qr(np.column_stack([best_x, np.eye(n)]))[0][:, :n]
    hm = Q.T @ h @ Q
    gm = np.einsum("pi,qj,rk,sl,pqrs->ijkl", Q, Q, Q, Q, g)
    I = np.eye(n)
    Hp = (np.einsum("ik,jl->ijkl", hm, I) + np.einsum("ik,jl->ijkl", I, hm)
          + np.einsum("ikjl->ijkl", gm)).reshape(n * n, n * n)
    P12 = np.einsum("il,jk->ijkl", I, I).reshape(n * n, n * n)
    E_fci = float(np.linalg.eigvalsh(Hp + 10.0 * (np.eye(n * n) - P12))[0]) + ZA * ZB / R
    result = (float(E_fci), float(E_rhf))
    return result
