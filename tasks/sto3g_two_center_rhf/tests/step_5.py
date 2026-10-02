import numpy as np
from scipy.special import erf as _t_erf
from scipy.integrate import quad as _t_quad

# Independent targets for the energy curve and the vibrational levels.
# - integrals: closed-form s-Gaussian formulas re-implemented here, vectorized over many R
#   (F0 from erf with its small-argument series), cross-checked against steps 2 and 3 tests;
# - full CI: lowest eigenvalue of the two-electron Hamiltonian in the symmetric (singlet)
#   combinations of the NON-orthogonal products phi_i(1) phi_j(2), generalized eigenproblem
#   (no orbitals, no configuration state functions);
# - dissociation limit: one-centre integrals by adaptive radial quadrature of the
#   Gaussian expansion (no closed forms);
# - levels: finite differences on three fine grids, extrapolated twice by Richardson.
_T_A = np.array([0.109818, 0.405771, 2.22766])
_T_D = np.array([0.444635, 0.535328, 0.154329])
_T_CM = 219474.6313632
_T_ME = 1822.888486209


def _t_F0(t):
    t = np.asarray(t, float)
    out = np.empty_like(t)
    small = t < 1e-8
    out[small] = 1.0 - t[small] / 3.0
    r = np.sqrt(t[~small])
    out[~small] = 0.5 * np.sqrt(np.pi) * _t_erf(r) / r
    return out


def _t_curve(ZA, ZB, za, zb, R):
    # full-CI total energy for an array of distances R
    R = np.asarray(R, float)
    cen = [np.zeros_like(R), R]
    Z = [ZA, ZB]
    ex = [_T_A * za ** 2, _T_A * zb ** 2]
    nrm = [_T_D * (2 * e / np.pi) ** 0.75 for e in ex]
    n = R.size
    S = np.zeros((n, 2, 2))
    H = np.zeros((n, 2, 2))
    for m in range(2):
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    a, b = ex[m][i], ex[k][j]
                    p = a + b
                    d2 = (cen[m] - cen[k]) ** 2
                    w = nrm[m][i] * nrm[k][j] * np.exp(-a * b / p * d2)
                    P = (a * cen[m] + b * cen[k]) / p
                    S[:, m, k] += w * (np.pi / p) ** 1.5
                    H[:, m, k] += w * a * b / p * (3 - 2 * a * b / p * d2) * (np.pi / p) ** 1.5
                    for c in range(2):
                        H[:, m, k] -= w * Z[c] * 2 * np.pi / p * _t_F0(p * (P - cen[c]) ** 2)
    G = np.zeros((n, 2, 2, 2, 2))
    for q1 in range(2):
        for q2 in range(2):
            for q3 in range(2):
                for q4 in range(2):
                    acc = np.zeros(n)
                    for i in range(3):
                        for j in range(3):
                            a, b = ex[q1][i], ex[q2][j]
                            p = a + b
                            P = (a * cen[q1] + b * cen[q2]) / p
                            kp = nrm[q1][i] * nrm[q2][j] * np.exp(-a * b / p * (cen[q1] - cen[q2]) ** 2)
                            for k in range(3):
                                for l in range(3):
                                    c, d = ex[q3][k], ex[q4][l]
                                    q = c + d
                                    Q = (c * cen[q3] + d * cen[q4]) / q
                                    kq = nrm[q3][k] * nrm[q4][l] * np.exp(-c * d / q * (cen[q3] - cen[q4]) ** 2)
                                    acc += kp * kq * 2 * np.pi ** 2.5 / (p * q * np.sqrt(p + q)) * _t_F0(p * q / (p + q) * (P - Q) ** 2)
                    G[:, q1, q2, q3, q4] = acc
    # product basis |ij> = phi_i(1) phi_j(2): <ij|H|kl> = h_ik S_jl + S_ik h_jl + (ik|jl)
    Hp = (np.einsum('nik,njl->nijkl', H, S) + np.einsum('nik,njl->nijkl', S, H)
          + np.einsum('nikjl->nijkl', G)).reshape(n, 4, 4)
    Sp = np.einsum('nik,njl->nijkl', S, S).reshape(n, 4, 4)
    B = np.array([[1, 0, 0], [0, 0, np.sqrt(0.5)], [0, 0, np.sqrt(0.5)], [0, 1, 0]], float)   # singlet functions
    Hp = B.T @ Hp @ B
    Sp = B.T @ Sp @ B
    L = np.linalg.cholesky(Sp)
    Li = np.linalg.inv(L)
    E = np.linalg.eigvalsh(Li @ Hp @ np.swapaxes(Li, 1, 2))[:, 0]
    return E + ZA * ZB / R


def _t_atom(Z, zeta):
    # radial quadrature: norm, <phi| -1/2 Lap - Z/r |phi>, (phi phi | phi phi) on one centre
    a = _T_A * zeta ** 2
    c = _T_D * (2 * a / np.pi) ** 0.75
    f = lambda r: np.sum(c * np.exp(-a * r * r))
    lap = lambda r: np.sum(c * (4 * a * a * r * r - 6 * a) * np.exp(-a * r * r))
    top = 12.0 / np.sqrt(a[0])
    opts = dict(epsabs=0.0, epsrel=1e-13, limit=400, points=[1 / np.sqrt(x) for x in a])
    s = _t_quad(lambda r: 4 * np.pi * r * r * f(r) ** 2, 0, top, **opts)[0]
    t = _t_quad(lambda r: -0.5 * 4 * np.pi * r * r * f(r) * lap(r), 0, top, **opts)[0]
    v = _t_quad(lambda r: -Z * 4 * np.pi * r * f(r) ** 2, 0, top, **opts)[0]
    # potential of the charge cloud rho = phi**2: U(r) = (1/r) int_0^r rho 4 pi s^2 ds + int_r^inf rho 4 pi s ds
    def U(r):
        inner = _t_quad(lambda x: 4 * np.pi * x * x * f(x) ** 2, 0, r, epsabs=0, epsrel=1e-13, limit=200)[0]
        outer = _t_quad(lambda x: 4 * np.pi * x * f(x) ** 2, r, top, epsabs=0, epsrel=1e-13, limit=200)[0]
        return inner / r + outer
    j = _t_quad(lambda r: 4 * np.pi * r * r * f(r) ** 2 * U(r), 1e-12, top, epsabs=0.0, epsrel=1e-11,
                limit=200, points=[1 / np.sqrt(x) for x in a])[0]
    return s, t + v, j


_T_LIMIT_CACHE = {}


def _t_limit(ZA, ZB, za, zb):
    key = (ZA, ZB, za, zb)
    if key not in _T_LIMIT_CACHE:
        _T_LIMIT_CACHE[key] = _t_limit_raw(ZA, ZB, za, zb)
    return _T_LIMIT_CACHE[key]


def _t_limit_raw(ZA, ZB, za, zb):
    # lowest separated-fragment arrangement and the product of the fragment charges
    sA, hA, jA = _t_atom(ZA, za)
    sB, hB, jB = _t_atom(ZB, zb)
    opts = [((2 * hA * sA + jA) / sA ** 2, (ZA - 2) * ZB), ((2 * hB * sB + jB) / sB ** 2, (ZB - 2) * ZA),
            (hA / sA + hB / sB, (ZA - 1) * (ZB - 1))]
    return min(opts)


def _t_numerov_levels(ZA, ZB, za, zb, mA, mB, nlev=5, h=0.004, r0=0.25, r1=22.0):
    # second-order finite differences of the radial equation on steps h, h/2 and h/4
    # (tridiagonal eigenproblems, Dirichlet walls), extrapolated twice by Richardson (error O(h**6))
    from scipy.linalg import eigh_tridiagonal as _t_eigt
    mu = mA * mB / (mA + mB) * _T_ME
    e_inf = _t_limit(ZA, ZB, za, zb)[0]
    R = np.arange(r0, r1 + 0.125 * h, 0.25 * h)
    V = _t_curve(ZA, ZB, za, zb, R) - e_inf
    out = []
    for k in (4, 2, 1):
        step = 0.25 * h * k
        Vk = V[k::k][:-1]                              # interior points of the coarser grid
        diag = 1.0 / (mu * step * step) + Vk
        off = np.full(Vk.size - 1, -0.5 / (mu * step * step))
        out.append(_t_eigt(diag, off, select="i", select_range=(0, nlev - 1), eigvals_only=True))
    r1_ = (4.0 * out[1] - out[0]) / 3.0
    r2_ = (4.0 * out[2] - out[1]) / 3.0
    return (16.0 * r2_ - r1_) / 15.0 * _T_CM


_T_MH, _T_MD, _T_MHE = 1.00782503207, 2.01410177812, 4.00260325413


def _t_check_levels(args):
    lev = vibrational_levels(*args)
    assert isinstance(lev, np.ndarray) and lev.shape == (5,), lev
    target = _t_numerov_levels(*args)
    err = np.abs(lev - target)
    assert np.all(err < 0.01), (args, lev, target)
    return lev


# --- test case 0: H2 (zeta = 1.24); neutral fragments H + H, levels from the full-CI curve ---
lev = _t_check_levels((1.0, 1.0, 1.24, 1.24, _T_MH, _T_MH))
assert 4000.0 < lev[1] - lev[0] < 6000.0              # fundamental near the minimal-basis value

# --- test case 1: HeH+; the lowest separated arrangement is He + H+ (both electrons on He) ---
_t_check_levels((2.0, 1.0, 2.0925, 1.24, _T_MHE, _T_MH))

# --- test case 2: ion-pair dissociation H- + H+ (attractive -1/R tail): the full-CI energy at
# 60 bohr is still 1/60 hartree (about 3658 cm^-1) below the true limit ---
_t_check_levels((1.0, 1.0, 1.24, 2.69, _T_MH, _T_MH))

# --- test case 3: fragments with charges 0.5 and 0.5 (repulsive +0.25/R tail) ---
_t_check_levels((1.5, 1.5, 2.0, 1.24, 3.0, 3.0))

# --- test case 4: isotopes on the same curve (D2), heavier nuclei and lower, denser levels ---
lev_d = _t_check_levels((1.0, 1.0, 1.24, 1.24, _T_MD, _T_MD))
assert lev_d[0] < lev[0] and lev_d[1] - lev_d[0] < lev[1] - lev[0]

# --- test case 5: ends of the allowed ranges: zeta = 0.8 (shallow, wide well) and zeta = 3
# (tight well at small R), both with the heaviest nuclei 10 u, and the lightest nuclei 1 u ---
_t_check_levels((1.0, 1.0, 0.8, 0.8, 10.0, 10.0))
_t_check_levels((1.0, 1.0, 3.0, 3.0, 10.0, 10.0))
_t_check_levels((1.0, 1.0, 1.24, 1.24, 1.0, 1.0))

# --- test case 6: no five bound levels (He2 2+ only has a metastable well above He+ + He+),
# or a mass that is not finite or outside [1, 10] u, raise ValueError ---
for _t_bad in ((2.0, 2.0, 2.0925, 2.0925, _T_MHE, _T_MHE), (1.0, 1.0, 1.24, 1.24, np.nan, _T_MH),
               (1.0, 1.0, 1.24, 1.24, _T_MH, np.inf), (1.0, 1.0, 1.24, 1.24, 10.5, _T_MH),
               (1.0, 1.0, 1.24, 1.24, 0.5, _T_MH)):
    try:
        vibrational_levels(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("vibrational_levels%r must raise ValueError" % (_t_bad,))
