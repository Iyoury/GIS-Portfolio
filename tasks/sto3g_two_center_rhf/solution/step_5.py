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


def vibrational_levels(ZA, ZB, zetaA, zetaB, massA, massB):
    '''Lowest five vibrational levels (J = 0) on the full-CI potential curve, from the dissociation limit.

    Inputs:
      ZA, ZB: float, nuclear charges, 1 <= Z <= 3.
      zetaA, zetaB: float, Slater exponents of the 1s functions on A and B, 0.8 <= zeta <= 3.
      massA, massB: float, nuclear masses in unified atomic mass units (u), > 0.

    Output:
      levels: float numpy array of shape (5,), the energies of the vibrational states
              v = 0..4 in cm^-1, measured from the exact R -> infinity limit of the
              full-CI energy (so all are negative), increasing. Absolute error below
              0.01 cm^-1 for each level.

    Raises:
      ValueError if a mass is not positive and finite, or if fewer than five bound
      vibrational levels lie below the dissociation limit.
    '''
    for m in (massA, massB):
        if not (np.isfinite(m) and m > 0.0):
            raise ValueError("nuclear masses must be positive and finite")
    # Dissociation limit in this basis: as R -> infinity all cross integrals vanish and the
    # lowest singlet is the lowest of the three fragment arrangements (both electrons on A,
    # both on B, one on each). The leftover Coulomb energy of the fragment charges
    # q_A q_B / R vanishes only in the limit, so the limit is not E(R) at any finite R.
    sA, hA, jA = _s5_atom(ZA, zetaA)
    sB, hB, jB = _s5_atom(ZB, zetaB)
    e_inf = min((2.0 * hA * sA + jA) / sA ** 2,
                (2.0 * hB * sB + jB) / sB ** 2,
                hA / sA + hB / sB)
    # Radial nuclear equation -(1/2 mu) chi'' + V chi = E chi, V(R) = E_fci(R) - e_inf,
    # solved with the sinc discrete-variable representation (Colbert-Miller) on a uniform
    # grid; the five lowest states are negligible outside [0.35, 16] bohr.
    mu = massA * massB / (massA + massB) * 1822.888486209
    h = 0.02
    R = np.arange(0.35, 16.0 + 0.5 * h, h)
    V = np.array([fci_energy(ZA, ZB, zetaA, zetaB, r)[0] for r in R]) - e_inf
    n = np.arange(R.size)
    diff = n[:, None] - n[None, :]
    with np.errstate(divide="ignore"):
        T = np.where(diff == 0, np.pi ** 2 / 3.0, 2.0 * (-1.0) ** diff / np.where(diff == 0, 1, diff) ** 2)
    T = T / (2.0 * mu * h * h)
    E = np.linalg.eigvalsh(T + np.diag(V))
    bound = E[E < 0.0]
    if bound.size < 5:
        raise ValueError("fewer than five vibrational levels lie below the dissociation limit")
    levels = bound[:5] * 219474.6313632
    return levels
