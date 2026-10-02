import numpy as np
from scipy.optimize import brentq


def _s7_ctm(T, h, chi, start=None):
    # Corner-transfer-matrix renormalization (Nishino and Okunishi) on the infinite lattice.
    # Each bond weight is split as exp(K s s') = sum_k W[s, k] W[s', k], which gives one site
    # tensor A with four equivalent legs whose full contraction is the partition function; one
    # corner C and one edge E then describe the whole environment. Returns the free energy per
    # spin, the magnetization and the converged (C, E) (used to warm-start nearby points).
    K = 1.0 / T
    W = np.array([[np.sqrt(np.cosh(K)), np.sqrt(np.sinh(K))],
                  [np.sqrt(np.cosh(K)), -np.sqrt(np.sinh(K))]])     # rows: s = +1, -1
    s = np.array([1.0, -1.0])
    weight = np.exp((h / T) * s)
    A = np.einsum("s,sl,su,sr,sd->lurd", weight, W, W, W, W)
    A_spin = np.einsum("s,s,sl,su,sr,sd->lurd", s, weight, W, W, W, W)
    if start is None:
        C = np.einsum("lurd,l,u->rd", A, W[0], W[0])     # start from a boundary of up spins
        E = np.einsum("lurd,l->urd", A, W[0])
    else:
        C, E = start
    history = []
    for _ in range(20000):
        big = np.einsum("ale,ab,buc,lurd->edcr", E, C, E, A, optimize=True)
        n = big.shape[0] * 2
        big = big.reshape(n, n)
        w, U = np.linalg.eigh(0.5 * (big + big.T))
        keep = np.argsort(-np.abs(w))[:chi]
        P = U[:, keep]
        C = np.diag(w[keep] / np.max(np.abs(w[keep])))
        grown = np.einsum("ale,lurd->aured", E, A, optimize=True).reshape(n, 2, n)
        E = np.einsum("xa,xry,yb->arb", P, grown, P, optimize=True)
        E = 0.5 * (E + E.transpose(2, 1, 0))
        E /= np.max(np.abs(E))
        half = np.einsum("ab,buc,cf->auf", C, E, C, optimize=True)
        env = np.einsum("auf,frg,gdj,jla->lurd", half, E, half, E, optimize=True)
        z9 = np.sum(env * A)
        m = float(np.sum(env * A_spin) / z9)
        # partition function per site: kappa = Z(4C+4E+A) Z(4C) / Z(4C+2E)**2, in which the
        # arbitrary normalizations of C and E cancel
        z4 = np.trace(np.linalg.matrix_power(C, 4))
        z6 = np.einsum("auf,fua->", half, half)
        f = float(-T * np.log(z9 * z4 / z6 ** 2))
        history.append((m, f))
        if len(history) > 10 and max(max(abs(m - a), abs(f - b)) for a, b in history[-6:-1]) < 1e-13:
            break
    return f, m, (C, E)


def bulk_thermodynamics(T, h):
    '''Free energy, magnetization, susceptibility and specific heat per spin of the infinite lattice in a field.

    Inputs:
      T: float, temperature, 2 <= T <= 3 (J = 1, k_B = 1).
      h: float, uniform field, 0.005 <= h <= 0.2; the energy is E = - sum_<ij> s_i s_j - h sum_i s_i.

    Output:
      (f, m, chi, c): tuple of four Python floats for the infinite square lattice (the limit of
      the strip quantities as the width goes to infinity):
        f: free energy per spin, f = -T lim ln(Z) / N (units of J), absolute accuracy 1e-10;
        m: magnetization per spin <s_i> = -df/dh, absolute accuracy 1e-9;
        chi: susceptibility per spin, chi = dm/dh at fixed T (units 1/J), relative accuracy 1e-5;
        c: specific heat per spin at fixed h, c = -T d2f/dT2 (units k_B), relative accuracy 1e-5.
    '''
    chi_bond = 16
    f0, m0, env = _s7_ctm(T, h, chi_bond)

    def at(TT, hh):
        return _s7_ctm(TT, hh, chi_bond, start=env)[:2]

    # Richardson-extrapolated central differences around the converged point
    dh, dT = 1e-4, 1e-3
    # MUTANT: susceptibility of independent spins, (1 - m**2) / T
    chi = (1.0 - m0 * m0) / T
    fp1, fm1 = at(T + dT, h)[0], at(T - dT, h)[0]
    fp2, fm2 = at(T + 2 * dT, h)[0], at(T - 2 * dT, h)[0]
    d2 = (4.0 * (fp1 - 2 * f0 + fm1) / dT ** 2 - (fp2 - 2 * f0 + fm2) / (4 * dT ** 2)) / 3.0
    result = (float(f0), float(m0), float(chi), float(-T * d2))
    return result
