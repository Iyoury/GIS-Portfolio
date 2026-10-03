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
    K = 1.0 / T
    labels = np.arange(2 ** L)
    spins = 1 - 2 * ((labels[:, None] >> np.arange(L)[None, :]) & 1)
    inside = np.sum(spins * np.roll(spins, -1, axis=1), axis=1)   # bonds inside each row
    field = np.sum(spins, axis=1)                                  # sum of the spins of each row
    half = 0.5 * K * inside + 0.5 * (h / T) * field                # half of a row's own weight
    M = np.exp(half[:, None] + half[None, :] + K * (spins @ spins.T))
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
    M = transfer_matrix(L, T)
    n = 2 ** L
    labels = np.arange(n)
    partner = labels ^ (n - 1)              # the same row with every spin flipped
    rep = labels[labels < partner]
    # M commutes with the global spin flip. In the basis (|a> +- |flip a>) / sqrt(2) it splits
    # into an even block M[a, b] + M[a, flip b] and an odd block M[a, b] - M[a, flip b].
    near = M[np.ix_(rep, rep)]
    far = M[np.ix_(rep, partner[rep])]
    even = np.linalg.eigvalsh(near + far)
    odd = np.linalg.eigvalsh(near - far)
    lam0 = even[-1]
    # s_0 is odd under the flip, so it joins the top state to the odd sector; e = s_0 s_1 is even,
    # so its connected correlation decays through the second state of the even sector.
    f = float(-T * np.log(lam0) / L)
    xi_spin = float(1.0 / np.log(lam0 / odd[-1]))
    xi_energy = float(1.0 / np.log(lam0 / even[-2]))
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
    M = transfer_matrix(L, T)
    n = 2 ** L
    labels = np.arange(n)
    partner = labels ^ (n - 1)              # the same row with every spin flipped
    rep = labels[labels < partner]
    # M commutes with the global spin flip. In the basis (|a> +- |flip a>) / sqrt(2) it splits
    # into an even block M[a, b] + M[a, flip b] and an odd block M[a, b] - M[a, flip b].
    near = M[np.ix_(rep, rep)]
    far = M[np.ix_(rep, partner[rep])]
    # G(r) = sum_k <0|s_0|k>^2 (lam_k / lam_0)^r; the slowest term comes from the top odd state.
    # Below Tc that state is degenerate with the top even state to machine precision, so the two
    # are taken from the separate sector blocks rather than from one eigensolver call.
    w_even, v_even = np.linalg.eigh(near + far)
    w_odd, v_odd = np.linalg.eigh(near - far)
    s0 = 1 - 2 * (rep & 1)                  # s_0 sends (|a> + |flip a>)/sqrt2 to s_0(a) (|a> - |flip a>)/sqrt2
    m_L = abs(float(np.sum(v_even[:, -1] * s0 * v_odd[:, -1])))
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
    M = transfer_matrix(L, T)
    n = 2 ** L
    labels = np.arange(n)
    partner = labels ^ (n - 1)              # the same row with every spin flipped
    rep = labels[labels < partner]
    # M commutes with the global spin flip. In the basis (|a> +- |flip a>) / sqrt(2) it splits
    # into an even block M[a, b] + M[a, flip b] and an odd block M[a, b] - M[a, flip b].
    near = M[np.ix_(rep, rep)]
    far = M[np.ix_(rep, partner[rep])]
    w_even, v_even = np.linalg.eigh(near + far)
    w_odd, v_odd = np.linalg.eigh(near - far)
    S = np.sum(1 - 2 * ((rep[:, None] >> np.arange(L)[None, :]) & 1), axis=1)   # row magnetization
    lam0 = w_even[-1]
    amp = v_odd.T @ (S * v_even[:, -1])     # <k|S|0> for every odd state k
    # chi = (1 / (T L)) sum over all r of <S(row 0) S(row r)>, summed in closed form state by state
    chi = float(np.sum(amp ** 2 * (lam0 + w_odd) / (lam0 - w_odd)) / (T * L))
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
    def discriminant(H):
        # (a - b)^2 for the two eigenvalues a, b of largest modulus: positive while they are real and
        # distinct, zero where they meet, negative once they form a complex-conjugate pair.
        w = np.linalg.eigvals(transfer_matrix(L, T, 1j * H))
        top = w[np.argsort(-np.abs(w))[:2]]
        return float(((top[0] - top[1]) ** 2).real)

    lo = hi = 1e-12 * T
    while discriminant(hi) > 0.0:           # the edge is tiny below Tc: walk up geometrically
        lo, hi = hi, 2.0 * hi
    H_edge = float(brentq(discriminant, lo, hi, xtol=1e-16, rtol=1e-13))
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
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    r = np.log((L + 1.0) / L)
    lengths = [strip_lengths(W, Tc) for W in (L, L + 1, L + 2)]
    widths = np.array([L, L + 1, L + 2], dtype=float)
    A = np.column_stack([np.ones(3), -np.pi * Tc / (6.0 * widths ** 2), widths ** -4.0])
    c = float(np.linalg.solve(A, np.array([x[0] for x in lengths]))[1])
    result = {
        "x_sigma": float(L / (2.0 * np.pi * lengths[0][1])),
        "x_energy": float(L / (2.0 * np.pi * lengths[0][2])),
        "beta_over_nu": float(np.log(strip_magnetization(L, Tc) / strip_magnetization(L + 1, Tc)) / r),
        "gamma_over_nu": float(np.log(strip_susceptibility(L + 1, Tc) / strip_susceptibility(L, Tc)) / r),
        "y_h": float(np.log(lee_yang_edge(L, Tc) / lee_yang_edge(L + 1, Tc)) / r),
        "c": c,
    }
    return result


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
    mp1, mm1 = at(T, h + dh)[1], at(T, h - dh)[1]
    mp2, mm2 = at(T, h + 2 * dh)[1], at(T, h - 2 * dh)[1]
    chi = (4.0 * (mp1 - mm1) / (2 * dh) - (mp2 - mm2) / (4 * dh)) / 3.0
    fp1, fm1 = at(T + dT, h)[0], at(T - dT, h)[0]
    fp2, fm2 = at(T + 2 * dT, h)[0], at(T - 2 * dT, h)[0]
    d2 = (4.0 * (fp1 - 2 * f0 + fm1) / dT ** 2 - (fp2 - 2 * f0 + fm2) / (4 * dT ** 2)) / 3.0
    result = (float(f0), float(m0), float(chi), float(-T * d2))
    return result
