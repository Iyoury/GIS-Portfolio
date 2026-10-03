import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def branch_magnetization(h, psi):
    '''Magnetization along the field on the zero-temperature descending branch of one particle.

    Inputs:
      h: float, reduced field along the field axis (negative means reversed field).
      psi: float, angle in radians between the field axis and the easy axis, 0 <= psi <= pi/2.

    Output:
      m: float, cos(theta - psi), where theta is the magnetization angle (from the easy
         axis) of the state reached when the field comes down from large positive values
         to h, the particle staying in its local energy minimum until that minimum
         disappears at h = -h_sw(psi). Absolute error below 1e-10 when
         |h + h_sw(psi)| >= 1e-3.

    Raises:
      ValueError if psi is outside [0, pi/2] or h is not finite.
    '''
    # Other method: all equilibria from the quartic in t = tan(theta/2), then pick the
    # minimum on the side of the positive easy direction while it still exists. The
    # switching field comes from the fold of the equilibrium curve (no astroid formula).
    h = float(h)
    psi = float(psi)
    if not np.isfinite(h) or not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("h must be finite and psi must be between 0 and pi/2")
    if psi == 0.0 or psi == 0.5 * np.pi:
        h_sw = 1.0
    else:
        def curvature_at_fold(t):
            return np.cos(2.0 * t) - np.sin(2.0 * t) * np.cos(t - psi) / (2.0 * np.sin(t - psi))

        t_fold = brentq(curvature_at_fold, -0.5 * np.pi + 1e-12, -1e-12, xtol=1e-15, rtol=1e-15)
        h_sw = float(np.sin(2.0 * t_fold) / (2.0 * np.sin(t_fold - psi)))
    s, c = np.sin(psi), np.cos(psi)
    coeffs = [h * s, 2.0 * h * c - 2.0, 0.0, 2.0 + 2.0 * h * c, -h * s]
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-14:
        coeffs = coeffs[1:]
    roots = np.roots(coeffs) if len(coeffs) > 1 else np.array([])
    candidates = [2.0 * np.arctan(r.real) for r in roots if abs(r.imag) < 1e-7 * (1.0 + abs(r))]
    candidates.append(np.pi)
    minima = []
    for th in candidates:
        for _ in range(4):
            d1 = 0.5 * np.sin(2.0 * th) + h * np.sin(th - psi)
            d2 = np.cos(2.0 * th) + h * np.cos(th - psi)
            if d2 != 0.0:
                th = th - d1 / d2
        if (abs(0.5 * np.sin(2.0 * th) + h * np.sin(th - psi)) < 1e-10
                and np.cos(2.0 * th) + h * np.cos(th - psi) > 1e-12):
            minima.append(th)
    if h >= -h_sw:
        theta = max(minima, key=np.cos)
    else:
        theta = min(minima, key=np.cos)
    m = float(np.cos(theta - psi))
    return m


def escape_barriers(h, psi):
    '''Reduced energy barriers that keep a particle in its original minimum.

    Inputs:
      h: float or numpy array (any shape), reduced field with |h| < h_sw(psi).
      psi: float, angle in radians between the field axis and the easy axis, 0 <= psi <= pi/2.

    Output:
      (low, high): two values with the shape of np.asarray(h). low <= high are
      e(theta_max) - e(theta_min) for the two energy maxima, theta_min being the original
      minimum of the descending branch. Absolute error below 1e-10 when
      h_sw(psi) - |h| >= 1e-3.

    Raises:
      ValueError if psi is outside [0, pi/2], if any h is not finite, or if any
      |h| >= h_sw(psi).
    '''
    # Other method: no polynomial. The original minimum is bracketed between its fold angle
    # and psi, the other minimum is the original minimum of the field -h turned by pi, and
    # each maximum is bracketed by the two minima (e' > 0 just after a minimum and e' < 0
    # just before the next one, going round the circle).
    h_arr = np.asarray(h, dtype=float)
    psi = float(psi)
    if not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("psi must be between 0 and pi/2")
    if not np.all(np.isfinite(h_arr)):
        raise ValueError("h must be finite")
    if psi == 0.0 or psi == 0.5 * np.pi:
        h_sw = 1.0
    else:
        def curvature_at_fold(t):
            return np.cos(2.0 * t) - np.sin(2.0 * t) * np.cos(t - psi) / (2.0 * np.sin(t - psi))

        t_fold = brentq(curvature_at_fold, -0.5 * np.pi + 1e-12, -1e-12, xtol=1e-15, rtol=1e-15)
        h_sw = float(np.sin(2.0 * t_fold) / (2.0 * np.sin(t_fold - psi)))
    if np.any(np.abs(h_arr) >= h_sw):
        raise ValueError("both energy minima exist only for |h| < h_sw(psi)")

    def slope(t, x):
        return 0.5 * np.sin(2.0 * t) + x * np.sin(t - psi)

    def energy(t, x):
        return 0.5 * np.sin(t) ** 2 - x * np.cos(t - psi)

    def original_minimum(x):
        if psi == 0.0:
            return 0.0
        fold = -0.5 * np.pi if psi == 0.5 * np.pi else -np.arctan(np.tan(psi) ** (1.0 / 3.0))
        lo = fold + 1e-13
        if slope(lo, x) >= 0.0:
            return lo
        return brentq(slope, lo, psi, args=(x,), xtol=1e-15, rtol=1e-15)

    low = np.empty(h_arr.size)
    high = np.empty(h_arr.size)
    for i, x in enumerate(h_arr.ravel()):
        t_orig = original_minimum(x)
        t_other = original_minimum(-x) + np.pi
        e_orig = energy(t_orig, x)
        found = []
        for start, stop in ((t_orig, t_other), (t_other, t_orig + 2.0 * np.pi)):
            gap = 1e-9 * (stop - start)
            t_max = brentq(slope, start + gap, stop - gap, args=(x,), xtol=1e-15, rtol=1e-15)
            found.append(energy(t_max, x) - e_orig)
        low[i], high[i] = min(found), max(found)
    low = low.reshape(h_arr.shape)
    high = high.reshape(h_arr.shape)
    return low, high


def survival_probability(h, psi, a, f0, rate):
    '''Probability that a particle has not left its original minimum when the sweep reaches h.

    Inputs:
      h: float, reduced field reached by the descending sweep.
      psi: float, easy-axis angle in radians, 0 <= psi <= pi/2.
      a: float, thermal stability ratio K V / (k_B T), > 0.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0.

    Output:
      P: float in [0, 1], P = 1 for h >= h_sw(psi) and P = 0 for h <= -h_sw(psi).
         Absolute error below 1e-10.

    Raises:
      ValueError if psi is outside [0, pi/2], if h is not finite, or if a, f0 or rate
      is not a positive finite number.
    '''
    # Other method: the escape integral I(h) = int_h^h_sw Gamma / f0 from an ODE run (DOP853)
    # downwards from h_sw, instead of a quadrature.
    h = float(h)
    psi = float(psi)
    if not np.isfinite(h) or not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("h must be finite and psi must be between 0 and pi/2")
    for name, value in (("a", a), ("f0", f0), ("rate", rate)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("%s must be a positive finite number" % name)
    if psi == 0.0 or psi == 0.5 * np.pi:
        h_sw = 1.0
    else:
        def curvature_at_fold(t):
            return np.cos(2.0 * t) - np.sin(2.0 * t) * np.cos(t - psi) / (2.0 * np.sin(t - psi))

        t_fold = brentq(curvature_at_fold, -0.5 * np.pi + 1e-12, -1e-12, xtol=1e-15, rtol=1e-15)
        h_sw = float(np.sin(2.0 * t_fold) / (2.0 * np.sin(t_fold - psi)))
    if h >= h_sw:
        P = 1.0
        return P
    if h <= -h_sw:
        P = 0.0
        return P

    lam = f0 / rate

    def rhs(x, y):
        if abs(x) >= h_sw:
            return [0.0]
        low, high = escape_barriers(x, psi)
        return [-float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))]

    def gone(x, y):
        # P < exp(-800): the particle has surely left
        return lam * y[0] - 800.0

    gone.terminal = True
    sol = solve_ivp(rhs, (h_sw, h), [0.0], method="DOP853", rtol=1e-12, atol=1e-13 / lam, events=gone)
    if sol.status == 1:
        P = 0.0
        return P
    integral = sol.y[0, -1]
    P = float(np.exp(-f0 / rate * integral))
    return P


def switching_field_statistics(psi, a, f0, rate):
    '''Median and mean switching field of one particle in the descending sweep.

    Inputs:
      psi: float, easy-axis angle in radians, 0 <= psi <= pi/2.
      a: float, thermal stability ratio K V / (k_B T), > 0.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0.

    Output:
      (h_median, h_mean): tuple of two floats, the median and the mean of the field at
      which the particle leaves its original minimum. Absolute errors below 1e-9.

    Raises:
      ValueError if psi is outside [0, pi/2] or if a, f0 or rate is not a positive
      finite number.
    '''
    # Other method: one ODE run from h_sw downwards for I(h) = int_h^h_sw Gamma / f0 and for
    # K(h) = int_h^h_sw P, with dense output; median by bisection on P = exp(-(f0/rate) I).
    psi = float(psi)
    if not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("psi must be between 0 and pi/2")
    for name, value in (("a", a), ("f0", f0), ("rate", rate)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("%s must be a positive finite number" % name)
    if psi == 0.0 or psi == 0.5 * np.pi:
        h_sw = 1.0
    else:
        def curvature_at_fold(t):
            return np.cos(2.0 * t) - np.sin(2.0 * t) * np.cos(t - psi) / (2.0 * np.sin(t - psi))

        t_fold = brentq(curvature_at_fold, -0.5 * np.pi + 1e-12, -1e-12, xtol=1e-15, rtol=1e-15)
        h_sw = float(np.sin(2.0 * t_fold) / (2.0 * np.sin(t_fold - psi)))
    lam = f0 / rate

    def rhs(x, y):
        if abs(x) >= h_sw:
            g = 0.0
        else:
            low, high = escape_barriers(x, psi)
            g = float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))
        return [-g, -np.exp(-lam * y[0])]

    def finished(x, y):
        return lam * y[0] - 80.0

    finished.terminal = True
    sol = solve_ivp(rhs, (h_sw, -h_sw), [0.0, 0.0], method="DOP853", rtol=1e-12,
                    atol=[1e-13 / lam, 1e-15], dense_output=True, events=finished)
    x_end = sol.t[-1]
    lo, hi = x_end, h_sw
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.exp(-lam * sol.sol(mid)[0]) > 0.5:
            hi = mid
        else:
            lo = mid
    h_median = 0.5 * (lo + hi)
    # P is below exp(-80) after x_end, so the rest of the integral and the atom at -h_sw vanish
    h_mean = float(h_sw - sol.y[1, -1])
    result = (float(h_median), h_mean)
    return result


def ensemble_switching(psis, weights, a, f0, rate):
    '''Dynamic coercive field and half-switching field of an ensemble during the sweep.

    Inputs:
      psis: 1-D array of easy-axis angles in radians, each in [0, pi/2].
      weights: 1-D array of the same length, nonnegative, not all zero (normalized by
               their sum).
      a: float, thermal stability ratio K V / (k_B T), > 0.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0.

    Output:
      (h_c, h_half): tuple of two floats, absolute errors below 1e-8.
        h_c: the ensemble magnetization is zero at h = -h_c during the sweep.
        h_half: half of the total weight has left its original minimum at h = -h_half.

    Raises:
      ValueError if psis and weights are not 1-D arrays of the same nonzero length, if
      a psi is outside [0, pi/2], if a weight is negative or not finite, if the weights
      add up to zero, or if a, f0 or rate is not a positive finite number.
    '''
    # Other method: one ODE with dense output per particle for the escape integral, then
    # bisection on the ensemble magnetization and on the switched weight.
    psis = np.asarray(psis, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if psis.ndim != 1 or weights.shape != psis.shape or psis.size == 0:
        raise ValueError("psis and weights must be 1-D arrays of the same nonzero length")
    if not np.all((psis >= 0.0) & (psis <= 0.5 * np.pi)):
        raise ValueError("every psi must be between 0 and pi/2")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("weights must be finite, nonnegative and not all zero")
    for name, value in (("a", a), ("f0", f0), ("rate", rate)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("%s must be a positive finite number" % name)
    w = weights / weights.sum()
    lam = f0 / rate

    def fold_field(psi):
        if psi == 0.0 or psi == 0.5 * np.pi:
            return 1.0

        def curvature_at_fold(t):
            return np.cos(2.0 * t) - np.sin(2.0 * t) * np.cos(t - psi) / (2.0 * np.sin(t - psi))

        t_fold = brentq(curvature_at_fold, -0.5 * np.pi + 1e-12, -1e-12, xtol=1e-15, rtol=1e-15)
        return float(np.sin(2.0 * t_fold) / (2.0 * np.sin(t_fold - psi)))

    runs = []
    for psi in psis:
        psi = float(psi)
        h_sw = fold_field(psi)

        def rhs(x, y, psi=psi, h_sw=h_sw):
            if abs(x) >= h_sw:
                return [0.0]
            low, high = escape_barriers(x, psi)
            return [-float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))]

        def finished(x, y):
            return lam * y[0] - 80.0

        finished.terminal = True
        sol = solve_ivp(rhs, (h_sw, -h_sw), [0.0], method="DOP853", rtol=1e-12, atol=1e-13 / lam,
                        dense_output=True, events=finished)
        runs.append((psi, h_sw, sol.t[-1], sol))

    def survival(x, run):
        psi, h_sw, x_end, sol = run
        if x >= h_sw:
            return 1.0
        if x <= x_end:
            return 0.0
        return float(np.exp(-lam * sol.sol(x)[0]))

    def magnetization(x):
        total = 0.0
        for wi, run in zip(w, runs):
            P = survival(x, run)
            # the other minimum is the original minimum of the field -x turned by pi
            m_other = -branch_magnetization(-x, run[0])
            m_orig = branch_magnetization(x, run[0]) if P > 0.0 else m_other
            total += wi * (P * m_orig + (1.0 - P) * m_other)
        return total

    def left(x):
        return sum(wi * (1.0 - survival(x, run)) for wi, run in zip(w, runs))

    # the magnetization rises with h, the switched weight falls with h; both cross on [-1, 1]
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if magnetization(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    h_c = -0.5 * (lo + hi)
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if left(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    h_half = -0.5 * (lo + hi)
    result = (float(h_c), float(h_half))
    return result


# ---- step 6, second method: Brown's equation in Legendre polynomials, extended precision ----
def _g6_band_solve(rows, rhs, N):
    # Gaussian elimination with partial pivoting on a pentadiagonal system given as dict rows[i] = {j: a_ij}
    A = [dict(r) for r in rows]
    b = list(rhs)
    for k in range(N):
        cand = [i for i in range(k, min(k + 3, N)) if k in A[i]]
        p = max(cand, key=lambda i: abs(A[i][k]))
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
        piv = A[k][k]
        for i in range(k + 1, min(k + 3, N)):
            if k in A[i] and A[i][k] != 0:
                f = A[i][k] / piv
                for j, v in A[k].items():
                    A[i][j] = A[i].get(j, 0) - f * v
                del A[i][k]
                b[i] -= f * b[k]
    x = [mp.mpf(0)] * N
    for k in range(N - 1, -1, -1):
        s = b[k] - mp.fsum(v * x[j] for j, v in A[k].items() if j > k)
        x[k] = s / A[k][k]
    return x

def _g6_legendre(sigma, h, N=None, dps=None):
    if dps is None: dps = 30 + int(0.5 * sigma * (1 + abs(h)) ** 2)
    if N is None: N = int(60 + 3 * sigma)
    with mp.workdps(dps):
        s = mp.mpf(sigma); xi = 2 * s * mp.mpf(h)
        def Mz(v):
            out = [mp.mpf(0)] * (len(v) + 1)
            for l, c in enumerate(v):
                if c == 0: continue
                out[l + 1] += c * (l + 1) / mp.mpf(2 * l + 1)
                if l >= 1: out[l - 1] += c * l / mp.mpf(2 * l + 1)
            return out
        def Tz(v):
            out = [mp.mpf(0)] * (len(v) + 1)
            for l, c in enumerate(v):
                if c == 0 or l == 0: continue
                f = c * l * (l + 1) / mp.mpf(2 * l + 1)
                out[l - 1] += f; out[l + 1] -= f
            return out
        cols = {}
        for l in range(0, N + 1):
            e = [mp.mpf(0)] * (l + 1); e[l] = mp.mpf(1)
            g = [-2 * s * x for x in Mz(e)] + [mp.mpf(0)]
            for k in range(len(e)): g[k] -= xi * e[k]
            Tg, Mg = Tz(g), Mz(g)
            col = {}
            col[l] = col.get(l, 0) - l * (l + 1)
            for k, v in enumerate(Tg): col[k] = col.get(k, 0) + v
            for k, v in enumerate(Mg): col[k] = col.get(k, 0) - 2 * v
            cols[l] = {k: v / 2 for k, v in col.items() if v != 0 and k <= N}
        # L' (rows/cols 1..N), stored by rows
        rows = [dict() for _ in range(N)]
        for l in range(1, N + 1):
            for k, v in cols[l].items():
                if k >= 1: rows[k - 1][l - 1] = v
        # equilibrium density: L' b + L[1:,0] b0 = 0, b0 = 1
        r0 = [-cols[0].get(k, 0) for k in range(1, N + 1)]
        beq = [mp.mpf(1)] + _g6_band_solve(rows, r0, N)
        # lam1: inverse iteration on L'
        x = [mp.mpf(1)] * N
        lam_old = None
        for it in range(60):
            y = _g6_band_solve(rows, x, N)
            nrm = mp.sqrt(mp.fsum(t * t for t in y)); y = [t / nrm for t in y]
            Ay = [mp.fsum(v * y[j] for j, v in rows[i].items()) for i in range(N)]
            lam = mp.fsum(a * b for a, b in zip(y, Ay))
            x = y
            if lam_old is not None and abs(lam - lam_old) < abs(lam) * mp.mpf(10) ** (-(dps - 10)): break
            lam_old = lam
        lam1 = -lam
        # integral relaxation time of z: tau = -int z Psi dz / int z rho0 dz, L Psi = -rho0... (L^{-1} rho0)
        zW = Mz(beq)
        zmean = zW[0] / beq[0]
        rho0 = [zW[k] - zmean * (beq[k] if k < len(beq) else 0) for k in range(N + 1)]
        y = _g6_band_solve(rows, rho0[1:], N)           # L' y = rho0  ->  Psi = -y solves L Psi = -rho0
        tau = -y[0] / rho0[1]                        # int z Psi = (2/3)(-y_1), C(0) = (2/3) rho0_1
        return lam1, tau


def brown_relaxation(sigma, h):
    '''Exact thermal relaxation of one particle with its field along the easy axis (Brown's equation).'''
    # Other method: W = sum b_l P_l(z); the Fokker-Planck operator is pentadiagonal in l and is
    # solved in extended precision (inverse iteration for lam1, a linear solve for tau_int).
    sigma = float(sigma)
    h = float(h)
    if not (np.isfinite(sigma) and np.isfinite(h)):
        raise ValueError("sigma and h must be finite")
    if not (0.0 <= sigma <= 60.0) or abs(h) > 0.9:
        raise ValueError("need 0 <= sigma <= 60 and |h| <= 0.9")
    lam1, tau = _g6_legendre(sigma, h)
    result = (float(lam1), float(tau))
    return result
