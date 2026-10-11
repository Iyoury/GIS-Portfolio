import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import mpmath as mp


def _s_fold_field(psi):
    # Switching field from the fold of the equilibrium curve (no astroid formula). Along e' = 0 the
    # field is h(theta) = -sin(2 theta) / (2 sin(theta - psi)); the descending branch folds at the
    # lowest such field, h = -h_sw, for theta = -pi/2 + d with 0 < d < pi/2. Written in d,
    # h_sw = max_d sin(d) cos(d) / (cos(d) cos(psi) + sin(d) sin(psi)), a sum of positive terms
    # (no cancellation even next to psi = 0 or pi/2), and the fold condition e'' = 0 reads
    # cos(d)**3 cos(psi) = sin(d)**3 sin(psi), solved by bracketing; at the maximum an error in d
    # enters h_sw only to second order.
    if psi == 0.0 or psi == 0.5 * np.pi:
        return 1.0
    cp, sp = np.cos(psi), np.sin(psi)
    d = brentq(lambda t: np.cos(t) ** 3 * cp - np.sin(t) ** 3 * sp, 0.0, 0.5 * np.pi, xtol=1e-16, rtol=1e-15)
    return float(np.sin(d) * np.cos(d) / (np.cos(d) * cp + np.sin(d) * sp))


def branch_magnetization(h, psi):
    '''Magnetization along the field on the zero-temperature descending branch of one particle.

    Inputs:
      h: float, reduced field along the field axis (negative means reversed field).
      psi: float, angle in radians between the field axis and the easy axis, 0 <= psi <= pi/2.

    Output:
      m: float, cos(theta - psi), where theta is the magnetization angle (from the easy
         axis) of the state reached when the field comes down from large positive values
         to h, the particle staying in its local energy minimum until that minimum
         disappears at h = -h_sw(psi). Absolute error below 1e-8 when
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
    h_sw = _s_fold_field(psi)
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
      (low, high): two values with the shape of np.asarray(h): float numpy arrays, or float
      scalars (Python float or numpy floating) for a scalar h. low <= high are
      e(theta_max) - e(theta_min) for the two energy maxima, theta_min being the original
      minimum of the descending branch. Absolute error below 1e-8 when
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
    h_sw = _s_fold_field(psi)
    if np.any(np.abs(h_arr) >= h_sw):
        raise ValueError("both energy minima exist only for |h| < h_sw(psi)")

    def slope(t, x):
        return 0.5 * np.sin(2.0 * t) + x * np.sin(t - psi)

    def energy(t, x):
        return 0.5 * np.sin(t) ** 2 - x * np.cos(t - psi)

    def original_minimum(x):
        if psi == 0.0:
            return 0.0
        if psi == 0.5 * np.pi:
            return np.arcsin(x)          # e' = cos(t) (sin(t) - x): no bracketing near h = 1
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
            if psi == 0.5 * np.pi:
                # the maxima are the zeros of cos(t); near h = 1 they are too close to the
                # minima for a bracket
                t_max = 0.5 * np.pi if start == t_orig else 1.5 * np.pi
            else:
                gap = 1e-9 * (stop - start)
                a_, b_ = start + gap, stop - gap
                if slope(a_, x) > 0.0 and slope(b_, x) < 0.0:
                    t_max = brentq(slope, a_, b_, args=(x,), xtol=1e-15, rtol=1e-15)
                else:
                    # within rounding of |h| = h_sw a minimum and the maximum next to it nearly
                    # merge and the slope signs next to the minimum are not reliable; e is
                    # unimodal between two neighbouring minima, so its maximum is found by
                    # golden-section search (the barrier is insensitive to the angle there)
                    lo_, hi_ = start, stop
                    for _ in range(120):
                        m1 = hi_ - 0.6180339887498949 * (hi_ - lo_)
                        m2 = lo_ + 0.6180339887498949 * (hi_ - lo_)
                        if energy(m1, x) < energy(m2, x):
                            lo_ = m1
                        else:
                            hi_ = m2
                    t_max = 0.5 * (lo_ + hi_)
            found.append(energy(t_max, x) - e_orig)
        low[i], high[i] = min(found), max(found)
    low = low.reshape(h_arr.shape)[()]
    high = high.reshape(h_arr.shape)[()]
    return low, high


def survival_probability(h, psi, a, f0, rate):
    '''Probability that a particle has not left its original minimum when the sweep reaches h.

    Inputs:
      h: float, reduced field reached by the descending sweep.
      psi: float, easy-axis angle in radians, 0 <= psi <= pi/2.
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      P: float in [0, 1], P = 1 for h >= h_sw(psi) and P = 0 for h <= -h_sw(psi).
         Absolute error below 1e-8; for 0 < psi < pi/2 not required at fields less than
         1e-6 below h_sw(psi).

    Raises:
      ValueError if psi is outside [0, pi/2], if h is not finite, if a, f0 or rate is
      not a positive finite number, if a is outside [40, 1000] or if f0 / rate is
      outside [1e5, 1e13].
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
    if not (40.0 <= a <= 1000.0 and 1e5 <= f0 / rate <= 1e13):
        raise ValueError("need 40 <= a <= 1000 and 1e5 <= f0 / rate <= 1e13")
    h_sw = _s_fold_field(psi)
    if h >= h_sw:
        P = 1.0
        return P
    if h <= -h_sw:
        P = 0.0
        return P

    lam = f0 / rate

    def rhs(x, y):
        # the run covers [h, h_sw]; at its ends the rate is taken as its limit from inside (at and
        # near psi = pi/2 both minima and a maximum (nearly) merge at h_sw and the rate tends to
        # f0 there, not to 0)
        inside = np.nextafter(h_sw, 0.0)
        x = min(max(x, -inside), inside)
        low, high = escape_barriers(x, psi)
        return [-float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))]

    def gone(x, y):
        # P < exp(-800): the particle has surely left
        return lam * y[0] - 800.0

    gone.terminal = True
    sol = solve_ivp(rhs, (h_sw, h), [0.0], method="DOP853", rtol=1e-12, atol=1e-13 / lam, events=gone)
    if sol.status < 0:
        raise RuntimeError(sol.message)
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
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      (h_median, h_mean): tuple of two floats, the median and the mean of the field at
      which the particle leaves its original minimum. Absolute errors below 1e-7.

    Raises:
      ValueError if psi is outside [0, pi/2], if a, f0 or rate is not a positive finite
      number, if a is outside [40, 1000] or if f0 / rate is outside [1e5, 1e13].
    '''
    # Other method: u = (f0 / rate) I(H_s), with I(h) = int_h^h_sw Gamma / f0, is an exponential
    # random variable with mean 1 (P = exp(-u)). With s = ln u the switching field is a monotone
    # function h(s), the field where (f0 / rate) I = e^s; its median is h(ln ln 2) and its mean
    # int h(s) exp(s - e^s) ds (a Gumbel weight), plus the atom exp(-u_max) at -h_sw. One DOP853 run for
    # I(h) (dense output, purely relative accuracy down to (f0 / rate) I ~ 1e-30) gives h(s) by
    # bisection on the dense output, all quadrature nodes at once. The s-integral runs from s = -40
    # (the weight below is e^-40) to ln u_end and is taken by tanh-sinh quadrature, its step halved until
    # the result settles to 1e-12. h(s) is nearly a jump where the escape rate has a deep valley between
    # two switching windows (psi above about pi/4: one window at positive fields, one towards -h_sw); the
    # valley lies near h = 0, where both barriers are 1/2, so the s-range is split at the s of h = 0 and
    # the steep part sits at the ends of two tanh-sinh ranges, where the nodes cluster.
    psi = float(psi)
    if not (0.0 <= psi <= 0.5 * np.pi):
        raise ValueError("psi must be between 0 and pi/2")
    for name, value in (("a", a), ("f0", f0), ("rate", rate)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("%s must be a positive finite number" % name)
    if not (40.0 <= a <= 1000.0 and 1e5 <= f0 / rate <= 1e13):
        raise ValueError("need 40 <= a <= 1000 and 1e5 <= f0 / rate <= 1e13")
    h_sw = _s_fold_field(psi)
    lam = f0 / rate
    inside = np.nextafter(h_sw, 0.0)

    def gamma_over_f0(x):
        # the rate at the ends of [-h_sw, h_sw] is taken as its limit from inside (at and near
        # psi = pi/2 both minima and a maximum (nearly) merge at h_sw and the rate tends to f0)
        x = min(max(x, -inside), inside)
        low, high = escape_barriers(x, psi)
        return float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))

    def rhs_i(x, y):
        return [-gamma_over_f0(x)]

    def far_enough(x, y):
        return lam * y[0] - 200.0

    far_enough.terminal = True
    run = solve_ivp(rhs_i, (h_sw, -h_sw), [0.0], method="DOP853", rtol=1e-12, atol=1e-30 / lam,
                    dense_output=True, events=far_enough)
    if run.status < 0:
        raise RuntimeError(run.message)
    u_end = lam * run.y[0, -1]

    def field_at(u):
        # the fields where (f0 / rate) I = u (an array), by bisection on the dense output (I falls with h)
        u = np.asarray(u, dtype=float)
        lo = np.full(u.shape, run.t[-1])
        hi = np.full(u.shape, h_sw)
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            below = lam * run.sol(mid)[0] < u
            hi = np.where(below, mid, hi)
            lo = np.where(below, lo, mid)
        return 0.5 * (lo + hi)

    h_median = float(field_at(np.array([np.log(2.0)]))[0])

    s_low, s_top = -40.0, float(np.log(u_end))
    cuts = [s_low, s_top]
    if run.t[-1] < 0.0 and run.sol(0.0)[0] > 0.0:
        s_zero = float(np.log(lam * run.sol(0.0)[0]))
        if s_low < s_zero < s_top:
            cuts.insert(1, s_zero)
    previous = None
    for level in range(3, 10):
        # tanh-sinh nodes on each range: t on [-3.5, 3.5] with step 2^-level; the position is written
        # as a fraction of the range from its lower end, (1 + tanh(g)) / 2 = 1 / (1 + exp(-2 g))
        step = 2.0 ** -level
        t = step * np.arange(-int(3.5 / step), int(3.5 / step) + 1)
        g = 0.5 * np.pi * np.sinh(t)
        frac = 1.0 / (1.0 + np.exp(-2.0 * g))
        weight = step * 0.5 * np.pi * np.cosh(t) / np.cosh(g) ** 2
        s = np.concatenate([sa + (sb - sa) * frac for sa, sb in zip(cuts[:-1], cuts[1:])])
        ws = np.concatenate([0.5 * (sb - sa) * weight for sa, sb in zip(cuts[:-1], cuts[1:])])
        value = float(np.sum(ws * field_at(np.exp(s)) * np.exp(s - np.exp(s))))
        if previous is not None and abs(value - previous) < 1e-12:
            break
        previous = value
    else:
        raise RuntimeError("the tanh-sinh quadrature of the mean did not settle")
    # below s_low the field is h_sw to within rounding; if the run reached -h_sw, the particle is there
    # with the probability exp(-u_end) (if it stopped at u = 200, that remainder is negligible)
    atom = np.exp(-u_end) if run.status == 0 else 0.0
    h_mean = float(value + h_sw * (1.0 - np.exp(-np.exp(s_low))) - h_sw * atom)
    result = (float(h_median), h_mean)
    return result


# ---- step 5, second method: finite-volume chain with an inertia count for lam1, and Garanin's
# quadrature with the inner integral in closed form (erfi) and the outer one by tanh-sinh
# quadrature in extended precision for tau_int ----
def _s5_chain(sigma, h, n):
    # Finite-volume discretization in theta (z = cos theta) of the Sturm-Liouville form
    # (sin(t) w g')' = -2 lam sin(t) w g on [0, pi]: n cells of width pi / n with nodes at the cell
    # centres, cell masses int w sin(t) dt and two-point conductances 1 / int dt / (sin(t) w)
    # between neighbouring nodes (exact for a constant flux), both by 16-point Gauss-Legendre.
    # w = exp(-beta E) is scaled to 1 at z = 1 (h >= 0 here, so that is the deep well).
    edges = np.pi * np.arange(n + 1) / n
    nodes = 0.5 * (edges[:-1] + edges[1:])
    xg, wg = np.polynomial.legendre.leggauss(16)
    top = sigma + 2.0 * sigma * h

    def log_w(t):
        z = np.cos(t)
        return sigma * z * z + 2.0 * sigma * h * z - top

    lo, hi = edges[:-1, None], edges[1:, None]
    tq = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xg[None, :]
    mass = 0.5 * (hi - lo)[:, 0] * np.sum(wg[None, :] * np.exp(log_w(tq)) * np.sin(tq), axis=1)
    lo, hi = nodes[:-1, None], nodes[1:, None]
    tq = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xg[None, :]
    resistance = 0.5 * (hi - lo)[:, 0] * np.sum(wg[None, :] * np.exp(-log_w(tq)) / np.sin(tq), axis=1)
    return mass, 1.0 / resistance


def _s5_count_below(mu, mass, cond):
    # Number of eigenvalues of the chain, A g = mu M g (A the weighted graph Laplacian), below each
    # entry of mu: Sylvester's law of inertia on the LDL^T pivots of A - mu M. Each pivot is written
    # as d_i = cond_i + t_i, which removes the cancellation of the Laplacian rows analytically:
    # t_0 = -mu m_0, t_i = cond_{i-1} t_{i-1} / (cond_{i-1} + t_{i-1}) - mu m_i, d_last = t_last.
    # Without subtractions of nearly equal numbers the count stays right for mu ~ 1e-24 next to
    # conductances of order n**2.
    t = -mu * mass[0]
    count = np.zeros(mu.shape, dtype=int)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(1, mass.size):
            d = cond[i - 1] + t
            count += d < 0.0
            d = np.where(d == 0.0, -1e-300, d)
            t = cond[i - 1] * t / d - mu * mass[i]
    count += t < 0.0
    return count


def _s5_slowest_rate(sigma, h, n):
    # lam1 of the n-cell chain: the smallest mu with two eigenvalues (0 and mu1) below it, by
    # multisection in log(mu) between 1e-60 and 1e4; lam1 = mu1 / 2
    mass, cond = _s5_chain(sigma, h, n)
    lo, hi = np.log(1e-60), np.log(1e4)
    for _ in range(14):
        grid = np.linspace(lo, hi, 66)[1:-1]
        above = _s5_count_below(np.exp(grid), mass, cond) >= 2
        k = int(np.argmax(above)) if np.any(above) else grid.size
        lo, hi = (grid[k - 1] if k > 0 else lo), (grid[k] if k < grid.size else hi)
        if hi - lo < 1e-14:
            break
    return 0.5 * np.exp(0.5 * (lo + hi))


def _s5_tau_int(sigma, h):
    # Garanin: tau = 2 int Q^2 / ((1 - z^2) w) dz / int (z - <z>)^2 w dz. With w = exp(sigma (z + h)^2
    # - c0) the inner integrals are closed forms: int w dz with erfi, int (z + h) w dz = w / (2 sigma).
    # Q is taken from the end z = -1 of the shallow well (h >= 0), where 1 / w is large; the outer
    # integrals use tanh-sinh quadrature in extended precision, split at the barrier z = -h and near
    # both wells.
    if sigma == 0.0:
        return 1.0
    dps = 30 + int(max(0.0, -np.log10(sigma)))
    with mp.workdps(dps):
        s = mp.mpf(sigma)
        hh = mp.mpf(h)
        rs = mp.sqrt(s)
        c0 = s * (1 + hh) ** 2

        def w(z):
            return mp.exp(s * (z + hh) ** 2 - c0)

        def int_w(lo, hi):
            return mp.sqrt(mp.pi) / (2 * rs) * mp.exp(-c0) * (mp.erfi(rs * (hi + hh)) - mp.erfi(rs * (lo + hh)))

        Z = int_w(-1, 1)
        zbar = ((w(1) - w(-1)) / (2 * s) - hh * Z) / Z

        def Q(z):
            return (w(z) - w(-1)) / (2 * s) - (hh + zbar) * int_w(-1, z)

        cuts = sorted({mp.mpf(-1), -1 + min(1 / (2 * s * (1 - hh) + 1), (1 - hh) / 2), -hh,
                       1 - min(1 / (2 * s * (1 + hh) + 1), (1 + hh) / 2), mp.mpf(1)})
        num = mp.quad(lambda z: Q(z) ** 2 / ((1 - z * z) * w(z)), cuts)
        den = mp.quad(lambda z: (z - zbar) ** 2 * w(z), cuts)
        return float(2 * num / den)


def brown_relaxation(sigma, h):
    '''Exact thermal relaxation of one particle with its field along the easy axis (Brown's equation).

    Inputs:
      sigma: float, K V / (k_B T), 0 <= sigma <= 60.
      h: float, reduced field H / H_K along the easy axis, -0.9 <= h <= 0.9.

    Output:
      (lam1, tau_int): tuple of two Python floats, in units of 1/tau_N and tau_N, respectively.
        lam1: smallest nonzero eigenvalue of the Fokker-Planck operator (relaxation rate times
              tau_N), relative error below 1e-5.
        tau_int: integral relaxation time of z = cos(theta) divided by tau_N,
                 int_0^inf C(t) dt / C(0) with C(t) = <z(t) z(0)> - <z>**2 in equilibrium,
                 relative error below 1e-5.

    Raises:
      ValueError if sigma or h is not finite, if sigma is outside [0, 60] or if |h| > 0.9.
    '''
    # Other method: lam1 from the inertia count of a finite-volume chain (2000 and 4000 cells,
    # Richardson extrapolation of the O(cell**2) error); tau_int from Garanin's quadrature with
    # closed-form inner integrals and extended-precision outer quadrature.
    sigma = float(sigma)
    h = float(h)
    if not (np.isfinite(sigma) and np.isfinite(h)):
        raise ValueError("sigma and h must be finite")
    if not (0.0 <= sigma <= 60.0) or abs(h) > 0.9:
        raise ValueError("need 0 <= sigma <= 60 and |h| <= 0.9")
    h = abs(h)                     # z -> -z maps h onto -h
    coarse = _s5_slowest_rate(sigma, h, 2000)
    fine = _s5_slowest_rate(sigma, h, 4000)
    lam1 = (4.0 * fine - coarse) / 3.0
    tau = _s5_tau_int(sigma, h)
    result = (float(lam1), float(tau))
    return result


def ensemble_switching(psis, weights, a, f0, rate):
    '''Dynamic coercive field and half-switching field of an ensemble during the sweep.

    Inputs:
      psis: 1-D array of easy-axis angles in radians, each in [0, pi/2].
      weights: 1-D array of the same length, nonnegative, not all zero (normalized by
               their sum).
      a: float, thermal stability ratio K V / (k_B T), 40 <= a <= 1000.
      f0: float, attempt frequency in 1/s, > 0.
      rate: float, sweep rate |dh/dt| in units of H_K per second, > 0, with
            1e5 <= f0 / rate <= 1e13.

    Output:
      (h_c, h_half): tuple of two floats, absolute errors below 1e-6.
        h_c: the ensemble magnetization is zero at h = -h_c during the sweep.
        h_half: half of the total weight has left its original minimum at h = -h_half.

    Raises:
      ValueError if psis and weights are not 1-D arrays of the same nonzero length, if
      a psi is outside [0, pi/2], if a weight is negative or not finite, if the weights
      add up to zero, if a, f0 or rate is not a positive finite number, if a is outside
      [40, 1000] or if f0 / rate is outside [1e5, 1e13].
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
    if not (40.0 <= a <= 1000.0 and 1e5 <= f0 / rate <= 1e13):
        raise ValueError("need 40 <= a <= 1000 and 1e5 <= f0 / rate <= 1e13")
    w = weights / weights.sum()
    lam = f0 / rate

    runs = []
    for psi in psis:
        psi = float(psi)
        h_sw = _s_fold_field(psi)

        def rhs(x, y, psi=psi, h_sw=h_sw):
            # at the ends of [-h_sw, h_sw] the rate is taken as its limit from inside (at and near
            # psi = pi/2 it tends to f0 there, not to 0)
            inside = np.nextafter(h_sw, 0.0)
            x = min(max(x, -inside), inside)
            low, high = escape_barriers(x, psi)
            return [-float(np.exp(-2.0 * a * low) + np.exp(-2.0 * a * high))]

        def finished(x, y):
            return lam * y[0] - 80.0

        finished.terminal = True
        sol = solve_ivp(rhs, (h_sw, -h_sw), [0.0], method="DOP853", rtol=1e-12, atol=1e-13 / lam,
                        dense_output=True, events=finished)
        if sol.status < 0:
            raise RuntimeError(sol.message)
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
