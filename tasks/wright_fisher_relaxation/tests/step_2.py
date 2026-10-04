import numpy as np
import mpmath as _t_mp

# Independent targets: the Wright-Fisher matrix in extended precision (mpmath binomials) and
# plain LU solves at high precision; no elimination ordering, no subtraction-free tricks.


def _t_P(N, s, u, v):
    # call inside _t_mp.workdps(...)
    s, u, v = _t_mp.mpf(s), _t_mp.mpf(u), _t_mp.mpf(v)
    P = _t_mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        p = _t_mp.mpf(i) / N
        ps = (1 + s) * p / (1 + s * p)
        pm = (1 - u) * ps + v * (1 - ps)
        for j in range(N + 1):
            P[i, j] = _t_mp.binomial(N, j) * pm ** j * (1 - pm) ** (N - j)
    return P


def _t_rel(a, b):
    return abs(a - b) / abs(b)


def _t_absorbing(N, s, dps=80):
    # h_fix, h_loss and the fundamental-matrix products G h by plain LU in extended precision
    with _t_mp.workdps(dps):
        P = _t_P(N, s, 0, 0)
        M = _t_mp.matrix(N - 1, N - 1)
        fix, loss = _t_mp.matrix(N - 1, 1), _t_mp.matrix(N - 1, 1)
        for a in range(1, N):
            fix[a - 1], loss[a - 1] = P[a, N], P[a, 0]
            for b in range(1, N):
                M[a - 1, b - 1] = (1 if a == b else 0) - P[a, b]
        hf, hl = _t_mp.lu_solve(M, fix), _t_mp.lu_solve(M, loss)
        gf, gl = _t_mp.lu_solve(M, hf), _t_mp.lu_solve(M, hl)
        return [(float(hf[k]), float(hl[k]), float(gf[k] / hf[k]), float(gl[k] / hl[k])) for k in range(N - 1)]


def _t_check(out, target, tol=1e-8):
    assert isinstance(out, tuple) and len(out) == 4 and all(isinstance(x, float) for x in out), out
    for a, b in zip(out, target):
        assert _t_rel(a, b) < tol, (out, target)


# --- test case 0: neutral drift: p_fix = i0 / N exactly, also when p_loss is the small one, and N = 2 ---
N = 300
for i0 in (1, 150, 299):
    out = fixation_statistics(N, 0.0, i0)
    assert _t_rel(out[0], i0 / N) < 1e-8 and _t_rel(out[1], (N - i0) / N) < 1e-8, (i0, out)
# smallest population, N = 2: from one copy the next generation has 0, 1 or 2 copies with probabilities
# 1/4, 1/2, 1/4, so p_fix = p_loss = 1/2 and both conditional times are 1 / (1/2) = 2 generations
out = fixation_statistics(2, 0.0, 1)
for a, b in zip(out, (0.5, 0.5, 2.0, 2.0)):
    assert _t_rel(a, b) < 1e-8, out

# --- test case 1: strong selection against A (p_fix ~ 3e-18 from one copy) and for A (p_loss tiny),
# all four quantities against extended precision ---
for N, s in ((30, -0.5), (30, 0.4), (12, 0.05)):
    target = _t_absorbing(N, s)
    for i0 in (1, N // 2, N - 1):
        _t_check(fixation_statistics(N, s, i0), target[i0 - 1])

# --- test case 2: relabeling the alleles: a has relative fitness 1 / (1 + s) = 1 + s' with
# s' = -s / (1 + s), so p_fix(s, i0) = p_loss(s', N - i0) and t_fix(s, i0) = t_loss(s', N - i0),
# exactly; at N = 400 the small probabilities reach about 3e-141 ---
for N, s, s2 in ((400, 0.5, -1.0 / 3.0), (400, 0.25, -0.2), (250, -0.1, 1.0 / 9.0)):
    for i0 in (1, N // 3, N - 1):
        a = fixation_statistics(N, s, i0)
        b = fixation_statistics(N, s2, N - i0)
        assert _t_rel(a[0], b[1]) < 1e-8 and _t_rel(a[1], b[0]) < 1e-8, (N, s, i0, a, b)
        assert _t_rel(a[2], b[3]) < 1e-8 and _t_rel(a[3], b[2]) < 1e-8, (N, s, i0, a, b)
a = fixation_statistics(400, -1.0 / 3.0, 1)
assert 0.0 < a[0] < 1e-100 and _t_rel(a[0] + a[1], 1.0) < 1e-8, a

# --- test case 3: N not an integer in [2, 400], i0 outside [1, N - 1], or s not finite or out of range ---
for _t_bad in ((1, 0.1, 1), (401, 0.1, 1), (10.0, 0.1, 1), (10, 0.1, True), (10, 0.1, 2.0), (10, 0.1, 0), (10, 0.1, 10), (10, 0.7, 3), (10, float("inf"), 3)):
    try:
        fixation_statistics(*_t_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("fixation_statistics%r must raise ValueError" % (_t_bad,))
