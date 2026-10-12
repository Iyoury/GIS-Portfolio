# --- test case 0 ---
# vertical hole
# Targets: exact straight-hole and constant-build-arc closed forms, the closed form of a constant-azimuth segment with
# inclination linear in MD (written without cancellation), 8-decimal values checked by adaptive quadrature of the arc
# tangent, and adaptive quadrature (scipy.integrate.quad) of the down component of the great-circle tangent for random
# three-dimensional surveys. Accuracy 1e-6 m.
import numpy as np


def _crown_case():
    import numpy as np
    out = np.asarray(true_vertical_depth([0, 100, 250, 600], [0, 0, 0, 0], [0, 45, 90, 10], [0, 33.3, 600]))
    assert np.max(np.abs(np.asarray(out, float) - np.asarray([0, 33.3, 600], float))) <= 1e-6


_crown_case()

# --- test case 1 ---
# straight inclined hole [inc = 15.0]
import numpy as np


def _crown_case():
    import math
    import numpy as np
    inc = 15.0
    md = [0, 150, 300, 450]
    got = true_vertical_depth(md, [inc] * 4, [200.0] * 4, 321.7)
    assert abs(got - 321.7 * math.cos(math.radians(inc))) < 1e-6


_crown_case()

# --- test case 2 ---
# straight inclined hole [inc = 35.0]
import numpy as np


def _crown_case():
    import math
    import numpy as np
    inc = 35.0
    md = [0, 150, 300, 450]
    got = true_vertical_depth(md, [inc] * 4, [200.0] * 4, 321.7)
    assert abs(got - 321.7 * math.cos(math.radians(inc))) < 1e-6


_crown_case()

# --- test case 3 ---
# straight inclined hole [inc = 60.0]
import numpy as np


def _crown_case():
    import math
    import numpy as np
    inc = 60.0
    md = [0, 150, 300, 450]
    got = true_vertical_depth(md, [inc] * 4, [200.0] * 4, 321.7)
    assert abs(got - 321.7 * math.cos(math.radians(inc))) < 1e-6


_crown_case()

# --- test case 4 ---
# constant build arc is exact between stations
import numpy as np


def _crown_case():
    import math
    import numpy as np
    # inclination builds uniformly from 0 to 90 deg over 900 m: a circular arc
    # of radius R = 900 / (pi/2); TVD(m) = R sin(m / R) at ANY measured depth
    _t1_L = 900.0
    _t1_Rad = _t1_L / (math.pi / 2)
    md = np.array([0.0, 300.0, 600.0, 900.0])
    inc = md / _t1_L * 90.0
    q = np.array([37.5, 137.5, 300.0, 451.2, 777.7, 900.0])
    got = np.asarray(true_vertical_depth(md, inc, [45.0] * 4, q))
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(_t1_Rad * np.sin(q / _t1_Rad), float))) <= 1e-6


_crown_case()

# --- test case 5 ---
# three dimensional segment
import numpy as np


def _crown_case():
    import numpy as np
    # build and turn in one segment; values checked by numerical integration
    # of the unit tangent along the great-circle arc
    got = np.asarray(true_vertical_depth([0, 400], [20, 50], [0, 90], [137.0, 250.0, 400.0]))
    assert np.max(np.abs(np.asarray(got, float) - np.asarray([128.63223148, 228.74044021, 341.01702731], float))) <= 1e-6


_crown_case()

# --- test case 6 ---
# azimuth turn at constant inclination
import numpy as np


def _crown_case():
    import numpy as np
    got = np.asarray(true_vertical_depth([0, 100, 200, 300], [40] * 4, [0, 60, 120, 180], [50.0, 175.0, 300.0]))
    assert np.max(np.abs(np.asarray(got, float) - np.asarray([39.73029462, 139.32487515, 238.38176775], float))) <= 1e-6


_crown_case()

# --- test case 7 ---
# scalar and zero query
import numpy as np


def _crown_case():
    import math
    import numpy as np
    got = true_vertical_depth([0, 100], [30, 30], [10, 10], 0.0)
    assert abs(got) < 1e-6
    # a scalar query inside the straight segment: MD cos(30 deg)
    got = true_vertical_depth([0, 100], [30, 30], [10, 10], 64.0)
    assert abs(got - 64.0 * math.cos(math.radians(30.0))) < 1e-6


_crown_case()

# --- test case 8 ---
# array query values match scalar queries
import numpy as np


def _crown_case():
    import math
    import numpy as np
    q = np.array([[10.0, 20.0], [30.0, 40.0]])
    md, inc, azi = [0, 100], [10, 12], [0, 0]
    out = true_vertical_depth(md, inc, azi, q)
    one = true_vertical_depth(md, inc, azi, np.array([75.0]))
    ref = [[true_vertical_depth(md, inc, azi, float(v)) for v in row] for row in q]
    # output contract: an array query keeps its shape (a one-element array included), a scalar query gives a float
    assert (isinstance(out, np.ndarray) and out.shape == q.shape and isinstance(one, np.ndarray) and one.shape == (1,)
            and np.issubdtype(out.dtype, np.floating) and np.issubdtype(one.dtype, np.floating)
            and all(type(v) is float for row in ref for v in row))
    # two computed outputs (array call against scalar calls): twice the 1e-6 m accuracy
    assert np.max(np.abs(np.asarray(out, float) - np.asarray(ref, float))) <= 2e-6
    # constant azimuth, inclination linear in MD: TVD = (sin(I(m)) - sin(I1)) / k
    k = math.radians(2.0) / 100.0
    assert np.max(np.abs(np.asarray(out, float) - np.asarray((np.sin(math.radians(10.0) + k * q) - math.sin(math.radians(10.0))) / k, float))) <= 1e-6
    assert abs(float(one[0]) - (math.sin(math.radians(10.0) + k * 75.0) - math.sin(math.radians(10.0))) / k) <= 1e-6


_crown_case()

# --- test case 9 ---
# tiny dogleg on a long segment [dinc_rad = 8e-10]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_exact_build_tvd(inc1_deg, dinc_rad, seg_len, m):
        k = dinc_rad / seg_len
        h = 0.5 * k * m
        sinc = 1.0 - h * h / 6.0 if h < 1e-6 else math.sin(h) / h
        return m * math.cos(math.radians(inc1_deg) + h) * sinc

    dinc_rad = 8e-10
    # a dogleg of 8e-10 rad over 20 km bends the path by about sin(30 deg) dinc L / 2 = 4e-6 m of TVD at the
    # end of the segment: treating a small non-zero dogleg as straight fails the 1e-6 m accuracy
    _t1_L, inc1 = 20000.0, 30.0
    inc2 = inc1 + math.degrees(dinc_rad)
    q = np.array([5000.0, 13000.0, 20000.0])
    got = np.asarray(true_vertical_depth([0.0, _t1_L], [inc1, inc2], [75.0, 75.0], q))
    ref = np.array([_t1_exact_build_tvd(inc1, dinc_rad, _t1_L, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 10 ---
# tiny dogleg on a long segment [dinc_rad = 3e-8]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_exact_build_tvd(inc1_deg, dinc_rad, seg_len, m):
        k = dinc_rad / seg_len
        h = 0.5 * k * m
        sinc = 1.0 - h * h / 6.0 if h < 1e-6 else math.sin(h) / h
        return m * math.cos(math.radians(inc1_deg) + h) * sinc

    dinc_rad = 3e-8
    # a dogleg of 3e-8 rad over 20 km bends the path by about sin(30 deg) dinc L / 2 = 1.5e-4 m of TVD at the
    # end of the segment: treating a small non-zero dogleg as straight fails the 1e-6 m accuracy
    _t1_L, inc1 = 20000.0, 30.0
    inc2 = inc1 + math.degrees(dinc_rad)
    q = np.array([5000.0, 13000.0, 20000.0])
    got = np.asarray(true_vertical_depth([0.0, _t1_L], [inc1, inc2], [75.0, 75.0], q))
    ref = np.array([_t1_exact_build_tvd(inc1, dinc_rad, _t1_L, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 11 ---
# tiny dogleg on a long segment [dinc_rad = 2e-6]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_exact_build_tvd(inc1_deg, dinc_rad, seg_len, m):
        k = dinc_rad / seg_len
        h = 0.5 * k * m
        sinc = 1.0 - h * h / 6.0 if h < 1e-6 else math.sin(h) / h
        return m * math.cos(math.radians(inc1_deg) + h) * sinc

    dinc_rad = 2e-6
    # a dogleg of 2e-6 rad over 20 km bends the path by about sin(30 deg) dinc L / 2 = 1e-2 m of TVD at the
    # end of the segment: treating a small non-zero dogleg as straight fails the 1e-6 m accuracy
    _t1_L, inc1 = 20000.0, 30.0
    inc2 = inc1 + math.degrees(dinc_rad)
    q = np.array([5000.0, 13000.0, 20000.0])
    got = np.asarray(true_vertical_depth([0.0, _t1_L], [inc1, inc2], [75.0, 75.0], q))
    ref = np.array([_t1_exact_build_tvd(inc1, dinc_rad, _t1_L, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 12 ---
# random three dimensional surveys [seed = 1]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 1
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 13 ---
# random three dimensional surveys [seed = 2]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 2
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 14 ---
# random three dimensional surveys [seed = 3]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 3
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 15 ---
# random three dimensional surveys [seed = 4]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 4
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 16 ---
# random three dimensional surveys [seed = 5]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 5
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 17 ---
# random three dimensional surveys [seed = 6]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 6
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 18 ---
# random three dimensional surveys [seed = 7]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 7
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 19 ---
# random three dimensional surveys [seed = 8]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    def _t1_quad_tvd(md, inc_deg, azi_deg, m):
        from scipy.integrate import quad
        inc, azi = np.radians(inc_deg), np.radians(azi_deg)
        t = [np.array([math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]) for a, b in zip(inc, azi)]
        total = 0.0
        for i in range(len(md) - 1):
            a, b = md[i], min(md[i + 1], m)
            if b <= a:
                break
            t1, t2, L = t[i], t[i + 1], md[i + 1] - md[i]
            axis = np.cross(t1, t2)
            beta = math.atan2(np.linalg.norm(axis), t1 @ t2)
            if beta <= 0.0:
                total += (b - a) * t1[2]
                continue
            u = axis / np.linalg.norm(axis)
            w = np.cross(u, t1)
            total += quad(lambda s: t1[2] * math.cos(beta * (s - a) / L) + w[2] * math.sin(beta * (s - a) / L),
                          a, b, epsabs=1e-11, epsrel=1e-13, limit=200)[0]
        return total

    seed = 8
    rng = np.random.default_rng(seed)
    md = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 300.0, 7))])
    inc = np.clip(np.cumsum(rng.normal(0.0, 12.0, 8)) + 5.0, 0.0, 120.0)
    azi = np.cumsum(rng.normal(0.0, 40.0, 8)) % 360.0
    q = np.sort(rng.uniform(0.0, md[-1], 6))
    got = true_vertical_depth(md, inc, azi, q)
    ref = np.array([_t1_quad_tvd(md, inc, azi, m) for m in q])
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6


_crown_case()

# --- test case 20 ---
# dogleg just outside the band is accepted
import numpy as np


def _crown_case():
    import math
    import numpy as np
    # dogleg = 180 deg - 2e-6 rad: outside the 1e-6 rad rejection band, so the arc is defined
    inc, azi = [10.0, 170.0 - math.degrees(2e-6)], [0.0, 180.0]
    v = true_vertical_depth([0.0, 100.0], inc, azi, [40.0, 100.0])
    v = np.asarray(v, float)
    # the path stays in the north-down plane; the angle of the tangent from vertical (positive toward north)
    # decreases at the constant rate k = (pi - 2e-6) / 100 rad per m from theta0 = 10 deg (the short way round), so
    # TVD(m) = (sin(theta0) - sin(theta0 - k m)) / k: 33.6324974019 m at MD 40 and 11.0548560664 m at MD 100
    # (turning the long way round would give 25.99 m and -11.05 m)
    th0, k = math.radians(10.0), (math.pi - 2e-6) / 100.0
    for m, got in zip((40.0, 100.0), v):
        assert abs(got - (math.sin(th0) - math.sin(th0 - k * m)) / k) <= 1e-6, (m, got)


_crown_case()

# --- test case 21 ---
# stated error rules raise ValueError
import numpy as np


def _crown_case():
    import math
    import numpy as np
    # the two stated error rules: a query depth outside the survey (below and above it), and a dogleg within
    # 1e-6 rad of 180 degrees (the arc between opposite directions is not defined)
    for case in ["beyond", "negative", "dogleg_near_180"]:
        _t_msg = " (case %r)" % (case,)
        md, inc, azi, q = [0.0, 100.0, 200.0], [10.0, 12.0, 14.0], [0.0, 0.0, 0.0], 50.0
        if case == "beyond":
            q = 250.0
        elif case == "negative":
            q = -1.0
        else:
            # opposite azimuths: dogleg = I1 + I2 = 180 deg - 5e-7 rad, inside the 1e-6 rad band
            inc, azi = [10.0, 170.0 - math.degrees(5e-7), 20.0], [0.0, 180.0, 0.0]
        try:
            true_vertical_depth(md, inc, azi, q)
        except ValueError:
            pass
        else:
            raise AssertionError("true_vertical_depth must raise ValueError" + _t_msg)


_crown_case()
