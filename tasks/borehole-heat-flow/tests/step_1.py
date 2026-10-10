# Tests for step 1: true_vertical_depth (minimum-curvature TVD).
# Targets: exact straight-hole and constant-build-arc closed forms, the closed form of a constant-azimuth
# segment with inclination linear in MD (written without cancellation), 8-decimal values checked by
# adaptive quadrature of the arc tangent, and adaptive quadrature (scipy.integrate.quad) of the down
# component of the great-circle tangent for random three-dimensional surveys. Accuracy 1e-6 m.

# --- test case 0 ---
# vertical hole
import numpy as np
out = np.asarray(true_vertical_depth([0, 100, 250, 600], [0, 0, 0, 0], [0, 45, 90, 10], [0, 33.3, 600]))
assert np.max(np.abs(np.asarray(out, float) - np.asarray([0, 33.3, 600], float))) <= 1e-6

# --- test case 1 ---
# straight inclined hole [inc = 15.0]
import math
import numpy as np
inc = 15.0
md = [0, 150, 300, 450]
got = true_vertical_depth(md, [inc] * 4, [200.0] * 4, 321.7)
assert abs(got - 321.7 * math.cos(math.radians(inc))) < 1e-6

# --- test case 2 ---
# straight inclined hole [inc = 35.0]
import math
import numpy as np
inc = 35.0
md = [0, 150, 300, 450]
got = true_vertical_depth(md, [inc] * 4, [200.0] * 4, 321.7)
assert abs(got - 321.7 * math.cos(math.radians(inc))) < 1e-6

# --- test case 3 ---
# straight inclined hole [inc = 60.0]
import math
import numpy as np
inc = 60.0
md = [0, 150, 300, 450]
got = true_vertical_depth(md, [inc] * 4, [200.0] * 4, 321.7)
assert abs(got - 321.7 * math.cos(math.radians(inc))) < 1e-6

# --- test case 4 ---
# constant build arc is exact between stations
import math
import numpy as np
# inclination builds uniformly from 0 to 90 deg over 900 m: a circular arc
# of radius R = 900 / (pi/2); TVD(m) = R sin(m / R) at ANY measured depth
L = 900.0
Rad = L / (math.pi / 2)
md = np.array([0.0, 300.0, 600.0, 900.0])
inc = md / L * 90.0
q = np.array([37.5, 137.5, 300.0, 451.2, 777.7, 900.0])
got = np.asarray(true_vertical_depth(md, inc, [45.0] * 4, q))
assert np.max(np.abs(np.asarray(got, float) - np.asarray(Rad * np.sin(q / Rad), float))) <= 1e-6

# --- test case 5 ---
# three dimensional segment
import numpy as np
# build and turn in one segment; values checked by numerical integration
# of the unit tangent along the great-circle arc
got = np.asarray(true_vertical_depth([0, 400], [20, 50], [0, 90], [137.0, 250.0, 400.0]))
assert np.max(np.abs(np.asarray(got, float) - np.asarray([128.63223148, 228.74044021, 341.01702731], float))) <= 1e-6

# --- test case 6 ---
# azimuth turn at constant inclination
import numpy as np
got = np.asarray(true_vertical_depth([0, 100, 200, 300], [40] * 4, [0, 60, 120, 180], [50.0, 175.0, 300.0]))
assert np.max(np.abs(np.asarray(got, float) - np.asarray([39.73029462, 139.32487515, 238.38176775], float))) <= 1e-6

# --- test case 7 ---
# scalar and zero query
import math
import numpy as np
got = true_vertical_depth([0, 100], [30, 30], [10, 10], 0.0)
assert type(got) is float and abs(got) < 1e-6
# a scalar query inside the straight segment: a float equal to MD cos(30 deg)
got = true_vertical_depth([0, 100], [30, 30], [10, 10], 64.0)
assert type(got) is float and abs(got - 64.0 * math.cos(math.radians(30.0))) < 1e-6

# --- test case 8 ---
# array query values match scalar queries
import math
import numpy as np
q = np.array([[10.0, 20.0], [30.0, 40.0]])
md, inc, azi = [0, 100], [10, 12], [0, 0]
out = true_vertical_depth(md, inc, azi, q)
assert isinstance(out, np.ndarray) and out.shape == q.shape, (type(out), np.shape(out))
ref = np.array([[true_vertical_depth(md, inc, azi, float(v)) for v in row] for row in q])
# two computed outputs (array call against scalar calls): twice the 1e-6 m accuracy
assert np.max(np.abs(np.asarray(out, float) - np.asarray(ref, float))) <= 2e-6
# constant azimuth, inclination linear in MD: TVD = (sin(I(m)) - sin(I1)) / k
k = math.radians(2.0) / 100.0
assert np.max(np.abs(np.asarray(out, float) - np.asarray((np.sin(math.radians(10.0) + k * q) - math.sin(math.radians(10.0))) / k, float))) <= 1e-6

# --- test case 9 ---
# tiny dogleg on a long segment [dinc_rad = 8e-10]
import math
import numpy as np

def _mp_build_tvd(inc1_deg, dinc_rad, seg_len, m):
    """Exact TVD on a constant-azimuth segment whose inclination changes by dinc_rad over seg_len
    (inclination linear in MD): (sin(I1 + k m) - sin(I1)) / k written without cancellation as
    m cos(I1 + k m / 2) sinc(k m / 2)."""
    k = dinc_rad / seg_len
    h = 0.5 * k * m
    sinc = 1.0 - h * h / 6.0 if h < 1e-6 else math.sin(h) / h
    return m * math.cos(math.radians(inc1_deg) + h) * sinc

dinc_rad = 8e-10
# a dogleg of 8e-10 rad over 20 km already bends the path by ~4e-6 m of TVD:
# treating a small non-zero dogleg as straight fails the 1e-6 m accuracy
L, inc1 = 20000.0, 30.0
inc2 = inc1 + math.degrees(dinc_rad)
q = np.array([5000.0, 13000.0, 20000.0])
got = np.asarray(true_vertical_depth([0.0, L], [inc1, inc2], [75.0, 75.0], q))
ref = np.array([_mp_build_tvd(inc1, dinc_rad, L, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 10 ---
# tiny dogleg on a long segment [dinc_rad = 3e-8]
import math
import numpy as np

def _mp_build_tvd(inc1_deg, dinc_rad, seg_len, m):
    """Exact TVD on a constant-azimuth segment whose inclination changes by dinc_rad over seg_len
    (inclination linear in MD): (sin(I1 + k m) - sin(I1)) / k written without cancellation as
    m cos(I1 + k m / 2) sinc(k m / 2)."""
    k = dinc_rad / seg_len
    h = 0.5 * k * m
    sinc = 1.0 - h * h / 6.0 if h < 1e-6 else math.sin(h) / h
    return m * math.cos(math.radians(inc1_deg) + h) * sinc

dinc_rad = 3e-8
# a dogleg of 8e-10 rad over 20 km already bends the path by ~4e-6 m of TVD:
# treating a small non-zero dogleg as straight fails the 1e-6 m accuracy
L, inc1 = 20000.0, 30.0
inc2 = inc1 + math.degrees(dinc_rad)
q = np.array([5000.0, 13000.0, 20000.0])
got = np.asarray(true_vertical_depth([0.0, L], [inc1, inc2], [75.0, 75.0], q))
ref = np.array([_mp_build_tvd(inc1, dinc_rad, L, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 11 ---
# tiny dogleg on a long segment [dinc_rad = 2e-6]
import math
import numpy as np

def _mp_build_tvd(inc1_deg, dinc_rad, seg_len, m):
    """Exact TVD on a constant-azimuth segment whose inclination changes by dinc_rad over seg_len
    (inclination linear in MD): (sin(I1 + k m) - sin(I1)) / k written without cancellation as
    m cos(I1 + k m / 2) sinc(k m / 2)."""
    k = dinc_rad / seg_len
    h = 0.5 * k * m
    sinc = 1.0 - h * h / 6.0 if h < 1e-6 else math.sin(h) / h
    return m * math.cos(math.radians(inc1_deg) + h) * sinc

dinc_rad = 2e-6
# a dogleg of 8e-10 rad over 20 km already bends the path by ~4e-6 m of TVD:
# treating a small non-zero dogleg as straight fails the 1e-6 m accuracy
L, inc1 = 20000.0, 30.0
inc2 = inc1 + math.degrees(dinc_rad)
q = np.array([5000.0, 13000.0, 20000.0])
got = np.asarray(true_vertical_depth([0.0, L], [inc1, inc2], [75.0, 75.0], q))
ref = np.array([_mp_build_tvd(inc1, dinc_rad, L, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 12 ---
# random three dimensional surveys [seed = 1]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 13 ---
# random three dimensional surveys [seed = 2]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 14 ---
# random three dimensional surveys [seed = 3]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 15 ---
# random three dimensional surveys [seed = 4]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 16 ---
# random three dimensional surveys [seed = 5]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 17 ---
# random three dimensional surveys [seed = 6]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 18 ---
# random three dimensional surveys [seed = 7]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 19 ---
# random three dimensional surveys [seed = 8]
import math
import numpy as np

def _quad_tvd(md, inc_deg, azi_deg, m):
    """TVD by adaptive quadrature of the down component of the unit tangent along the arcs."""
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
        if beta <= 0.0:                           # straight segment (beta >= 0)
            total += (b - a) * t1[2]
            continue
        u = axis / np.linalg.norm(axis)
        w = np.cross(u, t1)                       # t(phi) = t1 cos(phi) + w sin(phi)
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
assert isinstance(got, np.ndarray) and got.shape == q.shape
ref = np.array([_quad_tvd(md, inc, azi, m) for m in q])
assert np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))) <= 1e-6

# --- test case 20 ---
# dogleg just outside the band is accepted
import math
import numpy as np
# dogleg = 180 deg - 2e-6 rad: outside the 1e-6 rad rejection band, so the arc is defined
inc, azi = [10.0, 170.0 - math.degrees(2e-6)], [0.0, 180.0]
v = true_vertical_depth([0.0, 100.0], inc, azi, [40.0, 100.0])
v = np.asarray(v, float)
# the path is almost a half circle (radius ~ 100/pi m): it turns back up, so the TVD at the
# end of the segment is smaller than at MD 40, and both stay within [0, MD]
assert np.all(np.isfinite(v)) and np.all(v >= -1e-9) and np.all(v <= 100.0 + 1e-9) and v[1] < v[0]

# --- test case 21 ---
# invalid raises
import math
import numpy as np
for case in ["beyond", "negative", "nan", "start", "order", "lengths", "one_station", "two_d", "inc_neg", "inc_180", "dogleg_180", "dogleg_near_180", "md_nan", "inc_nan", "azi_nan"]:
    _t_msg = " (case %r)" % (case,)
    md, inc, azi, q = [0.0, 100.0, 200.0], [10.0, 12.0, 14.0], [0.0, 0.0, 0.0], 50.0
    if case == "beyond":
        q = 250.0
    elif case == "negative":
        q = -1.0
    elif case == "nan":
        q = float("nan")
    elif case == "start":
        md = [5.0, 100.0, 200.0]
    elif case == "order":
        md = [0.0, 200.0, 100.0]
    elif case == "lengths":
        inc = [10.0, 12.0]
    elif case == "one_station":
        md, inc, azi, q = [0.0], [10.0], [0.0], 0.0
    elif case == "two_d":
        md, inc, azi = [[0.0, 100.0, 200.0]], [[10.0, 12.0, 14.0]], [[0.0, 0.0, 0.0]]
    elif case == "inc_neg":
        inc = [-5.0, 12.0, 14.0]
    elif case == "inc_180":
        inc = [10.0, 180.0, 14.0]
    elif case == "dogleg_180":
        inc, azi = [10.0, 170.0, 20.0], [0.0, 180.0, 0.0]
    elif case == "dogleg_near_180":
        # opposite azimuths: dogleg = I1 + I2 = 180 deg - 5e-7 rad, inside the 1e-6 rad band
        inc, azi = [10.0, 170.0 - math.degrees(5e-7), 20.0], [0.0, 180.0, 0.0]
    elif case == "md_nan":
        md = [0.0, float("nan"), 200.0]
    elif case == "inc_nan":
        inc = [10.0, float("nan"), 14.0]
    else:
        azi = [0.0, float("nan"), 0.0]
    try:
        true_vertical_depth(md, inc, azi, q)
    except ValueError:
        pass
    else:
        raise AssertionError("true_vertical_depth must raise ValueError" + _t_msg)
