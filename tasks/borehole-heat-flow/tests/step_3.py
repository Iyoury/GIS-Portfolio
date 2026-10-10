# Tests for step 3: paleoclimate_perturbation (half-space response to a piecewise-constant history).
# Targets: erfc forms evaluated with math.erfc, and values checked by numerical Duhamel convolution with the
# half-space impulse response; absolute accuracy 1e-6 K per kelvin of the largest |dT|.

# --- test case 0 ---
# single step matches erfc [z = 0.0]
import math
import numpy as np

YEAR = 365.25 * 86400.0

z = 0.0
# surface warmed by 1.5 K 10 kyr ago and stayed there
kappa = 1.2e-6
exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * YEAR)))
got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
assert type(got) is float
assert abs(got - exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|

# --- test case 1 ---
# single step matches erfc [z = 50.0]
import math
import numpy as np

YEAR = 365.25 * 86400.0

z = 50.0
# surface warmed by 1.5 K 10 kyr ago and stayed there
kappa = 1.2e-6
exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * YEAR)))
got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
assert type(got) is float
assert abs(got - exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|

# --- test case 2 ---
# single step matches erfc [z = 300.0]
import math
import numpy as np

YEAR = 365.25 * 86400.0

z = 300.0
# surface warmed by 1.5 K 10 kyr ago and stayed there
kappa = 1.2e-6
exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * YEAR)))
got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
assert type(got) is float
assert abs(got - exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|

# --- test case 3 ---
# single step matches erfc [z = 1000.0]
import math
import numpy as np

YEAR = 365.25 * 86400.0

z = 1000.0
# surface warmed by 1.5 K 10 kyr ago and stayed there
kappa = 1.2e-6
exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * YEAR)))
got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
assert type(got) is float
assert abs(got - exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|

# --- test case 4 ---
# single step matches erfc [z = 3000.0]
import math
import numpy as np

YEAR = 365.25 * 86400.0

z = 3000.0
# surface warmed by 1.5 K 10 kyr ago and stayed there
kappa = 1.2e-6
exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * YEAR)))
got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
assert type(got) is float
assert abs(got - exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|

# --- test case 5 ---
# equal steps collapse to one
import math
import numpy as np

YEAR = 365.25 * 86400.0

z = np.array([100.0, 700.0, 1500.0])
a = np.asarray(paleoclimate_perturbation(z, [2e3, 3e4, 8e4], [-2.0, -2.0, -2.0], 1e-6))
b = np.asarray(paleoclimate_perturbation(z, [8e4], [-2.0], 1e-6))
# two computed outputs, each within 1e-6 K per K of the largest |dT| = 2 K
assert np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) <= 4e-6
# and the single step itself is the erfc response: -2 erfc(z / (2 sqrt(kappa t)))
exp = np.array([-2.0 * math.erfc(v / (2 * math.sqrt(1e-6 * 8e4 * YEAR))) for v in z])
assert np.max(np.abs(np.asarray(b, float) - np.asarray(exp, float))) <= 2e-6

# --- test case 6 ---
# past pulse
import math
import numpy as np

YEAR = 365.25 * 86400.0

# cold interval 10-100 ka only: difference of two erfc responses
kappa, z = 1.2e-6, 900.0
e = lambda t: math.erfc(z / (2 * math.sqrt(kappa * t * YEAR)))
got = paleoclimate_perturbation(z, [1.0e4, 1.0e5], [0.0, -6.0], kappa)
assert abs(got - (-6.0) * (e(1.0e5) - e(1.0e4))) < 6e-6

# --- test case 7 ---
# multistep history
import numpy as np
# checked against Duhamel convolution with the half-space impulse response
got = np.asarray(paleoclimate_perturbation([80.0, 400.0, 1200.0], [1e3, 1.1e4, 9e4, 1.3e5],
                      [0.8, 0.0, -1.0, 0.3], 1.15e-6))
assert np.max(np.abs(np.asarray(got, float) - np.asarray([0.56809482, -0.10491562, -0.44221609], float))) <= 1e-6

# --- test case 8 ---
# linearity and decay
import numpy as np
z = np.array([10.0, 200.0, 20000.0])
a = np.asarray(paleoclimate_perturbation(z, [5e3, 5e4], [0.4, -1.0], 1e-6))
b = np.asarray(paleoclimate_perturbation(z, [5e3, 5e4], [0.8, -2.0], 1e-6))
assert np.max(np.abs(np.asarray(b, float) - np.asarray(2 * a, float))) <= 4e-6
assert abs(a[2]) < 1e-6

# --- test case 9 ---
# two dimensional depth array
import math
import numpy as np
# the output keeps the full shape of z; z = 0 gives the present surface departure dT[0]
hist_t, hist_d, kappa = [2e3, 2.5e4, 1.1e5], [0.6, -2.5, 0.4], 1.1e-6
z = np.array([[0.0, 150.0, 400.0], [900.0, 1600.0, 3000.0]])
got = paleoclimate_perturbation(z, hist_t, hist_d, kappa)
assert isinstance(got, np.ndarray) and got.shape == (2, 3)
flat = np.array([paleoclimate_perturbation(float(v), hist_t, hist_d, kappa) for v in z.ravel()])
# two computed outputs, each within 1e-6 K per K of the largest |dT| = 2.5 K
assert np.max(np.abs(np.asarray(got.ravel(), float) - np.asarray(flat, float))) <= 5e-6
assert abs(got[0, 0] - 0.6) < 2.5e-6
YEAR_S = 365.25 * 86400.0
edges = [0.0] + [t * YEAR_S for t in hist_t]
exp = 0.0
for i, d in enumerate(hist_d):
    e_old = math.erfc(900.0 / (2 * math.sqrt(kappa * edges[i + 1])))
    e_new = 0.0 if i == 0 else math.erfc(900.0 / (2 * math.sqrt(kappa * edges[i])))
    exp += d * (e_old - e_new)
assert abs(got[1, 0] - exp) < 2.5e-6

# --- test case 10 ---
# invalid raises
import numpy as np
for case in ["t0", "order", "t_nan", "dT_nan", "kappa", "kappa_nan", "len", "two_d", "zneg", "znan"]:
    _t_msg = " (case %r)" % (case,)
    z, t, d, k = 100.0, [1e3, 1e4], [0.5, -1.0], 1e-6
    if case == "t0":
        t = [0.0, 1e4]
    elif case == "order":
        t = [1e4, 1e3]
    elif case == "t_nan":
        t = [1e3, float("nan")]
    elif case == "dT_nan":
        d = [0.5, float("nan")]
    elif case == "kappa":
        k = 0.0
    elif case == "kappa_nan":
        k = float("nan")
    elif case == "len":
        d = [0.5]
    elif case == "two_d":
        t, d = [[1e3, 1e4]], [[0.5, -1.0]]
    elif case == "zneg":
        z = -1.0
    else:
        z = float("nan")
    try:
        paleoclimate_perturbation(z, t, d, k)
    except ValueError:
        pass
    else:
        raise AssertionError("paleoclimate_perturbation must raise ValueError" + _t_msg)
