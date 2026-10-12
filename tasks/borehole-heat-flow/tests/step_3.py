# --- test case 0 ---
# single step matches erfc [z = 0.0]
# Targets: erfc forms evaluated with math.erfc, and values checked by numerical Duhamel convolution with the half-
# space impulse response; absolute accuracy 1e-6 K per kelvin of the largest |dT|.
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    z = 0.0
    # surface warmed by 1.5 K 10 kyr ago and stayed there
    kappa = 1.2e-6
    _t3_exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * _t3_YEAR)))
    got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
    assert abs(got - _t3_exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|


_crown_case()

# --- test case 1 ---
# single step matches erfc [z = 50.0]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    z = 50.0
    # surface warmed by 1.5 K 10 kyr ago and stayed there
    kappa = 1.2e-6
    _t3_exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * _t3_YEAR)))
    got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
    assert abs(got - _t3_exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|


_crown_case()

# --- test case 2 ---
# single step matches erfc [z = 300.0]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    z = 300.0
    # surface warmed by 1.5 K 10 kyr ago and stayed there
    kappa = 1.2e-6
    _t3_exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * _t3_YEAR)))
    got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
    assert abs(got - _t3_exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|


_crown_case()

# --- test case 3 ---
# single step matches erfc [z = 1000.0]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    z = 1000.0
    # surface warmed by 1.5 K 10 kyr ago and stayed there
    kappa = 1.2e-6
    _t3_exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * _t3_YEAR)))
    got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
    assert abs(got - _t3_exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|


_crown_case()

# --- test case 4 ---
# single step matches erfc [z = 3000.0]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    z = 3000.0
    # surface warmed by 1.5 K 10 kyr ago and stayed there
    kappa = 1.2e-6
    _t3_exp = 1.5 * math.erfc(z / (2 * math.sqrt(kappa * 1.0e4 * _t3_YEAR)))
    got = paleoclimate_perturbation(z, [1.0e4], [1.5], kappa)
    assert abs(got - _t3_exp) < 1.5e-6   # 1e-6 K per K of the largest |dT|


_crown_case()

# --- test case 5 ---
# equal steps collapse to one
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    z = np.array([100.0, 700.0, 1500.0])
    a = np.asarray(paleoclimate_perturbation(z, [2e3, 3e4, 8e4], [-2.0, -2.0, -2.0], 1e-6))
    b = np.asarray(paleoclimate_perturbation(z, [8e4], [-2.0], 1e-6))
    # two computed outputs, each within 1e-6 K per K of the largest |dT| = 2 K
    assert np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) <= 4e-6
    # and the single step itself is the erfc response: -2 erfc(z / (2 sqrt(kappa t)))
    _t3_exp = np.array([-2.0 * math.erfc(v / (2 * math.sqrt(1e-6 * 8e4 * _t3_YEAR))) for v in z])
    assert np.max(np.abs(np.asarray(b, float) - np.asarray(_t3_exp, float))) <= 2e-6


_crown_case()

# --- test case 6 ---
# past pulse
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    # cold interval 10-100 ka only: difference of two erfc responses
    kappa, z = 1.2e-6, 900.0
    _t3_e = lambda t: math.erfc(z / (2 * math.sqrt(kappa * t * _t3_YEAR)))
    got = paleoclimate_perturbation(z, [1.0e4, 1.0e5], [0.0, -6.0], kappa)
    assert abs(got - (-6.0) * (_t3_e(1.0e5) - _t3_e(1.0e4))) < 6e-6


_crown_case()

# --- test case 7 ---
# multistep history
import numpy as np


def _crown_case():
    import numpy as np
    # checked against Duhamel convolution with the half-space impulse response
    got = np.asarray(paleoclimate_perturbation([80.0, 400.0, 1200.0], [1e3, 1.1e4, 9e4, 1.3e5],
                          [0.8, 0.0, -1.0, 0.3], 1.15e-6))
    assert np.max(np.abs(np.asarray(got, float) - np.asarray([0.56809482, -0.10491562, -0.44221609], float))) <= 1e-6


_crown_case()

# --- test case 8 ---
# linearity and decay
import numpy as np


def _crown_case():
    import numpy as np
    z = np.array([10.0, 200.0, 20000.0])
    a = np.asarray(paleoclimate_perturbation(z, [5e3, 5e4], [0.4, -1.0], 1e-6))
    b = np.asarray(paleoclimate_perturbation(z, [5e3, 5e4], [0.8, -2.0], 1e-6))
    assert np.max(np.abs(np.asarray(b, float) - np.asarray(2 * a, float))) <= 4e-6
    assert abs(a[2]) < 1e-6


_crown_case()

# --- test case 9 ---
# two dimensional depth array
import numpy as np


def _crown_case():
    import math
    import numpy as np
    # the output keeps the full shape of z; z = 0 gives the present surface departure dT[0]
    hist_t, hist_d, kappa = [2e3, 2.5e4, 1.1e5], [0.6, -2.5, 0.4], 1.1e-6
    z = np.array([[0.0, 150.0, 400.0], [900.0, 1600.0, 3000.0]])
    got = paleoclimate_perturbation(z, hist_t, hist_d, kappa)
    one = paleoclimate_perturbation(np.array([900.0]), hist_t, hist_d, kappa)
    flat = [paleoclimate_perturbation(float(v), hist_t, hist_d, kappa) for v in z.ravel()]
    # output contract: the shape of z (a one-element z included); a float for a scalar z
    assert (isinstance(got, np.ndarray) and got.shape == (2, 3) and isinstance(one, np.ndarray) and one.shape == (1,)
            and all(type(v) is float for v in flat))
    # two computed outputs, each within 1e-6 K per K of the largest |dT| = 2.5 K
    assert np.max(np.abs(np.asarray(got.ravel(), float) - np.asarray(flat, float))) <= 5e-6
    assert abs(got[0, 0] - 0.6) < 2.5e-6
    _t3_YEAR_S = 365.25 * 86400.0
    edges = [0.0] + [t * _t3_YEAR_S for t in hist_t]
    _t3_exp = 0.0
    for i, d in enumerate(hist_d):
        e_old = math.erfc(900.0 / (2 * math.sqrt(kappa * edges[i + 1])))
        e_new = 0.0 if i == 0 else math.erfc(900.0 / (2 * math.sqrt(kappa * edges[i])))
        _t3_exp += d * (e_old - e_new)
    assert abs(got[1, 0] - _t3_exp) < 2.5e-6 and abs(float(one[0]) - _t3_exp) < 2.5e-6


_crown_case()

# --- test case 10 ---
# a non-positive diffusivity raises ValueError (stated error rule)
import numpy as np


def _crown_case():
    import numpy as np
    try:
        paleoclimate_perturbation(100.0, [1e3, 1e4], [0.5, -1.0], 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("paleoclimate_perturbation must raise ValueError for kappa = 0")


_crown_case()

# --- test case 11 ---
# shallow depths below the surface
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t3_YEAR = 365.25 * 86400.0

    # within a few metres of the surface the response to the latest interval changes on time scales of
    # hours to days (z^2 / kappa): a recent two-step history checked against the erfc forms (math.erfc)
    kappa, t_y, d = 1.0e-6, [2.0, 150.0], [1.2, -0.7]
    z = np.array([0.05, 0.3, 1.0, 2.5])
    edges = [0.0] + [t * _t3_YEAR for t in t_y]
    _t3_exp = []
    for v in z:
        tot = 0.0
        for i, di in enumerate(d):
            e_old = math.erfc(v / (2 * math.sqrt(kappa * edges[i + 1])))
            e_new = 0.0 if i == 0 else math.erfc(v / (2 * math.sqrt(kappa * edges[i])))
            tot += di * (e_old - e_new)
        _t3_exp.append(tot)
    got = np.asarray(paleoclimate_perturbation(z, t_y, d, kappa))
    assert np.max(np.abs(np.asarray(got, float) - np.asarray(_t3_exp, float))) <= 1.2e-6   # 1e-6 K per K, largest |dT| = 1.2


_crown_case()
