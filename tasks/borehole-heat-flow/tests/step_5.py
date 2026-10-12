# --- test case 0 ---
# uniform column is the half space
# Targets: the erfc half-space for a uniform column, the image series of Carslaw and Jaeger (1959, sec. 12.8) for a
# layer over a half-space evaluated with math.erfc, and 90-digit mpmath values (transfer-matrix transform, mpmath
# Talbot inversion) stored inline for two multilayer columns and for six depths of a 50-layer column (forward transfer
# matrices, Talbot and Gaver-Stehfest inversion in mpmath). Accuracy 1e-8 K per kelvin of the largest |dT|;
# comparisons between two computed outputs use twice that.
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t5_YEAR = 365.25 * 86400.0

    def _t5_halfspace(z, kappa, t_years, dT):
        edges = [0.0] + [t * _t5_YEAR for t in t_years]
        if z <= 0.0:
            return dT[0]
        tot = 0.0
        for i, d in enumerate(dT):
            e_old = math.erfc(z / (2 * math.sqrt(kappa * edges[i + 1])))
            e_new = 0.0 if i == 0 else math.erfc(z / (2 * math.sqrt(kappa * edges[i])))
            tot += d * (e_old - e_new)
        return tot

    # equal conductivities in every layer: kappa = k / rho_c everywhere, the erfc solution of step 3
    tops, k, rho_c = [0.0, 300.0, 800.0], [3.0, 3.0, 3.0], 2.5e6
    t_y, d = [1e3, 1.1e4, 9e4, 1.3e5], [0.8, 0.0, -1.0, 0.3]
    z = [0.0, 1.0, 50.0, 299.9, 300.0, 300.1, 700.0, 1500.0, 3000.0]
    got = np.asarray(layered_paleoclimate_perturbation(np.array(z), tops, k, rho_c, t_y, d))
    _t5_exp = [_t5_halfspace(v, 3.0 / 2.5e6, t_y, d) for v in z]
    assert np.max(np.abs(got - np.array(_t5_exp))) < 1e-8      # 1e-8 K per K of the largest |dT|
    # a single layer (the smallest valid column) is the same half-space
    got = np.asarray(layered_paleoclimate_perturbation(np.array(z), [0.0], [3.0], rho_c, t_y, d))
    assert np.max(np.abs(got - np.array(_t5_exp))) < 1e-8


_crown_case()

# --- test case 1 ---
# layer over half space [case = (2.0, 5.0, 400.0)]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t5_YEAR = 365.25 * 86400.0

    def _t5_two_layer_step(x, t, l, k1, k2, rho_c):
        a1, a2 = k1 / rho_c, k2 / rho_c
        al = (math.sqrt(k1 * rho_c) - math.sqrt(k2 * rho_c)) / (math.sqrt(k1 * rho_c) + math.sqrt(k2 * rho_c))
        tot, n = 0.0, 0
        while True:
            if x <= l:
                term = (-al) ** n * (math.erfc((2 * n * l + x) / (2 * math.sqrt(a1 * t)))
                                     + al * math.erfc((2 * (n + 1) * l - x) / (2 * math.sqrt(a1 * t))))
            else:
                term = (1 + al) * (-al) ** n * math.erfc(((2 * n + 1) * l / math.sqrt(a1) + (x - l) / math.sqrt(a2))
                                                         / (2 * math.sqrt(t)))
            tot += term
            n += 1
            if abs(al) ** n < 1e-18 or n > 2000:
                return tot

    def _t5_two_layer(x, l, k1, k2, rho_c, t_years, dT):
        if x <= 0.0:
            return dT[0]
        edges = [0.0] + [t * _t5_YEAR for t in t_years]
        tot = 0.0
        for i, d in enumerate(dT):
            u_old = _t5_two_layer_step(x, edges[i + 1], l, k1, k2, rho_c)
            u_new = 0.0 if i == 0 else _t5_two_layer_step(x, edges[i], l, k1, k2, rho_c)
            tot += d * (u_old - u_new)
        return tot

    case = (2.0, 5.0, 400.0)
    # a poorly and a highly conducting cover, and a thin cover: image series of the step response
    k1, k2, l = case
    rho_c, t_y, d = 2.4e6, [300.0, 1.2e4, 1.0e5, 1.25e5], [0.5, 0.0, -6.0, 1.5]
    z = [0.0, 3.0, 0.5 * l, 0.999 * l, l, 1.001 * l, 2.0 * l, l + 900.0, l + 2500.0]
    got = np.asarray(layered_paleoclimate_perturbation(np.array(z), [0.0, l], [k1, k2], rho_c, t_y, d))
    _t5_exp = [_t5_two_layer(v, l, k1, k2, rho_c, t_y, d) for v in z]
    assert np.max(np.abs(got - np.array(_t5_exp))) < 6e-8      # 1e-8 K per K, largest |dT| = 6


_crown_case()

# --- test case 2 ---
# layer over half space [case = (6.0, 0.8, 400.0)]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t5_YEAR = 365.25 * 86400.0

    def _t5_two_layer_step(x, t, l, k1, k2, rho_c):
        a1, a2 = k1 / rho_c, k2 / rho_c
        al = (math.sqrt(k1 * rho_c) - math.sqrt(k2 * rho_c)) / (math.sqrt(k1 * rho_c) + math.sqrt(k2 * rho_c))
        tot, n = 0.0, 0
        while True:
            if x <= l:
                term = (-al) ** n * (math.erfc((2 * n * l + x) / (2 * math.sqrt(a1 * t)))
                                     + al * math.erfc((2 * (n + 1) * l - x) / (2 * math.sqrt(a1 * t))))
            else:
                term = (1 + al) * (-al) ** n * math.erfc(((2 * n + 1) * l / math.sqrt(a1) + (x - l) / math.sqrt(a2))
                                                         / (2 * math.sqrt(t)))
            tot += term
            n += 1
            if abs(al) ** n < 1e-18 or n > 2000:
                return tot

    def _t5_two_layer(x, l, k1, k2, rho_c, t_years, dT):
        if x <= 0.0:
            return dT[0]
        edges = [0.0] + [t * _t5_YEAR for t in t_years]
        tot = 0.0
        for i, d in enumerate(dT):
            u_old = _t5_two_layer_step(x, edges[i + 1], l, k1, k2, rho_c)
            u_new = 0.0 if i == 0 else _t5_two_layer_step(x, edges[i], l, k1, k2, rho_c)
            tot += d * (u_old - u_new)
        return tot

    case = (6.0, 0.8, 400.0)
    # a poorly and a highly conducting cover, and a thin cover: image series of the step response
    k1, k2, l = case
    rho_c, t_y, d = 2.4e6, [300.0, 1.2e4, 1.0e5, 1.25e5], [0.5, 0.0, -6.0, 1.5]
    z = [0.0, 3.0, 0.5 * l, 0.999 * l, l, 1.001 * l, 2.0 * l, l + 900.0, l + 2500.0]
    got = np.asarray(layered_paleoclimate_perturbation(np.array(z), [0.0, l], [k1, k2], rho_c, t_y, d))
    _t5_exp = [_t5_two_layer(v, l, k1, k2, rho_c, t_y, d) for v in z]
    assert np.max(np.abs(got - np.array(_t5_exp))) < 6e-8      # 1e-8 K per K, largest |dT| = 6


_crown_case()

# --- test case 3 ---
# layer over half space [case = (1.2, 4.4, 35.0)]
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t5_YEAR = 365.25 * 86400.0

    def _t5_two_layer_step(x, t, l, k1, k2, rho_c):
        a1, a2 = k1 / rho_c, k2 / rho_c
        al = (math.sqrt(k1 * rho_c) - math.sqrt(k2 * rho_c)) / (math.sqrt(k1 * rho_c) + math.sqrt(k2 * rho_c))
        tot, n = 0.0, 0
        while True:
            if x <= l:
                term = (-al) ** n * (math.erfc((2 * n * l + x) / (2 * math.sqrt(a1 * t)))
                                     + al * math.erfc((2 * (n + 1) * l - x) / (2 * math.sqrt(a1 * t))))
            else:
                term = (1 + al) * (-al) ** n * math.erfc(((2 * n + 1) * l / math.sqrt(a1) + (x - l) / math.sqrt(a2))
                                                         / (2 * math.sqrt(t)))
            tot += term
            n += 1
            if abs(al) ** n < 1e-18 or n > 2000:
                return tot

    def _t5_two_layer(x, l, k1, k2, rho_c, t_years, dT):
        if x <= 0.0:
            return dT[0]
        edges = [0.0] + [t * _t5_YEAR for t in t_years]
        tot = 0.0
        for i, d in enumerate(dT):
            u_old = _t5_two_layer_step(x, edges[i + 1], l, k1, k2, rho_c)
            u_new = 0.0 if i == 0 else _t5_two_layer_step(x, edges[i], l, k1, k2, rho_c)
            tot += d * (u_old - u_new)
        return tot

    case = (1.2, 4.4, 35.0)
    # a poorly and a highly conducting cover, and a thin cover: image series of the step response
    k1, k2, l = case
    rho_c, t_y, d = 2.4e6, [300.0, 1.2e4, 1.0e5, 1.25e5], [0.5, 0.0, -6.0, 1.5]
    z = [0.0, 3.0, 0.5 * l, 0.999 * l, l, 1.001 * l, 2.0 * l, l + 900.0, l + 2500.0]
    got = np.asarray(layered_paleoclimate_perturbation(np.array(z), [0.0, l], [k1, k2], rho_c, t_y, d))
    _t5_exp = [_t5_two_layer(v, l, k1, k2, rho_c, t_y, d) for v in z]
    assert np.max(np.abs(got - np.array(_t5_exp))) < 6e-8      # 1e-8 K per K, largest |dT| = 6


_crown_case()

# --- test case 4 ---
# multilayer columns
import numpy as np


def _crown_case():
    import numpy as np

    _t5_T_ABITIBI = {1.0: -0.0008520484521674185183899386, 75.0: -0.06374574375906197936180743, 150.0: -0.1265498095117345146312706, 300.0: -0.1864289701210462478056366, 420.0: -0.230553633845341526486891, 650.0: -0.3377790202469069022517831, 900.0: -0.4209209384432630549218601, 1100.0: -0.4649965444213023133078132, 1500.0: -0.4847821038676746412810152, 2500.0: -0.3593394742270351026293501}

    _t5_T_CONTRAST = {5.0: 1.074603418068411517440318, 30.0: 0.4517810019483242366355346, 60.0: -0.2713805757456721570727339, 61.0: -0.2742609672749097256974367, 130.0: -0.4567504980928152345608845, 199.0: -0.6059285087183417791936461, 200.0: -0.6078422224580352027394069, 400.0: -1.475117606748449669874537, 900.0: -2.388730544290318546369798}

    # a five-layer Shield column with the glacial history of the task, and a strongly contrasted column
    # (k from 0.8 to 6.5) with a recent warming: values against 90-digit targets
    got = layered_paleoclimate_perturbation(np.array([float(v) for v in _t5_T_ABITIBI]), [0, 150, 420, 900, 1300], [2.3, 4.6, 3.1, 2.8, 3.5],
                 2.5e6, [1e4, 1e5, 1.2e5], [0.0, -1.0, 0.25])
    assert np.max(np.abs(np.asarray(got) - np.array(list(_t5_T_ABITIBI.values())))) < 1e-8
    got = layered_paleoclimate_perturbation(np.array([float(v) for v in _t5_T_CONTRAST]), [0, 60, 200], [0.8, 6.5, 2.0], 2.1e6,
                 [500.0, 1.2e4, 9e4], [1.2, -0.4, -5.5])
    assert np.max(np.abs(np.asarray(got) - np.array(list(_t5_T_CONTRAST.values())))) < 5.5e-8


_crown_case()

# --- test case 5 ---
# shape scalar and surface
import numpy as np


def _crown_case():
    import numpy as np
    # the output keeps the full shape of z; a scalar depth gives a float; z = 0 gives dT[0]
    tops, k, rc, t_y, d = [0, 150, 420], [2.3, 4.6, 3.1], 2.5e6, [2e3, 2.5e4, 1.1e5], [0.6, -2.5, 0.4]
    z = np.array([[0.0, 150.0, 400.0], [900.0, 1600.0, 3000.0]])
    got = layered_paleoclimate_perturbation(z, tops, k, rc, t_y, d)
    one = layered_paleoclimate_perturbation(400.0, tops, k, rc, t_y, d)
    arr1 = layered_paleoclimate_perturbation(np.array([400.0]), tops, k, rc, t_y, d)
    # output contract: the shape of z (a one-element z included); a float for a scalar z
    assert (isinstance(got, np.ndarray) and got.shape == (2, 3) and isinstance(arr1, np.ndarray) and arr1.shape == (1,)
            and type(one) is float)
    assert abs(got[0, 2] - one) < 5e-8 and abs(float(arr1[0]) - one) < 5e-8   # two computed outputs, each within 2.5e-8
    assert abs(got[0, 0] - 0.6) < 2.5e-8


_crown_case()

# --- test case 6 ---
# many depths layers and intervals
import numpy as np


def _crown_case():
    import math
    import numpy as np

    _t5_YEAR = 365.25 * 86400.0

    def _t5_halfspace(z, kappa, t_years, dT):
        edges = [0.0] + [t * _t5_YEAR for t in t_years]
        if z <= 0.0:
            return dT[0]
        tot = 0.0
        for i, d in enumerate(dT):
            e_old = math.erfc(z / (2 * math.sqrt(kappa * edges[i + 1])))
            e_new = 0.0 if i == 0 else math.erfc(z / (2 * math.sqrt(kappa * edges[i])))
            tot += d * (e_old - e_new)
        return tot

    # 1000 depths, 50 layers and 20 history intervals in one call. With one conductivity in every layer
    # the column is a half-space (erfc solution). With conductivities rising from 1.5 to 5 W/m/K, six depths
    # (z[j] for the keys j below, from 30 m to the bottom half-space) are compared with values computed
    # once in mpmath: forward transfer-matrix propagation of (F, k dF/dz) from the surface with the
    # working precision raised above the cancellation of the growing exponentials (270 to 390 digits),
    # fixed-Talbot inversion of degree 60 and 90 (identical to 30 digits) and Gaver-Stehfest inversion with
    # 60 terms (agreement 2e-16); every depth also agrees with a single-depth call
    _t5_T_50 = {10: 0.068114570909863410906, 137: 0.0079265466347630448992, 300: -0.069516821647734808719,
             500: -0.044596166873253997665, 700: -0.041732061948786446801, 999: -0.074038663511355877646}
    z = np.linspace(0.0, 3000.0, 1000)
    t_y, d = np.logspace(2.0, 5.5, 20), np.sin(np.arange(20.0))
    tops = np.linspace(0.0, 2450.0, 50)
    out = np.asarray(layered_paleoclimate_perturbation(z, tops, np.full(50, 2.9), 2.4e6, t_y, d))
    want = np.array([_t5_halfspace(float(x), 2.9 / 2.4e6, list(t_y), list(d)) for x in z])
    assert np.max(np.abs(out - want)) < 1e-8 * np.max(np.abs(d))
    k = np.linspace(1.5, 5.0, 50)
    out = np.asarray(layered_paleoclimate_perturbation(z, tops, k, 2.4e6, t_y, d))
    for j, v in _t5_T_50.items():
        assert abs(out[j] - v) < 1e-8 * np.max(np.abs(d)), (j, out[j], v)
    for j in (0, 137, 500, 999):
        one = layered_paleoclimate_perturbation(float(z[j]), tops, k, 2.4e6, t_y, d)
        assert abs(out[j] - one) < 2e-8 * np.max(np.abs(d)), (j, out[j], one)


_crown_case()

# --- test case 7 ---
# a non-positive conductivity or volumetric heat capacity raises ValueError (stated error rules)
import numpy as np


def _crown_case():
    import numpy as np
    for case in ["k0", "rc"]:
        k, rc = ([2.0, 0.0], 2.5e6) if case == "k0" else ([2.0, 3.0], 0.0)
        try:
            layered_paleoclimate_perturbation(100.0, [0.0, 200.0], k, rc, [1e3, 1e4], [0.5, -1.0])
        except ValueError:
            pass
        else:
            raise AssertionError("layered_paleoclimate_perturbation must raise ValueError (case %r)" % (case,))


_crown_case()
