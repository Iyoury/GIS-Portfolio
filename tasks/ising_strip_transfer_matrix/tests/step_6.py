# --- test case 0 ---
import numpy as np

out = critical_exponents(4)
expected = {
    "x_sigma": 0.12892232785193444,
    "x_energy": 0.952153234423053,
    "beta_over_nu": 0.1192889730965362,
    "gamma_over_nu": 1.8112793749049496,
    "y_h": 1.9318245860885825,
    "c": 0.4950743416965247,
}
assert isinstance(out, dict) and set(out) == set(expected)
assert all(isinstance(v, float) for v in out.values())
for key, value in expected.items():
    assert abs(out[key] - value) <= 1e-4, (key, out[key], value)

# --- test case 1 ---
import numpy as np

out = critical_exponents(6)
expected = {
    "x_sigma": 0.12656074717880184,
    "x_energy": 0.9778985038020392,
    "beta_over_nu": 0.12244327675045068,
    "gamma_over_nu": 1.7759995588747937,
    "y_h": 1.8986874467776071,
    "c": 0.49823978745756425,
}
assert isinstance(out, dict) and set(out) == set(expected)
assert all(isinstance(v, float) for v in out.values())
for key, value in expected.items():
    assert abs(out[key] - value) <= 1e-4, (key, out[key], value)

# --- test case 2 ---
import numpy as np

# width endpoints L = 3 and L = 7 (values cross-checked by the two solutions, 1e-12)
for L, expected in ((3, {"x_sigma": 0.1322568028591277, "x_energy": 0.9190468875900993, "beta_over_nu": 0.11479064707719389,
                         "gamma_over_nu": 1.8496126199116005, "y_h": 1.9673692810059342, "c": 0.4987546714035908}),
                    (7, {"x_sigma": 0.12611511593158228, "x_energy": 0.9836222211786327, "beta_over_nu": 0.12310599848615947,
                         "gamma_over_nu": 1.768763273117929, "y_h": 1.8919597616888932, "c": 0.49906830035366995})):
    out = critical_exponents(L)
    assert isinstance(out, dict) and set(out) == set(expected)
    assert all(isinstance(v, float) for v in out.values())
    for key, value in expected.items():
        assert abs(out[key] - value) <= 1e-4, (L, key, out[key], value)
