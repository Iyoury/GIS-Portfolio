# --- test case 0: exact Tc, weakest field: largest correlation length ---
import numpy as np

out = bulk_thermodynamics(2.0 / np.log(1.0 + np.sqrt(2.0)), 0.005)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.1129520308985383)) < 1e-10, f
assert abs(m - 0.7040845281630687) < 1e-9, m
assert abs(chi / 9.346238387834271 - 1.0) < 1e-5, chi
assert abs(c / 1.6888219521953 - 1.0) < 1e-5, c

# --- test case 1: above Tc, weak field: susceptibility peak region ---
import numpy as np

out = bulk_thermodynamics(2.4, 0.01)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.159232093634584)) < 1e-10, f
assert abs(m - 0.4420230056580343) < 1e-9, m
assert abs(chi / 22.504846099719447 - 1.0) < 1e-5, chi
assert abs(c / 1.6439292454428767 - 1.0) < 1e-5, c

# --- test case 2: below Tc, weakest field ---
import numpy as np

out = bulk_thermodynamics(2.1, 0.005)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.0731964092038884)) < 1e-10, f
assert abs(m - 0.8731105307995517) < 1e-9, m
assert abs(chi / 0.8041762325829612 - 1.0) < 1e-5, chi
assert abs(c / 0.9303023081108641 - 1.0) < 1e-5, c

# --- test case 3: above Tc, weakest field, small magnetization ---
import numpy as np

out = bulk_thermodynamics(2.6, 0.005)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.2438619104244353)) < 1e-10, f
assert abs(m - 0.06848983737974215) < 1e-9, m
assert abs(chi / 13.537370908985856 - 1.0) < 1e-5, chi
assert abs(c / 0.7226708601463608 - 1.0) < 1e-5, c

# --- test case 4: upper ends of the T and h ranges ---
import numpy as np

out = bulk_thermodynamics(3.0, 0.2)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.510941028682692)) < 1e-10, f
assert abs(m - 0.5422176797141599) < 1e-9, m
assert abs(chi / 1.4564677820698886 - 1.0) < 1e-5, chi
assert abs(c / 0.9086788254064615 - 1.0) < 1e-5, c

# --- test case 5: lower end of T, strongest field ---
import numpy as np

out = bulk_thermodynamics(2.0, 0.2)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.238272302396574)) < 1e-10, f
assert abs(m - 0.9479316136292613) < 1e-9, m
assert abs(chi / 0.1036596195774786 - 1.0) < 1e-5, chi
assert abs(c / 0.4714598825413437 - 1.0) < 1e-5, c
