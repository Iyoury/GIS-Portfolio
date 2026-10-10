# --- test case 0 ---
# near Tc, weak field
import numpy as np

out = bulk_thermodynamics(2.3, 0.008)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.124125511541119)) < 1e-10, f
assert abs(m - 0.6697168488050375) < 1e-9, m
assert abs(chi / 10.04315519161016 - 1.0) < 1e-5, chi
assert abs(c / 1.7258720135952346 - 1.0) < 1e-5, c

# --- test case 1 ---
# above Tc, moderate field
import numpy as np

out = bulk_thermodynamics(2.5, 0.05)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.218362198255085)) < 1e-10, f
assert abs(m - 0.6053470269921418) < 1e-9, m
assert abs(chi / 3.997818624283852 - 1.0) < 1e-5, chi
assert abs(c / 1.372746234277559 - 1.0) < 1e-5, c
