import time as _t_time


def _t_timed(T, h):
    # the prompt requires every call to finish within 60 s on one CPU core
    start = _t_time.perf_counter()
    out = bulk_thermodynamics(T, h)
    elapsed = _t_time.perf_counter() - start
    assert elapsed <= 60.0, ("bulk_thermodynamics took %.1f s" % elapsed, T, h)
    return out


# --- test case 0: near Tc, weak field ---
import numpy as np

out = _t_timed(2.3, 0.008)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.124125511541119)) < 1e-10, f
assert abs(m - 0.6697168488050375) < 1e-9, m
assert abs(chi / 10.04315519161016 - 1.0) < 1e-5, chi
assert abs(c / 1.7258720135952346 - 1.0) < 1e-5, c

# --- test case 1: above Tc, moderate field ---
import numpy as np

out = _t_timed(2.5, 0.05)
assert isinstance(out, tuple) and len(out) == 4 and all(type(x) is float for x in out), out
f, m, chi, c = out
assert abs(f - (-2.218362198255085)) < 1e-10, f
assert abs(m - 0.6053470269921418) < 1e-9, m
assert abs(chi / 3.997818624283852 - 1.0) < 1e-5, chi
assert abs(c / 1.372746234277559 - 1.0) < 1e-5, c
