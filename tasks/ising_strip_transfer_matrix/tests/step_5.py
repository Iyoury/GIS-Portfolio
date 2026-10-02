# --- test case 0 ---
import numpy as np

H_edge = lee_yang_edge(6, 2.0 / np.log(1.0 + np.sqrt(2.0)))
assert isinstance(H_edge, float)
assert abs(H_edge / 0.029976092836555227 - 1.0) < 1e-8, H_edge

# --- test case 1 ---
import numpy as np

H_edge = lee_yang_edge(8, 3.0)
assert isinstance(H_edge, float)
assert abs(H_edge / 0.17998354627962412 - 1.0) < 1e-8, H_edge

# --- test case 2 ---
import numpy as np

H_edge = lee_yang_edge(7, 1.5)
assert isinstance(H_edge, float)
assert abs(H_edge / 0.00013216250013612187 - 1.0) < 1e-8, H_edge

# --- test case 3 ---
import numpy as np

H_edge = lee_yang_edge(4, 10.0)
assert isinstance(H_edge, float)
assert abs(H_edge / 5.939074347274278 - 1.0) < 1e-8, H_edge

# --- test case 4 ---
import numpy as np

H_edge = lee_yang_edge(5, 2.0)
assert isinstance(H_edge, float)
assert abs(H_edge / 0.015847934855761598 - 1.0) < 1e-8, H_edge

# --- test case 5 ---
import numpy as np

# lower width endpoint L = 3, at both ends of the temperature range (cross-checked by the two solutions, 1e-13)
H_edge = lee_yang_edge(3, 1.5)
assert isinstance(H_edge, float)
assert abs(H_edge / 0.013280762164504314 - 1.0) < 1e-8, H_edge
H_edge = lee_yang_edge(3, 10.0)
assert isinstance(H_edge, float)
assert abs(H_edge / 6.125990933020653 - 1.0) < 1e-8, H_edge
