# --- test case 0 ---
import numpy as np

chi = strip_susceptibility(10, 1.5)
assert isinstance(chi, float)
assert abs(chi / 142550.12792116308 - 1.0) < 1e-6, chi

# --- test case 1 ---
import numpy as np

chi = strip_susceptibility(8, 2.0 / np.log(1.0 + np.sqrt(2.0)))
assert isinstance(chi, float)
assert abs(chi / 47.24062768591631 - 1.0) < 1e-6, chi

# --- test case 2 ---
import numpy as np

chi = strip_susceptibility(6, 3.0)
assert isinstance(chi, float)
assert abs(chi / 3.554466982574681 - 1.0) < 1e-6, chi

# --- test case 3 ---
import numpy as np

chi = strip_susceptibility(10, 1.0)
assert isinstance(chi, float)
assert abs(chi / 1390953074.0910573 - 1.0) < 1e-6, chi

# --- test case 4 ---
import numpy as np

chi = strip_susceptibility(4, 10.0)
assert isinstance(chi, float)
assert abs(chi / 0.1566621818513722 - 1.0) < 1e-6, chi

# --- test case 5 ---
import numpy as np

# lower width endpoint L = 3 (two temperatures; values cross-checked by the two solutions, 1e-15)
chi = strip_susceptibility(3, 2.5)
assert isinstance(chi, float)
assert abs(chi / 5.143306865572443 - 1.0) < 1e-6, chi
chi = strip_susceptibility(3, 10.0)
assert isinstance(chi, float)
assert abs(chi / 0.15627539609908636 - 1.0) < 1e-6, chi
