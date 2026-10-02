# --- test case 0 ---
import numpy as np

m_L = strip_magnetization(10, 0.5)
assert isinstance(m_L, float)
assert abs(m_L - 0.9999997746272085) < 1e-9, m_L

# --- test case 1 ---
import numpy as np

m_L = strip_magnetization(10, 0.6)
assert isinstance(m_L, float)
assert abs(m_L - 0.9999967442274905) < 1e-9, m_L

# --- test case 2 ---
import numpy as np

m_L = strip_magnetization(8, 0.5)
assert isinstance(m_L, float)
assert abs(m_L - 0.9999997746272085) < 1e-9, m_L

# --- test case 3 ---
import numpy as np

m_L = strip_magnetization(6, 2.2)
assert isinstance(m_L, float)
assert abs(m_L - 0.8643773472737563) < 1e-9, m_L

# --- test case 4 ---
import numpy as np

m_L = strip_magnetization(7, 4.0)
assert isinstance(m_L, float)
assert abs(m_L - 0.5091279359347339) < 1e-9, m_L

# --- test case 5 ---
import numpy as np

# both endpoints of the stated domain: L = 3 and T = 10 (value cross-checked by the two solutions, 1e-15)
m_L = strip_magnetization(3, 10.0)
assert isinstance(m_L, float)
assert abs(m_L - 0.6388000647922388) < 1e-9, m_L
