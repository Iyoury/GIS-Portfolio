# --- test case 0 ---
import numpy as np

# target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
# the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
# far below Tc the top pair is degenerate to 9e-18 relative; the value equals Yang's bulk
# (1 - sinh(2/T)**-4)**(1/8) to better than 1e-24 at this temperature
m_L = strip_magnetization(10, 0.5)
assert isinstance(m_L, float)
assert abs(m_L - 0.9999997746272086) < 1e-9, m_L

# --- test case 1 ---
import numpy as np

# target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
# the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
# top pair degenerate to 7e-15 relative; equals Yang's bulk value to better than 1e-24
m_L = strip_magnetization(10, 0.6)
assert isinstance(m_L, float)
assert abs(m_L - 0.9999967442274907) < 1e-9, m_L

# --- test case 2 ---
import numpy as np

# target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
# the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
# top pair degenerate to 3e-14 relative; equals Yang's bulk value to better than 1e-24
m_L = strip_magnetization(8, 0.5)
assert isinstance(m_L, float)
assert abs(m_L - 0.9999997746272086) < 1e-9, m_L

# --- test case 3 ---
import numpy as np

# target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
# the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
# just below Tc: the finite width matters (Yang's bulk value is 0.79)
m_L = strip_magnetization(6, 2.2)
assert isinstance(m_L, float)
assert abs(m_L - 0.8643773472737563) < 1e-9, m_L

# --- test case 4 ---
import numpy as np

# target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
# the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
# above Tc, where m_L is still defined by the slowest term of G(r)
m_L = strip_magnetization(7, 4.0)
assert isinstance(m_L, float)
assert abs(m_L - 0.5091279359347337) < 1e-9, m_L

# --- test case 5 ---
import numpy as np

# target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
# the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
# both endpoints of the stated domain: L = 3 and T = 10
m_L = strip_magnetization(3, 10.0)
assert isinstance(m_L, float)
assert abs(m_L - 0.6388000647922389) < 1e-9, m_L
