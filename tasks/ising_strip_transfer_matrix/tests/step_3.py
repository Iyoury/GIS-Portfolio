# --- test case 0 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
    # the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
    # far below Tc the top pair is degenerate to 9e-18 relative; the value equals Yang's bulk
    # (1 - sinh(2/T)**-4)**(1/8) to better than 1e-24 at this temperature
    m_L = strip_magnetization(10, 0.5)
    assert isinstance(m_L, float)
    assert abs(m_L - 0.9999997746272086) < 1e-9, m_L


_crown_case()

# --- test case 1 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
    # the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
    # top pair degenerate to 7e-15 relative; equals Yang's bulk value to better than 1e-24
    m_L = strip_magnetization(10, 0.6)
    assert abs(m_L - 0.9999967442274907) < 1e-9, m_L


_crown_case()

# --- test case 2 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
    # the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
    # top pair degenerate to 3e-14 relative; equals Yang's bulk value to better than 1e-24
    m_L = strip_magnetization(8, 0.5)
    assert abs(m_L - 0.9999997746272086) < 1e-9, m_L


_crown_case()

# --- test case 3 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
    # the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
    # just below Tc: the finite width matters (Yang's bulk value is 0.79)
    m_L = strip_magnetization(6, 2.2)
    assert abs(m_L - 0.8643773472737563) < 1e-9, m_L


_crown_case()

# --- test case 4 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
    # the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
    # above Tc, where m_L is still defined by the slowest term of G(r)
    m_L = strip_magnetization(7, 4.0)
    assert abs(m_L - 0.5091279359347337) < 1e-9, m_L


_crown_case()

# --- test case 5 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath) of the zero-momentum block of the transfer matrix, built from
    # the energy, with m_L = |<even|S|odd>| / L between its top flip-even and top flip-odd states
    # both endpoints of the stated domain: L = 3 and T = 10
    m_L = strip_magnetization(3, 10.0)
    assert abs(m_L - 0.6388000647922389) < 1e-9, m_L


_crown_case()

# --- test case 6 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: Yang's closed form for the spontaneous magnetization of the infinite lattice,
    # (1 - sinh(2/T)**-4)**(1/8), evaluated here. Below Tc m_L approaches it exponentially fast in L: the
    # 50-digit zero-momentum diagonalization gives m_L - Yang = 5.2e-19 at (L, T) = (10, 0.9) and 8.0e-21 at
    # (8, 0.7), far below the tolerance, while the top pair is degenerate to 1.1e-9 and 2.8e-10 relative
    m_L = strip_magnetization(10, 0.9)
    m_yang = (1.0 - np.sinh(2.0 / 0.9) ** -4) ** 0.125
    assert abs(m_L - m_yang) < 1e-9, (m_L, m_yang)
    m_L = strip_magnetization(8, 0.7)
    m_yang = (1.0 - np.sinh(2.0 / 0.7) ** -4) ** 0.125
    assert abs(m_L - m_yang) < 1e-9, (m_L, m_yang)


_crown_case()

# --- test case 7 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath eigsy) of the full 2**L-state transfer matrix summed from the
    # energy (no orbit reduction), m_L = |<top|s_0|top flip-odd>|; the zero-momentum route agrees to 1e-22
    # corner of the domain: narrowest strip at the lowest temperature (relative gap of the top pair 1.2e-5)
    m_L = strip_magnetization(3, 0.5)
    assert abs(m_L - 0.9999997746274376) < 1e-9, m_L


_crown_case()

# --- test case 8 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: 50-digit diagonalization (mpmath eigsy) of the full 2**L-state transfer matrix summed from the
    # energy (no orbit reduction), m_L = |<top|s_0|top flip-odd>|; the zero-momentum route agrees to 1e-22
    # below Tc on a narrow strip, where m_L differs from Yang's value by 6.7e-4
    m_L = strip_magnetization(4, 1.6)
    assert abs(m_L - 0.980292105519335) < 1e-9, m_L
    # above Tc, where m_L is not a magnetization of the infinite lattice but the amplitude still exists
    m_L = strip_magnetization(5, 2.8)
    assert abs(m_L - 0.7342911230665126) < 1e-9, m_L


_crown_case()
