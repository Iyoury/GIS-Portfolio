# --- test case 0 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath), of Re((a - b)**2) for the two eigenvalues a, b of
    # largest modulus of the zero-momentum block of the transfer matrix in the field 1j * H (both leading
    # states are translation invariant), bracketed on a logarithmic grid and refined to 1e-32 relative
    # exact critical temperature: a lower pair of eigenvalues leaves the real axis before the leading pair meets
    H_edge = lee_yang_edge(6, 2.0 / np.log(1.0 + np.sqrt(2.0)))
    assert isinstance(H_edge, float)
    assert abs(H_edge / 0.02997609283655467 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 1 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath), of Re((a - b)**2) for the two eigenvalues a, b of
    # largest modulus of the zero-momentum block of the transfer matrix in the field 1j * H (both leading
    # states are translation invariant), bracketed on a logarithmic grid and refined to 1e-32 relative
    # above Tc, widest strip of the range
    H_edge = lee_yang_edge(8, 3.0)
    assert abs(H_edge / 0.17998354627962332 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 2 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath), of Re((a - b)**2) for the two eigenvalues a, b of
    # largest modulus of the zero-momentum block of the transfer matrix in the field 1j * H (both leading
    # states are translation invariant), bracketed on a logarithmic grid and refined to 1e-32 relative
    # below Tc the edge closes exponentially with the width
    H_edge = lee_yang_edge(7, 1.5)
    assert abs(H_edge / 0.00013216250013609133 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 3 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath), of Re((a - b)**2) for the two eigenvalues a, b of
    # largest modulus of the zero-momentum block of the transfer matrix in the field 1j * H (both leading
    # states are translation invariant), bracketed on a logarithmic grid and refined to 1e-32 relative
    # high temperature
    H_edge = lee_yang_edge(4, 10.0)
    assert abs(H_edge / 5.939074347274278 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 4 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath), of Re((a - b)**2) for the two eigenvalues a, b of
    # largest modulus of the zero-momentum block of the transfer matrix in the field 1j * H (both leading
    # states are translation invariant), bracketed on a logarithmic grid and refined to 1e-32 relative
    # below Tc
    H_edge = lee_yang_edge(5, 2.0)
    assert abs(H_edge / 0.015847934855761532 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 5 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath), of Re((a - b)**2) for the two eigenvalues a, b of
    # largest modulus of the zero-momentum block of the transfer matrix in the field 1j * H (both leading
    # states are translation invariant), bracketed on a logarithmic grid and refined to 1e-32 relative
    # ends of the domain: L = 3 at both ends of the temperature range, and the smallest edge of the domain,
    # L = 8 at T = 1.5
    H_edge = lee_yang_edge(3, 1.5)
    assert abs(H_edge / 0.013280762164504243 - 1.0) < 1e-8, H_edge
    H_edge = lee_yang_edge(3, 10.0)
    assert abs(H_edge / 6.12599093302067 - 1.0) < 1e-8, H_edge
    H_edge = lee_yang_edge(8, 1.5)
    assert abs(H_edge / 4.821088064950643e-05 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 6 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath, Illinois iteration), of Re((a - b)**2) for the two
    # eigenvalues a, b of largest modulus of the full 2**L-state complex symmetric matrix in the field 1j * H
    # (mpmath eig, no orbit reduction), bracketed by a double-precision scan on a logarithmic grid from 1e-7;
    # the zero-momentum route agrees to 5e-30 relative
    # above Tc on the narrowest strip
    H_edge = lee_yang_edge(3, 4.0)
    assert abs(H_edge / 0.8832949852754638 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 7 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath, Illinois iteration), of Re((a - b)**2) for the two
    # eigenvalues a, b of largest modulus of the full 2**L-state complex symmetric matrix in the field 1j * H
    # (mpmath eig, no orbit reduction), bracketed by a double-precision scan on a logarithmic grid from 1e-7;
    # the zero-momentum route agrees to 5e-30 relative
    # above Tc, near the critical region
    H_edge = lee_yang_edge(4, 2.5)
    assert abs(H_edge / 0.11262495622664848 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 8 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath, Illinois iteration), of Re((a - b)**2) for the two
    # eigenvalues a, b of largest modulus of the full 2**L-state complex symmetric matrix in the field 1j * H
    # (mpmath eig, no orbit reduction), bracketed by a double-precision scan on a logarithmic grid from 1e-7;
    # the zero-momentum route agrees to 5e-30 relative
    # below Tc
    H_edge = lee_yang_edge(5, 1.8)
    assert abs(H_edge / 0.006365236037625519 - 1.0) < 1e-8, H_edge


_crown_case()

# --- test case 9 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: root, found with 40-digit arithmetic (mpmath, Illinois iteration), of Re((a - b)**2) for the two
    # eigenvalues a, b of largest modulus of the full 2**L-state complex symmetric matrix in the field 1j * H
    # (mpmath eig, no orbit reduction), bracketed by a double-precision scan on a logarithmic grid from 1e-7;
    # the zero-momentum route agrees to 5e-30 relative
    # lowest temperature of the range: the edge is 3.8e-4
    H_edge = lee_yang_edge(6, 1.5)
    assert abs(H_edge / 0.00037587734733559507 - 1.0) < 1e-8, H_edge


_crown_case()
