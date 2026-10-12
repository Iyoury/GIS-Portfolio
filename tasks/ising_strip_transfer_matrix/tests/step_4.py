# --- test case 0 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = -d2f/dh2 from a centred second difference (step 1e-25, Richardson-checked) of
    # f(h) = -T ln(lambda_0(h)) / L, with lambda_0(h) the top eigenvalue of the zero-momentum block of the
    # transfer matrix in the field h, diagonalized with 90-digit arithmetic (mpmath)
    # below Tc: the curvature of f(h) is confined to |h| of the order of the tiny spin gap
    chi = strip_susceptibility(10, 1.5)
    assert isinstance(chi, float)
    assert abs(chi / 142550.12792161922 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 1 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = -d2f/dh2 from a centred second difference (step 1e-25, Richardson-checked) of
    # f(h) = -T ln(lambda_0(h)) / L, with lambda_0(h) the top eigenvalue of the zero-momentum block of the
    # transfer matrix in the field h, diagonalized with 90-digit arithmetic (mpmath)
    # exact critical temperature
    chi = strip_susceptibility(8, 2.0 / np.log(1.0 + np.sqrt(2.0)))
    assert abs(chi / 47.240627685916294 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 2 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = -d2f/dh2 from a centred second difference (step 1e-25, Richardson-checked) of
    # f(h) = -T ln(lambda_0(h)) / L, with lambda_0(h) the top eigenvalue of the zero-momentum block of the
    # transfer matrix in the field h, diagonalized with 90-digit arithmetic (mpmath)
    # above Tc
    chi = strip_susceptibility(6, 3.0)
    assert abs(chi / 3.5544669825746844 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 3 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = -d2f/dh2 from a centred second difference (step 1e-25, Richardson-checked) of
    # f(h) = -T ln(lambda_0(h)) / L, with lambda_0(h) the top eigenvalue of the zero-momentum block of the
    # transfer matrix in the field h, diagonalized with 90-digit arithmetic (mpmath)
    # widest strip at the lowest temperature: chi of order 1e9
    chi = strip_susceptibility(10, 1.0)
    assert abs(chi / 1390953077.0438237 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 4 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = -d2f/dh2 from a centred second difference (step 1e-25, Richardson-checked) of
    # f(h) = -T ln(lambda_0(h)) / L, with lambda_0(h) the top eigenvalue of the zero-momentum block of the
    # transfer matrix in the field h, diagonalized with 90-digit arithmetic (mpmath)
    # high temperature
    chi = strip_susceptibility(4, 10.0)
    assert abs(chi / 0.15666218185137215 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 5 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = -d2f/dh2 from a centred second difference (step 1e-25, Richardson-checked) of
    # f(h) = -T ln(lambda_0(h)) / L, with lambda_0(h) the top eigenvalue of the zero-momentum block of the
    # transfer matrix in the field h, diagonalized with 90-digit arithmetic (mpmath)
    # lower width endpoint L = 3, at two temperatures
    chi = strip_susceptibility(3, 2.5)
    assert abs(chi / 5.143306865572443 - 1.0) < 1e-6, chi
    chi = strip_susceptibility(3, 10.0)
    assert abs(chi / 0.15627539609908636 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 6 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = (T / L) d2 ln(lambda_0) / dh2 from a centred second difference (step 1e-25; the partner with
    # step 2e-25 agrees to 1e-41) of the top eigenvalue of the full 2**L-state transfer matrix summed from the
    # energy, diagonalized with 90-digit arithmetic (mpmath eigsy, no orbit reduction); the zero-momentum
    # route agrees to 1e-25
    # lowest temperature of the range on a narrow strip: chi of order 1e4
    chi = strip_susceptibility(4, 1.0)
    assert abs(chi / 9671.307908926903 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 7 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = (T / L) d2 ln(lambda_0) / dh2 from a centred second difference (step 1e-25; the partner with
    # step 2e-25 agrees to 1e-41) of the top eigenvalue of the full 2**L-state transfer matrix summed from the
    # energy, diagonalized with 90-digit arithmetic (mpmath eigsy, no orbit reduction); the zero-momentum
    # route agrees to 1e-25
    # just below Tc
    chi = strip_susceptibility(5, 2.0)
    assert abs(chi / 58.475656692287025 - 1.0) < 1e-6, chi


_crown_case()

# --- test case 8 ---
import numpy as np


def _crown_case():
    import numpy as np

    # target: chi = (T / L) d2 ln(lambda_0) / dh2 from a centred second difference (step 1e-25; the partner with
    # step 2e-25 agrees to 1e-41) of the top eigenvalue of the full 2**L-state transfer matrix summed from the
    # energy, diagonalized with 90-digit arithmetic (mpmath eigsy, no orbit reduction); the zero-momentum
    # route agrees to 1e-25
    # below Tc on a wider strip: the relative spin gap is 1e-3 and chi is of order 3e4
    chi = strip_susceptibility(6, 1.25)
    assert abs(chi / 26516.25729406536 - 1.0) < 1e-6, chi


_crown_case()
