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
    assert isinstance(chi, float)
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
    assert isinstance(chi, float)
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
    assert isinstance(chi, float)
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
    assert isinstance(chi, float)
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
    assert isinstance(chi, float)
    assert abs(chi / 5.143306865572443 - 1.0) < 1e-6, chi
    chi = strip_susceptibility(3, 10.0)
    assert isinstance(chi, float)
    assert abs(chi / 0.15627539609908636 - 1.0) < 1e-6, chi


_crown_case()
