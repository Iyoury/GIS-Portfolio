# --- test case 0 ---
import numpy as np


def _crown_case():
    import numpy as np

    # targets: the estimators of the prompt evaluated in extended precision (mpmath) from independent inputs:
    # f_W, xi_spin and xi_energy from Kaufman's exact strip spectrum; m_W, chi_W and H_edge(W) from the
    # zero-momentum block of the transfer matrix (50-digit diagonalization, 90-digit second difference in h,
    # 40-digit root of the leading-pair discriminant)
    out = critical_exponents(4)
    assert isinstance(out, dict) and set(out) == {"x_sigma", "x_energy", "beta_over_nu", "gamma_over_nu", "y_h", "c"} \
        and all(isinstance(v, float) for v in out.values())
    assert abs(out["x_sigma"] - 0.12892232785193442) <= 1e-4, out["x_sigma"]
    assert abs(out["x_energy"] - 0.952153234423053) <= 1e-4, out["x_energy"]
    assert abs(out["beta_over_nu"] - 0.11928897309653642) <= 1e-4, out["beta_over_nu"]
    assert abs(out["gamma_over_nu"] - 1.8112793749049443) <= 1e-4, out["gamma_over_nu"]
    assert abs(out["y_h"] - 1.931824586088568) <= 1e-4, out["y_h"]
    assert abs(out["c"] - 0.4950743416963288) <= 1e-4, out["c"]


_crown_case()

# --- test case 1 ---
import numpy as np


def _crown_case():
    import numpy as np

    # targets: the estimators of the prompt evaluated in extended precision (mpmath) from independent inputs:
    # f_W, xi_spin and xi_energy from Kaufman's exact strip spectrum; m_W, chi_W and H_edge(W) from the
    # zero-momentum block of the transfer matrix (50-digit diagonalization, 90-digit second difference in h,
    # 40-digit root of the leading-pair discriminant)
    out = critical_exponents(6)
    assert abs(out["x_sigma"] - 0.12656074717880197) <= 1e-4, out["x_sigma"]
    assert abs(out["x_energy"] - 0.9778985038020386) <= 1e-4, out["x_energy"]
    assert abs(out["beta_over_nu"] - 0.1224432767504538) <= 1e-4, out["beta_over_nu"]
    assert abs(out["gamma_over_nu"] - 1.7759995588747775) <= 1e-4, out["gamma_over_nu"]
    assert abs(out["y_h"] - 1.8986874467775703) <= 1e-4, out["y_h"]
    assert abs(out["c"] - 0.4982397874572249) <= 1e-4, out["c"]


_crown_case()

# --- test case 2 ---
import numpy as np


def _crown_case():
    import numpy as np

    # targets: the estimators of the prompt evaluated in extended precision (mpmath) from independent inputs:
    # f_W, xi_spin and xi_energy from Kaufman's exact strip spectrum; m_W, chi_W and H_edge(W) from the
    # zero-momentum block of the transfer matrix (50-digit diagonalization, 90-digit second difference in h,
    # 40-digit root of the leading-pair discriminant)
    # lower width endpoint L = 3
    out = critical_exponents(3)
    assert abs(out["x_sigma"] - 0.13225680285912744) <= 1e-4, out["x_sigma"]
    assert abs(out["x_energy"] - 0.9190468875900993) <= 1e-4, out["x_energy"]
    assert abs(out["beta_over_nu"] - 0.11479064707719534) <= 1e-4, out["beta_over_nu"]
    assert abs(out["gamma_over_nu"] - 1.849612619911595) <= 1e-4, out["gamma_over_nu"]
    assert abs(out["y_h"] - 1.967369281005889) <= 1e-4, out["y_h"]
    assert abs(out["c"] - 0.4987546714036298) <= 1e-4, out["c"]


_crown_case()

# --- test case 3 ---
import numpy as np


def _crown_case():
    import numpy as np

    # targets: the estimators of the prompt evaluated in extended precision (mpmath) from independent inputs:
    # f_W, xi_spin and xi_energy from Kaufman's exact strip spectrum; m_W, chi_W and H_edge(W) from the
    # zero-momentum block of the transfer matrix (50-digit diagonalization, 90-digit second difference in h,
    # 40-digit root of the leading-pair discriminant)
    # upper width endpoint L = 7
    out = critical_exponents(7)
    assert abs(out["x_sigma"] - 0.12611511593158242) <= 1e-4, out["x_sigma"]
    assert abs(out["x_energy"] - 0.983622221178631) <= 1e-4, out["x_energy"]
    assert abs(out["beta_over_nu"] - 0.12310599848616365) <= 1e-4, out["beta_over_nu"]
    assert abs(out["gamma_over_nu"] - 1.7687632731179646) <= 1e-4, out["gamma_over_nu"]
    assert abs(out["y_h"] - 1.8919597616887955) <= 1e-4, out["y_h"]
    assert abs(out["c"] - 0.49906830035341704) <= 1e-4, out["c"]


_crown_case()

# --- test case 4 ---
import numpy as np


def _crown_case():
    import numpy as np

    # targets: the estimators of the prompt evaluated in extended precision (mpmath) from independent inputs:
    # f_W, xi_spin and xi_energy from Kaufman's exact strip spectrum; m_W, chi_W and H_edge(W) from the
    # zero-momentum block of the transfer matrix (50-digit diagonalization, 90-digit second difference in h,
    # 40-digit root of the leading-pair discriminant)
    # middle of the range; the m_W, chi_W and H_edge(W) inputs at W = 5 and 6 were reproduced to 25 digits by a
    # second extended-precision route on the full 2**W-state matrix (no orbit reduction)
    out = critical_exponents(5)
    assert abs(out["x_sigma"] - 0.12735340094255937) <= 1e-4, out["x_sigma"]
    assert abs(out["x_energy"] - 0.968613687511057) <= 1e-4, out["x_energy"]
    assert abs(out["beta_over_nu"] - 0.12134219322496173) <= 1e-4, out["beta_over_nu"]
    assert abs(out["gamma_over_nu"] - 1.7885391229520624) <= 1e-4, out["gamma_over_nu"]
    assert abs(out["y_h"] - 1.9104404106801347) <= 1e-4, out["y_h"]
    assert abs(out["c"] - 0.49673455609477923) <= 1e-4, out["c"]


_crown_case()
