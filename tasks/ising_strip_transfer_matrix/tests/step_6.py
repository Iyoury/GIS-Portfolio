# --- test case 0 ---
import numpy as np


def _crown_case():
    import numpy as np

    # targets: the estimators of the prompt evaluated in extended precision (mpmath) from independent inputs:
    # f_W, xi_spin and xi_energy from Kaufman's exact strip spectrum; m_W, chi_W and H_edge(W) from the
    # zero-momentum block of the transfer matrix (50-digit diagonalization, 90-digit second difference in h,
    # 40-digit root of the leading-pair discriminant)
    out = critical_exponents(4)
    expected = {
        "x_sigma": 0.12892232785193442,
        "x_energy": 0.952153234423053,
        "beta_over_nu": 0.11928897309653642,
        "gamma_over_nu": 1.8112793749049443,
        "y_h": 1.931824586088568,
        "c": 0.4950743416963288,
    }
    assert isinstance(out, dict) and set(out) == set(expected)
    assert all(isinstance(v, float) for v in out.values())
    for key, value in expected.items():
        assert abs(out[key] - value) <= 1e-4, (key, out[key], value)


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
    expected = {
        "x_sigma": 0.12656074717880197,
        "x_energy": 0.9778985038020386,
        "beta_over_nu": 0.1224432767504538,
        "gamma_over_nu": 1.7759995588747775,
        "y_h": 1.8986874467775703,
        "c": 0.4982397874572249,
    }
    assert isinstance(out, dict) and set(out) == set(expected)
    assert all(isinstance(v, float) for v in out.values())
    for key, value in expected.items():
        assert abs(out[key] - value) <= 1e-4, (key, out[key], value)


_crown_case()

# --- test case 2 ---
import numpy as np


def _crown_case():
    import numpy as np

    # targets: the estimators of the prompt evaluated in extended precision (mpmath) from independent inputs:
    # f_W, xi_spin and xi_energy from Kaufman's exact strip spectrum; m_W, chi_W and H_edge(W) from the
    # zero-momentum block of the transfer matrix (50-digit diagonalization, 90-digit second difference in h,
    # 40-digit root of the leading-pair discriminant)
    # width endpoints L = 3 and L = 7
    for L, expected in ((3, {"x_sigma": 0.13225680285912744, "x_energy": 0.9190468875900993, "beta_over_nu": 0.11479064707719534, "gamma_over_nu": 1.849612619911595, "y_h": 1.967369281005889, "c": 0.4987546714036298}),
                        (7, {"x_sigma": 0.12611511593158242, "x_energy": 0.983622221178631, "beta_over_nu": 0.12310599848616365, "gamma_over_nu": 1.7687632731179646, "y_h": 1.8919597616887955, "c": 0.49906830035341704})):
        out = critical_exponents(L)
        assert isinstance(out, dict) and set(out) == set(expected)
        assert all(isinstance(v, float) for v in out.values())
        for key, value in expected.items():
            assert abs(out[key] - value) <= 1e-4, (L, key, out[key], value)


_crown_case()
