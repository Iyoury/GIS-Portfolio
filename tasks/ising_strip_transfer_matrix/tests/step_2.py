# --- test case 0 ---
import numpy as np


def _crown_case():
    import numpy as np

    # Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
    # With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
    #   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
    # ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
    # even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
    # odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1. The spectrum is evaluated with
    # 60-digit decimal arithmetic (standard library): below Tc ln lambda_0 - ln lambda_1 is an exponentially
    # small difference of two sums of order L (1.4e-8 at L = 10, T = 1), which a direct double-precision
    # evaluation of these sums would give only to about 1e-7 relative.
    def _t_kaufman(L, T):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 60
            one = Decimal(1)
            pi = Decimal(0)                          # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
            for mult, x in ((16, Decimal(5)), (-4, Decimal(239))):
                term, k, s = one / x, 0, Decimal(0)
                while abs(term) > Decimal(10) ** -65:
                    s += term / (2 * k + 1)
                    term = -term / (x * x)
                    k += 1
                pi += mult * s

            def cos(x):                              # Taylor series after reduction to 0 <= x <= pi
                x = x if x <= pi else 2 * pi - x
                term, s, k = one, one, 0
                while abs(term) > Decimal(10) ** -65:
                    k += 2
                    term = -term * x * x / (k * (k - 1))
                    s += term
                return s

            def cosh(x):
                return (x.exp() + (-x).exp()) / 2

            def sinh(x):
                return (x.exp() - (-x).exp()) / 2

            K = one / Decimal(T)                     # Decimal(T) is the exact value of the float T
            Kd = -((1 - (-2 * K).exp()) / (1 + (-2 * K).exp())).ln() / 2
            g = [2 * (K - Kd)]
            for q in range(1, 2 * L):
                ch = cosh(2 * K) * cosh(2 * Kd) - sinh(2 * K) * sinh(2 * Kd) * cos(pi * q / L)
                g.append((ch + (ch * ch - 1).sqrt()).ln())
            base = L * (2 * sinh(2 * K)).ln() / 2
            l0 = base + sum(g[1::2]) / 2
            l1 = base + sum(g[0::2]) / 2
            return float(-Decimal(T) * l0 / L), float(1 / (l0 - l1)), float(1 / (2 * g[1]))


    # above Tc
    T = 3.0
    out = strip_lengths(4, T)
    assert isinstance(out, tuple) and len(out) == 3
    assert all(isinstance(v, float) for v in out)
    f, xi_spin, xi_energy = out
    _t_f, _t_xs, _t_xe = _t_kaufman(4, T)
    assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
    assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
    assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)


_crown_case()

# --- test case 1 ---
import numpy as np


def _crown_case():
    import numpy as np

    # Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
    # With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
    #   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
    # ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
    # even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
    # odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1. The spectrum is evaluated with
    # 60-digit decimal arithmetic (standard library): below Tc ln lambda_0 - ln lambda_1 is an exponentially
    # small difference of two sums of order L (1.4e-8 at L = 10, T = 1), which a direct double-precision
    # evaluation of these sums would give only to about 1e-7 relative.
    def _t_kaufman(L, T):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 60
            one = Decimal(1)
            pi = Decimal(0)                          # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
            for mult, x in ((16, Decimal(5)), (-4, Decimal(239))):
                term, k, s = one / x, 0, Decimal(0)
                while abs(term) > Decimal(10) ** -65:
                    s += term / (2 * k + 1)
                    term = -term / (x * x)
                    k += 1
                pi += mult * s

            def cos(x):                              # Taylor series after reduction to 0 <= x <= pi
                x = x if x <= pi else 2 * pi - x
                term, s, k = one, one, 0
                while abs(term) > Decimal(10) ** -65:
                    k += 2
                    term = -term * x * x / (k * (k - 1))
                    s += term
                return s

            def cosh(x):
                return (x.exp() + (-x).exp()) / 2

            def sinh(x):
                return (x.exp() - (-x).exp()) / 2

            K = one / Decimal(T)                     # Decimal(T) is the exact value of the float T
            Kd = -((1 - (-2 * K).exp()) / (1 + (-2 * K).exp())).ln() / 2
            g = [2 * (K - Kd)]
            for q in range(1, 2 * L):
                ch = cosh(2 * K) * cosh(2 * Kd) - sinh(2 * K) * sinh(2 * Kd) * cos(pi * q / L)
                g.append((ch + (ch * ch - 1).sqrt()).ln())
            base = L * (2 * sinh(2 * K)).ln() / 2
            l0 = base + sum(g[1::2]) / 2
            l1 = base + sum(g[0::2]) / 2
            return float(-Decimal(T) * l0 / L), float(1 / (l0 - l1)), float(1 / (2 * g[1]))


    # high temperature
    T = 5.0
    out = strip_lengths(6, T)
    assert isinstance(out, tuple) and len(out) == 3
    assert all(isinstance(v, float) for v in out)
    f, xi_spin, xi_energy = out
    _t_f, _t_xs, _t_xe = _t_kaufman(6, T)
    assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
    assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
    assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)


_crown_case()

# --- test case 2 ---
import numpy as np


def _crown_case():
    import numpy as np

    # Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
    # With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
    #   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
    # ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
    # even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
    # odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1. The spectrum is evaluated with
    # 60-digit decimal arithmetic (standard library): below Tc ln lambda_0 - ln lambda_1 is an exponentially
    # small difference of two sums of order L (1.4e-8 at L = 10, T = 1), which a direct double-precision
    # evaluation of these sums would give only to about 1e-7 relative.
    def _t_kaufman(L, T):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 60
            one = Decimal(1)
            pi = Decimal(0)                          # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
            for mult, x in ((16, Decimal(5)), (-4, Decimal(239))):
                term, k, s = one / x, 0, Decimal(0)
                while abs(term) > Decimal(10) ** -65:
                    s += term / (2 * k + 1)
                    term = -term / (x * x)
                    k += 1
                pi += mult * s

            def cos(x):                              # Taylor series after reduction to 0 <= x <= pi
                x = x if x <= pi else 2 * pi - x
                term, s, k = one, one, 0
                while abs(term) > Decimal(10) ** -65:
                    k += 2
                    term = -term * x * x / (k * (k - 1))
                    s += term
                return s

            def cosh(x):
                return (x.exp() + (-x).exp()) / 2

            def sinh(x):
                return (x.exp() - (-x).exp()) / 2

            K = one / Decimal(T)                     # Decimal(T) is the exact value of the float T
            Kd = -((1 - (-2 * K).exp()) / (1 + (-2 * K).exp())).ln() / 2
            g = [2 * (K - Kd)]
            for q in range(1, 2 * L):
                ch = cosh(2 * K) * cosh(2 * Kd) - sinh(2 * K) * sinh(2 * Kd) * cos(pi * q / L)
                g.append((ch + (ch * ch - 1).sqrt()).ln())
            base = L * (2 * sinh(2 * K)).ln() / 2
            l0 = base + sum(g[1::2]) / 2
            l1 = base + sum(g[0::2]) / 2
            return float(-Decimal(T) * l0 / L), float(1 / (l0 - l1)), float(1 / (2 * g[1]))


    # widest strip at the lowest temperature: the spin gap is 1.4e-8 relative
    T = 1.0
    out = strip_lengths(10, T)
    assert isinstance(out, tuple) and len(out) == 3
    assert all(isinstance(v, float) for v in out)
    f, xi_spin, xi_energy = out
    _t_f, _t_xs, _t_xe = _t_kaufman(10, T)
    assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
    assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
    assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)


_crown_case()

# --- test case 3 ---
import numpy as np


def _crown_case():
    import numpy as np

    # Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
    # With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
    #   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
    # ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
    # even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
    # odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1. The spectrum is evaluated with
    # 60-digit decimal arithmetic (standard library): below Tc ln lambda_0 - ln lambda_1 is an exponentially
    # small difference of two sums of order L (1.4e-8 at L = 10, T = 1), which a direct double-precision
    # evaluation of these sums would give only to about 1e-7 relative.
    def _t_kaufman(L, T):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 60
            one = Decimal(1)
            pi = Decimal(0)                          # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
            for mult, x in ((16, Decimal(5)), (-4, Decimal(239))):
                term, k, s = one / x, 0, Decimal(0)
                while abs(term) > Decimal(10) ** -65:
                    s += term / (2 * k + 1)
                    term = -term / (x * x)
                    k += 1
                pi += mult * s

            def cos(x):                              # Taylor series after reduction to 0 <= x <= pi
                x = x if x <= pi else 2 * pi - x
                term, s, k = one, one, 0
                while abs(term) > Decimal(10) ** -65:
                    k += 2
                    term = -term * x * x / (k * (k - 1))
                    s += term
                return s

            def cosh(x):
                return (x.exp() + (-x).exp()) / 2

            def sinh(x):
                return (x.exp() - (-x).exp()) / 2

            K = one / Decimal(T)                     # Decimal(T) is the exact value of the float T
            Kd = -((1 - (-2 * K).exp()) / (1 + (-2 * K).exp())).ln() / 2
            g = [2 * (K - Kd)]
            for q in range(1, 2 * L):
                ch = cosh(2 * K) * cosh(2 * Kd) - sinh(2 * K) * sinh(2 * Kd) * cos(pi * q / L)
                g.append((ch + (ch * ch - 1).sqrt()).ln())
            base = L * (2 * sinh(2 * K)).ln() / 2
            l0 = base + sum(g[1::2]) / 2
            l1 = base + sum(g[0::2]) / 2
            return float(-Decimal(T) * l0 / L), float(1 / (l0 - l1)), float(1 / (2 * g[1]))


    # exact critical temperature
    T = 2.0 / np.log(1.0 + np.sqrt(2.0))
    out = strip_lengths(8, T)
    assert isinstance(out, tuple) and len(out) == 3
    assert all(isinstance(v, float) for v in out)
    f, xi_spin, xi_energy = out
    _t_f, _t_xs, _t_xe = _t_kaufman(8, T)
    assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
    assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
    assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)


_crown_case()

# --- test case 4 ---
import numpy as np


def _crown_case():
    import numpy as np

    # Independent targets: Kaufman's exact spectrum of the periodic strip (no transfer matrix is built).
    # With K = 1/T and the dual coupling K* = -ln(tanh K) / 2, the fermion energies are
    #   cosh g_q = cosh 2K cosh 2K* - sinh 2K sinh 2K* cos(pi q / L),  q = 0, ..., 2L - 1,  g_0 = 2 (K - K*),
    # ln lambda = (L/2) ln(2 sinh 2K) + (1/2) sum of g_q over odd q (top of the even sector, lambda_0) or over
    # even q (top of the odd sector, lambda_1); the second state of the even sector carries the two lowest
    # odd-q excitations q = +-1, so ln lambda_2 = ln lambda_0 - 2 g_1. The spectrum is evaluated with
    # 60-digit decimal arithmetic (standard library): below Tc ln lambda_0 - ln lambda_1 is an exponentially
    # small difference of two sums of order L (1.4e-8 at L = 10, T = 1), which a direct double-precision
    # evaluation of these sums would give only to about 1e-7 relative.
    def _t_kaufman(L, T):
        from decimal import Decimal, localcontext
        with localcontext() as ctx:
            ctx.prec = 60
            one = Decimal(1)
            pi = Decimal(0)                          # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
            for mult, x in ((16, Decimal(5)), (-4, Decimal(239))):
                term, k, s = one / x, 0, Decimal(0)
                while abs(term) > Decimal(10) ** -65:
                    s += term / (2 * k + 1)
                    term = -term / (x * x)
                    k += 1
                pi += mult * s

            def cos(x):                              # Taylor series after reduction to 0 <= x <= pi
                x = x if x <= pi else 2 * pi - x
                term, s, k = one, one, 0
                while abs(term) > Decimal(10) ** -65:
                    k += 2
                    term = -term * x * x / (k * (k - 1))
                    s += term
                return s

            def cosh(x):
                return (x.exp() + (-x).exp()) / 2

            def sinh(x):
                return (x.exp() - (-x).exp()) / 2

            K = one / Decimal(T)                     # Decimal(T) is the exact value of the float T
            Kd = -((1 - (-2 * K).exp()) / (1 + (-2 * K).exp())).ln() / 2
            g = [2 * (K - Kd)]
            for q in range(1, 2 * L):
                ch = cosh(2 * K) * cosh(2 * Kd) - sinh(2 * K) * sinh(2 * Kd) * cos(pi * q / L)
                g.append((ch + (ch * ch - 1).sqrt()).ln())
            base = L * (2 * sinh(2 * K)).ln() / 2
            l0 = base + sum(g[1::2]) / 2
            l1 = base + sum(g[0::2]) / 2
            return float(-Decimal(T) * l0 / L), float(1 / (l0 - l1)), float(1 / (2 * g[1]))


    # narrowest strip at the highest temperature
    T = 10.0
    out = strip_lengths(3, T)
    assert isinstance(out, tuple) and len(out) == 3
    assert all(isinstance(v, float) for v in out)
    f, xi_spin, xi_energy = out
    _t_f, _t_xs, _t_xe = _t_kaufman(3, T)
    assert abs(f / _t_f - 1.0) < 1e-10, (f, _t_f)
    assert abs(xi_spin / _t_xs - 1.0) < 1e-6, (xi_spin, _t_xs)
    assert abs(xi_energy / _t_xe - 1.0) < 1e-6, (xi_energy, _t_xe)


_crown_case()
