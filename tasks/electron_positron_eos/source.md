# Sources, method and ground truth

## Physics
Ideal relativistic Fermi gas of electrons and positrons with pairs in equilibrium with radiation
(Chandrasekhar 1939; Cox and Giuli 1968; Timmes and Arnett 1999). Definitions follow the textbook
integrals over momentum. The constants are CODATA 2018 (m_e c^2, hbar/(m_e c)) and the exact SI-2019
values (k, N_A).

## Reference solution
- **Quadrature.** Composite 24-point Gauss-Legendre.
  - Below the Fermi edge, in t = sqrt(E) (E the kinetic energy in m_e c^2), with doubling panels.
  - Over the edge E_F +- 60 kT, in y = (E - E_F)/kT with panels of width 1. The electron occupation
    argument is then y itself, not a difference of two large energies; this keeps the entropy accurate
    to 1e-15 even at psi = 1e6.
- **Net electron density.** Computed from f(b - psi) - f(b + psi) = sinh(psi)/(cosh b + cosh psi). The
  integrand is positive and summed in logarithms, so there is no cancellation and no underflow.
- **Entropy.** Uses the positive integrand -[f ln f + (1-f) ln(1-f)] = log1p(e^{-|x|}) + |x|/(e^{|x|}+1).
- **psi from rho Y_e.** Bisection-safeguarded false position with the Illinois modification in ln psi on
  ln n_net (version 3; before, plain false position, see below). The variable psi includes the rest
  energy, so a pair plasma with psi = 2e-21 is represented exactly; mu/kT would round it away.
- **Specific heat.** Computed as c_V = (1/kT^2) det / <dN^2>. The determinant
  det = <dE^2><dN^2> - <dE dN>^2 = 1/2 sum_ij w_i w_j (e_i n_j - e_j n_i)^2 is evaluated term by term:
  - same-species pairs give two-pass centred second moments;
  - electron-positron pairs give (e_i + e_j)^2 > 0.

  There is no cancellation. The Schur form <dE^2> - <dE dN>^2/<dN^2>, evaluated directly, loses about
  (kT/E_F)^2 relative to its terms: the shipped schur_complement mutant is off by 1.9e-7 at
  rho_Ye = 1e13, T = 1e7 K (measured in version 4; versions 1-3 quoted 2.8e-6, which is not reproduced).
- **Helper names** (version 4). The module-level helpers and constants of the per-step files and of
  solution.py are named _epeos_* and _EPEOS_*, so that a helper with a generic name (_nodes, _log_net,
  PREF, ...) defined by a candidate in a later step cannot replace them in the shared namespace.
- **Adiabatic exponents** (version 6). gamma1 and nabla_ad of the electron-positron gas alone, at
  constant entropy per unit mass and fixed Y_e, with the pairs in equilibrium at every instant. They come
  from the grand-canonical response moments on the same edge-centred nodes. solution/step_5.py rebuilds
  these nodes under _epeos5_* names, so step 5 does not use the private helpers of another step.
  - Moments, with the weights f(1-f) of both species:
    - n_psi = kT (dn/dmu)_T = PREF sum w [f_-(1-f_-) + f_+(1-f_+)];
    - X = T (dn/dT)_mu = PREF sum w [x_- f_-(1-f_-) - x_+ f_+(1-f_+)].
  - Gibbs-Duhem with only n_net conserved (mu_+ = -mu_-) is dP = s dT + n_net dmu. It gives
    (dP/dn)_T = n_net kT/n_psi and (dP/dT)_n = s - n_net k X/n_psi.
  - The first law per baryon at constant s/n_net gives Gamma3 - 1 = (dP/dT)_n / c_V. Then
    gamma1 = chi_rho + chi_T (Gamma3 - 1) and nabla_ad = (Gamma3 - 1)/gamma1, with
    chi_rho = n_net (dP/dn)_T / P and chi_T = T (dP/dT)_n / P.
  - On these nodes x_- is the distance from the Fermi edge in units of kT, not eps/theta - psi. The
    antisymmetric edge moment X is only about kT/E_F of its own scale in a degenerate gas, and it keeps
    its relative accuracy. (dP/dT)_n then cancels only by a factor of about 3.
  - Checks (version 6): against the zero-temperature closed forms (x = p_F/(m_e c) >= 47, T <= 1.2e7 K)
    within 4.2e-12, the size of their thermal correction. Against the classical closed forms with
    thermal pairs within 2.2e-13 at five states with T >= 1e8 K, and within 2.4e-12 at rho_Ye = 1e-10
    g cm^-3, T = 1e7 K, where psi = 564 amplifies rounding. Against every stored target within 3.1e-12.
  - Cost: 0.4 ms per call on top of the 13 ms of the rest of electron_positron_eos at the degenerate
    corner.

## Second solution (versions 4 and 6)
- **Quadrature.** Adaptive QUADPACK (scipy.integrate.quad, epsrel 1e-13) instead of fixed panels: in the
  momentum p itself on the first panel, from E = 0 up to E_F - 40 kT (up to E_F when 0 < E_F < 40 kT),
  and on every panel of a non-degenerate gas (E_F <= 0); in y = (E - E_F)/kT on the panels above, split
  at E_F and E_F + 40 kT. The integrals end at max(E_F, 0) + 80 kT, where the occupations are below
  e^{-80}. quad is called with full_output=1, so QUADPACK's round-off notices are returned instead of
  raised as warnings; no warning filter is set.
- **Net electron density.** For psi >= 1 the plain difference n_minus - n_plus. Pointwise
  f_+/f_- = (e^{b-psi} + 1)/(e^{b+psi} + 1) >= e^{-2 psi} (b = (E + 1)/theta >= 1/theta), so n_plus/n_minus
  is at least e^{-2 psi}, not at most. Over the domain it stays below 0.163 (measured: largest at psi = 1,
  T = 1e11 K; the pointwise bound at b = 1/theta_max = 0.059 gives f_+/f_- <= 0.358). The difference
  therefore amplifies the relative quadrature error of n_minus and n_plus by (n_minus + n_plus)/(n_minus -
  n_plus) <= 1.39 (measured; rigorously at most 2.1). For psi < 1 the charge-susceptibility integral
  n_net(psi) = int_0^psi chi(psi') dpsi', chi = d n_net/d psi = PREF int p^2 [f_-(1-f_-) + f_+(1-f_+)] dp > 0
  (n_net(0) = 0 by charge-conjugation symmetry), with 16-point Gauss-Legendre in psi'; chi is analytic
  for |Im psi'| < pi, so on [0, 1] the rule is converged to rounding: 8 nodes already agree with 24
  within 2e-14 (measured 1.7e-14; 16 nodes within 1.5e-14, the level of the QUADPACK error of the chi
  values). This replaces the sinh identity of the reference.
- **Pressure.** From the grand potential, P = kT PREF int p^2 [ln(1 + e^{-x_-}) + ln(1 + e^{-x_+})] dp, not
  from the p^4/eps moment.
- **psi.** Brent's method on ln n_net(ln psi).
- **Specific heat.** The thermodynamic Jacobian in (T, mu): c_V = u_T - u_mu n_T / n_mu, with u_T, n_T at
  fixed mu and u_mu, n_mu at fixed T integrated from the analytic derivatives of the occupation numbers
  (df/dT at fixed mu = f(1-f) x/T, df_-+/dpsi = +- f(1-f)). The integrands are written with the occupation
  arguments x (eps = theta (x_- + psi) = theta (x_+ - psi)), so that u_T = k (A + psi B),
  u_psi = kT (B + psi W), n_T = B/T, n_psi = W with A, B, W the moments x^2, n x and 1 of the weights
  f(1-f): the mu n_T term is carried by the same integral B in u_T and in u_mu n_T/n_mu. This replaces
  the pairwise determinant of the reference.
- **Entropy.** The same stable form of the defining integrand as the reference (log1p(e^{-|x|}) +
  |x|/(e^{|x|}+1)), integrated with the quadrature above.
- **Accuracy (measured in version 4).** Every stored target to 1.1e-13 or better (the largest,
  1.0e-13, is n_plus at (rho, T, Ye) = (1e8, 3e8, 0.43), where d ln n_plus/d ln psi = -72.6). On a
  240-state (rho_Ye, T) grid with Ye = 1 the complete equations of state of the two solutions agree to
  2.7e-12; on a 190-point (T, psi) grid (ten temperatures from 1e7 to 1e11 K, psi from 1e-30 to 1e6,
  including 0.999999, 1 and 1.000001 around the switch of the n_net route) pair_densities and
  pair_thermodynamics agree to 3.2e-12 (P at T = 2.2e8 K, psi = 1e4, where 30-digit mpmath quadrature
  puts the reference within 6e-16 and the second solution within 3.2e-12).
- Versions 1-3 used QUADPACK in E, the sinh identity for n_net, the p^4/eps form of P and the same
  pairwise determinant for c_V as the reference; it agreed with the reference to 9.8e-12.
- **Adiabatic exponents (version 6).** From their definitions, without fluctuation moments, Maxwell
  relations or c_V.
  - Richardson-extrapolated five-point difference quotients (steps 4e-3 and 2e-3) of ln P and of the
    entropy per net electron sigma = s/n_net, in ln n_net and in ln T.
  - At each stencil state psi is solved afresh (Brent, then two Newton steps in psi itself). P (grand
    potential) and s come from the integrals above, without the range checks of the public functions,
    so the stencils may step just outside T = 1e7 K or rho Ye = 1e13 g cm^-3.
  - Gamma3 - 1 = -(d sigma/d ln n)_T / (d sigma/d ln T)_n,
    gamma1 = (d ln P/d ln n)_T + (d ln P/d ln T)_n (Gamma3 - 1) and nabla_ad = (Gamma3 - 1)/gamma1.
    nabla_ad never uses (dP/dT)_n. In gamma1, (d ln P/d ln T)_n (below 1e-9 in the coldest states, where
    its difference quotient keeps only a few digits) is multiplied by Gamma3 - 1.
  - The Newton steps matter in a cool classical gas: brentq stops at a tolerance in ln psi, which leaves
    psi uncertain by about 1e-15 |ln psi| psi. At psi = 564 (T = 1e7 K) ln P and sigma move by that
    absolute error, and the difference quotients divided it by 2e-3. That gave 1.6e-9 in the exponents
    before the Newton steps and 3.3e-10 after them.
  - Cost: up to 0.9 s per call (12 stencil states).
- **First momentum panel (version 6).** The panel from p = 0 up to the Fermi window is split at
  p = 1, 2, 4, .... Before the split, QUADPACK sometimes accepted a single 21-point Gauss-Kronrod rule
  over the whole panel (up to p_F = 173 at rho_Ye = 5e12 g cm^-3, T = 1.2e7 K), with an error estimate of
  8e-14 relative. The actual error was 4.9e-11, checked against the reference and against 30-digit
  mpmath quadrature. That error jumps as psi changes, and the difference quotients of the exponents turned
  it into 2e-9. After the split, pair_densities and pair_thermodynamics agree with the reference within
  1.1e-13 on 486 (T, psi) points: six temperatures from 1e7 to 1e11 K, psi from 1e-20 to 1e6, each with
  offsets of +-1 %.
- **Accuracy (measured in version 6).** On a 40-state (rho_Ye, T) grid with Ye = 1 (rho_Ye from 1e-10 to
  1e13 g cm^-3, T from 1e7 to 1e11 K, the four corners included), the complete equations of state of the
  two solutions agree:
  - within 3.2e-13 for psi, the densities, P, u and s;
  - within 1.5e-12 for c_V;
  - within 2.1e-10 for gamma1 and 3.3e-10 for nabla_ad. The largest exponent difference is at the cool
    classical corner (1e-10 g cm^-3, 1e7 K), where psi is about 1/theta, about 590, and the
    double-precision rounding of the occupation exponent (about 6e-14 absolute) is divided by the
    difference step.

  At the 19 tested (rho, T, Ye) states the exponents agree within 3.1e-10 (at (2e-10, 2e8, 0.5), the
  h^6 truncation of the stencil in T where the pair number varies as e^{-1/theta}). The second solution
  meets every stored target within 1.8e-11 for gamma1 and nabla_ad and within 1e-13 for the other keys.

## Targets (all independent of the solution code)
- **Classical gas.** The Maxwell-Juettner closed forms with modified Bessel functions, in mpmath:
  - n = theta K2(1/theta) e^{+-psi}/(pi^2 lambda^3);
  - P = n_tot k T;
  - <eps> = K1/K2 + 3 theta;
  - s per particle = k(<eps>/theta -+ psi + 1);
  - psi from n in closed form.

  These are used only where the Fermi corrections are below 1e-11. Positron densities of a classical
  positron gas also come from the Maxwell-Juettner formula.
- **Twelve (T, psi) states, from psi = 2.2e-21 to 1e6 and from 1e7 to 1e11 K.** mpmath tanh-sinh
  quadrature at 40 digits of the defining integrals (n_minus, n_plus, n_minus - n_plus computed in
  40-digit arithmetic, P, u, s), with breakpoints every kT over the Fermi edge.
- **Twelve (rho_Ye, T) states** (ten in versions 1-2). psi from an mpmath root search on those
  integrals. c_V from a centred difference of u at constant n_net (T +- 1e-7 T, psi re-solved at each
  temperature), all at 40 digits. This is a different route from the fluctuation formula of the reference.
- **Four (rho, T, Ye) states for the complete equation of state.** Same methods.
- **Two corner states of the complete equation of state** (version 3): (rho, T, Ye) = (2e13, 1e7, 0.5)
  and (2e-10, 1e11, 0.5); psi and c_V are the stored (rho_Ye, T) targets, P, u, s and the densities come
  from 40-digit mpmath quadrature at that psi (see Version 3).
- **Sommerfeld expansion** (version 5) for strongly degenerate gases. With mu = psi theta (rest energy
  included, units m_e c^2), int_1^inf H f deps = int_1^mu H deps + sum_k 2 (1 - 2^{1-2k}) zeta(2k)
  theta^{2k} H^{(2k-1)}(mu), with H = eps p (n), p^3/3 (P/m_e c^2) and (eps - 1) eps p (u/m_e c^2). The
  zero-temperature integrals are in closed form (p^3/3; [m (2m^2 - 5) p + 3 ln(m + p)]/24;
  m (2m^2 - 1) p/8 - ln(m + p)/8 - p^3/3), the derivatives of H by mpmath.diffs, terms to k = 8, 60 digits.
  s = (dP/dT) at constant mu; psi from n = rho_Ye N_A; c_V = (k/(pi^2 lambda^3)) (u_theta - u_mu n_theta /
  n_mu). Neglected: positrons and the lower end of the integral, both below e^{-190} at the states used,
  and the terms beyond k = 8 (k = 6 and k = 8 agree within 1e-22 for every new target). It reproduces
  the stored 40-digit targets of the degenerate states: n_minus, P, u, s at (T, psi) = (1e7, 1.3e5),
  (1e7, 2.8e4), (1e8, 600), (2e7, 1e6), (1e11, 1e6) and (1e11, 50) within 8e-25; psi and c_V at
  (rho_Ye, T) = (1e13, 1e7), (1e11, 1e7), (1e9, 1e8) within 2e-19, and (1e5, 1e7), (4e5, 1.2e7) (E_F
  about 60 and 120 k T) within 4e-16.
- **Bessel-function series of the Fermi-Dirac integrals** (version 5), exact for psi < 1/theta:
  1/(e^x + 1) = sum_j (-1)^{j+1} e^{-jx}, ln(1 + e^{-x}) = sum_j (-1)^{j+1} e^{-jx}/j (x > 0), with
  int p^2 e^{-a eps} dp = K2(a)/a, int p^2 eps e^{-a eps} dp = K1(a)/a + 3 K2(a)/a^2 and
  int p^2 eps^2 e^{-a eps} dp = K0(a)/a + 5 K1(a)/a^2 + 12 K2(a)/a^3 (a = j/theta). P from the grand
  potential, s from the Euler relation in 50-digit arithmetic, psi by a root search on the series and c_V =
  u_T - u_psi n_T/n_psi from the term-by-term derivatives (d/dtheta of a moment = a/theta times the next
  moment), terms until below 1e-45. It reproduces the stored 40-digit targets at (T, psi) =
  (3e9, 1.37e-16), (5e9, 0.0213) and (1e10, 2.29e-8) within 3e-25, and those at (1e8, 20) within 2e-25
  except n_plus = 1.8e-7 cm^-3, which differs by 3.3e-15 (it was taken in version 1 from the
  Maxwell-Juettner formula, which the series confirms to 1e-35 when evaluated with the decimal constants;
  the difference is far inside the 1e-10 tolerance).
- **Classical gas with thermal pairs** (version 5), in the tests: when both species are Maxwell-Juettner,
  psi = asinh(rho_Ye N_A / 2B) with B = theta K2(1/theta)/(pi^2 lambda^3), n_-+ = B e^{+-psi}, P = (n_- + n_+)
  kT, u = m_e c^2 [(n_- + n_+) E/N - n_net], s = k [(n_- + n_+)(E/(N theta) + 1) - psi n_net] and
  c_V = (2k/(pi^2 lambda^3 theta^2)) [F cosh psi - (E^2/N) tanh psi sinh psi] (N, E, F the three moments
  above at a = 1/theta). Used where the electron occupation exponent exceeds 27; the closed forms then agree
  with the exact Bessel series within 3.3e-13. This includes classical pair plasmas at T = 2e8 K (steps 1
  and 2 at psi = 0.0156, steps 3 and 4 at rho_Ye = 1e-10, 1e-9 and 1e-8 g cm^-3, step 5 at (2e-10, 2e8,
  0.5)), with up to 0.97 positrons per electron and occupation exponents above 28.4, where the closed forms
  agree with the series within 1.6e-13.
- **General case 1** (version 4) has no stored target. At (rho, T, Ye) = (1e6, 3e9, 0.5) it checks
  identities of the returned equation of state (charge neutrality, the Euler relation, c_V = du/dT =
  T ds/dT at constant rho Ye), each within its first-order error budget (see Version 4).
- **Version 6 targets for gamma1, nabla_ad and the new states.** None comes from a solution; the scripts
  (mpmath) stay outside the task folder. Every route takes the exponents from their definitions, by the
  Jacobian chain rule on directly computed n_net, P and s (sigma = s/n_net):
  gamma1 = (n/P) d(P, sigma)/d(n, sigma), nabla_ad = (P/T) d(T, sigma)/d(P, sigma), and
  c_V = T d(s, n)/d(T, n). No Maxwell relation, no fluctuation moment and no c_V input is used.
  - **M** (degenerate states). The Sommerfeld expansion above, carried to theta^24 (k = 12) in 60
    digits, with the (theta, mu) derivatives taken term by term. k = 10 and k = 12 agree within 1e-33 at
    every M target state except (rho_Ye, T) = (4.3e7, 3e8), where they agree within 1.2e-17. There E_F
    is only 53 kT, so route Q is used and M is the cross-check.
  - **S** (pair plasmas, partly degenerate pairs). The Bessel-function series above, 5 to 200 terms
    (e^{-J(1/theta - psi)} < 1e-48), 50 digits. The Jacobian comes from five-point differences in
    (ln T, ln psi) at h = 1e-10; h and 2h agree within 4e-32.
  - **P0** (pair plasmas with psi < 1e-10). The charge-conjugation limit: P, s and u_tot are even in
    psi and n_net is odd. With n_net = psi A + O(psi^3), the definitions give
    gamma1 = T (dP0/dT)^2/(P0 c_V0) and nabla_ad = P0/(T dP0/dT), with c_V = c_V0 and
    s = (U0 + P0)/T of the psi = 0 gas. These are 1-D integrals in 50-digit tanh-sinh quadrature, with
    error O(psi^2) <= 1.4e-21. At T = 1e11 K its A agrees with a 2400-term Bessel series within 3e-41.
  - **Q** (generic states). 50-digit tanh-sinh quadrature of n_net, P and s (Gibbs integrand), psi
    re-solved, and five-point differences in (ln T, ln psi) at h = 1e-10. h and 2h agree within 3e-38.
    The difference quotients of the quadrature hold Q to about 1e-18.
  - **G** (cross-check of Q). 34-digit Gauss-Legendre quadrature in p of the response moments n_psi, X
    and A = PREF sum w (x_-^2 f_-(1-f_-) + x_+^2 f_+(1-f_+)), with the analytic (T, mu) Jacobian:
    (dP/dT)_n = s - n k X/n_psi and c_V = k (A - X^2/n_psi). It uses no differences.
  - **C** (classical gases with thermal pairs, evaluated inside the tests, 40 digits).
    N_tot = sqrt(n^2 + 4B^2), (dP/dT)_n = k N_tot + 4 kT B (dB/dT)/N_tot,
    (dP/dn)_T = kT n/N_tot, gamma1 = n (dP/dn)_T/P + T (dP/dT)_n^2/(P c_V) and
    nabla_ad = (dP/dT)_n/(c_V gamma1). They agree with the definition-level route S within 2.2e-14 for
    the exponents and 3.2e-13 for the other keys at the three tested states.
  - **Zero-temperature closed forms** (cross-check only). Gamma1 = 8x^5/(3 sqrt(1+x^2)
    [x(2x^2 - 3) sqrt(1+x^2) + 3 asinh x]) and Gamma3 - 1 = (2+x^2)/(3(1+x^2)), x = p_F/(m_e c). They
    hold up to the thermal correction, of order (kT/E_F)^2.
  - Route-pair agreement for gamma1 and nabla_ad (DESIGN gate B):
    - S with P0 within 1.2e-24 (1.1e-22 at (2e-6, 1.5e9, 0.5));
    - M with Q within 3.8e-22 to 1.7e-21 at the cold degenerate states, 1.8e-19 at (2e9, 1.2e7, 0.5),
      4.2e-18 at (4e6, 1e7, 0.5) and 1.5e-18 at (1e8, 3e8, 0.43);
    - Q with S within 4.3e-18 at (1e6, 3e9, 0.5);
    - Q with G within 9.3e-19 at (2e6, 1e9, 0.5) and with P0 within 4.0e-20 at (2e-10, 1e11, 0.5);
    - M with the zero-temperature forms within 6.3e-15, 2.3e-14, 1.4e-13, 2.9e-13 and 1.9e-9 at x = 217,
      173, 101, 118 and 10.1, their thermal correction.

    The version-3 7-key targets kept in step 5 cases 0, 1, 3 and 5 agree with these routes within 1e-14.
    The only differences above 1e-16 are c_V, by the 1e-14 truncation of the centred difference that
    produced them, and, in the pair plasmas at 1e10 and 1e11 K, psi and the densities. The version-3 40-digit
    quadrature of n_minus - n_plus put them 2.4e-16 off the exact P0 values, through that cancellation.

The reference agrees with all targets to 1.4e-13 or better (remeasured in version 4, same values as in
version 3): 1.4e-13 for n_net = 9.0e-250 cm^-3 of the cool classical gas at T = 1e7 K, psi = 1e-18 (and
6.7e-14 for its n_minus, n_plus of 4.5e-232 cm^-3); up to 1.05e-13 for c_V (at rho_Ye = 1e-10, T = 1e7 K,
where d ln c_V/d ln psi = 564, and 9.4e-14 at 1e13, 1e7 K); up to 4.8e-14 for the other low-temperature
classical states (values down to 1e-109); and below 2.7e-14 for every other value (the largest is n_plus
at (rho, T, Ye) = (1e8, 3e8, 0.43)). These deviations come mostly from the rounding of theta and of
exponents near 1/theta = 593 at T = 1e7 K (condition number about 600).

## Why the naive formulas fail the stated accuracy
Measured with the shipped mutants in version 4 (versions 1-3 quoted 1.6e-5 for the entropy at
psi = 1.3e5 and 2.8e-6 for the Schur form at rho_Ye = 1e13; neither is reproduced by the shipped mutants).
- n_minus - n_plus: 100 % error in the pair plasma at T = 1e11 K, psi = 2.2e-21, 76 % at T = 3e9 K,
  psi = 1.37e-16 (net_by_subtraction).
- (u_tot + P - mu n)/T for the entropy: 7.5e-7 at psi = 1.3e5, T = 1e7 K, 4.5e-8 at psi = 2.8e4,
  T = 1e7 K, 3.2e-5 at psi = 1e6, T = 2e7 K and 2.7e-6 at psi = 1e6, T = 1e11 K (required 1e-10;
  entropy_from_euler).
- The Schur form of c_V: 1.9e-7 at rho_Ye = 1e13, T = 1e7 K, and 4.5e-8 at 1e11 g cm^-3, T = 1e7 K
  (required 1e-8; schur_complement).
- Adiabatic exponents (version 6, measured with the shipped mutants and two unshipped controls against
  the stored targets; required 1e-8):
  - **Entropy density s instead of s/rho held fixed** (adiabat_at_constant_entropy_density):
    nabla_ad off by 3.0 relative in cold degenerate gases (4 times the true value), 4.2 at
    (2e6, 1e9, 0.5) and 5.7 at (1e6, 3e9, 0.5); gamma1 near 0 in every pair plasma.
  - **Pair number frozen during the change** (pairs_frozen_in_response): gamma1 off by 8.2 to 60
    relative in pair plasmas (61 times the true value at (2e-10, 1e9, 0.5)) and by 2.4 at (1e6, 3e9, 0.5).
  - **(dP/dT)_n through (T, psi)** (thermal_pressure_from_T_psi_jacobian): s + n k psi - n kT
    (dn/dT)_psi/n_psi cancels by about psi^2. nabla_ad is off by 1.7e-7 at (2e13, 1e7, 0.5), 1.5e-7 at
    (1e13, 1.2e7, 0.5), 4.3e-7 at (2e12, 1e7, 0.5) and 6.5e-8 at (3.5e12, 2e7, 0.46). This is rounding
    noise; the values are identical with OMP_NUM_THREADS=1.
  - **c_V at constant psi** (cv_at_fixed_psi): c_V off by 5.2e-2 at (1e6, 3e9, 0.5) up to 5e9 relative
    in cold degenerate gases, and nabla_ad off by 1.0 there.
  - **Finite differences of P at fixed n_net** (control, h = 1e-4 T): nabla_ad off by 0.16 at the
    corner (2e13, 1e7, 0.5), by 3.3e-2, 1.4e-2 and 4.2e-3 at the other cold states and by 4.9e-5 at
    (4e6, 1e7, 0.5); gamma1 off by 2e-8 to 3e-4 in pair plasmas. It fails step 5 cases 0-3, 5 and 7-9
    and general cases 0-3.
  - **Isothermal exponent gamma1 = chi_rho** (control): it fails step 5 cases 0-5 and 7-9 and general
    cases 0-1. General case 1 (d) rejects it by 0.77 against a budget of 9.4e-8.
- Error budget of (dP/dT)_n in the cold, completely degenerate gases (version 6).
  - There E_F reaches 1.3e5 kT, (dP/dT)_n is chi_T = 8e-10 to 4e-9 of P/T, and it is set by the
    antisymmetric thermal moment (dn/dT)_mu, itself only about kT/E_F ~ 1e-5 of its own scale.
  - Quadrature nodes placed in the momentum p, with the Fermi argument formed as eps/theta - psi, locate
    the edge only to about p eps_mach, roughly 1e-11 kT at psi ~ 1e5. Divided by kT/E_F, this leaves a
    floor near 1e-8 in (dP/dT)_n and therefore in nabla_ad, independent of the requested quadrature
    tolerance.
  - Measured on the three blind codes of the version-6 pre-probe, written from the agent-visible text
    only: QUADPACK in p at its 1.2e-14 floor gave 6e-9 to 6.5e-8; 60- or 80-node Gauss-Legendre panels
    in p gave 1.4e-8 to 7.6e-8.
  - Routes centred on the edge reach about 1e-12: nodes in x = (E - E_F)/kT as in the reference
    (3.1e-12), the Sommerfeld expansion, and the second solution's differences of a directly computed
    entropy (within 8.1e-12 at the cold states).
  - The (T, psi) Jacobian is therefore not the only route that fails at the cold corners. Uncentred
    (T, mu) moments in p fail too, and so does the Schur-form c_V with uncentred moments. Their errors are
    1.01 to 7.4 times the tolerance, so a p-space code can occasionally clear a corner by chance (2 of 9
    cold-state results in the pre-probe).

## Version 2 (grading_fix: authoring-guide conformance)

No science, reference algorithm, stored target or input domain was changed. Each item gives the finding,
its cause, the change, the regression test and what was rerun.

- Time budgets in the tests and in the visible text.
  - Finding: every test file wrapped the evaluated functions in a `_t_budget` decorator that armed a
    SIGALRM interval timer and asserted on `time.perf_counter()` (10 s per call for steps 1-4, 30 s for
    step 5), and every step prompt, function_header docstring, problem_io, scaffold, per-step solution,
    solution.py and mutant docstring stated "Each call (must finish) within 10 s / 30 s (on one CPU core)".
  - Cause: per-call budgets were part of the version 1 design.
  - Change: the wrapper, the `signal` and `time` imports and the rebinding lines were removed from all six
    test files, and every time sentence was deleted (no sentence carried a problem size). The reference
    needs at most about 0.015 s per electron_positron_eos call (measured for the four step 5 states), so
    runtime stays reasonable without a budget.
  - Regression test: crown_check format (no clock or signal pattern left) and every case of every mode.
- Test-case format and self-containment.
  - Finding: the markers carried descriptions ("# --- test case 0: classical gas ... ---"), and all
    imports, constants, the Maxwell-Juettner oracle, the target tables and the check helpers lived in a
    preamble before case 0, so no case could run alone.
  - Cause: the files were written to be executed as a whole.
  - Change: markers are exactly "# --- test case N ---" (N = 0..n-1) with the description as comment
    lines below; every case repeats the imports, constants, oracle, target table and helper it uses
    (split with tools/selfcontain.py); the target-provenance note stays as a leading comment. Test-level
    globals were renamed with the `_t_` prefix (general case 0 used `out`, `k`, `want`, `tol`; case 1
    used `psi`, `P`, `u`, `s`, `lhs`, `rhs`; step 3 case 0 used `th`, `base`) so that a case cannot
    overwrite a module-level name of the candidate. Two comments were corrected: step 1 case 0 said
    "Mawell-Juettner", and step 1 case 2 claimed to test T = 1e7 K although it uses only T = 1e11 K.
  - Regression test: every case run alone in a fresh process (crown_check ref, second, mut, shift,
    controls).
- General case 1: bound inconsistent with the stated accuracies.
  - Finding: n_net at the psi of degeneracy_parameter (rho Ye = 5e5 g cm^-3, T = 1e9 K) was required to
    match rho Ye N_A within 3e-10, justified by "d ln n / d ln psi < 3". Measured with the reference,
    d ln n_net / d ln psi = 4.84 there (psi = 6.74, partially degenerate), so a psi with its allowed
    relative error 1e-10 plus an n_net with its allowed 1e-10 can be off by 5.8e-10 and fail.
  - Cause: the sensitivity was estimated, not computed.
  - Change: bound 6e-10 (the propagated error budget, rounded up) and the comment now states the
    computed sensitivity. The Euler-relation part keeps its 1e-8; its comment now gives the computed
    budget (3.4e-10: the largest terms n_net m_e c^2 and T s are 1.6 and 0.7 times the result) instead
    of "each side to 1e-10". Reference and second solution meet the identity to below 1e-14.
  - Regression test: general case 1 with solution.py and second_solution.py.
- Stated but untested endpoint.
  - Finding: the step 5 domain includes Ye = 1 (0 < Ye <= 1), but no test used it.
  - Cause (added in version 4): the version 1 tests were chosen by physical regime (white dwarf, pair
    plasma, helium burning, degenerate core) and never by the stated domain boundaries.
  - Change: step 5 case 2 evaluates (rho, T, Ye) = (1e6, 1e9, 1.0), which has rho Ye = 1e6 g cm^-3 exactly
    like the stored state (2e6, 1e9, 0.5), against the same 40-digit targets (the equation of state depends
    on rho and Ye only through rho Ye, as the prompt defines). n_test_cases of step 5 is now 3.
  - Regression test: step 5 case 2 passes with the reference, solution.py and the second solution and
    fails the cv_at_fixed_psi mutant and both controls.
- Visible specification.
  - Finding: the step 1 tests require a tuple of Python floats, while the step 1 prompt said only "return
    n_minus, n_plus and n_net"; the step 4 contract lists step 1 as a dependency, but the step 4 prompt
    did not name pair_densities.
  - Cause (added in version 4): the type contract and the dependency were written only in the
    function_header docstring and in the contract, not in the step prompt.
  - Change: the step 1 prompt now says "return the tuple (n_minus, n_plus, n_net) of Python floats"; the
    step 4 prompt says "the net electron density n_net of pair_densities (step 1)". No new requirement.
  - Regression test: crown_check ref and second on steps 1 and 4.
- Undeclared imports.
  - Finding: second_solution.py imports warnings, scipy.integrate.quad and IntegrationWarning, and
    scipy.optimize.brentq, which were not in required_dependencies.
  - Cause (added in version 4): required_dependencies listed only the imports of the reference.
  - Change: declared in required_dependencies; the scaffolds start with the same import lines.
  - Regression test: crown_check second (all cases).
- Metadata.
  - Finding: the version 1 problem.yaml had no subfield, tags, expert_time_estimate_hours,
    relevant_experience, author, affiliation, difficulty_explanation, solution_explanation,
    verification_explanation or edit_label, and source.md stated a second-solution agreement of "about
    1e-13".
  - Cause (added in version 4): version 1 predates the current skeleton's metadata fields; the
    agreement figure was not remeasured after the second solution was written.
  - Change: the fields were added (relevant_experience as the placeholder for the author, edit_label
    grading_fix) and the agreement was corrected to the measured 1e-11.
  - Regression test: crown_check format (metadata keys present); the corrected agreement is the version 2
    rerun (worst 9.8e-12, n_minus at T = 2e7 K, psi = 1e6).
- Floating-point comparisons: no exact float equality and no zero tolerance were present (all checks are
  relative errors with stated bounds or the stated absolute 1e-250 for tiny n_plus); nothing changed.
- Independent re-check of stored targets: n_minus, n_net, P, u and s at (T, psi) = (1e7 K, 2.8e4),
  (1e9 K, 7.8) and (1e11 K, 2.2e-21) were recomputed with 30-digit mpmath Gauss-Legendre quadrature in
  t = sqrt(E) (panels of width k T in E over and above the Fermi edge, geometric panels near E = 0), a
  route different from the tanh-sinh generation; all fifteen values agree with the stored targets to
  7e-17 or better.
- Rerun after all changes: tools/crown_check.py (all modes, every case alone in a fresh process, -j 2):
  FORMAT OK; REF steps 1-5 4/4, 3/3, 3/3, 2/2, 3/3 and general 2/2 (solution.py also passes every step
  file); SECOND steps 1-5 and general all pass; mutants fail net_by_subtraction 2/4 (cases 1, 2),
  entropy_from_euler 1/3 (case 1), positrons_ignored 1/3 (case 1), schur_complement 1/2 (case 0),
  positron_fluctuations_ignored 1/2 (case 0), cv_at_fixed_psi 2/3 (cases 0, 2),
  whole_task_classical_occupation 1/2 general (case 0); SHIFT +-1e-12 all pass; all-None and all-zero
  controls pass 0 cases. Repeat runs: solution.py and second_solution.py each run twice in clean
  processes (and once more with one BLAS/OpenMP thread) on 21 calls covering all five functions give
  bitwise identical outputs; the methods use no random numbers.

## Version 3 (grading_fix: authoring-guide conformance)

No stored target, input domain or physical model was changed. Three findings of an independent verifier
were confirmed and fixed (items 1-3). One reference bug found while checking them was repaired (item 4).
Each item gives the finding, its cause, the change, the regression test and what was rerun.

- 1. Error propagated from psi into the complete equation of state.
  - Finding: the step 5 text gave each output "the accuracy stated in its step". Step 5 cases 0 and 2
    and general case 0 compare n_minus, n_plus, P, u and s with their exact values at the exact psi,
    within 1e-10. These quantities change faster than psi. Computed d ln X / d ln psi:
    - at (rho Ye, T) = (1e6, 1e9): n_minus 4.6, n_plus -7.8, P 5.66, u 5.78, s 2.92;
    - at (4.3e7, 3e8): n_plus -72.6;
    - over the domain: up to 587 for n_minus, P and u, 548 for s, 598 for n_plus and 586 for c_V. In a
      cool classical gas the sensitivity is about psi, and psi is close to m_e c^2 / k T there.

    Reproduced: a whole solution whose degeneracy_parameter returns psi (1 +- 9e-11) passes
    tests/step_3.py 3/3 but fails general case 0 and step 5 cases 0 and 2 (n_minus 6.022142991659798e29
    against 6.022142989167459e29, relative error 4.1e-10).
  - Cause: the per-step accuracies are defined at a given psi. The integrated tests start from
    (rho, T, Ye), and the propagation through psi was never stated.
  - Change (no test tolerance changed): the step 5 step_description_prompt, function_header and
    problem_io now state the accuracies with respect to the exact values for the given (rho, T, Ye). They
    add that n_minus, n_plus, P, u and s amplify a relative error of psi up to about 600 times, so psi
    must be found to about 1e-13 relative, far better than the 1e-10 of step 3. The same docstring is in
    steps/step_5.py, solution/step_5.py, solution.py and the two mutants that define
    electron_positron_eos. The step 4 prompt now says that its 1e-8 is with respect to the exact c_V at
    the given rho_Ye and T, and that an error in psi enters c_V amplified up to about 600 times. That
    sentence has no grading effect: the step 4 context uses the reference step 3, and the general c_V
    checks are at states with d ln c_V / d ln psi below 4 (3.7 and 0).
  - Why not per-state widened tolerances: they would loosen stated tolerances. For an equation of state,
    the end-to-end accuracy at (rho, T, Ye) is the meaningful requirement, and the reference meets it
    (psi to about 1e-14).
  - Regression test: with psi scaled by 1 +- 9e-11, the whole solution fails step 5 cases 0, 2 and 3 and
    general case 0, now as specified. With psi scaled by 1 +- 1e-13 it passes every case of steps 3-5
    and general.
- 2. Inclusive endpoints of the step 5 domain were untested.
  - Finding: no step 5 or general case used T = 1e7 K, T = 1e11 K, rho Ye = 1e-10 or rho Ye = 1e13
    g cm^-3, and step 5 did not check the T > 1e11 K error path. A step 5 with exclusive bounds passed
    3/3.
  - Cause: the version 2 endpoint fix covered Ye = 1 only.
  - Change: new step 5 case 3 evaluates (rho, T, Ye) = (2e13, 1e7, 0.5) and (2e-10, 1e11, 0.5). In binary,
    rho Ye is exactly 1e13 and 1e-10.
    - psi and c_V come from the stored 40-digit (rho_Ye, T) targets.
    - P, u, s, n_minus and n_plus were computed for this version with 40-digit mpmath quadrature in
      t = sqrt(E), using tanh-sinh and Gauss-Legendre rules with breakpoints every k T over +-60 k T of
      the edge. The two rules agree to 25 digits, except for the 6e-56207 cm^-3 positron density, which
      enters only as 0. The same script reproduces the stored (T, psi) rows (1e7, 1.3e5) and
      (1e11, 2.2e-21) to all 25 printed digits.
    - Degenerate corner: n_minus = rho Ye N_A + n_plus = 6.02214076e36, since n_plus is about 6e-56207
      cm^-3; the quadrature confirms it to 5e-25.
    - Pair corner: the quadrature gives n_minus - n_plus = rho Ye N_A = 6.02214076e13 to the printed
      digits.

    (1e6, 2e11, 0.5) was added to the error list of step 5 case 1. n_test_cases of step 5 went from 3 to 4.
  - Regression test: the exclusive-bounds step 5 fails case 3 (ValueError). The reference, solution.py
    and the second solution pass, the reference within 9.4e-14 and the second solution within 4.6e-12.
- 3. n_net accuracy was promised below the range of double precision.
  - Finding: step 1 promised n_net with relative error 1e-10 for every 0 < psi <= 1e6. At T = 1e7 K and
    psi = 1e-100, the exact n_net = 2 sinh(psi) theta K2(1/theta) / (pi^2 lambda^3) is 9.0e-332 cm^-3,
    below the smallest double, and both solutions return 0.0. At psi = 1e-85 it is subnormal
    (8.998209e-317, about 7 digits). Confirmed.
  - Cause: the absolute floor given to n_plus was not given to n_net.
  - Change: the step 1 prompt and function_header now require n_plus and n_net each to have relative
    error 1e-10 when the exact value is at least 1e-250 cm^-3, and otherwise to lie within 1e-250 cm^-3.
    The same docstring is in steps/step_1.py, solution/step_1.py, solution.py and the two mutants that
    define pair_densities. The two step 1 check helpers apply the same rule; no existing target is below
    1e-250.
    - n_minus needs no floor: its smallest value in the domain is about 4.5e-232 cm^-3 (T = 1e7 K,
      psi -> 0).
    - The step 5 accuracy sentence needs no floor: step 5 does not return n_net, and rho Ye >= 1e-10
      keeps n_net >= 6.0e13 cm^-3.
  - Regression test: new step 1 case 4 checks two states at T = 1e7 K against the 40-digit
    Maxwell-Juettner closed form (Fermi corrections below 1e-257):
    - psi = 1e-18: n_net = 9.0e-250 cm^-3 with relative 1e-10 required, and n_minus = n_plus = 4.5e-232
      cm^-3;
    - psi = 1e-100: n_net within 1e-250.

    A pair_densities that sets n_net to 0 below 1e-240 fails this case. net_by_subtraction now fails
    cases 1, 2 and 4. n_test_cases of step 1 went from 4 to 5.
- 4. The reference root finder could stall (found in this pass).
  - Finding: a comparison of the complete equations of state of the two solutions on a 240-state
    (rho_Ye, T) grid showed psi differing by 5.5e-8 at rho_Ye = 1e5 g cm^-3, T = 1e7 K.
    - A 40-digit mpmath root search gives psi = 654.76623553247548. The second solution agrees to 1e-16;
      the reference returned 654.7662712217661, which makes n_net 9.1e-7 too large.
    - A scan of 8000 random (rho_Ye, T) states, half of them with T in [1e7, 1e8] K, found 39 states with
      a psi error above 1e-11, up to 1.4e-6. All of them lie in 3.8e4 <= rho_Ye <= 1.1e6 g cm^-3 and
      1.01e7 <= T <= 2.04e7 K, with psi from 350 to 660 (degenerate, non-relativistic electrons).
    - c_V inherits the error: 3.6e-7 at (1e5, 1e7) and 4.3e-6 at (4e5, 1.2e7), against the stated 1e-8.
  - Cause: the reference used plain false position, with bisection only while the bracket was wider than
    0.5 in ln psi. The slope of ln n_net against ln psi falls from about 500 on the classical side of the
    bracket to 17 at the root, so the lower end stayed fixed at ln psi = 6 (residual -195). The 300
    iterations ran out with the bracket still 0.48 wide.
  - Change: the root finder now uses the Illinois modification: the secant weight of an end kept twice in
    a row is halved. The fix is in solution/step_3.py and solution.py. It is also in the two mutants that
    copy the root finder (step_3_positrons_ignored and whole_task_classical_occupation), so each still
    differs from the reference only by its named mistake. The stopping criteria are unchanged. Over the
    same 8000 states, the psi error estimated as the n_net residual divided by d ln n_net / d ln psi is
    now at most 1.4e-14.
  - Regression test: (rho_Ye, T) = (1e5, 1e7) and (4e5, 1.2e7) were added to the 40-digit tables of step 3
    case 1 and step 4 case 0. Their targets were computed for this version:
    - psi from an mpmath root search on the n_net integral;
    - c_V from the centred difference of u_tot at constant n_net (T +- 1e-7 T, psi re-solved);
    - both with tanh-sinh and with Gauss-Legendre quadrature, which agree to 30 digits;
    - c_V cross-checked against the 40-digit fluctuation formula, which agrees within 1e-17.

    The script reproduces the stored psi and c_V at (1e6, 1e9) to 25 digits. With the old root finder,
    step 3 case 1 fails (psi 654.7662712217661 against 654.7662355324754), and so does step 4 case 0
    with the old step 3 as context (c_V 696370982242.6793 against 696370732683.6246). The fixed
    reference is within 1.6e-15 for psi and 7.1e-15 for c_V, and the second solution passes.
- 5. Other text.
  - solution_explanation: the root finder description.
  - difficulty_explanation: the 1e-250 floor and the end-to-end accuracy.
  - verification_explanation: twelve (rho_Ye, T) states, the new targets, the repeat runs and the grid
    comparison.
  - The method and target paragraphs above, and the reference-agreement statement. The earlier "below
    2e-14 for the others" did not hold for the low-temperature classical states of step 1 case 0 and
    step 2 case 0, which agree to 4.8e-14 and 2.7e-14.
- Rerun after all changes:
  - tools/crown_check.py, all modes, every case alone in a fresh process, -j 2.
    - FORMAT OK.
    - REF: steps 1-5 pass 5/5, 3/3, 3/3, 2/2 and 4/4, and general passes 2/2. solution.py also passes
      every step file.
    - SECOND: steps 1-5 and general all pass.
    - Mutants: net_by_subtraction fails 3/5 (cases 1, 2, 4), entropy_from_euler 1/3 (case 1),
      positrons_ignored 1/3 (case 1), schur_complement 1/2 (case 0), positron_fluctuations_ignored 1/2
      (case 0), cv_at_fixed_psi 3/4 (cases 0, 2, 3) and whole_task_classical_occupation 1/2 general
      (case 0).
    - SHIFT +-1e-12: all pass.
    - Controls: all-None and all-zero pass 0 cases.
  - Repeat runs: solution.py and second_solution.py were each run three times in clean processes, once
    with one BLAS/OpenMP thread, on 32 calls covering all five functions and every new state. The
    outputs were bitwise identical; the methods use no random numbers.
  - On a 240-state grid (rho_Ye = 1e-10 ... 1e13 g cm^-3 by decades, ten temperatures from 1e7 to
    1e11 K, Ye = 1), the complete equations of state of the two solutions agree to 4.6e-12.

## Version 4 (grading_fix: authoring-guide conformance, pass 2)

No stored target, input domain, tolerance of a stored-target check or reference algorithm was changed.
The findings come from the round-1 independent verifier and from the pass-2 rules (every general case
calls the final function, no hidden test inputs in the visible text, truthful explanations, a genuinely
different second solution, mutants built on the current reference). Each item gives the finding, its
cause, the change, the regression test and what was rerun.

- 1. General case 1 never called electron_positron_eos.
  - Finding: general case 1 called degeneracy_parameter, pair_densities and pair_thermodynamics only
    (crown_check format: "general case 1 never calls the final function"). Its Euler-relation tolerance
    of 1e-8 was also about 30 times its own error budget (3.4e-10).
  - Cause: the case was written as a cross-step consistency test of steps 1-3 before the final function
    was required in every general case.
  - Change: case 1 now tests the complete equation of state at (rho, T, Ye) = (1e6 g cm^-3, 3e9 K, 0.5),
    where psi = 0.65 and n_plus/n_minus = 0.29, so the positrons enter every quantity. It needs no stored
    target. It checks three identities of the returned dict:
    - charge neutrality n_minus - n_plus = rho Ye N_A;
    - the Euler relation u + n_net m_e c^2 + P - T s = psi k T n_net, with n_net = rho Ye N_A;
    - c_V = du/dT and c_V = T ds/dT at constant rho Ye, by the five-point centred difference of the
      returned u and s at T +- h, T +- 2h with h = 5e-3 T.

    Each tolerance is the first-order error budget when every returned quantity has its stated accuracy,
    computed in the test from the returned values: 1e-10 (n_minus + n_plus)/(rho Ye N_A) = 1.8e-10 for
    neutrality; 1e-10 (|u| + |P| + T|s| + |psi k T n_net|)/|psi k T n_net| = 2.6e-9 for the Euler relation;
    for c_V, 1e-8 (c_V itself) + the propagated 1e-10 of the four u or s values (6.5e-9 and 1.0e-8) +
    1.2e-8 for the h^4 truncation of the stencil. The truncation is a property of the exact u(T) and s(T):
    the five-point difference of the reference deviates from c_V by 4.8e-9 (u) and 6.0e-9 (s) at
    h = 5e-3 T and by 7.7e-8 and 9.7e-8 at h = 1e-2 T (ratio 16, so the h^4 term dominates); the
    allowance is about twice the larger value. The old cross-step checks are covered by the new case: the
    neutrality check tests degeneracy_parameter with pair_densities end to end, the Euler relation links
    psi with P, u and s.
  - Regression test: crown_check format no longer reports the case. Reference and second solution pass
    with deviations 0 and 1.0e-14 (neutrality), 3.9e-15 and 2.5e-14 (Euler), 4.8e-9 and 6.0e-9 (both,
    c_V from u and s; the truncation). Each of six whole solutions with one plausible error fails case 1:
    psi from n_minus = rho Ye N_A (positrons ignored), s without positrons, P without positrons, u without
    the positron rest energy, c_V without positron fluctuations, c_V as du/dT at constant psi. The shift
    mode (all five functions scaled by 1 +- 1e-12 at once) passes.
- 2. Exact key set of the step 5 dict.
  - Finding: tests/step_5.py requires set(out) == the seven keys, but the prompt and docstrings said only
    "a dict with the keys ..."; a dict with an extra key such as "n_net" fails every value case.
  - Cause: the wording described the required keys, not the exclusion of others.
  - Change: the step 5 prompt says "a dict with exactly the seven keys ...", and the step 5
    function_header, problem_io, steps/step_5.py, solution/step_5.py, solution.py and the two mutants that
    define electron_positron_eos say "dict with exactly the seven keys ..., whose values are Python floats".
    The tests are unchanged.
  - Regression test: all docstrings of the five functions in steps/, solution/, solution.py and mutants/
    were compared with function_header (and problem_io) by a script; all are identical.
- 3. Hidden test inputs in the prompts of steps 1-4.
  - Finding: the step prompts named exact test states with their approximate outputs: step 1 (T = 1e11 K,
    psi = 2.2e-21), step 2 (psi = 1.3e5, T = 1e7 K), step 3 (rho_Ye = 1e13, T = 1e7 K and rho_Ye = 1e-10,
    T = 1e11 K) and step 4 (rho_Ye = 1e13, T = 1e7 K). The step 1 prompt also called the Fermi edge "many
    hundreds of k T wide", which contradicts itself (the edge is a few k T wide; the Fermi sea is up to
    1e6 k T deep).
  - Cause: the regimes were illustrated with the corner states that the tests also use.
  - Change: the prompts state regimes and magnitudes that hold over the domain:
    - step 1: for psi up to 1e6 the Fermi energy reaches about 1e6 k T with an edge a few k T wide; in a
      pair plasma n_net/n_minus is of order psi for small psi;
    - step 2: the entropy of a degenerate gas is about 1e-11 of (u + P)/T for psi near 1e6 (measured
      9.9e-12 at T = 2e7 K and at 1e11 K);
    - step 3: psi ranges from above 1e5 in the densest, coolest gases to about 2e-21 in the most dilute,
      hottest pair plasmas (1.289e5 and 2.17e-21 at the corners);
    - step 4: c_V T/u falls below 1e-9 in the densest, coolest gases (7.97e-10 at the corner).

    The only numbers left in the agent-visible text are constants, tolerances, domain bounds and these
    magnitudes (checked by a script listing every number in problem_description_main, the prompts, the
    function headers, problem_io and required_dependencies).
  - Regression test: crown_check ref and second (no test changed by this item).
- 4. step_dependencies of step 2 and the wording of step 4.
  - Finding: step 2 listed step 1 as a dependency although pair_thermodynamics does not use
    pair_densities; its prompt said "occupation numbers ... as in pair_densities, step 1".
  - Cause: the shared conventions were referred to through the step 1 function.
  - Change: step 2 step_dependencies is [] and its prompt and docstring refer to the conventions of
    problem_description_main. The step 4 prompt now says to use degeneracy_parameter (step 3) for psi; it
    keeps [1, 2, 3] because c_V is defined as the temperature derivative of the energy density of
    pair_thermodynamics along the path on which n_net of pair_densities is fixed (stated in its contract
    assumptions).
  - Regression test: crown_check ref, second and mut for steps 2 and 4.
- 5. Test comments.
  - Finding: step 3 case 0 said "positrons below 1e-30 of the electrons"; 40-digit values are e^{-2 psi}
    = 1.4e-19 at (1e-8, 1.2e8) and 2.1e-24 at (1e-10, 1e8).
  - Cause: the bound was not recomputed when the states were chosen.
  - Change: "positrons below 2e-19 of the electrons, shifting psi by less than 1e-20 relative" (6.6e-21);
    the target is unchanged. The other numbers quoted in test comments were remeasured with the reference
    (step 3 case 1 and step 4 case 0: E_F/kT = 61.8 and 121.6, d ln n_net/d ln psi = 16.7 and 8.4,
    d ln c_V/d ln psi = 6.6 and 3.8; step 1 cases 0 and 4: smallest occupation exponents 39.3 and 593.1).
  - Regression test: crown_check ref (comments only).
- 6. Comparison tolerance in step 1 case 2.
  - Finding: n_minus(1e11 K, 2e-20) and n_minus(1e11 K, 1e-20), two computed values that differ by a
    relative 1e-20, were compared within 1e-9, five times the convention for two outputs each within
    1e-10.
  - Cause: the bound was set loosely in version 1.
  - Change: 2e-10 (twice the single-output tolerance), with the reason in the comment; the ratio check
    n_net(2 psi)/n_net(psi) = 2 within 4e-10 was already at that convention.
  - Regression test: reference, solution.py and second solution pass (deviation below 1e-15); shift
    +-1e-12 passes; net_by_subtraction still fails case 2.
- 7. Explanation numbers that were not reproduced.
  - Finding: the entropy from the Euler relation was said to be wrong by 1.6e-5 at psi = 1.3e5, T = 1e7 K
    and the Schur-form c_V by 2.8e-6 at rho_Ye = 1e13, T = 1e7 K; the shipped mutants give 7.5e-7 and
    1.9e-7. The verification text said the 1 +- 1e-12 scaling passed "every case", while crown_check then
    shifted only the step cases.
  - Cause: the numbers came from earlier, differently summed versions of the naive formulas; the shift
    claim predated the checker's general-case shift.
  - Change: difficulty_explanation, the Reference solution paragraph and "Why the naive formulas fail"
    give the values measured with the shipped mutants (entropy 7.5e-7 at psi = 1.3e5, T = 1e7 K and 3.2e-5
    at psi = 1e6, T = 2e7 K; Schur form 1.9e-7 at 1e13 and 4.5e-8 at 1e11 g cm^-3, T = 1e7 K), labelling
    the old figures as versions 1-3. The reference-agreement paragraph gives the remeasured values per
    category (largest 1.4e-13, c_V up to 1.05e-13, everything except the low-temperature classical states
    and c_V below 2.7e-14). verification_explanation now describes what crown_check runs today (step
    cases shifted one function at a time, general cases with all five functions shifted at once), the
    new second solution, general case 1 and the remeasured grid agreement.
  - Regression test: every quoted number was remeasured in this version with the scripts kept outside the
    task folder (mutant errors, reference and second solution against all stored targets, grids).
- 8. Second solution.
  - Finding: second_solution.py reused the reference's reformulations (the sinh identity for n_net, the
    same pairwise determinant for c_V) with only QUADPACK and Brent's method differing, and called
    warnings.simplefilter("ignore", IntegrationWarning) at module level, which changes process-wide state
    and was the only reason for "import warnings" and IntegrationWarning in the visible
    required_dependencies.
  - Cause: the version 1 second solution was written as a check of the numerics, not of the
    reformulations.
  - Change: second_solution.py was rewritten (see Second solution above): n_net from n_minus - n_plus for
    psi >= 1 and from the integral of the charge susceptibility over psi for psi < 1; P from the grand
    potential; c_V from the Jacobian of (u_tot, n_net) in (T, mu) with analytic derivative integrands;
    quadrature in the momentum below the Fermi window (which also removed its earlier 1e-11 error at
    psi = 1e6). quad is called with full_output=1, which returns QUADPACK's round-off notices instead of
    issuing warnings, so no filter and no catch_warnings block is needed; "import warnings" and
    IntegrationWarning were removed from required_dependencies and from the five scaffolds. The entropy
    still uses the stable form of the defining integrand; it is the definition itself, and the
    alternative routes (the Euler relation, d P/d T at fixed mu) cancel in a degenerate gas.
  - Regression test: crown_check second passes every case of every step and of general; against every
    stored target the second solution is within 1.1e-13 (before: 9.8e-12); on the 240-state (rho_Ye, T)
    grid the two complete equations of state agree to 2.7e-12, and on a 190-point (T, psi) grid around
    the psi = 1 switch the densities, P, u and s agree to 3.2e-12 (30-digit mpmath quadrature confirms the
    reference at the worst point). A call of electron_positron_eos takes at most 0.06 s.
- 9. Generic helper names in the shared evaluation namespace.
  - Finding: the reference functions of steps 1-3 looked up module-level helpers and constants with
    generic names (_nodes, _occ, _xp, _log_net, _net, _check_T_psi, _sigma, PREF, KB, MEC2, NA, ...) at
    call time. A candidate's later step defining a helper with the same name and another meaning replaced
    them: reference steps 1-3 plus the reference step 4 plus "def _log_net(...): return 0.0" failed step 4
    case 0 with "OverflowError: math range error" inside degeneracy_parameter (reproduced with the
    version 3 files).
  - Cause: the per-step reference files were written as stand-alone modules.
  - Change: the helpers and constants of solution/step_1..4.py, solution.py and the six mutants built on
    them were renamed with the prefixes _epeos_ and _EPEOS_ (no other change).
  - Regression test: the same construction, with a candidate step 4 that also redefines _nodes, _occ,
    PREF, KB, MEC2 and NA, now passes both cases of step 4; crown_check ref, mut, shift and controls pass.
- 10. Whole-task mutant.
  - Finding: whole_task_classical_occupation replaced only the occupation function by e^{-x}. It kept
    the Fermi-Dirac net density (so psi was Fermi-Dirac), the Fermi-Dirac entropy integrand, and used the
    fluctuation weight e^{-x}(1 - e^{-x}), which is negative for x < 0: not one coherent mistake.
  - Cause: the mutant was made by editing a single helper.
  - Change: the mutant is the current reference with classical (Maxwell-Boltzmann) statistics throughout:
    occupation e^{-x} without Pauli blocking (weight f instead of f(1 - f)), the classical net-density
    integrand e^{psi - b}(1 - e^{-2 psi}) and the classical entropy per state (x + 1) e^{-x}.
  - Regression test: it fails general case 0 (psi off by 10 % at (2e6, 1e9, 0.5)); as expected for a
    thermodynamically consistent classical gas, it passes the identities of general case 1.
- 11. Version 2 notes without a Cause line.
  - Finding: the Version 2 items "Stated but untested endpoint", "Visible specification", "Undeclared
    imports" and "Metadata" had no Cause line, and the metadata item had no finding/change/regression
    structure.
  - Change: the Cause lines were added there, marked "(added in version 4)", and the metadata item was
    given the same structure; the content of the changes is as recorded in version 2.
- 12. Wrong inequality in the description of the second solution's net density (round-2 verifier).
  - Finding: the Second solution paragraph and the header comment of second_solution.py said that for
    psi >= 1, n_plus/n_minus <= e^{-2 psi}, "so the difference loses at most a factor 1.3", and that 8
    psi-nodes agree with 24 "within 1e-14". Confirmed. Pointwise, with b = (E + 1)/theta,
    f_+/f_- = (e^{b-psi} + 1)/(e^{b+psi} + 1) >= e^{-2 psi} (equivalent to e^{2 psi} >= 1), so n_plus/n_minus
    is at least e^{-2 psi}. Measured with the reference pair_densities at psi = 1: n_plus/n_minus = 0.1353
    (1e7 and 1e8 K), 0.1356 (1e9 K), 0.1454 (3e9 K), 0.1596 (1e10 K), 0.1629 (1e11 K), and the error
    amplification (n_minus + n_plus)/(n_minus - n_plus) = 1.313, 1.314, 1.340, 1.380, 1.389. On a 168-point
    grid (21 temperatures from 1e7 to 1e11 K, psi from 1 to 10) the largest ratio is 0.1629 (psi = 1,
    T = 1e11 K), amplification 1.389. The pointwise bound at b = 1/theta_max = 0.0593 (T = 1e11 K) and
    psi = 1 gives f_+/f_- <= 0.358, hence a rigorous amplification bound of 2.11. For the psi-quadrature,
    on a 252-point grid (21 temperatures from 1e7 to 1e11 K, twelve psi from 1e-20 to 0.999999), 8 nodes
    agree with 24 within 1.70e-14 and 16 nodes within 1.44e-14 (both largest at T = 1e7 K), the level of
    the QUADPACK error of the chi values; the Gauss-Legendre truncation is below it.
  - Cause: the e^{-2 psi} limit of large b (the cool gas) was written as an upper bound, although it is
    the lower limit of f_+/f_-; the node-comparison figure was given as 1e-14 instead of the value on a
    stated grid.
  - Change: the Second solution paragraph and the second_solution.py header comment now say
    n_plus/n_minus >= e^{-2 psi}, below 0.163 over the domain (measured; pointwise f_+/f_- <= 0.36), so the
    difference amplifies the quadrature error by at most 1.39 (rigorously at most 2.1), and "8 nodes
    already agree with 24 within 2e-14". In the same review, the quadrature description in both places was
    made exact (which panels are integrated in p and which in y, and the cut at max(E_F, 0) + 80 kT); the
    code is unchanged. No test, target, tolerance or code statement changed.
  - Effect on the tests: none. The n_net error of the second solution for psi >= 1 stays near 1.4 times
    its QUADPACK error; every stored target is still met to 1.1e-13 (largest: n_plus at (rho, T, Ye) =
    (1e8, 3e8, 0.43)).
  - Regression test: the measurement scripts above (kept outside the task folder) and crown_check all
    modes (results below, same as before this item: only comments and source.md changed).
- Rerun after all changes (including item 12):
  - tools/crown_check.py, all modes, every case alone in a fresh process, one BLAS/OpenMP thread, -j 2:
    - FORMAT OK (no problem reported).
    - REF: steps 1-5 pass 5/5, 3/3, 3/3, 2/2 and 4/4, general passes 2/2; solution.py passes every step
      file.
    - SECOND: steps 1-5 and general all pass.
    - Mutants: net_by_subtraction fails 3/5 (cases 1, 2, 4), entropy_from_euler 1/3 (case 1),
      positrons_ignored 1/3 (case 1), schur_complement 1/2 (case 0), positron_fluctuations_ignored 1/2
      (case 0), cv_at_fixed_psi 3/4 (cases 0, 2, 3) and whole_task_classical_occupation 1/2 general
      (case 0).
    - SHIFT +-1e-12: every step case and every general case (all five functions shifted) pass.
    - Controls: all-None and all-zero pass 0 cases of every step and of general.
    - SUMMARY ALL OK.
  - Repeat runs: solution.py and second_solution.py were each run three times in clean processes (once
    with one BLAS/OpenMP thread) on 62 calls covering all five functions and every test state; the
    outputs were bitwise identical. The methods use no random numbers.

## Version 5 (grading_fix: platform content check, edge-case inputs and science vs validation)

No reference algorithm, input domain, tolerance or existing stored target was changed. Each item gives
the finding, its cause, the change, the regression test and what was rerun.

- 1. Non-finite inputs in the tests (platform content check of v2, blocking, steps 1-5).
  - Finding: "This step's tests pass inputs at representation limits: a non-finite value float("nan") /
    float("inf"). Each should be an input the science meets and the prompt states how to handle." The
    prompts did state "raise ValueError if T or psi is not finite"; the check still reported it.
  - Cause: the validation cases of steps 1-5 passed NaN and infinity to the evaluated functions. No
    equation-of-state calculation meets a non-finite density or temperature, so the platform does not
    accept them as scientific edge cases even when the prompt states the behaviour.
  - Change: removed every non-finite input (step 1: (nan, 1), (1e9, inf); step 2: (nan, 1), (1e9, nan);
    step 3: (nan, 1e9), (1, inf); step 4: (nan, 1e9), (1, nan); step 5: (nan, 1e9, 0.5), (1e6, inf, 0.5),
    (1e6, 1e9, nan)) and the "not finite" clauses from every agent-visible text and copy: problem_io, the
    five step prompts and function headers in problem.yaml, steps/*.py, solution/*.py, solution.py and
    the mutant docstrings. The solutions still reject non-finite input internally; no text promises it. The
    contracts never mentioned it. Redundant range entries were trimmed (step 1: psi = -1 next to psi = 0;
    step 5: T = 5e6 K, already covered by steps 1-4, and the non-finite entries). Remaining raise entries:
    4, 4, 4, 4 and 5 for steps 1-5, all range or Ye conditions stated in the prompts.
  - Regression test: grep of the task folder finds no float("nan"), float("inf"), np.nan, np.inf,
    math.nan or math.inf in tests/ and no "not finite" in any visible text.
- 2. Science outweighs validation (platform content check of v2, blocking, step 4).
  - Finding: "This step's tests check validation, exception, type or shape behaviour 3 times (6 expecting a
    raise, 6 of them stated in the prompt and counted at half weight) and computed values 1 times."
  - Cause: step 4 had one computed-value case (twelve states in one loop) and one validation case. Steps 2,
    3 and 5 had the same pattern (one or two computed cases, each with a single assertion over many states,
    while the type/shape check was repeated in every case), so their counts were close too.
  - Change (counts are assertion sites; a raise entry stated in the prompt counts one half; the type/shape
    contract of each function is now asserted once per step, in its case 0, instead of in every case):
    - Step 4 (computed 1 / validation 3 before; 8 / 3 after, of which 1 type check and 4 raise entries):
      case 0, classical gas without pairs at (rho_Ye, T) = (1e-10, 1e7), (1e-10, 1e8), (1e-8, 1.2e8):
      c_V = n k d/dtheta [K1(1/theta)/K2(1/theta) + 3 theta] in closed form, evaluated in the test; case 1,
      classical gas with thermal pairs at T = 1.4e8 K, rho_Ye = 1e-10 and 1e-9 (pair creation raises c_V by
      1.5e-5 and 1.5e-7), closed form evaluated in the test; case 2, a classical pair plasma at T = 2e8 K,
      rho_Ye = 1e-10, 1e-9, 1e-8 (0.97, 0.73 and 0.086 positrons per electron, c_V 3.9e4, 3.8e3 and 210
      times that of the electrons alone), same closed form; case 3, Sommerfeld targets at (1e13, 1e9),
      (2e6, 1e7), (1e11, 5e7); case 4, Bessel-series targets at (1e-10, 1e9) and (1e4, 2e9) (Fermi-Dirac
      corrections 1.2e-3 and 1.6e-2 of c_V); the twelve 40-digit states split by regime into cases 5 (five
      degenerate gases), 6 (five pair plasmas) and 7 (two partly degenerate or classical gases).
      n_test_cases 2 -> 9.
    - Step 3 (computed 2 / validation 3 before, plus the type check folded into both computed asserts;
      7 / 3 after, of which 1 type check and 4 raise entries): case 1, classical pair plasma at T = 2e8 K,
      psi = asinh(rho_Ye N_A / 2B) in closed form at rho_Ye = 1e-10, 1e-9, 1e-8 (solving n_minus = rho_Ye
      N_A instead puts psi off by up to a factor 220); case 2, Sommerfeld psi at the three degenerate states
      of step 4; case 3, Bessel-series psi at the two pair plasmas of step 4; the twelve 40-digit states split
      into cases 4 (degenerate), 5 (pair plasmas) and 6 (partly degenerate or classical). n_test_cases
      3 -> 8.
    - Step 2 (computed 2 / validation 5 before, of which 2 type/shape assertions; 18 / 3 after, of which 1
      type/shape assertion and 4 raise entries): P, u and s are asserted separately; case 0 gains a classical
      pair plasma at (T, psi) = (2e8, 0.0156) (0.97 positrons per electron); new case 1, Sommerfeld P, u, s
      at (3e7, 2e4), (1e9, 300), (5e9, 200); new case 2, Bessel-series P, u, s at (1e9, 1) and (2e9, 0.3)
      (Fermi-Dirac corrections 6e-4 to 1.1e-2); the twelve 40-digit states split into cases 3 (degenerate),
      4 (pair plasmas) and 5 (classical and partly degenerate). n_test_cases 3 -> 7.
    - Step 5 (computed 5 / validation 10.5 before, of which 6 type/shape assertions; 10 / 3.5 after, of
      which 1 contract assertion (exact key set, Python-float values) and 5 raise entries): the four
      40-digit states split into cases 0 and 1; new case 2, the complete equation of state of classical
      gases with thermal pairs in closed form, evaluated in the test, at (rho, T, Ye) = (2e-10, 1.4e8, 0.5),
      (1e-8, 1.2e8, 1) and the pair plasma (2e-10, 2e8, 0.5); new case 3, Sommerfeld targets for all seven
      quantities at (2e13, 1e9, 0.5) and (4e6, 1e7, 0.5). n_test_cases 4 -> 7.
    - Step 1 (computed 14 / validation 6.5 before; 14 / 3 after, of which 1 type/shape assertion and 4 raise
      entries): case 0 gains the classical pair plasma (2e8, 0.0156). The general tests (two computed
      cases, four computed assertions, no validation) needed no change.
  - Target provenance: the new targets come from the Sommerfeld expansion, the Bessel series and the
    classical closed forms described under Targets; none uses the reference or the second solution. The
    reference agrees with them to 1.6e-14 (Sommerfeld and Bessel targets) and 3.3e-13 (classical closed
    forms, which omit Fermi corrections up to 3.2e-13); the second solution to 9.3e-15 and 3.3e-13. At the
    T = 2e8 K pair plasmas the closed forms agree with the exact Bessel series within 1.6e-13, and both
    solutions agree with the series within 8.5e-15. The generator scripts (mpmath) were kept outside the
    task folder.
  - Regression test: each new case rejects at least one shipped mutant or control: entropy_from_euler fails
    step 2 cases 1 and 3; positrons_ignored fails step 3 cases 1, 3, 5 and 6; positron_fluctuations_ignored
    fails step 4 cases 1, 2, 4, 6 and 7; schur_complement fails step 4 case 5; cv_at_fixed_psi fails step 5
    cases 0-5; all-None and all-zero controls pass no case.
  - Scientific grounding and stated requirements (checked, no change needed): no output with a relative
    accuracy requirement can be exactly zero (n_minus, P, u, s, psi and c_V are positive on the whole
    domain; n_plus and n_net have the absolute 1e-250 cm^-3 rule below 1e-250 cm^-3, exercised in step 1
    case 4 and step 5 case 5); every stated output form is asserted once per step (a tuple of three Python
    floats, a Python float, a dict with exactly seven keys of Python floats); the inclusive endpoints are
    exercised with values: T = 1e7 and 1e11 K and psi = 1e6 (steps 1 and 2), rho_Ye = 1e-10 and 1e13 g cm^-3
    at T = 1e7 and 1e11 K (steps 3 and 4), Ye = 1 and rho Ye = 1e-10 and 1e13 g cm^-3 (step 5).
- Rerun after these changes:
  - tools/crown_check.py, all modes, every case alone in a fresh process, one BLAS/OpenMP thread, -j 2:
    - FORMAT OK (no non-finite literal in tests/, nothing before the first case marker).
    - REF: steps 1-5 pass 5/5, 7/7, 8/8, 9/9 and 7/7, general passes 2/2.
    - SECOND: steps 1-5 and general all pass.
    - Mutants: net_by_subtraction fails 3/5 (cases 1, 2, 4), entropy_from_euler 2/7 (cases 1, 3),
      positrons_ignored 4/8 (cases 1, 3, 5, 6), schur_complement 1/9 (case 5), positron_fluctuations_ignored
      5/9 (cases 1, 2, 4, 6, 7), cv_at_fixed_psi 6/7 (cases 0-5) and whole_task_classical_occupation 1/2
      general (case 0).
    - SHIFT +-1e-12: every step case and every general case pass.
    - Controls: all-None and all-zero pass 0 cases of every step and of general.
    - SUMMARY ALL OK.

## Version 6 (difficulty_increase: adiabatic exponents in the final step, recipes and hints removed)

Platform version 4 of the task; edit_label difficulty_increase. The design brief and the pre-probe
verdicts are kept outside the task folder. Each item gives the finding, its cause, the change, the
regression test and what was rerun.

- 1. Finding.
  - The previous version passed the content check, then failed the platform's blocking difficulty probe:
    a quick model solved the task end to end in 2 of 2 attempts (difficulty misses in a row: 1 of 3).
  - Two calibrated blind solvers that reproduce the platform's quick model also solved it fully.
- 2. Cause.
  - The final step added no science: it was a recipe that chained steps 1-4.
  - The integrated tests used only mild states: a white dwarf at 1e9 K, a pair plasma at 1e10 K and the
    identity check at (1e6, 3e9, 0.5).
  - The step 1 and 2 prompts handed over the defining integrals, including the non-cancelling form of
    the entropy integrand.
  - The step 1, 2 and 4 prompts named the cancellations to avoid: "n_net/n_minus of order psi", entropy
    "a tiny fraction of (u + P)/T", and c_V "smaller than u/T by (kT/E_F)^2".
- 3. Change: agent-visible text. It now equals the pre-probed specification word for word (checked with
  tools/visible_spec.py; the diff is empty).
  - Main text: the integrals are removed. The conventions now state the phase-space normalization (two
    states per (2 pi hbar)^3), that the radiation and the ions are not part of the gas, and that the input
    ranges bound the arguments and are not limits of the gas.
  - Steps 1 and 2: the integrals and the method hints are removed. The bounded domain facts are kept:
    Fermi energies up to about 1e6 kT, pair plasmas with up to about 20 decades of pair excess, and net
    densities below double precision.
  - Steps 2 and 4: the clauses "in which the entropy per electron is only about 1e-5 k" and "c_V is
    smaller than u/T by a factor proportional to (kT/E_F)^2 ..." are removed. Step 4 gains the domain
    sentence on Fermi energies up to about 1.3e5 kT in the densest, coolest gases.
  - Step 5: the call chain and "so psi must be found to about 1e-13" are removed. The result gains gamma1
    = (d ln P/d ln rho) and nabla_ad = (d ln T/d ln P), both at constant s/rho and fixed Ye, of the
    electron-positron gas alone with the pairs in equilibrium, at 1e-8 relative accuracy everywhere in
    the domain, boundaries included.
  - Contracts of steps 1, 2 and 5 are updated to match; the step 5 step_background is rewritten;
    background.md gains a paragraph on Gamma1 and nabla_ad.
- 4. Change: reference, second solution and targets.
  - The reference adds the response helper described under Reference solution.
  - The second solution adds the finite-difference route and the panel fix described under Second
    solution. Its docstrings are now those of the visible text.
  - New targets come from routes M, S, P0, Q, G and C (see Targets). Their agreements are listed there.
- 5. Change: tests.
  - tests/step_5.py goes from 7 to 10 cases; every computed case grades all nine values.
    - Case 0 adds the contract (exactly nine keys, Python floats).
    - Case 2 extends the classical closed forms to the exponents.
    - Case 7 (new): the cold degenerate gases (1e13, 1.2e7), (2e12, 1e7) and (2e9, 1.2e7).
    - Case 8 (new): the pair-softened plasmas (2e-10, 1e9) and (2e-6, 1.5e9).
    - Case 9 (new): the rescaled pair plasmas (2e-10, 5e9) and (2e2, 5e9). Their c_V, gamma1 and
      nabla_ad must meet the targets and agree within 2e-8 plus the difference of the two stored
      targets (1.3e-10). Requiring 2e-8 alone could reject a compliant pair with errors of +-1e-8, and
      the pre-probe verdict asked for this threshold.
  - tests/general.py goes from 2 to 4 cases.
    - Case 0 grades all nine values.
    - Case 1 adds (d), the adiabatic consistency of the returned P and s along constant s/rho by
      five-point differences with a first-order error budget computed in the test, and (e), the
      Bessel-series targets of (1e6, 3e9, 0.5).
    - Case 2 (new): the three cold degenerate states (1e13, 1.2e7), (2e13, 1e7) and (2e12, 1e7).
    - Case 3 (new): the pair-softened plasma (2e-10, 1e9).
  - Every case stays self-contained and wrapped in _crown_case(), with nothing before the first marker.
    No stored target or tolerance of steps 1-4 changed.
  - The general case 1 (d) budget at this state is 9.4e-8 for gamma1 and 1.4e-7 for nabla_ad. It is
    1e-8 + the propagation of 1e-10 errors of every returned P and s (6.5e-8 and 1.1e-7) + 2e-8 for the
    stencil truncation. The truncation is 5.5e-9 for gamma1 and 8.8e-9 for nabla_ad, measured on the
    exact Bessel-series equation of state at h and 2h, ratio 16. Both solutions show exactly these
    truncation errors.
- 6. Change: mutants.
  - New step 5 mutants: adiabat_at_constant_entropy_density, pairs_frozen_in_response and
    thermal_pressure_from_T_psi_jacobian.
  - cv_at_fixed_psi is rebuilt on the new step 5: the wrong c_V is returned and enters the exponents.
  - whole_task_classical_occupation is rebuilt: Boltzmann occupation throughout, including the response
    moments.
- 7. Regression tests.
  - Rejections (tools/crown_check.py):
    - cv_at_fixed_psi fails step 5 cases 0-5, 7 and 9;
    - adiabat_at_constant_entropy_density fails cases 0-5 and 7-9;
    - pairs_frozen_in_response fails cases 0, 2, 4, 5, 8 and 9;
    - thermal_pressure_from_T_psi_jacobian fails case 1 (6.5x the tolerance at (3.5e12, 2e7, 0.46)),
      case 5 (17x at the degenerate corner) and case 7 (43x at (2e12, 1e7, 0.5));
    - whole_task_classical_occupation fails general cases 0-3;
    - the unshipped finite-difference and isothermal controls fail as listed under "Why the naive
      formulas fail".
  - The three blind codes of the approved specification's pre-probe score 39/43, 40/43 and 40/43 on the
    final tests (tools/grade_candidate.py, end to end). Each fails only nabla_ad at the cold degenerate
    states, by 1.01 to 7.4 times the tolerance (step 5 cases 1, 5 and 7, general case 2).
  - The previous round's codes, written with the two hint clauses still in the text, score 43/43, 40/43
    and 39/43.
- 8. Rerun.
  - tools/crown_check.py, all modes (-j 3), with every case alone in a fresh process:
    - FORMAT OK;
    - REF: steps 1-5 pass 5/5, 7/7, 8/8, 9/9 and 10/10; general passes 4/4; solution.py passes every
      step file;
    - SECOND: every case passes;
    - every mutant fails at least one case of its target tests; the mutant mode repeated with
      OMP_NUM_THREADS=1 gives identical results and margins;
    - SHIFT +-1e-12: every case passes;
    - controls: all-None and all-zero pass no case;
    - SUMMARY ALL OK.
  - Both solutions, run three times in clean processes (once with one BLAS/OpenMP thread) on 20 states,
    gave bit-identical outputs.
  - Per-case runtime: at most 0.8 s for the reference, and at most 6.6 s for the second solution
    (general case 1, nine calls of about 0.7 s each).
  - Raw folder size: 334,129 bytes (du -sb, this note included), against a limit of 650 KB.
