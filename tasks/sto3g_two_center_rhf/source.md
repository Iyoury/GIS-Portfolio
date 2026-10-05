# Source

- A. Szabo and N. S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced
  Electronic Structure Theory" (Macmillan 1982; Dover reprint 1996). Chapter 3
  (Roothaan equations, the minimal-basis H2 and HeH+ examples, Section 3.5), Appendix A
  (integrals over s-type Gaussians and the F0 function) and Appendix B (the HeH+ SCF
  program and its output).
- W. J. Hehre, R. F. Stewart and J. A. Pople, "Self-consistent molecular-orbital
  methods. I. Use of Gaussian expansions of Slater-type atomic orbitals", J. Chem.
  Phys. 51, 2657 (1969). The STO-3G expansion.
- L. E. McMurchie and E. R. Davidson, "One- and two-electron integrals over Cartesian
  Gaussian functions", Journal of Computational Physics 26, 218-231 (1978).
- S. Obara and A. Saika, "Efficient recursive computation of molecular integrals over
  Cartesian Gaussian functions", Journal of Chemical Physics 84, 3963-3974 (1986).
- T. Helgaker, P. Jorgensen and J. Olsen, Molecular Electronic-Structure Theory (Wiley,
  2000), chapter 9 (integrals, Boys function) and chapter 10 (Hartree-Fock).
- S. F. Boys, "Electronic wave functions. I. A general method of calculation for the
  stationary states of any molecular system", Proc. R. Soc. London A 200, 542 (1950).
  Gaussian basis functions and the F0 function.
- CODATA 2018 recommended values for the hartree, the bohr radius, the atomic mass
  constant and the speed of light (given explicitly in the step 5 prompt).

Checks against the literature: for H2 (zeta = 1.24, R = 1.4 bohr) the reference code
gives S01 = 0.6593, H00 = -1.1204, H01 = -0.9584, (00|00) = 0.7746, (00|11) = 0.5697,
(01|01) = 0.2970, (00|01) = 0.4441 and E = -1.1167 hartree, matching the numbers printed
in Szabo and Ostlund, Section 3.5.2. For HeH+ it gives S01 = 0.4508, H00 = -2.6527,
H01 = -1.3472, H11 = -1.7318, matching their Appendix B output.

The combination of steps, the treatment of the dissociation limit and the parameter ranges are our own design.

Checks for steps 4 and 5: the reference full CI (configuration state functions in the RHF orbitals) agrees to about 1e-14 hartree with the lowest eigenvalue of the 4 x 4 non-orthogonal product-basis Hamiltonian and with the second solution (Loewdin orbitals). The analytic limit reproduces He = -2.643876 and H = -0.466582 hartree in this basis, and E_fci(R) - E_inf times R tends to the fragment-charge product (for example -1 for the ion pair). The sinc-DVR levels are converged to 2e-5 cm^-1 (grid step 0.02 vs 0.013 bohr, domain [0.35, 16] vs [0.22, 24]) over random valid parameters, and agree with Numerov shooting and with Chebyshev collocation to better than 1e-3 cm^-1.
The second solution is independent of the reference: shell-theorem radial quadrature
instead of closed-form Gaussian integrals, direct orbital minimization instead of the
self-consistent field, full CI in Loewdin orbitals, and for the levels a Chebyshev
interpolant of V in ln R with Chebyshev spectral collocation instead of the sinc DVR.

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| F0(t) (step 1) | 400-point Gauss-Legendre quadrature of the defining integral (cut at u = 10 / sqrt(t) when t > 100); exact limits F0(0) = 1, 1 - t/3 for tiny t, (1/2) sqrt(pi / t) for t = 1e4 | about 1e-13 relative; test tolerance 1e-11 relative (the stated requirement) |
| S, H (step 2) | shell theorem: every integral becomes a 1-D radial integral of one Gaussian averaged over spheres around the other centre, done with composite 16-point Gauss-Legendre; the nuclear attraction is the potential of a Gaussian charge cloud, also by 1-D quadrature (no F0, no erf) | about 1e-15; test tolerance 1e-10 absolute |
| (ij\|kl) (step 3) | same reduction: the potential of one Gaussian cloud averaged over spheres around the centre of the other cloud, then integrated over the radius | about 1e-15; test tolerance 1e-10 absolute |
| RHF energy (step 4) | direct minimization of 2 h + J over the single orbital-mixing angle with the test-side quadrature integrals (no Fock matrix, no iteration) | about 1e-15; test tolerance 1e-9 absolute |
| full-CI energy (step 4) | lowest eigenvalue of the two-electron Hamiltonian in the non-orthogonal product basis phi_i(1) phi_j(2) (generalized eigenproblem, overlap S (x) S) with the quadrature integrals | about 1e-15; test tolerance 1e-9 absolute |
| dissociation limit (step 5, general) | one-centre norm, core energy and (aa\|aa) by adaptive radial quadrature (scipy quad, the Coulomb term through the radial potential of the charge cloud); lowest of the three fragment arrangements; fragment-charge product checked for the ion-pair (-1) and fractional-charge (+0.25) cases | about 1e-12 hartree |
| vibrational levels (step 5, general) | full-CI curve from test-side closed-form integrals vectorized over R and the product-basis eigenproblem; renormalized Numerov shooting with node counting and bisection on steps 0.004 and 0.008 bohr over [0.25, 22], Richardson-extrapolated | below 1e-3 cm^-1; test tolerance 0.01 cm^-1 |
| sanity bands | Szabo and Ostlund values: S01, H00, H01, (00\|00), (00\|11), (01\|01), (00\|01) for H2; S01, H00, H01, H11 for HeH+; E_rhf = -1.1167 (H2) and -2.8607 (HeH+), E_fci = -1.1373 (H2); stretched H2 full CI near 2 x -0.466582 | bands of 1e-4 to 2e-3 only |

## Version 2 changes

Step 1: the stated and tested relative accuracy of F0 are the same, 1e-11 (quadrature target accurate
to ~1e-13); a 0-d result may be a 0-d array or a numpy float. Step 2: ZB = 0, zetaB = 0 and zetaB < 0
are tested as ValueError. Step 5: the bond length must lie within 1e-6 bohr of the minimum of the
test-side curve (slope / curvature), as stated; the isotope comparison allows twice that, since each
result carries its own 1e-6 error.

## Version 4 changes

Step 4 returns the full-CI and RHF energies. Step 5 computes the five lowest vibrational
levels from the exact R -> infinity limit, obtained from one-centre integrals and the
lowest fragment arrangement, instead of the energy at 60 bohr (which is wrong by
q_A q_B / 60 hartree for charged fragments). Step 5 is tested on an ion-pair limit
(-1/R tail) and a fractional-charge limit (+0.25/R tail) besides H2, D2 and HeH+; its
validation checks were cut to two (no five bound levels for He2 2+, a zero mass), with
no separate type or shape assertions.

## Version 5 changes

Step 5: the masses are restricted to 1 <= mass <= 10 u and validated (non-finite or
out-of-range masses raise ValueError, tested with NaN, inf and 10.5). The reference no
longer assumes a fixed radial domain: it widens [0.36, 16] bohr to [0.24, 24], [0.16, 36]
and [0.1, 54] until the five levels change by less than 1e-4 cm^-1. New converged cases at
the ends of the ranges: zeta = 0.8 and zeta = 3 with 10 u nuclei, and 1 u nuclei. The
output shape (5,) and the Python-float type of the step 4 energies are asserted; the
internal checks of the fragment-charge product were removed from the step 5 tests.

Version 6: the step 5 tests also check that a mass below 1 u (0.5 u) raises ValueError.

## Version 7 changes (difficulty increase)

Step 1 now returns F_0 .. F_nmax (n_max <= 16, 0 <= t <= 1e6, relative 1e-12); the tests
use the incomplete gamma function at 40 digits, and the upward recursion fails them.
Steps 2 and 3 call boys_function(0, t). New final step 6: singlet full-CI and RHF energies
in the 8-function basis STO-3G 1s + one Cartesian p shell per atom.
- Reference: McMurchie-Davidson integrals; full CI in the symmetric (singlet) product
  functions of Loewdin orbitals; RHF as the global minimum over closed-shell determinants
  (projected-gradient descent on the unit sphere from 48 starts, polished by a
  maximum-overlap SCF).
- Test oracle: Obara-Saika integrals with F_n from Kummer's function 1F1 (independent of
  steps 1-3), full CI as a generalized eigenproblem in the non-orthogonal symmetric product
  functions, RHF by BFGS from 40 random starts. Integrals agree with the reference to
  5e-15 and both energies to 2e-14 on 16 random points of the range.
- Second solution: Obara-Saika integrals, full CI in singlet configuration functions,
  RHF by steepest descent along great circles with Brent line searches.
- While building this step, two traps were found and are tested: the aufbau SCF converges
  to a solution 0.06-0.33 hartree too high for diffuse p shells, and the lowest eigenvalue
  of the full product space is a triplet 2e-7 hartree below the singlet at R = 10. The
  full-CI oracles of steps 4 and 5 were also restricted to singlet functions.

## Version 8 changes

Step 1: the stated and tested relative tolerance is 1e-11 (a 1e-12 tolerance left no room
for a result perturbed at the 1e-12 level). Step 5 runtime cut from about 270 s to about
70 s: the integrals of steps 2 and 3 are evaluated in one vectorized pass (identical results
to 1e-15), the SCF of step 4 stops at a density change of 1e-11 (energies identical to
4e-15), the sinc-DVR step is 0.03 bohr (levels within 3e-9 cm^-1 of the 0.02 bohr grid on
the hardest test cases) with the domain widened [0.36, 16] -> [0.27, 20] -> ..., and the
test oracle for the levels uses second-order finite differences on steps 0.004, 0.002 and
0.001 bohr with two Richardson extrapolations (tridiagonal eigenproblems) instead of
Numerov shooting in pure Python.

## Version 9 changes

Step 5 also tests HeH+ with the centres swapped (ZA = 1, ZB = 2), whose limit has both
electrons on atom B. Step 6 wording: the spatial symmetry of the full-CI ground state is
stated as a consequence of the singlet spin function.

Version 10 (reviewer's correction):
- Step 4 RHF reference replaced. The single self-consistent-field run from the core-Hamiltonian guess
  missed the global minimum at 115 of 960 points of the stated domain (charges 1-3, exponents
  0.5-3, R 0.3-30). It settled on the higher closed-shell branch of stretched asymmetric molecules,
  up to 2.6 hartree too high.
  - The reference now writes every normalized orbital as X (cos t, sin t), with X = S^(-1/2), scans
    the whole period and refines every local minimum of the scan.
  - It agrees with the test oracle and the second solution to 1e-12 on all 960 points.
- The old solver is kept as the mutant step_4_single_start_scf.
- New step-4 test case 5: three stretched asymmetric molecules where single-start SCF lands 2.07 to
  2.61 hartree too high.
- The prompt now says explicitly that E_rhf is the global minimum and that a second, higher local
  minimum exists.
- Test oracles:
  - The RHF oracle refines every local minimum of a 2001-point scan, not only the best grid point.
  - The nuclear-attraction quadrature of the step 2-4 tests (and of the second solution) now
    integrates only over [0, min(e, 9 / sqrt(p))]. The fixed 80-point rule on [0, e] missed the
    narrow peak of tight functions far from a nucleus, giving errors of 6e-9 at R = 30, zeta = 3.
    The reference was right there, as checked against the closed erf form in mpmath.
- Test metadata: every test case block now holds its own assert statements. Helper calls are
  asserted, and the error cases use an explicit raised flag. The step-4 n_test_cases is 6.

Version 11 (content-check fixes):
- Step 5 charge range 1 <= Z <= 2 (was 1 to 3). With a charge above 2 the separated fragments in
  this basis repel each other and no curve holds five bound levels. A scan over 36 exponent pairs
  at ZA = 3, ZB = 1 with 10 u nuclei found none, so an endpoint Z = 3 case with five levels does
  not exist. The upper endpoint Z = 2 is exercised on both centres (HeH+ and its swap). Charges
  outside [1, 2] or not finite now raise ValueError, stated and tested.
- Step 1 rejects an array with one bad element (NaN, negative or above 1e6).

## Version 12

- Step 5 reference rewritten so that no outer wall limits the states. Once the cross integrals have
  died out (Gaussian decay; checked to 1e-12 hartree over 2 bohr) the full-CI curve is exactly
  min_j (E_j - E_inf + c_j / R) over the three separated arrangements. The radial equation is solved
  with the computed curve on [r0, a] by a Lagrange-Legendre mesh with a free end at a (R-matrix
  u(a)/u'(a) = (1/2mu) sum_n u_n(a)^2/(E_n - E)) and with the analytic tail on [a, infinity) by outward
  integration; levels are located by Sturm counting (zeros of the solution regular at r0) and
  bisection to 1e-11 hartree. No R range is fixed in advance: the outer region extends as far as each
  energy requires, and the inner wall, tail start and mesh are refined until the five levels change by
  less than 1e-4 cm^-1. With an attractive tail (infinitely many levels) a level above -1e-12 hartree is
  returned as -0.5e-12 hartree. Checked on hydrogen (-1/r, exact levels to 1e-12 with a = 15 bohr while
  the 5th state reaches 50 bohr).
- New test case 6: (ZA, ZB, zetaA, zetaB) = (1, 2, 1.24, 1.592), masses 1 u, neutral fragments: exactly
  five levels, the fifth at -0.2689 cm^-1, reaching out to about 100 bohr. Target: the independent
  finite-difference/Richardson solver of the tests with the exact separated-fragment tail beyond 30 bohr
  (asserted equal to the curve there to 1e-12 hartree) and outer walls at 150 and 250 bohr, which agree
  to 1e-3 cm^-1. The v11 reference (walls up to 36 bohr) gives -0.1988 cm^-1 and fails.
- Second solution step 5: sinc DVR with the exact tail beyond the same kind of matching point, the grid
  length doubled and the step refined until converged (no fixed wall); it agrees with the reference to
  about 1e-6 cm^-1.
- Prompt: the far reach of near-threshold levels is stated.
