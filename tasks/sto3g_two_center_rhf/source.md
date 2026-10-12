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

Checks for steps 4 and 5: the reference full CI (configuration state functions in the RHF orbitals) is
the lowest singlet. It agrees to about 1e-14 hartree with the lowest eigenvalue of the non-orthogonal
product-basis Hamiltonian restricted to the three symmetric (singlet) product functions, and with the
second solution (version 15: the full product space of its RHF orbitals with a spin penalty on the
triplet, within 9.1e-13 on 540 points; version 14: Loewdin orbitals). It is not always the lowest
eigenvalue of the full 4 x 4 product space: for diffuse exponents at stretched R the triplet lies below the singlet in this basis (by
6.3e-3 hartree at ZA = ZB = 2, zeta = 0.8, R = 4 bohr; see version 14). The analytic limit reproduces
He = -2.643876 and H = -0.466582 hartree in this basis, and E_fci(R) - E_inf times R tends to the
fragment-charge product (for example -1 for the ion pair).

Methods of the current reference solution: boys_function sums a series for the highest order and
recurs downward for t < n_max + 25, and uses the erf form with the upward recursion above that; steps 2
and 3 use the closed-form s-Gaussian integrals of the Gaussian product theorem; fci_energy finds RHF
from a 720-point scan of the orbital angle in the symmetric orthogonalization S^(-1/2), every local
minimum refined by a bounded Brent search (since version 10), then diagonalizes the 3 x 3 singlet
configuration matrix in the RHF orbitals; vibrational_levels evaluates the curve with the step 2 and 3
integrals and the same singlet CI in Loewdin orbitals, takes the limit from one-centre integrals, and
solves the radial equation with a Lagrange-Legendre mesh on [r0, a] matched through the R-matrix to the
analytic tail integrated outward with solve_ivp, with Sturm node counting and bisection (since version
12); polarized_energies uses McMurchie-Davidson integrals, RHF by projected-gradient descent from 48
starts polished by a maximum-overlap SCF, and full CI in the symmetric product functions of Loewdin
orbitals.

The second solution (version 16) uses, function by function, a method different from the reference and
from the test oracles: composite Gauss-Legendre quadrature of the defining integral for F_n (since
version 13); momentum-space (Fourier) radial integrals for steps 2 and 3 (overlap and kinetic energy from
the Fourier transforms of the two primitives and Parseval's theorem, attraction and repulsion from the
Fourier transform of the Gaussian charge cloud and the Coulomb kernel 4 pi / k^2, the angular average
giving j0(k d), the k integral by composite Gauss-Legendre; no erf, no Boys function); for step 4 RHF
from the exact trigonometric-polynomial form of E(theta) (harmonics 0, 2, 4, coefficients from 8
samples) and the roots of its derivative (a quartic in exp(2 i theta)), and full CI in the whole
4-dimensional product space of the RHF orbitals with a spin penalty 20 (1 - P12) / 2 hartree on the
triplet; for step 5 the one-centre integrals of the limit from the same momentum-space integrals, its
own fci_energy sampled on 200 Chebyshev nodes in ln R on [0.3 bohr, r_c] and interpolated, the exact
separated-fragment tail beyond r_c, and a Colbert-Miller sinc DVR (at least 40 points) whose length is
doubled and whose step is refined until the five levels below the limit change by less than
1e-4 cm^-1 (since version 12); with fewer than five eigenvalues below the limit the box is doubled until
it holds every bound level (repulsive tail: the WKB attenuation integral of the tail beyond its last
non-positive point reaches 20) or up to 1000 bohr (neutral tail), and ValueError is raised if fewer than
five remain after a step refinement (version 16); for step 6 Gaussian-transform (Rys-type) integrals,
1/r = (2/sqrt(pi)) int exp(-s^2 r^2) ds with the x, y, z factors done exactly by 3-point Gauss-Hermite
rules and the transform variable by Gauss-Legendre, RHF by level-shifted Roothaan-Hall SCF (Fock
F = H + 2J - K, shift 1 hartree on the virtual space, no DIIS) from 64 fixed guess orbitals (every basis
function and every sum and difference of two), keeping the lowest converged energy, and full CI by
direct CI (sigma vectors from the integrals, no Hamiltonian matrix) with Davidson iteration in the
canonical RHF orbitals, restricted to symmetric coefficient matrices (version 16). Full CI is the same
eigenvalue problem in every method (it does not depend on the orbitals), so for the full-CI energies the
independence lies in the integrals, the orbitals and the eigensolver, not in the space.

Superseded descriptions, kept for the record (earlier versions of this file): until version 14 the
second solution reused the test-oracle code for the integrals of steps 2-3 (shell-theorem quadrature)
and step 6 (Obara-Saika), found the step 4 RHF with the scan-and-refine formulation of the step 4 oracle
and did both full CIs in Loewdin orbitals restricted to singlet functions (as the step 6 reference);
the step 5 reference was a sinc DVR on a widening fixed domain until version 11; the second solution
then used Chebyshev spectral collocation for the levels, and direct orbital minimization against the
self-consistent field that the step 4 reference used until version 9; the step 1 target was a
400-point Gauss-Legendre quadrature of F0 until version 6, and the levels target was renormalized
Numerov shooting until version 7; in version 15 the second solution found the step 6 RHF by steepest
descent along great circles from the eigenvectors of h and 40 random starts (the strategy of the
reference with another step rule) and did the step 6 full CI in the 64-dimensional product space of its
RHF orbitals with a spin penalty, and its sinc DVR had no explicit bound-state count. None of these is
used now.

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| F_n(t), n <= 16 (step 1) | F_n(t) = Gamma(n + 1/2) P(n + 1/2, t) / (2 t^(n + 1/2)) with the regularized lower incomplete gamma function P from mpmath at 40 digits; F_n(0) = 1/(2n + 1) exactly | far below 1e-13 relative; test tolerance 1e-11 relative (the stated requirement) |
| S, H (step 2) | shell theorem: every integral becomes a 1-D radial integral of one Gaussian averaged over spheres around the other centre, done with composite 16-point Gauss-Legendre; the nuclear attraction is the potential of a Gaussian charge cloud, its inner part by an 80-point rule on [0, min(e, 9 / sqrt(p))] and its outer part exact (no F0, no erf) | within 9e-16 of 50-digit mpmath closed forms at R = 0.02 and 100 bohr with zeta = 0.5 and 3 (version 14); test tolerance 1e-10 absolute |
| (ij\|kl) (step 3) | closed-form potential (erf) of one Gaussian cloud averaged over spheres around the centre of the other cloud, then a composite 16-point Gauss-Legendre radial integral; at R = 100 bohr also (00\|11) = N^2 / R for non-overlapping clouds | within 2.5e-15 of 50-digit mpmath at the same range ends; test tolerance 1e-10 absolute |
| RHF energy (step 4) | quadrature integrals of steps 2-3; 2 c.H.c + (cc\|cc) over the normalized orbital angle, every local minimum of a 2001-point scan of the whole period refined by a bounded Brent search (no Fock matrix, no iteration) | within 1.5e-14 of 50-digit mpmath at the range ends; test tolerance 1e-9 absolute |
| full-CI energy (step 4) | lowest eigenvalue of the generalized eigenproblem in the three symmetric (singlet) product functions of the non-orthogonal basis, overlap S (x) S, with the quadrature integrals; case 6 also evaluates the triplet function and asserts that it lies more than 1e-3 hartree below the singlet target | within 9e-12 of 50-digit mpmath at the checked points with R from 0.1 to 100 bohr (worst at R = 0.1, zeta = 0.5, where the two functions are nearly linearly dependent) and of the reference on a 432-point grid of the step 4 range; test tolerance 1e-9 absolute |
| dissociation limit (step 5) | one-centre norm, core energy and (aa\|aa) by adaptive radial quadrature (scipy quad, the Coulomb term through the radial potential of the charge cloud); lowest of the three fragment arrangements | about 1e-12 hartree |
| vibrational levels (step 5) | full-CI curve from test-side closed-form s-Gaussian integrals vectorized over R (F0 from erf with its small-argument series) and the singlet generalized eigenproblem; second-order finite differences with Dirichlet walls on steps 0.004, 0.002 and 0.001 bohr over [0.25, 22] bohr, extrapolated twice by Richardson (tridiagonal eigenproblems); for the near-threshold case 6 the exact separated-fragment tail beyond 30 bohr (asserted equal to the curve there to 1e-12 hartree) and outer walls at 150 and 250 bohr, which agree to 1e-3 cm^-1 | below 1e-3 cm^-1; test tolerance 0.01 cm^-1 |
| E_fci, E_rhf with p shells (step 6, general) | Obara-Saika overlap, nuclear-attraction and repulsion integrals, kinetic energy as (1/2) sum_i <d_i a\|d_i b>, F_m from Kummer's function 1F1 (scipy hyp1f1); full CI as the generalized eigenproblem in the non-orthogonal symmetric product functions; RHF by BFGS from 40 random starts with the fixed seed 2024 | integrals agreed with the reference to 5e-15 and energies to 2e-14 on 16 random points (version 7); test tolerance 1e-9 absolute |
| 1s-only bounds (step 6 case 0) | the step 6 oracle restricted to the two 1s functions (generalized-eigenproblem full CI, BFGS RHF); the candidate must lie below it by more than 1e-3 (full CI) and 1e-4 (RHF), the actual lowerings being 0.0130 and 0.0036 hartree | variational principle |
| F_n by other routes (step 1 cases 5-6, since version 17) | the defining integral by mpmath tanh-sinh quadrature at 30 digits (t = 0.5, 8, 60); the large-t closed form (2n - 1)!! sqrt(pi) / (2^(n+1) t^(n+1/2)) at t = 1000, 4.2e4 and the inclusive end 1e6; the Taylor series sum_k (-t)^k / (k! (2n + 2k + 1)) at 40 digits (t = 1e-3, 0.02, 0.3); F_0 = sqrt(pi / t) erf(sqrt t) / 2 at 40 digits; the identity (2n + 1) F_n - 2t F_(n+1) = exp(-t) between computed outputs (t = 1.5, 6) | the quadrature and the incomplete-gamma form agree to double precision; the closed form differs from it by Q(n + 1/2, t) < 1e-380 for t >= 1000 (2.2e-16 after rounding); tolerance 1e-11 relative, the identity 2e-11 times the sum of the two term magnitudes |
| one-centre and separated-atom integrals (step 2 case 6, step 3 case 5, since version 17) | closed forms of concentric primitive Gaussians: norm sum w_ij (pi / p)^(3/2) (S[0, 0] independent of zeta), kinetic sum w_ij 3 a_i a_j / p (pi / p)^(3/2), attraction sum w_ij 2 pi / p, repulsion sum w_ij w_kl 2 pi^(5/2) / (p q sqrt(p + q)); at R = 100 bohr the shell theorem (H[0, 0] = T_A - ZA U_A - ZB S_AA / R, (00\|11) = S_AA S_BB / R) and vanishing overlap densities; exact relations between computed outputs: lengths divided by lam = 2 with charges multiplied by lam (S unchanged, H times lam^2, (ij\|kl) times lam) and the mirror exchange of the atoms | every cross term below 1e-50 at R = 100 bohr (zeta >= 0.8 there); the reference agrees with the closed forms within 9e-16; tolerances 1e-10 absolute, the relations 2e-10, (1 + lam^2) 1e-10 and (1 + lam) 1e-10 (sum of the two output bounds) |
| separated-atom energies (step 4 case 8, since version 17) | R = 100 bohr: the three singlet configurations decouple, so E_fci is the lowest of 2 e_A + J_A + (ZA - 2) ZB / R, 2 e_B + J_B + (ZB - 2) ZA / R and e_A + e_B + (ZA - 1)(ZB - 1) / R, and E_rhf is the minimum on [0, 1] of the convex quadratic 2 (1 - x) h_A + 2 x h_B + (1 - x)^2 J_A + x^2 J_B + 2 x (1 - x) / R + ZA ZB / R (x the weight of B, h = e - Z_other / R), from the one-centre closed forms of the normalized functions; H2, HeH+, the ion pair (1, 1, 1.24, 2.69) and (1.5, 1.5, 2.0, 1.24) | the reference agrees within 1.8e-15 hartree; tolerance 1e-9 absolute |
| reduced-mass invariance (step 5 case 8, since version 17) | HeH+ with masses (2, 2) u against the finite-difference target; (1.5, 3) and (4, 4/3) u, the same reduced mass 1 u, against the first | reference within 6.1e-7 cm^-1 of the target and bitwise equal for the three pairs; relation tolerance 0.02 cm^-1 (two outputs, each within 0.01) |
| sanity bands | Szabo and Ostlund values: S01, H00, H01, (00\|00), (00\|11), (01\|01), (00\|01) for H2; S01, H00, H01, H11 for HeH+; E_rhf = -1.1167 (H2) and -2.8607 (HeH+), E_fci = -1.1373 (H2), also checked against the 1s-only oracle values of step 6 case 0; stretched H2 full CI near 2 x -0.466582 | bands of 1e-4 to 2e-3 only |

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
  (Version 16 note: the general claim "with a charge above 2 ... no curve holds five bound levels" was
  found inaccurate; it had been checked only at ZA = 3, ZB = 1. With the reference and only the charge
  cap raised, ZB = 1, zetaB = 1.24 and 10 u nuclei, five deep levels exist at ZA = 2.01, zetaA = 2.0925
  (-48094.9 to -41013.1 cm^-1), ZA = 2.05 (-35366.5 to -28725.4), ZA = 2.1 (-20491.5 to -14523.9),
  ZA = 2.1 with zetaA = 2.3 (-49656.0 to -41869.5) and ZA = 2.3 with zetaA = 2.5 (-9873.0 to
  -2993.7); at ZA = 2.5 (zetaA = 2.0925) and ZA = 3 (zetaA = 2.7) there are fewer than five. The
  domain limit 1 <= Z <= 2 is kept as a design choice; the false reason was removed from the step 5
  prompt in version 16.)
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

## Version 13 (grading_fix: authoring-guide conformance)

No science, reference algorithm, target, stated tolerance or input domain was changed. Each item gives
the finding, its cause, the change, the regression test and what was rerun.

- Test-case format and self-containment.
  - Finding: every test file kept its imports, oracle helpers and constants in a shared preamble before
    case 0, the markers carried descriptions ("# --- test case 0: ... ---"), and step 5 case 4 compared
    the D2 levels with the H2 levels `lev` computed in case 0, so the case failed when run alone.
  - Cause: the files were written to be executed as a whole.
  - Change: markers are exactly "# --- test case N ---" (N = 0..n-1), the descriptions are comment
    lines under them, and every case repeats the imports, helpers and constants it uses (split with
    tools/selfcontain.py); the target-provenance notes stay as leading comments. Step 5 case 4 now
    computes the H2 levels itself through the same check against the oracle before the isotope
    comparison. The general cases assert their helper call like the step cases.
  - Regression test: every case run alone in a fresh process (crown_check ref, second, mut, shift,
    controls).
- Step 5 n_test_cases.
  - Finding: problem.yaml said 7, tests/step_5.py has 8 cases (0..7).
  - Cause: case 6 (near-threshold fifth level, version 12) was added without updating the count.
  - Change: n_test_cases = 8. Regression test: crown_check format.
- Zero relative tolerance.
  - Finding: steps 2 and 3 compared with np.allclose(a, b, rtol=0.0, atol=1e-10).
  - Cause: written to express a pure absolute tolerance.
  - Change: np.max(np.abs(a - b)) <= 1e-10 (same meaning, stated 1e-10 unchanged). The symmetry checks
    S = S^T, H = H^T and the permutation symmetries of (ij|kl) compare two computed entries, each allowed
    1e-10 from a symmetric target, so they now allow 2e-10.
  - Regression test: steps 2 and 3, all cases, with the ±1e-12 shift.
- Step 4 variational bound with a slack below the stated accuracy.
  - Finding: E_fci <= E_rhf + 1e-12 (cases 4 and 5) compared two outputs that are each allowed an error
    of 1e-9 with a slack 1000 times smaller than that accuracy.
  - Cause: slack chosen before the rule for comparisons between computed outputs.
  - Change: slack 2e-9 (twice the stated single-output tolerance). In these cases E_rhf - E_fci is 0.15 to
    0.34 hartree (reference), so no test outcome changes. Regression test: step 4 cases 4-5.
- Undeclared imports in second_solution.py.
  - Finding: boys_function imported scipy.special.gammainc and gammaln, and the step 6 part imported
    functools.lru_cache; neither is in required_dependencies.
  - Cause: added when the second solution was extended in version 7.
  - Change: the second-solution boys_function is now a composite Gauss-Legendre quadrature of the
    defining integral (8 panels of the 20-point rule over [0, min(1, 13 / sqrt(t))]); against 40-digit
    mpmath its worst relative error is 2.3e-15 over 532 values of t in [0, 1e6] and all n <= 16. This
    also makes it independent of the incomplete-gamma identity used by the step 1 test oracle. lru_cache
    is replaced by a small dictionary memo with the same behaviour. Declaring the imports instead would
    have shown the agent a ready-made route to the Boys functions of step 1.
  - Regression test: SECOND on steps 1 and 6 and on the general tests.
- Timing: no clock read, time budget or per-call time sentence was present in tests, prompts,
  docstrings, solutions or mutants (checked by search); nothing removed.
- Metadata: subfield, tags, expert_time_estimate_hours, relevant_experience (placeholder for the author),
  author, affiliation, difficulty_explanation, solution_explanation and verification_explanation added;
  edit_label grading_fix.
- Rerun after all changes: tools/crown_check.py (all modes, every case alone in a fresh process):
  FORMAT OK; REF steps 1-6 5/5, 5/5, 4/4, 6/6, 8/8, 4/4 and general 3/3 (solution.py also passes every
  step file); SECOND steps 1-6 and general all pass; mutants fail upward_recursion 3/5,
  zeta_not_squared 3/5, wrong_boys_argument 3/4, full_exchange 6/6, orthonormal_atomic_basis 6/6,
  single_start_scf 1/6 (case 5), limit_at_finite_distance 3/8 (cases 2, 3, 6), total_mass 7/8,
  harmonic_levels 7/8, kinetic_without_angular_momentum 4/4, sigma_p_only 4/4, aufbau_scf 1/4 (case 2),
  whole_task_missing_hermite_sign 3/3; SHIFT ±1e-12 all pass; all-None and all-zero controls pass
  0 cases. Repeat runs in clean processes: solution.py gives bitwise identical outputs for all six
  functions in two runs and with one BLAS/OpenMP thread; second_solution.py is bitwise identical in two
  runs, and with one thread its vibrational levels move by at most 2e-9 cm^-1 (threaded eigensolver
  reductions), everything else identical. The second solution agrees with the reference to 2e-15
  (relative, F_n), 1e-15 (integrals), 1.2e-14 hartree (energies) and 1.5e-6 cm^-1 (levels) on those
  inputs.

## Version 14 (grading_fix: authoring-guide conformance)

Fixes for the blocking findings of an independent verification of version 13. No reference algorithm,
existing target or stated tolerance was changed. Each item gives the finding, its cause, the change,
the regression test and what was rerun.

- Step 4 did not say that E_fci is the singlet.
  - Finding: step 4 defined E_fci as the lowest eigenvalue of the electronic Hamiltonian in the space of
    all two-electron states and called it the ground-state energy. Its step_background said the triplet
    lies higher. The reference and every oracle compute the lowest singlet, but in this basis the triplet
    lies below it for diffuse exponents at stretched R. Rechecked here with the reference integrals, the
    singlet minus the triplet is 8.36e-3 hartree at (ZA, ZB, zetaA, zetaB, R) = (2, 2, 0.5, 0.5, 4),
    6.27e-3 at (2, 2, 0.8, 0.8, 4) and 3.19e-3 at (1.5, 1.5, 0.8, 0.8, 5). At every step-4 test input the
    singlet was also the lowest eigenvalue of the full product space (to 1e-15), so a solver returning
    that eigenvalue passed all six cases. The check paragraph of this file also claimed agreement with the
    lowest 4 x 4 product-basis eigenvalue.
  - Cause: the singlet wording added to step 6 in versions 7 and 9 was not carried back to step 4, and
    every step-4 input lay where the singlet is lowest.
  - Change: the step 4 prompt and header now ask for the lowest singlet eigenvalue (spatial two-electron
    function symmetric under exchange of the electrons), worded as in step 6. problem_description_main
    says singlet ground state. The step_background claim is corrected (in this basis the triplet can fall
    below the lowest singlet for diffuse exponents at stretched R), and so are the step 4 contract,
    background.md, difficulty_explanation and the check paragraph above. New step-4 test case 6 checks
    (2, 2, 0.8, 0.8, 4) and (1.5, 1.5, 0.8, 0.8, 5) against the existing singlet oracle, and asserts
    that the test-side triplet lies more than 1e-3 hartree below the singlet target there. New mutant
    step_4_lowest_product_eigenvalue takes the lowest eigenvalue of the 4 x 4 product space in the RHF
    orbitals.
  - Regression test: step 4 case 6. The reference and second solution pass it. The new mutant fails it
    (off by 6.3e-3 hartree at the first input) and passes the other seven cases, which shows that the
    earlier cases could not detect the mistake.
- The step 2-4 input domain was not stated.
  - Finding: the prompts and headers of steps 2-4 promised every positive input, but the reference raises
    ValueError once p |P - C|^2 exceeds 1e6. sto3g_one_electron(1, 1, 1.24, 1.24, 400) and
    fci_energy(1, 1, 3, 3, 160) fail (reproduced). The contracts gave other ranges (R <= 10 bohr for
    steps 2-3, R <= 30 for step 4) that did not cover step 4 case 5 (R = 20) or the distances at which
    the step 5 reference evaluates the integrals. The tests covered only a middle range.
  - Cause: the validated range was written only in the contracts, never in the visible text.
  - Change: the visible text now states the ranges, and the contracts match.
    - Steps 2 and 3: 1 <= ZA, ZB <= 3, 0.5 <= zetaA, zetaB <= 3 and 0.02 <= R <= 100 bohr. The ValueError
      for a non-positive input is kept. 0.02 bohr is the code floor of the inner wall in the step 5
      reference. Instrumented runs at the extremes of the step 5 range evaluated the integrals on
      [0.1, 22] bohr. Over this range the Boys argument stays below 40.1 x 1e4 = 4.0e5 < 1e6.
    - Step 4: the same charges and exponents, 0.1 <= R <= 100 bohr. Below about 0.1 bohr the two functions
      become nearly linearly dependent, and double-precision rounding of the integrals is amplified in
      E_fci. A 50-digit mpmath implementation of the closed-form integrals, the singlet CI and RHF (written
      for this check) gives, at R = 0.02 and zeta = 0.5, errors of 2e-9 for the reference, 8e-8 for the
      test oracle and 2.3e-7 for the second solution. At R = 0.1 the errors are 6.5e-13 for the reference,
      9e-12 for the oracle and 1.4e-10 for the second solution. With the reference integrals, four other
      formulations of the singlet CI (generalized eigh, inverse-overlap eig, Cholesky, canonical
      orthogonalization) all stay below 4.2e-11, and up to 3.2e-10 at R = 0.05.
  - Regression tests:
    - Step 2 case 5: (3, 3, 0.5, 0.5, 0.02), (1, 3, 3, 0.5, 0.02), (3, 1, 3, 0.5, 100) and
      (1, 1, 0.5, 0.5, 100).
    - Step 3 case 4: (0.5, 0.5, 0.02), (3, 0.5, 0.02), (3, 3, 100) and (0.5, 3, 100). At 100 bohr it also
      checks the shell-theorem value (00|11) = N^2 / R.
    - Step 4 case 7: (3, 3, 0.5, 0.5, 0.1), (1, 3, 3, 0.5, 0.1) and (3, 1, 3, 0.5, 100).
    - Before these cases were added, the oracles were checked at the range ends against the 50-digit
      values: integrals within 2.5e-15, energies within 9e-12 (full CI) and 1.5e-14 (RHF). On a 432-point
      grid of the step 4 range (Z = 1, 2, 3 x 1, 3; zeta = 0.5, 1.2, 3; R = 0.1 to 100 bohr) the
      reference agrees with the oracle to 9.1e-12 (full CI) and 5.7e-14 (RHF), and the second solution
      to 1.4e-10 and 1.4e-14.
    - n_test_cases: step 2 5 -> 6, step 3 4 -> 5, step 4 6 -> 8.
- Undeclared dependencies of step 5.
  - Finding: the step 5 reference helper _s5_fci calls sto3g_one_electron and sto3g_two_electron, but the
    step 5 prompt named only fci_energy and step_dependencies was [4].
  - Cause: when the curve was rebuilt from the integrals to save time (version 8), the declared
    dependencies were not updated.
  - Change: the step 5 prompt adds "you may also call sto3g_one_electron (step 2) and sto3g_two_electron
    (step 3) to evaluate the curve", step_dependencies is [2, 3, 4], and the helper comment says so.
    Calling fci_energy instead was not chosen: its RHF angle scan at every R would multiply the step 5
    runtime. In the same way, step 6 test case 0 called fci_energy (step 4) for its 1s-only comparison,
    although step 6 declares only step 1. It now uses the Szabo-Ostlund 1s-only values (full CI -1.1373,
    RHF -1.1167 hartree) with the same margins (1e-3, 1e-4); the actual gaps are 0.0130 and 0.0036
    hartree. Each test file now calls only its own step's function.
  - Regression test: crown_check REF and SECOND on steps 5 and 6 (step 5 code unchanged apart from the
    comment).
- Superseded methods in this file.
  - Finding: the target table and the check and second-solution paragraphs described methods no longer
    used. They listed a 400-point Gauss-Legendre F0 target, a Numerov levels target labelled for the
    general tests, sinc-DVR reference levels, and Chebyshev collocation and an SCF comparison for the
    second solution. The table had no row for the step 6 / general oracle.
  - Cause: version notes were appended without revising the summary sections.
  - Change: the table and paragraphs now describe the current oracles, reference and second solution
    (checked against the test headers and the code). The old descriptions are kept in an explicitly
    superseded paragraph.
  - Regression test: none (documentation).
- Metadata: difficulty_explanation (singlet selection in step 4), solution_explanation (the step 5 curve
  from the step 2-3 integrals) and verification_explanation (range-end checks, 13 step mutants) were
  updated. Docstrings of steps 2-4 were made identical to the new headers in solution.py, solution/,
  steps/, second_solution.py and the mutants.
- Rerun after all changes: tools/crown_check.py, all modes, every case alone in a fresh process.
  - FORMAT OK.
  - REF: steps 1-6 pass 5/5, 6/6, 5/5, 8/8, 8/8 and 4/4; general 3/3.
  - SECOND: all cases of steps 1-6 and general pass.
  - Mutants: each fails at least one case.
    - upward_recursion 3/5; zeta_not_squared 4/6 (cases 0, 1, 3, 5); wrong_boys_argument 4/5 (0, 1, 2, 4).
    - full_exchange 8/8; orthonormal_atomic_basis 8/8; single_start_scf 1/8 (case 5);
      lowest_product_eigenvalue 1/8 (case 6).
    - limit_at_finite_distance 3/8; total_mass 7/8; harmonic_levels 7/8.
    - kinetic_without_angular_momentum 4/4; sigma_p_only 4/4; aufbau_scf 1/4.
    - whole_task_missing_hermite_sign 3/3.
  - SHIFT ±1e-12: all pass.
  - All-None and all-zero controls pass 0 cases.
  - Repeat runs in clean processes (boundary inputs of steps 2-4 included): solution.py gives bitwise
    identical outputs in two runs and with one BLAS/OpenMP thread. second_solution.py is identical in two
    runs; with one thread only its vibrational levels move, by 4e-10 cm^-1.

## Version 15 (grading_fix: authoring-guide conformance, pass 2)

Fixes for the remaining findings of the round-1 independent verification and for the pass-2 rules. No
reference algorithm, target, stated tolerance or input domain was changed. Each item gives the finding,
its cause, the change, the regression test and what was rerun. The round-1 findings on the step 4
singlet, the step 2-4 input ranges, the undeclared step 5 dependencies and the superseded method
descriptions were already fixed in version 14 (see above) and were rechecked here.

- Second solution copied the test oracles (rule D).
  - Finding: in second_solution.py, sto3g_one_electron and sto3g_two_electron were the shell-theorem
    quadrature of the step 2-4 test oracles and the step 6 integrals (_g6_*) the Obara-Saika code of
    the step 6 / general oracle, line for line apart from names; the step 4 RHF used the oracle's
    formulation (v / sqrt(v S v), periodic scan, every local minimum refined by Brent), and the step 6
    full CI was the reference's construction (Loewdin orbitals restricted to symmetric product
    functions). Passing those tests added no independent evidence there.
  - Cause: the second solution of versions 7 and 10 was written from the oracle code.
  - Change: new methods, different from both the reference and the oracles.
    - Steps 2-3 (and the one-centre integrals of the step 5 limit): momentum space. Overlap and kinetic
      energy from the Fourier transforms of the two primitives and Parseval's theorem; attraction and
      repulsion from the Fourier transform of the Gaussian charge cloud and the Coulomb kernel
      4 pi / k^2; the angular average gives j0(k d), and the radial k integral is done by composite
      16-point Gauss-Legendre on panels no wider than half an oscillation period (no erf, no F0).
    - Step 4 RHF: E(theta) = 2 c.H.c + (cc|cc) with c = S^(-1/2) (cos theta, sin theta) is a
      trigonometric polynomial with harmonics 0, 2 and 4; its five coefficients come exactly from 8
      samples, its stationary points are the arguments of the roots of a quartic in exp(2 i theta)
      (numpy roots, then Newton steps), and the lowest value is kept. No scan, no Brent search.
    - Step 4 and step 6 full CI: the whole product space of the RHF orbitals (occupied orbital plus its
      orthogonal complement; 4 and 64 functions) with the spin penalty 20 (1 - P12) / 2 hartree, P12 the
      exchange of the electrons, which lifts every spatially antisymmetric (triplet) state, instead of
      a restriction to singlet functions. A first version in Loewdin orbitals erred by 1.3e-10 hartree
      at R = 0.1 bohr, zeta = 0.5 (rounding amplified by the nearly singular overlap), so the RHF-orbital
      basis is used.
    - Step 6 integrals: Gaussian transform 1/r = (2 / sqrt(pi)) int exp(-s^2 r^2) ds (Rys-type). With
      u^2 = s^2 / (rho + s^2) every attraction and repulsion integral is an integral over u in [0, 1] of
      exp(-T u^2) times a product of x, y and z factors, each the mean of a polynomial under a one- or
      two-dimensional Gaussian; the means are exact with 3-point Gauss-Hermite rules and the u integral
      uses 40-point Gauss-Legendre on [0, min(1, sqrt(46 / T))]. Overlap and kinetic integrals are
      one-dimensional Gauss-Hermite quadratures. No Boys function, no Hermite expansion, no recursion.
    - Unchanged: Gauss-Legendre quadrature of the Boys integral (step 1), the Chebyshev interpolant
      in ln R and sinc DVR of step 5, and RHF by great-circle steepest descent in step 6.
    - (Version 16 note: "different from both the reference and the oracles" did not hold for step 6.
      The great-circle RHF used the reference's strategy, multi-start first-order descent on the unit
      sphere from the eigenvectors of h and 40 random vectors, with only the step rule changed, and the
      step 6 full CI built the reference's 64 x 64 product-space Hamiltonian, with a spin penalty instead
      of the symmetric projection. The sinc DVR also passed the He2 2+ error case only through a numpy
      broadcasting ValueError. All three are fixed in version 16.)
  - Regression tests and checks (single-threaded BLAS):
    - Integrals of steps 2-3 against the reference on 288 points (Z = (1,1), (3,3), (1,3), (2,1);
      zetaA = 0.5, 1.24, 3; zetaB = 0.5, 2, 3; R = 0.02 to 100 bohr): within 7.1e-15 (one-electron), and
      within 4.4e-16 for the repulsion integrals on the 72 of them with Z = (1, 1) (they do not depend
      on Z). Against a 50-digit mpmath evaluation of the closed forms (written for this
      check) at R = 0.02, 7 and 100 bohr with zeta = 0.5 and 3: within 3.6e-15.
    - fci_energy against the reference on 540 points (ZA = 1, 2, 3; ZB = 1, 3; zeta = 0.5, 1.2, 3;
      R = 0.1 to 100 bohr): within 9.1e-13 (full CI) and 7.1e-14 (RHF). Against the 50-digit evaluation
      (singlet full CI by Cholesky-reduced generalized eigenproblem, RHF by scan and root of dE/dtheta)
      at (3, 3, 0.5, 0.5, 0.1), (1, 3, 3, 0.5, 0.1), (1, 1, 0.5, 0.5, 0.1), (3, 1, 3, 0.5, 100),
      (2, 2, 0.8, 0.8, 4), (1, 1, 1.24, 1.24, 1.4) and (3, 3, 3, 3, 0.1): within 3.5e-13 (full CI) and
      5.1e-14 (RHF); the reference is within 5.6e-13 and 3.7e-14 there.
    - Step 6 integrals against the reference McMurchie-Davidson integrals at four inputs, including the
      range ends (3, 1, 3, 0.5, 5, 0.1, 0.3) and (1, 3, 0.5, 3, 0.1, 5, 10): within 2.4e-15 (S),
      1.6e-14 (H) and 3.1e-15 (repulsion). polarized_energies against the reference on all ten step 6
      and general test inputs: within 1.7e-14 hartree.
    - vibrational_levels against the reference on all ten step 5 test inputs: within 1.5e-6 cm^-1.
    - crown_check SECOND: all cases of steps 1-6 and general pass (results below).
    - Runtime per call (one thread): step 4 below 0.05 s, step 5 about 2 s (10 s for the near-threshold
      case), step 6 4.5-27 s (the great-circle RHF from 48 starts dominates).
- Step 5 mutants built on a superseded solver (rule E).
  - Finding: step_5_total_mass, step_5_harmonic_levels, step_5_limit_at_finite_distance and the
    vibrational_levels of whole_task_missing_hermite_sign were the version-11 sinc-DVR solver (h = 0.03,
    walls up to 36 bohr, fci_energy calls), which is itself wrong for test case 6, so each carried two
    deviations; limit_at_finite_distance failed case 6 because of its wall, not its limit.
  - Cause: the mutants were not rebuilt when the step 5 reference was rewritten in version 12.
  - Change: each is now the current solution/step_5.py with one change: total_mass uses
    mu = (massA + massB) u; limit_at_finite_distance measures the levels from E_fci(60 bohr) instead of
    the exact limit (levels shifted by E_inf - E_fci(60) = -q_A q_B / 60 hartree); harmonic_levels
    replaces the radial solver by harmonic-oscillator levels V(R_e) + omega (v + 1/2) from the curvature
    at the minimum of the same curve. The whole-task mutant is now the current solution.py with only the
    sign (-1)^(tau + nu + phi) dropped.
  - Regression test: crown_check MUT. limit_at_finite_distance now fails exactly the two charged-fragment
    cases 2 and 3 (and passes the neutral-fragment case 6), as its description says.
- Step 6 prompt did not name boys_function (rule F).
  - Change: "the Coulomb integrals need the Boys functions F_0 .. F_4: use boys_function (step 1)".
    step_dependencies stays [1] (the step 6 reference calls only boys_function).
- Step 6 case 0 bound and slack (rule G, task note).
  - Finding: the case bounded the result by the 4-digit Szabo-Ostlund 1s-only values (version 14; it
    had used fci_energy from the evaluation context before) and required E_fci < E_rhf strictly.
  - Change: the 1s-only bound now comes from the case's own Obara-Saika oracle restricted to the two 1s
    functions (the Szabo-Ostlund values check that restricted oracle with 2e-4 bands); margins 1e-3 and
    1e-4 as before (actual lowerings 0.0130 and 0.0036 hartree); the ordering is E_fci <= E_rhf + 2e-9
    (two outputs, each within 1e-9), as in step 4.
  - Regression test: step 6 case 0 with the reference, the second solution and the ±1e-12 shift.
- General tests (rule A): every case already called polarized_energies against the independent
  oracle. Case 0 now also checks the integrated link with step 4: the 8-function basis contains the two
  1s functions, so polarized_energies <= fci_energy (both energies) at the same geometry, slack 2e-9
  (actual lowerings 0.0034 and 0.0037 hartree), and E_fci <= E_rhf + 2e-9.
- Hidden test inputs in visible text (rule B).
  - Finding: the step 4 step_background quoted the literature energies for H2 at R = 1.4 bohr with
    zeta = 1.24, which is a step 4 test input; the step 5 step_background named He2 2+, the system of the
    step 5 error case.
  - Change: the step 4 background states only the size of the correlation energy, the step 5 background
    says that a well above the limit gives no bound levels. The step 5 prompt sentence on the reach of
    near-threshold levels now says "with 1 u nuclei" (the reach shrinks for heavier nuclei, so the
    unqualified sentence was not accurate over the whole mass range). background.md still describes the
    Szabo-Ostlund textbook examples (H2 at 1.4 bohr, HeH+ at 1.4632 bohr) as literature context; they
    are not prompt or docstring text, and their 4-5 digit values are far from the 1e-9 grading
    tolerance.
- Stated behaviour not tested (rule F): the prompts say "float numpy array" for steps 1, 2, 3 and 5;
  the tests now assert a floating dtype. The step 5 levels oracle helper is renamed _t_fd_levels (it is
  finite differences with Richardson extrapolation, not Numerov).
- Unused dependency: "from scipy.integrate import quad" is removed from required_dependencies and from
  the scaffold, solution, second-solution and mutant headers (no solution used it). The step 5 tests
  import quad themselves for their one-centre oracle.
- Step 4 contract: the error figures for short R (1.4e-10 at 0.1 bohr, ...) are labelled as the version-14
  measurements of the formulations checked then.
- Step 5 runtime (round-1 advisory). Measured here with one BLAS thread, each case alone in a fresh
  process with the reference: cases 0-7 take 8.5, 16.7, 9.3, 9.1, 16.2, 35.6, 14.4 and 1.0 s (the oracle
  included). Single reference calls at the corners named by the verifier, (2, 1, 3, 3, 10, 10),
  (2, 1, 3, 1.24, 10, 10) and (1, 1, 3, 0.8, 10, 1), take 9.9, 11.9 and 7.5 s; the 33-53 s seen in round
  1 were under heavier CPU load. No case was split.
- Rerun after all changes: tools/crown_check.py, all modes, every case alone in a fresh process, one
  BLAS/OpenMP thread, -j 2.
  - FORMAT OK (no problems; every general case calls polarized_energies).
  - REF: steps 1-6 pass 5/5, 6/6, 5/5, 8/8, 8/8 and 4/4; general 3/3; solution.py also passes every
    step file.
  - SECOND: all cases of steps 1-6 and general pass.
  - Mutants: upward_recursion fails 3/5 (cases 0, 1, 3); zeta_not_squared 4/6 (0, 1, 3, 5);
    wrong_boys_argument 4/5 (0, 1, 2, 4); full_exchange 8/8; orthonormal_atomic_basis 8/8;
    single_start_scf 1/8 (5); lowest_product_eigenvalue 1/8 (6); limit_at_finite_distance 2/8 (2, 3);
    total_mass 7/8 (0-6); harmonic_levels 7/8 (0-6); kinetic_without_angular_momentum 4/4;
    sigma_p_only 4/4; aufbau_scf 1/4 (2); whole_task_missing_hermite_sign 3/3 general cases.
  - SHIFT ±1e-12: every step case passes, and every general case with all six functions shifted.
  - All-None and all-zero controls pass 0 cases in every step and in general.
  - Repeat runs: second_solution.py gives bitwise identical outputs of all six functions (range-end
    inputs of steps 2-4 and 6, one step 5 input) in two clean processes with one thread. Runs with more
    threads were not made here (shared CPU).

## Version 16 (grading_fix: authoring-guide conformance, pass 2)

Fixes for the three blocking findings of the independent verification of version 15. Only the step 5
prompt, second_solution.py (vibrational_levels and polarized_energies), verification_explanation and
this file changed. No reference algorithm, test, target, stated tolerance, input domain or mutant was
changed. Each item gives the finding, its cause, the change, the regression test and what was rerun.

- False reason in the step 5 prompt (rules B, C, F).
  - Finding: the Errors paragraph of the step 5 prompt ended with "(With a charge above 2 the separated
    fragments in this basis repel each other and no curve holds five bound levels, so the step is
    limited to 1 <= Z <= 2.)". This is false just above 2. Reproduced here with solution.py, with only
    the charge cap raised to 3 (ZB = 1, zetaB = 1.24, 10 u nuclei). Five deep levels exist at
    ZA = 2.01, 2.05 and 2.1 (zetaA = 2.0925), at ZA = 2.1 with zetaA = 2.3 and at ZA = 2.3 with
    zetaA = 2.5. ZA = 2.5 (zetaA = 2.0925) and ZA = 3 (zetaA = 2.7) raise ValueError. For a charge just
    above 2 the lowest limit leaves the small fragment charge ZA - 2 on A, so the repulsive tail
    (ZA - 2) ZB / R is weak, while the HeH+-like well is about 0.2 hartree deep.
  - Cause: the version-11 note generalized a scan made only at ZA = 3, ZB = 1.
  - Change: the parenthetical sentence is deleted. The prompt keeps the domain (inputs "1 <= Z <= 2")
    and its stated error rule ("raise ValueError if a charge is not finite or not in [1, 2]"). The
    range is a design choice and needs no physical reason. The version-11 note above is annotated with
    these counterexamples.
  - Regression test: step 5 case 7 still checks the ValueError for ZA = 2.5 and ZB = 0.9 (unchanged);
    crown_check FORMAT and REF.
- The step 5 second solution passed the error case by accident and did not terminate on another
  in-domain input (rule D).
  - Finding: reproduced here. For He2 2+, (2, 2, 2.0925, 2.0925, m_He, m_He), the first DVR had
    n = int(7.7 / 2.2067) = 3 points. nxt - cur on arrays of 5 and 3 eigenvalues then raised a numpy
    broadcasting ValueError, and case 7 counted it as the intended exception. For (2, 2, 3, 1.2, 10, 10),
    where the reference raises ValueError in under a second, the box was doubled from 11.7 to
    23962 bohr (n = 17163). Here the run was stopped by a MemoryError under a 3 GB limit after 112 s.
  - Cause: the loop required all five lowest eigenvalues to converge in the box length. Box
    (continuum) states above the limit never converge. The loop also had no minimum number of grid
    points, no count of bound levels and no length cap.
  - Change (second_solution.py, vibrational_levels only):
    - The DVR always has at least 40 points.
    - Only eigenvalues below 0 are compared. A box is accepted when five of them have converged to
      1e-4 cm^-1 between its half length and its length. A Dirichlet wall only raises the eigenvalues,
      so an eigenvalue below 0 marks a bound level.
    - With fewer than five, the box is doubled until it holds every bound level, or at most to
      1000 bohr. For a repulsive tail the box holds every bound level once it passes the point, beyond
      the last non-positive tail value, where the attenuation integral int sqrt(2 mu V_tail) dR
      reaches 20. A neutral tail has no such point and uses the cap.
    - The step is then refined at that box length until the number of levels below 0 no longer
      changes, and ValueError is raised.
    - Limitation, stated in the code: a fifth level that needs a box longer than 500-1000 bohr would
      be reported as missing. Such a level is bound by less than about 1e-3 cm^-1 with 1 u nuclei. No
      test input comes near this.
  - Regression tests (one BLAS thread):
    - He2 2+ now raises the intended ValueError (no level below the limit) in 1.8 s, and
      (2, 2, 3, 1.2, 10, 10) in 2.1 s.
    - The levels of all ten step 5 test inputs are bitwise identical to version 15 (1.7-10 s per
      call).
    - 30 further in-domain inputs were chosen near the five-level threshold from a WKB scan of the
      reference curve: Z from 1 to 2, zeta from 0.8 to 3, masses 1 and 10 u, 0.2 to 6.9 estimated
      levels. The reference and the second solution both raise ValueError on the same 16. On the other
      14 they give five levels (fifth level from -0.60 to -6650 cm^-1) that agree within
      1.1e-6 cm^-1. The second solution takes 1.7 to 15 s per input.
    - crown_check SECOND on step 5.
- The step 6 second solution followed the reference's strategy for RHF and full CI (rule D).
  - Finding: confirmed.
    - The version-15 RHF minimized E(x) = 2 x.h.x + (xx|xx) by multi-start first-order descent on the
      unit sphere in the Loewdin basis, from the 8 eigenvectors of h and 40 standard-normal vectors, as
      the reference does (seed 12345 there, 77 here). Only the step rule differed: an exact Brent line
      search along great circles instead of Armijo backtracking with a maximum-overlap SCF polish.
    - The full CI built the reference's 64 x 64 product-space Hamiltonian and differed only in how the
      triplet was excluded.
    - verification_explanation and this file called this a third method.
  - Cause: in version 15 only the step 6 integrals were rewritten.
  - Change (second_solution.py, polarized_energies):
    - RHF by Roothaan-Hall SCF in the AO basis: F(D) C = S C eps with F = H + 2J - K and D = c c^T,
      canonical orthogonalization, and a level shift of 1 hartree on the virtual space. The lowest
      orbital of the shifted Fock matrix is occupied. Each iteration is then a damped downhill step,
      which can keep a non-aufbau orbital occupied and does not settle on a saddle point. A run has
      converged when the commutator FDS - SDF is below 1e-11.
    - The SCF starts from 64 fixed guess orbitals: every normalized basis function and every
      normalized sum and difference of two of them. There are no random starts and no core-Hamiltonian
      eigenvectors. The lowest converged energy is kept.
    - Two variants were rejected. With DIIS, the extrapolation stalled on stretched molecules with
      nearly degenerate orbitals (at one random point 1e-3 hartree above the minimum, commutator 7e-7
      after 3000 iterations). A maximum-overlap SCF without level shift, started from the Fock
      eigenvectors of atomic-function densities, missed the delocalized minimum of stretched H2
      (R = 10 bohr) by 0.33 hartree.
    - Full CI by direct CI for the singlet. The spatial wave function is sum Y_ij chi_i(1) chi_j(2)
      with Y symmetric, and sigma(Y) = hY + Yh + sum_kl (ik|jl) Y_kl in the canonical RHF orbitals (the
      eigenvectors of the converged Fock matrix). Davidson iteration with the diagonal preconditioner
      starts from the RHF determinant and four random symmetric vectors (seed 31) and runs until the
      residual is below 1e-10.
    - No Hamiltonian matrix is formed, and no triplet enters, since sigma keeps Y symmetric. Full CI is
      the same eigenvalue problem in every method, so for E_fci the independence lies in the
      integrals, the orbitals and the eigensolver. verification_explanation now says so.
  - Regression tests and checks (second-solution integrals, one thread):
    - The check covered 202 inputs: the 10 step 6 and general test inputs; 32 corners of the range
      (Z = 1 and 3 on each atom, zetaA = 0.5 or 3 with zetaB = 3.5 - zetaA, alphaA = 0.1 or 5 with
      alphaB = 5.1 - alphaA, R = 0.3 and 10 bohr); and 160 random points (Z and zeta uniform, alpha and
      R log-uniform).
    - On every input, an independent 400-start BFGS minimization of the closed-shell energy agrees
      with the new RHF within 4.7e-14 hartree. A dense singlet full CI (generalized eigenproblem in the
      non-orthogonal symmetric product functions) agrees with the direct CI within 1.8e-13 hartree.
    - Every SCF run converged, and at every point at least 29 of the 64 guesses reached the global
      minimum.
    - On the ten test inputs, polarized_energies agrees with this check within 1.8e-14 hartree. It
      takes 1.9-2.6 s per call (was 4.5-27 s); the integrals dominate.
    - crown_check SECOND on step 6 and general.
- Texts: verification_explanation is rewritten. It drops "a third method wherever the reference and
  the oracles differ", adds the full-CI caveat, the two checks above and the bound-level count of step
  5. The second-solution paragraph near the top of this file is rewritten for version 16, the
  superseded paragraph records the version-15 methods, and the version-15 bullet is annotated.
- Rerun after all changes: tools/crown_check.py, all modes, every case alone in a fresh process, one
  BLAS/OpenMP thread, -j 2.
  - FORMAT OK.
  - REF: steps 1-6 pass 5/5, 6/6, 5/5, 8/8, 8/8 and 4/4; general 3/3.
  - SECOND: all cases of steps 1-6 and general pass.
  - Mutants: each fails at least one case, with the same cases as in version 15:
    - upward_recursion fails 3/5 (cases 0, 1, 3); zeta_not_squared 4/6 (0, 1, 3, 5);
      wrong_boys_argument 4/5 (0, 1, 2, 4).
    - full_exchange 8/8; orthonormal_atomic_basis 8/8; single_start_scf 1/8 (5);
      lowest_product_eigenvalue 1/8 (6).
    - limit_at_finite_distance 2/8 (2, 3); total_mass 7/8 (0-6); harmonic_levels 7/8 (0-6).
    - kinetic_without_angular_momentum 4/4; sigma_p_only 4/4; aufbau_scf 1/4 (2).
    - whole_task_missing_hermite_sign 3/3 general cases.
  - SHIFT ±1e-12: every step case passes, and every general case with every evaluated function
    shifted.
  - All-None and all-zero controls pass 0 cases in every step and in general.
  - Repeat runs: two clean processes with one thread give bitwise identical outputs of the new
    second-solution polarized_energies (two inputs) and vibrational_levels (two inputs).

## Version 17 (grading_fix: platform content check)

- Finding: the platform content check (2026-10-12) blocks non-finite literals anywhere in the tests
  (float("nan"), np.nan, np.inf and the like, also in internal helpers), even when a prompt promises a
  ValueError for non-finite input, and it counts assertion sites per step: validation, type, shape and
  exception checks (a raise stated in the prompt at half weight) against computed-value checks, which must
  clearly dominate. It also asks that a relative-error requirement never apply to an exactly zero output
  and that every stated output shape and inclusive endpoint be exercised with values. No platform report
  had been received for this task; the rules were applied proactively.
- Cause:
  - Non-finite inputs: step 1 case 4 passed t = NaN and an array containing NaN, step 5 case 7 a NaN
    mass, an infinite mass and a NaN charge, matching the "not finite" clauses of the step 1 and step 5
    prompts and function headers. The step 4 RHF oracle started its search from best = np.inf (8 copies).
  - Validation weight: the contract check was repeated in the helper of every case (step 2: four asserts
    per case; steps 3-6 and general: one or two), and the invalid-input lists tested rules several times
    (step 1: 9 entries, step 2: 7, step 3: 4, step 5: 8). Counted conservatively (an assert mixing a
    shape and a value check counted on both sides, the "raised" assert of an error case as validation,
    each listed invalid input as 0.5), computed : validation sites were step 1 15 : 9.5, step 2 31 : 24.5,
    step 3 21 : 7, step 4 28 : 16, step 5 26 : 12, step 6 17 : 4, general 10 : 4.
- Change:
  - Texts: "not finite" is removed from the step 1 and step 5 prompts and function headers and from every
    copy (steps/, solution/, solution.py, second_solution.py, the step 1 and step 5 mutants and the
    whole-task mutant); the error messages of the code no longer mention finiteness. The solutions still
    reject non-finite input internally. No domain, tolerance, signature or other requirement changed; the
    function headers still match the scaffolds, the per-step solutions, solution.py and the mutants, and
    problem_io matches the step 6 header.
  - Validation tests: every non-finite entry is removed and each stated error rule is tested once at its
    limit: step 1 n_max = 17, n_max = 2.5, t = -1e-3 and t = 1.000001e6 (both inside arrays, for the
    "any t" rule); step 2 R, ZA, ZB, zetaA and zetaB equal to 0 (one entry each); step 3 R, zetaA and
    zetaB equal to 0; step 5 He2 2+ (fewer than five levels), ZB = 0.99 and massA = 10.05 u. The step 4
    oracle starts from best = 1e300.
  - Contract checks kept once per step, in one assert: step 1 case 3, steps 2-6 case 0, general case 0
    (the type assert on the fci_energy output in general case 0 is dropped; the values are still used).
    Step 1 case 3 now also checks a one-row (1, 3) batch, shape (1, 3, 17) with the inclusive end
    t = 1e6 among its values, and a Python float, shape (3,).
  - New computed cases with independent targets (rows "since version 17" in the target table):
    - step 1 case 5 (independent_recomputation, special_case): F_0..F_16 at t = 0.5, 8, 60 from the
      defining integral by mpmath tanh-sinh quadrature at 30 digits; at t = 1000, 4.2e4 and 1e6 from the
      large-t closed form (2n - 1)!! sqrt(pi) / (2^(n+1) t^(n+1/2)), exact to Q(n + 1/2, t) < 1e-380.
    - step 1 case 6 (independent_recomputation, analytic identity): the Taylor series at t = 1e-3, 0.02,
      0.3 (40 digits), F_0 = sqrt(pi / t) erf(sqrt t) / 2 at t = 0.7, 5, 30, and the identity
      (2n + 1) F_n - 2t F_(n+1) = exp(-t) between computed outputs at t = 1.5 and 6 (tolerance 2e-11
      times the sum of the two term magnitudes).
    - step 2 case 6 (special_case, exact relations): S[0, 0] = S[1, 1] from the zeta-independent norm
      formula; at R = 100 bohr S[0, 1] = H[0, 1] = 0 and H[0, 0], H[1, 1] from one-centre closed forms
      and the shell theorem; lengths divided by 2 with charges doubled leave S unchanged and multiply H
      by 4 (tolerance 2e-10 and 5e-10); the mirror exchange of the atoms (2e-10).
    - step 3 case 5 (special_case, exact relations): (00|00) and (11|11) from the concentric closed
      form at R = 1.7 bohr; the whole array at R = 100 bohr ((00|11) = S_AA S_BB / R, overlap densities
      zero); lengths divided by 2 multiply every integral by 2 (tolerance 3e-10); mirror exchange
      (2e-10).
    - step 4 case 8 (special_case): separated atoms at R = 100 bohr for H2, HeH+, the ion pair
      (1, 1, 1.24, 2.69) and (1.5, 1.5, 2.0, 1.24): E_fci = the lowest decoupled fragment arrangement,
      E_rhf = the minimum of a convex quadratic in the weight of B, from one-centre closed forms.
    - step 5 case 8 (independent_recomputation, exact relation): HeH+ with masses (2, 2) u against the
      finite-difference target, and (1.5, 3) and (4, 4/3) u (the same reduced mass 1 u) against the
      first within 0.02 cm^-1.
  - Rule on relative errors: the only relative requirement (step 1, 1e-11) applies to F_n(t), which is
    positive on the whole domain (smallest value F_16(1e6) = 2.6e-87, representable); no exactly zero
    output exists, so nothing changed there.
  - verification_explanation mentions the new routes; n_test_cases: step 1 7, step 2 7, step 3 6,
    step 4 9, step 5 9, step 6 4 (unchanged), general 3 (unchanged).
- Counts after the change (same conservative count), computed : validation: step 1 19 : 4, step 2
  29 : 4.5, step 3 22 : 3.5, step 4 30 : 1, step 5 22 : 3.5, step 6 17 : 1, general 10 : 1.
- Regression tests: the new cases fail the mutants they target: upward_recursion fails step 1 cases 5
  and 6; zeta_not_squared step 2 case 6; wrong_boys_argument step 3 case 5; full_exchange,
  orthonormal_atomic_basis and single_start_scf step 4 case 8 (lowest_product_eigenvalue passes it: at
  R = 100 bohr the triplet is degenerate with the covalent singlet); total_mass and harmonic_levels step 5
  case 8 (limit_at_finite_distance passes it: the He + H+ limit is neutral). The reference and the
  second solution pass them with large margins (step 4 case 8 within 1.8e-15 hartree, step 5 case 8
  within 6.1e-7 cm^-1 for the reference and 3.4e-8 cm^-1 for the second solution).
- Rerun after all changes: tools/crown_check.py, all modes, every case alone in a fresh process, one
  BLAS/OpenMP thread, -j 2.
  - FORMAT OK (no non-finite literal left in tests/).
  - REF: steps 1-6 pass 7/7, 7/7, 6/6, 9/9, 9/9 and 4/4; general 3/3.
  - SECOND: all cases of steps 1-6 and general pass.
  - Mutants: upward_recursion fails 5/7 (cases 0, 1, 3, 5, 6); zeta_not_squared 5/7 (0, 1, 3, 5, 6);
    wrong_boys_argument 5/6 (0, 1, 2, 4, 5); full_exchange 9/9; orthonormal_atomic_basis 9/9;
    single_start_scf 2/9 (5, 8); lowest_product_eigenvalue 1/9 (6); limit_at_finite_distance 2/9
    (2, 3); total_mass 8/9 (0-6, 8); harmonic_levels 8/9 (0-6, 8); kinetic_without_angular_momentum
    4/4; sigma_p_only 4/4; aufbau_scf 1/4 (2); whole_task_missing_hermite_sign 3/3 general cases.
  - SHIFT ±1e-12: every step case passes, and every general case with every evaluated function shifted.
  - All-None and all-zero controls pass 0 cases in every step and in general.
