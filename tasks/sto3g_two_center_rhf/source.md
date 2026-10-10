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
second solution (Loewdin orbitals). It is not always the lowest eigenvalue of the full 4 x 4 product
space: for diffuse exponents at stretched R the triplet lies below the singlet in this basis (by
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

The second solution uses other methods where practical: composite Gauss-Legendre quadrature of the
defining integral for F_n (since version 13); shell-theorem radial quadrature instead of closed-form
Gaussian integrals for steps 2 and 3 (the formulation of the test oracles, so there it is a second
implementation of the oracle method rather than a third method); for step 4 RHF by direct
minimization over the angle of the normalized coefficient vector (1440-point scan, every local minimum
refined) and full CI in Loewdin orbitals; for step 5 its own fci_energy sampled on 200 Chebyshev nodes
in ln R on [0.3 bohr, r_c] and interpolated, the exact separated-fragment tail beyond r_c, and a
Colbert-Miller sinc DVR whose length is doubled and whose step is refined until the five levels change
by less than 1e-4 cm^-1 (since version 12); for step 6 Obara-Saika integrals, full CI in singlet
configuration functions and RHF by steepest descent along great circles with Brent line searches.

Superseded descriptions, kept for the record (earlier versions of this file): the step 5 reference
was a sinc DVR on a widening fixed domain until version 11; the second solution then used Chebyshev
spectral collocation for the levels, and direct orbital minimization against the self-consistent field
that the step 4 reference used until version 9; the step 1 target was a 400-point Gauss-Legendre
quadrature of F0 until version 6, and the levels target was renormalized Numerov shooting until
version 7. None of these is used now.

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
| sanity bands | Szabo and Ostlund values: S01, H00, H01, (00\|00), (00\|11), (01\|01), (00\|01) for H2; S01, H00, H01, H11 for HeH+; E_rhf = -1.1167 (H2) and -2.8607 (HeH+), E_fci = -1.1373 (H2), also as the 1s-only upper bounds in step 6 case 0; stretched H2 full CI near 2 x -0.466582 | bands of 1e-4 to 2e-3 only |

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
