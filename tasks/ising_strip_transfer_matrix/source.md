# Source

- L. Onsager, "Crystal statistics. I. A two-dimensional model with an order-disorder
  transition", Phys. Rev. 65, 117 (1944). Exact critical temperature
  Tc = 2 / ln(1 + sqrt(2)) for J = k_B = 1.
- B. Kaufman, "Crystal statistics. II. Partition function evaluated by spinor
  analysis", Phys. Rev. 76, 1232 (1949). Exact eigenvalues of the transfer matrix of a
  periodic strip; used for the step 2 targets and for the f and lengths entering the step 6
  targets (the second solution used it for step 2 before version 10).
- C. N. Yang, "The spontaneous magnetization of a two-dimensional Ising model", Phys.
  Rev. 85, 808 (1952). Limit of m_L below Tc.
- T. D. Lee and C. N. Yang, "Statistical theory of equations of state and phase
  transitions. II. Lattice gas and Ising model", Phys. Rev. 87, 410 (1952). Zeros of the
  partition function on the imaginary field axis.
- M. E. Fisher, "Yang-Lee edge singularity and phi^3 field theory", Phys. Rev. Lett. 40,
  1610 (1978).
- C. Itzykson, R. B. Pearson and J. B. Zuber, "Distribution of zeros in Ising and gauge
  models", Nucl. Phys. B 220, 415 (1983). Finite-size scaling of the zeros.
- M. P. Nightingale, "Scaling theory and finite systems", Physica A 83, 561 (1976).
  Finite-size scaling with strip transfer matrices.
- J. L. Cardy, "Conformal invariance and universality in finite-size scaling", J. Phys. A
  17, L385 (1984). xi_L = L / (2 pi x) on a periodic strip at criticality.
- H. W. J. Bloete, J. L. Cardy and M. P. Nightingale, Phys. Rev. Lett. 56, 742 (1986);
  I. Affleck, Phys. Rev. Lett. 56, 746 (1986). f_L = f_inf - pi c T / (6 L**2).
- M. Luscher, "Volume dependence of the energy spectrum in massive quantum field
  theories", Commun. Math. Phys. 104, 177 (1986). Exponential finite-size corrections with
  a power-law prefactor, which is why strips converge slowly near Tc in a weak field.
- T. Nishino and K. Okunishi, "Corner transfer matrix renormalization group method",
  J. Phys. Soc. Jpn. 65, 891 (1996). Used for the step 7 reference.
- V. Zauner-Stauber, L. Vanderstraeten, M. T. Fishman, F. Verstraete and J. Haegeman,
  "Variational optimization algorithms for uniform matrix product states", Phys. Rev. B 97,
  045145 (2018). Used for the step 7 second solution.

The row-labelling rule, the equal sharing of the in-row bonds and of the field between the
two factors, and the exact way the steps are combined are our own choices, made so that
every number in the tests is fixed.

## How the test targets are obtained (current state, version 10)

Steps 1 and 2 compute their targets inside the tests from independent formulas. The inline numbers
of steps 3 to 6 were regenerated in version 10 by an extended-precision route that shares no code
with either solution (before version 10 they were reference outputs, cross-checked by the second
solution). The step 7 and whole-task targets are outputs of the reference algorithm run at a higher
bond dimension, confirmed by a different algorithm (VUMPS, the second solution) and by the exact
h = 0 limit. The column "Agreement" was measured in version 10.

| Target | Method | Agreement |
|---|---|---|
| M entries (step 1) | brute-force sum over the energy inside the test (entry by entry for L = 3, 4 and 6; over all pairs of rows, column by column, at L = 10); hand-worked entries, including the ends of the stated range | reference and second solution within 3e-14 relative of a 40-digit (mpmath) sum over the energy: every entry for L = 3 and 4, 303 entries for L = 7 and 10, at T = 0.5, 1, Tc, 10 and h = 0, 0.1, +-20, +-20i, 20 exp(i pi/3), 11i; the reference is exactly symmetric there |
| f, xi_spin, xi_energy (step 2) | Kaufman's exact strip spectrum computed inside the test with 60-digit decimal arithmetic (standard library), no transfer matrix; xi_energy = 1 / (2 g_1) from the two lowest fermion modes q = 1 and 2L - 1 | reference within 2.2e-16 (f), 1.7e-8 (xi_spin, at L = 10, T = 1, where the spin gap is 1.4e-8 relative) and 2.7e-15 (xi_energy); second solution within 2.2e-16, 1.2e-9 and 1.3e-15 (tolerances 1e-10, 1e-6, 1e-6) |
| m_L (step 3) | zero-momentum block of the transfer matrix (basis of normalized translation orbits, entries summed from the energy in mpmath), diagonalized at 50 digits; m_L = abs(<even|S|odd>) / L between its top flip-even and top flip-odd states | reference 6.7e-16, second solution 1.2e-15 (tolerance 1e-9); at T = 0.5 and 0.6 the values equal Yang's bulk magnetization within 1e-24 |
| chi (step 4) | -d2f/dh2 from a centred second difference with field step 1e-25 of f(h) = -T ln(lambda_0(h)) / L, lambda_0(h) the top eigenvalue of the zero-momentum block at 90 digits; the Richardson partner with step 2e-25 agrees to 1e-32 | reference 1.5e-8 relative (at L = 10, T = 1, chi = 1.4e9), second solution 1.7e-8 (tolerance 1e-6) |
| H_edge (step 5) | root of Re((a - b)**2) for the two eigenvalues of largest modulus of the zero-momentum block in the field 1j H, at 40 digits (grid bracket, Illinois iteration to 1e-32 relative) | reference 1.1e-13 relative, second solution 5.7e-12 (tolerance 1e-8). In double precision, for L = 3 to 8 at ten temperatures from 1.5 to 10 and 120 fields below each edge, the leading pair of the full matrix stays real (imaginary parts below 1.6e-13 relative) and distinct, and at 0.5 and 0.99 of the edge it equals the leading pair of the zero-momentum block within 8e-14 |
| final dict (step 6) | the estimators of the prompt evaluated in mpmath on Kaufman's f and lengths and on the extended-precision m_L, chi and H_edge at widths 3 to 9 (Tc taken as the double value the tests pass) | reference 2.6e-13, second solution 4.5e-12 (tolerance 1e-4) |
| bulk f, m, chi, c (step 7, general) | corner-transfer-matrix renormalization at bond dimension 28 (the reference uses 16), free energy from kappa = Z(3x3) Z(4 corners) / Z(corners + 2 edges)**2, chi from Richardson central differences of m, c from Richardson second differences of f | rerun of the bond-dimension-28 computation on the eight points kept from version 9: within 2.2e-15 (f), 7e-14 (m), 1.7e-10 relative (chi) and 1.2e-8 relative (c) of the stored values; VUMPS (second solution: per-site eigenvalue of the row transfer matrix, chi = -d2f/dh2 and c from five-point stencils) within 5.2e-14 (f), 4.6e-11 (m), 5.3e-7 relative (chi) and 1e-7 relative (c) on all ten points; at h = 0 the corner-transfer-matrix free energy reproduces Onsager's within 1.6e-14 (bond dimension 16) and 1.3e-15 (28) at T = 2, 2.5 and 3, and its magnetization Yang's within 1e-14 at T = 2 and zero within 7e-14 above Tc |

## Why each step is not a direct transcription

Each step was run against the default that a practitioner would reach for first, on the test points
of that step. The tolerance in parentheses is the one stated in the prompt. Rerun in version 10 on the
current test points unless marked otherwise.

| Step | Default route | Error on the test points | Correct alternatives that pass |
|---|---|---|---|
| 2 (rel 1e-6) | xi_energy from the second eigenvalue of M | 1.1 to 2.4e8 (every case: that eigenvalue is the top odd state) | flip-sector blocks (reference); momentum and flip sectors (second solution); free fermions |
| 2 (rel 1e-6) | xi_energy from the third eigenvalue of M (mutant third_eigenvalue) | 0.22, 0.74, 0.83 at (L, T) = (4, 3), (6, 5), (3, 10), where the third eigenvalue is an odd state; passes at (10, 1) and (8, Tc), where it is the second even state (it is the second even state below a width-dependent crossover temperature, 2.05 at L = 3, 2.33 at L = 4 and 2.41 to 2.43 for L = 5 to 10, so at Tc and just above it for L >= 4, and an odd state above it) | as above |
| 3 (abs 1e-9) | matrix element between the two leading eigenvectors of one eigh call on M | 1.0 at (L, T) = (10, 0.5) and 1.8e-3 at (10, 0.6): the pair is degenerate to 9e-18 and 7e-15 relative and the solver returns an arbitrary mixture (an earlier entry of this table also recorded 3.3e-4 at (8, 0.5); the outcome there depends on the LAPACK build) | sector blocks; rotation of the pair into parity states |
| 3 (abs 1e-9) | Yang's bulk magnetization | 0.08 at (6, 2.2), 0.51 at (7, 4.0), 0.64 at (3, 10) | |
| 4 (rel 1e-6) | central second difference of f(h), step 1e-3 to 1e-7 | every step fails at least one case: at (10, 1.5) the error falls from 0.99 (step 1e-3) to 5.2e-5 (1e-7), at (10, 1.0) it stays 0.99 to 1.0, and at high T rounding dominates the small steps (3.3 at (4, 10) with step 1e-7); Richardson on 1e-4 and 5e-5 fails at (10, 1.0) (1.0), (10, 1.5) (0.72) and (4, 10) (1.0e-5) | spectral sum (reference); odd-sector linear solve (second solution) |
| 4 (rel 1e-6) | correlation summed over r >= 0 only (mutant one_sided_correlation_sum) | 0.11 to 0.50 (every case) | |
| 5 (rel 1e-8) | first field at which any eigenvalue becomes complex (mutant first_complex_eigenvalue) | 0.32 at (6, Tc), 0.99 at (8, 3.0): at these two points a lower pair leaves the real axis first (at 0.68 and 0.01 of the edge); in the other four cases, at (7, 1.5), (4, 10), (5, 2.0) and (case 5) (3, 1.5), (3, 10), (8, 1.5), no eigenvalue becomes complex before the leading pair meets and the mutant passes | root of the discriminant of the leading pair; bisection on the leading pair |
| 5 (rel 1e-8) | linear scan of H starting at 1e-3 | cannot bracket edges below its first grid point; the edges at (7, 1.5) and (8, 1.5) are 1.3e-4 and 4.8e-5 (version 9 recorded an error factor 6.6 at (7, 1.5)) | geometric bracketing |
| 5 (rel 1e-8) | Hermitian eigensolver (eigvalsh) on the complex symmetric matrix | it diagonalizes a different, Hermitian matrix whose leading pair never meets: the smallest (a - b)**2 over 0 < H < 2 H_edge is 8.5 to 1.2e3 on the five test points checked, so no edge is found | general (non-Hermitian) eigensolver |
| 7 (f abs 1e-10, m abs 1e-9, chi and c rel 1e-5) | widest diagonalizable strip, L = 10 | f off by 1.6e-8 to 3.5e-3 and m by 4.4e-7 to 0.38 on the eight points kept from version 9 (every point fails) | corner-transfer-matrix renormalization or VUMPS with the correct per-site partition function and converged derivatives |
| 7 | partition function per site without dividing out the corner contribution (mutant missing_corner_normalization) | f off by 0.05 to 0.6 (the size depends on the arbitrary normalization of the environment) | |
| 7 | specific heat as -d2f/dT2, no factor T (mutant specific_heat_without_T) | factor T = 2 to 3 | |
| 7 | susceptibility of free spins (1 - m**2) / T (mutant free_spin_susceptibility) | factor 2.0 to 67 on the ten points | |
| 7 | plain central differences, no extrapolation | at (Tc, 0.005) chi off by 1.2e-4 (step 1e-4) and 3.0e-3 (step 5e-4); at (2.6, 0.005) c off by 2.4e-5 (step 5e-3) | Richardson extrapolation or higher-order stencils on a well-converged f |

## Runtime

Measured in version 10 with single-threaded BLAS (OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = 1; numpy 2.4.6, scipy 1.17.1): every case of steps 1 to 6 runs in at most 3.7 s with
the reference and 6.5 s with the second solution (each case alone in a fresh process); one call of the
reference bulk_thermodynamics (one cold and eight warm-started corner-transfer-matrix runs) takes 1.3
to 11 s and one call of the second solution (VUMPS, one cold and eight warm-started runs) 1.7 to 53 s.
In version 9 the round-1 verifier measured the step 7 calls several times slower with the default
multithreaded BLAS on the loaded shared machine (reference 45 s at (Tc, 0.005), second solution more
than 10 min); this thread-configuration risk is recorded as an open issue, not as a time budget.

## Version history

Version 5: the free-energy tolerance of step 2 is 1e-10 instead of 1e-12 (a 1e-12 statement sat at the rounding level of the eigenvalue solver), and the step 2 targets are evaluated inside the test from Kaufman's formula.

Version 6: endpoint cases added to the tests — strip_magnetization(3, 10), strip_susceptibility(3, 2.5) and (3, 10), lee_yang_edge(3, 1.5) and (3, 10), critical_exponents(3) and (7); every value cross-checked by the two solutions (agreement 1e-12 or better). The Kaufman helper of the step 2 tests is written inside every test case (no preamble).

Version 7 (difficulty increase): the last step now returns the full thermodynamics of the
infinite lattice in a field, (f, m, chi, c), instead of m alone, over a wider and harder
range (h from 0.005 instead of 0.02, so the correlation length near Tc is about twice as
long). The free energy needs the correctly normalized per-site partition function of the
infinite-lattice method, and chi = dm/dh and c = -T d2f/dT2 need converged derivatives to
1e-5 relative. Targets from CTMRG at bond dimension 28, cross-checked with VUMPS.

Version 8: step 7 states a time budget of 60 s per call (the reference needs at most about
15 s), asserted on every call in the step 7 and whole-task tests; two rollout attempts had
timed out on step 7 without a stated budget. Step 6 states the scaling laws (Cardy's relation,
finite-size power laws between two widths, the conformal law of the free energy for three
widths) instead of the final estimator formulas; the expected values are unchanged.

## Version 9 (grading_fix: authoring-guide conformance)

1. Timing removed from the tests and the prompt.
   - Finding: tests/step_7.py and tests/general.py wrapped every call to bulk_thermodynamics in a
     perf_counter clock read with an assert of at most 60 s, and the step 7 prompt said "each call must
     finish within 60 s on one CPU core".
   - Cause: the budget was added in version 8 after two rollouts timed out; the authoring guide forbids
     clock reads and timing asserts in tests and per-call time budgets in prompts.
   - Change: the wrapper and the budget sentence are gone; every case calls bulk_thermodynamics directly.
     No other prompt, docstring, scaffold, solution or mutant stated a time budget, and no size or
     domain statement was attached to the removed sentence. The six step 7 cases and the two whole-task
     cases keep every value check with the same targets and tolerances.
   - Regression test: crown_check format finds no clock/timing pattern; the step 7 and general cases pass
     for the reference and the second solution and still reject every step 7 and whole-task mutant.
2. Test-case markers and preamble.
   - Finding: the markers of tests/step_7.py and tests/general.py carried descriptions
     ("# --- test case 0: exact Tc, weakest field ... ---") and both files had code before case 0.
   - Cause: the descriptive marker style predates the parser format.
   - Change: markers are exactly "# --- test case N ---" (0..5 for step 7, 0..1 for general), the
     descriptions are plain comments inside each case, each case has its own import. n_test_cases of
     step 7 stays 6. tools/selfcontain.py leaves every test file of the task unchanged (no PROBLEM line).
   - Regression test: crown_check format reports the markers of every test file as normalized.
3. Zero absolute tolerance in step 1.
   - Finding: the step 1 tests used np.allclose(..., rtol=1e-10, atol=0.0).
   - Cause: a relative-only comparison was written with an explicit zero absolute tolerance.
   - Change: the comparison with the brute-force matrix is written as the maximum relative entry error
     <= 1e-10 (the stated per-entry relative accuracy; every entry is an exponential, never zero, so the
     meaning is that of the old call). The symmetry check compares two computed entries M[n, m] and
     M[m, n], each allowed a relative error of 1e-10, so it now uses 2e-10 (it used 1e-10, which a
     candidate within the stated accuracy could fail). The stated accuracy itself is unchanged.
   - Regression test: step 1 passes for the reference, the second solution and the +-1e-12 shift; the
     field_not_shared mutant still fails every case with h != 0 (cases 0, 1, 3 and 4 after item 7; case
     2 has h = 0, where it equals the reference).
4. Metadata and labels: subfield, tags, expert_time_estimate_hours, the relevant_experience
   placeholder (to be written by the author), difficulty_explanation, solution_explanation,
   verification_explanation, author and affiliation added; edit_label is grading_fix.
5. Consistency of non-graded text: background.md still said that the last step asks for the bulk
   magnetization only; it now names the four quantities. (The step 1 contract range is handled in
   item 7.)
6. Target re-verification (no target changed): the CTMRG of the step 7 reference was rerun at bond
   dimension 28 on all eight step 7 and whole-task points; it reproduces the inline targets to
   3e-15 in f, 7e-14 in m, 2e-10 relative in chi and 1.3e-8 relative in c. The reference and the second
   solution agree on the step 3 to 6 test inputs to 1.1e-15 (m_L), 2.4e-9 relative (chi), 1e-12
   relative (H_edge) and 4.3e-12 (step 6 values).
7. Step 1 domain bounded to the validated range (found by an independent verifier after the first
   conformance pass of this version).
   - Finding: the step 1 prompt ("L is an integer with L >= 3, and T > 0"), the transfer_matrix
     docstring (problem.yaml and its six copies in steps/step_1.py, solution/step_1.py, solution.py,
     second_solution.py, mutants/step_1_field_not_shared.py and mutants/whole_task_field_not_scaled.py)
     promised every entry to relative accuracy 1e-10 for any L >= 3, any T > 0 and any h, and the
     contract had been widened earlier in this version from "L from 3 to about 11" to "L >= 3" to
     match. The reference cannot meet that: the largest exponent of an entry is L (2 + |Re h|) / T,
     so entries overflow to inf (and others underflow to 0) once it passes about 709;
     transfer_matrix(10, 0.02) and transfer_matrix(3, 0.005) contain inf and 0 entries, and memory
     grows as 4**L. The tests covered only L = 3, 4, 6 with T = 1.7 to 2.5.
   - Cause: the step 1 range was never bounded; it was written as the formal definition domain rather
     than the range that was validated and that the later steps use.
   - Change: the prompt now says "for every integer L from 3 to 10, every T from 0.5 to 10 and every
     real or complex h with |h| <= 20"; all seven docstring copies say 3 <= L <= 10,
     0.5 <= T <= 10 and |h| <= 20; the contract valid_input_ranges says the same and why. The range
     covers every call of the later steps: L up to 10 (steps 2 to 4; step 6 uses widths up to 9),
     T down to 0.5 (step 3) and up to 10, and imaginary fields up to 1e-12 T 2**40 = 11.0 at T = 10,
     the last point of the geometric bracket of the Yang-Lee edge in both solutions (the edge itself is
     at most 6.13, at L = 3 and T = 10). |h| <= 20 rather than 11 keeps a margin; in this range the
     exponent of every entry lies between -440 and 440 (largest entry exp(440) at L = 10, T = 0.5,
     h = -20), so no entry overflows or underflows. Validation: the reference and
     the second solution agree with a 40-digit sum over the energy (mpmath) to 3e-14 relative on every
     entry for L = 3 and 4 and on about 300 sampled entries for L = 7 and 10, at T = 0.5, 1, Tc
     and 10 and h = 0, 0.1, 20, -20, 20i, -20i, 20 exp(i pi/3) and 11i; the reference is exactly
     symmetric there.
     The reference, targets and science are unchanged.
   - Regression test: two new self-contained cases in tests/step_1.py (n_test_cases 3 -> 5).
     Case 3 compares the brute-force entry-by-entry sum and hand-worked entries at the low-T end
     (L = 3, T = 0.5, h = 0.1: M[0, 0] = exp(12.6), M[7, 7] = exp(11.4), M[0, 7] = 1) and at the
     high-T, largest-imaginary-field end (L = 3, T = 10, h = 20i: M[0, 0] = exp(0.6 + 6i),
     M[7, 7] = exp(0.6 - 6i), M[0, 7] = 1). Case 4 takes the widest strip at the lowest temperature,
     L = 10, T = 0.5, and compares every entry with a column-by-column sum of the energy over all
     pairs of rows, at h = 0 (M[0, 0] = exp(40), M[0, 1] = exp(32), M[0, 1023] = 1) and at the largest
     real field h = -20 (M[1023, 1023] = exp(440), M[0, 0] = exp(-360), M[0, 1023] = 1, symmetry to
     2e-10). Both pass for the reference, the second solution and the +-1e-12 shift; the
     field_not_shared mutant fails both (cases 0, 1, 3 and 4 in all); the all-None and all-zero
     controls fail both.

Rerun (after all seven items, including the two new step 1 cases): tools/crown_check.py (all modes,
every case alone in a fresh process, -j 2) and two clean-process runs of solution.py and
second_solution.py with identical outputs. Result:
FORMAT OK; REF 5/5, 5/5, 6/6, 6/6, 6/6, 3/3, 6/6 for steps 1 to 7 and 2/2 for general (solution.py
also passes every step file); SECOND the same counts, all pass; every mutant fails at least one case
(field_not_shared 4/5, third_eigenvalue 3/5, correlation_one_row 4/6, finite_difference 4/6,
first_complex_eigenvalue 2/6, inverted_edge_ratio 3/3, missing_corner_normalization 6/6,
specific_heat_without_T 6/6, free_spin_susceptibility 6/6, whole_task_field_not_scaled 2/2); SHIFT
+-1e-12 all pass; the all-None and all-zero controls pass 0 cases in every step and in general;
SUMMARY ALL OK. The run used OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1: on the
shared, heavily loaded machine, a first attempt with the default multithreaded BLAS made the second
solution's step 7 cases (VUMPS) run for more than ten minutes each, while one cold VUMPS run takes
about 10 s single-threaded. Repeat runs in clean processes gave bit-identical outputs (reference twice
with default threads and twice single-threaded, second solution twice single-threaded); between the
two thread settings the reference differs by at most 1.6e-8 relative (xi_spin at L = 10, T = 1, whose
gap is 1.4e-8 relative) and elsewhere by 1e-13 or less, far inside the stated tolerances. After item 7,
solution.py and second_solution.py were each run twice more in clean single-threaded processes on the
inputs of every test case of the task (the seven step 1 matrices compared through a hash of their
bytes, 44 outputs in all); both pairs of runs were identical.

## Version 10 (grading_fix: authoring-guide conformance, pass 2)

The findings come from the round-1 independent verifier of version 9, the earlier agents' design
notes and the pass-2 rules (every general case calls the final function; no hidden test input in the
visible text; truthful metadata; a second solution that is neither the reference's method nor a test
oracle; mutants built from the current reference). The science, the prompts' tolerances and the
reference solution are unchanged.

1. Precision of the step 2 Kaufman oracle.
   - Finding: the in-test target xi_spin at (L, T) = (10, 1) was computed as 1 / (l0 - l1) in double
     precision and was 1.6e-7 relative away from the exact value (16% of the 1e-6 tolerance); the
     source.md row quoted the reference error against the exact value (1.7e-8), not against the
     graded target (1.8e-7).
   - Cause: below Tc, l0 - l1 = ln(lambda_0 / lambda_1) is an exponentially small difference (1.4e-8 at
     L = 10, T = 1) of two sums of order L, so double precision loses about seven digits.
   - Change: _t_kaufman, in each of the five self-contained cases, evaluates the same formula with
     60-digit decimal arithmetic from the Python standard library (pi by Machin's formula, cosine by
     its Taylor series, Decimal exp, ln and sqrt; Decimal(T) is the exact value of the float T). Test
     points, formula and tolerances are unchanged.
   - Regression test: the helper agrees with a 60-digit mpmath evaluation of Kaufman's spectrum to the
     rounding of the returned floats (at most 1e-16 relative) at the five test points and at (10, 10),
     (3, 1) and (7, 1.37). The graded margin at (10, 1) is now the solution's own error: 1.7e-8 for the
     reference, 1.2e-9 for the second solution. The third_eigenvalue mutant still fails cases 0, 1, 4.
2. Steps 3 to 6 targets regenerated independently (rule C).
   - Finding: the inline values of steps 3 to 6 were reference outputs (the source.md table listed
     "reference" as their method), while the same file said that no target was copied from the
     reference.
   - Cause: these values had been produced by the reference and only cross-checked by the second
     solution and by earlier small-width 40-digit checks.
   - Change: every inline value of steps 3 to 6 was recomputed with the extended-precision
     zero-momentum-block route described in the table above (mpmath; entries summed from the energy;
     no code of either solution is used) and replaced by that value rounded to double. The values moved
     by at most 2.1e-9 relative in step 4 (chi at (10, 1), where the old reference value was off by
     2.1e-9; 3.2e-12 at (10, 1.5), at most 1e-15 elsewhere), 2.3e-13 relative in step 5, 2.2e-16 in
     step 3 and 3.4e-13 in step 6, all far inside the unchanged tolerances. Each test case now names its
     target route in a comment. The cross-checks behind the route were run in this version: the
     zero-momentum top even, top odd and second even levels reproduce Kaufman's f, xi_spin and
     xi_energy to 1e-39 for widths 3 to 8 at Tc, and the leading pair of the zero-momentum block equals
     the leading pair of the full matrix within 8e-14 below every edge checked (L = 3 to 8, ten
     temperatures).
   - Regression test: the reference and the second solution pass every step 3 to 6 case with the
     margins of the table above; every mutant of steps 3 to 6 still fails (item 6).
3. Smallest Yang-Lee edge of the domain tested.
   - Finding: the step 5 domain (L <= 8, T >= 1.5) has its smallest edge at L = 8, T = 1.5 (4.8e-5),
     which was untested (the smallest tested edge was 1.3e-4), and difficulty_explanation spoke of
     edges "of order 1e-4".
   - Cause: the endpoint case covered L = 3 only.
   - Change: lee_yang_edge(8, 1.5) is added to step 5 case 5 (target 4.821088064950643e-05 from the
     40-digit route; n_test_cases stays 6). difficulty_explanation now says "edges down to about
     5e-5 below Tc", and the step 5 contract names the smallest edge and the grid on which the leading
     pair was checked. No prompt or docstring mentions the point.
   - Regression test: the reference (1.1e-13) and the second solution (5.7e-12) pass; the
     first_complex_eigenvalue mutant fails cases 0 and 1.
4. Weak-field corners of the step 7 domain tested.
   - Finding: the corners (T, h) = (2.0, 0.005) and (3.0, 0.005) of the stated domain were untested.
     The low-temperature corner is where a weak field must select the up state rather than a metastable
     down branch.
   - Cause: the step 7 points were chosen around Tc and at h = 0.2.
   - Change: step 7 cases 6 (2.0, 0.005) and 7 (3.0, 0.005), with the same four checks and tolerances
     as the other cases (n_test_cases 6 -> 8). Targets from corner-transfer-matrix renormalization at
     bond dimension 28, confirmed by VUMPS within 1.2e-14 (f), 9.5e-15 (m), 2.5e-7 relative (chi) and
     9.7e-8 relative (c); the reference (bond dimension 16) is within 4e-15 (f), 1.2e-15 (m), 3.4e-11
     relative (chi) and 1.8e-7 relative (c).
   - Regression test: both solutions pass; the three step 7 mutants fail both new cases; the controls
     fail them.
5. Second solution (rule D).
   - Finding: second_solution.strip_lengths was Kaufman's closed-form spectrum, the very formula the
     step 2 tests use as their oracle, so it gave no independent check of step 2. Its
     bulk_thermodynamics had a one-line docstring that did not match the function_header, and an
     unused nested helper second(g, x, d).
   - Cause: the second solution was written before the step 2 oracle moved into the tests.
   - Change: strip_lengths now diagonalizes the transfer matrix in its full symmetry sectors, lattice
     momentum k = 2 pi j / L (translation of a row) times spin-flip parity, built from projected orbit
     states, and pools the levels of all momenta by parity (largest odd level for xi_spin, second even
     level for xi_energy). This differs from the reference (flip blocks only) and from the oracle (no
     matrix). bulk_thermodynamics carries the full docstring of the function_header and the dead
     helper is gone; its computation is unchanged.
   - Regression test: against the 60-digit Kaufman helper for every L from 3 to 10 at ten temperatures
     from 1 to 10 the new strip_lengths is within 2.2e-16 (f), 1.6e-9 (xi_spin) and 1.3e-15
     (xi_energy); crown_check second mode passes every case of every step and of the general tests.
6. Mutants rebuilt from the current reference (rule E).
   - Finding: three step mutants were not the current reference with one mistake:
     third_eigenvalue diagonalized the whole matrix (also changing f and xi_spin), finite_difference
     replaced the whole method, and first_complex_eigenvalue was a separate bisection code;
     correlation_one_row dropped the odd-sector diagonalization of the reference.
   - Cause: they were written against earlier solvers.
   - Change: third_eigenvalue is the reference with xi_energy taken from the third-largest eigenvalue of
     the pooled even and odd levels; correlation_one_row is the reference with its last line replaced
     by sqrt(G(1)); finite_difference is replaced by one_sided_correlation_sum (the reference with the
     fluctuation sum over r >= 0 only, lam_0 / (lam_0 - lam_k) instead of (lam_0 + lam_k) /
     (lam_0 - lam_k); file mutants/step_4_one_sided_sum.py); first_complex_eigenvalue is the reference
     (same bracket and brentq) whose root function watches the whole spectrum
     (1e-9 - max |Im lambda| / max |lambda|) instead of the leading pair. The finite-difference trap of
     step 4 stays documented, rerun, in the default-route table. The other mutants already were the
     current reference with one change (checked by diff).
   - Regression test: crown_check mut mode, results below.
7. Type checks of step 7 and of the general tests.
   - Finding: these cases required type(x) is float, stricter than the isinstance(x, float) used in
     steps 2 to 6, so a correct np.float64 would fail.
   - Cause: inconsistent style between steps.
   - Change: isinstance(x, float) everywhere; the prompt still asks for Python floats (np.float64 is a
     subclass of float). Values and tolerances unchanged.
   - Regression test: controls still pass no case.
8. Truthfulness of source.md and problem.yaml.
   - Finding: source.md said "No target is copied from the reference solution" while listing the
     reference as the method of steps 3 to 6; a stray paragraph split the default-route table (step 7
     rows rendered without a header) and named a test point, (2.0, 0.05), that no longer exists; it said
     "five warm-started CTMRG runs" (the code does one cold and eight warm-started runs) and gave a
     self-contradictory runtime sentence; the bond-dimension-28 agreement figures differed between
     source.md (2e-10 chi, 1.3e-8 c) and problem.yaml (2e-8); verification_explanation claimed a
     shift of "every reference output" while the checker then skipped complex outputs; the step 2 to 5
     default-route numbers had not been rerun on the current points; Version 9 items 4 and 5 lacked a
     cause and a regression line.
   - Cause: text carried over from earlier versions.
   - Change: the target table, the default-route table and the runtime paragraph above are rewritten
     with numbers measured in this version (earlier numbers that were not rerun are marked with their
     version); problem.yaml difficulty_explanation, solution_explanation ("one cold and eight
     warm-started runs") and verification_explanation now describe the current targets, methods and
     checks, with one set of bond-dimension-28 numbers (2.2e-15 f, 7e-14 m, 1.7e-10 chi, 1.2e-8 c).
     The shift check of tools/crown_check.py now scales real and complex outputs, and in the general
     cases it shifts every evaluated function of solution.py; the text says so. For Version 9 items 4
     and 5: the cause was missing metadata and stale background text, and their regression check was
     crown_check format (metadata keys present, no problem reported), rerun here.
   - Regression test: crown_check format reports no problem; every number quoted in the current-state
     sections was produced in this version by the scripts described there, except the marked
     version-9 measurements.
9. Unchanged, checked: both general cases call bulk_thermodynamics (the final function) and assert its
   four outputs against the independently confirmed targets above; step_dependencies match the earlier
   functions each prompt names (step 7 restates the energy and calls none); prompts and docstrings state
   the domain, not test points; relevant_experience stays the placeholder for the author.

10. Physics statements of the metadata limited to the part of the domain where they hold (found by
    the independent verifier after the first version of this pass-2 note).
    - Finding: three statements were written as general but hold only in part of the domain.
      (a) difficulty_explanation said that "the second and third eigenvalues of the whole matrix belong
      to the odd sector above Tc"; the description of the step 2 mutant third_eigenvalue ("above Tc that
      eigenvalue belongs to the odd sector") and its default-route row ("the third eigenvalue is odd
      above Tc") said the same. Just above Tc the third eigenvalue is the second even state, as at Tc.
      (b) difficulty_explanation said that "below Tc the two leading states are degenerate to machine
      precision"; the code comments of strip_magnetization in solution.py, solution/step_3.py,
      second_solution.py, mutants/step_3_correlation_one_row.py and mutants/whole_task_field_not_scaled.py
      said the same, and the step 4 contract called the gap "tiny" everywhere below Tc. (c)
      difficulty_explanation said that the leading pair must be tracked "since lower pairs leave the real
      axis first".
    - Cause: each statement generalized what was seen at a few test points (T = 3, 5, 10 for step 2;
      T = 0.5 and 0.6 at L = 10 for step 3; (6, Tc) and (8, 3.0) for step 5).
    - Measurements (this version; reference transfer_matrix, flip-sector blocks, single-threaded BLAS):
      (a) at 96 temperatures from 0.5 to 10 and every L from 3 to 10 the second eigenvalue of the whole
      matrix is the top odd state; the third is the second even state below a crossover temperature
      (brentq root of ln(odd_1) - ln(even_1)) of 2.045 at L = 3, 2.329 at L = 4 and 2.410, 2.431,
      2.432, 2.425, 2.416, 2.406 for L = 5 to 10, and an odd state above it. The mutant's relative
      error in xi_energy is 0 at (4, 2.3), (10, 2.4), (10, 1) and (8, Tc), 0.081 at (10, 2.5), and 0.22,
      0.74, 0.83 at the failing test points (4, 3), (6, 5), (3, 10). (b) The relative gap
      (lam_even0 - lam_odd0) / lam_even0 is 0 in double precision at (10, 0.5) (9e-18 at 50 digits from the step 3 target route, item 2),
      7.7e-15 at (10, 0.6), 1.0e-12 at (10, 0.7), 1.4e-8 at (10, 1.0), 2.6e-14 at (8, 0.5), 5.2e-7 at
      (8, 1.0), 2.0e-5 at (6, 1.0), 0.10 at (6, 2.2) and 1.2e-5 at (3, 0.5). (c) For L = 3 to 8 at
      T = 1.5, 2, Tc, 2.5, 3, 5 and 10 (42 points), on the fields k H_edge / 200, k = 1 to 199, a lower
      pair has an eigenvalue with |Im| > 1e-9 max |lambda| before the leading pair meets at 17 points:
      (4, 3), (5, 3), L = 6 at Tc, 2.5, 3 and 5, L = 7 at Tc to 10 and L = 8 at 2 to 10 (16 of them at
      or above Tc, 14 at L >= 6); at the other 25 points, among them every L = 3 point and every
      T = 1.5 point, no eigenvalue becomes complex before the edge.
    - Change: difficulty_explanation now says that the second eigenvalue of the whole matrix is the top
      odd state at every temperature and the third is the second even state below a width-dependent
      crossover (2.05 at L = 3, 2.33 at L = 4, 2.41 to 2.43 for L = 5 to 10; so at Tc and just above it
      for L >= 4) and an odd state above it; that below Tc the two leading states are nearly degenerate,
      with a gap that closes exponentially with the width and reaches the level of machine precision far
      below Tc at the larger widths (about 1e-14 relative or less for T <= 0.6 at L = 10); and that the
      leading pair must be tracked because at many points of the range, most of them at or above Tc on
      the wider strips, a lower pair leaves the real axis before the leading pair meets. The
      third_eigenvalue description and the default-route rows of third_eigenvalue and
      first_complex_eigenvalue (the latter now names the cases in which the mutant passes) say the same.
      The five code comments say "nearly degenerate ... to machine precision far below Tc at the larger
      widths" (comments only: the code of both solutions and of the mutants is unchanged, and each
      mutant still differs from the reference by its one mistake, checked by diff). The step 4 contract
      says that the gap closes exponentially with L (1.4e-8 relative at L = 10, T = 1). The step 3
      contract was already worded this way. No prompt, docstring, test, target or tolerance changed.
    - Regression test: crown_check format (no problem) and, because comments of the solution and
      mutant files changed, all modes again; results below.

Rerun after all items: tools/crown_check.py, all modes, every case alone in a fresh process, -j 2,
single-threaded BLAS; results are recorded below. solution.py and second_solution.py were each run twice
in clean single-threaded processes on the inputs of every test case (step 1 matrices compared by a
hash of their bytes); the two runs of each were identical.

crown_check result (version 10, all modes): FORMAT OK (no problem); REF 5/5, 5/5, 6/6, 6/6, 6/6,
3/3, 8/8 for steps 1 to 7 and 2/2 for general (solution.py also passes every step file); SECOND the
same counts, all pass; mutants: field_not_shared fails 4/5 cases [0, 1, 3, 4], third_eigenvalue 3/5
[0, 1, 4], correlation_one_row 4/6 [1, 3, 4, 5], one_sided_correlation_sum 6/6,
first_complex_eigenvalue 2/6 [0, 1], inverted_edge_ratio 3/3, missing_corner_normalization 8/8,
specific_heat_without_T 8/8, free_spin_susceptibility 8/8, whole_task_field_not_scaled 2/2 general
cases; SHIFT +-1e-12 (real and complex outputs; every evaluated function in the general cases) all
pass; the all-None and all-zero controls pass 0 cases in every step and in general; SUMMARY ALL OK
(7 min 12 s wall time with -j 2).

Rerun after item 10 (all modes, every case alone in a fresh process, -j 2, single-threaded BLAS):
the same result in every line, FORMAT OK with no problem; REF and SECOND 5/5, 5/5, 6/6, 6/6, 6/6,
3/3, 8/8 and 2/2 general; the mutants fail the same cases as above (first_complex_eigenvalue [0, 1],
third_eigenvalue [0, 1, 4]); SHIFT +-1e-12 all pass; both controls pass 0 cases; SUMMARY ALL OK
(7 min 30 s wall time). crown_check format was run once more after the last edit of this file.
