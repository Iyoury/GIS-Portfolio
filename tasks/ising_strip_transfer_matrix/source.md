# Source

- L. Onsager, "Crystal statistics. I. A two-dimensional model with an order-disorder
  transition", Phys. Rev. 65, 117 (1944). Exact critical temperature
  Tc = 2 / ln(1 + sqrt(2)) for J = k_B = 1.
- B. Kaufman, "Crystal statistics. II. Partition function evaluated by spinor
  analysis", Phys. Rev. 76, 1232 (1949). Exact eigenvalues of the transfer matrix of a
  periodic strip; used for the step 2 targets and in the second solution.
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

## How the test targets are obtained

No target is copied from the reference solution. Every expected value is written inline in
the tests and was produced by the method below; the column "check" gives the agreement with
an independent route.

| Target | Method | Check |
|---|---|---|
| M entries (step 1) | brute-force sum over the energy inside the test (entry by entry for L = 3, 4 and 6; over all pairs of rows, column by column, at L = 10); spot entries worked out by hand, including the ends of the stated range | reference and second solution within 3e-14 relative of a 40-digit sum over the energy at the corners of the range (L = 3, 4, 7, 10; T = 0.5, 1, Tc, 10; h = 0, 0.1, +-20, +-20i, 20 exp(i pi/3), 11i) |
| f, xi_spin, xi_energy (step 2) | Kaufman's exact strip spectrum, now computed inside the test file in double precision (no pinned numbers); xi_energy = 1 / (2 g_1), the two lowest fermion modes q = 1 and 2L - 1; tolerances 1e-10 (f) and 1e-6 (lengths) | reference within 1.7e-8 (xi_spin at L = 10, T = 1, limited by the 1.4e-8 relative gap), elsewhere within 1e-14 |
| m_L (step 3) | reference (separate spin-flip sector blocks) | second solution (rotation of the leading pair of the whole matrix into parity states) within 3e-16; coefficient of the slowest term of G(r) at 40 digits for L <= 6 within 1e-15 |
| chi (step 4) | reference (spectral sum over the odd states) | second solution (odd-sector linear solve) within 2.4e-9; 40-digit finite differences of ln lam0(h) for L <= 6 within 1e-13 |
| H_edge (step 5) | reference (root of the discriminant of the leading pair) | second solution (bisection on the leading pair leaving the real axis) within 1.3e-12; 40-digit bisection for L <= 4 within 1e-15; the leading pair was checked real and distinct below the edge on a fine grid for every L from 3 to 8 and T from 1.5 to 10 |
| final dict (step 6) | the formulas of step 6 applied to the values above | second solution within 1e-12 |
| bulk f, m, chi, c (step 7, general) | corner-transfer-matrix renormalization at bond dimension 28 (the reference uses 16), free energy from kappa = Z(3x3) Z(4 corners) / Z(corners + 2 edges)**2, chi from Richardson central differences of m, c from Richardson second differences of f | VUMPS (second solution: per-site eigenvalue of the row transfer matrix, chi = -d2f/dh2 and c from five-point stencils) within 4e-14 (f), 5e-11 (m), 6e-7 relative (chi) and 3e-8 relative (c) on all eight test points; at h = 0 the same free energy reproduces Onsager's to 1e-14 at T = 2, 2.5 and 3 |

## Why each step is not a direct transcription

Each step was run against the default that a practitioner would reach for first, on the
test points of that step. The tolerance in parentheses is the one stated in the prompt.

| Step | Default route | Error on the test points | Correct alternatives that pass |
|---|---|---|---|
| 2 (rel 1e-6) | xi_energy from the second eigenvalue of M | 1.1 to 6.8 (every case) | eigenvectors of M classified by spin-flip parity; sector blocks; free fermions |
| 2 (rel 1e-6) | xi_energy from the third eigenvalue of M | 0.22, 0.74, 0.83 at T = 3, 5, 10 (the third eigenvalue is odd above Tc) | as above |
| 3 (abs 1e-9) | matrix element between the two leading eigenvectors of one eigensolver call | 1.0 at (L, T) = (10, 0.5), 1.8e-3 at (10, 0.6), 3.3e-4 at (8, 0.5): the pair is degenerate to machine precision and the solver returns an arbitrary mixture | sector blocks; rotation of the pair into parity states |
| 3 (abs 1e-9) | Yang's bulk magnetization | 0.08 at (6, 2.2), 0.51 at (7, 4.0) | |
| 4 (rel 1e-6) | central second difference of f(h), step 1e-3 to 1e-7 | every step fails at least one case (relative error up to 0.99 below Tc, roundoff-dominated at Tc and above for the small steps); Richardson on 1e-4 and 5e-5 fails at 1.0 | spectral sum (with or without sector blocks); odd-sector linear solve |
| 4 (rel 1e-6) | correlation summed over r >= 0 only | 0.11 to 0.50 | |
| 5 (rel 1e-8) | first field at which any eigenvalue becomes complex | 0.32 at (6, Tc), 0.99 at (8, 3.0): degenerate lower pairs leave the real axis first | root of the discriminant of the leading pair; bisection on the leading pair |
| 5 (rel 1e-8) | linear scan of H from 1e-3 | factor 6.6 at (7, 1.5): the edge is 1.3e-4 below Tc | geometric bracketing |
| 5 (rel 1e-8) | Hermitian eigensolver on the complex symmetric matrix | factor 1.6 to 8.5e3 | general (non-Hermitian) eigensolver |

The strip routes pass only at the two test points far from the critical region
((T, h) = (2.0, 0.05) and (3.0, 0.2)), where the correlation length is about one lattice
spacing.

| 7 (f 1e-10, m 1e-9, chi and c rel 1e-5) | widest diagonalizable strip, or extrapolation of strips | m alone was already off by 1e-4 to 1e-2 at h = 0.02 near Tc; the new range goes down to h = 0.005 (longer correlation length) | corner-transfer-matrix renormalization or VUMPS with the correct per-site partition function and converged derivatives |
| 7 | partition function per site without dividing out the corner contribution | f off by about 0.3 | |
| 7 | specific heat as -d2f/dT2 (no factor T) | factor T | |
| 7 | susceptibility of free spins (1 - m**2) / T | factor 3 to 60 | |
| 7 | central differences with steps that are too large at h = 0.005 | chi off by 2e-5 relative and more (truncation error of a second difference with step 5e-4) | Richardson extrapolation or smaller steps on a well-converged f |

Every test case takes under 15 s with the reference solution (numpy 2.4.6, scipy 1.17.1), except the step 7 cases, which take up to about 15 s each (five warm-started CTMRG runs).

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
