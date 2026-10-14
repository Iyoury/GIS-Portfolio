# Source

- R. A. Fisher, The Genetical Theory of Natural Selection (Clarendon Press, 1930), and S. Wright,
  "Evolution in Mendelian populations", Genetics 16, 97-159 (1931). The Wright-Fisher model and
  the stationary distribution of the diffusion limit.
- M. Kimura, "On the probability of fixation of mutant genes in a population", Genetics 47,
  713-719 (1962), and M. Kimura and T. Ohta, "The average number of generations until fixation of a
  mutant gene in a finite population", Genetics 61, 763-771 (1969). Fixation probability and
  conditional fixation time.
- W. Feller, "Diffusion processes in genetics", Proc. Second Berkeley Symposium, 227-246 (1951),
  and C. Cannings, "The latent roots of certain Markov chains arising in genetics", Advances in
  Applied Probability 6, 260-290 (1974). Exact eigenvalues of the neutral chain.
- S. Karlin, Total Positivity, Vol. I (Stanford University Press, 1968). Real, positive, distinct
  eigenvalues of totally positive kernels.
- W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative analysis and steady state
  distributions for Markov chains", Operations Research 33, 1107-1116 (1985), and C. A. O'Cinneide,
  "Entrywise perturbation theory and error analysis for Markov chains", Numerische Mathematik 65,
  109-120 (1993). State reduction without subtraction, and its entrywise relative accuracy.
- W. J. Ewens, Mathematical Population Genetics I, 2nd ed. (Springer, 2004). Absorption
  probabilities and times of the Wright-Fisher chain, and the origin-fixation regime.

The combination of steps, the haploid life cycle (selection, then mutation, then sampling), the
parameter ranges and the accuracy requirements are our own design.

Checks used while building the task:
- Transition matrix: every entry of matrices up to N = 60 and selected rows at N = 300 and 400
  agree with mpmath binomials (50 digits) to about 1e-13 relative, including entries down to
  1e-250. Row sums, means N p_mut and variances N p_mut (1 - p_mut) are exact. For N = 1 the matrix
  is [[1 - v, v], [u, 1 - u]].
- Reference: subtraction-free elimination (state reduction) for the absorption probabilities and
  times, the stationary distribution, the relaxation rate (inverse iteration on a GTH-factored
  I - P, eliminating towards the most probable state) and the passage times. Second solution
  (versions 1-12; replaced for steps 2-6 in version 13 and for step 7 in version 14, see there): the
  same linear systems solved by plain LU in mpmath, with the precision doubled until two runs agree to
  1e-13. The two agree to about 1e-14 relative on all tested cases.

The version sections below describe the task as it was at that version; numbers and methods in an
older section are historical when a later section replaces them.
- Analytic checks:
  - neutral fixation probability i0 / N;
  - exact neutral first two stationary moments;
  - neutral relaxation rate u + v, down to 2e-12;
  - N = 1 closed forms;
  - exact allele-relabeling symmetry s -> -s / (1 + s), u <-> v, i -> N - i, which tests the
    results at N = 400 where they span more than 100 decades.
- Rare-mutation consistency:
  - the relaxation rate equals 1 / t_up + 1 / t_down to 2e-10;
  - t_up equals 1 / (N v p_fix(1)) to 2e-10;
  - pi[N] / pi[0] equals t_down / t_up to 4e-7.
- Dense linear algebra fails these tests:
  - np.linalg.solve for the passage times is off by 2e-5;
  - eig for the gap is off by 5e-3 at u + v = 2e-12;
  - the dense eigenvector for pi gives relative errors up to 1e10 on the small entries.

Version 2 (content-check fixes):
- Step 2 tests the smallest population N = 2 (neutral: p_fix = p_loss = 1/2, both conditional times 2
  generations).
- Step 4 tests u = 0.1 (neutral gap 0.11 for u = 0.1, v = 0.01).
- Step 5 tests v = 0.1 (N = 1: t_up = 1 / v = 10, t_down = 1 / u = 20).
- Step 4 prompt: the lambda_2**t decay is stated for a generic initial distribution (nonzero
  component along the second eigenvector).

Version 3 (difficulty increase, the quick probe solved v2 three times out of three):
- New step 6: the quasi-stationary distribution and absorption rate with one-way mutation (u = 0). The
  rate is down to about 1e-246 and is required to 1e-8 relative, with the distribution entrywise.
  - Reference: inverse iteration for the left Perron vector with a subtraction-free LU of I - Q. Each
    pivot is the sum of the remaining off-diagonal rates plus the accumulated absorption probability,
    and the iteration stops when every entry has settled, not only the rate.
  - Second solution: plain LU in mpmath with doubled precision.
  - Tests:
    - the neutral rate is exactly v, for any N (Feller-Cannings spectrum);
    - extended-precision oracle;
    - exact identities at N = 400: rate = sum_i q_i P[i, N] and (q Q)_j = (1 - rate) q_j;
    - the general test links it to step 5: rate * t_up = 1 to 8e-11.
- Steps 2-6 state a time budget of 20 s per call at N up to 400, asserted in the tests. The
  reference needs well under 2 s; extended-precision linear algebra at N = 400 does not fit.
- The prompts and background no longer describe the numerical remedy: no state reduction, no
  remark on dense eigensolvers or on diagonal entries close to 1.

Version 4 (content-check fixes):
- Independent N = 400 targets, computed once by plain Gaussian elimination with partial pivoting in
  mpmath at 400 digits (transition matrix built from mpmath binomials). They are written into the
  tests and compared at 1e-8 relative; the reference agrees with them to about 1e-13:
  - step 2: all four outputs for s = -1/3, i0 = 1 (p_fix about 3e-141) and for s = 0.5, i0 = 399;
  - step 3: 18 entries of pi for s = 0.5, u = v = 1e-12, from the largest down to about 6e-150;
  - step 5: t_up and t_down (about 8.7e149) for s = 0.5, u = v = 1e-12;
  - step 6: the rate (about 2.3e-246) and 15 entries of the distribution for s = -0.5, v = 1e-9,
    including its tail between 1e-250 and 1e-240.
- Steps 1 and 3: entries whose exact value is below 1e-250 are compared with that exact value
  (|got - exact| <= 1e-250), as the prompts state.

Version 5: the v4 rollouts (5/8) lost credit only on one step-6 check, the eigenvector equation at the
far tail of the N = 400, s = 0.5, v = 0.1 distribution (qsd[0] about 4e-221). The requirement was
already covered by the general accuracy rule; the prompt now states it explicitly, with that case and
the size of its tail, and says that every tail entry is checked against (qsd Q)_j = rho qsd_j.

Version 6: the step-6 sentence on the far tail is a behavioral requirement only (each tail entry
must itself have a relative error below 1e-8); it no longer describes how the tests check it.

Version 7: the stated far-tail requirement of step 6 (N = 400, s = 0.5, v = 0.1, qsd[0] about 4e-221)
is now checked against independent targets: inverse iteration on (I - Q)^T shifted to 0.3987, started
from a uniform vector, Gaussian elimination with partial pivoting in mpmath at 400 digits, run once
(converged to 1e-29 in every entry; the same result to 30 digits from a second start). The test asserts
the rate and the entries i = 0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 399 to a relative 1e-8, so a
solution that returns 0 (or any inaccurate value) in the tail fails. An unshifted inverse iteration from
a uniform start had not converged in the tail after 200 iterations (qsd[0] stuck near 1e-85), which is
why the shifted form is used for this target.

Version 8 (difficulty increase; v7 rollouts solved 8/8): new step 7, smallest_eigenvalue(N, s, u, v), the
smallest eigenvalue lambda_min of the transition matrix with two-way mutation, to a relative 1e-8 within
20 s. It is about 1e-173 to 1e-216 at N = 400, far below what a dense eigensolver resolves next to the
eigenvalue 1, and entrywise relative errors of P do not determine it (P is totally positive, its small
eigenvalues are ill-conditioned with respect to its entries). Reference: P = diag(q**N) V diag(C(N, j))
with V the Vandermonde matrix of the increasing nodes x_i = p_i / q_i; P^-1 has the checkerboard sign
pattern, so 1 / lambda_min is the Perron root of |P^-1|, whose entries
e_{N-j}(x without x_a) / (C(N, j) prod_{k != a}|x_a - x_k| q_a**N) are sums and products of positive
terms; the node differences come without cancellation from p_a - p_k = (1 - u - v)(1 + s)(a - k) /
(N (1 + s a/N)(1 + s k/N)); everything in logarithms; power method (ratio lambda_min / next eigenvalue,
about 1/N). Second solution: prefix/suffix convolution for the elementary symmetric functions and a
dense eigensolver after an Osborne-balancing diagonal similarity. Targets: the neutral closed form
(1 - u - v)**N N!/N**N (Feller; Cannings), mpmath eig at N <= 30 with selection, and inverse iteration with
a plain mpmath LU at 400 digits for (400, 0.5, 1e-12, 1e-12) -> 2.76656467220724976e-176 and
(400, -0.5, 0.1, 1e-12) -> 5.87144975069309366e-197 (reference within 3e-13). A dense eigensolver gives
1e-17 to 1e-31 for these (wrong by 100 to 190 decades); the mutants dense_eigenvalues and
numerical_inverse fail. General tests: one step-7 check at N = 18.

Version 9 (content-check fixes on v8): independent 400-digit targets (plain mpmath LU, run once) added for
fixation_statistics(400, -0.5, 1) (p_fix = 5.71042258020734e-240, all four values), for the stationary
distribution at N = 400, s = -0.5, u = v = 1e-12 (entries down to 1.3e-248, many between 1e-160 and 1e-250)
and for substitution_times(400, -0.5, 1e-12, 1e-12) (t_up = 4.37795967826815e248); the reference agrees to
about 1e-13. Comparisons between two computed outputs (the relabeling checks of steps 2, 3, 4, 5 and 7, the
origin-fixation link of step 5) use 2.1e-8, the identities of step 6 between rate and qsd 2.5e-8 and 3e-8,
the general rate * t_up link 2.5e-8; comparisons with independent targets stay at 1e-8. Step 1 transcription
advisory: no change (the binomial kernel is the definition of the model).

Version 10 (content-check fixes on v9, step 6): the complete quasi-stationary distribution, every entry, and the
rate are now compared with independent 400-digit targets (mpmath LU on (I - Q)^T, inverse iteration until every
entry changed by less than 1e-60 relative, run once) for (400, -0.5, 1e-9), (400, 0.5, 0.1), the lower-end case
(400, -0.5, 1e-12) (rate 2.28416905628920e-249) and (300, -0.2, 1e-12) (rate 1.32212623303458e-68); the
reference agrees to 4e-13 (rates) and 2e-12 (entries). The identity checks are gated on the target entries
(>= 1e-240), not on the submitted ones; a solution that zeroes tail entries now fails. Step 1 transcription
advisory: no change (the binomial kernel is the definition of the model).

Version 11 (difficulty increase; v10 rollouts solved 7/8): step 7 becomes fastest_mode(N, s, u, v), which returns
the smallest eigenvalue together with its right eigenvector (scaled to max |entry| = 1, first nonzero entry
positive), every entry with a relative error below 1e-8; at N = 400 the entries span up to 176 decades. The
domain now includes u = 0 and/or v = 0 (absorbing end states, eigenvalue 1 repeated, the mode exactly zero on
absorbing states; ValueError for N = 1 with u = v = 0). The hint "the binomial kernel with increasing p_mut is
totally positive" was removed from step 4 and from the step 7 contract. Reference: Perron pair of |P_II^-1| on
the transient block, P_II = diag(q^N x^i0) V(x_I) diag(C), entries from elementary symmetric functions and
cancellation-free node differences, power method in logarithms (about 0.6 s at N = 400). Second solution:
prefix/suffix convolutions, Osborne balancing, dense eigensolver, then log-space polishing of the vector;
the two agree to 2e-13 at N = 400. Targets: mpmath eig plus shifted inverse iteration with plain LU
(80-150 digits) for N <= 30, and inverse iteration with plain LU at 400 digits (run once, about 3 min each)
for (400, 0.5, 1e-12, 1e-12), (400, -0.5, 0.1, 1e-12), (400, 0.3, 0, 0) and (400, -0.4, 0, 0.05); the
reference agrees to 3e-13 (eigenvalues) and 8e-13 (entries). New mutants: regularized absorbing states
(u, v -> 1e-300) and sign fixed by the largest entry.

## Version 12 (grading_fix: authoring-guide conformance)

v11 is queued for rollouts on the platform; this version is prepared next to it and changes no science,
reference algorithm, target value or domain.

1. Test-case format and self-contained cases.
   - Finding: the markers read "# --- test case N: description ---", and every test file had a shared
     preamble (imports, the mpmath transition matrix, the extended-precision solvers, the check helpers)
     before case 0; in tests/general.py the helper _t_qsd sat between cases 2 and 3. A case run alone
     could not execute.
   - Cause: the files were written to be executed as a whole.
   - Change: markers are exactly "# --- test case N ---" (0..n-1) with the description as comments below;
     every case carries its own imports, helpers and setup (tools/selfcontain.py, no case uses a name
     defined only in another case); _t_qsd is now inside general case 3. The provenance comments stay as
     leading comments. No check was dropped; n_test_cases unchanged (5, 4, 5, 5, 5, 5, 6; general 5).
   - Regression test: crown_check format (markers, n_test_cases, no code before case 0) and every case
     run alone in a fresh process.
2. Per-call time budget.
   - Finding: the prompts of steps 2-7 said "Each call must finish within 20 s on one CPU core", and
     tests/step_2..7 wrapped the evaluated function in perf_counter timing asserts.
   - Cause: the budget (version 3) was meant to exclude extended-precision brute force.
   - Change: the sentence is removed from the six prompts (no size statement was attached to it; the
     domain statements stay) and the wrappers and "import time" are removed from the tests. The N = 400
     cases remain as value tests.
   - Regression test: crown_check format (no clock reads); all cases still pass for the reference.
3. Exact zeros in step 7.
   - Finding: the prompt required "entries of the mode that are exactly zero must be returned as 0.0",
     and the tests asserted mode[N] == 0.0, mode[0] == 0.0 and compared zero targets with a zero
     tolerance (|got - 0| <= 1e-8 * 0).
   - Cause: exact floating-point equality on computed outputs.
   - Change: step_description_prompt, function_header, problem_io and the docstrings of steps/step_7.py,
     solution/step_7.py, solution.py and the step-7 mutants now state that on the absorbing states the
     exact entries of the mode are zero and the returned entries must be below 1e-250 in absolute value
     (0.0 is accepted). 1e-250 is the floor already used in steps 1, 3 and 6, and it separates these
     entries from every true nonzero entry: over a grid of s in {-0.5, -0.2, 0, 0.2, 0.5} and u, v in
     {0, 1e-12, 0.1} at N = 400 the smallest nonzero |entry| of the reference mode is 3.3e-176. The sign
     convention "first nonzero entry is positive" became "first entry that is not on an absorbing state
     is positive", which is the same for the exact mode but stays unambiguous once tiny nonzero values
     are accepted on absorbing states. The tests build the absorbing-state mask from u and v
     (_t_absorbing_mask), require |entry| <= 1e-250 there and the relative 1e-8 (2.1e-8 for the
     relabeling comparison) elsewhere; the relabeling case fixes the sign of the reversed mode by its
     first entry off the absorbing states.
   - Regression test: tests/step_7.py cases 0-4; the reference and the second solution pass, the
     mutant of item 4 fails.
4. Mutant regularized_absorbing.
   - Finding: with the 1e-250 bound, replacing u = 0 or v = 0 by 1e-300 leaves the absorbing-state
     entries near 1e-300, inside the bound; that mutant still failed, but only because its own sign
     normalization picked those tiny entries (checked by running it on every case of tests/step_7.py).
   - Change: the regularization is now 1e-15 (a tiny mutation rate used instead of treating the absorbing
     states), and its description in problem.yaml says so.
   - Regression test: crown_check mut: it fails cases 0-4 of tests/step_7.py.
5. Size of tests/step_7.py.
   - Finding: the four N = 400 eigenvectors were stored in full (4 x 401 entries, about 44 KB).
   - Change: each now stores the entries i = 0, 8, 16, ..., 400 and the entries next to an absorbing state
     (51 to 53 per vector) as "index:value" strings, identical to the stored 400-digit-run values (checked
     string by string against v11), and the candidate entries at those indices are compared. The file went
     from 50.3 KB to 19.7 KB. The 400-digit provenance comment is kept.
   - Regression test: case 3 of tests/step_7.py; reference, second solution and shift check pass, the
     dense_eigenvalues, numerical_inverse, regularized_absorbing and largest_entry_positive mutants fail it.
6. Tolerances in tests/general.py that did not match the stated accuracy.
   - Finding: case 0 (gap = 1 / t_up + 1 / t_down) and case 2 (t_up = 1 / (N v p_fix)) compare two
     computed outputs, each allowed 1e-8, plus a rare-mutation link that holds to about 2e-10, with 1e-8;
     case 4 required the residual |P mode - lam_min mode| <= 1e-12 max(|P| |mode|) and |max |mode| - 1| <
     1e-12, while the stated accuracy of every entry of the mode is 1e-8 relative (a mode with entries
     correct to 5e-9 meets the specification and fails 1e-12).
   - Change: 2.1e-8 for the two links (as in step 5, case 3), 2.1e-8 for the residual (bound about
     1.01e-8 from the stated accuracies of mode, lam_min and P) and 1e-8 for the normalization.
   - Regression test: general cases 0, 2 and 4 for the reference and the second solution; the whole-task
     mutant still fails them.
7. Whole-task mutant incomplete.
   - Finding: mutants/whole_task_mutation_before_selection.py defined only the functions of steps 1-5,
     so general cases 3 and 4 failed by NameError rather than by the scientific error.
   - Change: it now also defines quasi_stationary (the reference algorithm on its mutated matrix) and
     fastest_mode with the mutation-before-selection frequencies (selection acting on the mutated
     frequency, node differences (1 + s)(m_a - m_k) / ((1 + s m_a)(1 + s m_k))); at N = 5-7 its eigenpair
     is exact for its own matrix (residual about 1e-16) and differs from the reference.
   - Regression test: crown_check mut: it fails all 5 general cases.
8. Metadata: subfield, tags, expert_time_estimate_hours, the relevant_experience placeholder for the
   author, difficulty_explanation, solution_explanation, verification_explanation, author and
   affiliation added; edit_label grading_fix.

Rerun: tools/crown_check.py on the whole folder, all modes, every case alone in a fresh process (results
below), and solution.py (N = 400 inputs of every step) and second_solution.py (N = 24) run twice in
clean processes, solution.py also with one BLAS thread: bit-identical outputs.

crown_check results (each case alone in a fresh process, -j 2):
- format: OK (relevant_experience is a placeholder for the author to fill in).
- ref: every step 5/5, 4/4, 5/5, 5/5, 5/5, 5/5, 6/6 and general 5/5 pass; solution.py also passes every
  step file.
- second: step 1 5/5, step 7 6/6 and general 5/5 pass; steps 2-6 pass 2/4, 4/5, 3/5, 4/5, 3/5, and the 8
  other cases (step 2 cases 0 and 2, step 3 case 3, step 4 cases 1 and 3, step 5 case 2, step 6 cases 1
  and 3) stop at the checker's 900 s per-case limit: they call steps 2-6 several times at N = 250 to 400,
  and the second solution's plain LU in pure-Python mpmath at up to several hundred digits (precision
  doubled until two runs agree) needs much longer there. This limitation exists since the large-N cases
  were added (versions 3 and 4); the 20 s timing asserts used until version 11 would have failed these
  cases as well (each of them takes more than 900 s for at most 21 calls).
  At that size the analytic results, the relabeling symmetry and the stored 400-digit targets are the
  independent checks of the reference.
- mut: every step mutant fails at least one case (step 1: [0, 2, 3] and [0, 1, 2, 3]; step 2: [1, 2] and
  [1, 2]; step 3: [0, 2, 3]; step 4: [1, 2, 3] and [2]; step 5: [0, 1, 2, 3]; step 6: [1, 2, 3] and
  [1, 2, 3]; step 7: [1, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4] and [0, 1, 2, 3, 4]); the whole-task
  mutant fails general [0, 1, 2, 3, 4].
- shift: every reference output scaled by 1 + 1e-12 and by 1 - 1e-12: every case passes.
- controls: all-None and all-zero stubs pass no case of any step or of the general tests.

## Version 13 (grading_fix: authoring-guide conformance)

Prepared next to the queued v11 like version 12. No reference algorithm, target value, tolerance, test case or
mutant changed; the second solution of steps 2-6 is new, and the domain statement of step 5 excludes the inputs
whose result is not representable.

1. Second solution of steps 2-6 never ran on the large-N cases.
   - Finding (independent verification of version 12): crown_check second stopped 8 of the 24 step 2-6 cases
     at its 900 s per-case limit (step 2 cases 0 and 2, step 3 case 3, step 4 cases 1 and 3, step 5 case 2,
     step 6 cases 1 and 3), so the alternative method was never compared with any case at N >= 250, not even
     the analytic neutral case of step 2 (N = 300). The guide's self-review item "a valid alternative passes"
     was unverified there.
   - Cause: steps 2-6 of second_solution.py solved every linear system by plain LU in pure-Python mpmath with
     the precision doubled until two runs agreed; the cost grows like N^3 times the number of digits
     (fixation_statistics(50, -0.5, 1) took 2.5 s, at N = 100 about 20 s, at N = 400 of the order of an hour
     per call), so splitting the cases could not help.
   - Change: steps 2-6 of second_solution.py now use a double-precision regenerative (hub) decomposition, a
     different method from the reference's state reduction (no state is eliminated one at a time, no pivot is
     formed):
     - hub states: 0, N and the grid state nearest to the stable equilibrium of the deterministic map
       p -> p_mut(p) (where p_mut(p) - p changes sign from + to -); outside the hubs (and the absorbing
       states) the chain reaches a hub or an absorbing state within O(N) generations on average;
     - the fundamental matrix G = sum_k K^k of the chain restricted to the other states is its Neumann
       series, summed by repeated squaring (G <- G + K^(2^m) G, K^(2^(m+1)) = (K^(2^m))^2, 7 to 14 doublings
       at N = 400, at most 20 allowed, otherwise an error is raised): only products and sums of nonnegative
       numbers;
     - the skeleton chain on the 1 to 3 hubs (jump probabilities P_HH + P_HF G P_FH between distinct hubs,
       exit probabilities P_H,t + P_HF G P_F,t) is solved by the Markov chain tree theorem: its stationary
       law as sums over spanning trees, its fundamental matrix (I - S)^-1 by the all-minors forest formula;
       1 - S_kk is never formed;
     - step 2: h = G P[., N], G P[., 0] over the interior states (no hub needed without mutation) and the
       conditional times (G h) / h; step 3: pi_H from the tree theorem, pi_F = pi_H P_HF G; step 5: mean
       passage times from the forest inverse of the skeleton killed at the target; steps 4 and 6: inverse
       iteration (as the reference, which is the natural eigen-iteration) whose linear solves are done by
       this decomposition (step 4 grounded at the most probable hub, step 6 a left solve with exit at N);
     - the transition matrix for steps 2-6 is built with exact integer binomial coefficients (relative
       error about 1.5e-13 against 60-digit mpmath rows at N = 400; the second solution's own step-1
       function, a ratio recursion, has about 4e-12, which is within the step-1 tolerance but leaves less
       margin after propagation).
     Steps 1 and 7 of the second solution are unchanged.
   - Regression test: crown_check second, every case alone in a fresh process: all 40 step cases and 5
     general cases pass (about 40 s in total, previously more than two hours with 8 time-outs). Beyond the
     tests, the second solution was compared with the reference on a grid (N in {1, 2, 5, 20, 60, 150, 400}
     for steps 3-6 with s in {-0.5, -0.2, 0, 0.2, 0.5} and u, v in {1e-12, 1e-6, 1e-3, 0.1}; N up to 400 and
     three i0 for step 2) and on 120 random parameter sets (N in [1, 400], s uniform, u and v log-uniform in
     [1e-12, 0.1], with extra weight on u = 0.1 and v = 1e-12): largest relative difference 2.1e-12 (step 2),
     5.2e-12 (step 3), 4.8e-12 (step 4), 6.0e-12 (step 5) and 5.2e-12 (step 6) on the grid and 2.2e-11 over the
     random sets (entries below 1e-250 within 1e-250); the doubling bound was never reached.
2. Passage times above the double-precision range.
   - Finding: the grid of item 1 showed that substitution_times(400, -0.5, 0.1, 1e-12) has t_up above the
     largest double (about 1.8e308): the reference returns nan there, the second solution inf. The step-5
     prompt nevertheless promised a relative error of 1e-8 for every input in the stated ranges.
   - Cause: the domain statement did not exclude results that cannot be represented.
   - Change: the step-5 prompt, its function_header, the contract (valid_input_ranges) and the docstrings of
     steps/step_5.py, solution/step_5.py, solution.py, second_solution.py, mutants/step_5_dense_solve.py and
     mutants/whole_task_mutation_before_selection.py state that inputs for which t_up or t_down would exceed
     1e300 generations are outside the domain, and that within the ranges this happens only to t_up near the
     corner N = 400, s = -0.5, u = 0.1, v = 1e-12. A scan with the second solution over N in {300, 330, 360,
     380, 390, 400}, s in {-0.5, -0.48, -0.46, -0.44, -0.4, 0.4, 0.5}, u in {0.05, ..., 0.1}, v in {1e-12,
     1e-6, 1e-3, 0.01, 0.1} and the swapped (u, v) found t_up >= 1e300 only for N >= 380, s <= -0.48,
     u >= 0.07, v <= 1e-3, and every other value below 9.2e299 (t_down never above 1e290). No test uses these
     inputs (the largest tested time is 4.4e248), so no test changed.
     (Superseded by version 14, item 10: the visible text no longer names the corner, which is a step-7 test
     input, and finer scans place the region at N >= 376, s <= -0.467, u >= 0.068, v <= 2.54e-3; the v bound of
     this coarse grid, 1e-3, was too small.)
   - Regression test: crown_check format and ref (unchanged results).
3. Metadata: verification_explanation describes the new second solution and that it passes every case;
   edit_label stays grading_fix.

Rerun: tools/crown_check.py on the whole folder, all modes, every case alone in a fresh process (-j 2), and
solution.py and second_solution.py (13 calls covering every step, up to N = 400) run twice in clean processes
and once with one BLAS thread: both are bit-identical between the two clean runs; solution.py is also
bit-identical with one BLAS thread, second_solution.py within 6e-16 relative of its multithreaded result (BLAS
summation order in the matrix products).

crown_check results:
- format: OK (relevant_experience is a placeholder for the author to fill in).
- ref: every step 5/5, 4/4, 5/5, 5/5, 5/5, 5/5, 6/6 and general 5/5 pass.
- second: every step 5/5, 4/4, 5/5, 5/5, 5/5, 5/5, 6/6 and general 5/5 pass (no time-out; version 12 had 8).
- mut: every step mutant fails at least one case (step 1: [0, 2, 3] and [0, 1, 2, 3]; step 2: [1, 2] and
  [1, 2]; step 3: [0, 2, 3]; step 4: [1, 2, 3] and [2]; step 5: [0, 1, 2, 3]; step 6: [1, 2, 3] and
  [1, 2, 3]; step 7: [1, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4] and [0, 1, 2, 3, 4]); the whole-task
  mutant fails general [0, 1, 2, 3, 4].
- shift: every reference output scaled by 1 + 1e-12 and by 1 - 1e-12: every step case passes.
- controls: all-None and all-zero stubs pass no case of any step or of the general tests.

## Version 14 (grading_fix: authoring-guide conformance, pass 2)

Each item lists the finding, its cause, the change and the regression test; what was rerun follows at the end.
No tolerance was loosened.

1. Whole-task tests that never called the final function.
   - Finding: tests/general.py cases 0-3 tested steps 1-6 only; fastest_mode appeared only in case 4, whose mode
     check was a residual at 2.1e-8 max(|P| |mode|). With lambda_min (1.3e-8) below that bound, a mode
     contaminated by 0.1 %, 1 % or 5 % of the next eigenvector passed it (checked with the eigenvectors of the
     matrix in 100-digit mpmath).
   - Cause: the general cases were written before step 7 existed (versions 3-9) and were not extended when step 7
     became the integrated result.
   - Change: every general case now calls fastest_mode and asserts a scientific property of its result.
     Case 0 (N = 20, s = 0.3, u = v = 1e-12): lambda_min and every entry of the mode against mpmath eig plus shifted
     inverse iteration with plain LU at 80 digits (1e-8), 0 < lambda_min < 1 - gap (step 4), and the stationary
     distribution of step 3 orthogonal to the mode. Case 1 (N = 18, s = -0.4, u = 1e-12, v = 1e-8):
     0 < lambda_min < 1 and the same orthogonality with pi. Case 2 (N = 16, s = 0.2, u = v = 0, the chain of
     fixation_statistics): lambda_min and the mode against the 80-digit eigenpair, the mode within 1e-250 of zero on
     both absorbing states. Case 3 (N = 20, s = -0.5, u = 0, v = 1e-12): the quasi-stationary distribution of step 6
     orthogonal to the mode on the transient states, |mode[N]| <= 1e-250, 0 < lambda_min < 1 - rate; the stale
     comment "rate about 1e-28" now gives the actual 6e-23. Case 4: every entry of the mode against the 100-digit
     eigenvector at 1e-8 (the residual check stays as an extra check).
     The orthogonality identities are exact: pi P = pi (or q Q = rho q) and P x = lambda_min x with
     lambda_min != 1 (!= rho) give sum_i pi_i x_i = 0. With every entry of pi or q of at least 1e-250 within 1e-8
     relative (smaller entries within 1e-250) and every entry of the mode within 1e-8 relative, the computed sum
     lies within (2e-8 + rounding) sum_i pi_i |x_i| + 1e-250 sum_i |x_i| of zero; the tests use 2.1e-8, as for two
     computed outputs. The reference meets them to 1.3e-15, 6.4e-16 and 1.7e-16 (cases 0, 1, 3).
   - Regression test: crown_check format reports no general case without a call of fastest_mode; the contaminated
     modes above now fail case 4.
2. Prompts that named hidden test inputs, and inaccurate magnitude statements.
   - Finding: the prompts quoted exact test tuples: step 2 (N = 400, s = -1/3, i0 = 1 and N = 400, s = 0.5,
     i0 = N - 1), step 4 (u = v = 1e-12 without selection), step 5 (N = 400, s = 0.5, u = v = 1e-12, also in the
     function_header and every docstring of substitution_times), step 6 (N = 400, s = -0.5, v = 1e-9 and N = 400,
     s = 0.5, v = 0.1 with qsd[0]), step 7 (N = 400, s = 0, u = v = 0.1). Several magnitudes were inaccurate: step 4
     said the gap goes down to about 1e-12 (it is 2e-12), step 6 said 1e-246 for a rate of 2.3e-246, step 7 said
     about 1e-173 for N!/N^N = 9.6e-173 and "of similar size with selection" although lambda_min reaches 1.4e-225,
     and difficulty_explanation gave lambda_min between 2e-211 and 1e-172 and passage times "up to about 4e248"
     (the domain reaches 1e300).
   - Cause: the sentences were added in versions 4-11 to make the accuracy requirement concrete, using the cases
     at hand.
   - Change: the prompts now state regimes and domain-wide magnitudes only: step 2 "with strong selection at
     N = 400 the smaller of the two probabilities can be far below 1e-100 (p_fix comes down to about 6e-240 within
     the stated ranges)"; step 4 "with rare mutation in both directions the gap ... comes down to about 2e-12";
     step 5 (prompt, function_header and the docstrings of steps/, solution/, solution.py, second_solution.py and
     the two mutants that carry it) "above 1e248 generations for some inputs at N = 400"; step 6 "comes down to
     about 2e-249 within the stated ranges" and "at N = 400 the distribution can span more than 200 decades";
     step 7 "at N = 400 it lies between about 1e-225 and 1e-172 over the stated ranges of s, u and v ... the entries
     of the mode span up to about 175 decades", and "(as stated in step 4)" instead of "(step 4)" so that it does
     not read as a dependency. background.md: the Kimura example uses N s = -100 (below 1e-80) instead of the
     tested corner N s = -200. difficulty_explanation corrected accordingly.
   - Evidence for the new numbers: lambda_min and the span of the mode over s in 41 values of [-0.5, 0.5] and
     u, v in {0, 1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.03, 0.06, 0.08, 0.1} at N = 300, 350, 380, 400 with the
     second solution: at N = 400 the smallest lambda_min is 1.357e-225 (s = -0.5, u = v = 0.1) and the largest span
     175.5 decades (s = -0.5, u = 0.1, v = 1e-12); the reference on 11 values of s and u, v in
     {0, 1e-12, 1e-6, 1e-3, 0.03, 0.1} at N = 400 gives the same extremes and the largest lambda_min, 9.603e-173 =
     N!/N^N (s = 0, u = v = 0); the minimum gap over N in {1, 2, 3, 5, 10, 40, 150, 400}, 11 values of s and
     u, v in {1e-12, 2e-12} is about 2.0e-12 (1.99999999999947e-12, neutral); the step-6 rate at s = -0.5,
     v = 1e-12 is 2.1e-237, 9.1e-249 and 2.28e-249 for N = 380, 399, 400; p_fix = 5.7e-240, t_up = 4.4e248 and the
     221-decade quasi-stationary distribution are the stored 400-digit targets.
     (Later correction, item 9: this scan had no positive u or v below 1e-12, which the step-7 domain then still
     allowed and where the span statement was false; item 9 restricts the domain and repeats the scan.)
   - Regression test: crown_check format; an AST comparison of every docstring in steps/, solution/, solution.py,
     second_solution.py and mutants/ with the function_header docstrings (31 files, no mismatch; problem_io equals
     the step-7 header).
3. Reference defect in fastest_mode at N = 1 with small u + v.
   - Finding: comparing the reference with mpmath eig plus inverse iteration for N = 1..8 showed wrong results at
     N = 1 when u + v is small, inside the domain: (u, v) = (1e-6, 1e-12) gave the mode (0.693, -1) instead of
     (1e-6, -1) and lambda_min with a relative error of 1.7e-6; (1e-12, 3e-12) gave (1, -1) instead of (1, -1/3);
     the v13 second solution also failed there (mode entry 1.000033e-6 instead of 1e-6).
   - Cause: the power method on |P_II^-1| contracts like the ratio of the two eigenvalues, 1 - u - v, and its
     stopping rule (vector change below 1e-11 per iteration) is met long before convergence when the contraction
     is that slow; for N >= 2 the ratio is at most about 1/2.
   - Change: for N = 1 with u, v > 0 the reference returns the exact 2 x 2 eigenpair, lambda_min = p_1 - p_0 =
     1 - u - v with the right eigenvector (p_0, -q_1) = (v, -u), in solution/step_7.py, solution.py, the two step-7
     mutants built from the reference (largest_entry_positive, regularized_absorbing) and the whole-task mutant
     (there with its own mutation-before-selection frequencies).
   - Regression test: tests/step_7.py case 0 adds N = 1 with (s, u, v) = (0.2, 1e-6, 1e-12) and
     (-0.4, 2e-12, 5e-12) against the exact eigenpair; the v13 reference and the v13 second solution fail it, the new
     reference passes. After the fix the reference agrees with the 60-digit mpmath eigenpair to 1e-11 or better for
     N = 1..8, s in {-0.5, -0.1, 0, 0.3, 0.5}, u, v in {0, 1e-12, 1e-6, 0.1}, except N = 1 with u = v = 1e-12 where the
     60-digit oracle itself is wrong (eigenvalues 1 and 1 - 2e-12); at 100 digits it gives (1, -1), as the reference.
4. Second solution of step 7.
   - Finding (round-1 verification, advisory): the step-7 second solution used the reference's reduction (the
     Perron root of |P_II^-1| built from elementary symmetric functions and node differences) and differed only in
     how the e_k were accumulated and in the eigen-solve; its docstring was a one-line summary.
   - Change: new method. Inverse iteration P_T y = r on the transient block, every solve through
     P_T = diag(q^N x^lo) V(x) diag(C(N, j)) with the Vandermonde factor solved by the Bjorck-Pereyra algorithm
     (Newton divided differences, then conversion to monomial coefficients). With positive increasing nodes and an
     alternating right-hand side every update combines two numbers of the same sign, so the solve runs in
     log-magnitude arithmetic with known signs and no cancellation; no inverse matrix and no elementary symmetric
     function is formed. A single transient state is its diagonal entry, and N = 1 with u, v > 0 uses the 2 x 2
     eigenpair (inverse iteration would contract only like 1 - u - v there). The function carries the full step-7
     docstring. What it still shares with the reference is the factorization into diagonal and Vandermonde factors
     and the cancellation-free node differences, i.e. the structure of the model; verification_explanation says so.
   - Regression test: crown_check second (all cases pass). Against the stored 400-digit targets of tests/step_7.py
     case 3 the new second solution is within 1.0e-12 (lambda_min) and 1.0e-12 (entries), the reference within
     2.6e-13 and 7.9e-13. Grid and random comparison with the reference: N in {1, 2, 3, 5, 20, 60, 150, 400},
     s in {-0.5, -0.2, 0, 0.2, 0.5}, u, v in {0, 1e-12, 1e-6, 1e-3, 0.1}: largest relative difference 8.5e-12 (at
     N = 2, where both iterations stop at changes of about 1e-11 and contract by about 1/2); 120 random sets (N
     uniform in [1, 400], s uniform, u and v log-uniform in [1e-12, 0.1], each set to 0 with probability 0.2):
     1.4e-12; both solutions return exact zeros on the absorbing states. At N = 400 it takes about 0.1 s per call
     (reference about 1.5 s).
5. Step 7, test case 3 (compacted N = 400 eigenvectors).
   - Finding (advisory): the version-12 compaction kept i = 0, 8, 16, ..., 400, so for the two vectors without
     absorbing states only even indices remained, all of the same sign in a checkerboard mode; the sign alternation
     and the odd entries at N = 400 were no longer checked.
   - Change: the stored entries are now i = 0, 5, 10, ..., 400 (both parities, 40-42 odd indices per vector) plus
     the entries next to an absorbing state, taken from the full version-11 400-digit vectors (git history) and
     checked string by string against the entries kept in version 12. tests/step_7.py is 23.7 KB.
   - Regression test: case 3 passes for the reference, the second solution and the shift check; the dense,
     numerical-inverse, regularized and largest-entry mutants fail it.
6. Step 4: the neutral_gap mutant was rejected by one case only.
   - Finding (advisory): gap = u + v for every s failed only case 2; the relabeling check of case 3 cannot see it and
     no N = 400 gap with selection was compared with an independent value.
   - Change: case 3 adds relaxation_rate(400, 0.5, 1e-12, 3e-12) against an independent target,
     6.99071848429455715699225366235e-10: the matrix from mpmath binomials at 400 digits, pi from a plain LU solve of
     the normalized system, inverse iteration on I - P + 1 pi^T with one plain LU factorization (scaled partial
     pivoting), converged to 60 digits (run once, 534 s). The reference agrees to 8.3e-15, the second solution to
     1.9e-13. The same script at N = 30 (80 digits) agrees with the reference to 6e-15.
   - Regression test: crown_check mut: neutral_gap now fails cases [2, 3].
7. Step 5, test case 3 tolerance.
   - Finding (advisory): the case compares one computed output (t_up) with an extended-precision value
     1 / (N v p_fix) and used 2.1e-8, justified by a comment with an arithmetic slip (1e-8 + 2e-10 is 1.02e-8, not
     2.1e-8); 2.1e-8 is the tolerance for two computed outputs.
   - Change: 1.05e-8 (1e-8 for t_up plus the origin-fixation discrepancy, measured 1.73e-10 here), comment
     corrected. This is a consistency fix of the error budget, not a tuning change: the reference is at 1.7e-10.
   - Regression test: crown_check ref/second/shift on tests/step_5.py.
8. Metadata and truthfulness.
   - difficulty_explanation (magnitudes of item 2, about 4 minutes per 400-digit factorization as measured for item
     6), solution_explanation (the N = 1 eigenpair of item 3) and verification_explanation (the whole-task identities
     of item 1, the new step-7 second solution and what it shares with the reference, the step-4 target, the shift
     check now covering the whole-task tests with every evaluated function scaled) were rewritten to match the code
     and tests; the unmeasured "about 40 s" timing claim was removed. The top of this file marks the version sections
     as historical where later versions replace them. relevant_experience stays the placeholder for the author;
     edit_label stays grading_fix.
   - Not changed, raised for the lead: removing the per-call time budget in version 12 makes plain arbitrary-precision
     linear algebra a valid but slow route (several minutes per factorization at N = 400); this is a design risk for
     the rollouts' overall time limit, not a rule violation. expert_time_estimate_hours = 14 is an estimate for the
     author to confirm.

9. Step 7: positive mutation rates below 1e-12, and the sign rule of the reference (independent verification of
   pass 2, blocking).
   - Finding: step 7 promised every u, v in [0, 0.1], hence any tiny positive rate. The entry of the mode at the end
     fed by a tiny rate scales like that rate: at N = 400, s = -0.5, u = 0.1 the reference gives mode[0] = 3.27e-176,
     3.27e-177, 3.27e-264 and 3.27e-314 (subnormal) for v = 1e-12, 1e-13, 1e-100 and 1e-150. So (a) the span of the
     mode is 176.5 decades already at v = 1e-13 and 263.5 at v = 1e-100, not "up to about 175"; (b) entries below
     the normal double range cannot carry the stated 1e-8 relative accuracy; (c) at v = 1e-200 and 1e-300 mode[0]
     underflows, the rule "divide by the sign of the first nonzero entry" then read mode[1] (negative) and the
     reference returned every entry with the wrong sign (mode[0] = -0.0, mode[1] = +8.164e-167, mode[400] = -0.90895
     instead of +3e-364, -8.164e-167, +0.90895; relative error 2). The second solution, which sets the signs by
     position, kept them right (mode[0] = 0.0). The item-2 scan used u, v in {0, 1e-12, 1e-9, ...} only, and no test
     had 0 < u or v < 1e-12, so this part of the domain had never been checked.
   - Cause: the domain of version 11 extended the lower bound 1e-12 of steps 3-6 down to 0 in order to include the
     absorbing cases, which also admitted arbitrarily small positive rates; and the reference read the sign from the
     computed entries (np.nonzero(mode)[0][0]) instead of from the position of the first transient state.
   - Change: (a) u and v are each either 0 or in [1e-12, 0.1], the lower bound of steps 3-6 and of
     problem_description_main ("down to 1e-12"); ValueError for 0 < u < 1e-12 and for 0 < v < 1e-12. This is stated
     in the step-7 prompt (domain sentence, Inputs and Errors), the function_header and problem_io, the contract
     (assumptions and valid_input_ranges) and the docstrings of steps/step_7.py, solution/step_7.py, solution.py,
     second_solution.py, the four step-7 mutants and the whole-task mutant, and the check is added to the validation
     of the eight implementations. (b) solution/step_7.py and solution.py no longer divide by the sign of the first
     nonzero entry: the checkerboard sign array is + on the first transient state I[0] by construction, which is the
     stated normalisation, so the sign is set by position. Mutants rebuilt from the current reference:
     regularized_absorbing and the whole-task mutant carry the same change; largest_entry_positive keeps its single
     mistake (flip so that the largest entry is positive, now marked as such); dense_eigenvalues and numerical_inverse
     now normalise by the sign of the first entry off the absorbing states, as stated, instead of the first nonzero
     entry, so that their only mistake is the dense eigen-solve. solution_explanation says how the sign is set.
   - Re-scan on the restricted domain: second solution on N in {200, 300, 350, 380, 400}, 11 values of s in
     [-0.5, 0.5] and u, v in {0, 1e-12, 2e-12, 1e-11, 1e-9, 1e-6, 1e-3, 0.03, 0.1} (4455 inputs), reference on
     N = 400, s in {-0.5, -0.25, 0, 0.25, 0.5}, u, v in {0, 1e-12, 1e-6, 0.1}: lambda_min at N = 400 between
     1.357e-225 (s = -0.5, u = v = 0.1) and 9.603e-173 (s = 0, u = v = 0), largest span 175.49 decades and smallest
     entry off the absorbing states 3.27e-176 (both at s = -0.5, u = 0.1, v = 1e-12, the lower bound of v, since the
     end entry is proportional to the rate), every mode with max |entry| = 1, + on the first transient state,
     strictly alternating signs and |entry| <= 1e-250 on the absorbing states. The prompt's "between about 1e-225
     and 1e-172" and "up to about 175 decades" stay true; the 1e-250 bound on the absorbing states stays about 74 decades
     below the smallest true entry.
   - Regression test: tests/step_7.py case 5 adds (10, 0.1, 1e-13, 0.01), (10, 0.1, 0.01, 5e-13),
     (10, 0.1, 0, 1e-300), (400, -0.5, 0.1, 1e-200) and (3, 0.2, 9.9e-13, 0), each required to raise ValueError; the
     previous reference fails case 5 (no error at (10, 0.1, 1e-13, 0.01)), the new reference and the second solution
     pass. Inside the domain the new reference is bit-identical to the previous one (lam_min, mode and the signs of
     zeros on 46 inputs: 40 random ones with N = 1 to 119 and rates 0 or 1e-12 to 0.1, and six fixed ones from N = 1
     to 400); with its
     validation removed it returns mode[1] = -8.164e-167 and mode[400] = +0.90895 at v = 1e-200 and 1e-300, i.e. the
     correct signs, where the previous rule flipped them. crown_check (below).
10. Step 5: the visible text named a hidden test input (independent verification of pass 2, blocking, rule B).
   - Finding: the step-5 prompt, function_header, contract and the six docstring copies of substitution_times said
     that t_up can exceed 1e300 only "near (the corner) N = 400, s = -0.5, u = 0.1, v = 1e-12"; the same tuple is a
     hidden input of tests/step_7.py case 3 (fastest_mode(400, -0.5, 0.1, 1e-12), target 5.87144975069e-197), the
     input with the largest span of the mode.
   - Cause: version 13 described the excluded region by its extreme corner.
   - Change: every copy now gives the region in general terms by necessary bounds: "only t_up can be that large, and
     only if N >= 370, s <= -0.46, u >= 0.06 and v <= 3e-3" (the prompt adds "for large N with A strongly
     deleterious, frequent mutation A -> a and rare mutation a -> A ... all hold"; the contract says the same). Files:
     problem.yaml (prompt, function_header, contract), steps/step_5.py, solution/step_5.py, solution.py,
     second_solution.py, mutants/step_5_dense_solve.py, mutants/whole_task_mutation_before_selection.py.
   - Evidence for the bounds (second solution; the reference agrees with it to 5e-13 at four inputs near the
     boundary, while at the corner itself, outside the domain, the reference returns nan and the second solution
     inf): along the four lines through the corner, t_up is monotone (increasing in N and u,
     decreasing in s and v) and reaches 1e300 at N = 376 (7.4e299 at N = 375, 4.4e300 at N = 376), s = -0.46704,
     u = 0.067998 and v = 2.538e-3 (bisection); on 1200 random inputs with N in [350, 400], s in [-0.5, -0.43],
     u in [0.055, 0.1] and v log-uniform in [1e-12, 10^-1.5] the 16 inputs with t_up > 1e300 all lie in the box
     N >= 376, s <= -0.467, u >= 0.068, v <= 2.54e-3, the largest t_up outside it is 2.2e292, and no t_down exceeds
     1e300 (the largest t_down over N in {300, 350, 400}, s in {-0.5, -0.3, 0, 0.3, 0.5}, u, v in {1e-12, 1e-6, 0.1}
     is 1.2e233). The stated bounds are these thresholds rounded outwards. They replace the coarse-grid region of
     version 13, item 2, whose v bound (1e-3) was too small (t_up = 1.2e300 at v = 2.5e-3).
   - Regression test: the AST comparison of item 2 repeated (31 files, 49 docstrings of the seven step functions,
     no mismatch; problem_io equals the step-7 function_header); a search of problem_description_main, every
     step_description_prompt, function_header and problem_io for parameter assignments finds no tested tuple (only
     N = 400 as a size, N = 1 with u = v = 0 as the stated error case, and u = 0 / v = 0 as the absorbing cases);
     crown_check format. No test changed for this item.
   - verification_explanation now cites the scans of items 9 and 10.

Rerun (after items 9 and 10, the last change to code, tests and problem.yaml): tools/crown_check.py on the whole
folder, all modes, every case alone in a fresh process (-j 2, one BLAS thread); solution.py on every case of
tests/step_1..7.py, each alone in a fresh process (all pass); fastest_mode of solution.py and second_solution.py
(six inputs from N = 1 to 400) run twice in clean processes with one BLAS thread and once with two: bit-identical
outputs in all three runs (second_solution.substitution_times differs by 3e-16 relative with two BLAS threads,
as noted in version 13). The results below are those of this final run; they are the same as before items 9
and 10 (the new entries of step-7 case 5 raise ValueError in the reference, the second solution and every
step-7 mutant, so the failing-case lists of the mutants are unchanged).

crown_check results:
- format: OK (no general case without a call of fastest_mode; relevant_experience is the placeholder for the author).
- ref: every step 5/5, 4/4, 5/5, 5/5, 5/5, 5/5, 6/6 and general 5/5 pass; solution.py also passes every step file.
- second: every step 5/5, 4/4, 5/5, 5/5, 5/5, 5/5, 6/6 and general 5/5 pass.
- mut: step 1: [0, 2, 3] and [0, 1, 2, 3]; step 2: [1, 2] and [1, 2]; step 3: [0, 2, 3]; step 4: dense_eigenvalues
  [1, 2, 3], neutral_gap [2, 3] (was [2]); step 5: [0, 1, 2, 3]; step 6: [1, 2, 3] and [1, 2, 3]; step 7:
  dense_eigenvalues [0, 1, 2, 3, 4], numerical_inverse [0, 1, 2, 3, 4], regularized_absorbing [0, 1, 2, 3, 4],
  largest_entry_positive [1, 2, 3, 4]; the whole-task mutant fails general [0, 1, 2, 3, 4].
- shift: every step output scaled by 1 + 1e-12 and by 1 - 1e-12, and the whole-task tests with every evaluated
  function so scaled: every case passes.
- controls: all-None and all-zero stubs pass no case of any step or of the general tests.

Re-verification of items 9 and 10 (second check of the two blocking findings, no change to code, tests or
problem.yaml):
- Item 9 confirmed against the pass-1 files (git history): the pass-1 solution/step_7.py accepts 0 < v < 1e-12 and,
  at N = 400, s = -0.5, u = 0.1, returns mode[0] = 3.27e-176, 3.27e-177, 3.27e-264 and 3.27e-314 for v = 1e-12,
  1e-13, 1e-100 and 1e-150, and at v = 1e-200 and 1e-300 mode[0] = -0, mode[1] = +8.164e-167, mode[400] = -0.90895
  (every sign wrong). The current reference raises ValueError there; with its validation line disabled it returns
  the same lambda_min (relative difference 0) and mode[1] = -8.164e-167, mode[400] = +0.90895 at all six v (signs set
  by position). Scan of the second solution at N = 400, 11 values of s in [-0.5, 0.5], u, v in
  {0, 1e-12, 1e-6, 1e-3, 0.1} (275 inputs): lambda_min from 1.357e-225 (s = -0.5, u = v = 0.1) to 9.603e-173
  (s = 0, u = v = 0), largest span 175.49 decades and smallest entry off the absorbing states 3.27e-176 (both at
  s = -0.5, u = 0.1, v = 1e-12); every mode with max |entry| = 1, + on the first transient state, strictly
  alternating signs, |entry| <= 1e-250 on the absorbing states. The step-7 statements stay true.
- Item 10 confirmed against the pass-1 problem.yaml (the step-5 prompt, function_header and contract named
  N = 400, s = -0.5, u = 0.1, v = 1e-12). No copy of the visible text names it now (search of problem.yaml, steps/,
  solution/, solution.py, second_solution.py, mutants/). Spot checks of the stated bounds, each with the other three
  parameters at the extreme corner: t_up = 1.66e295 at N = 369, 9.94e295 at s = -0.4599, 7.78e294 at u = 0.0599 and
  2.67e298 at v = 3.01e-3 (second solution; the reference agrees to 4e-13 at these four inputs), and 4.4e300 at
  N = 376, 3.5e300 at s = -0.468, 1.0e300 at u = 0.068, 1.35e300 at v = 2.5e-3 (second solution), as in item 10.
  400 new random inputs with N in [340, 400], s in [-0.5, -0.42], u in [0.05, 0.1], v log-uniform in
  [1e-12, 10^-1.5] (seed 20261011): the 326 outside the stated box have t_up at most 1.1e283, and no t_down exceeds
  2.2e8.
- AST comparison of the docstrings: 31 files, 49 docstrings of the seven step functions, no mismatch with the
  function_header docstrings; problem_io equals the step-7 header.
- fastest_mode of solution.py and second_solution.py on six inputs (N = 1 to 400, absorbing and non-absorbing):
  bit-identical outputs in three clean processes (one BLAS thread twice, two threads once).
- tools/crown_check.py, all modes, every case alone in a fresh process (-j 2, one BLAS thread): SUMMARY ALL OK,
  FORMAT OK, with exactly the results listed above (ref, solution.py on every step file, second, the failing-case
  lists of every mutant, shift with every step output and every evaluated function in the whole-task tests scaled
  by 1 +- 1e-12, and the all-None and all-zero controls passing no case).

## Version 15 (grading_fix: platform content-check rules, applied before the platform v11 report was read)

- Finding: the platform content check now enforces (2026-10-12): no non-finite literal anywhere in the tests; no
  finiteness promise in the visible text; computed values must clearly outweigh validation per step (assertion
  sites: type/shape/exception checks, a stated raise at half weight, against computed-value checks); no relative
  requirement on an exactly zero output; every stated output shape and inclusive endpoint exercised with values.
  Written before the platform v11 report was read; the rules were applied proactively.
- Cause (audit of version 14, commit 9a6c5ee): (1) Non-finite inputs and promises: every raise list contained a
  non-finite input (float("nan") in steps 1, 3, 4, 6 and 7, float("inf") in steps 2, 5 and 7), and 15 lines of
  problem.yaml (problem_description_main, the step prompts and the function_header docstrings, with their copies in the
  scaffolds, solution.py, solution/step_1..7.py, second_solution.py and the mutants) promised a ValueError when "s, u or
  v is not finite"; the references, the second solution and the mutants carried an explicit np.isfinite check with
  the message "must be finite" ahead of the range checks. (2) Validation weight: the raise lists had 8, 9, 9, 8, 8, 8
  and 18 entries (steps 1 to 7), several testing the same rule; asserts made only of type/shape/len/dtype checks were
  2 of 13, 2 of 9, 3 of 14, 4 of 9, 3 of 8, 8 of 20 and 10 of 27 asserts (the contract check repeated in several
  cases); the steps had 5, 4, 5, 5, 5, 5 and 6 cases. (3) No relative requirement applies to an exact zero: entries of
  P, pi and the quasi-stationary distribution below 1e-250 have the absolute rule, the exact zeros of the mode on
  absorbing states have the 1e-250 rule, the transient entries of the mode are nonzero (strict sign alternation of the
  last eigenvector of a totally positive block), and p_fix, p_loss, the times, the gap, the rate and lambda_min are
  positive in the domain. (4) Every endpoint was already exercised with values; the (N + 1, N + 1) shape of step 1 was
  checked only at N >= 40.
- Change:
  - Text: the "is not finite or" clauses are removed from the 15 lines of problem.yaml and from every copy of the
    docstrings; the stated errors are now the integer rules and the ranges only.
  - Code: the np.isfinite checks are removed from solution.py, solution/step_1..7.py, second_solution.py and every
    mutant (step 2: `np.isfinite(s) and` dropped from the range test). A NaN fails every range comparison and +-inf
    lies outside every range, so non-finite s, u, v still raise ValueError (checked once on nine calls of solution.py
    and second_solution.py, outside the tests). No copy of the code or text mentions finiteness.
  - Raise lists (no non-finite input left; each stated rule once, just outside the domain), version 14 -> 15:
    step 1, 8 -> 6 entries: N = 0 and s = nan removed, N = 2.5 -> 10.0 (an integer-valued float), s = 0.6 -> -0.51,
    u = -1e-3 -> -1e-6, v = 0.2 -> 0.11; step 2, 9 -> 5: N = 401, i0 = 2.0, i0 = 0 and s = inf removed,
    s = 0.7 -> 0.51; step 3, 9 -> 6: N = 401, v = 1e-13, u = 0.2 and u = nan removed, v = 0.11 added,
    s = -0.6 -> -0.51; step 4, 8 -> 6: N = 0 and s = nan removed, N = 500 -> 401, s = 0.6 -> 0.51; step 5, 8 -> 6:
    N = 0 and u = inf removed, v = 0 -> u = 0, v = 0.15 -> 0.11; step 6, 8 -> 5: N = 0, v = 0.2 and s = nan removed,
    s = 0.6 -> 0.51; step 7, 18 -> 8: N = 0, s = -0.6, u = -1e-12, v = -1e-12, u = 0.2, s = nan, u = inf,
    v = 5e-13, (400, -0.5, 0.1, 1e-200) and (3, 0.2, 9.9e-13, 0) removed, (10, 0.1, 0, 1e-300) -> (10, 0.1, 0.01,
    1e-200) for the stated rule 0 < v < 1e-12, s = 0.6 -> 0.51 (N = 401, N = 5.0, N = True, v = 0.11, u = 1e-13 and
    N = 1 without mutation kept).
  - One contract check per step (asserts made only of type/shape/len/dtype checks: 2, 2, 3, 4, 3, 8, 10 -> 1 each):
    step 1 case 2 (type, shape and float dtype, now also at N = 1 with v = 0: a (2, 2) matrix with an exact zero; the
    shape check of case 3 removed), step 2 case 1 (that of case 2 removed), step 3 case 0 (float numpy array; the shape
    asserts of cases 2 and 3 removed, entries read by position), step 4 case 0 (those of cases 1, 2, 3 removed), step 5
    case 0 (those of cases 1 and 2 removed), step 6 case 0 (type and shape in one assert; cases 1, 2, 3 no longer
    repeat them), step 7 case 0 (cases 1 to 4 no longer repeat them). Value asserts read entries by index, so a wrong
    shape still fails them.
  - New computed-value cases with independent targets (case counts 5, 4, 5, 5, 5, 5, 6 -> 8, 8, 8, 11, 12, 7, 8):
    - step 1 case 5: every entry of small matrices in exact rational arithmetic (Python fractions from the exact binary
      values of the inputs), mutation in both directions, one way only (exact zeros) and a rate at its floor 1e-12;
      relative 1e-10, exact zeros within 1e-250.
    - step 1 case 6: allele relabeling, P(s, u, v)[i, j] = P(s', v, u)[N - i, N - j] with s' = -s / (1 + s), at
      N = 400 with u = 1e-12 (rows near i = N, where 1 - p_mut is of order u, against the relabeled rows near i = 0);
      two computed outputs, 2.1e-10 relative or 2e-250 absolute.
    - step 2 case 4: N = 2 with selection in closed form, p_fix = (1 + s)**2 / (1 + (1 + s)**2) and
      t_fix = t_loss = (2 + s)**2 / (1 + (1 + s)**2).
    - step 2 case 5: p_fix, p_loss and the conditional times of small N for every i0, from the absorbing-chain systems
      solved exactly with Python fractions.
    - step 2 case 6: first-step identities on the outputs for every i0 (h harmonic with its boundary values, and
      g = h t with g(i) = h(i) + sum_j P[i, j] g(j)), all terms nonnegative, down to p_fix of about 4e-90.
    - step 3 case 5: the stationary distribution of small N in exact rational arithmetic, a mutation rate at its
      floor 1e-12 included.
    - step 3 case 6: stationarity of the output entry by entry, (pi P)_j = pi_j, on every entry of at least 1e-235 at
      N = 400, s = -0.5, u = v = 1e-12 (entries spanning more than 200 decades).
    - step 4 case 5: the gap of small populations with selection as the smallest root of the characteristic
      polynomial of I - P (principal-minor sums in exact rational arithmetic, roots in mpmath at 80 digits), gaps of
      order 1e-12 included.
    - step 4 case 6: origin-fixation limit, gap = N v pi_A + N u pi_a with rare mutation in both directions
      (corrections below 1e-9 relative), pi_A and pi_a from the mutation-free absorbing chain.
    - step 4 case 7: one rare and one frequent mutation rate, and moderate rates, by inverse iteration on
      I - P + 1 pi^T in extended precision.
    - step 4 case 8: N = 400, s = 0.5, u = 1e-12, v = 3e-12 (gap about 7e-10) by the same inverse iteration with one
      plain LU factorization in mpmath at 400 digits.
    - step 4 case 9: neutral drift with rare mutation, gap = u + v exactly (2e-12 to 2e-11).
    - step 5 case 5: the hitting times of small N in exact rational arithmetic, with u = v = 2**-30 (t_up about 1e9).
    - step 5 case 6: the commute-time identity t_up + t_down = 1 / (pi_0 e) with frequent mutation.
    - step 5 case 7: neutral drift with rare and frequent mutation, and A deleterious with v at its floor, against
      plain LU of the hitting-time systems in extended precision.
    - step 5 case 8: N = 400 with rare mutation and strong selection (t_up about 4.4e248) by Gaussian elimination in
      mpmath at 400 digits.
    - step 5 case 9: first-passage times from the fundamental matrix Z = (I - P + 1 pi^T)^-1 of the ergodic chain,
      (Z[b, b] - Z[a, b]) / pi_b, in mpmath at 80 digits (t_up about 9e15 included).
    - step 5 case 10: the commute-time identity with rare mutation and strong selection (times about 1e16 and 8e19
      against about 1e11).
    - step 6 case 5: N = 2 in closed form (the 2 x 2 restricted matrix, Perron root and left eigenvector in mpmath at
      50 digits, v down to 1e-12).
    - step 7 case 6: selection without mutation (both end states absorbing, N = 2 included), a rate at its floor
      1e-12 with the other 0, and frequent mutation in both directions, against the matrix in extended precision;
      exact zeros of the mode on absorbing states within 1e-250.
    - step 1 case 7: the binomial ratio identity P[i, j + 1] (j + 1) q_i = P[i, j] (N - j) p_i, with p_i and q_i formed
      from the model without subtraction, on every pair of neighbouring entries whose exact value exceeds 1e-238
      (log-gamma estimate) in every row of N = 400 at three corners of (s, u, v) (about 1.4e5 pairs each), plus
      positivity there; tolerance 2.1e-10 (two outputs at 1e-10). Reference deviation at most 7.7e-13.
    - step 2 case 7: neutral drift at N = 400: p_fix = i0 / N and p_loss = (N - i0) / N exactly (martingale), and
      t_fix(i0) = t_loss(N - i0) (allele exchange), for i0 = 1, 2, 137, 200, 263, 398, 399.
    - step 3 case 7: exact neutral stationary moments E_pi[i**k], k = 1..4, from the triangular system of the binomial
      factorial moments (signed Stirling numbers), solved with Python fractions; checked against the brute-force
      exact stationary vector at N = 4 and against the E[p (1 - p)] formula of case 1 (both exact agreement). At
      N = 400 with u = v = 0.1, N = 400 with u = 0.1, v = 1e-12 (moments set by the small entries next to i = 0),
      N = 250 and N = 37; tolerance 1.01e-8 (sums of nonnegative terms).
    - step 4 case 10: the gap at the corners of the mutation range with the strongest selection (N = 30, s = 0.5,
      u = 0.1, v = 1e-12 and N = 24, s = -0.5, u = 1e-12, v = 0.1) by inverse iteration with plain LU in mpmath at 60
      digits (the method of case 7), and the neutral gap u + v = 0.2 at N = 400, u = v = 0.1.
    - step 5 case 11: N = 400, s = -0.45, u = 0.1, v = 1e-12, next to the excluded corner: t_up from the hitting-time
      system by plain mpmath LU at 380 and at 450 digits (identical to 40 digits: 3.452011881615877833e290, 253 s and
      283 s) and t_down at 100 digits (12.41572647226185732); the reference agrees to 7.6e-14 and 9.5e-14.
    - step 6 case 6: origin-fixation limit of the one-way absorption rate, rate = N v pi_A (1 + O(N v T)), pi_A by
      plain mpmath LU at 60 digits; first-order perturbation of the Perron root only involves row 0 (left vector e_0,
      right vector the loss probability). N = 60, s = 0.5 and N = 40, s = 0.3 with v = 1e-12: T = 12.5 and 12.2
      generations (mean segregation time of one copy), N v T = 7.5e-10 and 4.9e-10; tolerance 1.1e-8; the measured
      deviations of the reference are 1.8e-11 and 1.0e-11.
    - step 7 case 7: lambda_min and the mode of N = 4 to 7 chains (one absorbing state, both, the floor 1e-12 next to
      a zero rate) from the Faddeev-LeVerrier characteristic polynomial in exact rational arithmetic, the smallest
      root by mpmath polyroots at 120 digits, and the null vector of P - lambda_min I with the row of the first
      transient state replaced by x_t = 1 (plain LU at 120 digits); no eigensolver, no inverse iteration. Checked
      against mpmath eig (agreement to 1e-16). At N = 5, s = -0.5, u = 0, v = 1e-12 the first transient entry of the
      mode is 5.3e-13 and must be positive.
  - n_test_cases: 8, 8, 8, 11, 12, 7, 8 (general 5, unchanged). verification_explanation lists the new evidence
    (without naming test inputs).
- Regression: the new cases reject step_1 cancelled_complement and mutation_before_selection (case 7), step_3
  dense_eigenvector (case 7), step_4 neutral_gap (case 10), step_5 dense_solve (case 11), step_6
  absorption_from_mutation and dense_eigenvector (case 6), and all four step-7 mutants (case 7); step 2 case 7 is
  passed by both step-2 mutants (neutral drift does not separate them; they fail the cases listed below).
- Rerun (after the last change to code, tests and problem.yaml): tools/crown_check.py on the whole folder, all
  modes, every case alone in a fresh process (-j 2, one BLAS thread): SUMMARY ALL OK, FORMAT OK (no non-finite
  literal in the tests). ref: 8/8, 8/8, 8/8, 11/11, 12/12, 7/7, 8/8 and general 5/5, and solution.py passes every step
  file; second: the same counts, all pass. mut: step 1 cancelled_complement [0, 2, 3, 6, 7], mutation_before_selection
  [0, 1, 2, 3, 5, 7]; step 2 loss_by_complement [1, 2, 6], unconditional_times [1, 2, 5, 6]; step 3 dense_eigenvector
  [0, 2, 3, 5, 6, 7]; step 4 dense_eigenvalues [1, 2, 3, 5, 6, 7, 8, 9], neutral_gap [2, 5, 6, 7, 8, 10]; step 5
  dense_solve [0, 1, 2, 3, 5, 7, 8, 9, 10, 11]; step 6 dense_eigenvector and absorption_from_mutation [1, 2, 3, 5, 6];
  step 7 dense_eigenvalues, numerical_inverse and regularized_absorbing [0, 1, 2, 3, 4, 6, 7], largest_entry_positive
  [1, 2, 3, 4, 6, 7]; the whole-task mutant fails general [0, 1, 2, 3, 4]. shift: every step output, and every
  evaluated function in the whole-task tests, scaled by 1 +- 1e-12: every case passes. controls: all-None and all-zero
  stubs pass no case of any step or of the general tests. Raw folder size 425,859 bytes (du -sb).

## Version 16 / platform v12 (grading_fix: platform v11 returned by an administrator)

- Version mapping: platform v11 is commit 0054371 (local Version 11, submitted 5 Oct). Local Versions 12 to 15
  were prepared while v11 was on the platform (rollouts finished 7 Oct, then in review until its return on
  13 Oct) and go up together with this note as platform v12.
- Finding (platform v11 report): content check "1 to consider"; exploit probe clean; difficulty probe 0/3; rollouts
  in band 1/8 (rewards 0.00, 0.86, 0.57, 0.86, 0.86, 0.86, 1.00, 0.86); attempts 0, 2 and 7 invalid ("possibly
  timed out"); lost credit: code synthesis 4, process 1, no single cause 2; task-level notes "verifier correctness"
  and "solution discoverability" at 1/6 each; the one per-step grading shown (attempt 1, under the verifier-correctness note) fails step 7 on the final
  file; attempts 1, 3, 4, 5 and 7 scored 0.86 (6 of 7 steps). Returned "Needs
  revision" by an administrator on 13 Oct 04:31, in the same minute as another task of the author; no reason shown.
- Cause (what v11 still contained, checked in commit 0054371): a per-call budget "within 20 s on one CPU core" in
  the prompts of steps 2 to 7 (six sentences) and perf_counter timing asserts in the tests of steps 2 to 7; exact
  floating-point zero asserts (mode[N] == 0.0 at absorbing states); the finiteness clauses and non-finite inputs.
  All are against the authoring guide or the platform rules of 10-12 Oct. They were removed by Versions 12 (budget,
  timing asserts, exact zeros: absolute 1e-250 rule) and 15 (finiteness). The verifier-correctness note could not
  be traced to a test: an independent fairness audit (a valid step-7 implementation written from the visible text
  only, and step-7 targets recomputed by a different high-precision route) found no defect.
- Change (pre-upload audit, two stated behaviours that no test exercised):
  - Step 7 case 8 (new): the low end of the stated range of lambda_min, N = 400, s = -0.5, u = v = 0.1, lambda_min
    = 1.3567455827803434e-225, with 81 mode entries of strictly alternating sign (i = 0, 5, ..., 400). Target:
    independent_recomputation, the matrix from mpmath binomials at 400 digits, inverse iteration with one plain LU
    factorization in mpmath (no use of the Vandermonde structure); cross-checked by a Lagrange-inverse route at 120
    digits followed by the power method at 60 digits (lambda within 3.6e-30, entries within 4.6e-18).
  - Step 3 case 8 (new): the "within 1e-250" rule for the far tail, N = 400, s = -0.5, u = 0.1, v = 1e-12; the
    distribution crosses 1e-250 between i = 340 and 341 and 60 entries lie below it. Target:
    independent_recomputation, pi (I - P) = 0, sum(pi) = 1 solved by one plain LU in mpmath at 400 digits;
    cross-checked by GTH elimination in mpmath at 40 digits (55 stored entries within 4.4e-18).
  - n_test_cases: step 3 and step 7 go from 8 to 9.
- Regression: the reference meets the new targets within 1.2e-13 (lambda) and 6.3e-13 (mode) in step 7 and 5.6e-13
  relative / 3.4e-264 absolute in step 3; the second solution within 6.9e-13 / 1.3e-12 and 1.4e-13 / 2.4e-264.
  The new cases reject the step 3 mutant dense_eigenvector and the step 7 mutants dense_eigenvalues,
  numerical_inverse and largest_entry_positive.
- Rerun: tools/crown_check.py, all modes; every case alone in a fresh single-threaded process (69 cases, about 88 s
  in all, slowest about 7.6 s); raw folder size after this note about 437 KB (du -sb).
