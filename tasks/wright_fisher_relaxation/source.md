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
  (versions 1-12; replaced for steps 2-6 in version 13, see there): the same linear systems solved
  by plain LU in mpmath, with the precision doubled until two runs agree to 1e-13. The two agree to
  about 1e-14 relative on all tested cases.
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
