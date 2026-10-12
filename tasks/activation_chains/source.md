# Source

- H. Bateman, "Solution of a system of differential equations occurring in the theory of radioactive
  transformations", Proceedings of the Cambridge Philosophical Society 15, 423-427 (1910). The chain
  solution.
- J. Cetnar, "General solution of Bateman equations for nuclear transmutations", Annals of Nuclear
  Energy 33, 640-645 (2006). Branching networks, production and the numerical problems of the
  Bateman form for close decay constants.
- A. C. McCurdy, K. C. Ng and B. N. Parlett, "Accurate computation of divided differences of the
  exponential function", Mathematics of Computation 43, 501-528 (1984), and A. H. Al-Mohy and
  N. J. Higham, "A new scaling and squaring algorithm for the matrix exponential", SIAM Journal on
  Matrix Analysis and Applications 31, 970-989 (2009). Scaling and squaring, and recomputing the
  diagonal of triangular exponentials exactly.
- M. Pusa and J. Leppanen, "Computing the matrix exponential in burnup calculations", Nuclear Science
  and Engineering 164, 140-150 (2010). Stiffness of depletion matrices and the failure of standard
  methods for short-lived nuclides.
- Half-lives and cross sections: NUBASE2020 (F. G. Kondev et al., Chinese Physics C 45, 030001, 2021)
  and the Atlas of Neutron Resonances (S. F. Mughabghab, 6th ed., Elsevier, 2018): 197Au 98.65 b,
  198Au 2.6941 d and about 25100 b, 199Au 3.139 d, 59Co 20.7 b to 60mCo, 60Co 5.2714 y and 2 b, and
  the 238U series. The 199Au cross section is rounded to 30 b, and the 214Bi -> 210Tl branch to
  2e-4.

The combination of steps, the subtraction-free scaling and squaring, the parameter ranges and the
accuracy requirements are our own design.

Checks used while building the task:
- Two independent oracles agree to all printed digits on the full 238U series (15 members, decay
  constants over 21 decades) at 1 y, 1e4 y and 1e9 y: the Bateman sum in 600-digit arithmetic and
  mpmath's matrix exponential at 300 digits.
- The reference reproduces them to about 1e-14 relative on random chains and networks, including:
  - nearly equal decay constants (relative differences down to 1e-15);
  - entries down to 1e-250;
  - production terms and nuclides given in a non-topological order.
- Closed forms are reproduced:
  - single decay;
  - Erlang amounts for equal decay constants;
  - saturation with production, including lam t = 1e-10;
  - two-way branching;
  - target burnup exp(-sigma phi t) down to exp(-200).
- The second solution agrees with the reference to about 1e-14. (Versions 1-13: mpmath expm in
  adaptively doubled precision, bisection plus secant for the flux. From version 14: the generalized
  Bateman solution in extended precision, bisection plus secant; agreement about 4e-14 on 60 random
  networks. Its precision control was first "double until two evaluations agree" and is, at the end of
  version 14, a running error bound; see the last items of version 14.)
- Standard methods fail the tests:
  - scipy.linalg.expm and the Bateman sum in double precision both fail step 1;
  - squaring the diagonal in scaling and squaring (instead of resetting it) loses the amounts on the
    238U series completely.
- The gold-monitor response: d ln A / d ln phi for 198Au after 5 d of irradiation and 1 d of cooling
  is 0.96 at 1e13, 0.65 at 1e14, and 0 near 1.6e15 (burnup maximum); 199Au rises with slope at least
  1.1 on [1e-2, 1e15].

Version 2 (content-check fixes):
- Accuracy rule: the relative bound applies to exact entries that are positive and at least
  1e-250 * S; every other entry, exact zeros included, must lie within max(1e-250 * S, 1e-300). This
  covers S = 0 (an empty inventory), which is now tested in steps 1-3. (Historical: since version 14
  the relative bound applies to exact entries of at least F = max(1e-250 * S, 1e-300).)
- Branching column sums: one rule everywhere, at most 1 + 1e-12 (1 up to rounding). Step 2 tests a
  sum of 1 + 5e-13 (accepted) and 1 + 1e-9 (rejected). Step 3 checks the branching matrix itself
  before the history, so an empty history no longer skips the check.
- New endpoint tests:
  - step 1: t = 0, lam = 1e10, n = 30 (equal and distinct constants), 2-D and empty arrays;
  - step 2: t = 0, t = 1e20, lam = 1e10, n = 30, n = 31;
  - step 3: sigma = 1e7 and above, capture_to = -2 and n, flux = 1e18 and above, duration = 1e12
    and above, an empty history;
  - step 4: t_cool = 0, t_irr = t_cool = 1e9, flux_hi = 1e18, the times above 1e9, k = -1 and n,
    a NaN activity.

Version 3 (content-check fixes):
- Step 3 new tests:
  - lam = 1e10 under no flux, and lam above 1e10 rejected;
  - a combined rate of exactly 1e10 per s (lam = 1e10 - 10, capture rate 10 per s), with its
    closed form;
  - n = 30 with an empty history, and n = 31 rejected;
  - a NaN cross section and an infinite duration rejected.
- Step 4 now states the step-3 data contract (types, shapes, ranges, acyclicity) itself.
- Step 4 makes the interval explicitly closed: an activity within a relative 1e-9 of A(flux_lo) or
  A(flux_hi) gives that end. This is tested at both ends.

Version 4 (content-check fixes):
- Step 4: the end rule is two-sided and checked before the bracket. If
  |ln(activity / A(flux_lo))| <= 1e-9 the result is flux_lo, else if the same holds at flux_hi it is
  flux_hi; on either side of the end value. Tested at both ends with activities equal to the end
  values and 5e-10 inside them.
- Step 2 tests a branching column summing exactly to the float 1 + 1e-12 (accepted).
- Step 3:
  - accepts the same bound with an empty history;
  - rejects a nonzero diagonal with an empty history;
  - ignores capture links of nuclides without cross section (a zero-sigma capture loop is accepted);
  - turns a history that is not a sequence (None) into ValueError;
  - rescales the period fractions so that rounding cannot push a column at the bound above it.

Version 5 (content-check fixes):
- Step 4 tests both sides of each end: activities 5e-10 outside A(flux_lo) and A(flux_hi) also give
  the ends.
- Step 3 requires history to be a list or tuple (ordered); None, a set and a generator are rejected.
- Steps 1-3 test amounts of exp(-550), about 1e-239, near the bottom of the relative-accuracy range:
  - step 1: pure decay;
  - step 2: pure decay in a network;
  - step 3: capture burnup at sigma = 1e7 b and 1e18 n / (cm^2 s) for 5.5e6 s.

Version 6 (content-check fixes):
- Step 3 new tests:
  - a valid history given as a tuple;
  - a history entry with a non-numeric value rejected;
  - a capture loop with positive cross sections rejected even with an empty history;
  - sum(n0) = 1e100 accepted and above rejected.
- Domain bound on the amounts, so that every exact result is representable in double precision:
  - sum(n0) <= 1e100 in steps 1, 3 and 4;
  - sum(n0) + t * sum(source) <= 1e100 in step 2.
  Each bound is stated, enforced (ValueError), and tested at the bound and above it.

Version 7 (advisory fixes after the v6 rollouts):
- Steps 1-3 test a positive exact amount below the 1e-250 S threshold: exp(-600), about 3e-261, is
  checked with the absolute rule.
  - step 1: pure decay;
  - step 2: pure decay in a network;
  - step 3: burnup at sigma = 100 b, flux 1e18, 6e6 s.
- Step 2 rejects an empty network, a scalar source and a 2-D n0.
- Step 3 rejects a scalar sigma, a branching matrix that is not n x n, and an n0 of the wrong length.

Version 8 (content-check fixes):
- Steps 1-2: the number of halvings is 0 whenever t * max(lam) <= 0.5. The logarithm is no longer
  taken of a product that underflows to 0 (lam = 1e-300, t = 1e-300), and that case is now tested.
- Step 4: activities 2e-9 (in |ln|) outside A(flux_lo) and A(flux_hi) must raise ValueError,
  next to the accepted 5e-10 cases.

Version 9 (content-check fixes):
- Domain of the cross sections: sigma = 0 or 1e-6 <= sigma <= 1e7 b (steps 3-4). Every capture
  rate sigma * 1e-24 * phi stays far above the floating-point underflow, so the promised flux
  accuracy is achievable.
  - Tested in step 3 with 1e-6 b (accepted) and 1e-7 b (rejected).
  - Tested in step 4 with 1e-300 b (rejected) and a round trip on a 1e-6 b monitor.
- Step 4:
  - activity = 0 and flux_lo = flux_hi raise ValueError;
  - the end tolerance is tested at |ln| = 0.999e-9 on both sides of each end (the end is
    returned) and at 1.01e-9 outside (ValueError).

Version 10 (content-check fixes):
- (Historical: since version 14 the rule is A(flux_lo) >= lam[k] * max(1e-250 * sum(n0), 1e-300).)
- Step 4 domain closed: A(flux_lo) >= 1e-250 * lam[k] * sum(n0), i.e. the measured nuclide holds at
  least 1e-250 of the initial atoms at flux_lo, the range where step 3 promises relative accuracy.
  Otherwise ValueError. This excludes the case where the per-atom product fraction underflows
  (t_irr = 1e-300 with n0 = 1e100), which is tested as a ValueError.
- Step 4 end rule: the tests use the activities closest to |ln| = 1e-9 from inside (1e-9 - 1e-14,
  a few rounding units of the logarithms; end returned, at both ends and on both sides) and
  1.01e-9 outside (ValueError).
- Step 4 new tests: a round trip at slope 0.52 (flux 1.547e14 on the gold monitor), and the
  precedence of flux_lo when both end activities are within the tolerance.
- Steps 1-3 test exp(-575), about 8e-250, just inside the relative-accuracy range.

## Version 11

- Step 4 end rule tested at the boundary itself: on both sides of both ends, the activity closest to
  |ln(activity / A(end))| = 1e-9 from inside (within 1e-14) returns the end, so a cutoff anywhere
  measurably below 1e-9 fails. The prompt no longer says which activities are used in the tests.
- Step 4 domain check done on the atoms in separate logarithms, ln x[k] >= ln 1e-250 + ln sum(n0),
  since 1e-250 * lam[k] * sum(n0) underflows to 0 for small lam[k] (then -inf >= -inf passed). The
  prompt states that the rule holds in that case; a new test with lam[k] = 1e-75 and a two-capture
  product holding 5e-267 of the atoms at flux_lo raises ValueError (the v10 reference returned flux_hi).
  ln A is formed as ln lam[k] + ln x[k] so that lam[k] * x[k] cannot underflow.

## Version 12

- Step 4 end rule: the tolerance 1e-9 is stated to hold up to the rounding of the end values, 1e-12 in
  ln (for 1e-9 - 1e-12 < |ln| <= 1e-9 + 1e-12 the end, the root and, outside the interval, ValueError
  are all correct), a normative rule since A(end) itself carries rounding errors of order 1e-15 in any
  implementation. Tests, at both ends and on both sides: the representable activity adjacent to
  |ln| = 1e-9 - 1e-12 from inside (exact logarithms in mpmath against the extended-precision A(end)) gives
  the end; the representable activity adjacent to |ln| = 1e-9 + 1e-12 from beyond raises ValueError
  outside the interval and gives a flux within 1e-8 of the end inside it. Cutoffs at 0.998e-9 and
  1.002e-9 fail.
- Step 4 accuracy edge: a round trip at 1.645e14 on the gold monitor, where d ln A / d ln flux is just
  above 0.5 (asserted in the test; 0.5 is reached at about 1.6454e14).
- Steps 1-3: one-nuclide cases at exp(-575.636) = 1.01e-250 S, just above the threshold of the relative
  rule (pure decay, decay without source, burnup).

## Version 13 (grading_fix: authoring-guide conformance)

Audit of every file against the current authoring guide. The science, the reference algorithms, the
targets, the tolerances of the individual outputs and the domain are unchanged.

- Test-case format and self-contained cases (tests/step_1.py to step_4.py, tests/general.py).
  - Finding: the markers read "# --- test case N: description ---" instead of the parser format
    "# --- test case N ---", and the imports, the target helpers (_t_check, _t_bateman, _t_expm_apply,
    _t_activation, _t_activity, _t_roundtrip) and the nuclide constants were defined once before case 0.
    tests/step_1.py case 5 used _t_fact, imported only in case 1. tests/step_4.py defined _T_DAY twice
    and gold constants that no case used.
  - Cause: the files were written for a whole-file run, before the per-case parser format.
  - Change: markers normalized to "# --- test case N ---" (0..n-1) with the description kept as
    comments; every case now carries its own imports, helpers and constants (tools/selfcontain.py, then
    checked by hand); step 1 case 5 imports factorial itself; the duplicate and unused step 4 constants
    were removed. No case was dropped, merged or split: n_test_cases stays 7, 6, 6, 6.
  - Regression test: tools/crown_check.py runs every case alone in a fresh process. Before the change the
    reference passed 1/7, 1/6, 1/6, 0/6 and 0/2 cases (NameError on the shared helpers); after it, all.
- Comparison of two computed outputs (tests/step_3.py case 3).
  - Finding: the invariance check (a period split into two halves, or a zero-length period inserted)
    compared two outputs of the candidate with the single-output tolerance 1e-10.
  - Cause: each output may carry the stated error, so their difference may reach about twice it.
  - Change: the pair is compared with twice the stated tolerance (relative 2e-10; absolute
    2 max(1e-250 S, 1e-300)) and the shape of the output is asserted.
  - Regression test: reference, second solution and the +-1e-12 shift pass case 3.
- Floating-point equality in target generation (tests/step_2.py case 0).
  - Finding: the closed-form target branched on "_t_lam == 0.0" (not an assertion).
  - Cause: the target special-cases lam = 0, where (1 - exp(-lam t)) / lam becomes t, and the branch was
    written as an exact float comparison, a pattern the conformance check flags.
  - Change: written as "if _t_lam > 0.0 ... else ..." (same meaning, lam >= 0), so no float equality
    pattern remains in the tests. No assertion compares floats exactly or with zero tolerance.
  - Regression test: crown_check format finds no float-equality pattern; reference, second solution and
    the +-1e-12 shift pass step 2 case 0 with the same targets as before.
- Stated but untested error rules.
  - Finding: the prompts promise ValueError for a non-finite initial amount and a non-finite t (step 1),
    a non-finite entry (step 2), a capture_to of the wrong shape (step 3) and non-finite flux bounds
    (step 4); none of these inputs was tested.
  - Cause: the error cases tested one representative input per rule and were not extended when the
    finiteness and shape rules were stated for every input.
  - Change: one entry each added to the existing error cases: n0 = [1, NaN] and t = NaN (step 1 case 6),
    n0 = [1, NaN] (step 2 case 5), capture_to of length 1 for n = 2 (step 3 case 5), flux_lo = NaN
    (step 4 case 5). Reference and second solution raise ValueError for each.
  - Regression test: a validation written as np.any(n0 < 0) (False for NaN) lets n0 = [1, NaN] through and
    returns NaN amounts, so the case fails; the reference and the second solution pass every error case.
- tests/general.py case 1 comment.
  - Finding: the comment called the history "two-cycle" and the flux "of the second cycle"; the history
    is one irradiation (3e6 s) and one cooling (2e5 s).
  - Cause: the comment was not updated to the history the case actually uses (comment-only defect).
  - Change: comment corrected; test unchanged.
  - Regression test: none needed for a comment; the case was rerun unchanged (reference, second solution,
    shift, whole-task mutant).
- Step 2 prompt and contract.
  - Finding: the prompt said "Generalize step 1 (chain_inventory)", naming a function that step 2 does
    not use, while its step_dependencies is [].
  - Cause: the prompt referred to step 1 by its function name although the step 2 reference exponentiates
    the augmented network matrix itself and never calls chain_inventory.
  - Change: it now reads "Generalize the linear chain of step 1 to a branching network with constant
    production."
  - Regression test: step_dependencies [] matches the prompt; the step 2 reference passes all its cases
    with no earlier step in the context (rerun in version 14).
- Metadata added to problem.yaml.
  - Finding: subfield, tags, expert_time_estimate_hours, relevant_experience, difficulty_explanation,
    solution_explanation, verification_explanation, author, affiliation and edit_label were missing.
  - Cause: the task was built on an earlier skeleton without these fields.
  - Change: all added (relevant_experience as a placeholder for the author; edit_label: grading_fix).
  - Regression test: crown_check format reports no missing metadata key.
- Mutants built from a stale reference (mutants/step_4_linear_response.py,
  mutants/whole_task_barn_in_square_metres.py).
  - Finding (independent verifier): both mutants had been copied from monitor_flux as it was before
    version 11. Besides the named mistake, each kept two old defects that version 11 fixed in the
    reference: the v10 domain rule ln A(flux_lo) >= ln(1e-250 lam[k] sum(n0)), whose right-hand side
    underflows to ln 0 = -inf for a small lam[k], and ln(lam[k] x[k]), whose product can underflow. A
    mutant should introduce one explainable scientific mistake. The step 4 mutant's failure of case 5 came
    from the stale domain rule, not from the linear-response mistake.
  - Cause: the reference was fixed in version 11 but the mutant files were not regenerated from it.
  - Change: both files were regenerated from the current reference and differ from it in one line only
    (diff checked): step 4 replaces the brentq root by flux = flux_lo exp(ln activity - ln A(flux_lo))
    (activity assumed proportional to the flux); the whole-task mutant replaces sigma 1e-24 by
    sigma 1e-28 in activation_inventory. Names, descriptions and problem.yaml entries are unchanged.
  - Regression test: per case, with the crown_check harness: the old step 4 mutant failed cases 0, 1, 2,
    3, 5; its stale code alone (brentq restored) fails case 5 only ("A(flux_lo) below 1e-250 lam[k]
    sum(n0) with a tiny lam[k] must raise ValueError"); the regenerated mutant fails cases 0, 1, 2, 3
    (the flux-inversion round trips, where burnup of 198Au, the flux**2 growth of 199Au or the domain
    ends make the activity not proportional to the flux) and passes case 5, as it should. The old whole-task mutant's
    stale code alone fails no general case; the regenerated mutant fails general case 1.
- Checked without change: no timing assert, clock read or per-call time budget in any test, prompt,
  docstring or solution; every solution import (numpy, scipy.optimize.brentq, mpmath) is declared in
  required_dependencies; the per-step solutions do not redefine earlier step functions; every step has a
  complete contract and at least one named mutant, and there is a whole-task mutant.

Rerun: tools/crown_check.py, all modes, every case alone in a fresh process:
- FORMAT OK.
- REF: step 1 7/7, step 2 6/6, step 3 6/6, step 4 6/6, general 2/2 (solution.py also passes every step file).
- SECOND: 7/7, 6/6, 6/6, 6/6, general 2/2.
- Mutants: float_bateman fails 5/7 (cases 1-5), step 1 squared_diagonal 3/7 (3, 4, 5),
  transposed_branching 4/6 (1-4), step 2 squared_diagonal 2/6 (3, 4), barn_in_square_metres 4/6
  (0, 1, 2, 4), linear_response 4/6 (0, 1, 2, 3), whole-task barn mutant 1/2 general cases (1); every
  mutant differs from its reference in its named mistake only.
- SHIFT +-1e-12: every case passes.
- Controls: all-None and all-zero stubs pass 0 cases in every step and in general.
- solution.py, the concatenated solution/step_*.py and second_solution.py were each run twice in clean
  processes and once with single-threaded BLAS on the 238U series, a random network, the gold monitor and
  two flux inversions: identical outputs (bitwise) in every run.

## Version 14 (grading_fix: authoring-guide conformance, pass 2)

Second audit against the authoring guide and the round-1 independent verification. The science, the
reference algorithms (solution/, solution.py), the existing targets and the 1e-10 and 1e-8 output
tolerances are unchanged. The input domain is unchanged except for the floor of 1e-300 atoms added to
the accuracy threshold of steps 1-3 and to the domain rule of step 4 after the verifier's round on this
pass (last item before the rerun). edit_label stays grading_fix.

- General case 0 did not call the final function.
  - Finding: tests/general.py case 0 checked only chain_inventory and network_inventory on the 238U
    series; whole-task tests must call monitor_flux, and only case 1 did (crown_check format: "general
    case 0 never calls the final function monitor_flux").
  - Cause: case 0 was written as an integrated check of the decay part only.
  - Change: the existing chain and network checks are kept, and the case continues with the aged ore
    (the exact 600-digit amounts after 3e5 years, 206Pb holding the rest of the 1e20 atoms) irradiated
    for one day and cooled for three days. The network has 18 nuclides: the 15 members of the series,
    then 239U (23.45 min), 239Np (2.356 d) and 239Pu (2.411e4 y, its decay leaves the network); only
    238U captures (2.68 b, into 239U), the captures of the products are left out. These half-lives and
    the cross section are rounded standard values; the test only needs physically plausible in-domain
    inputs, and its target is computed from exactly these inputs. For fluxes 5e11 and 2e14 the 239Np
    activity is computed independently as the 600-digit Bateman sum of the capture branch (during the
    irradiation the linear chain lam0 + r, l239U, l239Np, l239Pu with its products scaled by
    r / (lam0 + r), r = sigma 1e-24 flux; during the cooling the chain l239U, l239Np, l239Pu), and
    monitor_flux on [1e8, 1e16] must return the flux to the stated 1e-8. Link with step 1: the same
    activity predicted with two chain_inventory calls (the same decomposition) must also give back the
    flux, to 2e-8, about twice the single-output tolerance since that input carries the step-1 error
    (<= 2e-10, i.e. <= 2e-10 in the flux at slope 1).
  - Regression test: crown_check format no longer reports the case; reference and second solution
    recover both fluxes to 3e-15 or better by both routes; the whole-task barn mutant now fails general
    cases 0 and 1 (before: 1 only); the general shift check (every evaluated function scaled by
    1 +- 1e-12) passes.
- Step 4 end-tolerance band inconsistent with the accuracy of step 3.
  - Finding (verifier and lead): the prompt let the end tolerance |ln(activity / A(end))| <= 1e-9 be
    uncertain only within 1e-12 in ln, and case 3 tested it there. This requires A(flux_lo) and
    A(flux_hi) to about 1e-12, a hundred times better than the relative 1e-10 that activation_inventory,
    which step 4 is built on, promises. With the step-3 context scaled by (1 + 5e-12) or (1 - 5e-12),
    still within its contract, the version-13 case 3 fails.
  - Cause: in version 12 the band was sized to the rounding of a forward model accurate to about 1e-14
    (the reference), not to the stated accuracy of step 3.
  - Change: the prompt and the docstring now say that A is computed with activation_inventory (step 3),
    which gives it to a relative 1e-10, so the tolerance holds up to 1e-10 in ln: for
    1e-9 - 1e-10 < |ln| <= 1e-9 + 1e-10 the end, the root and (outside the interval) ValueError are all
    correct. Case 3 tests, at both ends and on both sides, the representable activity adjacent to
    |ln| = 1e-9 - 1e-10 from inside (the end) and to |ln| = 1e-9 + 1e-10 from beyond (ValueError outside
    the interval, a flux within 1e-8 of the end inside it). The threshold 1e-9, the 5e-10 and 2e-9
    checks and every other tolerance are unchanged; this fixes an inconsistency of the error budget and
    is not a tuning of the pass rate.
  - Regression test: the step 4 tests run with the step-3 context scaled by (1 + 9e-11), by (1 - 9e-11),
    and by opposite factors at the two ends (-9e-11 at 1e8, +9e-11 at 1e13): every case passes; the
    version-13 step 4 tests fail case 3 for each of these perturbations and for +-5e-12. The reference
    with end cutoffs at 0.85e-9 or 1.15e-9 fails case 3; cutoffs at 0.95e-9 and 1.05e-9 lie inside the
    stated band and pass.
- Unstated accuracy of the returned end.
  - Finding (verifier): cases 3 and 5 compare a returned end with flux_lo or flux_hi to a relative
    1e-11, a number that no visible text stated (the output accuracy stated is 1e-8, and the roots of
    the inside-band activities lie only 5e-10 to 1.1e-9 from the ends).
  - Cause: "return flux_lo" was meant as an exact discrete result and the float slack was not written.
  - Change: prompt and docstring: the end is returned as the given bound itself (not a root estimate),
    equal to it to a relative 1e-11; the docstring also states that flux_lo is checked first.
  - Regression test: tests unchanged; a variant of the reference that returns the root instead of the
    end for activities inside the interval fails case 3 ((100000000.05, 1e8) for an activity 5e-10
    inside A(1e8)).
- Second solution used the oracle's method (rule: genuinely different method, not a test oracle).
  - Finding: second_solution.py computed every inventory with mpmath's expm in adaptively doubled
    precision, which is the algorithm of the tests' 300-digit oracle (_t_expm_apply) with a different
    precision.
  - Cause: it was written as an extended-precision alternative to the double-precision reference
    without considering the test oracles.
  - Change: a new engine, _exp_poly_apply: the generalized Bateman solution. Every amount is a finite
    sum of terms c t**p exp(-a t); going down the network in topological order, each feeding term is
    convolved with exp(-r_j t) in closed form (t**(p+1) / (p+1) for a = r_j, so equal rates are exact;
    p! / d**(p+1) [exp(-r_j t) - exp(-a t) sum_m (d t)**m / m!] for d = a - r_j otherwise), the
    production entering as a term with a = 0. (Its precision control was replaced later in this version
    by a running error bound, see the item "Accuracy threshold and step 4 domain below 1e-300 atoms".)
    It is evaluated in mpmath from 30 digits, doubling the
    precision until two evaluations agree to 1e-13 in every entry (or to 1e-3 of the absolute floor
    max(1e-250 S, 1e-300)). chain_inventory, network_inventory and every period of
    activation_inventory (built directly from decay and capture rates, without branching fractions) use
    it; monitor_flux keeps bisection on ln flux followed by secant steps (the reference uses brentq).
    Remaining overlap, stated plainly: for linear chains with distinct constants the second solution
    and the Bateman oracle of steps 1-3 and of the general tests expand the same closed form, but by
    different algorithms (recursive convolution with confluent terms in adaptive precision, versus the
    explicit product formula at a fixed 600 digits); the reference uses an unrelated method
    (subtraction-free scaling and squaring), and the network oracle is mpmath expm.
  - Regression test: the second solution passes every case (steps 7/7, 6/6, 6/6, 6/6, general 2/2).
    It agrees with the reference to 3.7e-14 (relative rule above 1e-250 S, absolute below) on 60 random
    networks (2 to 30 nuclides, decay constants 1e-12 to 1e6, many exactly equal or 1e-13 apart, some
    zero, production, non-topological order, t from 1e-3 to 1e9), to 2e-14 on the 238U series from 1e-3
    to 1e20 s and on chains with constants one to a few rounding units apart, and to 2e-15 on the gold
    monitor inventories; both solutions recover the flux of three gold round trips (1e8, 1e12 and
    1.6454e14) to 5e-15 or better.
- Stated error rules and boundaries tested only through one representative (verifier).
  - Finding: negative decay constants or branching entries and infinite values (step 2); negative
    decay constants, cross sections, initial amounts or durations (step 3); infinite activity, flux
    bound or time (step 4) were stated as errors but not tested; n = 30 was tested in step 3 only with
    an empty history, which returns n0 trivially.
  - Cause: the error cases had one representative input per rule.
  - Change, entries added to existing cases only (n_test_cases unchanged: 7, 6, 6, 6): step 2 case 5:
    lam = [-1, 1], a branching entry -0.1, source = [inf, 0]; step 3 case 5: lam = [-1e-3, 1], sigma =
    [-5, 0], n0 = [1, -1], a duration of -1 s; step 4 case 5: activity = inf, flux_hi = inf,
    t_irr = inf. Step 3 case 4: a 30-nuclide network with a non-empty history, a stable target followed
    by 29 radioactive products each made by a capture on the previous one (sigma 1e2 to 1e4 b, lam 1e-6
    to 1e-3 per s, 1e5 s at 1e15 n / (cm^2 s), then 2e4 s of cooling), against the 600-digit Bateman sum
    of a chain with transfer rates sigma 1e-24 phi and removal rates lam + sigma 1e-24 phi (amounts down
    to 8e-54 atoms, 8e-74 of n0).
  - Regression test: reference and second solution raise ValueError for every added entry and pass the
    30-nuclide check (3e-15 and 2e-15); the step 3 barn mutant still fails cases 0, 1, 2 and 4.
- Prompts exposing test inputs or misstating the domain (rule B).
  - Finding: step 1 said "relative differences down to 1e-13", the spacing used in step 1 case 2 and
    inaccurate as a domain statement (closer constants are allowed and handled); step 4 said "This
    includes activities as small as 1e-11 Bq", pointing at the 199Au round trip (2.2e-11 Bq) and
    understating the domain (there is no lower limit other than the domain rule).
  - Cause: regime statements written from the test inputs.
  - Change: step 1: "Decay constants may be equal or arbitrarily close to one another, and they may span
    more than 20 decades in one chain"; step 4: the accuracy holds "for activities of any size within
    the domain below" (prompt and docstring). So that this statement is tested, step 4 case 1 gains a
    round trip near the bottom of the double range: a two-capture product with lam[k] = 1e-295 per s
    (sigma 1e7 b, 100 s at 1e12 n / (cm^2 s), 10 s of cooling), about 5e-302 Bq at the root, whose
    activity at flux_lo (about 5e-330 Bq) is below the smallest positive double while the product holds
    5e-35 of the target atoms (inside the domain); target from the 300-digit forward model.
  - Regression test: the reference and the second solution pass the new round trip; it covers the stated
    behaviour but is not a discriminator: a variant that forms lam[k] * x[k] before the logarithm also
    passes (brentq accepts the -inf at the lower end). The reference agrees with the second solution to
    about 1e-15 for chains whose constants are one to two rounding units apart; every prompt and
    docstring was reread for exact test inputs.
- Earlier functions named in the prompts (step dependencies).
  - Finding: step 3 named network_inventory only as a definition ("decays as in network_inventory") and
    step 4 did not say that A is computed with activation_inventory, although their references call
    these functions (step_dependencies [2] and [3]).
  - Cause: the prompts described the model by reference to the earlier steps without saying which
    function to use.
  - Change: step 3: "use network_inventory (step 2) for the periods"; step 4: "The predicted activity
    A(phi) is computed with activation_inventory (step 3)". step_dependencies unchanged ([], [2], [3]).
  - Regression test: the step references pass all their cases with only the listed dependencies in the
    context: step 2 with no earlier step, step 3 with step 2 only, step 4 with steps 2 and 3 (step 3
    calls step 2).
- Contracts did not match the prompts (verifier).
  - Finding: step 2 gave "column sums of branching <= 1" (the prompt accepts 1 + 1e-12) and omitted
    sum(n0) + t sum(source) <= 1e100; step 3 omitted sum(n0) <= 1e100, the capture_to rules, the history
    container and lam + sigma 1e-24 flux <= 1e10; step 4 omitted k, activity, monotonicity and the domain
    rule A(flux_lo) >= 1e-250 lam[k] sum(n0).
  - Cause: the contracts were written as short summaries.
  - Change: valid_input_ranges of steps 1-4 now list the prompts' ranges and conditions in full.
  - Regression test: every contract range was read against its prompt; crown_check format checks the
    contract keys.
- Duplicate mutant names (verifier).
  - Finding: step 1 and step 2 both had a mutant named squared_diagonal.
  - Cause: the same mistake was introduced in both steps under one name.
  - Change: renamed chain_squared_diagonal (step 1) and network_squared_diagonal (step 2); files and
    descriptions unchanged.
  - Regression test: crown_check mut reports both names, failing step 1 cases 3, 4, 5 and step 2 cases
    3, 4.
- Inaccurate test comments (verifier).
  - Finding: step 4 case 3 said the exact root next to |ln| = 1e-9 + 1e-12 was "about 2e-9 away" (the
    slope d ln A / d ln flux is 1.00 at 1e8 and 0.96 at 1e13, so about 1e-9); the header of general
    case 0 said "against the Bateman sum" for a network checked with mpmath expm; the file headers of
    tests/step_1.py and tests/step_4.py listed oracles those files do not use.
  - Cause: comments copied between files and not updated.
  - Change: comments corrected (step 4 case 3 now: the root next to |ln| = 1e-9 + 1e-10 is about
    1.1e-9 away); each test file header lists the oracles it uses.
  - Regression test: comments only; every case rerun.
- Metadata and source.md truthfulness.
  - Finding: verification_explanation described the old second solution and the old end band; the
    "Checks used while building the task" section described the old second solution without a version;
    several version-13 items had no cause or regression test.
  - Cause: the texts were not updated with this pass and the version-13 notes were incomplete.
  - Change: difficulty_explanation and verification_explanation rewritten for the current code and
    tests (new second solution, band 1e-10, general case 0, error-budget check, the runs below); the
    second-solution note in the checks section is labelled by version; the version-13 items now each
    give finding, cause, change and regression test. relevant_experience is still the author placeholder.
  - Regression test: the explanations were read against the code and the crown_check output below.

- Accuracy threshold and step 4 domain below 1e-300 atoms (verifier, round on this pass; blocking).
  - Finding: the relative 1e-10 rule of steps 1-3 applied to every exact entry of at least 1e-250 * S,
    and S has no lower bound. For S below about 2.2e-58 that threshold is a subnormal double, for S below
    about 2.5e-64 entries near it cannot be represented to a relative 1e-10 by any double (subnormals are
    spaced 4.9e-324), and for S below about 2.5e-74 it underflows to 0 (from there to about 7e-74 it rounds
    to the smallest subnormal, 4.9e-324). Example: chain_inventory([1], [1e-70],
    570) has the exact value 2.83e-318 > 1e-250 S = 1e-320, and both solutions return it with a relative
    error of 1.7e-7 (the nearest subnormal). The step 4 domain rule x[k](flux_lo) >= 1e-250 sum(n0)
    admitted such amounts, so its promise of 1e-8 "for activities of any size within the domain" failed:
    lam = [0, 1e-3], sigma = [1, 0], capture_to = [1, -1], n0 = [1e-70, 0], t_irr = 1e-225, t_cool = 0,
    activity 1e-320 Bq on [1, 1e4] (x[1](flux_lo) is about 1e-319, 1e-249 of the atoms): the reference
    returned 99.998862 against the exact root 99.998887 (relative error 2.5e-7), while the second
    solution raised ValueError.
  - Cause: the threshold was designed as a fraction of the inventory and validated only for inventories
    of order 1 and above (every tested S is 0 or at least 7e-5); the floor 1e-300 was written into the
    absolute branch of the rule but not into its relative threshold, nor into the step 4 domain rule.
  - Change: with F = max(1e-250 * S, 1e-300), every exact entry of at least F must have a relative error
    below 1e-10 and every other entry, exact zeros included, must lie within F (prompts of steps 1-3, the
    docstrings of chain_inventory, network_inventory and activation_inventory in problem.yaml, steps/,
    solution/, solution.py, second_solution.py and mutants/). The step 4 domain rule is now
    x[k](flux_lo) >= max(1e-250 * sum(n0), 1e-300), i.e. A(flux_lo) >= lam[k] * max(1e-250 * sum(n0),
    1e-300), evaluated in logarithms (prompt, Errors list, problem_io, function_header, contract,
    solution/step_4.py, solution.py, second_solution.py and, so that they stay the current reference with
    one mistake, the domain line of mutants/step_4_linear_response.py and
    mutants/whole_task_barn_in_square_metres.py). Every amount the rule then promises to a relative 1e-10
    is a normal double, and so is x[k] on the whole bracket of step 4 (A increases from flux_lo). The
    helpers _t_check of steps 1-3 and of tests/general.py (and _t_check_pair of step 3 case 3) use the
    same threshold; for every existing entry (S = 0 or S >= 7e-5) they decide exactly as before. New
    entries in existing cases (n_test_cases unchanged: 7, 6, 6, 6): step 1 case 0 and step 2 case 0, a
    single nuclide with n0 = 1e-70 at lam t = 500 (about 7e-288, relative rule) and 570 (about 3e-318, a
    subnormal above 1e-250 S but below F, absolute rule), and in step 2 a production of 1e-200 per unit
    time alone (S = 2e-200, amount 1.26e-200, relative rule); step 3 case 0, burnup of 1e-70 atoms at
    sigma phi t = 500 and 570; step 4 case 1, a round trip with n0 = [1e-70, 0], t_irr = 1e-200 s and
    t_cool = 10 s on [1, 1e4] (about 1e-294 atoms of the product at flux_lo, inside the domain; about
    1e-295 Bq at the root, target from the 300-digit forward model); step 4 case 5, the verifier's corner
    above (outside the domain now, with the activity between A(flux_lo) and A(flux_hi), so the domain rule
    is the only reason for the ValueError). Targets: closed forms n0 exp(-lam t), n0 exp(-sigma phi t) and
    q (1 - exp(-lam t)) / lam (mpmath at 40-50 digits where the tests already used it).
  - Second solution, defect found by the new step 4 round trip: its precision control (double the digits
    until two evaluations agree to 1e-13 or to 1e-3 of the floor) accepted 0 for the product amount,
    about 1e-294, because the closed form exp(-r1 t) - exp(-r0 t) at t = 1e-200 cancels to 0 at both 30
    and 60 digits (it raised ValueError: x[k](flux_lo) below the domain floor). It now carries, next to
    every coefficient of the generalized Bateman form, the sum of the magnitudes it was built from (the
    same operations on absolute values) and raises the precision until 1e5 eps times that magnitude sum
    is below 1e-14 of every amount or of the floor F (a running error bound, which cancellation cannot
    fool). The method (generalized Bateman solution in extended precision) is unchanged.
  - Regression tests: under the version-14 rule the new entry n0 = 1e-70, lam t = 570 of step 1 (and
    its counterparts in steps 2 and 3) promises a relative 1e-10 of the exact 2.832340e-318, which no
    double meets: in 60-digit mpmath both solutions are 1.66e-7 away (the nearest subnormal), and for
    n0 = 1e-80, lam t = 560 0.21 away; under the new rule it is an absolute-rule entry (within 1e-300) and
    passes. (Corrected in the verifier round below: the test entry itself does not separate the two
    rules, because its double target n0 exp(-570) rounds to the same subnormal as the output, so its
    computed relative difference is 0; the impossibility was shown with the mpmath comparison, not by a
    failing test case.) A variant of the reference with the version-14
    domain line (no floor) fails step 4 case 5 ("fewer than 1e-300 atoms of the measured nuclide at flux_lo
    must raise ValueError"). The second solution with the old precision control fails step 4 case 1
    (ValueError); with the running error bound it passes every case and agrees with the reference to
    3.7e-14 on the 60 random networks used above (the same set), to 1.4e-14 on 60 further random networks
    (2 to 30 nuclides, rates 1e-12 to 1e6 with many exactly equal or 1e-13 apart, some zero, production,
    non-topological order), to 2.8e-15 on 30 networks with inventories scaled to 1e-280 to 1e-50 and to
    2.2e-16 on 30 networks over times of 1e-250 to 1e-20 (relative above F; below F both agree to far
    better than F); on 12 of the tiny-time networks (up to 8 nuclides) both agree with mpmath expm at 600
    digits to 2.1e-16 or better above F. Both solutions recover the gold-monitor flux at 1e8, 1e12 and
    1.6454e14 to 5.1e-15 or better. The error budget of step 4 was rerun with the new tests (step-3 context
    scaled by 1 +- 9e-11, by opposite factors at the two ends, and by 1 +- 5e-12: every case passes; end
    cutoffs 0.85e-9 and 1.15e-9 fail case 3, 0.95e-9 and 1.05e-9 pass; the variant returning the root
    inside the end tolerance fails case 3). The step references pass with only their listed dependencies
    in the context (step 2 alone, step 3 with step 2, step 4 with steps 2 and 3).
  - Metadata: difficulty_explanation (the floor 1e-300), solution_explanation (the domain rule
    x[k](flux_lo) >= max(1e-250 sum(n0), 1e-300)) and verification_explanation (the tiny-inventory tests,
    the domain floor tested on both sides, the second solution's running error bound and its agreement
    figures) were updated; problem_description_main now says that the amounts are required with a small
    relative error down to the floor stated in each step (it said "every amount").

- Independent verifier, round after pass 2 (blocking): accuracy contract unattainable for tiny
  inventories, and the step 4 domain admitting a subnormal x[k].
  - Finding: the verifier reported that the relative 1e-10 rule of steps 1-3 applied to every exact entry
    of at least 1e-250 S with no lower bound on S (chain_inventory([1], [1e-70], 570): exact 2.832340e-318
    above 1e-250 S = 1e-320, relative error 1.66e-7 in both solutions; 0.21 for n0 = 1e-80, t = 560), and
    that the step 4 domain rule admitted lam = [0, 1e-3], sigma = [1, 0], capture_to = [1, -1],
    n0 = [1e-70, 0], t_irr = 1e-225 s, t_cool = 0, activity 1e-320 Bq on [1, 1e4], where the reference
    returned 99.998862 against the exact root 99.998887 (2.5e-7 > 1e-8) and the second solution raised
    ValueError.
  - Cause: the verifier read the files as they were before the item "Accuracy threshold and step 4 domain
    below 1e-300 atoms" above was applied. That item makes exactly the change the verifier asks for
    (F = max(1e-250 S, 1e-300) in the prompts and every docstring copy of steps 1-3, the step 4 domain rule
    x[k](flux_lo) >= max(1e-250 sum(n0), 1e-300) evaluated in logarithms in the prompt, Errors list,
    problem_io, function_header, contract, both solutions and both step 4 mutants, absolute-rule entries
    with n0 = 1e-70 at lam t = 570 in steps 1-3, and the verifier's corner as a ValueError entry of step 4
    case 5).
  - Confirmation on the current files (rerun in this round): in 60-digit mpmath, n0 = 1e-70 at lam t = 570
    and n0 = 1e-80 at 560 are below F = 1e-300 (absolute rule) and both solutions are within F; n0 = 1e-70
    at 500 (7.1e-288, relative rule) has a relative error of 9e-17 in both. The verifier's corner raises
    ValueError in both solutions (x[1](flux_lo) = 1e-319 < 1e-300). The reference with the version-14
    domain line (no floor) reproduces the verifier's numbers (99.99886201 against the exact
    99.99888672 = activity / (lam1 n0 sigma 1e-24 t_irr) in mpmath, valid since lam1 t_irr and the capture
    rate times t_irr are below 1e-200; relative error 2.47e-7) and fails step 4 case 5 ("fewer than 1e-300
    atoms of the measured nuclide at flux_lo must raise ValueError") while passing case 1. An AST
    comparison shows that every docstring of chain_inventory, network_inventory, activation_inventory
    and monitor_flux in steps/, solution/, solution.py, second_solution.py and mutants/ is identical to
    problem.yaml (function_header, and problem_io for monitor_flux). The only remaining statements of the
    old threshold 1e-250 S alone are in notes labelled historical.
  - Two statements of the item above corrected (source.md only): the S below which 1e-250 S underflows to
    0 is about 2.5e-74, not 5e-74 (from there to about 7e-74 it rounds to the smallest subnormal); and the
    new tiny-inventory entries of steps 1-3 do not fail under the old rule, as that item said, because
    their double target rounds to the same subnormal as the output (computed relative difference 0). The
    impossibility of the old promise was shown with the mpmath comparison only. Checked by running case 0
    of steps 1-3 with the reference under the old rule (it passes) and under the new rule (it passes).
  - Related clarification, step 4 domain boundary. Finding: the domain rule compares x[k](flux_lo), which
    step 3 gives only to a relative 1e-10, with max(1e-250 sum(n0), 1e-300), so two valid implementations
    can decide differently for inputs within 1e-10 of the bound; the end rule already states such a band,
    the domain rule did not. Cause: the band was written for the end rule only. Change: the prompt, the
    Raises section of the docstring (problem_io, function_header, steps/step_4.py, solution/step_4.py,
    solution.py, second_solution.py and both step 4 mutants) and the step 4 contract now say that
    ValueError and a result are both correct when x[k](flux_lo) lies within a relative 1e-10 of
    max(1e-250 sum(n0), 1e-300). Code and tests unchanged: no case lies in that band (the closest domain
    inputs hold about 1e-294 atoms, 1e6 times the floor, and about 1e-319 atoms, 1e-19 of it; the cases at
    1e-250 sum(n0) are about 5e-17 and 1e-80 of it). Regression: the docstring comparison above, each mutant
    still differs from its reference in the named line only (diff), and crown_check below.

Rerun: tools/crown_check.py, all modes, every case alone in a fresh process (single-threaded BLAS, -j 2),
after the last item above (final state of version 14; run again after the verifier round, with the same
results):
- FORMAT OK (zero problems; every general case calls monitor_flux).
- REF: step 1 7/7, step 2 6/6, step 3 6/6, step 4 6/6, general 2/2; solution.py also passes every step
  file (7/7, 6/6, 6/6, 6/6).
- SECOND: 7/7, 6/6, 6/6, 6/6, general 2/2.
- Mutants: float_bateman fails 5/7 (cases 1-5), chain_squared_diagonal 3/7 (3, 4, 5),
  transposed_branching 4/6 (1-4), network_squared_diagonal 2/6 (3, 4), barn_in_square_metres 4/6
  (0, 1, 2, 4), linear_response 4/6 (0, 1, 2, 3), whole-task barn mutant 2/2 general cases (0, 1).
- SHIFT +-1e-12: every step case passes, and every general case passes with every evaluated function
  shifted.
- Controls: all-None and all-zero stubs pass 0 cases in every step and in general.
- SUMMARY ALL OK.
- Error budget of step 4 (perturbed step-3 context) and the end-cutoff and root-returning variants: as
  given in the item "Accuracy threshold and step 4 domain below 1e-300 atoms" (rerun with the final tests
  of that item; the verifier round changed docstrings and texts only, no executable code or test).
- solution.py, the concatenated solution/step_*.py and second_solution.py were each run twice in clean
  processes with single-threaded BLAS (and solution.py once more with default threads) on the 238U series
  at 3e5 y, a random 12-nuclide network with production, the gold monitor at 1e14 and two flux
  inversions: identical outputs (bitwise) within each solution, and the reference and the concatenated
  steps identical to each other. In the verifier round, solution.py and second_solution.py were run twice
  each in clean processes with single-threaded BLAS on a random 12-nuclide network with production, a
  chain with two equal constants, the gold monitor at 1e14 and two flux inversions (gold at 1e12 and the
  round trip with 1e-70 target atoms): identical outputs (bitwise) within each solution; solution.py was
  also run alone on every case of tests/step_1.py to tests/step_4.py (7/7, 6/6, 6/6, 6/6).

## Version 15 (grading_fix: platform content check of upload v13)

Platform content check of upload v13: six blocking items ("Edge-case inputs" in steps 1-4, "Science
outweighs validation" in steps 2 and 3) and one advisory ("Scientific grounding", step 2). The science,
the reference algorithms (apart from the column normalization below), the targets of the kept cases, the
1e-10 and 1e-8 tolerances and the input domain are unchanged. edit_label stays grading_fix.

- Non-finite inputs (blocking, steps 1-4).
  - Finding: the error tests passed NaN or infinite inputs (step 1 case 6: a NaN decay constant, a NaN
    amount, a NaN time; step 2 case 5: a NaN time, a NaN amount, an infinite production rate; step 3
    case 5: a NaN cross section, an infinite duration; step 4 case 5: NaN or infinite activity, flux
    bounds and t_irr), and step 4 case 3 used np.inf internally as the direction of np.nextafter.
  - Cause: the prompts promised ValueError for values that are not finite, and the tests checked that
    promise; the platform now refuses every non-finite literal in tests, internal ones included.
  - Change: those entries were removed; np.nextafter(x, +-np.inf) became np.nextafter(x, +-1e308), which
    gives the same neighbour for every activity used (all far below 1e308). The clauses "or not finite",
    "(NaN and infinite values included)", "all values finite" and "all finite" were removed from the four
    prompts, the function headers, problem_io, the contracts' valid_input_ranges and every docstring copy
    (steps/, solution/, solution.py, second_solution.py, mutants/; still identical to problem.yaml).
    The solutions still reject non-finite values internally (unstated, untested); their error messages
    no longer mention finiteness.
  - Regression: crown_check format (non-finite-literal scan) reports nothing; step 4 case 3 passes with
    the reference and the second solution, and the end-cutoff variants below still fail it.
- Science outweighs validation (blocking for steps 2 and 3; applied to every step).
  - Finding: counted per step as the platform does (assertion sites, stated raises at half weight), the
    validation and computed-value sites were 12.5 / 13 (step 1), 14.5 / 11 (step 2), 22 / 10 (step 3) and
    18 / 9 (step 4); the shape assert of the _t_check helper was repeated in every case, the float check
    in every round trip, and the error lists tested several entries per stated rule.
  - Cause: each case had been made self-contained with its own copy of the helper, shape check included,
    and the error lists had grown with every review round.
  - Change: one contract check per step (steps 1-3, case 0: a one-nuclide input returns a numpy array of
    shape (1,), not a scalar; step 4, case 0: the flux is a Python float); elsewhere the helpers compare
    entry by entry after reshape(len(target)), so a wrong number of entries still fails. The error lists
    keep each stated rule once, at its boundary where it has one: step 1 nine entries (1.0001e10,
    sum(n0) = 1.01e100, t = -1 and 1.0001e20, ...), step 2 nine (column sum 1 + 1e-9, 1.0001e10,
    1.0001e20, sum(n0) + t sum(source) = 1.01e101, ...), step 3 thirteen (a decay-capture cycle with an
    empty history, positive sigma 1e-7 b, a duration of 1.0001e12 s, a history given as a set, ...),
    step 4 seven (t_irr = 0, k of a stable nuclide, k = 1.0 and True, activity 0, flux_lo = flux_hi,
    flux_lo = 9.99e-3), plus the three domain-floor cases, the step-3 data case and the two outside-interval
    checks of case 3. The old step 4 case 4 (two activities outside the interval) was dropped, since case 3
    tests the same rule at both ends (2e-9 outside, and the representable activity just beyond
    1e-9 + 1e-10). New computed cases, each with independent targets:
    - step 1 case 7: a stable member inside a chain (closed forms at 50 digits; the members after it stay
      exact zeros; conservation of the atoms above it); case 8: the 232Th series as a linear chain through
      212Po (1.6e-18 to 2.3e6 per s) after 1 d, 10 y and 1e10 y, against the Bateman sum at 600 digits,
      208Pb as the 600-digit remainder, and the total conserved;
    - step 2 case 6: a diamond network given out of topological order, every amount the sum over its decay
      paths of the product of fractions times the 600-digit Bateman amount along the path, and the balance
      (atoms lost = 30 % of the decays of nuclide 2); case 7: a chain fed by a constant source, the source
      treated as a parent of decay constant 0 in the 600-digit Bateman sum, and the saturation limit
      lam[k] x[k] = q at lam t >= 200; case 4 also checks conservation with a normalized column;
    - step 3 case 6: two-member activation with burnup of target and product in closed form (cobalt from
      sigma phi t = 1e-6 to 111, a product removed at the target's rate, where the textbook formula divides
      by zero) and the 198Au saturation activity equal to the production rate n0 sigma 1e-24 phi; case 7:
      decay competing with capture with decay branching (shares 0.3 lam : 0.7 lam : c of the losses, then
      cooling), also with a column summing to 1 + 1e-12, and conservation of the atoms; case 8: the order
      of the periods (irradiation then cooling against the reverse, and their ratio exp(-0.6) between two
      outputs at 2e-10) and a four-period history propagated with the closed-form two-member solution;
    - step 4 case 5: round trips on a cobalt monitor whose activity is the closed-form two-member solution
      with burnup (at 1e9, 1e13 and 3e14, slopes 1.00, 0.99 and 0.83; maximum of A near 2.7e15); case 6:
      the power laws at low flux on the gold monitor (10 times the 198Au activity gives 10 times the flux, 4
      times the 199Au activity twice the flux; burnup terms below 1e-14); case 7: 30 nuclides and
      sum(n0) = 1e100, the three-capture product of the capture chain inverted at 1e10 and 3e16 against the
      600-digit Bateman sum of the chain.
    n_test_cases: 9, 8, 9, 8 (were 7, 6, 6, 6).
  - Counts after the change (same counting): 5.5 / 19 (step 1), 5.5 / 18 (step 2), 7.5 / 19 (step 3),
    7.5 / 18 (step 4).
  - Regression: every new case passes with the reference and the second solution and fails at least one
    mutant (crown_check below); the error budget of step 4 was rerun on the new cases.
- Column sums in (1, 1 + 1e-12] (advisory, step 2).
  - Finding: such sums were accepted as rounding of 1 but used as given, so a column summing to
    1 + 1e-12 made more daughter atoms than parent decays.
  - Cause: the bound was written for the validation only.
  - Change: network_inventory divides such a column by its sum before building the rates, and
    activation_inventory normalizes the decay branching the same way before combining it with capture
    (reference, second solution through a helper _unit_columns, mutants rebuilt from the reference with
    their one change). Stated in the step 2 prompt and header ("divided by its sum before use, so that its
    fractions add up to exactly 1"), in the step 3 prompt and in the step 4 header and prompt
    ("normalized as in step 2"), and in the step 2 contract. Targets updated: step 2 case 4 now expects
    0.5 / s and (0.5 + 5e-13) / s of the decays (s the column sum) and 1 - exp(-2) for the column at
    1 + 1e-12. The change moves results by at most a relative 1e-12, below the stated 1e-10, so no
    tolerance can single it out; the tests use the normalized targets and check conservation.
- Zero outputs and stated shapes and endpoints (checked, no change needed beyond the above). The relative
  rule of steps 1-3 applies only to exact values of at least F = max(1e-250 S, 1e-300) > 0, so an exact
  zero is always under the absolute rule (tested: empty inventories, the members behind a stable one);
  the flux of step 4 is at least 1e-2. The (1,) shape of a one-nuclide result is now asserted in steps
  1-3; the inclusive ends n = 30, lam = 1e10, sigma = 1e-6 and 1e7 b, sum(n0) = 1e100 (now also in step
  4), t = 0 and 1e20, duration 0 and 1e12 s, flux 0 and 1e18, t_irr and t_cool = 1e9 s, t_cool = 0,
  flux_lo = 1e-2 and flux_hi = 1e18 are exercised with values.

Rerun after these changes (single-threaded BLAS, every case alone in a fresh process, -j 2):
- tools/crown_check.py, all modes: FORMAT OK; REF 9/9, 8/8, 9/9, 8/8 and general 2/2; SECOND the same;
  mutants: float_bateman fails 7/9 (cases 1-5, 7, 8), chain_squared_diagonal 4/9 (3, 4, 5, 8),
  transposed_branching 6/8 (1-4, 6, 7), network_squared_diagonal 3/8 (3, 4, 6), barn_in_square_metres 7/9
  (0, 1, 2, 4, 6, 7, 8), linear_response 7/8 (0-3, 5, 6, 7), whole-task barn mutant 2/2; SHIFT +-1e-12
  passes every step and general case; all-None and all-zero controls pass no case; SUMMARY ALL OK.
- Error budget of step 4 on the new test file: the step-3 context scaled by 1 + 9e-11, by 1 - 9e-11, and
  by 1 + 9e-11 below a flux of 10**10.5 and 1 - 9e-11 above it (and the reverse), which gives opposite
  signs at the two ends of the end-rule tests: every case passes; end cutoffs 0.85e-9 and 1.15e-9 fail
  case 3.
- Second solution against the reference (scratch comparison script, seeds 5, 6, 7): largest relative
  difference 2.7e-14 on 120 random networks with production, 2.8e-15 on 30 networks scaled to 1e-280 to
  1e-50, 2.2e-16 on 30 networks over times of 1e-250 to 1e-20 (no difference above 1.8e-29 F below F);
  problem.yaml now quotes 2.7e-14 (it quoted 3.7e-14 from an earlier run).
- solution.py (twice), the concatenated solution/step_*.py and second_solution.py (twice) on the 238U
  series, a random network, the gold monitor and two inversions in clean processes: bitwise identical
  outputs within each solution, and the reference identical to the concatenated steps.
