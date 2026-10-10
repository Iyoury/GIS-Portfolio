# Source

- E. C. Stoner and E. P. Wohlfarth, "A mechanism of magnetic hysteresis in
  heterogeneous alloys", Philosophical Transactions of the Royal Society of London A
  240, 599-642 (1948). The single-domain model, the switching field and the hysteresis
  loops for different field angles.
- L. Neel, "Theorie du trainage magnetique des ferromagnetiques en grains fins avec
  applications aux terres cuites", Annales de Geophysique 5, 99-136 (1949), and W. F.
  Brown, "Thermal fluctuations of a single-domain particle", Physical Review 130,
  1677-1686 (1963). Thermally activated escape over the anisotropy barrier with an
  attempt frequency (Neel-Brown model).
- M. P. Sharrock, "Time dependence of switching fields in magnetic recording media",
  Journal of Applied Physics 76, 6413-6418 (1994). Dependence of the switching field and
  of the coercivity on the sweep rate and the thermal stability ratio K V / (k_B T).
- W. T. Coffey and Yu. P. Kalmykov, The Langevin Equation, 4th ed. (World Scientific, 2017),
  chapter on the Brown model: Legendre-polynomial (matrix continued fraction) solution of the
  axially symmetric Fokker-Planck equation, smallest eigenvalue and integral relaxation time.
- D. A. Garanin, "Integral relaxation time of single-domain ferromagnetic particles", Physical
  Review E 54, 3250-3256 (1996). Exact integral formula for the integral relaxation time.
- C. Tannous and J. Gieraltowski, "The Stoner-Wohlfarth model of ferromagnetism",
  European Journal of Physics 29, 475-487 (2008). Teaching account of the model and the
  astroid.

The combination of steps (zero-temperature branch, barriers, survival probability under
a linear sweep, switching-field statistics, dynamic coercivity of a weighted ensemble),
the escape over both maxima without back jumps, and the parameter ranges are our own
design.

Checks used while building the task:
- The barriers were computed by three independent methods (quartic in a shifted
  tan(theta/2) solved by companion-matrix eigenvalues, bracketing of each extremum
  between known neighbours, and a grid search on the circle refined by bisection); they
  agree to better than 1e-15. At psi = 0 and psi = pi/2 they equal the closed forms
  (1 + h)**2 / 2 and (1 -+ |h|)**2 / 2.
- At psi = 0 the survival probability has the closed form
  P(h) = exp(-(f0 / rate) sqrt(pi / a) [erfc(sqrt(a) (1 + h)) - erfc(2 sqrt(a))]); the
  reference reproduces it to about 1e-15 and its median to about 1e-14.
- The second solution (second_solution.py) uses a different method in every step of the
  sweep chain (numbering of version 13):
  - step 1: all equilibria from the quartic in tan(theta/2) (np.roots, Newton-polished), with
    h_sw from the fold of the equilibrium curve; the reference brackets the followed minimum
    with Brent's method and uses the astroid formula;
  - step 2: bracketing of each extremum between its neighbours (Brent's method); the reference
    uses the companion-matrix quartic in a shifted tan(theta/2);
  - step 3: one DOP853 ODE run for the escape integral; the reference uses adaptive quadrature;
  - step 4: one ODE run with dense output for the escape integral and for int P dh, median by
    bisection; the reference uses Brent's method and a composite Gauss-Legendre window;
  - step 6 (ensemble): per-particle ODE runs with dense output and bisection on the ensemble
    magnetization and on the switched weight; the reference uses Brent's method over nested
    quadratures.
  For Brown's equation (step 5) it uses the same extended-precision Legendre code as the test
  oracle, so there the independent comparison is reference against oracle. Reference and second
  solution agree to 2e-12 or better (reproducibility run of version 13).

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| zero-temperature branch (step 1) | all minima of e on a 20000-point circle grid refined by bisection; with two minima the descending branch is the one with cos(theta) > 0 | exact to rounding for points at least 1e-3 from the jump |
| barriers (step 2) | minima and maxima of e from the same grid search, refined by bisection; original minimum = the one with cos(theta) > 0 | exact to rounding; closed forms at h = 0, psi = 0 and psi = pi/2 |
| survival probability (step 3) | closed form with erfc at psi = 0; for other angles an adaptive quadrature (purely relative tolerance) of the escape rate built on the grid barriers | about 1e-14 |
| median switching field (steps 4, 6) | closed form with the inverse erfc at psi = 0 (and the erf form at psi = pi/2); otherwise one Newton step on P = 1/2 from the returned value, with dP/dh = (f0 / rate) (Gamma / f0) P | must move it by less than 1e-7 |
| mean switching field (step 4) | E[H_s] = h_sw - int P dh, with P from the closed form (psi = 0) or from one ODE run for the escape integral on the grid barriers | about 1e-12 |
| h_c and h_half (step 6, general) | one Newton step on the expected ensemble magnetization (or on the switched weight minus 1/2), with analytic derivatives: dP/dh as above and dm/dh = sin(theta - psi)**2 / e''(theta) along each minimum | must move them by less than 1e-6; for one particle at psi = 0 both equal minus the closed-form median; for a psi = 0 particle (weight 2) with a psi = pi/2 particle (weight 1), whose magnetization is h in both minima, 4 P0(-h_c) - 2 - h_c = 0 and P0(-h_half) = 3/4 with the closed-form P0 |
| lam1 and tau_int (step 5) | Legendre expansion of Brown's operator (pentadiagonal), solved in extended precision with mpmath by banded elimination: inverse iteration for lam1, one linear solve for tau_int | 30 + sigma (1 + abs(h))**2 / 2 digits; exact 1 and 1 at sigma = 0 |
| sanity values | barriers 1/2 at h = 0; P = 1 above h_sw and 0 below -h_sw; P depends on f0 and rate only through f0 / rate | exact |

## Version 3 changes (difficulty increase)

New final step 6, brown_relaxation(sigma, h): the smallest nonzero eigenvalue lam1 and the
integral relaxation time tau_int of Brown's axially symmetric Fokker-Planck equation,
0 <= sigma <= 60, |h| <= 0.9, both to 1e-8 relative.
- Reference: Sturm-Liouville form with the Boltzmann weight w = exp(-beta E); inverse iteration
  in which every step is a cumulative integral of w g (flux) and of 2 F / ((1 - z^2) w),
  represented by piecewise Chebyshev interpolation in theta (panels keep their own relative
  accuracy, so wells whose weights differ by 1e-94 are both resolved). The flux is taken from
  the nearer end on each side of the barrier to avoid cancelling the two well masses, and the
  iteration starts from a step at the barrier so that a slow mode living in the shallow well is
  not lost below rounding. lam1 is the Rayleigh quotient of positive integrals; tau_int is
  Garanin's formula 2 int Q^2 / ((1 - z^2) w) dz / int (z - <z>)^2 w dz.
- Test oracle and second solution: Legendre expansion W = sum b_l P_l, pentadiagonal operator
  from the recurrences of z P_l and (1 - z^2) P_l', solved in extended precision (mpmath,
  30 + sigma (1 + |h|)^2 / 2 digits) by banded elimination: inverse iteration for lam1, a
  linear solve for tau_int.
- The two agree to 3e-15 (lam1) and 2e-13 (tau_int) on 48 points (random and corners);
  lam1 = 1, tau_int = 1 at sigma = 0; Brown's asymptote is approached at sigma = 60; a third
  check by symmetric finite differences reproduces lam1 = 5.150415 at sigma = 60, h = 0.9,
  where a smooth starting vector converges instead to an intrawell mode of the deep well.

Version 4: the sigma = 0 checks of step 6 use 1e-9 instead of 1e-12 (inside the stated 1e-8; a 1e-12 bound left no room for a result perturbed at the 1e-12 level).

Version 5 (content-check fixes):
- Steps 3-5: the domain 40 <= a <= 1000, 1e5 <= f0 / rate <= 1e13 is now enforced (ValueError
  outside it) and stated in the prompts and docstrings. In it f0 / rate * int Gamma / f0 over
  (-h_sw, h_sw) is at least about 430 for every psi (checked on a psi grid at a = 1000,
  f0 / rate = 1e5), so the survival probability always crosses 1/2 inside (-h_sw, h_sw) and the
  median is a proper root; the tests cover all four corners of the domain.
- psi = pi/2: h_sw = 1 is taken exactly (cos(pi/2) = 6e-17 in floating point would put it 2e-11
  below 1, which matters for f0 / rate up to 1e13); quadrature nodes that round onto +-h_sw are
  moved one ulp inside. New closed-form tests at psi = pi/2 (barriers (1 -+ h)^2 / 2) in steps 3
  and 4, and an ensemble with a particle at psi = pi/2 and one with a zero weight in step 5.
- Step 2 returns numpy float scalars for a scalar h (tested with isinstance(..., np.floating));
  the near-edge fields are 1.1e-3 inside the range, where the 1e-10 accuracy is promised.
- Step 1 is tested exactly at |h + h_sw| = 1e-3 (psi = 0, h = -0.999 and -1.001).
- Step 3 is tested at h = +-h_sw exactly (psi = 0 and pi/2).
- Step 5: fewer invalid-input entries, three new computed cases.
- Step 6: the whole computation (panel construction, inverse iteration, Rayleigh quotient,
  Garanin's tau_int) is in the body of brown_relaxation; h = -0.9 is tested against the oracle
  and against h = +0.9; the docstring gives the units as 1/tau_N and tau_N, respectively.
- Step 5 (v5 follow-up): the ValueError test now covers every invalid input named in the prompt
  (equal-shaped 2-D arrays, empty arrays, infinite and NaN weights, rate = 0, negative f0, NaN a,
  infinite rate) next to the domain bounds.

Version 7: step 6 states a time budget (each call within 30 s on one CPU core; the reference takes
about 1.2 s at sigma = 60, |h| = 0.9 and the second solution about the same), and the step-6 and
general tests assert it on every checked call.

Version 8: v7 reached 1/8 with 3 of 8 attempts cut by the time limit. Failures were spread over
all six steps and were mostly numerical-method misses while chasing 1e-10. The required
accuracies are relaxed by a factor of 100, everywhere at once (prompts, docstrings and tests); the
science and the test cases are unchanged:
- steps 1-3: 1e-10 to 1e-8 absolute;
- step 4: 1e-9 to 1e-7;
- step 5: 1e-8 to 1e-6;
- step 6: 1e-8 to 1e-6 relative.
Every mutant still fails by many orders of magnitude more than the new bounds.

Version 9: step 5 also rejects a negative psi, a = 1001 and f0 / rate = 1e4 (the lower and upper bounds that were not yet tested).

Version 10 (content-check fixes):
- Step 5: two new computed test cases, so that the science outweighs the input validation:
  - one particle at psi = 0 over four (a, f0/rate) pairs, where h_c = h_half = minus the
    closed-form median;
  - dynamic coercivity of a two-particle ensemble: both fields fall with slower sweeps and with
    lower a, plus an oracle check of the hot case.
- Step 2 rejects negative fields beyond -h_sw (h = -1 at psi = 0, an array with -0.6 at psi = pi/4).
- Step 1: the near-jump checks use |h + h_sw| = 1.1e-3, safely inside the accurate range, instead
  of the boundary value 1e-3.

## Version 11

v10 was in band (1/8) but held for an admin: 2 of 8 attempts were flagged "possibly timed out". Only step 6
had a time budget, checked after the call returned, so a very slow solution could hold up the grader
until its global time-out instead of failing. Now every step states a budget per call (steps 1-3: 10 s,
steps 4-6: 30 s; the reference needs at most 0.1 s, 3 s, 6 s and 1.4 s) and the tests interrupt a call
once it exceeds its budget (SIGALRM, one second of grace; measured only where no alarm is available), so
a too-slow solution fails the check promptly. A solution made to sleep 60 s per call now fails step 5
after 32 s. No other change.

## Version 12

v11 went 0/8 (best 5/6 steps, mean 0.75): slightly too hard. Small easing, no change to the science or the
test cases:
- per-call time budgets doubled (steps 1-3: 10 s -> 20 s, steps 4-6: 30 s -> 60 s), in prompts and tests;
- step 6 prompt now gives the two exact forms the reference relies on: the self-adjoint Sturm-Liouville
  form d/dz[(1 - z^2) w g'] = -2 lam w g for lam1, and Garanin's quadrature for tau_int
  (2 int Q^2 / ((1 - z^2) w) dz / int (z - <z>)^2 w dz, Q = int_{-1}^{z} (z' - <z>) w dz');
- step 6 relative accuracy relaxed from 1e-6 to 1e-5 (2.1e-5 for the h -> -h comparison of two outputs).
All mutants still fail by orders of magnitude more than the new bounds.

Version 12 content-check fixes: the step 6 prompt now states the 60 s budget used by the tests (one line had
kept 30 s); the step 2 array tests also check that both outputs are ndarrays with a floating dtype.

## Version 13 (grading_fix: authoring-guide conformance)

Steps 5 and 6 change places in this version (item 12): Brown's equation (brown_relaxation) is now step 5
and the ensemble step (ensemble_switching) is the final step 6. The notes of versions 3 to 12 above use
the old numbering (Brown's equation = step 6, ensemble = step 5); the notes below use the new one. No
change to the science, the reference algorithms, the domain or any single-output tolerance; no existing
target was changed (items 11 and 12 add targets). Each item gives the finding, its cause, the change and
the regression test.

1. Time budgets in tests and prompts.
   - Finding: every step test file and tests/general.py wrapped the evaluated functions in a SIGALRM
     `_t_budget` wrapper (clock reads with perf_counter, an elapsed-time assert), the Brown and general
     tests also asserted the elapsed time of each brown_relaxation call, and the step prompts stated
     "Each call must finish within 20 s / 60 s on one CPU core".
   - Cause: the per-call budgets of versions 7, 11 and 12, added to fail slow solutions promptly. The
     authoring guide forbids timing asserts, clock reads and per-call time budgets in the specification.
   - Change: wrapper, clock reads and elapsed asserts removed from all seven test files; the budget
     sentence removed from every prompt. The Brown sentence also named the hardest corner of the
     domain, which is kept without a time: "relative error below 1e-5 on the whole domain, the hardest
     corner sigma = 60 with |h| = 0.9 included". The large cases (sigma = 60, |h| = 0.9;
     f0 / rate = 1e13) stay as value tests.
   - Regression test: crown_check format reports no clock/timing pattern; every reference case passes.
2. Test-case format and self-contained cases.
   - Finding: markers carried descriptions ("# --- test case 0: ... ---") and all helpers (grid
     search, closed forms, the mpmath Legendre oracle) sat in a preamble before case 0, so a case run
     alone failed with NameError (30 of 38 step and general cases in a per-case run of version 12).
   - Cause: the files were written for whole-file execution.
   - Change: tools/selfcontain.py split every file on its markers; every case now starts with
     "# --- test case N ---" (N = 0..n-1), keeps its description as a comment and carries its own
     imports and the helpers it uses. No step case was added, dropped or merged; n_test_cases is
     unchanged (steps 1-4: 5, 5, 6, 5; Brown, now step 5: 6; ensemble, now step 6: 8).
   - Regression test: crown_check runs every case alone in a fresh process (results below).
3. Zero relative tolerance.
   - Finding: step 2 case 1 used np.allclose(..., rtol=0.0, atol=1e-8) for the closed-form barriers at
     psi = 0 and pi/2.
   - Change: written as np.max(np.abs(np.asarray(out) - np.asarray(target))) <= 1e-8, the same check.
4. Comparisons between two computed outputs.
   - Finding: step 3 case 3 compared P(f0 = 1e9, rate = 1) with P(f0 = 1e11, rate = 100) to 1e-8, and
     Brown case 3 compared (sigma, h) = (60, 0.9) with (60, -0.9) to 1e-5 relative, although each output
     is only promised to that accuracy (inconsistent with the stated tolerance).
   - Change: 2e-8 and 2.1e-5 (the latter already used by Brown case 4). Each output is still checked
     against its independent target at 1e-8 and 1e-5.
5. Stated but untested behaviour.
   - Finding: brown_relaxation promises ValueError for a non-finite h; only a non-finite sigma was tested.
   - Change: (sigma, h) = (10, nan) added to Brown case 5.
6. Stale docstrings.
   - Finding: the three Brown mutants and the old whole-task mutant still stated "relative error below
     1e-6" (before version 12); second_solution.py had a one-line docstring for brown_relaxation.
   - Change: 1e-5 in the mutant docstrings; second_solution.py carries the function_header docstring.
     All docstrings of scaffolds, solutions, second solution and mutants equal the function_header
     docstrings, and problem_io equals the final step's header (checked with ast.get_docstring).
7. Metadata: subfield, tags, expert_time_estimate_hours, relevant_experience (placeholder for the
   author), difficulty_explanation, solution_explanation, verification_explanation, author and
   affiliation added; edit_label grading_fix.

Items 8-12 come from an independent verification of the first version-13 draft.

8. Step 2 return type for a scalar field.
   - Finding: SHIFT +-1e-12 failed step 2 case 0 (4/5 pass), only on `isinstance(low, np.floating)`; no
     value check was near its bound (targets 1/2, tolerance 1e-8).
   - Cause: the test and the text required numpy floating scalars, an incidental dtype convention that
     the guide says not to test; the shift check returns a numpy float as a Python float, and a correct
     solution returning Python floats would have lost the whole case.
   - Change: case 0 accepts float scalars (Python float or numpy floating). The step 2 prompt, the
     function_header (and its copies in steps/, solution/, solution.py, second_solution.py and the
     mutants) and the contract output_shape now say "float scalars (Python float or numpy floating)
     for a scalar h"; float numpy arrays for array input are unchanged.
   - Regression test: SHIFT +1e-12 and -1e-12 pass step 2 5/5.
9. Wrong claims about the second solution.
   - Finding: verification_explanation said the second solution used "bracketing instead of the
     quartic, Gauss-Legendre and ODE runs instead of library quadrature", and the checks above said it
     used "an adaptive Gauss-Legendre rule ... for the survival probability".
   - Cause: description not checked against the code. The composite Gauss-Legendre window is in the
     reference (solution/step_4.py); in step 1 the second solution uses the quartic and the reference
     brackets; the second solution's survival probability is a DOP853 ODE run.
   - Change: both texts now describe the second solution step by step (see "Checks used while building
     the task").
   - Regression test: checked against the code: np.roots in its branch_magnetization, brentq in its
     escape_barriers, solve_ivp (DOP853) in its steps 3, 4 and 6, the mpmath Legendre code in its
     brown_relaxation; np.polynomial.legendre.leggauss appears only in the reference step 4.
10. step_dependencies.
   - Finding: the step 2 prompt names branch_magnetization (step 1) with step_dependencies [], and the
     step 4 prompt names survival_probability (step 3) and escape_barriers (step 2) with [2].
   - Change: step 2 [1], step 4 [2, 3].
   - Regression test: the dependencies of every step now list exactly the earlier functions its prompt
     names (step 3: [2]; Brown step 5: []; ensemble step 6: [1, 3]).
11. Constant outputs earning credit.
   - Finding: a stub that validates its inputs and returns (1.0, 1.0) passed Brown cases 0 (only the
     sigma = 0 limit, whose targets are 1 and 1), 4 (h -> -h comparison without a target) and 5
     (ValueError). The same audit on step 2 found that a stub returning 1/2 for both barriers passed
     case 0 (h = 0 only) and the ValueError case 4.
   - Change: Brown case 0 also checks (sigma, h) = (0.5, 0.6) against the Legendre oracle (approach to
     free diffusion: lam1 = 0.84900, tau_int = 1.17080); Brown case 4 checks (45, 0.6) and (45, -0.6)
     against the oracle and keeps the 2.1e-5 comparison of the two outputs. Step 2 case 0 also checks
     scalar fields h = +-0.05 against the closed forms (1 + h)**2 / 2 (psi = 0) and (1 -+ |h|)**2 / 2
     (psi = pi/2). Case counts unchanged.
   - Regression test: the (1.0, 1.0) stub now passes only the Brown ValueError case (1/6) and the
     constant-1/2 stub only the step 2 ValueError case (1/5); the oracle agrees with the reference to
     1.3e-15 at (0.5, 0.6) and 5e-14 at (45, +-0.6). A stub returning only the stated boundary values
     P = 1 (h >= h_sw) and P = 0 (h <= -h_sw) still passes step 3 case 2, which tests exactly that
     stated behaviour, and the ValueError case (2/6); this was left as is.
12. The final step did not deliver the integrated result.
   - Finding: the final step (brown_relaxation) used none of steps 1-5; the integrated result of the
     chain is the dynamic coercivity of the ensemble. The general tests and the whole-task mutant
     (2 sigma in Brown's equation) only exercised brown_relaxation, so no whole-task test ran the
     hysteresis and thermal-switching chain on solution.py, and difficulty_explanation called Brown's
     relaxation part of the chain.
   - Cause: Brown's equation was appended as a new final step in version 3.
   - Change: steps 5 and 6 swapped (scaffolds, per-step solutions, step tests and step mutants
     renamed; solution.py and second_solution.py define brown_relaxation before ensemble_switching);
     their prompts, headers and tests are otherwise unchanged except for items 8-11 and the step 6
     prompt opener ("Back to the field sweep of steps 3 and 4."). problem_io is the ensemble_switching
     header. problem_description_main, difficulty_explanation and solution_explanation say that step 5
     stands apart from the chain. tests/general.py now has four self-contained ensemble_switching cases
     with independent targets and parameters not used in the step tests:
     (0) one particle at psi = 0, a = 250, f0 / rate = 1e10: h_c = h_half = minus the closed-form
     median (0.723368);
     (1) psi = 0 (weight 2) with psi = pi/2 (weight 1), a = 70, f0 / rate = 1e8: the hard-axis particle
     has the magnetization h in both minima and has left just below h = 1, so h_c solves
     4 P0(-h_c) - 2 - h_c = 0 and P0(-h_half) = 3/4 with the closed-form P0 (h_c = 0.526881,
     h_half = 0.520070);
     (2) four particles at 0.1, 0.45, 0.95, 1.35 rad with weights 1, 2, 1.5, 0.5, a = 120,
     f0 / rate = 1e8, Newton-step oracle;
     (3) six angles from 0 to pi/2 with weights sin(psi) (zero weight at psi = 0), a = 45,
     f0 / rate = 1e12, Newton-step oracle.
     The new whole-task mutant whole_task_barriers_from_global_minimum is solution.py with the barriers
     measured from the deeper minimum, an error that propagates through the survival probability to both
     ensemble fields; the old doubled-energy mutant is kept as the step 5 mutant doubled_energy.
   - Regression test: reference and second solution pass all four general cases (closed forms matched to
     1e-15, Newton steps below 3e-14); the whole-task mutant fails all four (it returns h_c = h_half =
     1 for the psi = 0 particle instead of 0.723).

Why step 5 (Brown's equation) keeps a relative tolerance of 1e-5 (2.1e-5 for the h -> -h comparison of two
outputs). lam1 runs from 1 (sigma = 0) down to about 4.5e-24 (sigma = 60, h = 0), and tau_int from 1 up to
about 2e23; only a relative criterion is meaningful over such a range, and 1e-5 relative resolves every
value in it equally well (five significant digits of a rate of 1e-24 are as sharp a statement as five
digits of a rate of 1). The difficulty of the step is the exponential scale separation, not extra digits:
lam1 must be found next to eigenvalues of order 100, and in a strong field the slow mode lives in a well
whose relative Boltzmann weight is near 1e-94. A method that does not handle this returns the wrong mode
or loses the shallow well and is wrong by factors, not by 1e-5: the scientific mutants miss by a factor 2
(missing 1/2 of Brown's equation), by a factor of about 40 in tau_int at sigma = 60, h = 0.9 (tau_int taken
as 1 / lam1, whereas lam1 tau_int = 0.023 there) and by 1.7 % or more (Brown's high-barrier asymptote at
sigma = 60, h = 0). The reference (double-precision Chebyshev Sturm-Liouville iteration) and the
extended-precision Legendre method (second_solution.py, the same code as the test oracle) agree to
1.2e-13 relative on the Brown points of the reproducibility run below, so 1e-5 leaves a wide margin for
any valid double-precision method while rejecting every mode confusion.

Rerun for version 13 (after items 8-12):
- Reproducibility: solution.py and second_solution.py each run twice in clean processes (python3 -I) on
  22 inputs covering all six functions; the two runs of each gave byte-identical outputs. Reference
  against second solution on these inputs: 1.1e-16 (step 1), 2.8e-17 (step 2), 1.1e-14 (step 3),
  1.0e-14 (step 4), 1.2e-13 relative (step 5) and 1.7e-12 (step 6), all absolute except step 5.
  (The second solution's DOP853 runs emit a scipy RuntimeWarning, "invalid value encountered in scalar
  divide", from the step-size control near h = h_sw, where the escape rate over f0 is about 4e-174 at
  psi = 0, a = 100 and the error norm underflows to 0/0; it does not affect the results.)
- tools/crown_check.py, all modes, every case alone in a fresh process (-j 2):
  - FORMAT OK.
  - REF steps 1-6: 5/5, 5/5, 6/6, 5/5, 6/6, 8/8; REF general (solution.py): 4/4; solution.py also passes
    every step file.
  - SECOND steps 1-6: 5/5, 5/5, 6/6, 5/5, 6/6, 8/8; SECOND general: 4/4.
  - MUTANTS: step 1 global_minimum fails cases [0, 1, 2, 3], wrong_astroid_power [1]; step 2
    barrier_from_global_minimum [0, 1, 2, 3]; step 3 lower_route_only [0, 1, 4]; step 4
    wrong_mean_formula [0, 1, 2, 3]; step 5 brown_asymptote, tau_from_slowest_mode, missing_half and
    doubled_energy [0, 1, 2, 3, 4] each; step 6 athermal_particles [0-6]; whole-task
    whole_task_barriers_from_global_minimum fails all 4 general cases.
  - SHIFT +1e-12 and -1e-12: every case of steps 1-6 passes (5/5, 5/5, 6/6, 5/5, 6/6, 8/8).
  - CONTROLS: all-None and all-zero stubs pass 0 cases in every step and in general.
  - SUMMARY ALL OK.
