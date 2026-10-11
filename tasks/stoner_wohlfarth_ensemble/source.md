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
  agree to 2.2e-16 (rerun in version 14 on 49 fields at seven angles from 0.05 to 1.55,
  |h| up to 0.99 h_sw). At psi = 0 and psi = pi/2 they equal the closed forms
  (1 + h)**2 / 2 and (1 -+ |h|)**2 / 2.
- At psi = 0 the survival probability has the closed form
  P(h) = exp(-(f0 / rate) sqrt(pi / a) [erfc(sqrt(a) (1 + h)) - erfc(2 sqrt(a))]); the
  reference reproduces it to 2.5e-14 and its median (inverse erfc) to 1e-15 (rerun in version 14
  at six (a, f0 / rate) pairs including the four corners of the domain, nine fields each across
  the switching window).
- The second solution (second_solution.py) uses a different method in every step (state of
  version 14; numbering of version 13 onwards):
  - step 1: all equilibria from the quartic in tan(theta/2) (np.roots, Newton-polished), with
    h_sw from the fold of the equilibrium curve, written as the maximum over d in (0, pi/2) of
    sin(d) cos(d) / (cos(d) cos(psi) + sin(d) sin(psi)) with the fold condition
    cos(d)**3 cos(psi) = sin(d)**3 sin(psi) (no cancellation next to psi = 0 or pi/2); the
    reference brackets the followed minimum with Brent's method and uses the astroid formula;
  - step 2: bracketing of each extremum between its neighbours (Brent's method), with a
    golden-section search for a maximum that nearly merges with a minimum within rounding of
    |h| = h_sw; the reference uses the companion-matrix quartic in a shifted tan(theta/2);
  - step 3: one DOP853 ODE run for the escape integral; the reference uses adaptive quadrature;
  - step 4: the median by bisection on the dense output of one DOP853 run for the escape
    integral I(h); the mean from the exact statement that u = (f0 / rate) I(H_s) is an
    exponential variable with mean 1, so that with s = ln u the mean switching field is
    int h(s) exp(s - e^s) ds, h(s) being integrated as an ODE (dh/ds = -e^s / ((f0 / rate)
    Gamma / f0)) from the median at s = ln ln 2; the reference uses Brent's method and a
    composite Gauss-Legendre window on E[H_s] = h_sw - int P dh, and the test oracle one ODE
    run for I and int P dh;
  - step 5 (Brown's equation): lam1 from a finite-volume chain in theta (2000 and 4000 cells,
    exact two-point conductances, Richardson extrapolation) whose smallest nonzero eigenvalue is
    located by an inertia (Sturm) count of A - mu M with the pivots written in differential form,
    and tau_int from Garanin's quadrature with the inner integrals in closed form (erfi) and the
    outer ones by tanh-sinh quadrature in mpmath (30 digits); the reference uses Chebyshev
    inverse iteration with cumulative integrals and the test oracle an extended-precision
    Legendre matrix;
  - step 6 (ensemble): per-particle ODE runs with dense output and bisection on the ensemble
    magnetization and on the switched weight; the reference uses Brent's method over nested
    quadratures.
  Cross-checks of version 14 (each output against the other method, in clean processes): Brown
  step against the Legendre oracle on 26 points 6e-12 (lam1) and 4e-15 (tau_int) relative, and
  against the reference on 58 more points (random and corners, sigma from 1e-9 to 60) 6e-12 and
  2e-13; step 4 against the reference on 26 inputs (corners of the domain, psi = 0, pi/2, 1e-9 and
  pi/2 - 1e-9, random) 1.6e-15 (median) and 3.5e-14 (mean); step 3 next to the hard axis
  (psi = pi/2 - 1e-12 ... pi/2 - 0.3, the four corners of (a, f0 / rate) and a = 300 with
  1e9, fields 1e-6 ... 1 below h_sw) 2e-11, and at 60 random (psi, a, f0 / rate) with six fields
  each 2.3e-13; step 6 against the reference on 10 random ensembles 1.0e-14.

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| zero-temperature branch (step 1) | all minima of e on a 20000-point circle grid refined by bisection; with two minima the descending branch is the one with cos(theta) > 0 | exact to rounding for points at least 1e-3 from the jump |
| barriers (step 2) | minima and maxima of e from the same grid search, refined by bisection; original minimum = the one with cos(theta) > 0 | exact to rounding; closed forms at h = 0, psi = 0 and psi = pi/2 |
| survival probability (step 3) | closed form with erfc at psi = 0 and with erf / erfc at psi = pi/2 (barriers (1 -+ h)**2 / 2, h >= 0); for other angles an adaptive quadrature (purely relative tolerance) of the escape rate built on the grid barriers, at fields far below h_sw | about 1e-14 |
| median switching field (steps 4, 6) | closed form with the inverse erfc at psi = 0 (and the erf form at psi = pi/2); otherwise one Newton step on P = 1/2 from the returned value, with dP/dh = (f0 / rate) (Gamma / f0) P | must move it by less than 1e-7 |
| mean switching field (step 4) | E[H_s] = h_sw - int P dh, with P from the closed forms (psi = 0 and psi = pi/2, adaptive quadrature of P) or from one ODE run for the escape integral and int P dh on the grid barriers | about 1e-12 |
| h_c and h_half (step 6, general) | one Newton step on the expected ensemble magnetization (or on the switched weight minus 1/2), with analytic derivatives: dP/dh as above and dm/dh = sin(theta - psi)**2 / e''(theta) along each minimum | must move them by less than 1e-6; for one particle at psi = 0 both equal minus the closed-form median; for a psi = 0 particle (weight 2) with a psi = pi/2 particle (weight 1), whose magnetization is h in both minima, 4 P0(-h_c) - 2 - h_c = 0 and P0(-h_half) = 3/4 with the closed-form P0 |
| lam1 and tau_int (step 5) | Legendre expansion of Brown's operator (pentadiagonal, N = 60 + 3 sigma modes), solved in extended precision with mpmath by banded elimination: inverse iteration for lam1, one linear solve for tau_int | 30 + sigma (1 + abs(h))**2 / 2 digits; exact 1 and 1 at sigma = 0; converged (see version 14, item 1); agrees with the version-14 second solution (finite-volume inertia count, closed-form Garanin quadrature) to 6e-12 (lam1) and 4e-15 (tau_int) relative on 26 points that include every step 5 test point (up to the sign of h) |
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
extended-precision Legendre method (in version 13 the second solution, the same code as the test oracle)
agree to 1.2e-13 relative on the Brown points of the reproducibility run below; the version-14 second
solution (double-precision finite-volume inertia count for lam1, closed-form Garanin quadrature in
mpmath for tau_int) agrees with both to 6e-12. So 1e-5 leaves a wide margin for any valid
double-precision method while rejecting every mode confusion.

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

## Version 14 (grading_fix: authoring-guide conformance, pass 2)

Second conformance pass, after an independent verification of version 13. Step numbering as in version
13 (Brown's equation = step 5, ensemble = final step 6). No reference algorithm, test target, test input
or single-output tolerance was changed; the test changes are the float type check (item 6) only. Each
item gives the finding, its cause, the change, the regression test and what was rerun.

1. Second solution of Brown's equation was the test oracle.
   - Finding: second_solution.py computed brown_relaxation with the extended-precision Legendre code of the
     step 5 test oracle (_g5_band_solve / _g5_legendre were _t_band_solve / _t_brown with renamed
     identifiers), so the second solution passing step 5 said nothing beyond the oracle.
   - Cause: in version 3 the oracle code was reused as the "second method".
   - Change: a different method. lam1: finite-volume discretization of the Sturm-Liouville form in theta
     (cell masses int w sin(t) dt and exact two-point conductances 1 / int dt / (sin(t) w), both by
     16-point Gauss-Legendre per cell); the smallest nonzero eigenvalue of the chain A g = mu M g is
     located by Sylvester's law of inertia, counting the negative LDL^T pivots of A - mu M with each pivot
     written as d_i = c_i + t_i (t_i = c_{i-1} t_{i-1} / (c_{i-1} + t_{i-1}) - mu m_i), which removes the
     cancellation of the Laplacian rows, by multisection in log(mu); 2000 and 4000 cells, Richardson
     extrapolation of the O(cell**2) error. tau_int: Garanin's quadrature with the inner integrals in
     closed form (int w dz with erfi, int (z + h) w dz = w / (2 sigma)), Q taken from the end of the
     shallow well, and the outer integrals by tanh-sinh quadrature in mpmath (30 digits, more for
     sigma < 1), split at the barrier and next to both wells.
   - Regression test: against the oracle on 26 points that include every step 5 test point (up to the
     sign of h): 5.8e-12 (lam1) and 4.2e-15 (tau_int) relative; against the reference on 58 more points
     (sigma = 0, 1e-9, 1e-6, 1e-3, 0.02 and up to 60, |h| up to 0.9, 50 of them random): 5.8e-12 and
     1.8e-13; crown_check SECOND step 5 6/6. Without Richardson extrapolation (4000 cells alone) the lam1 error is up to 9e-7, which
     shows that the extrapolated chain, not a coincidence, carries the agreement. The oracle itself is
     converged: at (sigma, h) = (60, 0.9), (60, 0) and (45, 0.6), 80 more Legendre modes and 25 more
     digits change lam1 and tau_int by less than 1e-16 relative.
2. Second solution of step 4: the mean came from the test oracle's ODE, and the ODE could step over the
   switching window.
   - Finding: the mean switching field of second_solution.py was one DOP853 run for I(h) = int Gamma / f0
     and K(h) = int P dh with the same right-hand side, tolerances (rtol 1e-12, atol [1e-13 rate / f0,
     1e-15]) and stopping rule (f0 / rate I = 80) as the step 4 test oracle _t_mean. A randomized check
     against the reference (26 inputs) also showed in-domain failures: at psi = 0.7, a = 1000,
     f0 / rate = 1e13 it returned h_median = h_mean = -h_sw (error 3.9e-2), with errors of 3.5e-2 and
     2.2e-2 at two more inputs with a near 1000, and it raised ValueError at psi = 0.0083, a = 562.5,
     f0 / rate = 2.4e11.
   - Cause: the right-hand side was set to 0 for |h| >= h_sw. With a = 1000 the rate underflows to 0
     above the switching window, and one DOP853 step went from h = 0.046 to h = -h_sw, its last stage at
     -h_sw saw a zero rate, and the step was accepted although it jumped over the window. The ValueError
     came from escape_barriers (item 3).
   - Change: the mean now uses a different formulation. u = (f0 / rate) I(H_s) is an exponential
     variable with mean 1 (P = exp(-u)); with s = ln u the switching field is a monotone function h(s)
     with dh/ds = -e^s / ((f0 / rate) Gamma(h) / f0), and E[H_s] = int h(s) exp(s - e^s) ds (Gumbel
     weight; the atom at -h_sw has weight exp(-u_max)), integrated along the ODE for h(s) from s = -40
     up to s = 5, the direction in which the ODE contracts (a perturbation of h decays like
     e^-(s - s_start); integrated downwards from the median it diverged). The median and the starting
     field h(-40) come from bisection on the dense output of one run for I(h), now with a purely
     relative accuracy down to (f0 / rate) I ~ 1e-30. In every ODE of the second solution (steps 3, 4
     and 6) the rate at the ends of [-h_sw, h_sw] is now its limit from inside, so a step that jumps over
     the window ends where the rate is large and is rejected; a solver failure now raises instead of
     returning a value.
   - Regression test: against the reference on 26 inputs (the four corners of (a, f0 / rate) at
     psi = pi/2 and two at psi = 0, psi = 1e-9 and pi/2 - 1e-9, psi = pi/4, 0.7, 0.3 and 1.2, and 12
     random inputs): median 1.6e-15, mean 3.5e-14; crown_check SECOND step 4 5/5.
3. Second solution near |h| = h_sw and next to the hard axis.
   - Finding: escape_barriers of the second solution raised ValueError ("f(a) and f(b) must have
     different signs") for fields within rounding of +-h_sw (psi = 0.0083 above, psi = pi/2 - 1e-9), and
     its h_sw from the fold of the equilibrium curve, sin(2t) / (2 sin(t - psi)) at the fold angle t, lost up
     to 2.2e-12 next to psi = pi/2 (cancellation of t + pi/2 and t - psi + pi). With the end-point fix of
     item 2 a further defect appeared: next to the hard axis the maximum that merges with the other
     minimum at h_sw was misidentified within an ulp of h_sw (the slope next to a nearly degenerate
     minimum has an unreliable sign), the rate there jumped from about f0 to 2 f0, DOP853 stopped
     ("required step size is less than spacing between numbers") and survival_probability returned
     P = 1 from the unfinished run.
   - Cause: brackets built on slope signs right next to a minimum that is about to vanish; a fold formula
     written as a ratio of two small differences; the solver status was not checked.
   - Change: h_sw = max over d in (0, pi/2) of sin(d) cos(d) / (cos(d) cos(psi) + sin(d) sin(psi))
     (theta = -pi/2 + d), with the fold condition cos(d)**3 cos(psi) = sin(d)**3 sin(psi) solved by
     bracketing: sums of positive terms only, and an error in d enters h_sw at second order. A maximum
     whose bracket fails the slope-sign test is found by golden-section search of e between the two
     minima (e is unimodal there). Solver failures raise RuntimeError.
   - Regression test: h_sw against the astroid formula in 40-digit mpmath for psi from 1e-12 to
     pi/2 - 1e-12: 1.8e-16 (the old form: 2.2e-12); survival probability against the reference next to the
     hard axis (psi = pi/2 - 1e-12, -1e-9, -1e-6, -1e-3, -0.02, -0.07, -0.15, -0.3; the four corners of
     (a, f0 / rate) and a = 300 with 1e9; fields 1e-6 to 1 below h_sw): 2.0e-11; random check of steps 3
     and 6 (item 4); crown_check SECOND all steps.
4. Step 3 promised an accuracy that no double-precision method reaches next to the hard axis.
   - Finding: for psi close to pi/2 the lower barrier vanishes at h_sw (top barrier 0.0153 at
     psi = pi/2 - 0.02, 3e-4 at pi/2 - 1e-3), so the particle leaves within about rate / f0 below
     h_sw(psi), and P(h) there depends on h_sw - h to full relative precision. h_sw is known to about
     1e-16 in double precision, so at f0 / rate = 1e13 P is uncertain by up to about 1e-3 there. Example:
     psi = pi/2 - 1e-9, a = 207.8, f0 / rate = 5.99e12, h = h_sw - 1.16e-13: the reference gives
     P = 0.500045, the value from h_sw in 50-digit mpmath (the rate is 1 - 1.4e-9 there) is 0.499518; the
     old second solution gave 0.757.
   - Cause: the step 3 accuracy statement (absolute error below 1e-8 for every h) did not consider the
     conditioning of P at fields within about rate / f0 of h_sw when 0 < psi < pi/2 (at psi = pi/2, h_sw = 1
     is exact and at psi = 0 the rate at h_sw is negligible).
   - Change: the step 3 prompt, function_header (and its copies) and contract now say that for
     0 < psi < pi/2 the 1e-8 accuracy is not required at fields less than 1e-6 below h_sw(psi), and why.
     Bound: the rate is practically constant over the last 1e-6 below h_sw, so for h <= h_sw - 1e-6 the
     escape integral is at least that rate times 1e-6, and an error delta in h_sw changes P by at most
     max_u (u e^-u) delta / 1e-6 = 0.37 delta / 1e-6, below 1e-10 for delta = 2e-16. No test field lies in
     the excluded band (step 3 tests 0 < psi < pi/2 only at fields far below h_sw and at h_sw + 1e-6, where
     P = 1); the step 4 and step 6 outputs are fields with absolute tolerances and are not affected.
   - Regression test: next to the hard axis reference and second solution agree to 2.0e-11 at every field
     at least 1e-6 below h_sw (item 3). Random check of step 3: 60 draws (psi from {0, pi/2, 1e-9,
     pi/2 - 1e-9, uniform}, a log-uniform in [40, 1000], f0 / rate log-uniform in [1e5, 1e13]), six fields
     each (the median, +-0.002, +-0.01 and 0): reference and second solution agree to 2.3e-13 (this also
     holds inside the excluded band, where both use h_sw to about 1e-16 and share its rounding). Random
     check of step 6: 10 ensembles of one to three particles (angles including 0, pi/2 and 1.561): 1.0e-14.
5. Hidden test inputs in the step 5 prompt (rule B).
   - Finding: the prompt gave lam1 "about 1e-24 at sigma = 60, h = 0", the shallow-well weight "about
     1e-94 at sigma = 60, |h| = 0.9" and "the hardest corner sigma = 60 with |h| = 0.9 included"; (60, 0)
     and (60, +-0.9) are test inputs.
   - Cause: magnitude examples written with the test corners, and a leftover of the removed time-budget
     sentence.
   - Change: "on the stated domain it can be as small as about 4.5e-24, for the largest sigma in weak
     fields", "a relative Boltzmann weight near exp(-4 sigma |h|), which can be as small as about 1e-94 on
     the stated domain", and "a relative error below 1e-5 on the whole stated domain". Both magnitudes hold
     over the whole domain: lam1 is smallest at sigma = 60, h = 0 (4.51e-24, oracle) and
     exp(-4 * 60 * 0.9) = 1.6e-94. difficulty_explanation and background.md give the same magnitudes.
   - Regression test: grep of the prompts and docstrings for the step 5 test tuples finds none.
6. Float type check of step 5.
   - Finding: the step 5 tests required type(x) is float, the other steps isinstance(x, float), while all
     prompts say "Python float".
   - Cause: inconsistent test helpers.
   - Change: the step 5 tests use isinstance(x, float) like every other step; problem_description_main
     states that wherever an output is a Python float any instance of float is accepted (numpy.float64
     included). Step 2 keeps its own stated rule for a scalar h (Python float or numpy floating).
   - Regression test: crown_check REF, SECOND and SHIFT step 5.
7. Mutants that were not the current reference with one error (rule E).
   - Finding: step_1_global_minimum was a separate grid-search solver; step_1_wrong_astroid_power also
     dropped the exact h_sw = 1 at psi = pi/2; step_5_brown_asymptote was a separate function with two
     errors (asymptotic lam1 and tau_int = 1 / lam1) and a fallback lam1 = 1 when the asymptote vanished.
   - Cause: written before the reference reached its present form.
   - Change: all three rebuilt from the current per-step reference with one change each:
     global_minimum switches to the other minimum (the global one) as soon as h < 0 instead of at
     h = -h_sw; wrong_astroid_power changes only the outer exponent (-2/3 for -3/2); brown_asymptote
     replaces the inverse iteration by Brown's two-well high-barrier asymptote for lam1 and keeps the exact
     Garanin tau_int. Descriptions in problem.yaml updated. The other step mutants and the whole-task
     mutant already differ from the current reference by one change (checked with diff).
   - Regression test: crown_check MUT (below).
8. Stale or unsupported statements (rule C).
   - Finding: the "Checks used while building the task" list described the second solution of version 13
     and gave unrerun numbers; the target table did not mention the psi = pi/2 closed forms of steps 3
     and 4, and the step 5 row did not state the oracle's mode count or convergence; the step 5 tolerance
     paragraph called the second solution the oracle's code; verification_explanation described the
     second solution of version 13; the version 13 items 3 to 7 had no cause or regression test.
   - Cause: texts not updated with the code.
   - Change: all corrected (numbers rerun in this version, labelled as such). difficulty_explanation now
     says that the step 5 prompt states the Sturm-Liouville form and Garanin's quadrature (the easing of
     version 12, kept by the author). Completion of the version 13 items:
     - item 3 (zero relative tolerance): cause, np.allclose(..., rtol=0.0) used to write a pure absolute
       tolerance; regression test, crown_check format (no zero-tolerance pattern) and REF/SECOND step 2.
     - item 4 (two computed outputs compared at the single-output tolerance): cause, the comparisons were
       written before the shift check existed; regression test, SHIFT +-1e-12 passes step 3 case 3 and
       step 5 cases 3 and 4, and each output is still checked against its own target.
     - item 5 (ValueError for a non-finite h untested): cause, the invalid-input list predates the
       statement; regression test, step 5 case 5 with (10, nan) passes for the reference and the second
       solution.
     - item 6 (stale docstrings): cause, mutant docstrings copied before the version 12 tolerance change;
       regression test, ast.get_docstring of every scaffold, solution, second-solution and mutant function
       equals the function_header docstring and problem_io equals the final header (rerun in version 14:
       0 mismatches).
     - item 7 (metadata): cause, the version 12 problem.yaml predates the metadata fields of the current
       skeleton; regression test, crown_check format (metadata keys present).
   - Regression test: the numbers in the checks list, the target table and the step 5 tolerance
     paragraph were rerun in version 14 (lam1 at sigma = 60, h = 0: 4.51e-24; tau_int there 2.2e23;
     lam1 tau_int = 0.0229 at sigma = 60, h = 0.9; the asymptote 1.7 % above lam1 at sigma = 60, h = 0).
     Numbers inside the notes of versions 3 to 13 are historical and refer to their version.
9. Checked and unchanged.
   - Every general case calls ensemble_switching (the final function) and checks it against an
     independent target (closed forms or the Newton-step oracle), rule A.
   - step_dependencies list exactly the earlier functions named in each prompt: step 2 [1], step 3 [2],
     step 4 [2, 3], step 5 [], step 6 [1, 3].
   - relevant_experience keeps the placeholder for the author.

Rerun for version 14 (after items 1-8):
- Reproducibility: solution.py and second_solution.py each run twice in clean processes (python3 -I) on
  21 inputs covering all six functions; the two runs of each gave byte-identical outputs.
- tools/crown_check.py, all modes, every case alone in a fresh process (-j 2, single-threaded BLAS):
  - FORMAT OK.
  - REF steps 1-6: 5/5, 5/5, 6/6, 5/5, 6/6, 8/8; REF general (solution.py): 4/4.
  - SECOND steps 1-6: 5/5, 5/5, 6/6, 5/5, 6/6, 8/8; SECOND general: 4/4.
  - MUTANTS: step 1 global_minimum fails cases [0, 1, 2, 3], wrong_astroid_power [1]; step 2
    barrier_from_global_minimum [0, 1, 2, 3]; step 3 lower_route_only [0, 1, 4]; step 4
    wrong_mean_formula [0, 1, 2, 3]; step 5 brown_asymptote, tau_from_slowest_mode, missing_half and
    doubled_energy [0, 1, 2, 3, 4] each; step 6 athermal_particles [0-6]; whole-task
    whole_task_barriers_from_global_minimum fails all 4 general cases.
  - SHIFT +1e-12 and -1e-12: every case of steps 1-6 passes (5/5, 5/5, 6/6, 5/5, 6/6, 8/8), and every
    general case with every evaluated function shifted (4/4).
  - CONTROLS: all-None and all-zero stubs pass 0 cases in every step and in general.
  - SUMMARY ALL OK.
