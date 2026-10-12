# Source

- V. Ambegaokar and B. I. Halperin, "Voltage due to thermal noise in the dc Josephson
  effect", Physical Review Letters 22, 1364-1366 (1969). Exact mean voltage of the
  overdamped junction with thermal noise.
- Yu. M. Ivanchenko and L. A. Zil'berman, "The Josephson effect in small tunnel
  contacts", Soviet Physics JETP 28, 1272-1276 (1969). The same problem from the
  Fokker-Planck equation.
- R. L. Stratonovich, Topics in the Theory of Random Noise, Vol. II (Gordon and Breach,
  1967). Stationary current of Brownian motion in a tilted periodic potential.
- P. Reimann, C. Van den Broeck, H. Linke, P. Hanggi, J. M. Rubi and A. Perez-Madrid,
  "Giant acceleration of free diffusion by use of tilted periodic potentials", Physical
  Review Letters 87, 010602 (2001), and Physical Review E 65, 031104 (2002). Exact
  effective diffusion coefficient and the giant-diffusion peak near the critical tilt.
- B. Lindner, M. Kostur and L. Schimansky-Geier, "Optimal diffusive transport in a tilted
  periodic potential", Fluctuation and Noise Letters 1, R25-R39 (2001). Diffusion and
  first-passage moments in the tilted washboard.
- S. Lifson and J. L. Jackson, "On the self-diffusion of ions in a polyelectrolyte
  solution", Journal of Chemical Physics 36, 2410-2414 (1962). Diffusion in an untilted
  periodic potential.
- H. Risken, The Fokker-Planck Equation, 2nd ed. (Springer, 1989), chapters 10 and 11.
  Kramers equation, Brinkman hierarchy, Brownian motion in periodic potentials, matrix
  continued fractions.
- H. D. Vollmer and H. Risken, "Eigenvalues and their connection to transition rates for
  the Fokker-Planck equation with a periodic potential", Zeitschrift fur Physik B 52,
  259-266 (1983). Noisy underdamped junction (RCSJ) by matrix continued fractions.
- W. C. Stewart, Applied Physics Letters 12, 277 (1968), and D. E. McCumber, Journal of
  Applied Physics 39, 3113 (1968). The RCSJ model and the parameter beta_c.

The combination of steps (exact voltage and differential resistance, effective phase
diffusion, location of the giant-diffusion peak, inversion of the voltage for the noise
temperature and its thermometric sensitivity, criterion current and voltage noise of a series array with junction-
dependent noise strengths and time scales), the parameter ranges and the accuracy
requirements are our own design.

Consistency checks run while building the task:
- The reference (Stratonovich and Reimann double integrals: periodic trapezoid rule in
  the position, composite Gauss-Legendre in the shift graded toward the tilt boundary
  layer) and the high-precision Fourier continued fractions agree to about 1e-13
  (relative) on 40 random points over |i| <= 10, 0.02 <= theta <= 50, including
  exponentially small voltages (down to about 1e-17) and diffusion coefficients.
  Rechecked in version 12 on a new set of 40 random points (8 of them below the critical
  current at theta <= 0.05): largest relative difference 4.4e-15 for v, 3.6e-14 for dv/di and
  1.3e-15 for D_eff, with v and D_eff down to about 3e-25.
- The thermometric sensitivity kappa = d ln theta / d ln v of the reference (since the second round of
  version 12) agrees with the extended-precision central difference of the continued-fraction voltage
  (the step 4 test helper, 60 + 1.8 / theta digits) to 1.9e-11 relative on a geometric 14 x 10 grid of
  (i, theta) over 0.01 <= i <= 10, 0.02 <= theta <= 50, where kappa ranges from 0.0102 (at i = 0.01,
  theta = 0.02) to 2.42e7 (at i = 10, theta = 0.02); the reference before the second round was off by up
  to 2.2e-5 there (at i = 10, theta = 0.02), beyond the 1e-6 tolerance. (Measured again when the second
  round was completed; an interim record of this round gave 4.8e-11 and 4.4e-5 for a grid whose points
  were not recorded.)
- On a geometric 25 x 50 grid over 0.01 <= i <= 10, 0.02 <= theta <= 50, d ln v / d ln i = i r_d / v of
  the reference is never below 1 (minimum 1.000000000016 at i = 0.01, theta = 50, i.e. in linear
  response). For the array, J (dV/dJ) / V = sum r_k r_d,k / sum r_k v_k / i_k is then at least 1 as well,
  so a relative voltage error enters j_star (step 5) divided by a factor of at least 1.
- The giant-diffusion maximum is flat: d^2 ln D_eff / di^2 at the peak is about -216 at theta = 0.02,
  -60 at 0.05, -22 at 0.1, -3.8 at 0.3 and -0.26 at theta = 1 (reference, central difference with
  h = 1e-3). A relative error of D_eff bounded by delta but otherwise arbitrary in i can therefore move the
  apparent maximum by up to sqrt(4 delta / |d^2 ln D_eff / di^2|), i.e. 4.3e-6 to 1.2e-4 for
  delta = 1e-9, which is why the step 3 prompt states that the 1e-9 of effective_diffusion does not
  locate i_peak to 1e-6.
- The second solution (since version 12) uses a third exact representation for steps 1 to 5,
  independent of both the reference and the test targets. The Jacobi-Anger expansion
  exp(z cos x) = sum_m I_m(z) e^{imx}, z = 1 / theta, makes every shift integral of the
  Stratonovich and Reimann formulas elementary: v = a / Phi with
  Phi = I_0(z)**2 + 2 sum_{n>=1} (-1)**n I_n(z)**2 b**2 / (b**2 + n**2), a = |i|, b = a / theta
  (the Ambegaokar-Halperin voltage as a Bessel series), and D_eff = theta <W Gp**2 Gm> / Phi**3,
  a mean of products of Bessel-Fourier series (W = exp(-z cos x), Gp and Gm the shift integrals
  of I_+ and I_- up to a common factor) formed by exact integer convolutions. The alternating
  sums cancel by up to about 0.87 / theta digits and are evaluated in mpmath at
  30 + 1.8 / theta digits (at theta = 0.02, D_eff was found to need about 50 digits, i.e. about
  16 + 0.87 / theta). Derivatives in i and theta are analytic in the series. The diffusion peak
  is found by a grid scan and Illinois regula falsi on d ln D / di, the noise temperature by
  Illinois regula falsi in 1 / theta, the criterion current by safeguarded Newton iterations on
  ln V. On 16 nonzero-bias (i, theta) points of the step 1 tests and the 16 of step 2 (v and
  D_eff down to about 1e-17) it reproduces the continued-fraction values to the last
  double-precision bit, at i = 0 it reproduces 1 / I_0(1/theta)**2 and the Lifson-Jackson D to
  2.3e-15 relative, and it agrees with the reference to 4e-15 in i_peak and to about 1e-13 relative
  in every step 3 to 5 output. For step 6 it solves the Kramers hierarchy by a matrix
  continued fraction in a real (1, cos p x, sin p x) Fourier basis with Hermite functions
  centred at the overdamped mean velocity (N = 130, |p| <= 64), evaluated at the complex bias
  i + 1e-20 j: the real part is v and the imaginary part divided by 1e-20 is dv/di
  (complex-step differentiation, no difference error). Before version 12 the second solution
  solved the Fourier-Galerkin system of the test targets by tridiagonal elimination (see the
  version 12 note).

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| v and dv/di (step 1) | decaying solution of the Fourier recurrence of the stationary Fokker-Planck equation as a continued fraction in extended precision (30 + 1.8 / theta digits), v = i + Im r_1, dv/di from the differentiated recurrence | about 1e-15 relative, also for exponentially small v |
| v at low noise (steps 1, 4, 5) | single-integral form with the inner integral done exactly, v = theta (1 - exp(-2 pi i / theta)) / int_0^{2 pi} I_0(2 sin(y/2) / theta) exp(-i y / theta) dy, mpmath quadrature at 40 digits | agrees with the continued fraction to 1e-12 |
| dv/di at i = 0 (step 1) | linear response, 1 / I_0(1/theta)**2 | exact |
| v and dv/di, closed form (step 1 case 5, since version 13) | v = (theta / pi) sinh(pi i / theta) / |I_{j i / theta}(1 / theta)|**2, modified Bessel function of imaginary order in mpmath at 40 digits; dv/di by mpmath numerical differentiation of the same closed form at 60 digits | exact closed form (no recurrence, no quadrature); on the 8 tested points it reproduces the continued-fraction v exactly in double precision and dv/di to 8e-15 relative |
| D_eff (steps 2, 3, 5) | homogenization: D = theta int (1 + chi')**2 p dx with the corrector chi and the density p both from continued fractions in extended precision; Fourier convolution for the average | about 1e-15 relative |
| D_eff at i = 0 (step 2) | Lifson-Jackson, theta / I_0(1/theta)**2 | exact |
| diffusion peak (step 3) | one Newton step on dD/di = 0 with central differences (h = 2e-5) of the continued-fraction D must move i_peak by less than 1e-6; D lower at i_peak +- 0.05; d_peak against the continued-fraction D at the returned i_peak | truncation of the differences below 1e-8; d_peak relative 1e-9 |
| noise temperature (step 4) | inside the range the voltage is generated at a known theta by the continued fraction (or the Bessel integral) and the returned theta must recover it. The endpoint rule is defined with the voltages of mean_voltage (step 1) at theta = 0.02 and 50, so the endpoint inputs (v_lo and v_hi themselves in case 2, the probes beyond them in case 3) are built from mean_voltage, which each case first checks against the continued fraction to its stated 1e-9 | relative 1e-7 + 2e-9 |kappa| (kappa the extended-precision sensitivity at the known theta): theta is defined through mean_voltage, whose relative error (below 1e-9) enters theta multiplied by |kappa|; probes 0.9e-11 (accepted) and 1.1e-11 (rejected) beyond the endpoint voltage (since the second round of version 12; before: relative 1e-7, see the version 12 note) |
| thermometric sensitivity kappa (step 4) | central difference of ln v in ln theta with step 1e-15 theta, the continued fraction evaluated at 60 + 1.8 / theta digits, at the theta returned by the candidate (since the second round of version 12; before: at the known theta) | about 1e-15 relative; compared with relative tolerance 1e-6 |
| array criterion (step 5) | one Newton step on V(J) = v_crit with V and dV/dJ from the continued fractions must move j_star by less than the stated relative tolerance; r_diff and s_v compared with the continued-fraction sums at the returned j_star; for one junction j_star is known from the Bessel-integral voltage | j_star relative 1e-9 + 2e-9 / s, s = d ln V / d ln J of the continued-fraction array voltage (since the second round of version 12; before: relative 1e-9, see the version 12 note); r_diff and s_v relative 1e-8 |
| array in linear response (step 5 case 6, since version 13) | v_crit = 1e-9: every junction in equilibrium, r_diff = sum r_k / I_0(c_k / theta0)**2, j_star = v_crit / r_diff, s_v = 2 theta0 r_diff (Lifson-Jackson D_k and the Einstein relation, i.e. the Johnson-Nyquist noise of the differential resistance), 40 digits | nonlinear corrections of relative order J**2, about 1e-18 at J of order 1e-9 |
| v and dv/di with capacitance (step 6, general) | Kramers equation in Hermite functions centred at the shifted velocity v0 = 0.6 i (N = 120, |p| <= 60), its own matrix continued fraction; dv/di by the fourth-order five-point central difference (h = 1e-3); beta_c = 0 against the overdamped continued fraction; in the general test also rcsj_voltage(i, theta, 0) against mean_voltage | against N = 220, |p| <= 110 with the same stencil on the 13 tested points: v within 5e-5 of its tolerance, dv/di within 1.7e-7 relative at the smallest tested voltage (rounding in the shifted basis) and within 1e-12 elsewhere; against the analytic dv/di of the reference within about 3e-7 at that voltage (stencil truncation 2.6e-7 plus rounding) and 1e-9 elsewhere (version 12; before version 12: N = 150, |p| <= 72 and a central difference h = 1e-4, see the version 12 note) |

## Version 3 changes

Editorial clean-up only: wording of the source notes and of the problem description.
No change to any prompt requirement, function header, solution, test or mutant.

## Version 4 changes

Step 4: the endpoint tolerance is now part of the stated contract (a voltage at most a
relative 1e-11 beyond the voltage at theta = 0.02 or 50 returns that endpoint; farther out
raises ValueError) and is tested on both sides (1e-12 accepted, 1e-9 rejected). Step 5: the
validation tests cover 2-D arrays, empty arrays, a NaN entry, a zero resistance, theta0 / c_k
above 50 and v_crit = 0, and three more computed-value cases were added.

## Version 5 changes (difficulty increase)

New final step 6: exact stationary voltage and differential resistance of the noisy
junction with capacitance (RCSJ), |i| <= 2.5, 0.1 <= theta <= 2, beta_c = 0 or 0.1 to 2.
The reference solves the Kramers equation in Hermite functions (velocity) and Fourier
modes (phase) by a matrix continued fraction over the Hermite index (N = 110, |p| <= 56),
with dv/di carried analytically through the same recursion; it is converged to 2e-9
relative against N = 260, |p| <= 140 on 29 points of the range (random and corners). The test oracle expands in Hermite functions centred at a different velocity
v0 = 0.6 i (a different truncated system) with N = 220, |p| <= 110; the two agree to 6e-10
relative over a 112-point grid of the range. The second solution centres the basis at the
overdamped mean velocity. A sparse direct solve of the same hierarchy and a continued
fraction over the Fourier index were tried and rejected: both lose accuracy in the running
state at weak noise and strong inertia. Step 1 also tests the endpoint i = -10.

## Version 6 changes

Step 4: the endpoint tolerance is applied exactly as stated, as a relative difference of the
voltage to the endpoint voltage (no longer through ln v), and is tested just around the
cutoff at both endpoints (0.9e-11 accepted, 1.1e-11 rejected). Step 6: non-finite input,
i = -2.6, theta = 2.01 and beta_c = 2.01 are tested as ValueError. Step 2: i = -10.01 too.

## Version 7 changes

Step 6 runtime cut from about 150 s to under 15 s: the reference truncation is N = 110,
|p| <= 56 (converged to 2e-9 relative, tolerance 1e-7) and dv/di is computed analytically
by differentiating the continued-fraction recursion (agrees with finite differences to
their own accuracy), so one call takes about 0.25 s instead of 7 s. The test oracle uses
N = 150, |p| <= 72 in the shifted basis (unchanged to 5e-9 against N = 220, |p| <= 110) and
a central difference for dv/di.

## Version 8 changes

Step 6 test comment corrected to the truncation actually used by the oracle (N = 150,
|p| <= 72, central difference). Step 4: the accepted endpoint-tolerance cases also check
kappa against the independent kappa at the endpoint.

## Version 9 changes

Step 6 now states three things the tests rely on: (1) the tolerance on v is
1e-7 * |v| + 1e-12 (a purely relative 1e-7 on voltages of order 1e-9 demanded an absolute
accuracy near 1e-16 that the prompt did not announce); (2) in the bistable range the required
v is the average over the unique stationary distribution, not the voltage of the running or
locked branch, which a finite time integration started in one branch does not reach; (3) a
time budget of 30 s per call (the reference needs about 0.25 s), asserted on every call
in the step 6 and whole-task tests.

## Version 10 changes

Step 6: the 30 s per-call time budget stated in version 9 was enforced in the tests (every
rcsj_voltage call in tests/step_6.py and tests/general.py was wrapped in a perf_counter timer
with an assert), and the parenthetical "(the tests make about 15 calls)" was removed from the
step 6 prompt as test-strategy wording. (Recorded here in version 12; before that this version
was mentioned only in the version 11 introduction. Both the budget sentence and the timing
assert were removed again in version 11.)

## Version 11 (grading_fix: authoring-guide conformance)

Audit of every file against the current authoring guide. The science, the reference algorithms, the
targets, the tolerances of the individual outputs and the domain are unchanged. (The version 10 upload
asserted the step 6 time budget in the tests and removed the sentence about the number of test calls from
the step 6 prompt; see the version 10 note above, added in version 12. The Cause and Regression test lines
marked "added in version 12" below complete entries that originally gave only the change.)

- Per-call time limit (step 6 prompt, tests/step_6.py, tests/general.py).
  - Finding: the step 6 prompt required "Each call must finish within 30 s on one CPU core", and the step 6
    and whole-task tests wrapped every rcsj_voltage call in a perf_counter timer with an assert.
  - Cause: added in versions 9 and 10 to announce and enforce a budget; the guide forbids timing asserts,
    clock reads and stated per-call budgets (runtime must simply be reasonable).
  - Change: the sentence is removed from the prompt (it stated no problem size); the timing wrapper
    _t_timed is removed and the cases call rcsj_voltage directly. Every value check is kept.
  - Regression test: crown_check format reports no clock/timing pattern; reference and second solution pass
    every step 6 and general case. For information, the reference needs about 0.4 s per call with
    single-threaded BLAS here; with a multi-threaded OpenBLAS on a fully loaded shared machine one call took
    up to about 90 s (thread oversubscription), so the checks were run with OPENBLAS_NUM_THREADS=1.
- Exact floating-point equality (tests/step_1.py case 2, tests/step_6.py case 2).
  - Finding: "out[0] == 0.0" and "_t_z[0] == 0.0" on the computed voltage at i = 0, and the prompts and
    headers said "at i = 0 return v = 0.0 exactly".
  - Cause: zero-tolerance comparison of a computed float.
  - Change: at i = 0 the voltage must satisfy |v| <= 1e-12, stated in the step 1 and step 6 prompts, the
    function headers, problem_io, the scaffolds, the solutions and the mutant docstrings. For step 6 this
    is the absolute part of the stated tolerance 1e-7 |v| + 1e-12 at v = 0; for step 1 the same 1e-12 is
    used (equilibrium, detailed balance gives v = 0; 1e-12 accepts round-off of any method while the
    forward-slip mutant still gives order 0.1 at theta = 1).
  - Regression test: step 1 case 2 and step 6 case 2 pass for the reference, the second solution and the
    +-1e-12 shift; the forward_slips_only mutant still fails step 1 case 2.
- Comparison of two computed outputs (tests/step_1.py case 3).
  - Finding: the mirror-symmetry check compared mean_voltage(0.8, 0.07) with mean_voltage(-0.8, 0.07) to a
    relative 1e-12 (v) and 1e-10 (r_d), tighter than the stated accuracies 1e-9 and 1e-8.
  - Cause: a valid method that evaluates negative bias separately may differ by up to the sum of the two
    stated errors.
  - Change: relative 2e-9 for v and 2e-8 for r_d (twice the single-output tolerances).
  - Regression test: reference, second solution and shift pass case 3.
- Tolerance stated in the prompt applied exactly (tests/step_6.py case 1).
  - Finding: the beta_c = 0 check used a pure relative 1e-7 on v.
  - Cause (added in version 12): the check predates the absolute part of the tolerance stated in version 9.
  - Change: it now uses the stated 1e-7 |v| + 1e-12 (no practical change at v of order 0.5).
  - Regression test (added in version 12): step 6 case 1 passed for the reference, the second solution and
    the +-1e-12 shift in the version 11 crown_check run listed below.
- Hidden test inputs in the step 6 prompt.
  - Finding: the prompt named the bistable example "beta_c = 2, theta = 0.1, i = 0.7" and the small-voltage
    example "i = 0.05, theta = 0.1, beta_c = 2", both of which are test inputs.
  - Cause (added in version 12): the example coordinates had been taken from the test points when the
    sentences were written in version 9.
  - Change: the statements are kept without the coordinates ("in the bistable range (strong inertia and
    weak noise) ... the stationary state can be dominated by the locked state"; "below the critical current
    at weak noise and strong inertia v can be very small (of order 1e-9 and below)"). The reference gives
    v = 4.55e-9 at (0.05, 0.1, 2) and 6.5e-10 at (0.01, 0.1, 2), so the second statement holds.
  - Regression test (added in version 12): a search of every prompt, problem_io and function header for the
    coordinates of the step 6 and general test points finds none (run in version 12); REF and SECOND passed
    unchanged in the version 11 run.
- Stated but untested error rules.
  - Finding: the prompts promise ValueError for a non-finite i (step 4), a non-finite theta0 (step 5) and a
    non-finite theta or beta_c (step 6); none of these inputs was tested. A NaN theta0 or theta passes a
    plain range comparison, so these are distinct behaviours.
  - Cause (added in version 12): the error lists of the prompts had been written without a matching test
    entry for every clause.
  - Change: one entry each added to the existing error cases: (nan, 0.1) in step 4 case 4, theta0 = nan in
    step 5 case 5, (0.5, nan, 1.0) and (0.5, 0.3, inf) in step 6 case 3. Reference and second solution raise
    ValueError for each.
  - Regression test (added in version 12): those error cases passed for the reference and the second
    solution and failed for the all-None and all-zero controls in the version 11 run.
- Test-case format and self-contained cases (all test files).
  - Finding: the markers read "# --- test case N: description ---" instead of "# --- test case N ---", and
    the imports and target helpers (_t_cf, _t_v_bessel, _t_rel, _t_check_peak, _t_kappa, _t_check_temp,
    _t_array, _t_check_array, _t_kramers_v, _t_kramers, _t_check_rcsj) were defined once before case 0.
  - Cause: the files were written for a whole-file run, before the per-case parser format.
  - Change: markers normalized to "# --- test case N ---" (0..n-1) with the descriptions kept as comments;
    every case now carries its own imports and the helpers it uses (tools/selfcontain.py, then checked by
    hand); the provenance comments are kept as a header comment before case 0. No case was dropped, merged
    or split: n_test_cases stays 5, 5, 4, 5, 6, 4 and general has 2 cases.
  - Regression test: before the change, crown_check (every case alone in a fresh process) passed only the
    error-input case of each step and 0/2 general cases (NameError on the shared helpers); after it, all.
- Test header comment.
  - Finding: the garbled recurrence "c_{n+1} + (2 theta n + 2 j i) c_{n-1}... in the form".
  - Cause (added in version 12): an editing error in an earlier version of the header comment.
  - Change: it now reads c_{n+1} + a_n c_n - c_{n-1} = 0 with a_n = 2 theta n + 2 j i (the recurrence the
    helpers solve).
  - Regression test (added in version 12): comment only; no check changed (REF and SECOND unchanged).
- source.md target table.
  - Finding: the step lists claimed that the overdamped targets are used in the general tests, and the
    step 6 / general RCSJ oracle had no row.
  - Cause (added in version 12): the table was not updated when version 5 made rcsj_voltage the final step.
  - Change: general.py is described as testing only rcsj_voltage, and a row for the RCSJ oracle was added.
  - Regression test (added in version 12): documentation only; checked against tests/general.py by hand.
- Metadata.
  - Finding: the current template asks for subfield, tags, expert_time_estimate_hours, relevant_experience,
    difficulty_explanation, solution_explanation, verification_explanation, author and affiliation.
  - Cause (added in version 12): the task predates these template fields.
  - Change: the fields were added to problem.yaml (relevant_experience as a placeholder for the author);
    edit_label is now grading_fix.
  - Regression test (added in version 12): crown_check format reported no missing metadata field.
- Checked without change: every solution import (numpy, mpmath, scipy.optimize.brentq) is declared in
  required_dependencies; the per-step solutions do not redefine earlier step functions and do no top-level
  computation; every step has a complete contract and at least one named mutant, and there is a whole-task
  mutant; no np.allclose with rtol = 0 in the tests.

Rerun: tools/crown_check.py, all modes, every case alone in a fresh process (OPENBLAS_NUM_THREADS=1):
- FORMAT OK.
- REF: step 1 5/5, step 2 5/5, step 3 4/4, step 4 5/5, step 5 6/6, step 6 4/4, general 2/2 (solution.py
  also passes every step file).
- SECOND: 5/5, 5/5, 4/4, 5/5, 6/6, 4/4, general 2/2.
- Mutants: forward_slips_only fails 4/5 (cases 0-3), kramers_rate 4/5 (0-3), einstein_relation 3/5 (1-3),
  einstein_peak 3/4 (0-2), kramers_inversion 3/5 (0-2), inverted_sensitivity 4/5 (0-3),
  arrhenius_sensitivity 4/5 (0-3), squared_voltage_scale 4/6 (1-4), no_inertia 3/4 (0-2),
  velocity_noise_not_scaled 3/4 (0-2), missing_thermal_velocity 3/4 (0-2), whole-task
  velocity_noise_not_scaled 2/2 general cases.
- SHIFT +-1e-12: every case passes.
- Controls: all-None and all-zero stubs pass 0 cases in every step and in general.
- solution.py, the concatenated solution/step_*.py and second_solution.py were each run twice in clean
  processes on twelve calls covering all six functions: identical outputs in each pair of runs, and
  solution.py and the per-step files agree bitwise. With multi-threaded BLAS the step 6 voltage at
  (0.05, 0.1, 2) changed in the ninth significant digit (absolute 2e-17), far inside the stated tolerance.

## Version 12 (grading_fix: authoring-guide conformance, pass 2)

Second conformance pass, driven by the findings of an independent verifier on version 11. In the first round
of this pass the model, the reference algorithms (solution.py and solution/step_*.py, apart from the step 4
docstring and one comment), the domain and every stated tolerance were unchanged. A second round, after an
independent verification of the first round (see "Second round" at the end of this section), corrected the
first-round step 4 endpoint entry, changed the step 4 kappa algorithm of the reference, stated the
propagated tolerances of the step 4 theta and the step 5 j_star (the error of mean_voltage carried through
the inversion) and made the accuracy needed to locate the step 3 maximum visible; the model, the domain
and every other tolerance are unchanged. edit_label stays grading_fix.

- Second solution shared the test targets' method (second_solution.py; rule: a genuinely different method).
  - Finding: for steps 1 to 5 the second solution solved, by Thomas elimination, the same truncated Fourier
    tridiagonal system that the test helper _t_cf solves as a continued fraction, with the same working
    digits and truncation formulas and the same corrector formula for D_eff; for step 6 its dv/di was the
    same central difference (h = 1e-4) as the step 6 target. A pass of the second solution therefore added
    little evidence beyond the reference-versus-target comparison.
  - Cause: the second solution had been chosen to differ from the reference (quadrature) without checking
    its independence from the targets (continued fractions).
  - Change: new second solution. Steps 1 to 5: the Jacobi-Anger Bessel series of the Stratonovich and
    Reimann integrals, described under "Consistency checks" above (v = a / Phi as a Bessel series, D_eff as
    a mean of products of Bessel-Fourier series formed by exact integer convolutions, analytic derivatives,
    mpmath at 30 + 1.8 / theta digits; the precision was set after finding that D_eff at theta = 0.02 is
    wrong at 40 digits and correct to 2e-13 at 50). The root finders stay different from the reference
    (grid scan plus Illinois regula falsi on d ln D / di, Illinois in 1 / theta, safeguarded Newton on
    ln V). noise_temperature now applies the endpoint rule with its own mean_voltage, as the prompt now
    states. Step 6: the Kramers hierarchy in a real (1, cos p x, sin p x) Fourier basis with Hermite
    functions centred at the overdamped velocity (N = 130, |p| <= 64), evaluated at the complex bias
    i + 1e-20 j, so that dv/di = Im / 1e-20 without difference error (complex-step differentiation).
  - Limitation (for the project lead): for step 6 the reference, the target and the second solution are all
    Hermite-Fourier matrix continued fractions (different basis centres, truncations, Fourier
    representations and derivative methods). No method outside this family that reaches the stated
    1e-7 |v| + 1e-12 at voltages of order 1e-9 was found (a sparse direct solve of the hierarchy and a
    continued fraction over the Fourier index were tried in version 5 and rejected; no Langevin or
    velocity-grid solver was run: a Langevin estimate cannot reach a relative 1e-7 in reasonable time,
    and a double-precision grid solver gives v only with an absolute error, which would have to stay
    below about 1e-12 at that voltage).
    The checks of step 6 outside this family are the beta_c = 0 limit (overdamped continued fraction and,
    in general case 1, mean_voltage), v = 0 at i = 0, and the noiseless and ohmic bounds at i = 2.5.
  - Regression test: on the step 1 and step 2 test points the new second solution reproduces the
    continued-fraction v, dv/di and D_eff to the last double-precision bit (closed forms at i = 0 to
    2.3e-15); it agrees with the reference to 4e-15 in i_peak and about 1e-13 in every other step 3 to 5
    output, and with the reference rcsj_voltage to 5.4e-9 relative in dv/di and 3.3e-5 of the v tolerance
    on 13 RCSJ points. crown_check SECOND passes every case (results below).
- Step 6 target dv/di at the smallest tested voltage (tests/step_6.py, tests/general.py, helper
  _t_kramers).
  - Finding: at (i, theta, beta_c) = (0.05, 0.1, 2) the target dv/di was accurate only to a few 1e-6
    relative (the reference deviated from it by 0.27 of the 1e-5 tolerance), so a candidate inside the
    specification but erring in the same direction as the target could fail.
  - Cause: in the shifted basis (centre v0 = 0.6 i = 0.03) the voltage v = 4.6e-9 results from a
    cancellation of about 7 digits, which leaves a rounding error of about 2.5e-17 in v; the central
    difference with h = 1e-4 amplifies it to about 2e-6 relative, and its own truncation error there is
    about 1.4e-6 (the voltage varies on the scale theta / pi).
  - Change: dv/di of the target is now the fourth-order five-point central difference
    [8 (f(i+h) - f(i-h)) - (f(i+2h) - f(i-2h))] / (12 h) with h = 1e-3 (at that point its truncation
    error is 2.6e-7 relative, measured with the unshifted expansion of the reference, and the rounding of
    the shifted basis is of the same order), and the truncation is N = 120, |p| <= 60 instead of N = 150,
    |p| <= 72 (against N = 220, |p| <= 110 with the same stencil on all 13 tested points: dv/di within
    1e-12 relative except 1.7e-7 at that point, v within 5e-5 of its tolerance; still larger than the
    reference's N = 110, |p| <= 56). This also answers the verifier's runtime advisory: one check now costs
    five solves of about 0.2 to 0.3 s instead of three of about 0.5 s with single-threaded BLAS (measured
    here), i.e. somewhat less than before. Thread oversubscription of a multi-threaded BLAS on a loaded
    machine can still make the step 6 and general cases slow; no timing code is used in the tests.
    No tolerance was changed.
  - Regression test: deviation of the reference (analytic dv/di) from the target, in units of the 1e-5
    tolerance, at (0.05, 0.1, 2): 0.27 before, 0.025 now; at the other 12 RCSJ test points at most 9e-5 now
    (at most 0.042 before). The new second solution (complex-step dv/di) deviates from the target by the
    same amounts, and its v by at most 2.3e-5 of the v tolerance.
- Untested promised ValueErrors (tests/step_1.py case 4, tests/step_5.py case 5, tests/step_6.py case 3).
  - Finding: step 5 promised ValueError for v_crit not in (0, 0.5 sum(c_k r_k)], which includes NaN, and for
    every non-finite entry, but only v_crit = 0 and 0.51, a NaN critical current and a zero resistance were
    tested; step 6 promised ValueError for any beta_c that is neither 0 nor in [0.1, 2], but no negative
    beta_c was tested. The same audit found theta = NaN untested in step 1, where a check written as
    "theta < 0.02 or theta > 50" lets NaN through.
  - Cause: the error lists of the prompts had more clauses than the error cases had entries.
  - Change: entries added to the existing error cases (no new case, n_test_cases unchanged): step 1
    (0.5, nan); step 5 rs = [1, inf], ics = [-1, 1] and v_crit = nan; step 6 beta_c = -1 and nan,
    i = inf and theta = inf.
  - Regression test: the reference and the second solution raise ValueError for every entry; the all-None
    and all-zero controls fail these cases.
- Step 2 dependency (problem.yaml step 2) and earlier functions named in the prompts.
  - Finding: step 2 listed step_dependencies [1], although neither the reference nor the second solution
    calls mean_voltage; the prompt named mean_voltage only to identify the model.
  - Cause: the dependency recorded the shared model, not a function the step uses.
  - Change: step_dependencies [] and the prompt says "the same overdamped junction as in step 1". The step 3
    prompt now says to use effective_diffusion (step 2) for D_eff and the step 6 prompt says to use
    mean_voltage (step 1) for beta_c = 0, matching their step_dependencies [2] and [1] and the reference.
  - Regression test: crown_check REF (each step run with the earlier reference steps only) passes; the
    prompts and contracts were compared by hand for every step.
- Step 4 endpoint rule needed accuracy beyond the stated tolerances (problem.yaml step 4 prompt, contract
  and function header; steps/step_4.py; solution/step_4.py and solution.py docstring and comment; the step 4
  and whole-task mutants' identical docstring and comment; tests/step_4.py case 3).
  - Finding: the rule was relative to the exact voltage at theta = 0.02 and 50, and the test separates
    excesses of 0.9e-11 and 1.1e-11; a candidate computing the endpoint voltage itself within the 1e-9
    stated for mean_voltage could misclassify both probes.
  - Cause: the rule referred to a quantity (the exact endpoint voltage) that the specification only asks
    to be computed to 1e-9.
  - Change: the rule is now defined with the endpoint voltages of step 1, v_lo = mean_voltage(i, 0.02)[0]
    and v_hi = mean_voltage(i, 50)[0]: a v in [v_lo (1 - 1e-11), v_lo] returns theta = 0.02, a v in
    [v_hi, v_hi (1 + 1e-11)] returns theta = 50, and a v outside [v_lo (1 - 1e-11), v_hi (1 + 1e-11)]
    raises ValueError. The test builds its probes (0.9e-11 and 1.1e-11, unchanged) from mean_voltage, after
    checking it against the continued fraction to the stated 1e-9, and still checks theta and kappa against
    the independent targets. The reference already used mean_voltage for the rule.
  - Regression test: step 4 case 3 passes for the reference, the second solution and the +-1e-12 shift;
    the step 4 mutants still fail their cases.
  - Correction (second round): this change was incomplete. tests/step_4.py case 2 still passed the exact
    continued-fraction voltages at theta = 0.02 and 50 as endpoint inputs. Under the rule these inputs are
    valid only if mean_voltage is within a relative 1e-11 of the exact voltage, but step 1 promises only
    1e-9, so case 2 still depended on an accuracy that is not stated. The claim above that the defect was
    removed was therefore wrong for case 2. The second round fixes it (entry "Step 4 case 2 endpoint
    inputs" below).
- Test-namespace names (tests/step_1.py cases 0-3, tests/step_2.py cases 0-3).
  - Finding: the unprefixed module-level names out and D could rebind a candidate's global of the same name.
  - Cause: these two names were not given the _t_ prefix used by every other test name.
  - Change: renamed to _t_out and _t_res (a pure rename, no check changed).
  - Regression test: REF, SECOND and SHIFT pass steps 1 and 2.
- Whole-task test linked with step 1 (tests/general.py case 1).
  - Finding: the general cases checked rcsj_voltage only against the RCSJ target, never against an earlier
    step of the same candidate.
  - Change: general case 1 also requires rcsj_voltage(-0.6, 0.4, 0) to reproduce mean_voltage(-0.6, 0.4)
    within twice the step 6 tolerances (2e-7 |v| + 2e-12 for v, 2e-5 relative for r_d), as the step 6
    prompt says for beta_c = 0.
  - Regression test: REF, SECOND and the shift of every evaluated function pass general case 1; the
    whole-task mutant still fails both general cases.
- Wording and records (problem.yaml, source.md).
  - Finding: difficulty_explanation said the solver "has to use" the integral representations or continued
    fractions; verification_explanation said "a 60-digit central difference" (the code uses
    60 + 1.8 / theta digits) and did not say that the old second solution shared the targets' recurrence;
    source.md had no version 10 note and several version 11 entries gave only the change.
  - Change: difficulty_explanation says "for example"; verification_explanation describes the current
    targets, their error budget, the new second solution, the step 6 limitation and this pass's crown_check
    results; the source.md target table and second-solution description are updated, a version 10 note is
    added and the version 11 entries are completed (marked "added in version 12").
  - Regression test: documentation only; every number quoted was rechecked against a run in this pass.

- Checked without change: no prompt, problem_io or function header contains the coordinates of a test
  point (searched for every step 6 and general test point; the only hits were the definition
  "beta_c = 2 e I_c R**2 C / hbar" and domain bounds), and the magnitude statements ("about 1e-17",
  "of order 1e-9 and below") hold over the stated domain; every mutant differs from the current reference
  by its one marked change only (the step 4 and whole-task mutants received the same docstring and comment
  edits as the reference); both general cases call rcsj_voltage; relevant_experience is still the author
  placeholder; solution.py and second_solution.py were each run twice in clean processes on 13 calls
  covering all six functions with identical outputs, and solution.py agrees bitwise with the concatenated
  solution/step_*.py.

Rerun (version 12, first round; superseded by the second-round rerun below): tools/crown_check.py, all
modes, every case alone in a fresh process, single-threaded BLAS (OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1), -j 2:
- FORMAT OK (no problem reported; repeated after the last documentation edits, which changed no test or
  solution file, again OK).
- REF: step 1 5/5, step 2 5/5, step 3 4/4, step 4 5/5, step 5 6/6, step 6 4/4, general 2/2 (solution.py
  also passes every step file).
- SECOND: 5/5, 5/5, 4/4, 5/5, 6/6, 4/4, general 2/2.
- Mutants: forward_slips_only fails 4/5 (cases 0-3), kramers_rate 4/5 (0-3), einstein_relation 3/5 (1-3),
  einstein_peak 3/4 (0-2), kramers_inversion 3/5 (0-2), inverted_sensitivity 4/5 (0-3),
  arrhenius_sensitivity 4/5 (0-3), squared_voltage_scale 4/6 (1-4), no_inertia 3/4 (0-2),
  velocity_noise_not_scaled 3/4 (0-2), missing_thermal_velocity 3/4 (0-2), whole-task
  velocity_noise_not_scaled 2/2 general cases.
- SHIFT +-1e-12: every step case passes, and both general cases pass with every evaluated function shifted.
- Controls: all-None and all-zero stubs pass 0 cases in every step and in general.
- SUMMARY ALL OK.

### Second round (version 12, after an independent verification of the first round)

The verifier found two blocking defects in the first round, both about the error budget of step 4: an
endpoint input that relied on an accuracy of mean_voltage that is not stated, and a theta tolerance that did
not include the voltage error of step 1 amplified by kappa. Its evidence also noted that the step 5 j_star
tolerance had no margin left for the same reason. The related steps were fixed together (guide section 8).
This round was interrupted by a container restart after part of the step 4 work had been saved; it was
then completed, every check below was rerun on the final files, and the numbers quoted are from that
rerun. The earlier-reference perturbations below wrap the reference function of step 1 or 2 so that its
output is multiplied by the stated factor, and run each case of the later step alone in a fresh process.

- Step 4 case 2 endpoint inputs (tests/step_4.py case 2).
  - Finding: case 2 passed the exact continued-fraction voltages at theta = 0.02 and 50 as the endpoint
    inputs. The endpoint rule is defined with v_lo = mean_voltage(i, 0.02)[0] and
    v_hi = mean_voltage(i, 50)[0], so these inputs are valid only if mean_voltage is within a relative
    1e-11 of the exact voltage, but step 1 promises only 1e-9 (and case 3 accepts 1e-9).
  - Cause: the first-round change of the endpoint rule (entry "Step 4 endpoint rule" above) was applied to
    the probes of case 3 only.
  - Change: case 2 builds its two endpoint inputs as case 3 does, v = mean_voltage(0.9, 0.02)[0] and
    v = mean_voltage(0.7, 50)[0], each first checked against the continued fraction to the stated 1e-9,
    and expects theta = 0.02 and 50 within the stated theta tolerance and kappa against the
    extended-precision difference at the returned theta. This also tests the inclusive endpoints v = v_lo
    and v = v_hi exactly.
  - Regression test: the finding was reproduced on the first-round files (tests/step_4.py and
    solution/step_4.py of the first round, mean_voltage multiplied by 1 +- 5e-10): case 2 raised
    "ValueError: no theta in [0.02, 50] reproduces this voltage" for both signs. With the current files
    every step 4 case passes with mean_voltage multiplied by 1 +- 5e-10, 1 +- 9.9e-10,
    1 + 9.9e-10 sin(7 ln theta + 3 i) and 1 - 9.9e-10 cos(5 ln theta).
- Step 4 theta error budget (problem.yaml step 4 prompt, function header and contract; steps/step_4.py;
  solution/step_4.py, solution.py and the step 4 and whole-task mutants' docstring; tests/step_4.py cases
  0 to 3).
  - Finding: theta is defined through mean_voltage, whose relative error may reach 1e-9, but was graded to
    a relative 1e-7. The theta error is kappa times the relative voltage error, and kappa reaches about
    500 to 2500 at tested points (510 at i = 1.5, theta = 0.04; 2501 at i = 0.7, theta = 50) and 2.4e7
    over the domain. With mean_voltage multiplied by 1 +- 5e-10 the first-round reference missed theta at
    (1.5, 0.04) by a relative 2.55e-7 (step 4 case 1, AssertionError, reproduced here for both signs).
  - Cause: the error of step 1 was not propagated through the inversion (guide section 5, "Account for
    error propagation through the full calculation"; rule G).
  - Change (option (b) of the verifier): the prompt, the header and the contract state that theta is
    graded against the noise strength at which the exact stationary voltage equals v with a relative error
    below 1e-7 + 2e-9 |kappa|, i.e. the 1e-9 of mean_voltage, doubled as for a comparison of two computed
    outputs, times |kappa|; the tests take kappa from the extended-precision difference at the known
    theta. Option (a), theta to 1e-7 from an exact voltage, was not taken: where kappa is about 2.4e7 it
    would need the voltage to a relative 4e-15, at the limit of double precision. kappa is now compared
    with the extended-precision difference at the returned theta (before: at the known theta), as the
    prompt states, because with the propagated theta tolerance the two can differ by more than 1e-6 where
    kappa is large. The prompt and the header also state the range of kappa over the domain (about 0.01
    to about 2e7; 0.0102 to 2.42e7 on the grid under "Consistency checks"). This is an error-budget
    correction, not a tuning: where kappa is small (the Arrhenius regime) the tolerance stays 1e-7.
  - Regression test: as for the previous entry; the step 4 mutants still fail (kramers_inversion 3/5,
    inverted_sensitivity 4/5, arrhenius_sensitivity 4/5).
- Step 4 kappa of the reference where kappa is large (solution/step_4.py, solution.py, the three step 4
  mutants and the whole-task mutant; tests/step_4.py case 1).
  - Finding: while checking the step 4 budget, the reference kappa was found to deviate from the
    extended-precision difference by up to 2.2e-5 relative (at i = 10, theta = 0.02, kappa 2.4e7) and by
    5.3e-6 at i = 8, theta = 0.04 (kappa 2.4e6), beyond the stated 1e-6. No test exposed it: the largest
    tested kappa was 2501.
  - Cause: d ln v / d ln theta was obtained by differentiating exp(E / theta) under the integral, a
    difference of two nearly equal terms when d ln v / d ln theta is tiny (far above the critical current
    at weak noise).
  - Change: the theta-derivative is taken with the shift scaled as y = theta u, which turns it into the
    integral of the Boltzmann factor times F = cos x - cos(x - y) + y sin(x - y) = O(y**2) (evaluated
    without cancellation, with the series of y cos(y/2) - 2 sin(y/2) for y < 1) plus a boundary term in
    exp(-2 pi i / theta) (solution_explanation). Case 1 adds the point (8.0, 0.04), where
    d ln v / d ln theta is about 4e-7. The step 4 and whole-task mutants were rebuilt from the new
    reference; each still differs from it by its one marked change only (checked with diff).
  - Regression test: the new reference kappa agrees with the extended-precision difference to 1.9e-11 on
    the 14 x 10 grid and to 1.0e-11 at (8.0, 0.04); the first-round solution/step_4.py run on the current
    tests fails case 1 at that point (kappa off by a relative 5.3e-6, tolerance 1e-6) and passes cases 0,
    2, 3 and 4. The second solution (analytic theta-derivative of the Bessel series) passes case 1.
- Step 5 j_star error budget (problem.yaml step 5 prompt, function header and contract; steps/step_5.py,
  solution/step_5.py, solution.py and the step 5 and whole-task mutants' docstring; tests/step_5.py
  cases 0 to 4).
  - Finding: the relative voltage error of mean_voltage enters j_star divided by s = d ln V / d ln J, which
    is at least 1 and tends to 1 in linear response, so at the 1e-9 allowed for mean_voltage the j_star
    error alone could reach the whole 1e-9 tolerance, leaving nothing for the candidate's own root finding.
    The first round recorded this without changing it.
  - Cause: the same missing propagation as in step 4.
  - Change: j_star is graded against the criterion current of the exact array voltage with the stated
    relative tolerance 1e-9 + 2e-9 / s, s the logarithmic slope of the exact array voltage at j_star
    (the 1e-9 of mean_voltage, doubled, divided by s); where the voltage rises steeply the tolerance stays
    close to 1e-9, and it is at most 3e-9 in linear response. The tests check
    |V(j) - v_crit| / V'(j) < (1e-9 + 2e-9 V / (j V')) j with V and V' from the continued fractions at the
    returned j, and case 0 compares j_star with the Bessel-integral bias 0.8 with the same tolerance. The
    prompt and the header now also say that r_diff and s_v are evaluated at the returned j_star, which is
    what the tests always did. The r_diff and s_v tolerances (1e-8) are unchanged: they are sums of
    positive terms r_k r_d,k and 2 c_k r_k D_k evaluated at the returned j_star, so the errors of
    mean_voltage (1e-8 for r_d) and effective_diffusion (1e-9) pass through without amplification.
  - Regression test: every step 5 case passes with mean_voltage v and r_d and effective_diffusion
    multiplied by (1 +- 9.9e-10, 1 +- 9.9e-9, 1 +- 9.9e-10) with equal signs and with mixed signs, and by
    i-dependent factors (1 + 9.9e-10 sin 9i, 1 - 9.9e-9 cos 4i, 1 + 9.9e-10 sin 3i); the step 5 mutant
    still fails 4/6 cases.
- Step 3 accuracy needed to locate the flat maximum (problem.yaml step 3 prompt, function header and
  contract; steps/step_3.py, solution/step_3.py, solution.py and the step 3 and whole-task mutants'
  docstring). edit kind: clarification.
  - Finding: found while auditing the related budgets. The header defined i_peak as the bias "at which
    effective_diffusion(i, theta) is largest" to an absolute 1e-6, but the tests grade it against the
    maximum of the exact (continued-fraction) D_eff, and the 1e-9 guaranteed by effective_diffusion does not
    determine the flat maximum to 1e-6: a relative error bounded by 1e-9 that varies with i can move it by
    4.3e-6 at theta = 0.02 up to 1.2e-4 at theta = 1 (curvatures under "Consistency checks"). The needed
    accuracy was stated only in step_background and difficulty_explanation.
  - Cause: the requirement followed from the error budget but was not in the agent-visible text.
  - Change: the prompt, the header and the contract define i_peak as the maximum of the exact D_eff, and
    the prompt states that an error of D_eff at the 1e-9 level that varies with i can move the apparent
    maximum by several 1e-6 at weak noise and by about 1e-4 at the strong-noise end of the range, so
    D_eff or d ln(D_eff) / di must be evaluated far more accurately and smoothly in i; d_peak is stated at
    the returned i_peak (as tested). No test changed.
  - Regression test: the reference (which locates the peak from its own d ln D / di) passes every step 3
    case with effective_diffusion multiplied by 1 +- 9.9e-10, 1 + 9.9e-10 sin 40i,
    1 + 9.9e-10 sin(1e5 i) and 1 + 9.9e-10 sign(sin 3000i). A plausible candidate that maximises
    effective_diffusion itself by golden-section search (written for this check, not part of the task)
    passes every case with the unperturbed reference context but fails case 1 (theta = 0.3, i_peak off by
    1.1e-5) with the factor 1 + 9.9e-10 sin(1e5 i), which is inside the stated 1e-9 of step 2.
- Step 6 with a perturbed mean_voltage: every step 6 case passes with mean_voltage v and r_d multiplied by
  (1 +- 9.9e-10, 1 +- 9.9e-9) (beta_c = 0 uses mean_voltage); no change.
- Records (problem.yaml verification_explanation; source.md): the step 4 and step 5 budgets, the kappa
  grid (recomputed: 1.9e-11, previous reference 2.2e-5; an interim record of this round had 4.8e-11 and
  4.4e-5 on a grid whose points were not recorded), the 25 x 50 grid of d ln v / d ln i (minimum
  1.000000000016; the interim record had 1.0000000000000016) and the perturbation runs are described as
  run; the step 5 target-table row no longer mentions identical junctions (no test uses them) and the
  "no margin" sentence on step 5 is replaced by the propagated tolerance.

Rerun (version 12, second round, on the final files): tools/crown_check.py, all modes, every case alone in
a fresh process, single-threaded BLAS (OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1), -j 2:
- FORMAT OK (no problem reported; repeated after the last documentation edits, which changed no test or
  solution file, again OK).
- REF: step 1 5/5, step 2 5/5, step 3 4/4, step 4 5/5, step 5 6/6, step 6 4/4, general 2/2 (solution.py
  also passes every step file).
- SECOND: 5/5, 5/5, 4/4, 5/5, 6/6, 4/4, general 2/2.
- Mutants: forward_slips_only fails 4/5 (cases 0-3), kramers_rate 4/5 (0-3), einstein_relation 3/5 (1-3),
  einstein_peak 3/4 (0-2), kramers_inversion 3/5 (0-2), inverted_sensitivity 4/5 (0-3),
  arrhenius_sensitivity 4/5 (0-3), squared_voltage_scale 4/6 (1-4), no_inertia 3/4 (0-2),
  velocity_noise_not_scaled 3/4 (0-2), missing_thermal_velocity 3/4 (0-2), whole-task
  velocity_noise_not_scaled 2/2 general cases.
- SHIFT +-1e-12: every step case passes, and both general cases pass with every evaluated function shifted.
- Controls: all-None and all-zero stubs pass 0 cases in every step and in general.
- SUMMARY ALL OK.
- Also run: the earlier-reference perturbations listed in the entries above (steps 3 to 6, every case
  passes for the reference); solution.py and second_solution.py each run twice in clean processes on 13
  calls covering all six functions (identical outputs in each pair), and solution.py agrees bitwise with
  the concatenated solution/step_*.py; every function header in problem.yaml matches the scaffold, the
  per-step solution, solution.py and the mutants that define that function, and problem_io matches the
  step 6 header.
