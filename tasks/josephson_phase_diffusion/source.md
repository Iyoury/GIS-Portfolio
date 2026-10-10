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
- The second solution is independent of the reference: it solves the stationary
  Fokker-Planck equation and the homogenization cell problem by Galerkin (Fourier)
  truncation with tridiagonal eliminations in extended precision, gets dv/di from the
  linearized system, evaluates D_eff = theta int (1 + chi')**2 p from the Fourier
  coefficients, finds the diffusion peak by a double-precision scan followed by secant
  iterations on high-precision differences, the noise temperature by Illinois regula falsi in 1 / theta with the thermometric
  sensitivity from the Galerkin system differentiated in theta, and the
  criterion current by safeguarded Newton iterations on ln V.

## How the test targets are obtained

No target is copied from the reference solution. Every test computes its own target:

| Target | Method in the test | Note / limit of validity |
|---|---|---|
| v and dv/di (step 1) | decaying solution of the Fourier recurrence of the stationary Fokker-Planck equation as a continued fraction in extended precision (30 + 1.8 / theta digits), v = i + Im r_1, dv/di from the differentiated recurrence | about 1e-15 relative, also for exponentially small v |
| v at low noise (steps 1, 4, 5) | single-integral form with the inner integral done exactly, v = theta (1 - exp(-2 pi i / theta)) / int_0^{2 pi} I_0(2 sin(y/2) / theta) exp(-i y / theta) dy, mpmath quadrature at 40 digits | agrees with the continued fraction to 1e-12 |
| dv/di at i = 0 (step 1) | linear response, 1 / I_0(1/theta)**2 | exact |
| D_eff (steps 2, 3, 5) | homogenization: D = theta int (1 + chi')**2 p dx with the corrector chi and the density p both from continued fractions in extended precision; Fourier convolution for the average | about 1e-15 relative |
| D_eff at i = 0 (step 2) | Lifson-Jackson, theta / I_0(1/theta)**2 | exact |
| diffusion peak (step 3) | one Newton step on dD/di = 0 with central differences (h = 2e-5) of the continued-fraction D must move i_peak by less than 1e-6; D lower at i_peak +- 0.05 | truncation of the differences below 1e-8 |
| noise temperature (step 4) | the voltage is generated at a known theta (including the endpoints 0.02 and 50) by the continued fraction (or the Bessel integral); the returned theta must recover it | relative 1e-7 |
| thermometric sensitivity kappa (step 4) | central difference of ln v in ln theta with step 1e-15 theta, the continued fraction evaluated at 60 + 1.8 / theta digits | about 1e-15 relative |
| array criterion (step 5) | one Newton step on V(J) = v_crit with V and dV/dJ from the continued fractions must move j_star by less than 1e-9 relative; r_diff and s_v compared with the continued-fraction sums; for one junction or identical junctions j_star is known from the Bessel-integral voltage | relative 1e-9 / 1e-8 |
| v and dv/di with capacitance (step 6, general) | Kramers equation in Hermite functions centred at the shifted velocity v0 = 0.6 i (N = 150, |p| <= 72), its own matrix continued fraction; dv/di by a central difference (h = 1e-4); beta_c = 0 against the overdamped continued fraction | unchanged to 5e-9 relative against N = 220, |p| <= 110 (version 7) |

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

## Version 11 (grading_fix: authoring-guide conformance)

Audit of every file against the current authoring guide. The science, the reference algorithms, the
targets, the tolerances of the individual outputs and the domain are unchanged. (The version 10 upload
asserted the step 6 time budget in the tests and removed the sentence about the number of test calls from
the step 6 prompt; it is recorded in the version 9 note above.)

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
- Tolerance stated in the prompt applied exactly (tests/step_6.py case 1): the beta_c = 0 check used a pure
  relative 1e-7 on v; it now uses the stated 1e-7 |v| + 1e-12 (no practical change at v of order 0.5).
- Hidden test inputs in the step 6 prompt.
  - Finding: the prompt named the bistable example "beta_c = 2, theta = 0.1, i = 0.7" and the small-voltage
    example "i = 0.05, theta = 0.1, beta_c = 2", both of which are test inputs.
  - Change: the statements are kept without the coordinates ("in the bistable range (strong inertia and
    weak noise) ... the stationary state can be dominated by the locked state"; "below the critical current
    at weak noise and strong inertia v can be very small (of order 1e-9 and below)"). The reference gives
    v = 4.55e-9 at (0.05, 0.1, 2) and 6.5e-10 at (0.01, 0.1, 2), so the second statement holds.
- Stated but untested error rules.
  - Finding: the prompts promise ValueError for a non-finite i (step 4), a non-finite theta0 (step 5) and a
    non-finite theta or beta_c (step 6); none of these inputs was tested. A NaN theta0 or theta passes a
    plain range comparison, so these are distinct behaviours.
  - Change: one entry each added to the existing error cases: (nan, 0.1) in step 4 case 4, theta0 = nan in
    step 5 case 5, (0.5, nan, 1.0) and (0.5, 0.3, inf) in step 6 case 3. Reference and second solution raise
    ValueError for each.
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
- Test header comment: the garbled recurrence "c_{n+1} + (2 theta n + 2 j i) c_{n-1}... in the form" now
  reads c_{n+1} + a_n c_n - c_{n-1} = 0 with a_n = 2 theta n + 2 j i (the recurrence the helpers solve).
- source.md target table: the step lists no longer claim that the overdamped targets are used in the
  general tests (general.py tests only rcsj_voltage since version 5), and a row for the step 6 / general
  RCSJ oracle was added.
- Metadata added to problem.yaml: subfield, tags, expert_time_estimate_hours, relevant_experience (a
  placeholder for the author), difficulty_explanation, solution_explanation, verification_explanation,
  author, affiliation; edit_label is now grading_fix.
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
