# Sources, method and ground truth

## Physics and formulas
- **Prism potential, attraction and gradients.** Nagy (1966) and Nagy, Papp and Benedek (2000,
  J. Geodesy 74, 552), as corner sums with the limits at faces, edges and corners.
- **Columns with an exponentially decaying density contrast.** Cordell (1973); Chakravarthi and
  Sundararajan (2007, Geophysics 72, I23).
- **Basin inversion.** Bott (1960); Cordell and Henderson (1968).
- **Constant.** G = 6.6743e-11 (CODATA 2018).

## Reference solution
- **Prism potential, attraction and gradients, near the prism (within two diagonals of its centre).**
  The corner sums, with these details:
  - ln(a + r) for a < 0 is taken as ln((b^2 + c^2)/(r - a)), with no cancellation.
  - On the line of an edge (b^2 + c^2 = 0) the ln(b^2 + c^2) part is dropped. It cancels exactly between
    the two corners of that edge.
  - 0 ln 0 is taken as 0, and atan(n/0) as sign(n) pi/2. These are the limits used where the sums are
    continuous.
- **Prism, beyond two diagonals.** A 16^3-point Gauss-Legendre volume rule of 1/r and its derivatives.
  It is exact to rounding that far. The corner sums lose about (distance/size)^2 of their digits:
  1e-8 relative at 100 diagonals, 4e-5 at 1000, and nothing correct at 1e4.
- **Elongated prisms (version 3).** Within two diagonals the corner sums lose about 1e-16 times the shape
  factor diag^3 / volume (5.2 for a cube, about aspect^2 for a rod, 2.8 aspect for a plate): 1e-8 for a
  1 x 1 x 1000 rod. A prism whose shape factor exceeds 200 is cut into equal pieces with shape factor at
  most 200; each piece is taken by its corner sums within two of its own diagonals and by a
  Gauss-Legendre rule beyond (order per axis from the Bernstein-ellipse bound, error below 1e-24, per band
  of distance). For a point within a quarter piece of a cut, that axis uses the cuts shifted by half a
  piece, so no point is on an internal face, edge or corner, where the gradients of the pieces jump or
  diverge although their sum does not.
- **Columns.** g_z = G drho0 int_0^h e^{-lam z} Omega(z) dz, with Omega the solid angle of the
  horizontal lamina. It is evaluated with geometric panels from the top (ratio 2, down to 2^-60 h),
  20-point Gauss-Legendre each. This resolves the lamina term near the top for a station at any distance
  from an edge. A single 64-point rule gives 4e-5 relative error 1 mm from an edge, and 6e-6 even with
  1024 nodes. Near the column Omega is the corner sum of atan2(x y, Z r); at a horizontal distance of at
  least the larger width (version 3) it is the Van Oosterom-Strackee formula for the two triangles of the
  rectangle, tan(Omega_t / 2) = Z wx wy / (|a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|), whose terms are all
  positive there.
- **Basin depths (version 3).** Start from the infinite-slab thicknesses; scipy's trust-region
  reflective least squares on the depths, bounded by 1.5 times the largest depth of the stated domain
  (min(5e4, 3 w_min, 2 / lam)), with the analytic Jacobian dg_i/dh_j = G drho0 e^{-lam h_j} Omega_ij(h_j);
  then Newton's method with step halving until the relative update is below 1e-13 or the residual is at
  its rounding level. The depth panels start at a sixty-fourth of the smallest cell width and double.

## Second solution
- The corner sums evaluated point by point, with math.fsum of all the terms.
- A 24^3-point rule beyond three diagonals.
- Elongated prisms (diag^3 / volume > 100) bisected recursively along the longest edge, the cut moved to
  3/8 or 5/8 of the edge when the point is within 1/8 of its middle (version 3).
- The columns by adaptive QUADPACK with geometric breakpoints; beyond two lamina diagonals the lamina term
  by a 24 x 24 Gauss-Legendre rule of Z / r^3 over the lamina (version 3).
- The basin by scipy's trust-region reflective least squares in the slab-equivalent thicknesses
  u = (1 - e^{-lam h}) / lam, then Newton iterations with least-squares steps (numpy.linalg.lstsq) and
  step halving, on a forward model with 16-point Gauss-Legendre panels at h 2^-k (k = 24 .. 0) and the
  analytic Jacobian (version 3; before, undamped Newton with least-squares steps).

## Targets
- **Prism (steps 1 and 2).** The closed forms evaluated with mpmath at 60 digits, which have no
  cancellation at any distance, with the same limit conventions. The far-field points (2.5 to 1e6
  diagonals) agree with 48^3-point Gauss-Legendre volume integrals in double precision to 1e-15. The rod
  (2 m x 2 m x 2 km) and plate (2 km x 2 km x 2 m) of case 4 at 90 digits.
- **Columns (step 3).** mpmath tanh-sinh quadrature, at 30 digits, of the lamina term over depth, with
  breakpoints at h 2^-k. For lam = 0, the 60-digit closed form of the prism. The far stations of case 4
  (5e4 m to 1e7 m) at 60 and 70 digits, and the 90-digit prism closed form for lam = 0.
- **Basins (step 4 and tests/general.py).** A 5 x 4 basin with known depths, its anomaly computed with
  25-digit mpmath quadrature of all 400 station-column pairs; three more basins (version 3) with 30-digit
  quadrature of every pair (a 6 x 4 basin with lam depth up to 2, a 5 x 4 basin of 80 m cells with
  lam = 1e-2, a rough 6 x 6 basin with 30 m columns beside 1500 m ones), and a 10 x 10 basin with lam = 0
  whose anomaly is the sum of the 60-digit closed forms of its 100 prisms at each station.
- No HDF5 file: every target is stored inline in the test case that uses it, so there is no
  target_evidence record.

The reference agrees with all targets within the stated tolerances: U to 1e-12, g to 0.09 of its
tolerance, T to 1e-4 of its tolerance, columns to 1.5e-13, depths to 1e-12.

## Version 2 (grading_fix: authoring-guide conformance)

1. **Timing in the tests and time budgets in the specification.**
   - Finding: every test file wrapped the evaluated functions in a `_t_budget` wrapper (SIGALRM interval
     timer, `time.perf_counter`, an assert on the elapsed time), and the prompts, function headers,
     problem_io, scaffolds, solutions and mutants stated per-call budgets ("Each call must finish within
     10 s on one CPU core for up to 1000 points", "within 30 s").
   - Cause: the first version used wall-clock budgets to require efficient methods; the guide forbids
     clock reads in tests and per-call time budgets.
   - Change: the wrapper, the `signal`/`time` imports and every budget sentence are removed. The problem
     size of those sentences is kept without a time: "up to 1000 points (stations) per call" in the
     prompts, docstrings and contracts of steps 1-3.
   - Regression test: case 2 of tests/step_1.py and tests/step_2.py now also calls the function once with
     1000 points (the stored near and far points of the first prism, repeated) and checks every value
     against the stored targets with the stated tolerances.
2. **Test-case format.**
   - Finding: markers carried descriptions ("# --- test case 0: far from the prism ... ---"); constants,
     helpers and the time-budget wrapper sat before case 0, and case 2 of steps 1 and 2 used target lists
     defined in cases 0 and 1.
   - Cause: the files were written to run as one script.
   - Change: markers are exactly "# --- test case N ---" (the description follows as a comment); every
     case has its own imports, constants, helper functions and stored targets (tools/selfcontain.py, then
     checked by hand). No case was dropped; n_test_cases stays 4, 4, 4, 2 (general: 2).
   - Regression test: tools/crown_check.py runs every case alone in a fresh process.
3. **Tolerance inconsistent with the stated tolerances (tests/general.py, case 1).**
   - Finding: column_gz (lam = 0) and the z component of prism_gravity were compared with a relative
     2.1e-10 of g_z, but prism_gravity's stated tolerance is 1e-10 |g| + 1e-14 G |rho| L, relative to
     the norm |g|. At the station (-500, 900, -3), g_z / |g| = 0.15, so a prism_gravity within its
     stated tolerance could differ from the exact g_z by 6.5e-10 g_z and be rejected.
   - Cause: the comparison used g_z instead of |g| as the scale of prism_gravity's error.
   - Change: |column_gz - g_z| <= 2.1e-10 |g| + 2e-14 G |rho| L, about twice the single-output tolerance
     of step 1 (column_gz's own 1e-10 |g_z| is at most 1e-10 |g|). No stated tolerance changed.
   - Regression test: the case passes for the reference (agreement 4e-15 relative) and the second solution
     (5.6e-13); the whole-task mutant still fails case 0.
4. **Ambiguous far-field range.**
   - Finding: the prompts promised accuracy "up to a million times the size of the prism"; the farthest
     test points are 1e6 diagonals from the centre, i.e. 1.34e6 and 1.41e6 times the longest edge.
   - Change: "a million times the diagonal of the prism" in the prompts and docstrings of steps 1 and 2,
     and in their contracts.
5. **Stated error rules without a test.**
   - Finding: several ValueError rules of the visible specification were never exercised.
   - Change: entries added to the existing invalid-input cases: step 1, x1 = x2; step 2, a non-finite
     point and an empty (0, 3) array (rules "as in prism_gravity"); step 3, non-finite x2, depth and lam
     and y1 = y2; step 4, a 2-D edge array, an edge array with one entry, an infinite edge, a NaN
     observation and drho0 = -inf. The reference and the second solution raise for all of them.
6. **Undeclared and unused imports in second_solution.py.**
   - Finding: it imported `warnings`, `scipy.integrate.quad`, `IntegrationWarning` and
     `scipy.optimize.root` (never used), none declared in required_dependencies, and changed the global
     warning filter at import time; its header comment and this file said the basin was solved with
     scipy.optimize.root (hybrid Powell), but the code runs Newton iterations with least-squares steps.
   - Change: the unused import and the global filter are removed; QUADPACK is called with
     full_output=1, which returns its roundoff messages instead of emitting warnings (same values);
     `from scipy.integrate import quad` is declared in required_dependencies and in the scaffold headers;
     the comment and the "Second solution" section above describe the actual method.
   - Regression test: second_solution.py passes every case; run with UserWarnings as errors, it emits
     none.
7. **Contracts.** Step 3 listed step_dependencies [1], but its prompt names no earlier function and the
   reference does not call one: now []. Its assumptions are spelled out instead of "As in step 1", and
   the valid_input_ranges of all steps now match the prompts (distance range, points per call, parameter
   rules).
8. **Metadata.** Added subfield, tags, expert_time_estimate_hours, the relevant_experience placeholder
   (for the author to fill), difficulty_explanation, solution_explanation, verification_explanation,
   author, affiliation and edit_label: grading_fix.
9. **Spot checks of the stored targets (new in this version, independent of the task code).** My own
   mpmath implementation of the prism potential (60 digits, attraction by mpmath differentiation) at four
   points (two near the first prism, one at 100 diagonals of each prism) matches the stored U and g to
   1.5e-25 relative near the prism and 4e-17 at 100 diagonals; a 30-digit mpmath quadrature of the
   lamina term over depth matches four stored column values (1 mm and 1e-6 m from an edge, the deep
   lam = 1e-2 column, a station above the surface) to 1.5e-21; the 30-digit sum of the 20 columns at two
   stations of the basin matches the stored anomaly to 7e-17 (the precision of the stored doubles).
10. **Reproducibility.** solution.py and second_solution.py were each run twice in clean processes and
    once with single-threaded BLAS on the same inputs (all four functions): identical outputs (SHA-256 of
    the output bytes). Neither uses random numbers.

Rerun: `python3 tools/crown_check.py tasks/prism_basin_gravity -j 2` (all modes): FORMAT OK; REF steps
1-4 4/4, 4/4, 4/4, 2/2 and general 2/2; SECOND the same; mutants closed_form_everywhere (step 1) fails
cases 0, 2, direct_log fails 1, 2, closed_form_everywhere (step 2) fails 0, 2, uniform_gauss_legendre
fails 0, 2, uniform_density fails 0, 2, slab_thickness fails 0, whole_task_upward_z fails general case 0;
SHIFT +-1e-12 all pass; all-None and all-zero controls pass 0 cases in every step and in general.

Not changed: the science, the reference algorithms, the stored targets and every stated tolerance. The
size "up to 1000 stations per call" of step 3 is stated but exercised with at most 5 stations, because a
QUADPACK implementation such as second_solution.py needs about 0.14 s per station (about 145 s for 1000).

## Version 3 (grading_fix: authoring-guide conformance)

An independent verification after version 2 reported four blocking findings. Each was reproduced before
any change, by rerunning the verifier's probe scripts and with my own (60- to 90-digit mpmath closed forms
and quadratures, the conditioning of the inversion from its Jacobian, the inversion on random basins), and
each is fixed below.

1. **Step 4: part of the promised domain was ill-posed at 1e-8, and the reference failed on well-posed
   basins.**
   - Finding: the visible domain was every basin with depths in (0, 5e4] m, 0 <= lam <= 1e-2 and up to
     100 cells. The amplification of relative data errors into relative depth errors,
     ||diag(1/h) J^-1 diag(g)||_inf, reaches 3.6e11 for a 3 x 3 basin of 1 km cells 45 km deep (lam = 0)
     and 2.5e17 for depths of 3 km with lam = 1e-2, so rounding gz_obs to a double alone moves the depths
     by far more than 1e-8. The verification also reported that the reference, Newton's method with step
     halving, raised ValueError on well-posed basins (lam = 1e-2, depths 1000/1200 m on 1 km cells) after
     overshooting a column to a depth where exp(-lam h) makes the Jacobian singular; with my own random
     basins the same algorithm also stalls inside the new, restricted domain on rough basins (3 of 40).
   - Cause: the domain was stated without an analysis of the conditioning, and the Newton iteration was
     not globalized.
   - Change: the domain is now every depth <= 3 w_min (w_min the smallest cell width in x or y),
     <= 5e4 m and lam depth <= 2, stated in the prompt, the function header, problem_io, the contract and
     the step background, with the remark that no smoothness of the basin is assumed. Over uniform,
     irregular and anisotropic grids up to 10 x 10 and constant, random, chequered and bowl-shaped depths,
     the amplification is at most 7.9e5 in that domain (a 10 x 10 basin 3 w_min deep everywhere with
     lam depth = 2; 2.2e5 for lam = 0; the old test basin 4.1e3), so the rounding of the data contributes
     at most 1e-10; allowing lam depth up to 5 would raise it to 8e6, and depths up to 3 w_min with
     lam depth <= 2 keeps the old test basin (2.52 w_min, lam depth 1.01) inside the domain. The
     reference inversion is now a trust-region reflective least-squares phase (scipy.optimize.least_squares,
     bounded by 1.5 times the largest depth of the domain) followed by Newton's method to 1e-13;
     `from scipy.optimize import least_squares` is declared in required_dependencies and the scaffolds.
     Newton in the slab-equivalent thickness, Newton with step caps, Newton in log-depth,
     Levenberg-Marquardt, continuation in the data scale and continuation in the horizontal scale were
     tried first and each stalled on some rough basin (shallow columns beside deep ones): such a column
     is overestimated by the slab start, Newton then overshoots a deep neighbour into a region where its
     attraction hardly changes with depth, and the shallow column is pushed to zero depth.
   - Regression test: step 4 cases 2-5 (below) and, outside the tests, the final reference on 200 random
     basins of the domain (1 to 100 cells, uniform and irregular grids, constant, random, rough and
     bimodal depths, lam from 0 to lam depth = 2, both signs of drho0; data from the reference forward
     model, so this checks the convergence of the solver, the accuracy being checked by the independent
     targets of the tests): no failure, worst relative depth error 4.6e-11, at most 25 s on a loaded
     machine; the second solution on 100 of them: no failure, worst 3.9e-11. The old reference algorithm
     fails 3 of the first 40 (all rough). A first version of the trust-region phase with tolerances 1e-10
     stopped early (gradient test) on one ill-conditioned basin in the second solution, so both solutions
     now use xtol = ftol = gtol = 1e-14 there.
2. **Step 4 was graded on a single basin, and the second solution was wrong on most of the domain.**
   - Finding: one value case (a smooth 5 x 4 basin, lam = 4e-4, drho0 < 0) and one validation case;
     nothing tested lam = 0, drho0 > 0, 100 cells or a larger lam, and second_solution.py (undamped Newton
     with least-squares steps and no residual check) passed every case while returning depths off by
     factors of 1e3 to 1e7 on well-posed basins.
   - Cause: too few regimes in the tests.
   - Change: four value cases with independent targets: case 2, a 10 x 10 basin (the largest grid,
     irregular cells of 900 m to 1200 m) with lam = 0, drho0 = +300 and smooth depths up to 2.86 w_min,
     its anomaly the sum of the 60-digit closed forms of its 100 prisms at each station; case 3, a 6 x 4
     basin with lam = 1e-3 and depths up to 2 km (lam depth = 2, the limit); case 4, a 5 x 4 basin of 80 m
     cells with lam = 1e-2 (the largest) and depths up to 200 m; case 5, a rough 6 x 6 basin of 500 m
     cells with columns 30 m deep beside columns 1500 m deep (3 w_min, the limit), lam = 4e-4. The data of
     cases 3-5 are 30-digit mpmath tanh-sinh quadratures of every station-column pair. tests/general.py
     case 2 sums prism_gravity over the 100 prisms of the case-2 basin (within the sum of the stated
     tolerances of the terms) and inverts the anomaly. n_test_cases of step 4 is now 6 (general: 3).
     second_solution.py now runs trust-region reflective least squares in the slab-equivalent
     thicknesses (1 - e^{-lam h}) / lam, then least-squares Newton steps with step halving, on its own
     forward model. A new mutant, newton_step_halving (the version-2 reference algorithm), fails case 5.
   - Regression test: the reference and the second solution reproduce the depths of cases 2-5 to 9e-13
     and 1.3e-12 at most; the version-2 reference raises ValueError on case 5 and the version-2 second
     solution is off by 2.4e9 there. The forward model of the reference agrees with the four new anomalies
     to 1e-15.
3. **Step 3: the stated 1e-10 failed at far stations.**
   - Finding: the four arctangents of the lamina term cancel when the station is many column widths
     away; the reference lost 9e-8 relative 1e6 m from a 100 m x 200 m column and every digit 1e7 m
     from a 1 m column, and the second solution failed too. The prompt said "near or far" with no bound.
   - Cause: the lamina term was evaluated by its corner sum at every distance.
   - Change: beyond a horizontal distance of the larger width the reference takes the solid angle of the
     lamina by the Van Oosterom-Strackee formula for its two triangles (no cancellation: triple product
     Z wx wy, all dot products positive); near the column the corner sum is kept, so every near value is
     unchanged (of the 22 stored stations of step 3 cases 0-2, the 4 that are a width or more away moved by
     at most 1.5e-13 relative, all to within 3.3e-16 of their targets). The second solution integrates Z / r^3 over the lamina with a 24 x 24 Gauss-Legendre rule
     beyond two lamina diagonals. The prompt, header and contract now state the range: every station with
     z <= 0 up to 1e7 m from the column. New mutant corner_sum_far.
   - Regression test: step 3 case 4, nine far stations (5e4 m to 1e7 m; columns 1 m to 200 m wide, lam 0,
     5e-4 and 1e-2) against 60- and 70-digit tanh-sinh quadratures and 90-digit prism closed forms, also in
     one (n, 3) call. Reference and second solution within 4.4e-16; the old reference fails (up to 2.9
     relative). Outside the tests, the new lamina term against 50-digit mpmath on 3000 random rectangles
     and stations: 6e-16 in the far region, bit-identical to the old one in the near region.
4. **Steps 1 and 2: the stated accuracy failed for elongated prisms.**
   - Finding: no limit on the aspect ratio was stated, but within two diagonals the corner sums lose
     about 1e-16 times diag^3 / volume: for a 1 x 1 x 1000 rod, U to 1.6e-8 relative and T to 5 times its
     tolerance (1 x 1 x 1e4: 4e-7); plates are milder (1 x 1000 x 1000: 3.3e-11). The second solution
     failed similarly.
   - Cause: the near/far switch was on the diagonal only, which for an elongated prism leaves points far
     from its small edges in the corner-sum region.
   - Change: the domain now states that the longest edge is at most 1000 times the shortest (prompts,
     headers, contracts, problem_description_main), and the reference cuts a prism of shape factor above
     200 into equal compact pieces, as described above; the second solution bisects such a prism
     recursively. New mutants no_subdivision (steps 1 and 2).
   - Regression test: case 4 of steps 1 and 2: a 2 m x 2 m x 2 km rod and a 2 km x 2 km x 2 m plate
     (aspect 1000), points inside, on the surface and up to two diagonals away, single points and one
     (n, 3) call, against 90-digit closed forms. Reference within 0.03 of the tolerance of g and 3e-4 of
     those of U and T; the version-2 reference fails (U at 120 times its tolerance, T at 4.9 times).
     Outside the tests: 30 random prisms of aspect 1 to 822 at 9 points each (inside, on a face, near,
     far) against 90-digit closed forms: reference within 0.005 (U), 0.009 (g) and 0.0005 (T) of the
     tolerances, second solution within 0.011, 0.010 and 0.001. 1000 points near a 1 m x 1 km x 1 km
     plate take about 5 s.
5. **Metadata and descriptions.** difficulty_explanation, solution_explanation and
   verification_explanation describe the new methods, domain and checks; the test-file headers name the
   new targets; the sections above describe the version-3 methods. No target_evidence is needed: every
   target is inline.

Rerun: `python3 tools/crown_check.py tasks/prism_basin_gravity -j 2` (all modes): FORMAT OK; REF steps
1-4 5/5, 5/5, 5/5, 6/6 and general 3/3; SECOND the same; mutants: closed_form_everywhere (step 1) fails
cases 0, 2, 4, direct_log 1, 2, no_subdivision (step 1) 4, closed_form_everywhere (step 2) 0, 2, 4,
no_subdivision (step 2) 4, uniform_gauss_legendre 0, 2, uniform_density 0, 2, 4, corner_sum_far 4,
slab_thickness 0, 2, 3, 4, 5, newton_step_halving 5, whole_task_upward_z general 0, 2; SHIFT +-1e-12 all
pass; all-None and all-zero controls pass 0 cases in every step and in general. solution.py and
second_solution.py were each run twice in clean processes and once with single-threaded BLAS on the same
inputs (all four functions, including the rod, a far column and the rough basin): identical outputs.

Not changed: the physics, the stored targets of version 2, and every stated tolerance (U 1e-10,
g 1e-10 |g| + 1e-14 G |rho| L, T 1e-9 ||T||, columns 1e-10, depths 1e-8).
