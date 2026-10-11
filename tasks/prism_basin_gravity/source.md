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
  It is exact to rounding that far. The corner sums lose about (distance/size)^2 of their digits. Measured
  in version 4 (mutants/step_1_closed_form_everywhere.py against my own 60-digit mpmath closed form, 50
  random directions): for the 100 m x 200 m x 250 m prism of the tests, U is off by a median 6.5e-9
  (max 5.1e-8) relative at 100 diagonals, 9.1e-6 (4.0e-5) at 1000, 9.7e-3 (4.6e-2) at 1e4, and at 1e6
  diagonals by a factor up to 5e4 with the wrong sign in 36 of the 50 directions; for the
  1000 m x 1000 m x 2 m plate, 2.0e-6 (5.8e-6) at 100 diagonals, 1.5e-3 (8.4e-3) at 1000, and the wrong
  sign in about half the directions from 1e4 diagonals on. (Earlier versions of this file quoted
  "1e-8 at 100 diagonals, 4e-5 at 1000, nothing correct at 1e4", and difficulty_explanation "5e-9 at 100
  diagonals"; these were rounded single figures, replaced by the measured medians and maxima.)
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
  from an edge. A single 64-point rule over the 3 km column of the tests gives 4.1e-5 relative error 1 mm
  from an edge and 1.0e-3 for a station 1 m from two edges, and a 1024-point rule still 6.0e-6 1 mm from
  an edge (rechecked in version 4). Near the column Omega is the corner sum of atan2(x y, Z r); at a horizontal distance of at
  least the larger width (version 3) it is the Van Oosterom-Strackee formula for the two triangles of the
  rectangle, tan(Omega_t / 2) = Z wx wy / (|a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|), whose terms are all
  positive there.
- **Basin depths (version 3).** Start from the infinite-slab thicknesses; scipy's trust-region
  reflective least squares on the depths, bounded by 1.5 times the largest depth of the stated domain
  (min(5e4, 3 w_min, 2 / lam)), with the analytic Jacobian dg_i/dh_j = G drho0 e^{-lam h_j} Omega_ij(h_j);
  then Newton's method with step halving until the relative update is below 1e-13 or the residual is at
  its rounding level. The depth panels start at a sixty-fourth of the smallest cell width and double.

## Second solution (version 4)
Methods different from the reference's in every step (the version-3 second solution, which used the
reference's strategies with other parameters, is described under Version 4 below):
- **Prism, beyond three diagonals of its centre: multipole expansion.** 1/|R - s| is expanded in a
  Cartesian Taylor series in the source offset s about the centre of the prism; the coefficients
  a_k = d^k(1/|R|)/k! come from the recurrence
  |k| |R|^2 a_k + (2|k| - 1) sum_i R_i a_{k-e_i} + (|k| - 1) sum_i a_{k-2e_i} = 0, and the moments of the
  box are exact, int s^k dV = prod_i 2 h_i^(k_i+1) / (k_i+1) for even k_i (0 otherwise). U, g and T are
  the sums over k of the moments times a_k and its derivatives ((k_i + 1) a_{k+e_i}, ...). In units of
  the half-diagonal the order-n terms decrease like |R|^-n, so the order is ceil(19 / log10 |R|), at most
  26 (|R| >= 6 half-diagonals).
- **Prism, within three diagonals: reduction to one dimension.** Along the axis w whose cross-section is
  closest to a square, the prism is a stack of rectangular laminae. Over a lamina, 1/r, the attraction and
  the gradients have two-dimensional closed forms (corner sums of x ln(y + r) + y ln(x + r)
  - Z atan(x y / (Z r)), -ln(y + r), -ln(x + r), atan(x y / (Z r)), x / (r (y + r)), y / (r (x + r)),
  1 / r, Z / (r (y + r)), Z / (r (x + r)), with ln(y + r) and 1 / (y + r) rewritten for y < 0); a lamina
  whose centre is more than four lamina diagonals from the point is integrated by a 10 x 10
  Gauss-Legendre rule instead. The laminae are integrated over w with 12-point Gauss-Legendre panels
  whose ends are at +-L 2^-k (k = -2 .. 60) from the plane of the point, so the panels shrink toward
  the plane where the lamina integrals vary on the scale of the point's distance to the lamina edges,
  jump (inside the prism) or have integrable logarithmic singularities (on the surface). T_ww follows
  from Poisson's equation, T_ww = -4 pi G rho [inside] - T_uu - T_vv. Elongated prisms need no special
  treatment: the cross-section of the chosen axis has an aspect ratio of at most sqrt(1000).
- **Columns.** Adaptive QUADPACK (scipy.integrate.quad) over depth with geometric breakpoints, of the
  corner sum of the lamina term near the column and of a 24 x 24 Gauss-Legendre rule of Z / r^3 over the
  lamina beyond two lamina diagonals (unchanged since version 3).
- **Basin depths.** Forward model: for every column a tanh-sinh (double-exponential) rule over [0, h_j]
  (97 nodes, step 1/12) of e^{-lam z} times the corner sum of the lamina term; Jacobian
  dg_i/dh_j = G drho0 e^{-lam h_j} Omega_ij(h_j). Inversion: Levenberg-Marquardt iterations from the
  infinite-slab thicknesses (Marquardt scaling by diag(J^T J), Nielsen's update of the damping, each
  step the least-squares solution of [J; sqrt(mu D)] p = [-r; 0]), run until the step is below 1e-14
  relative (the Gauss-Newton limit), the iterates projected on [1e-6 w_min, 1.5 min(5e4, 3 w_min, 2/lam)];
  if that stalls away from the data (a column held at a bound), the same iteration again with the steps
  reflected at the bounds; ValueError if neither reaches a residual of 1e-10 max |g_obs|.

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
- **One-cell basin (tests/general.py case 1, version 4).** The vertical attraction at the centre of the
  top of the uniform prism 0..100 x 0..200 x 0..300 m (rho = -400) by my own 60-digit closed form,
  -1.1106331967251501474213766859e-5 m s^-2, confirmed to 30 digits by a 50-digit mpmath quadrature over
  depth of the solid angle of the centred laminae, 4 atan(a b / (z sqrt(a^2 + b^2 + z^2))). The depth
  (300 m) is the known input.
- No HDF5 file: every target is stored inline in the test case that uses it, so there is no
  target_evidence record.

Agreement with all stored targets (rechecked in version 4): the reference within 1.1e-3 of the U
tolerance, 0.09 of the g tolerance and 1.7e-4 of the T tolerance, columns to 4.4e-16 relative, depths to
8.9e-13; the second solution within 6.7e-5 (U), 8.9e-5 (g) and 8.0e-7 (T) of the tolerances, columns to
3.3e-16, depths to 1.8e-12. (The figure "columns to 1.5e-13" of earlier versions predates version 3.)

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
   - Cause (added in version 4): "size" was never defined, while the far test points were placed in
     units of the diagonal.
   - Change: "a million times the diagonal of the prism" in the prompts and docstrings of steps 1 and 2,
     and in their contracts.
   - Regression test (added in version 4): case 0 of tests/step_1.py and tests/step_2.py (points from
     2.5 to 1e6 diagonals) pass for the reference and the second solution; crown_check format OK.
5. **Stated error rules without a test.**
   - Finding: several ValueError rules of the visible specification were never exercised.
   - Cause (added in version 4): the invalid-input cases were written for the first error rules and not
     extended when the specification listed more.
   - Change: entries added to the existing invalid-input cases: step 1, x1 = x2; step 2, a non-finite
     point and an empty (0, 3) array (rules "as in prism_gravity"); step 3, non-finite x2, depth and lam
     and y1 = y2; step 4, a 2-D edge array, an edge array with one entry, an infinite edge, a NaN
     observation and drho0 = -inf. The reference and the second solution raise for all of them.
   - Regression test (added in version 4): case 3 of tests/step_1.py, step_2.py and step_3.py and case 1
     of tests/step_4.py, which pass for the reference and the second solution in every crown_check run.
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
   - Cause (added in version 4): the step-3 contract referred to step 1 for the shared conventions ("As in
     step 1") and listed it as a dependency for that reason; the ranges were written before the prompts
     stated the distance range and the number of points per call.
   - Regression test (added in version 4): crown_check format checks the contract keys (OK); by reading,
     solution/step_3.py and the column_gz of second_solution.py call no step-1 function.
8. **Metadata.** Added subfield, tags, expert_time_estimate_hours, the relevant_experience placeholder
   (for the author to fill), difficulty_explanation, solution_explanation, verification_explanation,
   author, affiliation and edit_label: grading_fix.
   - Cause (added in version 4): version 1 had none of these fields.
   - Regression test (added in version 4): crown_check format checks that the metadata keys are present
     (OK); the explanations were reread against the code (they are revised again in version 4).
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
[Historical, corrected in version 4: that runtime figure was wrong (the second solution takes 3 to 11 ms
per stored station), and step 3 case 2 now makes a 1000-station call.]

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

## Version 4 (grading_fix: authoring-guide conformance, pass 2)

The findings of the round-1 independent verification that version 3 had left open, and the pass-2 rules
(every general case calls the final function, no prompt text pointing at hidden tests, a second solution
by a genuinely different method, mutants built on the current reference, truthful metadata), handled as
follows. The four blocking round-1 findings (step-4 domain and solver, step-4 coverage, far columns,
elongated prisms) were fixed in version 3 and were rechecked, not changed.

1. **General case 1 never called invert_basin_depths.**
   - Finding: tests/general.py case 1 checked only column_gz against prism_gravity and the trace of
     prism_gradients; crown_check format reported "general case 1 never calls the final function". Its
     comment also called the bound 2.1e-10 |g| + 2e-14 G |rho| L "the sum of the two stated tolerances",
     which it is not (the sum is 2e-10 |g| + 1e-14 G |rho| L).
   - Cause: the case was written as a cross-step consistency check before the rule that every whole-task
     case must use the final function.
   - Change: the case now also inverts the one-cell basin made of its column (100 m x 200 m, 300 m deep,
     i.e. 3 w_min, the inclusive limit of the stated domain, lam = 0, drho0 = -400): from the independent
     value at the centre of the top (60-digit closed form, confirmed by a 50-digit quadrature; see
     Targets) the depth must be 300 m within the stated 1e-8, and from column_gz's own value there within
     2e-8 (about twice the single-output tolerance; the depth amplifies relative data errors by 6.7 here,
     so column_gz's 1e-10 adds at most 7e-10). The comment now says "about twice the step-1 tolerance".
   - Regression test: general case 1 passes for the reference (depth error 1.6e-15) and the second
     solution (2.2e-16); the whole-task mutant fails it (its depth is 6.6e-4 m instead of 300 m);
     crown_check format reports no problem.
2. **Step 3: "up to 1000 stations per call" was stated but never exercised.**
   - Finding: at most 5 stations were tested per call; version 2 justified this by a runtime of 0.14 s
     per station of the second solution, which was wrong.
   - Cause: a runtime estimate that was not measured.
   - Change: case 2 of tests/step_3.py now also calls column_gz once with 1000 stations (the 10 near and
     4 far stored stations of the first column, repeated) and checks every value at 1e-10 against the
     stored targets (the 4 far targets are copied from case 4 so that the case stays self-contained).
     Measured: the second solution takes 3 to 11 ms per stored station (about 7 s for the 1000), the
     reference about 0.1 s for the call. The Version 2 note is labelled as historical.
   - Regression test: step 3 case 2 passes for the reference and the second solution; the mutants
     uniform_gauss_legendre and uniform_density still fail it.
3. **The whole-task mutant left invert_basin_depths unchanged.**
   - Finding: whole_task_upward_z flipped g_z in prism_gravity and column_gz only, so the integrated
     result was the reference's and the mutant failed only through the column_gz assertions; its
     description ("every vertical component has the wrong sign") was inaccurate.
   - Cause: the sign was changed only where g_z is returned (prism_gravity, column_gz); invert_basin_depths
     has had its own forward model (_basin_forward) since version 1, so the mistake never reached the
     integrated result.
   - Change: it is replaced by whole_task_mgal, the current reference with one mistake carried through
     the whole task: the attraction is handled in mGal (1 mGal = 1e-5 m s^-2, the unit in which gravity
     anomalies are reported) instead of m s^-2, so prism_gravity and column_gz return it 1e5 times too
     large and invert_basin_depths reads gz_obs as mGal and returns a basin at least 1e5 times too shallow
     (a column needs at least the depth of the infinite slab of the same attraction).
     mutants/whole_task_upward_z.py is deleted.
   - Regression test: crown_check mut, whole_task_mgal fails general cases 0, 1 and 2 (in case 1 at the
     new inversion of the one-cell basin, since its column_gz and prism_gravity agree with each other).
4. **The second solution used the reference's strategies with other parameters (steps 1, 2 and 4).**
   - Finding: for the prism it used corner sums near the prism and a tensor Gauss-Legendre rule far away
     (three instead of two diagonals, 24^3 instead of 16^3 points) and cut elongated prisms (bisection
     instead of equal pieces); for the basin, trust-region reflective least squares (scipy's
     least_squares, as the reference) then Newton steps on a composite Gauss-Legendre forward model.
   - Cause: the second solution was derived from the reference by changing parameters rather than
     methods.
   - Change: new methods, described under "Second solution (version 4)" above: for the prism, the
     multipole (Cartesian Taylor) expansion with exact box moments beyond three diagonals, and within
     three diagonals a reduction to one dimension (two-dimensional closed forms of the laminae, or a
     10 x 10 Gauss-Legendre rule for laminae more than four lamina diagonals away, integrated along the
     axis with the most compact cross-section on 12-point Gauss-Legendre panels that shrink
     geometrically toward the plane of the point; T_ww from Poisson's equation), with no special
     treatment of elongated prisms; for the basin, a tanh-sinh rule over depth in the forward model and
     Levenberg-Marquardt iterations (projected on the admissible box, or reflected at its bounds if that
     stalls) run to convergence instead of trust-region reflective least squares plus Newton. The column
     (step 3, adaptive QUADPACK) was already a different method and is unchanged. second_solution.py no
     longer imports scipy.optimize. Prototypes tried on random basins and rejected: continuation in the
     data from the slab thicknesses (the path leaves the positive depths on rough basins, whose depths
     move by up to 1e6 times any relative change of the data), Powell's dogleg with projection (1 failure
     in 50 rough 50- to 100-cell basins) or in log-depth (depths running off to large values),
     Levenberg-Marquardt with projection only (1 failure in 50) or with reflection only (2 in 100); the
     projected and reflected variants fail on different basins, and their combination, as now in
     second_solution.py, failed on none of 200 such basins in the prototype.
   - Regression test: the second solution passes every case. Against the stored targets it is within
     6.7e-5 (U), 8.9e-5 (g) and 8.0e-7 (T) of the step-1 and step-2 tolerances, columns 3.3e-16, depths
     1.8e-12. Against my own 90-digit closed form of U (g and T by mpmath differentiation) on 28 random
     prisms (aspect ratio up to 1000, 8 of them at exactly 1000; 10 points each: inside, on a face, an
     edge and a corner, near, at 2 to 3.5 diagonals and up to 1e6 diagonals): within 9.2e-4 (U), 3.6e-4
     (g) and 4.4e-5 (T) of the tolerances (the reference on the same points: 3.2e-3, 4.3e-3, 2.3e-4).
     240 random basins of the stated domain (120 of 1 to 100 cells with constant, random, bimodal,
     bowl-shaped and rough depths; 120 of 50 to 100 cells with mostly rough and bimodal depths; lam from 0
     to the limit lam depth = 2; both signs of drho0): the reference inverts data computed with the second
     solution's forward model and the second solution data computed with the reference's (they agree to
     about 1e-15): no failure, worst depth error 9.6e-12 (reference) and 1.5e-11 (second), at most 12 s
     and 4.5 s per basin. Runtime of the new prism code for 1000 points within two diagonals: 3 to 4.5 s
     near a 1 x 1 x 1000 rod, 0.6 to 1.5 s near a plate or a cube. No warnings in the step cases and
     general cases 1 and 2 (run with -W error), and identical outputs (SHA-256 of the outputs of all four
     functions, the rod, a far column and the rough basin included) in two clean runs and with one and
     two BLAS threads, for the reference as well.
5. **Prompt text pointing at hidden test inputs.**
   - Finding: the step-3 prompt said "(a station a million column widths away included)" and "(a
     station on the surface 1e-6 m from an edge included)", remarks pointing at stored test stations;
     difficulty_explanation named the test rod (2 m x 2 m x 2 km).
   - Cause: the remarks were added to make the stated range concrete.
   - Change: the step-3 accuracy sentence states the regime only ("up to 1e7 m from the column, whatever
     the column's width ... at any horizontal distance from its edges, however small"); the metadata give
     measured figures for generic prisms and columns. Every magnitude in the prompts was rechecked over
     the whole stated domain (for example |g| < 1e-12 G |rho| L at a million diagonals holds for every
     box, since volume / diag^2 <= L / 3).
   - Regression test: crown_check format OK; the step-3 cases are unchanged apart from case 2.
6. **Inconsistent figures in source.md and the metadata.**
   - Finding: the round-1 report noted that source.md described the basin depth panels as starting at a
     quarter of the smallest cell width (the code starts at w_min / 64), and that the cancellation figures
     of the corner sums differed between source.md and difficulty_explanation; stale agreement figures
     ("columns to 1.5e-13") remained.
   - Cause: text not updated with the code, and rounded figures quoted from different measurements.
   - Change: the panel description had been corrected in version 3 (source.md, solution_explanation and
     the code comment all say a sixty-fourth); the cancellation figures were measured again
     (mutants/step_1_closed_form_everywhere.py against my own 60-digit closed form over 50 random
     directions, both test prisms) and the same medians and maxima are now quoted in source.md and
     difficulty_explanation; the step-3 figures (a 64-point rule 4.1e-5 1 mm from an edge and 1.0e-3 1 m
     from a corner, 1024 points 6.0e-6; the lamina corner sum 9e-8 at 5000 widths, of order one at 1.7e5
     and 1e7 widths) were remeasured against the stored targets; the agreement figures are recomputed.
   - Regression test: rereading of source.md, problem.yaml and the code; the figures come from the
     scripts named above.
7. **Version 2 items without cause or regression test.** Items 4, 5, 7 and 8 of Version 2 now carry
   Cause and Regression test lines (marked "added in version 4"); its wrong runtime claim is labelled.
8. **Mutants and dependencies.** Every step mutant was rechecked by diff to be the current reference with
   one change (closed_form_everywhere switches off both the far-field rule and the cutting of elongated
   prisms, i.e. evaluates the closed form as written; its description now says so, with measured
   figures). The step-4 prompt now says that the anomaly is the sum of the column_gz values of the
   columns, the use that its step_dependencies [3] records; step 2 keeps [1] because its prompt defines
   the tensor and its error rules through prism_gravity.
9. **Not changed, with reasons.** The g tolerance of step 1 (1e-10 |g| + 1e-14 G |rho| L) is dominated by
   its absolute term a million diagonals away, where |g| is about 1e-13 G |rho| L; this was a design
   choice, unchanged so as not to change tolerances. The per-step solution files repeat the same private
   helpers (identical code; only the evaluated functions are graded). relevant_experience is still the
   placeholder for the author.

Rerun: `OMP_NUM_THREADS=1 python3 tools/crown_check.py tasks/prism_basin_gravity -j 2` (all modes, with the
version-4 checker, whose format mode also reports general cases that never call the final function and
whose shift mode also shifts every evaluated function in the general cases): FORMAT OK (no problem); REF
steps 1-4 5/5, 5/5, 5/5, 6/6 and general 3/3; SECOND the same; mutants: closed_form_everywhere (step 1)
fails cases 0, 2, 4, direct_log 1, 2, no_subdivision (step 1) 4, closed_form_everywhere (step 2) 0, 2, 4,
no_subdivision (step 2) 4, uniform_gauss_legendre 0, 2, uniform_density 0, 2, 4, corner_sum_far 2, 4
(case 2 now has far stations), slab_thickness 0, 2, 3, 4, 5, newton_step_halving 5, whole_task_mgal
general 0, 1, 2; SHIFT +-1e-12 all pass (steps and general); all-None and all-zero controls pass 0 cases
in every step and in general; SUMMARY ALL OK.

Not changed: the physics, the reference solution (solution.py and solution/), every stored target of
earlier versions, and every stated tolerance (U 1e-10, g 1e-10 |g| + 1e-14 G |rho| L, T 1e-9 ||T||,
columns 1e-10, depths 1e-8); relevant_experience stays the placeholder for the author.
