# Sources, provenance of the targets and revision history

## Sources checked

| Element | Source | Used for |
|---|---|---|
| Minimum-curvature trajectory | Sawaryn & Thorogood (2005) | ratio factor, great-circle arc between stations |
| Layered steady conduction | Bullard (1939); Carslaw & Jaeger (1959) | R(z), S(z), geotherm with uniform heat production |
| Paleoclimate perturbation | Birch (1948); Beck (1977) | erfc step response, superposition of a history |
| Shield magnitudes | Jessop et al. (1984); Mareschal & Jaupart (2004) | inclined holes, q0 of 35-55 mW m^-2, glacial correction |
| Layer over a half-space (step 5 targets) | Carslaw & Jaeger (1959), ch. 12 | image series, alpha = (e1 - e2)/(e1 + e2), e = sqrt(k rho_c) |
| Fixed Talbot contour (step 5 reference, mpmath targets) | Abate & Valko (2004) | contour and trapezoid weights |
| Hyperbolic contour (step 5 second solution) | Weideman & Trefethen (2007) | contour parameters |
| Gaver-Stehfest (step 5 mutant, 50-layer cross-check) | Stehfest (1970) | weights and nodes |

Abate, J., Valko, P.P. (2004) Int. J. Numer. Meth. Eng. 60, 979-993. Weideman, J.A.C., Trefethen, L.N. (2007) Math.
Comp. 76, 1341-1356. Stehfest, H. (1970) Commun. ACM 13, 47-49. Carslaw, H.S., Jaeger, J.C. (1959) Conduction of
Heat in Solids, 2nd ed., Oxford. The other works are listed in background.md. Every formula taken from them was
checked numerically against independent targets. The author's own work: the decomposition, the branch-cut test
oracle, the data constructions of steps 6 and 7 and the likelihood derivations (standard theory). No code copied.

## Data and targets (no value comes from solution.py or second_solution.py)

All data are synthetic and built inside each test case (no data file). Each case header names its targets.

- Step 1: exact straight holes and constant-build arcs (R sin(MD/R) between stations), a constant-azimuth segment
  with inclination linear in MD (closed form without cancellation, also for doglegs of 8e-10 to 2e-6 rad on 20 km),
  8-decimal three-dimensional values checked by adaptive quadrature of the arc tangent (5e-9 m), and eight random
  surveys against scipy quad of the great-circle tangent computed in the case.
- Step 2: hand-computed layer sums (harmonic, not arithmetic, effective conductivity), R = z/k1 and S = z^2/(2 k1)
  under a deep stack, six random columns against quad of 1/k and z/k.
- Step 3: erfc forms with math.erfc (down to 0.05 m under a 2-year interval), a multistep history checked by Duhamel
  convolution (5e-9 K), linearity and decay.
- Step 4: exact recovery on noise-free designs (n = 4 included), numpy.linalg.lstsq with the OLS covariance, a
  P column with prescribed scaled singular values (3e-9 rejected, 3e-8 fitted), and a residual orthogonal to
  [1, R, P] with rms exactly 1e-6 K (standard errors from a QR factorization of the design).
- Step 5: the erfc half-space (also a single-layer column), the Carslaw-Jaeger image series, 90-digit mpmath values (transfer-matrix transform,
  mpmath Talbot) for a five-layer Abitibi column and a contrasted column (five of them recomputed by forward transfer
  matrices at 40 digits: 5e-17 K), and six depths of a 50-layer column computed at 267-392 digits with fixed Talbot
  of degree 60 and 90 and Gaver-Stehfest with 60 terms (agreement 2e-16).
- Oracle of steps 6 and 7 (author's derivation): the Bromwich contour collapsed on the cut s <= 0,
  U = 1 + (2/pi) int exp(-w^2) Im F+(w^2/t)/w dw and dU/dt = -(2/(pi t)) int exp(-w^2) w Im F+ dw, with real
  transfer matrices from the surface and the radiation condition of the bottom half-space, 24-point Gauss-Legendre
  panels on w in [0, 7]. Against the stored 90-digit values: 3.3e-16, 8.9e-16 and 4.4e-16 K; against erfc 6.7e-16,
  its time derivative 2.7e-16 relative; against a parabolic-contour inversion 1.5e-14. A uniform column uses
  math.erfc and its closed-form derivative.
- Step 6: on the used readings y = T + A S = T0 + q0 R + g P(t_end) + r, r a seeded vector projected orthogonal to
  the oracle Jacobian [1, R, P, g dP/dt_end] and scaled to the stated rms, so the generating values are the exact
  least-squares solution (RSS_min = |r|^2); shallower readings get values that follow no model. Wald errors from the
  oracle Jacobian; F(1, nu) quantile from the integrated Student t density (agrees with scipy.stats.f.ppf to 6e-13);
  interval ends = the oracle profile's nearest crossings (outward march, root finder). Dense scans (6000 times per
  case) find no point below RSS_min. Cases: inclined 7-layer hole (326 readings, 11132.9-11891.8 yr); short
  contrasted hole (38, F 4.13 against 3.84, asymmetric 7728-10729 yr); uniform column without heat production, the
  collar reading at z_min = 0 (121); short glaciation with a second minimum at 23.97 ka (0.54 Delta above RSS_min) and a second call
  whose bounds contain both basins; end-point minima and a precise log with the minimum 55 years inside either
  bound; uncontained intervals; five readings (F(1, 1) = 161.4) at rms 1e-4 K with a two-value history (m = 2,
  8725.6-11680.7 yr; 6000-point scan: single excursion below the threshold) and the three stated error rules.
- Step 7 and whole task: in whitened space u_h = L(phi_h) r_h = b_h + c_h (readings by MD, L the Prais-Winsten
  matrix); b_h orthogonal to the whitened oracle Jacobian with |b_h|^2 = n_h noise_h^2, turned on its sphere until
  the phi_h score n (sum u_i r_(i-1) + phi r_1^2) - |u|^2 phi/(1 - phi^2) vanishes; c_h in the span of the projected
  L P_h and L g dP_h/dt_end, one hole closing the weighted shared scores in closed form. Every score of the exact
  log-likelihood vanishes at the generating values (below 6e-9 standard deviations by central differences in all
  4H + 2 parameters). Dense scans (600 logarithmic times per region plus 101 near t_end, grid over g and every phi,
  Newton polish) find no value above l_max (largest excess 5.5e-12). Standard errors from the expected information
  of the whitened oracle Jacobians, l_max from the generating residuals, the chi-square quantile from math.erf.

| Region | Holes: n_used (s mK, phi) | g, t_end | 95 % interval (yr) | Feature |
|---|---|---|---|---|
| step 0 | 163 (4.0, 0.35); 137 (12.3, 0.70); 136 (30.0, 0.85) | 7.8, 11500 | 10864.8-12178.0 | types; s_h at 30 mK |
| step 1, 5 | 24 (2.4, -0.30); 220 (20.0, 0.60) as two logging runs | 6.0, 9000 | 8861.8-9134.6 | MD order; case 5 containment |
| step 2 | uniform 121 (6.0, 0.50) with the collar reading; 57 (15.8, 0.25), A = 0 | 6.2, 10500 | 8374.3-14770.2 | erfc oracle |
| step 3 | 326 (2.5, 0.60); 137 (5.1, 0.45); 136 (10.1, 0.75); then the whole 2 region | 7.8, 11100 | 10496.5-11837.4 | 2nd max 17.9 ka, 9.95 below |
| step 4 | step 0 region (25.6 mK); precise 326, 137, 136 at 1.01, 1.00, 1.02 mK | 7.8, 11500 | precise 11446.0-11558.9 | end points, 77-yr controls |
| step 6 | 8 of 10 readings (4.5, 0.0); 137 (8.0, 0.50); m = 2 | 7.0, 10000 | 8980.7-11239.8 | smallest hole; 7 readings raise |
| step 7 | 37 (3.0, 0.90); 136 (10.0, 0.40) | 7.5, 11000 | 8694.3-14519.6 | first reading matters |
| whole 0 | 220 (10.0, 0.80); 109 (3.4, 0.30); 136 (18.0, 0.55) | 5.0, 9800 | 9562.2-10040.9 | deep region |
| whole 1 | 24 (2.2, 0.40); 220 (20.0, 0.65) | 6.0, 9000 | 8834.1-9169.5 | offset and order invariances |
| whole 2 | 326 (5.0, 0.50); 137 (9.1, 0.70); 136 (15.1, 0.30) | 7.4, 11070 | 9983.2-12584.3 | 2nd max 18.17 ka 1.25 below |

Whole-task region 2: the statistic falls below the quantile again over 17767-18444 years, outside the interval.

Error budgets (reference run with P off by 1e-8 K per K in smooth, irregular and residual-aligned patterns, TVD
off by 1e-6 m, R and S by a relative 1e-9): at most 0.31 of a step 6 limit (0.45 at rms 1e-4 K); at most 0.61 of a
step 7 limit on the test regions (0.72 for the aligned pattern on the log-likelihood) and 0.77 on regions rebuilt
at the ends of the domain; smooth errors below 0.023 of every limit. At s_h = 1e-4 K the irregular pattern moved
phi by 3.5 times its limit, hence the 1e-3 K floor of the domain. Along the amplitude-timing ridge of the
short-glaciation regions (d ln g/d ln t_end about 2.05) the g limit binds first.

## Revision history

- Versions 1-7: steps 1-4 and the linear workflow surface_heat_flow; v5 added error-contract tests; v7 rollouts
  solved 8 of 8. Version 8 (difficulty increase): the layered perturbation (Laplace transform, Talbot inversion;
  mpmath and image-series targets; Stehfest and erfc mutants). Version 9: content-check fixes (no clock reads,
  size stated and checked by values, n = 4 fit, inclusive z_min reading).
- Revisions 10a and 10b (authoring-guide conformance, before the v10 upload): skeleton schema, layered step and
  workflow swapped (final step last), self-contained cases, per-step solutions without redefinitions, declared
  imports, one-error mutant files, stated tolerances consistent with the tests, contracts and metadata; second
  solution made independent (circular-arc TVD, ramp sums and integration by parts for R and S, similarity-variable
  Duhamel quadrature from tau = 0), reference S factored as (b - a)(b + a)/(2k) (exact zero below z), collar
  reading at z_min = 0, independent 50-layer targets. Each change has a regression case; crown_check all modes.
- Version 10 (difficulty_increase; v9 was "too easy", 8 of 8). Round 1: step 6 joint inversion of T0, q0, g and
  t_end with the layered perturbation, Wald errors (n - 4) and the F-calibrated profile interval, the branch-cut
  oracle and the orthogonal-residual targets; repair: second calls with both basins inside the bounds (a local
  search passed), domain-floor budget, prompt method hints removed. Round 2: regional step 7 (shared g, t_end);
  repair: end-cell refinement of the t_end scan (a maximum in an end cell was reported as an end point; near-boundary
  valid controls added) and test helpers renamed after a clash with candidate names. Round 3: stationary AR(1) noise per
  hole with exact likelihood, expected information and likelihood-ratio interval (whitened-space targets); repair:
  whole-task region 2 retuned so that the statistic falls below the quantile again outside the interval (new
  nearest-crossing check), tie rule for equal MD, budget claim limited to the measured regions. Internal blind-solver
  calibration (Claude subagents written from the visible text only; not platform rollouts, not an independent
  review) passed every case in rounds 1-3 (two solvers on 71 cases, 84 of 84 case runs, three solvers on 79 cases);
  it finds specification gaps but does not predict the platform band. Item 16: the step 6 prompt no longer lists the Jacobian columns or hints at variable
  projection. Each round reran crown_check in all modes (ALL OK).
- Version 10 upload refused (2026-10-12): domain mismatch and domain not accepted (domain set to physics),
  incomplete metadata block (author fields; author_profile and conflicts_of_interest pending from the author), and
  "too large to store" (1752 KB; limit 800 KB; raw folder 1530 KiB).
  Cause of the size: duplicated oracle code with long comments in every self-contained case of steps 6-7 and the
  whole task (383 KiB), 53 mutant files (750 KiB; the whole-task ones full solution copies) and a 161 KiB source.md.
  Change (edit_label unchanged): (1) mutants reduced to a representative set, stripped of docstrings and comments
  (the MUTANT markers kept): linear TVD, arithmetic conductivity, years not converted, heat-production sign, mean-
  diffusivity erfc, double-precision Gaver-Stehfest, step 6 half-space perturbation and local search without scan,
  step 7 independent errors and conditional likelihood, whole-task MD as depth and independent errors; (2) tests:
  every case keeps its assertions, inputs and targets; the oracle library of each case was rewritten compactly
  (2-space indentation, no comments, unused helpers pruned; the step 7 profile maximized over the phi_h by
  Nelder-Mead instead of damped Newton steps, over g by a 4001-point grid with bounded Brent refinement instead of
  derivative bisection; phi-score roots by Brent; step 6 crossings by Brent instead of Illinois steps, the t density
  integrated by quad); file preambles moved into the case-0 headers (the parser counts text before the first case);
  (3) problem.yaml: mutant lists and the three explanations condensed (prompts, headers, return lines and contracts
  unchanged); (4) this file condensed. Regression (every old and new case of steps 6, 7 and the whole task
  instrumented at the points its assertions evaluate): same data (largest difference 7e-15 K), targets within about
  3e-12 relative (sigma_amplitude, step 6 cases 1, 2, 5), thresholds within 5e-13, oracle profile values within about
  3e-6 (step 7 case 4, precise log; 1e-7 to 4e-7 elsewhere); every above/below-threshold outcome is unchanged and the
  smallest |l_p - threshold| at a checked point is 4.0e-3 (step 7 case 2), about 1500 times the largest difference.
  Rerun: crown_check ref, second, mut, shift and controls ALL OK; format also listed 58 non-finite literals present
  before the shrink (next item). The shrink had dropped the erfc oracle of step 6 case 2 (the branch-cut oracle,
  within 3.1e-16 of erfc on that column, was used) and left a removed mutant named in verification_explanation.
- Content-check preparation (2026-10-12, edit_label unchanged). Finding: 58 non-finite literals (platform check
  "Edge-case inputs", blocking): 29 NaN test inputs (steps 1-5: 19; step 6 and step 7 case 6: 5 each) and 29 test
  sentinels (math.inf as the bottom of the last layer in the layer sums, -math.inf for |phi| >= 1 in the step 7 and
  whole-task profiles). Cause: the prompts promised ValueError for non-finite inputs and the validation cases tested
  it. Change: NaN entries removed from the validation cases of steps 1-7 (step 6 keeps 24 invalid inputs, step 7
  29) and the non-finite clauses from the prompts, function headers, problem_io and scaffolds (the references keep
  their checks, now neither stated nor graded); the last layer's bottom is max(z, last top) (same arithmetic) and the
  profile returns -1e300 for |phi| >= 1 (Nelder-Mead only compares values: same path); step 6 case 2 again uses
  math.erfc and its closed-form time derivative (UHole, as UBore in step 7 case 2: data change 3.6e-15 K, targets
  3e-12 relative); the removed step 6 mutant is no longer named. Regression: data, targets, thresholds and
  oracle profile values of every step 6, step 7 and whole-task case are bitwise identical before and after, except
  step 6 case 2 (3.0e-12 relative, the erfc change). Rerun: crown_check in all modes, ALL OK (format: only the two
  pending metadata keys); raw folder below 600 KiB.
- Pre-upload content-check emulation (2026-10-12, last version; edit_label stays difficulty_increase). Finding: an
  emulation of the platform content check on steps 1-7 and the whole task found (1) validation outweighing computed
  values (counted as the platform does, raises at half weight): step 6 17 vs 12, step 7 22.5 vs 14, step 5 10 vs 11,
  repeated type/shape asserts in steps 1-3; (2) stated but untested behaviour: one-element array outputs, a single-
  layer column, m = 2 histories, heat_production = 0, the domain ends rms_residual = 1e-4 K (step 6), s_h = 1e-3 and
  0.03 K and phi_h = -0.9 and 0.95 (step 7), the equal-MD tie rule, the nearest-crossing rule only in general.py;
  (3) grounding: relative accuracy of the step 4 standard errors although exact data make them 0, zero tolerance for
  an all-zero dT, relative limits on g and q0 without q0 > 0 or g != 0. Cause: long error lists (dimensionality,
  lengths, list types, key sets, history layout) tested one by one, and domain statements without endpoint cases.
  Change: stated errors reduced to the scientific rules (query outside the survey, dogleg near 180 degrees; k <= 0;
  kappa <= 0; n < 4, A < 0, unresolvable design; k or rho_c <= 0; step 6: < 5 used readings, unresolvable at t_end,
  s_0 = s_1, end-point minimum, uncontained interval; step 7: < 8 used readings, end-point maximum, uncontained
  interval), each tested once, the other conditions stated as input ranges (references still check them) in prompts,
  headers, problem_io, scaffolds and contracts; one output-contract assert per step (one-element arrays included);
  step 4 case 4 rebuilt at rms_residual = 1e-6 K (QR standard errors), the standard-error accuracy now stated for
  rms_residual >= 1e-6 K; single-layer column (step 5); step 6 case 6 at rms 1e-4 K with m = 2; A = 0 in step 6
  and 7 case 2 (A S is added back exactly); step 7 case 0 hole 3 at s_h = 30.0 mK and case 4 precise hole 2 at
  1.000 mK (noise / sqrt(1 + 6.25/n)); step 7 case 6 with m = 2; step 7 case 3 also runs the whole 2 region; distinct
  MDs required instead of the tie rule; phi domain narrowed to the tested -0.3 to 0.9 (inside the budget regions);
  dT "not all zero", q0 > 0 and why g != 0 (g = 0 would make the profile constant) stated. After: validation vs
  computed 2.5/23, 1.5/24, 1.5/15, 5/20, 2/12, 4.5/12, 3.5/17. Regression: every changed case run with the
  reference, solution.py and the second solution (worst error/limit 0.018, second solution 0.009); oracle scans of
  the changed regions (step 6 case 6: 6000 times; step 7 cases 0, 4, 6: 241-341 times) find the extremum at t_end
  (largest excess 7.5e-12) and one interval. Rerun: crown_check in all modes on the final files (one thread,
  -j 4): ALL OK, FORMAT OK; every mutant still fails (step 4 sign mutant now also case 4); raw folder below 600 KiB.
