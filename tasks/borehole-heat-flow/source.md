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
  8-decimal three-dimensional values checked by adaptive quadrature of the arc tangent (5e-9 m), eight random
  surveys against scipy quad of the great-circle tangent computed in the case, and the planar closed form for a
  dogleg of 180 deg - 2e-6 rad (checked at 50 digits).
- Step 2: hand-computed layer sums (harmonic, not arithmetic, effective conductivity), R = z/k1 and S = z^2/(2 k1)
  under a deep stack, six random columns against quad of 1/k and z/k.
- Step 3: erfc forms with math.erfc (down to 0.05 m under a 2-year interval), a multistep history checked by Duhamel
  convolution (5e-9 K), linearity and decay.
- Step 4: exact recovery on noise-free designs (n = 4 included; P also in units of 1e-4), numpy.linalg.lstsq with
  the OLS covariance, residuals orthogonal to [1, R, P] (rms exactly 1e-6 K, standard errors from a QR factorization;
  rms 1e-3 K with P nearly proportional to R at scaled singular values 3e-8 and 1.05e-3, standard errors from the
  Frisch-Waugh identity; checked against 80-digit mpmath least squares on the same inputs) and 3e-9 rejected.
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
  8725.6-11680.7 yr; 6000-point scan: single excursion below the threshold) and three of the four stated error rules.
- Step 7 and whole task: in whitened space u_h = L(phi_h) r_h = b_h + c_h (readings by MD, L the Prais-Winsten
  matrix); b_h orthogonal to the whitened oracle Jacobian with |b_h|^2 = n_h noise_h^2, turned on its sphere until
  the phi_h score n (sum u_i r_(i-1) + phi r_1^2) - |u|^2 phi/(1 - phi^2) vanishes; c_h in the span of the projected
  L P_h and L g dP_h/dt_end, one hole closing the weighted shared scores in closed form. Every score of the exact
  log-likelihood vanishes at the generating values (below 6e-9 standard deviations by central differences in all
  4H + 2 parameters). Dense scans find no value above l_max beyond 7.5e-12: 600 logarithmic times plus 101 near
  t_end (grid over g and every phi, Newton polish) per original region, and 300 + 41 (cases 0, 6) or 200 + 41 (case 4,
  precise region) for the regions rebuilt at the domain ends. Standard errors from the expected information
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
| whole 1 | 24 (2.2, 0.40); 220 (20.0, 0.65) | 6.0, 9000 | 8834.1-9169.5 | offset, order, sign of g |
| whole 2 | 326 (5.0, 0.50); 137 (9.1, 0.70); 136 (15.1, 0.30) | 7.4, 11070 | 9983.2-12584.3 | 2nd max 18.17 ka 1.25 below |

Whole-task region 2: the statistic falls below the quantile again over 17767-18444 years, outside the interval.

Error budgets (re-measured 2026-10-12 on every current step 6, step 7 and whole-task call; reference with P off by
1e-8 K per K of max|dT| smooth in z and t, aligned with the residual sign or irregular in z and t, or with the TVD
off by 1e-6 m and R, S by a relative 1e-9): smooth and geometric errors at most 0.11 of a step 6 limit (rms_residual,
case 6) and 0.046 of a step 7 or whole-task limit; aligned 0.93 of the step 6 rms_residual limit at rms 1e-4 K (0.32
for the other step 6 outputs) and 0.72 of the step 7 log-likelihood limit (case 4, precise region; 0.30 otherwise);
irregular 0.62 of a step 7 limit (phi, case 4; 0.19 otherwise) but 2.65, 1.81 and 1.29 of the step 6 g limit in
cases 1, 2 and 5 (t_end 1.89, 1.38, 0.92; T0 1.60 in case 1; at most 0.53 in cases 0, 3, 4, 6): an error irregular
in t perturbs RSS(t) itself (at 1e-9: 0.08 at most; a dense oracle scan of the perturbed case 1 profile finds the
same displacement), so the step 6 accuracy is stated for step 5 errors smooth in t_end. Exact inputs: reference 1.7e-3 (step 6) and 0.019 (step 7, whole task), second solution (its own steps 1-5)
9e-4 and 0.009. At s_h = 1e-4 K (design region) the irregular pattern moved phi by 3.5 times its limit, hence the
1e-3 K floor of the domain. Along the amplitude-timing ridge of the short-glaciation regions (d ln g/d ln t_end about
2.05) the g limit binds first.

## Revision history

- Versions 1-7: steps 1-4 and the linear workflow surface_heat_flow; v5 added error-contract tests; v7 rollouts
  solved 8 of 8. Version 8 (difficulty increase): the layered perturbation (Laplace transform, Talbot inversion;
  mpmath and image-series targets; Stehfest and erfc mutants). Version 9: content-check fixes (no clock reads,
  size stated and checked by values, n = 4 fit, inclusive z_min reading).
- Revisions 10a and 10b (authoring-guide conformance, before the v10 upload): skeleton schema, final step last,
  self-contained cases, no redefinitions, declared imports, one-error mutants, tolerances as tested, contracts and
  metadata; second
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
  calibration (Claude subagents given only the visible text; not rollouts, not an independent review) passed every
  case in rounds 1-3; it finds specification gaps, not the platform band. The step 6 prompt no longer lists the
  Jacobian columns or hints at variable projection. Each round reran crown_check in all modes (ALL OK).
- Version 10 upload refused (2026-10-12): domain not accepted (set to physics), incomplete author metadata, and
  "too large to store" (1752 KB; limit 800 KB). Cause: oracle code with long comments repeated in every case of
  steps 6-7 and the whole task (383 KiB), 53 mutant files (750 KiB) and a 161 KiB source.md. Change (edit_label
  unchanged): mutants reduced to the representative set of problem.yaml, without docstrings; every test case keeps
  its inputs, targets and assertions, its oracle library rewritten compactly (step 7 profile over the phi_h by
  Nelder-Mead, over g by a 4001-point grid and bounded Brent; roots by Brent; t density by quad); preambles moved
  into the case-0 headers; texts condensed. Regression (assertion values instrumented): data within 7e-15 K,
  targets within 3e-12 relative, thresholds within 5e-13, oracle profile values within 3e-6 (step 7 case 4, precise
  log; 4e-7 elsewhere); every above/below-threshold outcome unchanged (smallest margin 4.0e-3, step 7 case 2).
  Rerun: crown_check ref, second, mut, shift, controls ALL OK; format listed 58 non-finite literals (next item). The
  shrink had dropped the erfc oracle of step 6 case 2 (branch-cut oracle used, 3.1e-16 from erfc) and left a
  removed mutant named in verification_explanation.
- Content-check preparation (2026-10-12, edit_label unchanged). Finding: 58 non-finite literals (platform check
  "Edge-case inputs", blocking): 29 NaN test inputs and 29 test sentinels (math.inf as the last layer's bottom,
  -math.inf for |phi| >= 1 in the profiles). Cause: the prompts promised ValueError for
  non-finite inputs. Change: NaN inputs and the non-finite clauses removed (prompts, headers, problem_io, scaffolds;
  the references keep their checks, neither stated nor graded); the last layer's bottom is max(z, last top) and the
  profile returns -1e300 for |phi| >= 1 (same arithmetic, same Nelder-Mead path); step 6 case 2 again uses math.erfc
  (data change 3.6e-15 K); the removed step 6 mutant no longer named. Regression: data, targets, thresholds and
  oracle profile values of every step 6, step 7 and whole-task case bitwise identical except step 6 case 2 (3.0e-12
  relative). Rerun: crown_check all modes ALL OK (format: two pending metadata keys).
- Pre-upload content-check emulation (2026-10-12, last version; edit_label unchanged). Findings: (1) validation
  outweighing computed values (raises at half weight: step 6 17 vs 12, step 7 22.5 vs 14, step 5 10 vs 11) and repeated
  type/shape asserts in steps 1-3; (2) stated but untested behaviour (one-element outputs, a single-layer column, m = 2, heat_production = 0, the domain ends of
  rms_residual, s_h and phi_h, the equal-MD tie rule, the nearest-crossing rule outside general.py); (3) grounding:
  relative accuracy of step 4 standard errors that exact data make 0, zero tolerance for an all-zero dT, relative
  limits on g and q0 without q0 > 0 or g != 0. Cause: long error lists tested one by one; domain statements without
  endpoint cases. Change: stated errors reduced to the scientific rules, each tested once, the other conditions
  stated as input ranges; one output-contract assert per step; step 4 case 4 at rms_residual = 1e-6 K; single-layer
  column (step 5); step 6 case 6 at rms 1e-4 K with m = 2; A = 0 in step 6 and 7 case 2; step 7 case 0 hole 3 at
  30.0 mK, case 4 precise hole 2 at 1.000 mK, case 6 with m = 2, case 3 also on the whole 2 region; distinct MDs
  instead of the tie rule; phi domain -0.3 to 0.9; dT "not all zero", q0 > 0 and g != 0 stated. After: validation
  vs computed 2.5/23, 1.5/24, 1.5/15, 5/20, 2/12, 4.5/12, 3.5/17. Regression: changed cases with the reference,
  solution.py and the second solution (worst error/limit 0.018 and 0.009); oracle scans of the changed regions (step
  6 case 6: 6000 times; step 7 cases 0, 4, 6: 241-341 times) find the extremum at t_end (excess 7.5e-12) and one
  interval. Rerun: crown_check all modes (one thread, -j 4): ALL OK, FORMAT OK; every mutant still fails.
- Content-check emulation 2 (2026-10-12, last version; edit_label unchanged). Findings: (1) step 4: the reference
  solved the normal equations (condition number squared) and missed its own limits near the resolvability threshold
  (80-digit check: g off by 0.6 K at a scaled singular value s_min of 3e-8, 1.9e-3 K at 1e-6); the near-threshold
  case checked only isfinite; absolute limits on g and q0 do not follow a rescaling of P or R. (2) Step 1 case 20
  (dogleg 180 deg - 2e-6 rad) checked bounds only; an arccos dogleg errs by 1.8e-4 m there. (3) The output asserts
  of steps 1, 2, 3, 5 did not check the stated float dtype. Change: fit_heat_flow by the thin SVD of the
  column-scaled design (reference, solution.py, mutant copies); accuracy restated for terms up to 1e3 K:
  rms_residual within 1e-9 K for every resolvable design; for s_min >= 1e-3 every fitted term within 1e-9 K (T0, q0
  max|R|, g max|P|) and the standard errors within 1e-6 (rms >= 1e-6 K); step 4 case 5 rebuilt (s_min 3.0e-8 with
  terms of 885 and 937 K: rms; 3.0e-9 rejected), new case 7 (s_min 1.05e-3, two coefficient sets), P in units of
  1e-4 in case 0; step 1 case 20 against the closed form; dtype in the existing asserts. Regression: targets of
  cases 5 and 7 within 6e-4 of each limit of the 80-digit solution (rms 4e-15 K), Frisch-Waugh standard errors
  within 1.1e-12; reference worst error/limit 8.8e-3 (second solution 1.5e-3); normal equations fail cases 5 (rms
  3.4e-7 K) and 7; arccos fails case 20, reference and second solution within 2.0e-9 m. Rerun: crown_check all modes
  (one thread, -j 4): ALL OK, FORMAT OK.
- Content-check emulation 3 (2026-10-12, last version; edit_label unchanged). Findings: (1) stale figures (scan
  counts, exact-input maxima, budget) from before the domain-end regions were rebuilt; (2) the step 6 rule "not
  separately resolvable at t_end" could not be isolated (an unresolvable design makes RSS(t) constant, so the
  end-point rule fires first); (3) step 7 states g on the whole real line, but every region had g > 0; (4) the step 7
  reference rejects a hole whose [1, R] is unresolvable, a condition no longer stated; (5) problem_description_main
  did not say which parameters carry standard errors; (6) the re-measured budget: a P error irregular in t at the
  full step 5 accuracy exceeds the step 6 g and t_end limits in cases 1, 2 and 5 (an inherent displacement of the
  minimizer, not a reference defect), against the step 6 propagation sentence. Cause: texts not updated after the
  regions were rebuilt. Change: (1) figures restated (above); (2) the clause dropped
  (prompt, header, scaffold), the step 6 case 6 input (unchanged) relabelled as the end-point rule; (3) general case
  1 adds a call with negated shape values on the same data: g -> -g, every other output unchanged (twice the
  single-output tolerances); (4) step 7 domain (prompt, header, problem_io, scaffold, contract): the used readings of
  a hole lie at distinct depths, so the raise is an out-of-domain guard; (5) "standard errors of the regression
  parameters"; (6) the step 6 propagation sentence (prompt, header, scaffold, contract) limited to step 5 errors
  smooth in t_end, tolerances unchanged (step 6 cases use the reference steps 1-5). Regression: general case 1
  passes with the reference, solution.py and the second solution; on the relabelled input the reference raises the
  end-point error (the second solution its resolvability guard). Rerun: crown_check in all modes (one thread, -j 4):
  ALL OK, FORMAT OK; every mutant still fails.
