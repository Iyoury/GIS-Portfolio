# Sources, method and ground truth

## Physics
Ideal relativistic Fermi gas of electrons and positrons with pairs in equilibrium with radiation
(Chandrasekhar 1939; Cox and Giuli 1968; Timmes and Arnett 1999). Definitions follow the textbook
integrals over momentum. The constants are CODATA 2018 (m_e c^2, hbar/(m_e c)) and the exact SI-2019
values (k, N_A).

## Reference solution
- **Quadrature.** Composite 24-point Gauss-Legendre.
  - Below the Fermi edge, in t = sqrt(E) (E the kinetic energy in m_e c^2), with doubling panels.
  - Over the edge E_F +- 60 kT, in y = (E - E_F)/kT with panels of width 1. The electron occupation
    argument is then y itself, not a difference of two large energies; this keeps the entropy accurate
    to 1e-15 even at psi = 1e6.
- **Net electron density.** Computed from f(b - psi) - f(b + psi) = sinh(psi)/(cosh b + cosh psi). The
  integrand is positive and summed in logarithms, so there is no cancellation and no underflow.
- **Entropy.** Uses the positive integrand -[f ln f + (1-f) ln(1-f)] = log1p(e^{-|x|}) + |x|/(e^{|x|}+1).
- **psi from rho Y_e.** Bisection-safeguarded false position with the Illinois modification in ln psi on
  ln n_net (version 3; before, plain false position, see below). The variable psi includes the rest
  energy, so a pair plasma with psi = 2e-21 is represented exactly; mu/kT would round it away.
- **Specific heat.** Computed as c_V = (1/kT^2) det / <dN^2>. The determinant
  det = <dE^2><dN^2> - <dE dN>^2 = 1/2 sum_ij w_i w_j (e_i n_j - e_j n_i)^2 is evaluated term by term:
  - same-species pairs give two-pass centred second moments;
  - electron-positron pairs give (e_i + e_j)^2 > 0.

  There is no cancellation. The Schur form loses (kT/E_F)^2, which is 2.8e-6 relative at
  rho_Ye = 1e13, T = 1e7 K.

## Second solution
- Adaptive QUADPACK (scipy.integrate.quad) in E, split at the edge window, instead of fixed panels.
- Brent's method for psi.
- Centred moments for c_V, by the same adaptive quadrature.
- Agrees with the reference to 1e-11 or better (version 2 rerun: worst 9.8e-12, n_minus at T = 2e7 K,
  psi = 1e6, where the reference is within 7e-16 of the 40-digit target; all others below 5e-12).

## Targets (all independent of the solution code)
- **Classical gas.** The Maxwell-Juettner closed forms with modified Bessel functions, in mpmath:
  - n = theta K2(1/theta) e^{+-psi}/(pi^2 lambda^3);
  - P = n_tot k T;
  - <eps> = K1/K2 + 3 theta;
  - s per particle = k(<eps>/theta -+ psi + 1);
  - psi from n in closed form.

  These are used only where the Fermi corrections are below 1e-11. Positron densities of a classical
  positron gas also come from the Maxwell-Juettner formula.
- **Twelve (T, psi) states, from psi = 2.2e-21 to 1e6 and from 1e7 to 1e11 K.** mpmath tanh-sinh
  quadrature at 40 digits of the defining integrals (n_minus, n_plus, n_minus - n_plus computed in
  40-digit arithmetic, P, u, s), with breakpoints every kT over the Fermi edge.
- **Twelve (rho_Ye, T) states** (ten in versions 1-2). psi from an mpmath root search on those
  integrals. c_V from a centred difference of u at constant n_net (T +- 1e-7 T, psi re-solved at each
  temperature), all at 40 digits. This is a different route from the fluctuation formula of the reference.
- **Four (rho, T, Ye) states for the complete equation of state.** Same methods.
- **Two corner states of the complete equation of state** (version 3): (rho, T, Ye) = (2e13, 1e7, 0.5)
  and (2e-10, 1e11, 0.5); psi and c_V are the stored (rho_Ye, T) targets, P, u, s and the densities come
  from 40-digit mpmath quadrature at that psi (see Version 3).

The reference agrees with all targets to 1.4e-13 or better (remeasured in version 3): up to about 1e-13
for c_V and for the densities of the cool classical gas at T = 1e7 K with tiny psi (n_minus about
4.5e-232 cm^-3, n_net 9.0e-250 cm^-3), up to 5e-14 for the other low-temperature classical states (values
down to 1e-109), and below 2e-14 for the rest.

## Why the naive formulas fail the stated accuracy
- n_minus - n_plus: 100 % error in the pair plasma (rho_Ye = 1e-10, T = 1e11 K).
- (u_tot + P - mu n)/T for the entropy: 1.6e-5 at psi = 1.3e5, T = 1e7 K (required 1e-10).
- The Schur form of c_V: 2.8e-6 at rho_Ye = 1e13, T = 1e7 K, and 4.5e-8 at 1e11 (required 1e-8).

## Version 2 (grading_fix: authoring-guide conformance)

No science, reference algorithm, stored target or input domain was changed. Each item gives the finding,
its cause, the change, the regression test and what was rerun.

- Time budgets in the tests and in the visible text.
  - Finding: every test file wrapped the evaluated functions in a `_t_budget` decorator that armed a
    SIGALRM interval timer and asserted on `time.perf_counter()` (10 s per call for steps 1-4, 30 s for
    step 5), and every step prompt, function_header docstring, problem_io, scaffold, per-step solution,
    solution.py and mutant docstring stated "Each call (must finish) within 10 s / 30 s (on one CPU core)".
  - Cause: per-call budgets were part of the version 1 design.
  - Change: the wrapper, the `signal` and `time` imports and the rebinding lines were removed from all six
    test files, and every time sentence was deleted (no sentence carried a problem size). The reference
    needs at most about 0.015 s per electron_positron_eos call (measured for the four step 5 states), so
    runtime stays reasonable without a budget.
  - Regression test: crown_check format (no clock or signal pattern left) and every case of every mode.
- Test-case format and self-containment.
  - Finding: the markers carried descriptions ("# --- test case 0: classical gas ... ---"), and all
    imports, constants, the Maxwell-Juettner oracle, the target tables and the check helpers lived in a
    preamble before case 0, so no case could run alone.
  - Cause: the files were written to be executed as a whole.
  - Change: markers are exactly "# --- test case N ---" (N = 0..n-1) with the description as comment
    lines below; every case repeats the imports, constants, oracle, target table and helper it uses
    (split with tools/selfcontain.py); the target-provenance note stays as a leading comment. Test-level
    globals were renamed with the `_t_` prefix (general case 0 used `out`, `k`, `want`, `tol`; case 1
    used `psi`, `P`, `u`, `s`, `lhs`, `rhs`; step 3 case 0 used `th`, `base`) so that a case cannot
    overwrite a module-level name of the candidate. Two comments were corrected: step 1 case 0 said
    "Mawell-Juettner", and step 1 case 2 claimed to test T = 1e7 K although it uses only T = 1e11 K.
  - Regression test: every case run alone in a fresh process (crown_check ref, second, mut, shift,
    controls).
- General case 1: bound inconsistent with the stated accuracies.
  - Finding: n_net at the psi of degeneracy_parameter (rho Ye = 5e5 g cm^-3, T = 1e9 K) was required to
    match rho Ye N_A within 3e-10, justified by "d ln n / d ln psi < 3". Measured with the reference,
    d ln n_net / d ln psi = 4.84 there (psi = 6.74, partially degenerate), so a psi with its allowed
    relative error 1e-10 plus an n_net with its allowed 1e-10 can be off by 5.8e-10 and fail.
  - Cause: the sensitivity was estimated, not computed.
  - Change: bound 6e-10 (the propagated error budget, rounded up) and the comment now states the
    computed sensitivity. The Euler-relation part keeps its 1e-8; its comment now gives the computed
    budget (3.4e-10: the largest terms n_net m_e c^2 and T s are 1.6 and 0.7 times the result) instead
    of "each side to 1e-10". Reference and second solution meet the identity to below 1e-14.
  - Regression test: general case 1 with solution.py and second_solution.py.
- Stated but untested endpoint.
  - Finding: the step 5 domain includes Ye = 1 (0 < Ye <= 1), but no test used it.
  - Change: step 5 case 2 evaluates (rho, T, Ye) = (1e6, 1e9, 1.0), which has rho Ye = 1e6 g cm^-3 exactly
    like the stored state (2e6, 1e9, 0.5), against the same 40-digit targets (the equation of state depends
    on rho and Ye only through rho Ye, as the prompt defines). n_test_cases of step 5 is now 3.
  - Regression test: step 5 case 2 passes with the reference, solution.py and the second solution and
    fails the cv_at_fixed_psi mutant and both controls.
- Visible specification.
  - Finding: the step 1 tests require a tuple of Python floats, while the step 1 prompt said only "return
    n_minus, n_plus and n_net"; the step 4 contract lists step 1 as a dependency, but the step 4 prompt
    did not name pair_densities.
  - Change: the step 1 prompt now says "return the tuple (n_minus, n_plus, n_net) of Python floats"; the
    step 4 prompt says "the net electron density n_net of pair_densities (step 1)". No new requirement.
  - Regression test: crown_check ref and second on steps 1 and 4.
- Undeclared imports.
  - Finding: second_solution.py imports warnings, scipy.integrate.quad and IntegrationWarning, and
    scipy.optimize.brentq, which were not in required_dependencies.
  - Change: declared in required_dependencies; the scaffolds start with the same import lines.
  - Regression test: crown_check second (all cases).
- Metadata: subfield, tags, expert_time_estimate_hours, relevant_experience (placeholder for the author),
  author, affiliation, difficulty_explanation, solution_explanation, verification_explanation and
  edit_label grading_fix added. The second-solution agreement stated above was corrected from "about
  1e-13" to the measured 1e-11 (see Second solution).
- Floating-point comparisons: no exact float equality and no zero tolerance were present (all checks are
  relative errors with stated bounds or the stated absolute 1e-250 for tiny n_plus); nothing changed.
- Independent re-check of stored targets: n_minus, n_net, P, u and s at (T, psi) = (1e7 K, 2.8e4),
  (1e9 K, 7.8) and (1e11 K, 2.2e-21) were recomputed with 30-digit mpmath Gauss-Legendre quadrature in
  t = sqrt(E) (panels of width k T in E over and above the Fermi edge, geometric panels near E = 0), a
  route different from the tanh-sinh generation; all fifteen values agree with the stored targets to
  7e-17 or better.
- Rerun after all changes: tools/crown_check.py (all modes, every case alone in a fresh process, -j 2):
  FORMAT OK; REF steps 1-5 4/4, 3/3, 3/3, 2/2, 3/3 and general 2/2 (solution.py also passes every step
  file); SECOND steps 1-5 and general all pass; mutants fail net_by_subtraction 2/4 (cases 1, 2),
  entropy_from_euler 1/3 (case 1), positrons_ignored 1/3 (case 1), schur_complement 1/2 (case 0),
  positron_fluctuations_ignored 1/2 (case 0), cv_at_fixed_psi 2/3 (cases 0, 2),
  whole_task_classical_occupation 1/2 general (case 0); SHIFT +-1e-12 all pass; all-None and all-zero
  controls pass 0 cases. Repeat runs: solution.py and second_solution.py each run twice in clean
  processes (and once more with one BLAS/OpenMP thread) on 21 calls covering all five functions give
  bitwise identical outputs; the methods use no random numbers.

## Version 3 (grading_fix: authoring-guide conformance)

No stored target, input domain or physical model was changed. Three findings of an independent verifier
were confirmed and fixed (items 1-3). One reference bug found while checking them was repaired (item 4).
Each item gives the finding, its cause, the change, the regression test and what was rerun.

- 1. Error propagated from psi into the complete equation of state.
  - Finding: the step 5 text gave each output "the accuracy stated in its step". Step 5 cases 0 and 2
    and general case 0 compare n_minus, n_plus, P, u and s with their exact values at the exact psi,
    within 1e-10. These quantities change faster than psi. Computed d ln X / d ln psi:
    - at (rho Ye, T) = (1e6, 1e9): n_minus 4.6, n_plus -7.8, P 5.66, u 5.78, s 2.92;
    - at (4.3e7, 3e8): n_plus -72.6;
    - over the domain: up to 587 for n_minus, P and u, 548 for s, 598 for n_plus and 586 for c_V. In a
      cool classical gas the sensitivity is about psi, and psi is close to m_e c^2 / k T there.

    Reproduced: a whole solution whose degeneracy_parameter returns psi (1 +- 9e-11) passes
    tests/step_3.py 3/3 but fails general case 0 and step 5 cases 0 and 2 (n_minus 6.022142991659798e29
    against 6.022142989167459e29, relative error 4.1e-10).
  - Cause: the per-step accuracies are defined at a given psi. The integrated tests start from
    (rho, T, Ye), and the propagation through psi was never stated.
  - Change (no test tolerance changed): the step 5 step_description_prompt, function_header and
    problem_io now state the accuracies with respect to the exact values for the given (rho, T, Ye). They
    add that n_minus, n_plus, P, u and s amplify a relative error of psi up to about 600 times, so psi
    must be found to about 1e-13 relative, far better than the 1e-10 of step 3. The same docstring is in
    steps/step_5.py, solution/step_5.py, solution.py and the two mutants that define
    electron_positron_eos. The step 4 prompt now says that its 1e-8 is with respect to the exact c_V at
    the given rho_Ye and T, and that an error in psi enters c_V amplified up to about 600 times. That
    sentence has no grading effect: the step 4 context uses the reference step 3, and the general c_V
    checks are at states with d ln c_V / d ln psi below 4 (3.7 and 0).
  - Why not per-state widened tolerances: they would loosen stated tolerances. For an equation of state,
    the end-to-end accuracy at (rho, T, Ye) is the meaningful requirement, and the reference meets it
    (psi to about 1e-14).
  - Regression test: with psi scaled by 1 +- 9e-11, the whole solution fails step 5 cases 0, 2 and 3 and
    general case 0, now as specified. With psi scaled by 1 +- 1e-13 it passes every case of steps 3-5
    and general.
- 2. Inclusive endpoints of the step 5 domain were untested.
  - Finding: no step 5 or general case used T = 1e7 K, T = 1e11 K, rho Ye = 1e-10 or rho Ye = 1e13
    g cm^-3, and step 5 did not check the T > 1e11 K error path. A step 5 with exclusive bounds passed
    3/3.
  - Cause: the version 2 endpoint fix covered Ye = 1 only.
  - Change: new step 5 case 3 evaluates (rho, T, Ye) = (2e13, 1e7, 0.5) and (2e-10, 1e11, 0.5). In binary,
    rho Ye is exactly 1e13 and 1e-10.
    - psi and c_V come from the stored 40-digit (rho_Ye, T) targets.
    - P, u, s, n_minus and n_plus were computed for this version with 40-digit mpmath quadrature in
      t = sqrt(E), using tanh-sinh and Gauss-Legendre rules with breakpoints every k T over +-60 k T of
      the edge. The two rules agree to 25 digits, except for the 6e-56207 cm^-3 positron density, which
      enters only as 0. The same script reproduces the stored (T, psi) rows (1e7, 1.3e5) and
      (1e11, 2.2e-21) to all 25 printed digits.
    - Degenerate corner: n_minus = rho Ye N_A + n_plus = 6.02214076e36, since n_plus is about 6e-56207
      cm^-3; the quadrature confirms it to 5e-25.
    - Pair corner: the quadrature gives n_minus - n_plus = rho Ye N_A = 6.02214076e13 to the printed
      digits.

    (1e6, 2e11, 0.5) was added to the error list of step 5 case 1. n_test_cases of step 5 went from 3 to 4.
  - Regression test: the exclusive-bounds step 5 fails case 3 (ValueError). The reference, solution.py
    and the second solution pass, the reference within 9.4e-14 and the second solution within 4.6e-12.
- 3. n_net accuracy was promised below the range of double precision.
  - Finding: step 1 promised n_net with relative error 1e-10 for every 0 < psi <= 1e6. At T = 1e7 K and
    psi = 1e-100, the exact n_net = 2 sinh(psi) theta K2(1/theta) / (pi^2 lambda^3) is 9.0e-332 cm^-3,
    below the smallest double, and both solutions return 0.0. At psi = 1e-85 it is subnormal
    (8.998209e-317, about 7 digits). Confirmed.
  - Cause: the absolute floor given to n_plus was not given to n_net.
  - Change: the step 1 prompt and function_header now require n_plus and n_net each to have relative
    error 1e-10 when the exact value is at least 1e-250 cm^-3, and otherwise to lie within 1e-250 cm^-3.
    The same docstring is in steps/step_1.py, solution/step_1.py, solution.py and the two mutants that
    define pair_densities. The two step 1 check helpers apply the same rule; no existing target is below
    1e-250.
    - n_minus needs no floor: its smallest value in the domain is about 4.5e-232 cm^-3 (T = 1e7 K,
      psi -> 0).
    - The step 5 accuracy sentence needs no floor: step 5 does not return n_net, and rho Ye >= 1e-10
      keeps n_net >= 6.0e13 cm^-3.
  - Regression test: new step 1 case 4 checks two states at T = 1e7 K against the 40-digit
    Maxwell-Juettner closed form (Fermi corrections below 1e-257):
    - psi = 1e-18: n_net = 9.0e-250 cm^-3 with relative 1e-10 required, and n_minus = n_plus = 4.5e-232
      cm^-3;
    - psi = 1e-100: n_net within 1e-250.

    A pair_densities that sets n_net to 0 below 1e-240 fails this case. net_by_subtraction now fails
    cases 1, 2 and 4. n_test_cases of step 1 went from 4 to 5.
- 4. The reference root finder could stall (found in this pass).
  - Finding: a comparison of the complete equations of state of the two solutions on a 240-state
    (rho_Ye, T) grid showed psi differing by 5.5e-8 at rho_Ye = 1e5 g cm^-3, T = 1e7 K.
    - A 40-digit mpmath root search gives psi = 654.76623553247548. The second solution agrees to 1e-16;
      the reference returned 654.7662712217661, which makes n_net 9.1e-7 too large.
    - A scan of 8000 random (rho_Ye, T) states, half of them with T in [1e7, 1e8] K, found 39 states with
      a psi error above 1e-11, up to 1.4e-6. All of them lie in 3.8e4 <= rho_Ye <= 1.1e6 g cm^-3 and
      1.01e7 <= T <= 2.04e7 K, with psi from 350 to 660 (degenerate, non-relativistic electrons).
    - c_V inherits the error: 3.6e-7 at (1e5, 1e7) and 4.3e-6 at (4e5, 1.2e7), against the stated 1e-8.
  - Cause: the reference used plain false position, with bisection only while the bracket was wider than
    0.5 in ln psi. The slope of ln n_net against ln psi falls from about 500 on the classical side of the
    bracket to 17 at the root, so the lower end stayed fixed at ln psi = 6 (residual -195). The 300
    iterations ran out with the bracket still 0.48 wide.
  - Change: the root finder now uses the Illinois modification: the secant weight of an end kept twice in
    a row is halved. The fix is in solution/step_3.py and solution.py. It is also in the two mutants that
    copy the root finder (step_3_positrons_ignored and whole_task_classical_occupation), so each still
    differs from the reference only by its named mistake. The stopping criteria are unchanged. Over the
    same 8000 states, the psi error estimated as the n_net residual divided by d ln n_net / d ln psi is
    now at most 1.4e-14.
  - Regression test: (rho_Ye, T) = (1e5, 1e7) and (4e5, 1.2e7) were added to the 40-digit tables of step 3
    case 1 and step 4 case 0. Their targets were computed for this version:
    - psi from an mpmath root search on the n_net integral;
    - c_V from the centred difference of u_tot at constant n_net (T +- 1e-7 T, psi re-solved);
    - both with tanh-sinh and with Gauss-Legendre quadrature, which agree to 30 digits;
    - c_V cross-checked against the 40-digit fluctuation formula, which agrees within 1e-17.

    The script reproduces the stored psi and c_V at (1e6, 1e9) to 25 digits. With the old root finder,
    step 3 case 1 fails (psi 654.7662712217661 against 654.7662355324754), and so does step 4 case 0
    with the old step 3 as context (c_V 696370982242.6793 against 696370732683.6246). The fixed
    reference is within 1.6e-15 for psi and 7.1e-15 for c_V, and the second solution passes.
- 5. Other text.
  - solution_explanation: the root finder description.
  - difficulty_explanation: the 1e-250 floor and the end-to-end accuracy.
  - verification_explanation: twelve (rho_Ye, T) states, the new targets, the repeat runs and the grid
    comparison.
  - The method and target paragraphs above, and the reference-agreement statement. The earlier "below
    2e-14 for the others" did not hold for the low-temperature classical states of step 1 case 0 and
    step 2 case 0, which agree to 4.8e-14 and 2.7e-14.
- Rerun after all changes:
  - tools/crown_check.py, all modes, every case alone in a fresh process, -j 2.
    - FORMAT OK.
    - REF: steps 1-5 pass 5/5, 3/3, 3/3, 2/2 and 4/4, and general passes 2/2. solution.py also passes
      every step file.
    - SECOND: steps 1-5 and general all pass.
    - Mutants: net_by_subtraction fails 3/5 (cases 1, 2, 4), entropy_from_euler 1/3 (case 1),
      positrons_ignored 1/3 (case 1), schur_complement 1/2 (case 0), positron_fluctuations_ignored 1/2
      (case 0), cv_at_fixed_psi 3/4 (cases 0, 2, 3) and whole_task_classical_occupation 1/2 general
      (case 0).
    - SHIFT +-1e-12: all pass.
    - Controls: all-None and all-zero pass 0 cases.
  - Repeat runs: solution.py and second_solution.py were each run three times in clean processes, once
    with one BLAS/OpenMP thread, on 32 calls covering all five functions and every new state. The
    outputs were bitwise identical; the methods use no random numbers.
  - On a 240-state grid (rho_Ye = 1e-10 ... 1e13 g cm^-3 by decades, ten temperatures from 1e7 to
    1e11 K, Ye = 1), the complete equations of state of the two solutions agree to 4.6e-12.
