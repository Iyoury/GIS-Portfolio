# Project Crown task authoring (AfterQuery) — working memory

The author (Ayoub) writes in informal French; always answer in French. Tasks live in `tasks/<folder>/`.
The authoritative standard is the authoring guide, saved in full here:

@docs/CROWN_AUTHORING_GUIDE.md

## How the author works with me

- He pastes pipeline reports (content check, rollouts, probes) and says "regle sa", "fixe sa", "le rendre
  plus difficile", or "next sujet" (new task).
- For each request: fix or build the task, validate it (below), zip it, commit and push to the session's
  designated branch, send the zip with SendUserFile, and reply in French with what changed and why.
- Zip from the repo root so paths start with `tasks/<folder>/...` (otherwise the platform says "No tasks
  found"): `zip -qr <scratchpad>/<folder>.zip tasks/<folder>` after deleting `__pycache__`.
- Never create a pull request. Commit messages end with the attribution lines the session asks for.
- A task whose rollouts are queued or in flight ("Paused / Measuring", "pending") is waiting for platform
  capacity: do not change or re-upload it; re-uploading restarts the queue.

## Non-negotiable rules (from the guide plus platform findings)

- Everything graded must be visible in problem_description_main, required_dependencies,
  step_description_prompt, function_header or return_line; every stated behaviour is tested, every
  tested behaviour is stated (boundaries included, e.g. inclusive endpoints, n = minimum, empty and 2-D
  inputs, float dtype when the prompt says "float array").
- Test files use the parser format `# --- test case N ---`; every case is self-contained (its own imports,
  helpers and setup; no variables or random state from other cases). n_test_cases equals the number of
  cases. tests/general.py calls the final function against solution.py.
- Every test-case body is wrapped in `def _crown_case(): ...` followed by `_crown_case()` (tools/wrapcases.py),
  so no test variable or helper leaks into the namespace shared with the candidate's code (a module-level
  `pi = ...` or a helper class `_Hole` in a test once overwrote the candidate's own names).
- No timing asserts and no clock reads in tests (no perf_counter, time.time, signal.alarm); do not state a
  per-call time budget in prompts. Runtime must simply be reasonable.
- No exact floating-point equality and no zero tolerance; near-zero targets get an absolute tolerance.
  Comparisons between two computed outputs use about twice the single-output tolerance.
- Tolerances are scientifically justified and never loosened (or tightened) to tune pass rates; change
  difficulty only through legitimate scientific reasoning and integration.
- Computed scientific values must dominate each step's tests over validation/type/shape checks.
- No test-describing language in prompts; do not expose hidden test cases.
- Targets come from independent evidence (analytic identities, special cases, independent high-precision
  recomputation with a different method, literature) — never from the reference or the candidate.
- Every solution import is declared in required_dependencies (e.g. `from scipy.integrate import solve_ivp`).
- Per-step solution files do not redefine earlier step functions; no substantial top-level computation.
- At least one named scientific mutant per step (file in mutants/) and one whole-task mutant; every
  mutant must fail at least one case of its target tests.
- second_solution.py uses a genuinely different method and passes every test.
- Each step has a contract: quantity, assumptions, result_kind (exact/approximate/asymptotic),
  valid_input_ranges, units, output_shape, step_dependencies.
- Metadata: subfield (<= 120 chars), tags, expert_time_estimate_hours > 0, relevant_experience,
  difficulty_explanation, solution_explanation, verification_explanation, author fields. Affiliation is the
  real one or exactly "Independent Researcher". Never invent credentials, links or citations.
- Every revision gets a source.md note: finding, cause, change, regression test, what was rerun; and the
  right edit_label (repair, clarification, grading_fix, robustness, difficulty_increase).
- Difficulty band target: about 3-6 of 8 rollouts.

## Platform "too easy" feedback (since 2026-10-12)

A too-easy report now explains, step by step, why the step gave way and suggests directions. Labels seen:
"Key formula given" (the prompt supplies the central formula: direct substitution), "Prompt reads as a recipe"
(every equation, index order, constant and update rule is spelled out). Suggested fixes, in line with the guide:
- remove the central formula/recipe but keep the definition, conventions and accuracy (the agent derives it);
- state the full numerical range and add demanding but explicitly bounded cases (asymptotic regimes,
  underflow vs true zero, ill-conditioned or rescaled inputs) checked against high-precision references;
- "converge on the output": grade energies/residuals/stationarity of the result, not a prescribed algorithm.
The team says the goal is deeper scientific reasoning, not extra steps or tighter tolerances; the suggestions are
starting points, use judgement and flag what does not fit. Update prompt, reference and tests together.

## Upload checks seen (2026-10-12)

- Accepted domains: physics, biology, materials, math, chemistry (earth refused: geophysics tasks use physics).
- Size limit 800 KB as the platform measures it (about 1.12 x the raw folder size): keep raw <= 650 KB.
- Metadata must be a `metadata:` block with author_name, author_email, author_organization, author_profile,
  conflicts_of_interest, relevant_experience, subfield, tags, expert_time_estimate_hours, difficulty_explanation,
  solution_explanation, verification_explanation (tools/metadata_block.py).
- Note "n_test_cases_mismatch" appeared on every step while markers matched n_test_cases; the files had a
  comment preamble before "# --- test case 0 ---" -> no text at all before the first marker (crown_check flags it).
- Content check (2026-10-12, blocking "Edge-case inputs"): no NaN/inf inputs in tests unless the science needs
  them, even when the prompt says "raise ValueError if not finite"; drop such tests AND the prompt clauses.
  "Science outweighs validation" counts assertion sites per step (stated raises at half weight): computed values
  must clearly dominate; add computed cases with independent targets rather than more validation.
- A failed content check returns the task as "Needs revision" (a send-back): on a task's LAST allowed version
  that closes it, so emulate the content check (all categories, per-step counts) before uploading a last version.
- Warning "tasks/_id_mapping.json is absent; falling back to delivered ids" is harmless (the version still maps).
- Pipeline after the content check (seen 2026-10-12): an exploit probe (an agent told to pass the tests without
  solving; any credit -> judge) then a blocking difficulty probe: a quick model solves the task end to end twice;
  2/2 = "too easy", sent back before any rollout ("Difficulty misses in a row: 1 of 3"; the miss that reaches the
  limit closes the task). Each report has a collapsed "Why each step gave way" feedback: ask the author to paste it.
  activation v14 and electron v3 failed it (2/2) after passing the content check.
- AfterQuery warning (2026-10-12): too many submissions fail validation and review; each run is costly. Upload only
  versions that are complete, content-check clean (emulated) and hard enough (blind quick-solver probe:
  tools/visible_spec.py + a fast model writing cand.py from the visible text only + tools/grade_candidate.py).
- Probe calibration (2026-10-12): on the two tasks the platform probe called too easy, Sonnet at medium effort and
  Opus at low effort (blind, ~3 min each) scored electron v3 38/38 and 38/38, activation v14 34/36 and 35/36 (only
  validation edges failed); Haiku-low and Sonnet-low scored 27-30/38 on electron. So the platform's quick model is
  about Sonnet-medium / Opus-low: use those two as the quick-probe proxy, and require that they fail for scientific
  reasons before uploading a hardened version.

- UPLOADS ON HOLD (2026-10-13): an admin paused new uploads and revisions from the author's account (tasks already
  uploaded keep moving), after a second AfterQuery warning about submissions failing validation and review. WF v11 and
  stoner v13 were both returned by an administrator at 13 Oct 04:31 (batch, no reason shown; both submitted 5 Oct with
  time budgets and finiteness clauses in their prompts). While on hold: no upload advice; prepare and audit locally only;
  when lifted, submit one fully audited task at a time and wait for its result.
- HOLD LIFTED (2026-10-15): Harry (AfterQuery team) unpaused the account. Upload only what passed the RELEASE GATE,
  one task at a time, each only after the previous one's pipeline result is read (zips8 WF/stoner withdrawn).

## Official skeleton and README (2026-10-18, docs/skeleton_2026-10-18/, from the platform's Instructions/Walkthrough)

- Domains now listed: physics, biology, materials, math, chemistry, EARTH, OCEAN (earth was refused on 2026-10-12;
  existing geo tasks stay in physics, new geoscience tasks may use earth).
- Environment: Python 3.12 stdlib + numpy 2.1.3, scipy 1.14.1, sympy 1.13.3, h5py 3.12.1, mpmath 1.3.0; offline.
- Limits: at most 50 tasks and 8 MB per zip (keep our 650 KB raw habit anyway).
- A step prompt must NAME every earlier function the step calls (the content check sends back a step that uses
  one its prompt never mentions); per-step solution files do not repeat earlier functions.
- Mutant names: letters, digits, _ and - only; each needs name, description, file in the right list.
- Placeholder affiliations ("none", "n/a", "independent", "self-employed") are refused; exactly "Independent
  Researcher" is accepted. author_profile: Scholar/LinkedIn/personal site.
- "Do not upload platform logs or review records as author submission files": keep source.md revision notes to
  finding/cause/change/regression/rerun without pipeline quotes (rewards, admin returns, probe counts).
- Tests: bare blocks after "# --- test case N ---", expected values inline, no preamble before case 0 (our
  _crown_case wrapping is compatible). "Test invalid inputs only when the prompt specifies how they are handled."
- Difficulty must come from scientific reasoning or implementation choices, not hidden assumptions, arbitrary
  precision demands or runtime. Audit 2026-10-18: all 9 repo tasks pass the mechanical rules (prompts name the
  earlier functions they call, dependencies, mutant names, metadata).

## RELEASE GATE (author's rule, 2026-10-15): never send a zip that has not passed all of these

Borehole v10 (the LAST version) came back "too easy" after only uncalibrated blind solvers had been run on it, and
prism v4 (local v7) failed the content check after a self-run emulation. From now on a zip is sent only when:
1. crown_check ALL OK (all modes) on the final folder, size <= 650 KB raw.
2. An INDEPENDENT content-check emulation (agent that did not write the change): every category, per-step counts,
   stated <-> tested both ways, no NaN/inf, no zero-relative tolerance, no hidden convention, no recipe, no test
   inputs in prompts; no blocking or should-fix finding left.
3. A CALIBRATED difficulty probe on the FINAL visible text (tools/visible_spec.py), graded end to end with
   tools/grade_candidate.py on the real tests. Pass only if NO probe passes every case, and the failures are fair
   scientific/numerical mistakes (judge reads the code). WARNING (2026-10-16): the weak proxy (2 x Sonnet medium +
   1 x Opus low, single pass) let WF v12 through while the platform solved it 2/2; use STRONG agentic proxies
   (2 x Opus high + 1 x Sonnet high, unlimited self-checking): recalibrated 2026-10-16 on WF v12.
4. An independent fairness audit of the hardest step (a valid alternative passes; targets recomputed by another route).
5. Every finding fixed and the gate re-run on the changed parts. Otherwise: no zip, and tell the author why.

## Validation before every zip

1. Reference: each case of tests/step_k.py run alone in a fresh process with solution/step_1..k; each case
   of tests/general.py alone with solution.py.
2. second_solution.py: every case passes.
3. Mutants: each fails at least one case of its target tests.
4. Shift check: perturbing every reference output by a relative ±1e-12 still passes every case.
5. All-None / all-zero controls fail.

`tools/crown_check.py` (repo root) runs 1-5 for one task folder.

## Task status (keep updated)

- borehole-heat-flow: v9 rollouts "Too easy 8/8"; versions used 9 of 10 -> v10 is the LAST version (a send-back
  closes the task); difficulty misses in a row 1 of 2. v10 = new integrated final step (joint inversion of q0, T0,
  g and the deglaciation time t_end with the layered paleoclimate perturbation, Wald errors and profile-likelihood
  95% interval). Design brief kept in the session scratchpad (bh10/DESIGN.md).
- Blind-solver calibration (2026-10-11): my spec-only blind solvers (Claude subagents) passed stoner v10 2/2 and
  wright-fisher v10 2/2, while the platform rollouts gave 1/8 and 7/8. The platform model is clearly weaker than
  the blind-solver proxy: a 100 % blind score does NOT mean "too easy" on the platform; use blind solvers to find
  spec gaps and unfair tests, not to predict the band.
- borehole-heat-flow v10 final design: step 6 single-hole joint inversion (q0, T0, g, t_end; Wald + profile
  interval) and step 7 regional ML over several holes with AR(1) noise (exact likelihood, expected Fisher
  information, likelihood-ratio interval). Blind solvers 7/7 passed (proxy optimistic, see above).
- wright_fisher_relaxation platform v11 = local Version 11 (commit 0054371, 5 Oct): in band 1/8, difficulty probe 0/3, 3 of 8 attempts
  invalid ("possibly timed out", agents ran out of session time on step 7), notes discoverability and verifier
  correctness 1/6 each; returned by an administrator 2026-10-13, reason not pasted (asked the author). v11 still had
  the per-call 20 s budget and the non-finite clauses; local v15 (zips6, identical to the repo) removes both.
  Reference cases take 0.6-7.4 s each on one core (92 s in all), so the timeouts are not the verifier's.
- stoner_wohlfarth_ensemble platform v13 = commit 000951e (local Version 12): in band 1/8, returned by an administrator
  13 Oct 04:31 (same minute as WF v11). Pre-upload audit (2026-10-14): unstated meaning of psi = np.pi/2 (a literal
  float reading failed step 3 case 4 by up to 0.95) -> stated in the main text, step 3 prompt, docstrings, contract;
  source.md records the v13 report. Ready as platform v14 (zips8), not uploaded (account on hold).
- wright_fisher_relaxation pre-upload audit (2026-10-14): step 7 case 8 (lambda_min 1.36e-225, low end of the stated
  range) and step 3 case 8 (1e-250 absolute rule) added with independent mp targets; source.md Version 16 records the
  v11 report. Ready as platform v12 (zips8), not uploaded (account on hold).
- Dashboard 2026-10-15: ACCEPTED two-center-rhf v12 (sto3g), ising-strip-transfer v8, josephson-phase-diffusion v10.
  Needs revision: borehole v10 too easy (last version), prism v4 content check FAIL (local v7, report not yet pasted),
  activation v14 and electron v3 too easy, stoner v13 and WF v11 returned by admin, layered-surface-wave v2 content
  fail, hodgkin-huxley v1 reference fail, hydrogen-stark-resonances v1 too easy; older geo tasks (basin-backstripping,
  gravity-basin-stripping, earthquake-swarm-location, gnss-euler-pole, zircon-discordia-ages, paleopole-synfolding)
  not in the repo.
- Release gate run 2026-10-15: ELECTRON v4 PASS (probes 39/43, 40/43, 40/43 on the final tests; background
  occupation-number typo fixed) -> zips9, first to upload. WF v12 FAIL on content only (probes 66, 68, 67 of 69, fair):
  every two-sided error rule tested on one side since the Version 15 trim -> fixed (Version 17), independent
  re-verification GATE PASS -> zips9, second to upload (after electron's result). STONER v14 FAIL: a Sonnet-medium probe
  solved 46/46; content fixes committed (both sides of rules, step 3 case 7, step 6 h_half exclusion); removing the
  step 5 key formulas (scratch copy gate/stoner_wohlfarth_ensemble/harden) did not help (47/47, 45, 45 of 47): the
  models know Brown/Garanin. Not releasable; author decides (park, or new science).
- WF v12 (uploaded 15 Oct 19:17): content check NO FINDINGS, exploit clean, but DIFFICULTY PROBE 2/2 TOO EASY
  (feedback: step 1 recipe, step 6 key formula, steps 2-5 textbook, step 7 solved every time; suggested harder
  settings saved in scratchpad wf13/platform_feedback_v12.md). The weak proxy (Sonnet-medium/Opus-low single pass)
  scored 66-68/69 and missed it: the platform quick model is STRONGER than that proxy. v11's 0/3 probe came partly
  from the 20 s budget and timing asserts. Recalibration with strong agentic proxies (Opus high x2, Sonnet high,
  unlimited self-checking) DONE 2026-10-16: on WF v12 they scored 69, 70, 70 of 70 (2 of 3 full solves = matches the
  platform's 2/2); on electron v4 43/43 x3 -> electron v4 WOULD BE TOO EASY: do NOT upload it (needs new science,
  author's call). Gate item 3 is now 2 x Opus-high + 1 x Sonnet-high, blind, unlimited self-checking; pass only at
  0 of 3 full solves with fair failures. WF v13 design (scratchpad wf13/DESIGN_FINAL.md): new final step 8
  relaxation_spectrum (all N+1 eigenvalues and rates to 1e-8, totally positive structure; dense eig gets 22-32 of 401
  right at N = 400; certified arb targets A/B/C), step 1 in words, step 6 as a Yaglom limit. PRE-BUILD STRONG PROBE:
  STOP - 2 of 3 strong proxies solved v13 end to end (all three wrote Koev's totally-nonnegative eigenvalue algorithm,
  401/401 eigenvalues; the third missed only the old step 7 N = 1 corner). Not built. Lesson: strong models know the
  specialised algorithms (GTH, Koev TN, Brown/Garanin, Sommerfeld); removing formulas or adding a known advanced
  algorithm does not create difficulty. Always run the strong probe on a DESIGN (spec + prototype harness) before
  building anything.
- New topic search 2026-10-18/20: lithosphere_strength_inversion (earth; geotherm + Byerlee + creep envelope +
  (Curie, LAB) inversion; design/proto/harness in scratchpad newtopic/geotherm) -> STRONG PROBE: Opus-high solved
  16/16 in ~10 min -> NOT built (too easy). Prism v8 (content-check v4 fixes, commit 8ed027b) strong probe running.
- All tasks: authoring-guide conformance passes 1-2 (2026-10-10/11). Author metadata filled 2026-10-12 with the
  author's own text (tools/set_author_meta.py): affiliation "Independent Researcher"; relevant_experience = B.Sc.
  Georesources and Environment (FST Fes, 2016), graduate certificate in geomatics (Universite Laval), GIS analyst
  at Morocco's Ministry of Energy and Mines (2016-2018), independent geospatial consultancy (Craton Geodonnees,
  Quebec, Abitibi drilling databases). Reuse it for new tasks. author_name Ayoub Bousfiha, author_email iyoury.ab@gmail.com, author_profile
  https://www.linkedin.com/in/ayoub-bousfiha-674a75219, conflicts_of_interest None (author's file, 2026-10-12);
  research_advisor and referred_by optional and left out.
- layered-surface-wave and basin-backstripping: not in the repo (zips never received).
- 2026-10-12 later: activation v14 and electron v3 "too easy" by the difficulty probe (2/2, miss 1 of 3); prism v3
  content check failed (4 untested step 4 error rules). Redesign/fix workflows run with blind quick-solver probes;
  design briefs in the scratchpad probe/<task>/DESIGN.md.
- Pre-probes 2026-10-13 (calibrated Sonnet-med x2 + Opus-low, harness with independent mp targets):
  activation redesign v1 (de-recipe, in-growth step 4) scored 14/14 by both Sonnet-med -> not built; activation plan B
  (Hogdahl thermal + epithermal spectrum, monitor_decays + 3-parameter WLS spectrum_parameters) 12/12 by Sonnet-med a and
  Opus-low -> STOP, author decides (park, or new science such as Westcott g(T_n) / Cd-covered monitors, re-gated).
  Electron v4 design with two hint clauses removed (step 2 entropy per electron, step 4 (kT/E_F)^2) -> 12, 11, 12 of 14,
  failing only nabla_ad at the three cold degenerate corners (p-space quadrature floor ~1e-8, edge-centred ~1e-12) ->
  PROCEED; approved visible text = scratchpad probe/electron_positron_eos/next2/spec.md. Electron v4 built (source.md
  Version 6 + second pass), independently verified (targets recomputed by a separate mp adiabat to <= 3.5e-17), method
  hints also removed from the backgrounds, crown_check ALL OK, 343704 bytes; zip in scratchpad zips7, sent, NOT uploaded
  (account on hold). Only visible-text change after the probe: step 1 "by 20 decades or more".
- prism v7 (2026-10-13): content-check v3 fixes + second pass, crown_check ALL OK, verifier READY; calibrated
  probe 0 of 3 full solves (34, 30, 34 of 36; step 2 in-plane/edge points and step 1 rods fail) -> zip sent (zips5).
- Content-check sweep 2 (2026-10-12): all 9 tasks without non-finite literals or finiteness promises, per-step
  validation trimmed, computed cases added; independent verifiers PASS. Zips sent: electron, prism, activation
  (zips3); josephson, sto3g, stoner, ising, wright-fisher and borehole v10 final (zips4; borehole crown_check ALL OK,
  614371 bytes, step 6 propagation sentence limited to step 5 errors smooth in t_end). Upload advice given: stoner, WF, activation only after their
  in-flight rollouts return; josephson, ising, sto3g only if sent back; borehole once ready (last version, Physics).
