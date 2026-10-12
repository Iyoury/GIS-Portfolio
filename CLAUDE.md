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
- Warning "tasks/_id_mapping.json is absent; falling back to delivered ids" is harmless (the version still maps).

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
- wright_fisher_relaxation v11 and activation_chains v12: queued for rollouts on the platform when last seen.
- stoner_wohlfarth_ensemble v12 (eased: budgets, step-6 hints, tolerance 1e-5): awaiting rollouts when last seen.
- All tasks: authoring-guide conformance passes 1-2 (2026-10-10/11). Author metadata filled 2026-10-12 with the
  author's own text (tools/set_author_meta.py): affiliation "Independent Researcher"; relevant_experience = B.Sc.
  Georesources and Environment (FST Fes, 2016), graduate certificate in geomatics (Universite Laval), GIS analyst
  at Morocco's Ministry of Energy and Mines (2016-2018), independent geospatial consultancy (Craton Geodonnees,
  Quebec, Abitibi drilling databases). Reuse it for new tasks.
- layered-surface-wave and basin-backstripping: not in the repo (zips never received).
