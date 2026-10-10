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

## Validation before every zip

1. Reference: each case of tests/step_k.py run alone in a fresh process with solution/step_1..k; each case
   of tests/general.py alone with solution.py.
2. second_solution.py: every case passes.
3. Mutants: each fails at least one case of its target tests.
4. Shift check: perturbing every reference output by a relative ±1e-12 still passes every case.
5. All-None / all-zero controls fail.

`tools/crown_check.py` (repo root) runs 1-5 for one task folder.
