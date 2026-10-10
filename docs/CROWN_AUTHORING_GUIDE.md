# AfterQuery Scientific Coding Task Authoring Guide (Project Crown)

Saved verbatim (formatting only) from the version the author received on 2026-10-10. This is the
authoritative standard for every task under `tasks/`.

## The standard

Create a scientific computation that a competent researcher would recognize as useful work. An agent
should have to reason about the science, implement the method, and produce results that can be
independently checked.

The task must be challenging because of that work, not because of missing information, brittle tests,
excessive runtime, or an unsupported environment.

You own the scientific design and correctness. Automated checks help find defects; passing them does not
prove that a task is scientifically sound or ready for approval. Follow the project's tool-use policy,
verify any assisted output, and never claim independent review or execution that did not happen.

Start with one polished task. Download the current task-folder skeleton from the upload panel and check
the accepted domains, supported packages, and submission limits. The skeleton's toy calculations
illustrate packaging, not acceptable scientific difficulty.

## 1. Design a real scientific problem

Choose a problem within your expertise. Write a short design brief before coding:

- Scientific goal: What quantity, prediction, inference, or comparison matters?
- Grounding: Which research method, established model, or derivation supports it?
- Nontrivial work: Which decisions require scientific judgment or numerical reasoning?
- Scope: Which inputs and regimes can you define and verify reliably?
- Verification: How will you establish the correct answer independently?

Use original problem construction, not a renamed benchmark task or a rejected task uploaded under a new
identity. Cite sources you actually checked in source.md.

Do not rely on the agent finding a paper, downloading data, or guessing a convention to complete the task.

Prefer a compact model with substantive reasoning over a large, expensive calculation. Scaling a matrix,
adding repetitive steps, or tightening tolerances without scientific justification does not make a
stronger task.

## 2. Make the specification complete without giving away the solution

For every evaluated function, state:

- Its exact name and signature, including parameter order and defaults.
- The scientific quantity it computes and the model assumptions.
- Input types, shapes, units, indexing, ordering, and valid ranges.
- Output type, shape, units, normalization, sign, and ordering conventions.
- Accuracy requirements, including absolute and relative tolerances.
- Any required numerical convention, branch selection, boundary condition, or approximation.
- Behavior at relevant boundary and undefined cases, including errors if tested.
- Earlier functions it may use, named explicitly in the step prompt.

Everything that affects grading must appear in the agent-visible text:

- problem_description_main
- required_dependencies
- step_description_prompt
- function_header
- return_line

Do not put a necessary formula, constant, unit, input exclusion, or exception rule only in background.md,
step_background, source.md, the structured contract, or reference code. Those fields are not all visible
to every agent.

Specify the scientific problem and what correctness means. Leave meaningful derivation, method selection,
or integration to the agent. Do not expose hidden test cases or provide a line-by-line solution recipe.

If two scientifically valid choices can differ by more than the stated tolerance, either specify the
required choice or grade a method-independent result with an appropriate tolerance. A test cannot secretly
select your preferred grid, stopping rule, eigenvector sign, or treatment of a near-zero quantity.

## 3. Use meaningful, ordered substeps

Meet the minimum shown in the upload panel. Do not add artificial steps just to reach it. At least one
evaluated step must require a substantive scientific decision rather than straightforward transcription.

Each step implements one function and has its own scientific output and tests.

- Number steps consecutively from 1.
- Dependencies must refer only to earlier steps.
- Independent steps are fine where scientifically justified. Do not invent dependencies.
- If a step uses earlier functions, name them in the visible prompt and list their step numbers in
  step_dependencies.

The final step should deliver the task's integrated scientific result. Its signature must agree with
problem_io.

Keep function names unique across the task and avoid shadowing imports or built-ins.

## 4. Build and independently check the reference

Provide the per-step reference implementations and a complete solution.py defining all required
functions. Their signatures and behavior must agree with the scaffolds and specification.

Per-step files should not redefine earlier step functions. They can call earlier functions as supplied by
the evaluation context.

Use the supported Python environment and declare every solution import in required_dependencies as Python
import lines, not package-installation requirements.

During execution, do not:

- Install packages or use the network.
- Inspect evaluator or source files.
- Manipulate processes.
- Rely on machine-specific paths.
- Perform substantial top-level computation or depend on shared mutable state.

As an authoring quality standard, also prepare second_solution.py using a genuinely different method and
run it against the tests. A rearranged copy of the reference is not an independent check.

If a second implementation is impractical, discuss the limitation with the project lead and provide
another defensible verification route. Do not fabricate one. The platform accepting a second-solution
field does not mean it automatically executed it.

Establish expected results through:

- Analytic identities or bounds.
- Independently derived special cases.
- An independently implemented computation.
- Verified literature with matching assumptions and units.

Re-running the same reference or copying its outputs is not independent evidence. Higher precision can
diagnose numerical error, but it does not alone establish that the underlying model or equation is
correct.

Repeat reference runs in clean processes. Check random seeds, hash/order sensitivity, and available
numerical thread configurations. A correct answer should not depend on an accidental seed or reduction
order.

If randomness is part of the intended problem, explain its error and define reproducibility conventions.

## 5. Test the science, not just your implementation

Create a coverage map before finalizing tests. For every promised behavior, identify:

- The test that checks it.
- Why the input matters scientifically.
- Where the expected result comes from.
- The tolerance and its justification.
- The plausible scientific mistake it detects.

Cover the following where relevant:

- Ordinary cases: More than one representative parameter regime.
- Scientific limits: Known limiting cases, symmetries, conservation laws, or scaling relations.
- Numerical stress: Cancellation, conditioning, stiffness, large or small scales, and near-degeneracy
  within the allowed domain.
- Boundaries: Included endpoints and near-boundary values, with defined behavior.
- Integrated result: The final function, not only intermediate helpers.
- Invalid inputs: Only behavior explicitly promised by the specification.

Do not claim all positive inputs are valid if you only validated a comfortable middle range.

For a ratio whose denominator can vanish, define the zero case or justify and state the exclusion. For a
branch-dependent solution, make branch selection reproducible.

Tests need not exhaust a continuous range, but your scientific argument and stress checks must support the
range you promise.

### Tolerances

Use scientifically justified tolerances that accept independent correct methods and reject errors that
matter.

- Compare floating-point results numerically.
- Do not use exact floating-point equality or zero relative tolerance.
- State the grading accuracy requirement in the visible specification.
- Handle near-zero outputs with an appropriate absolute tolerance.
- Account for error propagation through the full calculation.

Computed scientific results should dominate each step's tests, rather than input-validation checks.

Test each evaluated function directly. Do not test a hidden helper, implementation detail, dtype, exception
message, or ordering unless the public contract requires it.

### Test-case format

Separate cases using the current parser format:

```python
# --- test case 0 ---
import numpy as np
# Construct this case's inputs and independently justified expectation.
# Call the evaluated function and assert the promised numerical behavior.

# --- test case 1 ---
import numpy as np
# Another self-contained scientific case.
```

Each case runs independently. Repeat its imports and setup; do not rely on variables or random state from
another case.

Keep n_test_cases equal to the actual number of cases. Whole-task tests in tests/general.py call the final
function and run against the complete solution.py.

### Stored targets

If using HDF5 targets:

- Include the actual data file in the upload.
- Verify every group and target referenced by every test exists.
- Give each stored target independent target_evidence.
- Never generate expected results by calling the candidate being graded.

For new tasks, prefer compact inline expected values or directly evaluated analytic checks where practical.

An evidence record uses these exact keys:

```yaml
target_evidence:
  - test_case: 0
    target: 0
    method: special_case
    note: "Explain the independent derivation and how its assumptions match this case."
    limits: "State the regime where that derivation applies."
```

This is a format example, not a complete task.

test_case is zero-based. target is zero-based and required when a case reads multiple stored targets.

Valid evidence methods are: analytic_bound, special_case, independent_recomputation, literature.

Evidence belongs with the corresponding step or whole-task tests. Do not attach stored-target evidence to
an unrelated inline-only case.

## 6. Try to break your tests before submitting

Passing your reference is only the first check. Your test suite should:

- Accept the reference and a scientifically valid alternative.
- Reject missing work and all-functions-return-None or zero controls.
- Reject incorrect shapes where shape is part of the stated contract.
- Reject plausible scientific errors, not only nonsense outputs.
- Remain stable across repeat runs and reasonable environment variation.

Include at least one named scientific mutant per step and one complete whole-task mutant in the skeleton's
mutants lists.

A mutant preserves the required interface but introduces one explainable scientific mistake, such as:
missing normalization; incorrect sign or unit conversion; an omitted interaction; the wrong solution
branch; an approximation applied outside its valid regime.

Run each mutant and identify the case that rejects it.

A step mutant must fail its targeted step; other correctly implemented steps need not receive zero. A
whole-task mutant must fail the integrated tests.

Do not weaken tests so a wrong mutant passes or tighten them to exclude a valid alternative.

If a plausible wrong solution passes, add a scientifically relevant, in-domain discriminator. If a valid
alternative fails, investigate the specification, target provenance, and error budget first.

## 7. Package a complete, clean task

Start from the current downloaded skeleton, preserving its field names and relative paths:

```
tasks/<stable-task-folder>/
  problem.yaml
  background.md
  source.md
  solution.py
  second_solution.py
  steps/step_1.py ... step_N.py
  solution/step_1.py ... step_N.py
  tests/step_1.py ... step_N.py
  tests/general.py
  mutants/*.py
```

Keep the same folder identity for revisions. Do not add a placeholder problem_id or change the identity to
bypass a rejected task.

Enter the task's display name in the upload panel. Use sub_steps, not an invented alternative field name.

All declared paths must exist relative to the task folder. Scaffolds contain the function header and an
unimplemented body, not the reference answer.

### Structured contracts

Fill out each step's contract using: quantity, assumptions, result_kind, valid_input_ranges, units,
output_shape, step_dependencies.

result_kind is exact, approximate, or asymptotic.

These records help review. They do not replace agent-visible requirements.

### Metadata

Complete the metadata truthfully: subfield; tags; a positive expert_time_estimate_hours;
relevant_experience; difficulty_explanation; solution_explanation; verification_explanation; author fields
requested by the current template.

Use your real affiliation, or exactly "Independent Researcher" if appropriate.

Do not invent credentials, profile links, citations, or conflict disclosures. research_advisor and
referred_by are optional. Keep subfield to at most 120 characters.

Do not assume all identity metadata will be removed from delivered files.

### Cleanliness and upload limits

Keep personal information out of scientific prose, code, tests, and fixtures. Remove editor caches,
.DS_Store, secrets, local paths, unrelated documents, and review correspondence.

The platform generates the evaluation bundle. Do not add your own Harbor or container configuration to
this authoring format unless the project lead asks for it.

Follow the current upload limits. Start with one polished task rather than a large batch. Oversized
serialized task content can also be refused; compact test generation is preferable to huge pasted arrays.

Check the upload preview and resolve parser findings before submitting.

## 8. Revise with evidence, not trial and error

Read the full feedback and relevant failed case or trajectory before editing.

Identify whether the cause is: a scientific or specification defect; inadequate tests; a reference bug;
genuine difficulty; a platform or provider failure.

Infrastructure findings should be reported for investigation, not worked around by changing the science.

For each revision, document: the finding; its cause; the specific change; the regression test; what you
reran.

Use the appropriate edit_label: repair, clarification, grading_fix, robustness, difficulty_increase.

Fix related issues together. Do not resubmit after changing only the first error message.

Do not loosen tolerances, hide conventions, add irrelevant error checks, or make work deliberately slow to
influence model results. Change difficulty through legitimate scientific reasoning and integration while
keeping the contract fair and verifiable.

Version, difficulty-miss, in-flight, and author-trial limits apply. Follow the limits shown in the platform
and ask the lead before continuing a closed or held task.

Passing automated checks, reaching human review, receiving a reviewer Accept, and final approval are
different states.

## Final self-review

Do not upload until you can answer yes to each item:

- A domain expert can explain why this is meaningful scientific work.
- Every graded requirement is visible to the agent.
- All steps, signatures, dependencies, units, and conventions agree.
- The reference is accurate and defined throughout the promised domain.
- Expected results have a defensible independent basis.
- Tests cover representative regimes and relevant failure-prone boundaries.
- A valid alternative passes; scientific mutants fail the intended checks.
- Empty or no-op work does not earn scientific credit.
- Repeat runs are stable and runtime is reasonable for the problem.
- Every declared file and stored target is present.
- No private or unrelated material is included.
- Metadata and verification claims are truthful.
- Known gaps are resolved or explicitly raised with the lead.

If any answer is no, polish the task before spending another submission attempt.
