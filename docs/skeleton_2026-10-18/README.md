# Task folders: skeleton

This zip is one complete, correctly formatted task. Every value is a
placeholder: replace each <...> and each placeholder function with your own
task, and keep every file and key.

    tasks/
      my_task/                     one folder per task (a zip may hold several)
        problem.yaml               the task, its settings and its ordered steps
        background.md              background prose for the whole task
        source.md                  where the problem comes from
        solution.py                the whole-task reference solution
        second_solution.py         a second correct solution, by another method
        steps/step_1.py            step 1's scaffold: header, no body
        steps/step_2.py ... step_4.py
        solution/step_1.py         step 1's reference solution
        solution/step_2.py ... step_4.py
        tests/step_1.py            step 1's test cases
        tests/step_2.py ... step_4.py
        tests/general.py           whole-task tests, run against solution.py
        mutants/*.py               wrong versions the tests must reject

## Task folders

The folder name identifies the task. Give every task its own folder name, and
keep that name when you upload a new version of the task. Do not add a
`problem_id` to problem.yaml: the folder is the id.

## problem.yaml

The step list is `sub_steps:`, spelled exactly so, one entry per step in
order. Every path in problem.yaml is relative to the task folder. The comments
in problem.yaml say what every key holds.

Every solver is shown `problem_description_main`, `required_dependencies`,
and for each step its `step_description_prompt`, `function_header` and
`return_line`. Background prose (background.md and `step_background`) is
not shown to every solver, and the contract, tests, solutions, mutants and
source.md never are. So anything a test checks (a raised error, a type, a
shape, a unit, an ordering) must be stated in that step's prompt or header.

## Metadata

problem.yaml's `metadata:` block says who wrote the task and describes it:
the author's `author_name`, `author_email`, `author_organization`,
`author_profile`, `conflicts_of_interest` and `relevant_experience`
(`research_advisor` and `referred_by` are optional), and the task's
`subfield`, `tags`, `expert_time_estimate_hours` (a number above 0) and
three short explanations: what makes it difficult, how the solution works,
and how the tests verify it. It is written into the bundle's task.toml; no
solver is shown it, and reviewers never see the author fields. Fill in every
required field: uploads may be refused without them. Fill in your
affiliation, profile link and conflict-of-interest disclosure: placeholder
values ("none", "n/a", "independent", "self-employed") are refused. Keep
`subfield` to a short phrase (at most 120 characters).

## Imports

Task code may import only the Python 3.12 standard library and numpy 2.1.3, scipy 1.14.1, sympy 1.13.3, h5py 3.12.1, mpmath 1.3.0; tasks run offline, so nothing else can be installed.
Write every import the solutions use in `required_dependencies`, one import
per line. That list is what the code is graded with.

## Steps

Each step is one function. A step may call the functions of earlier steps;
its solution file does not repeat them, and its `step_description_prompt`
must name every earlier function it calls (the content check sends back a
step that uses one its prompt never mentions). The scaffold in `steps/` is the
step's `function_header` with `raise NotImplementedError` as its body.

## Test files

A tests file holds one or more test cases, each starting with its own
separator line, numbered from 0:

    # --- test case 0 ---
    assert ...

    # --- test case 1 ---
    assert ...

Each test case runs on its own, so repeat the imports it needs. A step's
tests call that step's function directly; the whole-task tests in
tests/general.py call the last step's function and run against solution.py.
Write every expected value inline in the test, as these do.

## Second solution

`second_solution.py` solves the whole task again by a different method (not
the same formula rewritten), defining every step's function. The tests must
accept it: set each tolerance so a correct method other than yours passes.

## Mutants

A mutant is a version of a function with one deliberate scientific error,
which the tests must reject. Every step needs at least one (its `mutants:`
list names the file, which defines the step's function), and the task needs
at least one whole-task mutant (the top-level `mutants:` list, a complete
wrong solution). Give each a short name and a one-line description of the
error it encodes.

## Limits

At most 50 tasks and 8 MB per zip.
