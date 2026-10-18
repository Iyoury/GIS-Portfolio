# --- test case 0 ---
# AUTHOR GUIDE: whole-task tests against solution.py.
# Call the final function and check the complete assembled output.
# Replace the toy assertions with scientific checks and independently justified
# expected values. Do not compute expected outputs by calling the reference.
# Each numbered block runs independently: repeat its imports and setup.
# Include ordinary cases, scientifically meaningful boundaries, and plausible
# wrong-method failures. Check type, shape, units, and conventions as relevant.
# Choose tolerances from numerical error or stability analysis, not to reject
# otherwise correct methods. State all graded rules in the visible prompts.
# Number blocks from 0 and update n_test_cases for per-step files.
# Keep these comments AFTER a test separator. A nonempty preamble before the
# first separator is parsed as an extra test block.
# Keep expected values inline for this layout. Stored-target files require
# their supported data wiring and evidence; never leave a missing data file.
#
# SUGGESTED COVERAGE (choose what is relevant; do not add arbitrary cases):
# - A small instance whose answer you can derive independently.
# - A typical instance in the stated valid range.
# - Boundary, degenerate, or multi-branch cases allowed by the prompt.
# - Shape and ordering checks where the return contract specifies them.
# - A case that detects each included mutant's intended mistake.
# For values near zero, consider absolute as well as relative error. Explain
# the chosen tolerance and check numerical stability across valid inputs.
# Test invalid inputs only when the prompt specifies how they are handled.
# If an empty, constant, or partial implementation passes, inspect coverage.
# Do not copy these toy checks as the test suite for a real submission.
#
# WRITE A SMALL COVERAGE MAP BEFORE ADDING CASES:
# Requirement -> valid input -> expected result/source -> assertion -> wrong approach
#   caught.
# Keep that reasoning beside each real test. These notes are not a new YAML field.
#
# CASE COMMENT TEMPLATE (replace with actual evidence, not a promise):
# Purpose: the scientific rule this case distinguishes.
# Input: why this input is within the stated domain and exercises that rule.
# Expected: derivation or independent calculation establishing this value.
# Accuracy: why this absolute/relative tolerance fits its scale and uncertainty.
# Failure caught: the plausible wrong method that this assertion rejects.
#
# ILLUSTRATIVE CHECKS, ONLY IF YOUR TASK HAS THESE FEATURES:
# - A fit returns coefficients AND a covariance: check both, not only residuals.
# - A weight or temperature is an argument: vary it so ignoring it cannot pass.
# - A spectrum has equivalent eigenvector signs: compare the specified invariant
#   or state a meaningful sign convention; do not require an accidental sign.
# - A response is a tiny difference of large terms: use a stable independent
#   target and an absolute error budget, not an arbitrary relative threshold.
# - A valid zero-input limit makes a denominator vanish: establish the limit or
#   the declared undefined-case behavior before writing an assertion.
# - A full pipeline uses multiple intermediates: include a case that catches
#   incorrect composition even when isolated step tests pass.
#
# LOCAL EXECUTION REMINDER:
# Each test block needs its own setup and imports. Count blocks, not assertions.
# Do not put an executable or comment-only preamble above the first separator.
# Run under the supported scientific Python environment with declared imports.
# Keep tests deterministic; do not depend on wall-clock timing, network access,
# private files, machine-specific paths, or another test block having run.
# A rejected mutant must fail an intended assertion, not just crash from missing
# imports or an unrelated setup error. Keep ordinary correct methods passing.
#
# MAKE THE VISIBLE CONTRACT SUFFICIENT
# - Specify arguments, dimensions, valid ranges, units, output components, and any
#   nonstandard ordering or normalization in each relevant step prompt or header.
# - Do not leave graded rules only in background.md, contract summaries, source
#   notes, reference comments, or author explanations.
# - State special-case and error behavior only when it is relevant. Tests must not
#   invent a new rule for empty inputs, exact dtypes, or exception classes.
# - Describe the scientific problem, not which test inputs the grader uses. Remove
#   references to hidden tests or the expected grading strategy from solver-facing
#   prose.
#
# CHECK COVERAGE IN BOTH DIRECTIONS
# - For each stated requirement, identify a test assertion that checks it. For each
#   assertion, identify the visible requirement it checks.
# - Inspect every returned component, important input parameter, allowed branch, and
#   scientific regime. Ask what plausible wrong implementation could still pass.
# - Whole-task assertions can cover intermediate requirements. Avoid duplicating the
#   same validation check for every argument without a scientific reason.
# - Do not require an exact container, dtype, helper call, or implementation pathway
#   unless that is genuinely part of the stated task.
#
# PROBE THE ALLOWED NUMERICAL DOMAIN
# - For every division, log, square root, inverse function, matrix solve, and
#   cancellation-prone difference, identify the allowed inputs where it becomes
#   singular or unstable.
# - Test relevant limits and regime changes, not only comfortable middle-of-range
#   inputs. Name the input and failure mechanism; do not add random edge cases for
#   volume.
# - If a quantity is undefined at an allowed input, define meaningful behavior or
#   justify a narrower domain. Update the prompt, references, and tests together.
# - Compare a fragile case with an analytic limit, higher-precision calculation, or
#   genuinely independent method before setting its expected value.
#
# MAKE ACCURACY REQUIREMENTS FAIR
# - Justify absolute and relative tolerances from numerical error and the scale of
#   the result, including values near zero.
# - Where valid grids, interpolation rules, solver stopping criteria, or equivalent
#   methods give different answers, either specify the scientifically necessary
#   choice or allow their justified numerical spread.
# - Check error propagation between substeps. A later step must not silently require
#   an earlier result to be more accurate than its own contract and tests require.
# - Avoid boundaries separated by one floating-point rounding step unless that
#   distinction is scientifically necessary and explicitly defined.
#
# ESTABLISH TRUSTWORTHY REFERENCES AND TARGETS
# - The reference must compute the requested answer, not recognize test inputs or
#   return cached test answers.
# - Record how expected values were obtained and the limits of that method. Agreement
#   with a copy of the same reference is not independent validation.
# - Use analytic cases, conservation laws, independent formulations, convergence
#   studies, or higher precision where appropriate. Explain what each check
#   establishes and what it does not.
# - Do not present an analytic bound as an exact value or a special-case formula as a
#   target for a different regime. Never invent citations or successful validation
#   runs.

import numpy as np
# Explain the origin of each expected value and the reason for its tolerance.
assert np.allclose(final_result([1.0, 2.0, 2.0]), [1 / 3, 2 / 3, 2 / 3])
