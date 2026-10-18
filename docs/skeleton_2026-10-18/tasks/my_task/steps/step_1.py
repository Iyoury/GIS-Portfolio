# AUTHOR GUIDE: scaffold mirror for subtask 1.
# Replace the signature and docstring to match function_header in problem.yaml.
# Keep the body unimplemented here; the answer belongs in solution/step_N.py.
# The importer reads the YAML function_header, not this mirror file.
# Put input shapes, units, output definitions, and relevant edge conventions in
# the visible prompt or header. Do not hide requirements in reference comments.
#
# Related files: solution/step_1.py and tests/step_1.py.
# A NotImplementedError is appropriate only in this scaffold, not in a
# submitted reference or a scientific mutant.
#
# SUBSTEP 1 FOCUS:
# Define the first independently useful scientific result.
# State how raw inputs become the mathematical representation you use.
# Check dimensions, units, ordering, and any symmetry or normalization.
# The toy squaring operation is wiring only, not a suggested scientific step.
# Complete the checklist beside sub_steps entry 1 in problem.yaml.
# Put the resulting input/output specification into the actual header and
# prompt, not just into these author comments.
#
#
# DEFINE THE SCIENTIFIC RESULT
# - Name the quantity being computed, its mathematical definition, and why it matters
#   in this scientific setting.
# - Distinguish an exact model result from an approximation, fitted quantity,
#   asymptotic limit, or discretized estimate.
# - State the assumptions that make the problem well posed. If multiple answers are
#   valid, define the requested branch or accept equivalent answers.
# - Check formulas, constants, signs, units, and source attributions against the
#   derivation. A plausible topic is not evidence that the computation is correct.
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
# KEEP DIFFICULTY SCIENTIFIC
# - Explain the research decision, derivation, method choice, or integration
#   challenge that the solver must resolve.
# - Simple setup steps can be useful, but the whole task must do more than transcribe
#   supplied equations or repackage a standard worked exercise.
# - Do not remove necessary conventions to make the task harder. Do not add arbitrary
#   precision, string formatting, excessive runtime, or busywork as difficulty.
# - Build realistic wrong-method mutants and confirm the tests distinguish them from
#   the reference for the intended scientific reason.

import numpy as np


def step_one(x):
    '''<What it takes and what it returns.>'''
    raise NotImplementedError
