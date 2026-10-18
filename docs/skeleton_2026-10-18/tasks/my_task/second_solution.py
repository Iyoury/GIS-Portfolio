# AUTHOR GUIDE: independent whole-task solution.
# Replace this toy alternative with a genuinely different method, derivation,
# or independently validated implementation of the same scientific contract.
# Define all step functions with the same signatures and output conventions.
# Run the same tests on both references. Investigate disagreements rather than
# loosening tolerances until both pass. Record justified validation evidence in
# problem.yaml. Do not label untested or duplicated code as independent.
#
# Use the same inputs, output definitions, and units as the primary method.
# Compare representative cases and relevant limits. Keep an explanation of
# how expected values were obtained in verification_explanation.
# This example is intentionally simple; its toy alternative is not evidence
# that a real scientific task has been independently validated.
#
# REFERENCE SELF-REVIEW: trace the computation against the visible contract.
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
#
# Check that no hidden clipping threshold, early exit, regularizer, default
# parameter, or branch silently changes the requested mathematical result.
# A stabilization that preserves the specified result is different from a
# new special-case rule. Document scientifically consequential choices.

import numpy as np


def step_one(x):
    '''<What it takes and what it returns.>'''
    # <Step 1 by a different method from solution.py.>
    values = np.asarray(x, dtype=float)
    y = values * values
    return y


def step_two(x):
    '''<What it takes and what it returns.>'''
    # <Step 2 by a different method: here, an inner product.>
    values = np.asarray(x, dtype=float)
    total = float(np.dot(values, values))
    return total


def step_three(x):
    '''<What it takes and what it returns.>'''
    # <Step 3 by a different method: here, a library norm.>
    norm = float(np.linalg.norm(np.asarray(x, dtype=float)))
    return norm


def final_result(x):
    '''<What it takes and what it returns.>'''
    # <Step 4 by a different method from solution.py.>
    values = np.asarray(x, dtype=float)
    result = values / np.linalg.norm(values)
    return result
