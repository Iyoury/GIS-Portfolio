# AUTHOR GUIDE: reference implementation for subtask 4.
# Replace the toy body with your scientific method; match the YAML header.
# Implement this step only. Earlier step functions are supplied by the harness.
# If you call an earlier function, name that dependency in this step prompt
# and keep the contract dependency list aligned. Do not repeat earlier code.
# Declare every required import in problem.yaml. Keep the implementation in
# solution.py consistent with this file.
# Verify ordinary cases and relevant boundaries using justified tolerances.
#
# Test file: tests/step_4.py. Full reference: solution.py.
# Local checks need the declared imports and any earlier step functions in
# the same namespace. Executing this file by itself is not a full step test.
# Check the requested quantity itself, not only agreement between two copies
# of the same algorithm.
#
# SUBSTEP 4 FOCUS:
# Define the integrated scientific output and every component it contains.
# Explain how earlier results contribute; name the functions it may call.
# State the valid domain of any final division, transformation, or average.
# Check the end-to-end answer, not only the individual intermediate values.
# This is the final step in this example; adapt the role if you add steps.
# Before writing code, define this result and its valid domain in the YAML
# prompt. Use that specification to design independent tests.
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


def final_result(x):
    '''<What it takes and what it returns.>'''
    # <Your reference implementation of step 4. It may call earlier steps.>
    values = np.asarray(x, dtype=float)
    result = values / step_three(values)
    return result
