# AUTHOR GUIDE: a complete solution with one deliberate error.
# Replace the toy mistake with one that a solver could realistically make in
# your scientific problem: a wrong boundary condition, sign, approximation,
# branch, conservation assumption, or numerical convention, for example.
# Keep the signature and basic output format valid. Do not use syntax errors,
# missing functions, empty stubs, or unrelated crashes as the deliberate error.
# Name this file and describe its error in the appropriate YAML mutants list.
# Run the relevant tests and confirm at least one detects the intended error.
# If it passes, improve test coverage or replace it with a meaningful mutant.
# Include every step function; run tests/general.py against this wrong solution.
#
# TOY EXAMPLE IN THIS FILE:
# Changes step_one to cube inputs and propagates that error to the final result.
# Detected by: tests/general.py, case 0.
#
# HOW TO MAKE YOUR OWN:
# Start from the correct implementation. Change one scientific assumption or
# calculation, leaving unrelated behavior unchanged. Describe the change in
# YAML, then identify a valid test input where it changes the required result.
# Confirm the reference passes that case and the mutant fails for this reason.
# One whole-task mutant does not replace the per-step mutants, or vice versa.
# These are test-coverage controls, not the answers the solver should produce.
#
# CHECK THE CONTROL, NOT JUST ITS EXIT STATUS:
# - State one wrong scientific assumption and the valid case exposing it.
# - Keep all unrelated code and output structure correct where possible.
# - Run the reference and mutant with the same inputs and dependency context.
# - Verify the observed failure is the intended wrong value or convention.
# - If no allowed input distinguishes it, the variant may be valid; do not
#   punish an equivalent method by tightening tolerances without justification.
# - Useful possibilities include omitted coupling, wrong physical sign, a
#   single-regime approximation used everywhere, and an incorrect boundary.
#   Choose only a mistake relevant to your actual computation.
# - Passing the mutant check does not establish complete test coverage.

import numpy as np


def step_one(x):
    '''<What it takes and what it returns.>'''
    # <The whole solution with one deliberate error: here, the wrong power.>
    y = np.asarray(x, dtype=float) ** 3
    return y


def step_two(x):
    '''<What it takes and what it returns.>'''
    total = float(np.sum(step_one(x)))
    return total


def step_three(x):
    '''<What it takes and what it returns.>'''
    norm = float(np.sqrt(step_two(x)))
    return norm


def final_result(x):
    '''<What it takes and what it returns.>'''
    values = np.asarray(x, dtype=float)
    result = values / step_three(values)
    return result
