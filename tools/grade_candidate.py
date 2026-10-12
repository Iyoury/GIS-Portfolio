"""Grade a candidate solution file end to end against a task's tests (each case alone, fresh process).

Usage: python3 tools/grade_candidate.py tasks/<folder> <candidate.py> [-j 3]

The candidate file defines every step function (as an agent writing from the visible text would).
Unlike crown_check's per-step mode, the candidate's own earlier functions are used for every step
(end-to-end solving, as the platform's difficulty probe does). Prints per-step and general pass counts
and the first error line of each failing case.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crown_check import Task, jobs_for_cases, run_many  # noqa: E402


def main():
    root, cand = sys.argv[1], sys.argv[2]
    j = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 3
    t = Task(root)
    code = open(cand).read()
    total = passed = 0
    with ThreadPoolExecutor(j) as pool:
        groups = [(f"step {int(s['step_number'])}", s["tests"]) for s in t.steps]
        groups.append(("general", t.y.get("general_tests", "tests/general.py")))
        for label, rel in groups:
            res = run_many(t, label, jobs_for_cases(code, rel, t), pool)
            ok = [i for i, o, _ in res if o]
            total += len(res)
            passed += len(ok)
            print(f"{label}: {len(ok)}/{len(res)} pass")
            for i, o, err in res:
                if not o:
                    print(f"  case {i} FAIL: {err}")
    print(f"TOTAL {passed}/{total}")


if __name__ == "__main__":
    main()
