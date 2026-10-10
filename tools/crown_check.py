"""Validate one Crown task folder case by case, as the guide prescribes.

Usage: python3 tools/crown_check.py tasks/<folder> [ref second mut shift controls format] [-j 3]
       (no mode = all modes)

Every test case runs alone in a fresh Python process:
  step k : solution/step_1..k-1 (evaluation context) + candidate step k + the case
  general: candidate whole solution + the case
Modes
  format   : markers "# --- test case N ---" numbered 0..n-1, n_test_cases matches, no clock reads,
             no exact float equality patterns, declared files exist, contract keys, metadata keys
  ref      : solution/step_k.py on tests/step_k.py; solution.py on tests/general.py
  second   : second_solution.py on every step's cases and on general
  mut      : every step mutant must fail >= 1 case of its step; every whole mutant >= 1 general case
  shift    : reference outputs scaled by (1 +- 1e-12) must still pass every case
  controls : all-None and all-zero stubs must not pass any computed-value case (reported per step)
"""
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import yaml

MARK = re.compile(r"^# --- test case (\d+) ---\s*$", re.M)
ANYMARK = re.compile(r"^# --- test case", re.M)
CONTRACT = {"quantity", "assumptions", "result_kind", "valid_input_ranges", "units", "output_shape",
            "step_dependencies"}
META = {"subfield", "tags", "expert_time_estimate_hours", "relevant_experience", "difficulty_explanation",
        "solution_explanation", "verification_explanation"}

PERT = r'''
import numpy as _ck_np
def _ck_pert(v, s):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float):
        return float(v * (1.0 + s))
    if isinstance(v, _ck_np.floating):
        return type(v)(v * (1.0 + s))
    if isinstance(v, _ck_np.ndarray) and _ck_np.issubdtype(v.dtype, _ck_np.floating):
        return v * (1.0 + s)
    if isinstance(v, tuple):
        return tuple(_ck_pert(x, s) for x in v)
    if isinstance(v, list):
        return [_ck_pert(x, s) for x in v]
    if isinstance(v, dict):
        return {k: _ck_pert(x, s) for k, x in v.items()}
    return v
'''


def fname(header):
    m = re.search(r"def\s+(\w+)\s*\(", header)
    return m.group(1)


def cases_of(path):
    text = open(path).read()
    pos = [m.start() for m in ANYMARK.finditer(text)]
    out = []
    for i, p in enumerate(pos):
        e = pos[i + 1] if i + 1 < len(pos) else len(text)
        out.append(text[p:e])
    return text[:pos[0]] if pos else text, out


class Task:
    def __init__(self, root):
        self.root = root
        self.y = yaml.safe_load(open(os.path.join(root, "problem.yaml")))
        self.steps = sorted(self.y["sub_steps"], key=lambda s: int(s["step_number"]))
        self.funcs = {int(s["step_number"]): fname(s["function_header"]) for s in self.steps}

    def p(self, rel):
        return os.path.join(self.root, rel)

    def src(self, rel):
        return open(self.p(rel)).read()

    def prefix(self, k):
        return "".join(self.src(s["solution"]) + "\n" for s in self.steps if int(s["step_number"]) < k)


def run_code(code, cwd, timeout=900):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=os.environ.get("CK_TMP")) as f:
        f.write(code)
        name = f.name
    try:
        r = subprocess.run([sys.executable, name], cwd=cwd, capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
        err = (r.stderr.strip().splitlines() or [""])[-1][:220]
    except subprocess.TimeoutExpired:
        ok, err = False, "TIMEOUT"
    finally:
        os.unlink(name)
    return ok, err


def jobs_for_cases(candidate, test_rel, task):
    pre, cases = cases_of(task.p(test_rel))
    return [(candidate + "\n" + c, i) for i, c in enumerate(cases)]


def run_many(task, label, jobs, pool):
    futs = [(i, pool.submit(run_code, code, task.root)) for code, i in jobs]
    res = []
    for i, f in futs:
        ok, err = f.result()
        res.append((i, ok, err))
    return res


def mode_format(t):
    probs = []
    y = t.y
    for s in t.steps:
        k = int(s["step_number"])
        for key in ("scaffold", "solution", "tests"):
            if not os.path.exists(t.p(s[key])):
                probs.append(f"step {k}: missing {key} {s[key]}")
        for m in s.get("mutants", []) or []:
            if "file" in m and not os.path.exists(t.p(m["file"])):
                probs.append(f"step {k}: missing mutant file {m['file']}")
            if "file" not in m:
                probs.append(f"step {k}: mutant {m.get('name')} has no file")
        if not (s.get("mutants") or []):
            probs.append(f"step {k}: no mutant")
        c = s.get("contract") or {}
        if set(c) != CONTRACT:
            probs.append(f"step {k}: contract keys {sorted(c)}")
        text = open(t.p(s["tests"])).read()
        nums = [int(n) for n in MARK.findall(text)]
        nall = len(ANYMARK.findall(text))
        if nums != list(range(len(nums))) or nall != len(nums):
            probs.append(f"step {k}: markers not normalized ({nall} markers, strict {nums})")
        if s.get("n_test_cases") != nall:
            probs.append(f"step {k}: n_test_cases={s.get('n_test_cases')} but {nall} cases")
        pre, _ = cases_of(t.p(s["tests"]))
        if re.search(r"^\s*(import|from|def|class|[A-Za-z_]\w*\s*=)", pre, re.M):
            probs.append(f"step {k}: code before the first test case (cases must be self-contained)")
    for rel in [s["tests"] for s in t.steps] + [y.get("general_tests", "tests/general.py")]:
        text = open(t.p(rel)).read()
        for pat, what in ((r"perf_counter|time\.time\(|time\.monotonic|signal\.|setitimer|alarm\(", "clock/timing"),
                          (r"rtol\s*=\s*0(\.0*)?\b", "rtol=0"),
                          (r"==\s*0\.0\b|==\s*1\.0\b|==\s*-?\d+\.\d+(e-?\d+)?\b", "float ==")):
            for m in re.finditer(pat, text):
                line = text[:m.start()].count("\n") + 1
                probs.append(f"{rel}:{line}: {what}: {text.splitlines()[line - 1].strip()[:90]}")
    g = y.get("general_tests", "tests/general.py")
    text = open(t.p(g)).read()
    pre, _ = cases_of(t.p(g))
    if re.search(r"^\s*(import|from|def|class|[A-Za-z_]\w*\s*=)", pre, re.M):
        probs.append("general: code before the first test case")
    for m in y.get("mutants", []) or []:
        if "file" not in m or not os.path.exists(t.p(m["file"])):
            probs.append(f"whole mutant {m.get('name')}: missing file")
    if not (y.get("mutants") or []):
        probs.append("no whole-task mutant")
    missing = META - set(y)
    if missing:
        probs.append(f"metadata missing: {sorted(missing)}")
    for p in probs:
        print("FORMAT", p)
    print("FORMAT", "OK" if not probs else f"{len(probs)} problem(s)")


def main():
    root = sys.argv[1]
    modes = [a for a in sys.argv[2:] if not a.startswith("-") and not a.isdigit()]
    j = 3
    if "-j" in sys.argv:
        j = int(sys.argv[sys.argv.index("-j") + 1])
    modes = modes or ["format", "ref", "second", "mut", "shift", "controls"]
    t = Task(root)
    gen = t.y.get("general_tests", "tests/general.py")
    pool = ThreadPoolExecutor(j)
    summary = []
    if "format" in modes:
        mode_format(t)
    if "ref" in modes:
        for s in t.steps:
            k = int(s["step_number"])
            res = run_many(t, f"ref {k}", jobs_for_cases(t.prefix(k) + t.src(s["solution"]), s["tests"], t), pool)
            bad = [(i, e) for i, ok, e in res if not ok]
            print(f"REF step {k}: {len(res) - len(bad)}/{len(res)} pass", *[f"| case {i}: {e}" for i, e in bad])
            summary.append(not bad)
        res = run_many(t, "ref general", jobs_for_cases(t.src("solution.py"), gen, t), pool)
        bad = [(i, e) for i, ok, e in res if not ok]
        print(f"REF general (solution.py): {len(res) - len(bad)}/{len(res)} pass", *[f"| case {i}: {e}" for i, e in bad])
        summary.append(not bad)
        # solution.py on every step file too
        for s in t.steps:
            k = int(s["step_number"])
            res = run_many(t, "", jobs_for_cases(t.src("solution.py"), s["tests"], t), pool)
            bad = [(i, e) for i, ok, e in res if not ok]
            if bad:
                print(f"REF solution.py on step {k}: FAIL", *[f"| case {i}: {e}" for i, e in bad])
                summary.append(False)
    if "second" in modes:
        sec = t.src(t.y.get("second_solution", "second_solution.py"))
        for s in t.steps:
            k = int(s["step_number"])
            res = run_many(t, "", jobs_for_cases(sec, s["tests"], t), pool)
            bad = [(i, e) for i, ok, e in res if not ok]
            print(f"SECOND step {k}: {len(res) - len(bad)}/{len(res)} pass", *[f"| case {i}: {e}" for i, e in bad])
            summary.append(not bad)
        res = run_many(t, "", jobs_for_cases(sec, gen, t), pool)
        bad = [(i, e) for i, ok, e in res if not ok]
        print(f"SECOND general: {len(res) - len(bad)}/{len(res)} pass", *[f"| case {i}: {e}" for i, e in bad])
        summary.append(not bad)
    if "mut" in modes:
        for s in t.steps:
            k = int(s["step_number"])
            for m in s.get("mutants", []) or []:
                code = t.src(m["file"]) if "file" in m else m["code"]
                res = run_many(t, "", jobs_for_cases(t.prefix(k) + code, s["tests"], t), pool)
                failed = [i for i, ok, e in res if not ok]
                print(f"MUTANT step {k} {m['name']}: fails {len(failed)}/{len(res)} cases {failed}",
                      "OK" if failed else "!!! SURVIVES")
                summary.append(bool(failed))
        for m in t.y.get("mutants", []) or []:
            code = t.src(m["file"]) if "file" in m else m["code"]
            res = run_many(t, "", jobs_for_cases(code, gen, t), pool)
            failed = [i for i, ok, e in res if not ok]
            print(f"MUTANT whole {m['name']}: fails {len(failed)}/{len(res)} general cases {failed}",
                  "OK" if failed else "!!! SURVIVES")
            summary.append(bool(failed))
    if "shift" in modes:
        for sgn in (1, -1):
            for s in t.steps:
                k = int(s["step_number"])
                f = t.funcs[k]
                wrap = (PERT + f"\n_ck_orig = {f}\ndef {f}(*a, **kw):\n    return _ck_pert(_ck_orig(*a, **kw), {sgn}e-12)\n")
                res = run_many(t, "", jobs_for_cases(t.prefix(k) + t.src(s["solution"]) + wrap, s["tests"], t), pool)
                bad = [(i, e) for i, ok, e in res if not ok]
                print(f"SHIFT {sgn:+d}e-12 step {k}: {len(res) - len(bad)}/{len(res)} pass", *[f"| case {i}: {e}" for i, e in bad])
                summary.append(not bad)
    if "controls" in modes:
        for val in ("None", "0.0"):
            for s in t.steps:
                k = int(s["step_number"])
                stub = f"def {t.funcs[k]}(*args, **kwargs):\n    return {val}\n"
                res = run_many(t, "", jobs_for_cases(t.prefix(k) + stub, s["tests"], t), pool)
                passed = [i for i, ok, e in res if ok]
                print(f"CONTROL return {val} step {k}: passes {len(passed)}/{len(res)} cases {passed}",
                      "OK" if not passed else "!!! earns credit")
            stubs = "".join(f"def {f}(*args, **kwargs):\n    return {val}\n" for f in t.funcs.values())
            res = run_many(t, "", jobs_for_cases(stubs, gen, t), pool)
            passed = [i for i, ok, e in res if ok]
            print(f"CONTROL return {val} general: passes {len(passed)}/{len(res)} cases {passed}",
                  "OK" if not passed else "!!! earns credit")
    print("SUMMARY", "ALL OK" if all(summary) else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
