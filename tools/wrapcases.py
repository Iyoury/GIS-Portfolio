"""Wrap the body of every test case in a function so that no test variable or helper leaks into the
module namespace shared with the candidate's code (a test assignment such as `pi = ...` or a helper class
named like one of the candidate's helpers would otherwise overwrite the candidate's globals).

Usage: python3 tools/wrapcases.py <tests/file.py> [--write]

Each case
    # --- test case N ---
    # description comments
    <body>
becomes
    # --- test case N ---
    # description comments
    import numpy as np
    def _crown_case():
        <body, indented>
    _crown_case()
Already-wrapped files are left unchanged. The body must not use `global` or `nonlocal` at its own top level.
"""
import ast
import re
import sys

MARK = re.compile(r"^# --- test case (\d+) ---\s*$")


def wrap(text):
    lines = text.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if MARK.match(l)]
    if not starts:
        raise SystemExit("no normalized markers")
    out = ["".join(lines[:starts[0]])]
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        block = lines[s:e]
        head = [block[0]]
        k = 1
        while k < len(block) and (block[k].startswith("#") or not block[k].strip()):
            if not block[k].strip():
                break
            head.append(block[k])
            k += 1
        body = "".join(block[k:]).strip("\n")
        if "def _crown_case():" in body:
            out.append("".join(block))
            continue
        ast.parse(body)  # must be valid on its own
        for node in ast.walk(ast.parse(body)):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise SystemExit(f"case {MARK.match(block[0]).group(1)}: global/nonlocal present")
        indented = "".join(("    " + l if l.strip() else l) for l in body.splitlines(keepends=True))
        if not indented.endswith("\n"):
            indented += "\n"
        case = "".join(head) + "import numpy as np\n\n\ndef _crown_case():\n" + indented + "\n\n_crown_case()\n\n"
        out.append(case)
    return "".join(out).rstrip("\n") + "\n"


def main():
    path = sys.argv[1]
    new = wrap(open(path).read())
    ast.parse(new)
    if "--write" in sys.argv:
        open(path, "w").write(new)
        print("wrapped", path, file=sys.stderr)
    else:
        sys.stdout.write(new)


if __name__ == "__main__":
    main()
