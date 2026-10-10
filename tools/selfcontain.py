"""Rewrite a Crown test file so that every test case is self-contained.

Usage: python3 tools/selfcontain.py <tests/step_k.py> --funcs f1,f2,... [--write]

The file is split on the test-case markers ("# --- test case N ..."). Everything before the first marker
is the preamble (imports, helpers, constants). Each case gets a normalized marker "# --- test case i ---",
its description as plain comments, the imports it needs and the preamble statements it uses
(transitively), then its own body. Names that a case uses but that are defined only in an earlier case are
reported: those need a manual fix. Without --write the result goes to stdout.
"""
import ast
import builtins
import re
import sys

MARK = re.compile(r"^# --- test case (\d+)(.*)$")


def stored_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


def top_defined(stmt):
    """Names a top-level statement binds in the module namespace."""
    out = set()
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(stmt.name)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for a in stmt.names:
            out.add((a.asname or a.name).split(".")[0])
    else:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
    return out


def loaded_names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def split(text):
    lines = text.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if MARK.match(l)]
    if not starts:
        raise SystemExit("no test-case markers found")
    pre = "".join(lines[:starts[0]])
    cases = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        block = lines[s:e]
        # description: rest of the marker line plus the comment lines that continue it (until "---")
        first = MARK.match(block[0]).group(2).strip()
        first = first.lstrip(":").strip()
        desc, k = [], 1
        if first.endswith("---"):
            desc.append(first[:-3].rstrip())
        else:
            if first:
                desc.append(first)
            while k < len(block) and block[k].lstrip().startswith("#"):
                c = block[k].strip()[1:].strip()
                k += 1
                if c.endswith("---"):
                    desc.append(c[:-3].rstrip())
                    break
                desc.append(c)
            else:
                # no closing "---": the comment lines read belong to the body, not the description
                k = 1
                desc = [first] if first else []
        body = "".join(block[k:])
        cases.append(([d for d in desc if d], body))
    return pre, cases


def main():
    path = sys.argv[1]
    funcs = set()
    if "--funcs" in sys.argv:
        funcs = set(sys.argv[sys.argv.index("--funcs") + 1].split(","))
    text = open(path).read()
    pre, cases = split(text)
    pre_tree = ast.parse(pre)
    pre_lines = pre.splitlines(keepends=True)
    stmts = []
    for st in pre_tree.body:
        src = "".join(pre_lines[st.lineno - 1:st.end_lineno])
        # keep decorators
        if getattr(st, "decorator_list", None):
            first = min(d.lineno for d in st.decorator_list)
            src = "".join(pre_lines[first - 1:st.end_lineno])
        stmts.append((st, src, top_defined(st), loaded_names(st)))
    header_comments = []
    for l in pre_lines:
        if l.startswith("#") or not l.strip():
            header_comments.append(l)
        else:
            break
    problems = []
    for st, src, defs, uses in stmts:
        if defs & funcs:
            problems.append(f"preamble rebinds an evaluated function: {sorted(defs & funcs)}: {src.strip()[:80]}")
        if not defs and not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
            problems.append(f"preamble side-effect statement (kept in every case): {src.strip()[:80]}")
    pre_defined = set().union(*[d for _, _, d, _ in stmts]) if stmts else set()
    known = set(dir(builtins)) | funcs | pre_defined
    out = []
    if header_comments and "".join(header_comments).strip():
        out.append("".join(header_comments).rstrip() + "\n\n")
    seen_case_defs = {}
    for i, (desc, body) in enumerate(cases):
        try:
            body_tree = ast.parse(body)
        except SyntaxError as e:
            raise SystemExit(f"case {i}: syntax error {e}")
        need = loaded_names(body_tree)
        own = stored_names(body_tree)
        # transitive closure over preamble statements
        chosen = set()
        frontier = set(need)
        while True:
            added = False
            for j, (st, src, defs, uses) in enumerate(stmts):
                if j in chosen:
                    continue
                is_side = not defs and not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))
                if (defs & frontier) or is_side:
                    chosen.add(j)
                    frontier |= uses
                    added = True
            if not added:
                break
        # always keep "import numpy as np" if the preamble has it
        for j, (st, src, defs, uses) in enumerate(stmts):
            if isinstance(st, (ast.Import, ast.ImportFrom)) and "np" in defs:
                chosen.add(j)
        missing = sorted(n for n in need - own - known)
        if missing:
            origin = {n: seen_case_defs.get(n) for n in missing}
            problems.append(f"case {i}: uses names not defined in the case or preamble: {origin}")
        for n in own:
            seen_case_defs.setdefault(n, i)
        parts = [f"# --- test case {i} ---\n"]
        parts += [f"# {d}\n" for d in desc]
        imports = [stmts[j][1] for j in sorted(chosen) if isinstance(stmts[j][0], (ast.Import, ast.ImportFrom))]
        others = [stmts[j][1] for j in sorted(chosen) if not isinstance(stmts[j][0], (ast.Import, ast.ImportFrom))
                  and not (isinstance(stmts[j][0], ast.Expr) and isinstance(stmts[j][0].value, ast.Constant))]
        parts += [s if s.endswith("\n") else s + "\n" for s in imports]
        if others:
            parts.append("\n")
            parts.append("\n\n".join(s.rstrip("\n") for s in others) + "\n\n")
        parts.append(body.strip("\n") + "\n\n")
        out.append("".join(parts))
    result = "".join(out).rstrip("\n") + "\n"
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    if "--write" in sys.argv:
        open(path, "w").write(result)
        print(f"wrote {path}: {len(cases)} cases", file=sys.stderr)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
