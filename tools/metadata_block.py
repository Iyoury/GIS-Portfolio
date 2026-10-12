"""Move the author/metadata fields of problem.yaml into the `metadata:` block the platform expects.

Usage: python3 tools/metadata_block.py [--email E] [--profile URL] [--coi TEXT] [--domain D] [tasks/<folder> ...]

The platform's upload check asks, in the metadata block, for: author_name, author_email, author_organization,
author_profile, conflicts_of_interest, relevant_experience, subfield, tags, expert_time_estimate_hours,
difficulty_explanation, solution_explanation, verification_explanation.
Top-level keys subfield, tags, expert_time_estimate_hours, relevant_experience, author, affiliation,
difficulty_explanation, solution_explanation, verification_explanation are moved (text kept verbatim, indented)
into `metadata:`; author -> author_name, affiliation -> author_organization. Existing metadata keys are kept and
the given --email/--profile/--coi values set or replace author_email/author_profile/conflicts_of_interest.
--domain replaces the top-level domain value. Idempotent.
"""
import glob
import json
import re
import sys

import yaml

MOVE = ["author_name", "author_email", "author_organization", "author_profile", "conflicts_of_interest",
        "relevant_experience", "subfield", "tags", "expert_time_estimate_hours", "difficulty_explanation",
        "solution_explanation", "verification_explanation"]
RENAME = {"author": "author_name", "affiliation": "author_organization"}
TOP = set(MOVE) | set(RENAME)


def blocks(lines):
    """Split into top-level blocks: (key or None, [lines])."""
    out = []
    for l in lines:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", l)
        if m:
            out.append([m.group(1), [l]])
        elif out and (l.startswith((" ", "\t")) or not l.strip()):
            out[-1][1].append(l)
        else:  # comment or stray line at column 0
            out.append([None, [l]])
    return out


def main():
    args = sys.argv[1:]
    opts = {}
    folders = []
    i = 0
    while i < len(args):
        if args[i] in ("--email", "--profile", "--coi", "--domain"):
            opts[args[i][2:]] = args[i + 1]; i += 2
        else:
            folders.append(args[i]); i += 1
    folders = folders or sorted(glob.glob("tasks/*/"))
    for f in folders:
        p = f.rstrip("/") + "/problem.yaml"
        text = open(p).read()
        bl = blocks(text.splitlines(keepends=True))
        meta = {}
        order = []
        new = []
        insert_at = None
        for key, ls in bl:
            if key == "metadata":
                insert_at = len(new) if insert_at is None else insert_at
                # existing metadata block: split its children (2-space indented keys)
                cur = None
                for l in ls[1:]:
                    m = re.match(r"^  ([A-Za-z_][A-Za-z0-9_]*):", l)
                    if m:
                        cur = m.group(1); meta[cur] = [l[2:]]; order.append(cur)
                    elif cur is not None:
                        meta[cur].append(l[2:] if l.startswith("  ") else l)
                continue
            if key in TOP:
                insert_at = len(new) if insert_at is None else insert_at
                name = RENAME.get(key, key)
                first = ls[0].replace(key + ":", name + ":", 1)
                meta[name] = [first] + ls[1:]
                order.append(name)
                continue
            if key == "domain" and "domain" in opts:
                ls = ["domain: %s\n" % opts["domain"]] + [l for l in ls[1:] if not l.strip()]
            new.append((key, ls))
        for k, v in (("author_email", opts.get("email")), ("author_profile", opts.get("profile")),
                     ("conflicts_of_interest", opts.get("coi"))):
            if v is not None:
                meta[k] = ["%s: %s\n" % (k, json.dumps(v, ensure_ascii=False))]
                if k not in order:
                    order.append(k)
        if insert_at is None:
            insert_at = 1
        mlines = ["metadata:\n"]
        for k in MOVE + [k for k in order if k not in MOVE]:
            if k in meta:
                body = meta[k]
                while body and not body[-1].strip():
                    body = body[:-1]
                mlines += [("  " + l) if l.strip() else l for l in body]
        mlines.append("\n")
        out = []
        for j, (key, ls) in enumerate(new):
            if j == insert_at:
                out += mlines
            out += ls
        if insert_at >= len(new):
            out += mlines
        res = "".join(out)
        y = yaml.safe_load(res)
        missing = [k for k in MOVE if k not in (y.get("metadata") or {})]
        open(p, "w").write(res)
        print(p, "metadata keys missing:", missing, "| domain:", y.get("domain"))


if __name__ == "__main__":
    main()
