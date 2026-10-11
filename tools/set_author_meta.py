"""Fill the author-owned metadata of every task's problem.yaml (text edit, formatting preserved).

Usage: python3 tools/set_author_meta.py --experience "..." [--affiliation "..."] [tasks/<folder> ...]
Replaces the relevant_experience placeholder line and, if given, the affiliation line.
"""
import glob
import json
import re
import sys


def main():
    args = sys.argv[1:]
    exp = aff = None
    folders = []
    i = 0
    while i < len(args):
        if args[i] == "--experience":
            exp = args[i + 1]; i += 2
        elif args[i] == "--affiliation":
            aff = args[i + 1]; i += 2
        else:
            folders.append(args[i]); i += 1
    folders = folders or sorted(glob.glob("tasks/*/"))
    for f in folders:
        p = f.rstrip("/") + "/problem.yaml"
        t = open(p).read()
        if exp is not None:
            t, n = re.subn(r"^relevant_experience:.*$", "relevant_experience: " + json.dumps(exp), t, flags=re.M)
            assert n == 1, (p, n)
        if aff is not None:
            t, n = re.subn(r"^affiliation:.*$", "affiliation: " + json.dumps(aff), t, flags=re.M)
            assert n == 1, (p, n)
        open(p, "w").write(t)
        print("updated", p)


if __name__ == "__main__":
    main()
