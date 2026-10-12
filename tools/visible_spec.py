"""Write the agent-visible text of a task (what a solver sees) to a Markdown file.

Usage: python3 tools/visible_spec.py tasks/<folder> <out.md>

Fields: problem_description_main, required_dependencies, problem_io, and per step the
step_description_prompt, function_header and return_line. Nothing else (no background, tests,
solutions, source.md or contracts).
"""
import sys

import yaml


def main():
    root, out = sys.argv[1], sys.argv[2]
    y = yaml.safe_load(open(root.rstrip("/") + "/problem.yaml"))
    parts = ["# Task\n", y["problem_description_main"].strip(), "\n## Required dependencies\n```python",
             y["required_dependencies"].strip(), "```\n"]
    for s in sorted(y["sub_steps"], key=lambda s: int(s["step_number"])):
        parts += [f"## Step {s['step_number']}\n", s["step_description_prompt"].strip(), "\n```python",
                  s["function_header"].rstrip(), "    ...", "    " + s["return_line"].strip(), "```\n"]
    parts += ["## Final function (problem_io)\n```python", y["problem_io"].rstrip(), "```\n"]
    open(out, "w").write("\n".join(parts) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
