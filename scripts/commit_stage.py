#!/usr/bin/env python3
"""
scripts/commit_stage.py
-----------------------
Cross-platform (Windows + Unix) replacement for commit_stage.sh.

Usage:
    python scripts/commit_stage.py <stage-number> "<message>"

What it does:
    1. Runs `python -m ruff check` on ml/ api/ scripts/
    2. Runs `python -m black --check` on ml/ api/ scripts/
    3. Runs `pytest` for the stage test target
    4. If all pass: git add -A, conventional-commit, lightweight tag stage-N
    5. Any failure stops the script and exits non-zero (no commit, no tag)

Commit types by stage (matches Section 8 of the plan):
    0 -> chore    1,2,3,4,5,6,7 -> feat    8 -> test    9 -> docs
"""

import subprocess
import sys

STAGE_TYPES: dict[int, str] = {
    0: "chore",
    1: "feat",
    2: "feat",
    3: "feat",
    4: "feat",
    5: "feat",
    6: "feat",
    7: "feat",
    8: "test",
    9: "docs",
}

SRC_DIRS = ["ml", "api", "scripts"]


def run(cmd: list[str], description: str) -> bool:
    """Run a subprocess command. Print output. Return True on success."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n[FAIL] {description} exited with code {result.returncode}")
    return result.returncode == 0


def main() -> int | None:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    try:
        stage = int(sys.argv[1])
    except ValueError:
        print(f"[ERROR] Stage number must be an integer, got: {sys.argv[1]}")
        return 1

    message = sys.argv[2]
    commit_type = STAGE_TYPES.get(stage, "feat")
    scope = f"stage-{stage}"
    full_msg = f"{commit_type}({scope}): {message}"
    tag = f"stage-{stage}"

    print(f"\n{'#' * 60}")
    print(f"  COMMIT GATE: Stage {stage}")
    print(f"  Commit message: {full_msg}")
    print(f"  Tag:            {tag}")
    print(f"{'#' * 60}")

    # 1. Ruff lint
    if not run([sys.executable, "-m", "ruff", "check"] + SRC_DIRS, "Ruff lint"):
        return 1

    # 2. Format check (Ruff format or Black)
    format_cmd = [sys.executable, "-m", "ruff", "format", "--check"] + SRC_DIRS
    if not run(format_cmd, "Code format check"):
        return 1

    # 3. Pytest
    if not run([sys.executable, "-m", "pytest", "-q"], "Pytest test suite"):
        return 1

    # 4. Git add + commit
    if not run(["git", "add", "-A"], "Git add all"):
        return 1

    if not run(["git", "commit", "-m", full_msg], f"Git commit: {full_msg}"):
        return 1

    # 5. Tag
    if not run(["git", "tag", tag], f"Git tag: {tag}"):
        return 1

    print(f"\n{'#' * 60}")
    print(f"  SUCCESS! Stage {stage} committed and tagged '{tag}'.")
    print(f"{'#' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
