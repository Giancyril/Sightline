#!/usr/bin/env bash
# scripts/commit_stage.sh
# Git Bash / Unix wrapper — delegates to the cross-platform Python script.
# Usage: bash scripts/commit_stage.sh <stage-number> "<message>"
set -euo pipefail
python scripts/commit_stage.py "$@"

