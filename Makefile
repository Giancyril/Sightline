# Object Detection System — Makefile
#
# Usage:
#   make lint           Run all Python linters
#   make test           Run all tests
#   make test-stage-N   Run tests for a specific stage (e.g. make test-stage-0)
#   make commit-stage N=0 MSG="bootstrap repository and tooling"

PYTHON  := python
RUFF    := $(PYTHON) -m ruff
BLACK   := $(PYTHON) -m black
MYPY    := $(PYTHON) -m mypy
PYTEST  := $(PYTHON) -m pytest

SRC_DIRS := ml api scripts

.PHONY: help lint lint-ruff lint-black lint-mypy format test test-stage-0 \
        test-stage-1 test-stage-2 test-stage-3 test-stage-4 test-stage-5 \
        test-stage-6 test-stage-7 test-stage-8 test-stage-9 commit-stage clean

help:
	@echo "Available targets:"
	@echo "  lint              Run ruff + black --check + mypy"
	@echo "  format            Auto-format with ruff and black"
	@echo "  test              Run all pytest tests"
	@echo "  test-stage-N      Run tests for stage N (0-9)"
	@echo "  commit-stage      Lint + test + git commit + tag  (N= MSG= required)"
	@echo "  clean             Remove caches and build artefacts"

# ── Linting ─────────────────────────────────────────────────────────────────

lint: lint-ruff lint-black lint-mypy

lint-ruff:
	$(RUFF) check $(SRC_DIRS)

lint-black:
	$(BLACK) --check $(SRC_DIRS)

lint-mypy:
	-$(MYPY) $(SRC_DIRS) --ignore-missing-imports

format:
	$(RUFF) check --fix $(SRC_DIRS)
	$(BLACK) $(SRC_DIRS)

# ── Tests ───────────────────────────────────────────────────────────────────

test:
	$(PYTEST)

test-stage-0:
	$(PYTEST) -m "stage0 or (not stage1 and not stage2 and not stage3 and not stage4 and not stage5 and not stage6 and not stage7 and not stage8 and not stage9)"

test-stage-1:
	$(PYTEST) -m "stage1"

test-stage-2:
	$(PYTEST) -m "stage2"

test-stage-3:
	$(PYTEST) -m "stage3"

test-stage-4:
	$(PYTEST) -m "stage4"

test-stage-5:
	$(PYTEST) -m "stage5"

test-stage-6:
	$(PYTEST) -m "stage6"

test-stage-7:
	$(PYTEST) -m "stage7"

test-stage-8:
	$(PYTEST) -m "stage8"

test-stage-9:
	$(PYTEST) -m "stage9"

# ── Commit gate ─────────────────────────────────────────────────────────────
# Usage: make commit-stage N=3 MSG="baseline inference with COCO SSD"

commit-stage:
ifndef N
	$(error N is not set. Usage: make commit-stage N=<number> MSG="<message>")
endif
ifndef MSG
	$(error MSG is not set. Usage: make commit-stage N=<number> MSG="<message>")
endif
	$(PYTHON) scripts/commit_stage.py $(N) "$(MSG)"

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache   -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

