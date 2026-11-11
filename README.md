# Object Detection System

> Fine-tuned SSD object detection: COCO-pretrained SSD with output layers re-trained on a 20-class Open Images subset, served by a FastAPI backend and a professional React/TypeScript frontend.

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd object-detection-system

# 2. Python environment (requires Python 3.11)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install (core + dev tools)
pip install -e ".[dev]"

# 4. Run lint
make lint

# 5. Run tests
make test

# 6. Start API (after Stage 6)
uvicorn api.main:app --reload

# 7. Start frontend (after Stage 7)
cd web && npm install && npm run dev
```

---

## Repository Layout

```
├── ml/
│   ├── data/        # Subset downloader, TFRecord conversion, tf.data pipeline
│   ├── train/       # Fine-tuning scripts and YAML configs
│   ├── eval/        # mAP evaluation, PR curves, error analysis
│   ├── export/      # SavedModel / TFLite export
│   ├── utils/       # Shared utilities (bbox, drawing, label maps)
│   └── tests/       # ML unit tests (bbox math, IoU, NMS)
├── api/
│   ├── main.py      # FastAPI app
│   ├── routers/     # /v1/detect, /v1/classes, /v1/model (Stage 6)
│   └── tests/       # pytest + httpx API tests
├── web/             # React + TypeScript + Vite + Tailwind (Stage 7)
├── scripts/
│   ├── commit_stage.py   # Cross-platform commit gate
│   └── commit_stage.sh   # Bash wrapper
├── docs/
│   ├── adr/         # Architecture decision records
│   ├── evaluation.md
│   └── model_card.md
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
└── .gitignore
```

---

## Implementation Stages

| Stage | Description | Status |
|-------|-------------|--------|
| **0** | Bootstrap — repo layout, tooling, CI | ✅ Done |
| **1** | Data acquisition — Open Images 20-class subset, EDA | ✅ Done |
| **2** | Input pipeline — TFRecords, tf.data, augmentations | ✅ Done |
| **3** | Baseline inference — COCO-pretrained SSD | ✅ Done |
| **4** | Fine-tuning — detection head on Open Images subset | ✅ Done |
| **5** | Evaluation — mAP, error analysis, threshold tuning | ⬜ In Progress |
| **6** | Inference API — FastAPI endpoints, OpenAPI docs | ⬜ |
| **7** | Frontend — React UI with canvas overlay, webcam | ⬜ |
| **8** | Integration tests & CI — Playwright, contract tests | ⬜ |
| **9** | Packaging — Docker, model card, README, v1.0.0 | ⬜ |

---

## Development Workflow

Every stage ends with an automated commit gate that runs linters and tests before committing:

```bash
# After completing a stage:
make commit-stage N=0 MSG="bootstrap repository and tooling"
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | Python 3.11, TensorFlow 2.x, OpenCV, NumPy, pycocotools |
| **Data** | FiftyOne / Open Images downloader |
| **Backend** | FastAPI, Uvicorn, Pydantic, pytest |
| **Frontend** | React, TypeScript, Vite, Tailwind CSS, Radix UI, Lucide |
| **Quality** | ruff, black, mypy, eslint, prettier, pre-commit, GitHub Actions |
| **Containers** | Docker, docker-compose |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| mAP@0.5 improvement on new classes | ≥ +10 points vs COCO baseline |
| Regression on shared COCO classes | ≤ 2 points drop |
| p95 API inference (CPU, ≤1280px) | ≤ 300 ms |
| Lighthouse accessibility score | ≥ 95 |
| Initial JS bundle (gzipped) | ≤ 250 KB |
| API test coverage | ≥ 80 % |

---

## Licence

See [LICENSE](LICENSE). Model weights are distributed separately — see the model card for licence information.

