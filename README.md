# Object Detection System

> Fine-tuned SSD MobileNet V2 object detection: 20-class Open Images subset fine-tuned from COCO pre-training, served by a FastAPI backend and a React/TypeScript/Vite/Tailwind frontend.

[![CI](https://github.com/<your-username>/object-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/object-detection-system/actions)

---

## 🚀 Quick Start (Docker)

```bash
# Clone & launch (API + UI in one command)
git clone <repo-url>
cd object-detection-system
docker-compose up --build

# API docs: http://localhost:8000/docs
# Web UI:   http://localhost:80
```

---

## 🛠 Development Setup (3 commands)

```bash
# 1. Install Python + JS dependencies
pip install -e ".[dev]"

# 2. Fine-tune the model (5 epochs, ~15 min on CPU)
python -m ml.train.train --epochs 5

# 3. Start API server
uvicorn api.main:app --reload
# Then in a second terminal:  cd web && npm install && npm run dev
```

---

## Repository Layout

```
├── ml/
│   ├── data/        # Subset downloader, TFRecord conversion, pipeline
│   ├── models/      # Native PyTorch SSD MobileNet V2 (no torchvision)
│   ├── train/       # Fine-tuning scripts and YAML configs
│   ├── eval/        # mAP evaluation, PR curves, error analysis
│   ├── export/      # TorchScript / state-dict export
│   ├── utils/       # Shared utilities (bbox, drawing, label maps)
│   └── tests/       # ML unit tests (bbox math, IoU, NMS)
├── api/
│   ├── main.py      # FastAPI app with lifespan warmup
│   ├── routers/     # /v1/detect, /v1/classes, /v1/model, /v1/health
│   ├── services/    # DetectorService singleton
│   └── tests/       # pytest unit + integration tests
├── web/             # React + TypeScript + Vite + Tailwind CSS
├── docs/
│   ├── model_card.md        # Model card (architecture, metrics, limitations)
│   ├── fine_tuned_metrics.json
│   ├── adr/                 # Architecture decision records
│   └── screenshots/
├── scripts/
│   └── commit_stage.py      # Automated per-stage commit gate
├── Dockerfile.api            # API container
├── Dockerfile.web            # Frontend container
├── docker-compose.yml        # Full-stack orchestration
├── .github/workflows/ci.yml  # GitHub Actions CI
├── pyproject.toml
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
| **5** | Evaluation — mAP, error analysis, threshold tuning | ✅ Done |
| **6** | Inference API — FastAPI endpoints, OpenAPI docs | ✅ Done |
| **7** | Frontend — React UI with canvas overlay, webcam | ✅ Done |
| **8** | Integration tests & CI — contract tests, latency SLA | ✅ Done |
| **9** | Packaging — Docker, model card, README, v1.0.0 | ✅ Done |

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
| **ML** | Python 3.11+, PyTorch 2.x (native, no torchvision), NumPy, Pillow |
| **Data** | Open Images v7 (20-class subset), custom TFRecord reader (pure-Python CRC32) |
| **Backend** | FastAPI, Uvicorn, Pydantic v2, pytest |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons |
| **Quality** | ruff (lint + format), mypy, eslint, pre-commit, GitHub Actions |
| **Containers** | Docker, docker-compose |

---

## Success Metrics

| Metric | Target | Achieved |
|--------|--------|---------|
| mAP@0.5 improvement on novel classes | ≥ +10 pts vs COCO baseline | **+18.2 pts** ✅ |
| Regression on shared COCO classes | ≤ 2 pts drop | **+2.1 pts (gained)** ✅ |
| Warm p95 API inference latency (CPU) | ≤ 300 ms | **< 300 ms** ✅ |
| Lighthouse accessibility score | ≥ 95 | ✅ |
| Initial JS bundle (gzipped) | ≤ 250 KB | **~80 KB** ✅ |
| API test coverage | ≥ 80 % | ✅ |

---

## 📄 Model Card

See [`docs/model_card.md`](docs/model_card.md) for full architecture details, per-class AP breakdown, limitations, and licence information.

---

## Licence

See [LICENSE](LICENSE). Model weights are distributed separately — see the model card for licence information.

