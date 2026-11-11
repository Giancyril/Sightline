# Model Card — SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)

> Model version: **v1.0.0** | Released: 2026-11

---

## Model Overview

| Field | Value |
|---|---|
| **Model name** | SSD MobileNet V2 (Open Images 20-Class Fine-Tuned) |
| **Task** | 2D Object Detection |
| **Architecture** | Single Shot MultiBox Detector (SSD) with MobileNet V2 backbone |
| **Input size** | 300 × 300 RGB |
| **Anchors** | 2 268 default boxes across 6 feature map scales |
| **Output classes** | 21 (20 object categories + background) |
| **Framework** | PyTorch 2.x (native, no torchvision) |
| **Export formats** | TorchScript (`.pt`), state-dict checkpoint (`.pth`) |

---

## Architecture Details

The backbone is a **MobileNet V2** network using inverted residual blocks with linear bottlenecks and depthwise separable convolutions, implemented from scratch in native PyTorch (no `torchvision` dependency).

**Detection head:** Six SSD feature-map layers at strides 8, 16, 32, 64, 100, 300. Each cell predicts bounding-box offsets and class scores for a fixed set of aspect-ratio anchors (1:1, 2:1, 1:2, 3:1, 1:3 at the appropriate scales).

**Loss:** Smooth-L1 regression loss + Focal-weighted cross-entropy classification loss. Hard negative mining maintains a 3:1 negative:positive ratio during training.

**Post-processing:** Confidence threshold → class-agnostic Non-Maximum Suppression (IoU ≥ 0.45 by default).

---

## Training Details

### Dataset

The model was fine-tuned on a **20-class subset of the Open Images v7** dataset:

**COCO shared classes (10):** Person, Laptop, Chair, Book, Backpack, Cell phone, Bottle, Clock, Car, Dog

**Novel classes (10):** Guitar, Microphone, Helmet, Headphones, Wheelchair, Whiteboard, Flashlight, Calculator, Stapler, Wrench

| Split | Images | Annotations |
|---|---|---|
| Train | ~160 | ~400 |
| Validation | 40 | 150 |

### Training Procedure

Training used a two-phase curriculum:

| Phase | Epochs | Frozen layers | LR |
|---|---|---|---|
| **Phase 1** (head-only) | 2 | All backbone layers | 1 × 10⁻³ |
| **Phase 2** (fine-tune) | 3 | None | 1 × 10⁻⁴ |

- **Optimizer:** Adam with cosine annealing schedule
- **Augmentations:** Random horizontal flip, random brightness/contrast jitter, random crop, aspect-ratio-preserving resize to 300 × 300
- **Hardware:** CPU-only (no GPU required)

---

## Evaluation Metrics

Evaluated on the 40-image Open Images validation split at IoU threshold 0.5.

### Overall Performance

| Metric | Baseline (COCO pre-trained) | Fine-Tuned | Δ |
|---|---|---|---|
| **mAP@0.5 — All 20 classes** | — | **0.264** | — |
| **mAP@0.5 — COCO shared (10)** | 0.326 | **0.347** | +0.021 ✅ |
| **mAP@0.5 — Novel classes (10)** | 0.000 | **0.182** | **+18.2 pts** ✅ |
| **COCO mAP@[0.5:0.95]** | — | **0.160** | — |

> **Success gates:** Novel class gain ≥ +10 pts ✅ (+18.2 achieved) | COCO shared drop ≤ 2 pts ✅ (gained +2.1 pts)

### Per-Class AP@0.5

| Class | Type | AP@0.5 |
|---|---|---|
| Book | COCO shared | 0.431 |
| Backpack | COCO shared | 0.414 |
| Car | COCO shared | 0.375 |
| Chair | COCO shared | 0.338 |
| Laptop | COCO shared | 0.337 |
| Dog | COCO shared | 0.334 |
| Clock | COCO shared | 0.327 |
| Cell phone | COCO shared | 0.317 |
| Person | COCO shared | 0.313 |
| Bottle | COCO shared | 0.284 |
| Guitar | Novel | 0.152 |
| Microphone | Novel | 0.228 |
| Helmet | Novel | 0.217 |
| Calculator | Novel | 0.192 |
| Stapler | Novel | 0.185 |
| Wrench | Novel | 0.173 |
| Flashlight | Novel | 0.174 |
| Whiteboard | Novel | 0.165 |
| Wheelchair | Novel | 0.165 |
| Headphones | Novel | 0.167 |

### Threshold Tuning

Optimal confidence threshold **0.25** maximises F1 (0.640):

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.10 | 0.42 | 0.78 | 0.546 |
| 0.20 | 0.58 | 0.68 | 0.626 |
| **0.25** | **0.65** | **0.63** | **0.640** |
| 0.35 | 0.76 | 0.51 | 0.610 |
| 0.50 | 0.87 | 0.36 | 0.509 |

### Inference Latency (CPU)

| Measurement | Value |
|---|---|
| **Single-threaded warm inference** | ~120–188 ms |
| **p95 warm inference (8 runs)** | < 300 ms |
| **Input resolution** | 300 × 300 (any input → auto-resized) |
| **Hardware** | Intel x86-64, single CPU core |

---

## Limitations

- **Small training set:** Validation set is only 40 images; per-class AP estimates carry high variance, especially for novel classes with fewer than 10 validation examples.
- **Resolution constraint:** Input is resized to 300 × 300; small objects in high-resolution images may be missed.
- **Domain shift:** The model was fine-tuned on web-crawled Open Images data; performance may degrade on out-of-domain images (medical, satellite, industrial).
- **No depth / orientation:** The model predicts 2D axis-aligned bounding boxes only.
- **CPU inference:** No GPU optimisation; latency ≥ 100 ms on CPU; not suitable for real-time video at > 10 FPS on a single core without further optimisation (quantisation, pruning).
- **Class imbalance:** "Person" is heavily over-represented in both training and validation splits, which may inflate its AP.

---

## Intended Use

- **In-scope:** Educational demonstrations of fine-tuning SSD detectors on small custom datasets; non-critical prototyping; offline batch image analysis.
- **Out-of-scope:** Safety-critical applications (autonomous vehicles, medical diagnosis, security surveillance); high-volume real-time video streams; production systems without further evaluation and monitoring.
- **Not validated for:** Faces (privacy), minors, biometric identification, or any purpose requiring fairness auditing.

---

## Data & License

- **Training data:** [Open Images v7](https://storage.googleapis.com/openimages/web/index.html) — licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Backbone weights:** Initialised from COCO-pretrained SSD MobileNet V2 (original training details in [SSD paper](https://arxiv.org/abs/1512.02325)); weights are publicly available.
- **Model weights & code:** See repository LICENSE file.

---

## Citation

If you use this model or codebase in research, please cite:

```
@software{ods2026,
  title  = {Object Detection System — SSD MobileNet V2 fine-tuned on Open Images 20-class subset},
  year   = {2026},
  url    = {https://github.com/<your-username>/object-detection-system}
}
```

---

## Contact

Please open a GitHub issue for bugs, questions, or feature requests.

