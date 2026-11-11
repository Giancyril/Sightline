# Model Evaluation & Benchmark Report

> Quantitative mAP@0.5 evaluations: COCO SSD MobileNet V2 Baseline vs Fine-Tuned Output Heads.

## 1. Executive Summary & Success Gates

| Metric | Baseline (Stage 3) | Fine-Tuned Target (Stage 5) | Status |
|:---|:---:|:---:|:---:|
| **Novel Classes mAP@0.5** (Classes not in COCO) | 0.0000 (0.0%) | >= 0.1000 (+10.0 pts) | Benchmark Set |
| **COCO Shared Classes mAP@0.5** | 0.3264 (32.6%) | >= 0.3064 (drop <= 2 pts) | Benchmark Set |
| **Overall 20-Class mAP@0.5** | 0.1632 (16.3%) | >= 0.2132 | Benchmark Set |

## 2. Stage 3 Baseline Per-Class AP@0.5

| ID | Class Name | Category Group | Baseline AP@0.5 | Notes |
|:---:|:---|:---:|:---:|:---|
| 1 | `Person` | COCO Shared | 0.3253 | Pre-trained knowledge |
| 2 | `Laptop` | COCO Shared | 0.3378 | Pre-trained knowledge |
| 3 | `Chair` | COCO Shared | 0.3432 | Pre-trained knowledge |
| 4 | `Book` | COCO Shared | 0.4389 | Pre-trained knowledge |
| 5 | `Backpack` | COCO Shared | 0.4313 | Pre-trained knowledge |
| 6 | `Cell phone` | COCO Shared | 0.3341 | Pre-trained knowledge |
| 7 | `Bottle` | COCO Shared | 0.0000 | Pre-trained knowledge |
| 8 | `Clock` | COCO Shared | 0.3301 | Pre-trained knowledge |
| 9 | `Car` | COCO Shared | 0.3834 | Pre-trained knowledge |
| 10 | `Dog` | COCO Shared | 0.3401 | Pre-trained knowledge |
| 11 | `Guitar` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 12 | `Microphone` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 13 | `Helmet` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 14 | `Headphones` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 15 | `Wheelchair` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 16 | `Whiteboard` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 17 | `Flashlight` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 18 | `Calculator` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 19 | `Stapler` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |
| 20 | `Wrench` | **Novel (Non-COCO)** | 0.0000 | Zero-shot (Untrained) |

## 3. Analysis & Key Takeaways

- **Novel Concepts (IDs 11-20)**: As expected, the COCO baseline model achieves 0.00 mAP on the 10 classes that do not exist in COCO (`Guitar`, `Microphone`, `Helmet`, `Headphones`, `Wheelchair`, `Whiteboard`, `Flashlight`, `Calculator`, `Stapler`, `Wrench`).
- **Shared Concepts (IDs 1-10)**: Pre-trained weights provide solid initial representations on common classes (`Person`, `Laptop`, `Chair`, `Book`, etc.).
- **Fine-Tuning Objective (Stage 4)**: Replace the classification output layers with 21 classes (20 targets + 1 background), freeze the MobileNetV2 backbone, and train the heads to give the model detection capability on all 20 classes.
