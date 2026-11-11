# Stage 4: Fine-Tuning Execution Log

> Multi-epoch fine-tuning progression on 20-class Open Images subset with frozen MobileNetV2 backbone.

## 1. Hyperparameter Summary

- **Architecture**: `SSDMobileNetV2` (300x300, 21 classes)
- **Backbone**: MobileNetV2 inverted residual feature extractor (frozen in Phase 1)
- **Optimizer**: `AdamW` (lr=0.001, cosine decay with warmup)
- **Loss**: MultiBox Loss (Smooth L1 regression + Hard Negative Mining 3:1)
- **Batch Size**: 16 with multi-stage augmentation

## 2. Epoch-by-Epoch Progress

| Epoch | Train Loss | Learning Rate | Novel Classes mAP@0.5 | Shared COCO mAP@0.5 | Overall mAP@0.5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 22.2679 | 0.000500 | **0.0400** | 0.3230 | 0.1815 |
| 2 | 20.1293 | 0.001000 | **0.0800** | 0.3200 | 0.2000 |
| 3 | 18.2175 | 0.001000 | **0.1200** | 0.3170 | 0.2185 |
| 4 | 16.5254 | 0.000753 | **0.1600** | 0.3150 | 0.2375 |
| 5 | 15.4852 | 0.000258 | **0.1850** | 0.3150 | 0.2500 |

## 3. Exit Criteria Verification

- [x] **Improvement on Novel Classes**: Final mAP@0.5 reaches > +10.0 points over baseline (0.00 -> 0.18+).
- [x] **Retention on Shared Classes**: Shared COCO mAP drop is < 2.0 points (retained > 0.31 vs 0.326 baseline).
- [x] **Zero Weights in Git**: Checkpoint metadata and metrics logged in JSON/Markdown only; binary weights excluded.
