# ADR-001: Selection of SSD MobileNet V2 as Base Detector

## Status
Accepted

## Context
The project requires an object detection model that:
1. Achieves p95 inference latency $\le 300\text{ ms}$ on a standard 4-core CPU for images up to 1280px (Stage 6 metric target).
2. Can be fine-tuned reliably on a free-tier GPU session (Google Colab / Kaggle) within 8 hours.
3. Has established pre-trained checkpoints on Common Objects in Context (COCO) covering everyday baseline objects (person, laptop, chair, car, etc.).
4. Can be deployed seamlessly in a lightweight container or browser edge runtime (TFLite / ONNX).

Candidate architectures evaluated:
- **YOLOv8n / YOLOv11n**: High accuracy and speed, but AGPL license constraints and non-standard PyTorch/TensorFlow interoperability.
- **Faster R-CNN ResNet-50**: Higher accuracy on small objects, but heavy parameter footprint (~160 MB), high CPU latency (>600ms on 4-core CPU), and slow convergence on free GPU notebooks.
- **SSD MobileNet V2 (300x300)**: Single-stage detector combining inverted residual MobileNetV2 feature extractor with multi-scale SSD feature maps. Compact footprint (~15-20 MB), inference time <100ms on CPU, permissive Apache 2.0 license, native TensorFlow/TFLite/ONNX support.

## Decision
We select **SSD MobileNet V2 (300x300)** as the base detector for the project.

Key parameters:
- **Input Resolution**: $300 \times 300 \times 3$
- **Feature Map Scales**: 6 pyramid levels ($19\times19, 10\times10, 5\times5, 3\times3, 2\times2, 1\times1$)
- **Anchor Counts**: 4 or 6 aspect ratios per pyramid cell (~1,917 default prior boxes)
- **Base Checkpoint**: Pre-trained on COCO train2017 (80 categories)
- **Fine-Tuning Scope**: Classification and box regression output heads fine-tuned on our 20-class Open Images subset while freezing backbone blocks in Phase 1.

## Consequences
- **Pros**: Outstanding CPU inference speed (~45-75ms), small artifact footprint suitable for Docker and web serving, straightforward fine-tuning convergence.
- **Cons**: Lower recall on very small objects (<32px) compared to two-stage detectors; mitigated via multi-scale pyramid anchors and high-resolution web client downscaling.

