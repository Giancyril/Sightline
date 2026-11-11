# Architecture Overview

## High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Browser                           │
│  React (Vite) · TypeScript · Tailwind · Radix UI · Lucide       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Service (api/)                       │
│  POST /v1/detect  ·  GET /v1/classes  ·  GET /v1/model          │
│  GET /healthz                                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ TF SavedModel
┌────────────────────────────▼────────────────────────────────────┐
│                     Detector (ml/export/)                        │
│  SSD MobileNetV2 backbone (frozen) + fine-tuned heads           │
│  Preprocessing · NMS · Label mapping                            │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. User uploads image (or sends webcam frame) via frontend.
2. Frontend POSTs to `POST /v1/detect` with optional `threshold` param.
3. API validates file type/size, strips EXIF, preprocesses with OpenCV.
4. TensorFlow SavedModel runs inference; API applies NMS and threshold.
5. Response: `{image_size, inference_ms, detections: [{label, score, box}]}`.
6. Frontend renders bounding boxes on HTML canvas; populates inspector table.

## ADR Index

| ADR | Decision |
|-----|----------|
| [ADR-001](adr/001-ssd-mobilenetv2.md) | Use SSD MobileNetV2 as the base model |
| [ADR-002](adr/002-open-images-subset.md) | 20-class Open Images subset strategy |

_ADRs are added as decisions are made throughout the project._

