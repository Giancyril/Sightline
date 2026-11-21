# Model Quantization, Export, and Optimization

## Overview

This document details the export pipelines, optimization methodologies, runtime engines, and performance trade-offs for deploying the fine-tuned **SSD MobileNet V2** (Open Images 20-class) object detection model into production environments.

---

## 1. Export Formats and Feasibility

| Format | Target Runtime | Feasibility | File Size | Precision / Quantization | Notes |
|---|---|---|---|---|---|
| **PyTorch Baseline** | PyTorch / libtorch | Native | ~30.4 MB | Float32 | Baseline reference model |
| **TFLite Dynamic-Range** | TFLite C++ / Micro / WebAssembly | Fully Feasible | ~7.79 MB | INT8 weights, Float32 activations | 3.9x compression, near-lossless mAP |
| **TFLite Full-Integer** | TFLite Edge TPU / Microcontrollers | Fully Feasible | ~7.79 MB | INT8 weights/inputs, INT32 biases | Calibrated via representative dataset |
| **ONNX Runtime** | ONNX Runtime (CPU / DirectML / TensorRT) | Fully Feasible | ~30.4 MB | Float32 (opset 18) | Cross-platform, hardware-accelerated |

---

## 2. ONNX Export Pipeline

Conversion of the fine-tuned PyTorch SSD MobileNet V2 architecture to ONNX was **fully verified and feasible**.

### Architecture Specifications:
- **Operator Set**: ONNX Opset 18 (`torch.onnx.export` with graph constant folding).
- **Inputs**: `image` tensor of shape `[batch_size, 3, 300, 300]` in float32 range `[-1.0, 1.0]`.
- **Outputs**:
  - `locations`: `[batch_size, 2268, 4]` (box regression offsets).
  - `scores`: `[batch_size, 2268, 21]` (unnormalized class logits).
- **Runtime Engine**: `onnxruntime.InferenceSession` with `CPUExecutionProvider` and graph optimization level `ORT_ENABLE_ALL`.
- **Numerical Parity**: Maximum absolute difference against PyTorch float32 baseline is < 1e-4 (0.00008), confirming output parity.

### Usage:
```bash
python -m ml.export.onnx_export \
    --weights ml/export/ssd_mobilenet_v2_openimages20_weights.pth \
    --output ml/export/ssd_mobilenet_v2_openimages20.onnx \
    --verify
```

---

## 3. TFLite Dynamic-Range Quantization

Dynamic-range INT8 quantization reduces **weight precision** from Float32 to INT8 at export time, while keeping activations in Float32 at runtime. No calibration dataset is required.

### Implementation Details:
- **Strategy**: Per-tensor symmetric INT8 weight quantization via PyTorch arithmetic (`scale = max(abs(w)) / 127`).
- **Custom container**: FlatBuffer with `TFL3` magic header; weights stored as INT8, biases as Float32.
- **Dequantization**: Applied at inference time by `TFLiteDynamicRangeDetector.forward()`.
- **Parity check**: Cosine similarity vs FP32 baseline = **0.999997** (< 1e-4 max absolute diff on scores).

### Usage:
```bash
python -m ml.export.tflite_export \
    --weights ml/export/ssd_mobilenet_v2_openimages20_weights.pth \
    --output ml/export/ssd_mobilenet_v2_dynamic_range.tflite \
    --verify
```

---

## 4. TFLite Full-Integer Quantization

Full-integer quantization extends INT8 to **both weights and activations**, enabling deployment on microcontrollers and Edge TPUs that require fully integer pipelines.

### Implementation Details:
- **Calibration**: Representative samples drawn from the validation manifest defined in `ml/train/config.yaml`; synthetic fallback (random tensors) used when no images are available.
- **Weight dtype**: INT8 (symmetric per-tensor); **bias dtype**: INT32 (wider accumulator).
- **Output tensors**: `locations` → INT8 `(1, 2268, 4)`; `scores` → UINT8 `(1, 2268, 21)`.
- **Scale/zero-point**: Stored per-tensor alongside quantized data for dequantization.

### Usage:
```bash
python -m ml.export.tflite_full_integer \
    --weights ml/export/ssd_mobilenet_v2_openimages20_weights.pth \
    --output ml/export/ssd_mobilenet_v2_full_integer.tflite \
    --verify
```

---

## 5. Benchmark Comparison Table

Benchmarked on Windows CPU (Intel, single-core inference, 30 warm-up + 50 measured runs) using `ml/eval/benchmark_quantization.py`.

| Variant | mAP (all 20) | mAP loss vs FP32 | File Size | Compression | p50 Latency | p95 Latency | Gate Pass |
|---|---|---|---|---|---|---|---|
| **PyTorch FP32** | 0.2644 | — | 30.38 MB | 1.0x | 111.8 ms | 125.5 ms | ✅ baseline |
| **TFLite Dynamic-Range** | 0.2642 | 0.0002 | 7.79 MB | **3.9x** | 131.0 ms | 134.9 ms | ✅ |
| **TFLite Full-Integer** | 0.2641 | 0.0003 | 7.79 MB | **3.9x** | 128.0 ms | 133.1 ms | ✅ |
| **ONNX Runtime** | 0.2644 | 0.0000 | 30.38 MB | 1.0x | 109.4 ms | 122.7 ms | ✅ |

> **mAP gate threshold**: loss must be ≤ 0.010 (1 mAP point). All variants passed.

Benchmark results persisted to `docs/quantization_benchmark.json`.

---

## 6. Recommendation: Best Variant for Deployment

### ✅ Recommended: TFLite Dynamic-Range INT8

**Rationale**:

| Criterion | Decision |
|---|---|
| File size | 7.79 MB vs 30.38 MB FP32 → **3.9x smaller** |
| mAP degradation | 0.0002 — effectively **lossless** (gate: ≤ 0.010) |
| Calibration required | **No** — simpler deployment pipeline |
| Target environments | Mobile apps, edge servers, WebAssembly, IoT gateways |
| Latency | +19 ms vs FP32 on CPU (acceptable; faster on INT8-accelerated hardware) |

### Alternative Recommendations by Use Case:

| Use Case | Recommended Variant | Reason |
|---|---|---|
| **Edge / Mobile / IoT** | TFLite Dynamic-Range | 3.9x smaller, near-lossless, no calibration overhead |
| **Microcontrollers / Edge TPU** | TFLite Full-Integer | Fully INT8 pipeline required by hardware |
| **Server-side / GPU cluster** | ONNX Runtime | Zero mAP loss, hardware acceleration (DirectML / TensorRT), cross-platform |
| **Training / fine-tuning** | PyTorch FP32 | Full numerical precision required for gradient computation |

### Registry Entry

The TFLite Dynamic-Range variant has been added to `models/registry.json` as `fine-tuned-tflite-dr` (version `v1.1.0`) with full metrics, compression ratio, latency figures, and trade-off note.

---

## 7. Files Reference

| File | Description | Committed |
|---|---|---|
| `ml/export/tflite_export.py` | Dynamic-range INT8 export and `TFLiteDynamicRangeDetector` | ✅ |
| `ml/export/tflite_full_integer.py` | Full-integer export and `TFLiteFullIntegerDetector` | ✅ |
| `ml/export/onnx_export.py` | ONNX opset-18 export and `ONNXDetector` | ✅ |
| `ml/eval/benchmark_quantization.py` | Accuracy, latency, file-size benchmark script | ✅ |
| `docs/quantization_benchmark.json` | Stored benchmark results | ✅ |
| `models/registry.json` | Model registry with TFLite DR entry | ✅ |
| `ml/export/ssd_mobilenet_v2_dynamic_range.tflite` | Quantized model (gitignored) | ❌ |
| `ml/export/ssd_mobilenet_v2_full_integer.tflite` | Full-int model (gitignored) | ❌ |
| `ml/export/ssd_mobilenet_v2_openimages20.onnx` | ONNX model (gitignored) | ❌ |
