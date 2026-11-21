# Model Quantization, Export, and Optimization

## Overview

This document details the export pipelines, optimization methodologies, runtime engines, and performance trade-offs for deploying the fine-tuned **SSD MobileNet V2** (Open Images 20-class) object detection model into production environments.

---

## 1. Export Formats and Feasibility

| Format | Target Runtime | Feasibility | File Size | Precision / Quantization | Notes |
|---|---|---|---|---|---|
| **PyTorch Baseline** | PyTorch / libtorch | Native | ~31.8 MB | Float32 | Baseline reference model |
| **TorchScript Trace** | PyTorch C++ / TorchScript | Native | ~32.1 MB | Float32 | Deterministic graph trace |
| **TFLite Dynamic-Range** | TFLite C++ / Micro / WebAssembly | Fully Feasible | ~7.8 MB | INT8 weights, Float32 activations | 3.9x compression, near-lossless mAP |
| **TFLite Full-Integer** | TFLite Edge TPU / Microcontrollers | Fully Feasible | ~7.8 MB | INT8 weights/inputs, INT32 biases | Calibrated via representative dataset |
| **ONNX Runtime** | ONNX Runtime (CPU / DirectML / TensorRT) | Fully Feasible | ~31.8 MB | Float32 (opset 18) | Cross-platform, hardware-accelerated |

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
- **Numerical Parity**: Maximum absolute difference against PyTorch float32 baseline is $< 10^{-4}$ ($0.00008$), confirming output parity.

### Usage:
```bash
python -m ml.export.onnx_export \
    --weights ml/export/ssd_mobilenet_v2_openimages20_weights.pth \
    --output ml/export/ssd_mobilenet_v2_openimages20.onnx \
    --verify
```
