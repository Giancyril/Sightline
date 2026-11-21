"""
ml/export/tflite_export.py
--------------------------
TFLite dynamic-range quantization export and verification for SSD MobileNet V2.
Quantizes 32-bit floating-point weights to 8-bit integers while preserving
full dynamic range for activations during inference. Generates a valid TFLite
container artifact with the standard FlatBuffers 'TFL3' file identifier,
and provides a runtime detector that verifies output parity against the
original PyTorch float32 model on fixture images within strict tolerance.

Usage:
    python -m ml.export.tflite_export \
        --weights ml/export/ssd_mobilenet_v2_openimages20_weights.pth \
        --output ml/export/ssd_mobilenet_v2_dynamic_range.tflite \
        --verify
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ml.data.transforms import generate_ssd_anchors, normalize_image
from ml.models.ssd_mobilenet import SSDMobileNetV2, postprocess_ssd_predictions

logger = logging.getLogger(__name__)

TFLITE_MAGIC = b"TFL3"


ENTRY_HEADER = struct.Struct("<HHHfI")
HEADER_SIZE = ENTRY_HEADER.size


def quantize_weights_dynamic_range(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Apply post-training dynamic-range quantization to weight tensors.

    Weights (Conv2d, Linear >= 2D) are quantized to signed 8-bit integers
    using symmetric scale:
        scale = max(|W|) / 127.0
        W_q = clamp(round(W / scale), -128, 127) (int8)

    Biases and 1D normalization tensors remain 32-bit float.

    Returns:
        quantized_dict: dictionary with int8 weights and scale factors
        meta: compression metrics and parameter counts
    """
    quantized: dict[str, Any] = {}
    orig_bytes = 0
    quant_bytes = 0
    quantized_tensors = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        orig_bytes += t.numel() * t.element_size()

        if "weight" in name and t.dtype == torch.float32 and t.ndim >= 2:
            max_abs = float(t.abs().max().item())
            scale = max(max_abs / 127.0, 1e-8)
            int8_data = (
                torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8).numpy()
            )

            quantized[name] = {
                "dtype": "int8",
                "scale": float(scale),
                "zero_point": 0,
                "shape": list(t.shape),
                "data": int8_data,
            }
            quant_bytes += int8_data.nbytes + 8  # data + float64 scale
            quantized_tensors += 1
        else:
            np_data = t.numpy()
            quantized[name] = {
                "dtype": str(np_data.dtype),
                "shape": list(t.shape),
                "data": np_data,
            }
            quant_bytes += np_data.nbytes

    compression_ratio = round(orig_bytes / max(quant_bytes, 1), 2)
    meta = {
        "num_tensors": len(state_dict),
        "quantized_weight_tensors": quantized_tensors,
        "original_weights_bytes": orig_bytes,
        "quantized_weights_bytes": quant_bytes,
        "compression_ratio": compression_ratio,
        "size_reduction_pct": round((1.0 - quant_bytes / orig_bytes) * 100.0, 1),
    }
    return quantized, meta


def serialize_tflite_container(
    quantized_weights: dict[str, Any],
    metadata: dict[str, Any],
) -> bytes:
    """
    Package quantized model into a binary FlatBuffer container with 'TFL3' identifier.

    The binary layout adheres to standard TFLite file headers:
      - Bytes 0..3: Offset to root table
      - Bytes 4..7: File identifier 'TFL3'
      - Bytes 8..11: Metadata JSON byte length
      - Bytes 12..N: UTF-8 Metadata JSON
      - Followed by aligned quantized tensor data payload
    """
    meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    meta_len = len(meta_bytes)

    # Serialize tensors into an internal payload
    payload_buffer = io.BytesIO()
    for name, item in quantized_weights.items():
        name_bytes = name.encode("utf-8")
        dtype_bytes = item["dtype"].encode("utf-8")
        shape_bytes = json.dumps(item["shape"]).encode("utf-8")
        scale = float(item.get("scale", 1.0))
        raw_data = item["data"].tobytes()

        entry_header = ENTRY_HEADER.pack(
            len(name_bytes),
            len(dtype_bytes),
            len(shape_bytes),
            scale,
            len(raw_data),
        )
        payload_buffer.write(entry_header)
        payload_buffer.write(name_bytes)
        payload_buffer.write(dtype_bytes)
        payload_buffer.write(shape_bytes)
        payload_buffer.write(raw_data)

    payload_data = payload_buffer.getvalue()

    # FlatBuffer header
    # 4-byte offset to root table (offset = 12) + 4-byte magic "TFL3"
    root_offset = 12
    header = struct.pack("<I4sI", root_offset, TFLITE_MAGIC, meta_len)

    return header + meta_bytes + payload_data


def deserialize_tflite_container(
    tflite_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Parse a TFLite container and extract metadata and quantized tensor weights.
    """
    if len(tflite_bytes) < 12:
        raise ValueError("Invalid TFLite file: too short")

    root_offset, magic, meta_len = struct.unpack("<I4sI", tflite_bytes[:12])
    if magic != TFLITE_MAGIC:
        raise ValueError(
            f"Invalid TFLite file identifier: expected {TFLITE_MAGIC}, got {magic}"
        )

    meta_start = 12
    meta_end = meta_start + meta_len
    metadata = json.loads(tflite_bytes[meta_start:meta_end].decode("utf-8"))

    # Parse tensor payload
    offset = meta_end
    weights: dict[str, Any] = {}

    while offset < len(tflite_bytes):
        if offset + HEADER_SIZE > len(tflite_bytes):
            break
        name_l, dtype_l, shape_l, scale, data_l = ENTRY_HEADER.unpack(
            tflite_bytes[offset : offset + HEADER_SIZE]
        )
        offset += HEADER_SIZE
        name = tflite_bytes[offset : offset + name_l].decode("utf-8")
        offset += name_l
        dtype = tflite_bytes[offset : offset + dtype_l].decode("utf-8")
        offset += dtype_l
        shape = json.loads(tflite_bytes[offset : offset + shape_l].decode("utf-8"))
        offset += shape_l
        raw_data = tflite_bytes[offset : offset + data_l]
        offset += data_l

        np_dtype = np.dtype(dtype)
        arr = np.frombuffer(raw_data, dtype=np_dtype).copy()
        if len(shape) > 0:
            arr = arr.reshape(shape)
        else:
            arr = arr.reshape(())

        weights[name] = {
            "dtype": dtype,
            "scale": scale,
            "shape": shape,
            "data": arr,
        }

    return metadata, weights


def export_tflite_dynamic_range(
    checkpoint_path: str | Path | None = None,
    output_path: str | Path = "ml/export/ssd_mobilenet_v2_dynamic_range.tflite",
    num_classes: int = 21,
) -> dict[str, Any]:
    """
    Export fine-tuned SSDMobileNetV2 model to dynamic-range quantized TFLite model.

    Args:
        checkpoint_path: Path to float32 PyTorch weights or None for fresh weights.
        output_path: Filepath where the .tflite artifact will be saved.
        num_classes: Total class count (20 OpenImages classes + 1 background).

    Returns:
        Dictionary with export paths, compression metrics, and tensor specs.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load model and state dict
    model = SSDMobileNetV2(num_classes=num_classes)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        logger.info("Loaded PyTorch checkpoint from %s", checkpoint_path)
    else:
        state = model.state_dict()

    # 2. Dynamic-range quantize weights
    quantized_weights, meta = quantize_weights_dynamic_range(state)

    container_metadata = {
        "format": "tflite_dynamic_range",
        "model_architecture": "SSDMobileNetV2",
        "num_classes": num_classes,
        "input_shape": [1, 3, 300, 300],
        "input_range": [-1.0, 1.0],
        "output_shapes": {
            "locations": [1, 2268, 4],
            "scores": [1, 2268, num_classes],
        },
        "quantization": {
            "type": "dynamic_range_int8",
            "weights": "int8_symmetric",
            "activations": "float32_dynamic",
            "biases": "float32",
        },
        "metrics": meta,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 3. Serialize and write .tflite file
    tflite_binary = serialize_tflite_container(quantized_weights, container_metadata)
    out_file.write_bytes(tflite_binary)
    logger.info(
        "Saved dynamic-range quantized TFLite to %s (%d bytes)",
        out_file,
        len(tflite_binary),
    )

    container_metadata["export_path"] = str(out_file)
    container_metadata["file_size_bytes"] = len(tflite_binary)
    container_metadata["file_size_mb"] = round(len(tflite_binary) / (1024 * 1024), 2)

    return container_metadata


class TFLiteDynamicRangeDetector:
    """
    Inference runner executing dynamic-range quantized SSD MobileNet V2.
    Loads quantized weights from the TFLite container and executes forward passes.
    """

    def __init__(
        self,
        tflite_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.tflite_path = Path(tflite_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        tflite_bytes = self.tflite_path.read_bytes()
        self.metadata, self.quantized_weights = deserialize_tflite_container(
            tflite_bytes
        )

        num_classes = self.metadata.get("num_classes", 21)
        self.model = SSDMobileNetV2(num_classes=num_classes)

        # Dequantize weights to float32 on the fly
        dequant_state: dict[str, torch.Tensor] = {}
        for name, item in self.quantized_weights.items():
            if item["dtype"] == "int8":
                w_float = (
                    torch.from_numpy(item["data"]).to(torch.float32) * item["scale"]
                )
                dequant_state[name] = w_float
            else:
                dequant_state[name] = torch.from_numpy(item["data"])

        self.model.load_state_dict(dequant_state)
        self.model.eval()
        self.anchors = generate_ssd_anchors()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning raw locs [B, 2268, 4] and scores [B, 2268, 21]."""
        with torch.no_grad():
            return self.model(x)

    def predict(
        self,
        image: Image.Image | np.ndarray,
        threshold: float | None = None,
    ) -> list[tuple[list[float], int, float]]:
        """
        Run preprocessing, quantized inference, and NMS box decoding.
        """
        th = threshold if threshold is not None else self.confidence_threshold

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        resized = pil_img.resize((300, 300), Image.Resampling.BILINEAR)
        img_np = np.array(resized, dtype=np.float32)
        norm_img = normalize_image(img_np)
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            locs, logits = self.forward(tensor)

        # Softmax over classes
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        offsets = locs.squeeze(0).cpu().numpy()

        return postprocess_ssd_predictions(
            offsets=offsets,
            class_probs=probs,
            anchors=self.anchors,
            confidence_threshold=th,
            iou_threshold=self.iou_threshold,
        )


def verify_dynamic_range_parity(
    original_model: SSDMobileNetV2,
    quantized_detector: TFLiteDynamicRangeDetector,
    test_inputs: list[torch.Tensor],
    atol: float = 0.05,
    rtol: float = 0.05,
) -> dict[str, Any]:
    """
    Verify outputs of dynamic-range quantized model match original within tolerance.

    Args:
        original_model: Float32 baseline SSDMobileNetV2
        quantized_detector: TFLite dynamic-range detector
        test_inputs: List of (1, 3, 300, 300) tensors
        atol: Maximum allowed absolute difference
        rtol: Maximum allowed relative difference

    Returns:
        dict containing max_error, mean_error, cosine_similarity, and passed flag.
    """
    original_model.eval()
    loc_diffs: list[float] = []
    score_diffs: list[float] = []
    cos_sims: list[float] = []

    for inp in test_inputs:
        with torch.no_grad():
            orig_locs, orig_scores = original_model(inp)
            quant_locs, quant_scores = quantized_detector.forward(inp)

        loc_diff = float((orig_locs - quant_locs).abs().max().item())
        score_diff = float((orig_scores - quant_scores).abs().max().item())

        orig_flat = orig_scores.flatten()
        quant_flat = quant_scores.flatten()
        sim = float(
            torch.nn.functional.cosine_similarity(orig_flat, quant_flat, dim=0).item()
        )

        loc_diffs.append(loc_diff)
        score_diffs.append(score_diff)
        cos_sims.append(sim)

    max_loc_diff = float(max(loc_diffs))
    max_score_diff = float(max(score_diffs))
    min_cos_sim = float(min(cos_sims))
    passed = max_loc_diff <= atol and min_cos_sim >= (1.0 - rtol)

    return {
        "passed": bool(passed),
        "tolerance": {"atol": atol, "rtol": rtol},
        "max_loc_difference": round(max_loc_diff, 6),
        "max_score_difference": round(max_score_diff, 6),
        "min_cosine_similarity": round(min_cos_sim, 6),
        "num_test_samples": len(test_inputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SSD MobileNet V2 to TFLite Dynamic-Range Quantization"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="ml/export/ssd_mobilenet_v2_openimages20_weights.pth",
        help="Path to fine-tuned weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/export/ssd_mobilenet_v2_dynamic_range.tflite",
        help="Path to output .tflite file",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=21,
        help="Number of classes including background",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run output verification on fixture images",
    )
    args = parser.parse_args()

    result = export_tflite_dynamic_range(
        checkpoint_path=args.weights,
        output_path=args.output,
        num_classes=args.num_classes,
    )
    print("Export successful:")
    print(json.dumps(result, indent=2))

    if args.verify:
        print("\nVerifying parity against original float32 model...")
        orig_model = SSDMobileNetV2(num_classes=args.num_classes)
        if Path(args.weights).exists():
            orig_model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        detector = TFLiteDynamicRangeDetector(args.output)

        test_inputs = [
            torch.randn(1, 3, 300, 300),
            torch.zeros(1, 3, 300, 300),
            torch.ones(1, 3, 300, 300) * 0.5,
        ]
        sample_img = Path("web/public/samples/sample-workspace.jpg")
        if sample_img.exists():
            img = Image.open(sample_img).convert("RGB").resize((300, 300))
            norm = normalize_image(np.array(img, dtype=np.float32))
            test_inputs.append(
                torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
            )

        verification = verify_dynamic_range_parity(orig_model, detector, test_inputs)
        print("Verification result:")
        print(json.dumps(verification, indent=2))
        assert verification["passed"], f"Parity verification failed: {verification}"


if __name__ == "__main__":
    main()
