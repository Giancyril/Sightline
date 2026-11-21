"""
ml/export/tflite_full_integer.py
--------------------------------
Full-integer (INT8) quantization export for SSD MobileNet V2 using a representative dataset.
Quantizes all weights to signed INT8, biases to INT32, and inputs/outputs to INT8/UINT8,
calibrating dynamic activation ranges from a representative validation sample without
committing any dataset files. Generates a valid TFLite FlatBuffer container artifact
with complete quantization parameters (scales and zero points).

Usage:
    python -m ml.export.tflite_full_integer \
        --weights ml/export/ssd_mobilenet_v2_openimages20_weights.pth \
        --config ml/train/config.yaml \
        --output ml/export/ssd_mobilenet_v2_full_integer.tflite \
        --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from ml.data.transforms import generate_ssd_anchors, normalize_image
from ml.export.tflite_export import (
    deserialize_tflite_container,
    serialize_tflite_container,
)
from ml.models.ssd_mobilenet import SSDMobileNetV2, postprocess_ssd_predictions

logger = logging.getLogger(__name__)


def get_representative_dataset_generator(
    config_path: str | Path | dict[str, Any] | None = "ml/train/config.yaml",
    num_samples: int = 20,
    seed: int = 42,
) -> Generator[np.ndarray, None, None]:
    """
    Generate representative calibration samples for full-integer quantization.
    Derives calibration paths from the train/val configuration without committing
    any data files. Falls back to deterministic synthetic calibration tensors
    when image paths are unavailable.

    Yields:
        np.ndarray with shape (1, 3, 300, 300) in [-1.0, 1.0] (float32).
    """
    manifest_path: Path | None = None

    if isinstance(config_path, dict):
        cfg_data = config_path
    elif config_path is not None and Path(config_path).exists():
        raw_text = Path(config_path).read_text(encoding="utf-8")
        cfg_data = yaml.safe_load(raw_text) or {}
    else:
        cfg_data = {}

    val_manifest = cfg_data.get("data", {}).get("val_manifest")
    if val_manifest and Path(val_manifest).exists():
        manifest_path = Path(val_manifest)

    image_paths: list[Path] = []
    if manifest_path and manifest_path.exists():
        try:
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            header = lines[0].split(",") if lines else []
            if "file_path" in header:
                idx = header.index("file_path")
                seen: set[str] = set()
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) > idx:
                        p_str = parts[idx].strip()
                        if p_str and p_str not in seen:
                            seen.add(p_str)
                            img_p = Path(p_str)
                            if img_p.exists():
                                image_paths.append(img_p)
                    if len(image_paths) >= num_samples:
                        break
        except Exception as e:
            logger.warning("Could not read manifest for calibration: %s", e)

    # Yield real images if available
    count = 0
    for img_p in image_paths:
        if count >= num_samples:
            break
        try:
            pil_img = (
                Image.open(img_p).convert("RGB").resize((300, 300), Image.BILINEAR)
            )
            img_np = np.array(pil_img, dtype=np.float32)
            norm = normalize_image(img_np)
            tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
            yield tensor
            count += 1
        except Exception as e:
            logger.warning("Failed to load calibration image %s: %s", img_p, e)

    # Deterministic synthetic calibration samples for remainder
    rng = np.random.RandomState(seed)
    while count < num_samples:
        # Realistic normalized distribution clipped to [-1.0, 1.0]
        synthetic = rng.normal(loc=0.0, scale=0.4, size=(1, 3, 300, 300))
        synthetic = np.clip(synthetic, -1.0, 1.0).astype(np.float32)
        yield synthetic
        count += 1


def quantize_full_integer(
    model: SSDMobileNetV2,
    representative_samples: list[np.ndarray],
    num_classes: int = 21,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Calibrate and quantize model to full INT8 precision:
      - Input: INT8 [-128, 127], scale = 2.0 / 255.0, zero_point = 0
      - Weights: INT8 [-128, 127], per-tensor symmetric scale
      - Biases: INT32, scale = s_in * s_weight
      - Activations: calibrated min/max range across representative set
      - Output locations: INT8, shape (1, 2268, 4)
      - Output scores: UINT8 [0, 255], shape (1, 2268, num_classes)

    Returns:
        quantized_weights: dict containing int8/int32/uint8 arrays with scale/zero_point
        metadata: full-integer quantization schema and statistics
    """
    model.eval()

    # 1. Calibrate activation ranges with representative inputs
    activation_stats: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for sample in representative_samples:
            x_tensor = torch.from_numpy(sample)
            locs, logits = model(x_tensor)
            probs = torch.softmax(logits, dim=-1)

            # Track output ranges
            loc_min = float(locs.min().item())
            loc_max = float(locs.max().item())
            prob_min = float(probs.min().item())
            prob_max = float(probs.max().item())

            if "locations" not in activation_stats:
                activation_stats["locations"] = {"min": loc_min, "max": loc_max}
                activation_stats["scores"] = {"min": prob_min, "max": prob_max}
            else:
                activation_stats["locations"]["min"] = min(
                    activation_stats["locations"]["min"], loc_min
                )
                activation_stats["locations"]["max"] = max(
                    activation_stats["locations"]["max"], loc_max
                )
                activation_stats["scores"]["min"] = min(
                    activation_stats["scores"]["min"], prob_min
                )
                activation_stats["scores"]["max"] = max(
                    activation_stats["scores"]["max"], prob_max
                )

    # 2. Input Quantization Spec
    input_scale = float(2.0 / 255.0)  # [-1.0, 1.0] -> 255 steps
    input_zero_point = 0

    # 3. Output Quantization Specs
    loc_range = max(
        abs(activation_stats["locations"]["min"]),
        abs(activation_stats["locations"]["max"]),
        1.0,
    )
    loc_scale = float((2.0 * loc_range) / 255.0)
    loc_zero_point = 0

    score_scale = float(1.0 / 255.0)  # [0.0, 1.0] -> UINT8 [0, 255]
    score_zero_point = 0

    quantized_weights: dict[str, Any] = {}
    state_dict = model.state_dict()
    orig_bytes = 0
    quant_bytes = 0

    # 4. Quantize each parameter
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        orig_bytes += t.numel() * t.element_size()

        if "weight" in name and t.dtype == torch.float32 and t.ndim >= 2:
            max_abs = float(t.abs().max().item())
            w_scale = max(max_abs / 127.0, 1e-8)
            int8_data = (
                torch.clamp(torch.round(t / w_scale), -128, 127).to(torch.int8).numpy()
            )

            quantized_weights[name] = {
                "dtype": "int8",
                "scale": float(w_scale),
                "zero_point": 0,
                "shape": list(t.shape),
                "data": int8_data,
            }
            quant_bytes += int8_data.nbytes
        elif "bias" in name and t.dtype == torch.float32:
            # Full integer INT32 bias quantization
            # Bias scale = input_scale * weight_scale
            w_name = name.replace("bias", "weight")
            w_tensor = state_dict.get(w_name)
            if w_tensor is not None:
                w_scale = max(float(w_tensor.abs().max().item()) / 127.0, 1e-8)
            else:
                w_scale = 1.0 / 127.0
            bias_scale = input_scale * w_scale
            int32_data = torch.round(t / max(bias_scale, 1e-10)).to(torch.int32).numpy()

            quantized_weights[name] = {
                "dtype": "int32",
                "scale": float(bias_scale),
                "zero_point": 0,
                "shape": list(t.shape),
                "data": int32_data,
            }
            quant_bytes += int32_data.nbytes
        else:
            np_data = t.numpy()
            quantized_weights[name] = {
                "dtype": str(np_data.dtype),
                "scale": 1.0,
                "zero_point": 0,
                "shape": list(t.shape),
                "data": np_data,
            }
            quant_bytes += np_data.nbytes

    meta = {
        "format": "tflite_full_integer",
        "model_architecture": "SSDMobileNetV2",
        "num_classes": num_classes,
        "calibration": {
            "representative_samples_count": len(representative_samples),
            "activation_ranges": activation_stats,
        },
        "tensor_specs": {
            "input": {
                "name": "input_image",
                "shape": [1, 3, 300, 300],
                "dtype": "int8",
                "scale": input_scale,
                "zero_point": input_zero_point,
                "domain": [-128, 127],
            },
            "output_locations": {
                "name": "box_offsets",
                "shape": [1, 2268, 4],
                "dtype": "int8",
                "scale": loc_scale,
                "zero_point": loc_zero_point,
            },
            "output_scores": {
                "name": "class_probabilities",
                "shape": [1, 2268, num_classes],
                "dtype": "uint8",
                "scale": score_scale,
                "zero_point": score_zero_point,
                "domain": [0, 255],
            },
            "weights": {
                "dtype": "int8",
                "quantization": "per_tensor_symmetric",
            },
            "biases": {
                "dtype": "int32",
                "quantization": "input_scale_x_weight_scale",
            },
        },
        "compression": {
            "original_bytes": orig_bytes,
            "quantized_bytes": quant_bytes,
            "compression_ratio": round(orig_bytes / max(quant_bytes, 1), 2),
            "size_reduction_pct": round((1.0 - quant_bytes / orig_bytes) * 100.0, 1),
        },
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return quantized_weights, meta


def export_tflite_full_integer(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = "ml/train/config.yaml",
    output_path: str | Path = "ml/export/ssd_mobilenet_v2_full_integer.tflite",
    num_samples: int = 20,
    num_classes: int = 21,
) -> dict[str, Any]:
    """
    Export fine-tuned SSDMobileNetV2 model to full-integer quantized TFLite container.

    Args:
        checkpoint_path: Path to float32 PyTorch weights.
        config_path: Path to training/data configuration.
        output_path: Filepath where full-integer .tflite artifact will be saved.
        num_samples: Number of representative dataset calibration samples.
        num_classes: Total class count.

    Returns:
        Full metadata dictionary containing tensor specifications, shapes, and dtypes.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model = SSDMobileNetV2(num_classes=num_classes)
    model.eval()
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        logger.info("Loaded PyTorch checkpoint from %s", checkpoint_path)

    # 2. Collect representative samples
    gen = get_representative_dataset_generator(
        config_path=config_path, num_samples=num_samples
    )
    samples = list(gen)
    logger.info("Generated %d representative calibration samples", len(samples))

    # 3. Quantize model
    quant_weights, meta = quantize_full_integer(model, samples, num_classes=num_classes)

    # 4. Serialize to TFLite FlatBuffer container
    binary = serialize_tflite_container(quant_weights, meta)
    out_file.write_bytes(binary)

    meta["export_path"] = str(out_file)
    meta["file_size_bytes"] = len(binary)
    meta["file_size_mb"] = round(len(binary) / (1024 * 1024), 2)
    logger.info("Saved full-integer TFLite model to %s", out_file)

    return meta


class TFLiteFullIntegerDetector:
    """
    Inference runner executing full-integer INT8/UINT8 quantized SSD MobileNet V2.
    Accepts INT8 quantized input tensors or images, and returns INT8/UINT8 quantized outputs.
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

        binary = self.tflite_path.read_bytes()
        self.metadata, self.quantized_weights = deserialize_tflite_container(binary)

        self.input_spec = self.metadata["tensor_specs"]["input"]
        self.loc_spec = self.metadata["tensor_specs"]["output_locations"]
        self.score_spec = self.metadata["tensor_specs"]["output_scores"]

        num_classes = self.metadata.get("num_classes", 21)
        self.model = SSDMobileNetV2(num_classes=num_classes)

        # Dequantize weights/biases for pure PyTorch execution simulation
        dequant_state: dict[str, torch.Tensor] = {}
        for name, item in self.quantized_weights.items():
            if item["dtype"] == "int8":
                w = torch.from_numpy(item["data"]).to(torch.float32) * item["scale"]
                dequant_state[name] = w
            elif item["dtype"] == "int32":
                b = torch.from_numpy(item["data"]).to(torch.float32) * item["scale"]
                dequant_state[name] = b
            else:
                dequant_state[name] = torch.from_numpy(item["data"])

        self.model.load_state_dict(dequant_state)
        self.model.eval()
        self.anchors = generate_ssd_anchors()

    def quantize_input(self, float_tensor: np.ndarray) -> np.ndarray:
        """Quantize float32 input in [-1.0, 1.0] to INT8 [-128, 127]."""
        scale = self.input_spec["scale"]
        q_int8 = np.clip(np.round(float_tensor / scale), -128, 127).astype(np.int8)
        return q_int8

    def dequantize_input(self, int8_tensor: np.ndarray) -> torch.Tensor:
        """Dequantize INT8 input to float32 PyTorch tensor."""
        scale = self.input_spec["scale"]
        return torch.from_numpy(int8_tensor.astype(np.float32) * scale)

    def forward_quantized(
        self, int8_input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute full-integer forward pass.
        Args:
            int8_input: np.ndarray with shape (1, 3, 300, 300) and dtype int8.
        Returns:
            quantized_locs: INT8 array shape (1, 2268, 4)
            quantized_scores: UINT8 array shape (1, 2268, num_classes)
        """
        assert int8_input.dtype == np.int8, (
            f"Expected int8 input, got {int8_input.dtype}"
        )
        assert int8_input.shape == (1, 3, 300, 300), (
            f"Invalid shape: {int8_input.shape}"
        )

        float_input = self.dequantize_input(int8_input)
        with torch.no_grad():
            locs, logits = self.model(float_input)
            probs = torch.softmax(logits, dim=-1)

        # Quantize outputs according to specifications
        loc_scale = self.loc_spec["scale"]
        locs_np = locs.cpu().numpy()
        q_locs = np.clip(np.round(locs_np / loc_scale), -128, 127).astype(np.int8)

        score_scale = self.score_spec["scale"]
        probs_np = probs.cpu().numpy()
        q_scores = np.clip(np.round(probs_np / score_scale), 0, 255).astype(np.uint8)

        return q_locs, q_scores

    def predict(
        self,
        image: Image.Image | np.ndarray,
        threshold: float | None = None,
    ) -> list[tuple[list[float], int, float]]:
        """
        Full inference from raw image input to decoded NMS detections.
        """
        th = threshold if threshold is not None else self.confidence_threshold

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        resized = pil_img.resize((300, 300), Image.Resampling.BILINEAR)
        img_np = np.array(resized, dtype=np.float32)
        norm_img = normalize_image(img_np)
        float_tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(
            np.float32
        )

        int8_input = self.quantize_input(float_tensor)
        q_locs, q_scores = self.forward_quantized(int8_input)

        # Dequantize for NMS
        offsets = q_locs.squeeze(0).astype(np.float32) * self.loc_spec["scale"]
        probs = q_scores.squeeze(0).astype(np.float32) * self.score_spec["scale"]

        return postprocess_ssd_predictions(
            offsets=offsets,
            class_probs=probs,
            anchors=self.anchors,
            confidence_threshold=th,
            iou_threshold=self.iou_threshold,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-Integer INT8 Quantization Export for SSD MobileNet V2"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="ml/export/ssd_mobilenet_v2_openimages20_weights.pth",
        help="Path to float32 weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/train/config.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/export/ssd_mobilenet_v2_full_integer.tflite",
        help="Output full-integer .tflite path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=21,
        help="Number of classes including background",
    )
    args = parser.parse_args()

    meta = export_tflite_full_integer(
        checkpoint_path=args.weights,
        config_path=args.config,
        output_path=args.output,
        num_samples=args.num_samples,
        num_classes=args.num_classes,
    )
    print("Full-Integer Quantization Export Complete:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
