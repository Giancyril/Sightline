"""
ml/export/onnx_export.py
------------------------
ONNX export and inference runtime for SSD MobileNet V2.
Exports PyTorch weights to an optimized ONNX computational graph (opset 18),
validates graph schema and node topologies using the ONNX model checker,
and executes high-throughput CPU inference using ONNX Runtime.

Usage:
    python -m ml.export.onnx_export \
        --weights ml/export/ssd_mobilenet_v2_openimages20_weights.pth \
        --output ml/export/ssd_mobilenet_v2_openimages20.onnx \
        --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None  # type: ignore[assignment]
    ort = None  # type: ignore[assignment]

from ml.data.transforms import generate_ssd_anchors, normalize_image
from ml.models.ssd_mobilenet import SSDMobileNetV2, postprocess_ssd_predictions

logger = logging.getLogger(__name__)


def export_onnx_model(
    checkpoint_path: str | Path | None = None,
    output_path: str | Path = "ml/export/ssd_mobilenet_v2_openimages20.onnx",
    num_classes: int = 21,
    opset_version: int = 18,
) -> dict[str, Any]:
    """
    Export SSDMobileNetV2 model to ONNX format and validate model graph.

    Args:
        checkpoint_path: Path to float32 PyTorch weights or None for uninitialized weights.
        output_path: Target path for the .onnx model file.
        num_classes: Total class count including background (default 21).
        opset_version: Target ONNX operator set version (default 18).

    Returns:
        dict containing export paths, graph input/output shapes, and model metadata.
    """
    if onnx is None:
        raise ImportError(
            "ONNX export requires 'onnx' and 'onnxruntime'. "
            "Install with: pip install onnx onnxruntime"
        )
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize PyTorch model
    model = SSDMobileNetV2(num_classes=num_classes)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        logger.info("Loaded PyTorch checkpoint from %s", checkpoint_path)

    # 2. Run ONNX export with dynamic batch dimension
    dummy_input = torch.randn(1, 3, 300, 300, dtype=torch.float32)
    input_names = ["image"]
    output_names = ["locations", "scores"]
    dynamic_axes = {
        "image": {0: "batch_size"},
        "locations": {0: "batch_size"},
        "scores": {0: "batch_size"},
    }

    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    export_duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # 3. Validate ONNX model graph structure
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    logger.info("Validated ONNX model structure: %s", out_file)

    file_size_bytes = out_file.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    meta = {
        "format": "onnx",
        "opset_version": opset_version,
        "model_architecture": "SSDMobileNetV2",
        "num_classes": num_classes,
        "input_tensor": {
            "name": "image",
            "shape": [1, 3, 300, 300],
            "dtype": "float32",
            "range": [-1.0, 1.0],
        },
        "output_tensors": {
            "locations": {"shape": [1, 2268, 4], "dtype": "float32"},
            "scores": {"shape": [1, 2268, num_classes], "dtype": "float32"},
        },
        "export_duration_ms": export_duration_ms,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "export_path": str(out_file),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return meta


class ONNXDetector:
    """
    High-performance detector running SSD MobileNet V2 via ONNX Runtime.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        if ort is None:
            raise ImportError(
                "ONNXDetector requires 'onnxruntime'. "
                "Install with: pip install onnxruntime"
            )
        self.onnx_path = Path(onnx_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = generate_ssd_anchors()

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute ONNX forward pass.
        Args:
            x: np.ndarray float32 of shape (B, 3, 300, 300) in [-1.0, 1.0]
        Returns:
            locations: np.ndarray of shape (B, 2268, 4)
            scores: np.ndarray of shape (B, 2268, num_classes)
        """
        assert x.dtype == np.float32, f"Expected float32 input, got {x.dtype}"
        outputs = self.session.run(None, {self.input_name: x})
        return outputs[0], outputs[1]

    def predict(
        self,
        image: Image.Image | np.ndarray,
        threshold: float | None = None,
    ) -> list[tuple[list[float], int, float]]:
        """
        Execute end-to-end detection on a PIL image or numpy array.
        """
        th = threshold if threshold is not None else self.confidence_threshold

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        resized = pil_img.resize((300, 300), Image.Resampling.BILINEAR)
        img_np = np.array(resized, dtype=np.float32)
        norm_img = normalize_image(img_np)
        input_tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(
            np.float32
        )

        locs, logits = self.forward(input_tensor)

        # Softmax over logits for scores
        logits_tensor = torch.from_numpy(logits)
        probs = torch.softmax(logits_tensor, dim=-1).squeeze(0).numpy()
        offsets = locs[0]

        return postprocess_ssd_predictions(
            offsets=offsets,
            class_probs=probs,
            anchors=self.anchors,
            confidence_threshold=th,
            iou_threshold=self.iou_threshold,
        )


def verify_onnx_parity(
    original_model: SSDMobileNetV2,
    onnx_detector: ONNXDetector,
    test_inputs: list[np.ndarray],
    atol: float = 1e-3,
) -> dict[str, Any]:
    """
    Verify numerical parity between PyTorch baseline and ONNX Runtime.
    """
    original_model.eval()
    max_loc_err = 0.0
    max_score_err = 0.0

    for x_np in test_inputs:
        x_torch = torch.from_numpy(x_np)
        with torch.no_grad():
            pt_locs, pt_scores = original_model(x_torch)

        ox_locs, ox_scores = onnx_detector.forward(x_np)

        loc_diff = float(np.max(np.abs(pt_locs.numpy() - ox_locs)))
        score_diff = float(np.max(np.abs(pt_scores.numpy() - ox_scores)))

        max_loc_err = max(max_loc_err, loc_diff)
        max_score_err = max(max_score_err, score_diff)

    passed = max_loc_err <= atol and max_score_err <= atol
    return {
        "passed": bool(passed),
        "max_loc_difference": round(max_loc_err, 6),
        "max_score_difference": round(max_score_err, 6),
        "tolerance": atol,
        "samples_evaluated": len(test_inputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SSD MobileNet V2 to ONNX format"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="ml/export/ssd_mobilenet_v2_openimages20_weights.pth",
        help="Path to float32 weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/export/ssd_mobilenet_v2_openimages20.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX operator set version",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify parity against PyTorch baseline",
    )
    args = parser.parse_args()

    meta = export_onnx_model(
        checkpoint_path=args.weights,
        output_path=args.output,
        opset_version=args.opset,
    )
    print("ONNX Export Successful:")
    print(json.dumps(meta, indent=2))

    if args.verify:
        print("\nVerifying numerical parity against PyTorch model...")
        orig_model = SSDMobileNetV2(num_classes=21)
        if Path(args.weights).exists():
            orig_model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        detector = ONNXDetector(args.output)
        test_inputs = [
            np.random.randn(1, 3, 300, 300).astype(np.float32),
            np.zeros((1, 3, 300, 300), dtype=np.float32),
        ]
        verification = verify_onnx_parity(orig_model, detector, test_inputs)
        print("Parity verification result:")
        print(json.dumps(verification, indent=2))
        assert verification["passed"], (
            f"ONNX parity verification failed: {verification}"
        )


if __name__ == "__main__":
    main()
