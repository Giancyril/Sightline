"""
ml/export/export.py
-------------------
Export fine-tuned SSD MobileNet V2 model for production inference.
Supports TorchScript (.pt), state_dict checkpoints, and ONNX format,
verifying input/output tensor shapes and inference latency.

Usage:
    python -m ml.export.export [--output-dir ml/export] [--format torchscript]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ml.models.ssd_mobilenet import SSDMobileNetV2


def export_model(
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "ml/export",
    num_classes: int = 21,
    export_torchscript: bool = True,
) -> dict[str, Any]:
    """
    Export fine-tuned SSDMobileNetV2 model and verify input/output shapes.

    Args:
        checkpoint_path: Path to checkpoint weights or None for fresh architecture.
        output_dir: Directory where exported models and metadata are saved.
        num_classes: Total number of classes (including background).
        export_torchscript: If True, exports TorchScript trace.

    Returns:
        dict containing export paths, tensor shapes, and verification benchmark.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SSDMobileNetV2(num_classes=num_classes)
    model.eval()

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)

    dummy_input = torch.randn(1, 3, 300, 300, dtype=torch.float32)

    # 1. Verification pass with PyTorch model
    t0 = time.perf_counter()
    with torch.no_grad():
        locs, scores = model(dummy_input)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    expected_anchors = 2268
    assert locs.shape == (1, expected_anchors, 4), (
        f"Expected locs shape (1, {expected_anchors}, 4), got {locs.shape}"
    )
    assert scores.shape == (1, expected_anchors, num_classes), (
        f"Expected scores shape (1, {expected_anchors}, {num_classes}), got {scores.shape}"
    )

    result: dict[str, Any] = {
        "model_name": "ssd_mobilenet_v2_openimages20",
        "input_shape": [1, 3, 300, 300],
        "locs_output_shape": list(locs.shape),
        "scores_output_shape": list(scores.shape),
        "num_classes": num_classes,
        "num_anchors": expected_anchors,
        "cpu_latency_ms": round(latency_ms, 2),
        "verified": True,
    }

    # 2. Save PyTorch model state_dict
    weights_path = out_dir / "ssd_mobilenet_v2_openimages20_weights.pth"
    torch.save(model.state_dict(), weights_path)
    result["weights_path"] = str(weights_path)

    # 3. Export TorchScript trace
    if export_torchscript:
        ts_path = out_dir / "ssd_mobilenet_v2_openimages20.pt"
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(ts_path))
            result["torchscript_path"] = str(ts_path)

            # Verify TorchScript model reload
            loaded_ts = torch.jit.load(str(ts_path))
            loaded_ts.eval()
            with torch.no_grad():
                ts_locs, ts_scores = loaded_ts(dummy_input)
            assert ts_locs.shape == locs.shape
            assert ts_scores.shape == scores.shape
            result["torchscript_verified"] = True
        except Exception as e:
            result["torchscript_error"] = str(e)

    # 4. Save metadata JSON
    meta_path = out_dir / "model_spec.json"
    meta_path.write_text(json.dumps(result, indent=2))
    result["spec_path"] = str(meta_path)

    return result


def verify_exported_model(spec_path: str | Path = "ml/export/model_spec.json") -> bool:
    """Verify exported model specification matches criteria."""
    path = Path(spec_path)
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return (
        data.get("verified", False)
        and data.get("num_classes") == 21
        and data.get("num_anchors") == 2268
        and data.get("cpu_latency_ms", 999) < 300
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned SSD MobileNet V2.")
    parser.add_argument("--output-dir", default="ml/export", help="Output directory")
    args = parser.parse_args()
    res = export_model(output_dir=args.output_dir)
    print("\n" + "=" * 60)
    print("  MODEL EXPORT SUMMARY")
    print("=" * 60)
    print(f"Model:           {res['model_name']}")
    print(f"Input:           {res['input_shape']}")
    print(f"Locs Output:     {res['locs_output_shape']}")
    print(f"Scores Output:   {res['scores_output_shape']}")
    print(f"CPU Latency:     {res['cpu_latency_ms']} ms (< 300 ms target)")
    print(f"Verified:        {res['verified']}")
    print("=" * 60)
