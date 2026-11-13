"""
api/services/detector.py
------------------------
Singleton Inference Service for SSD MobileNet V2.
Handles image validation (size cap, format/magic bytes, EXIF stripping),
pre-processing, PyTorch model execution, post-processing (anchor decoding + NMS),
and coordinate conversion.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from ml.data.classes import CLASS_BY_ID, CLASSES, ClassInfo
from ml.data.transforms import generate_ssd_anchors
from ml.models.coco_labels import COCO_CLASSES, map_coco_id_to_subset
from ml.models.registry import ModelEntry, get_model_entry
from ml.models.ssd_mobilenet import SSDMobileNetV2, postprocess_ssd_predictions

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MAGIC_SIGNATURES = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),  # WebP RIFF container
]


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnknownModelError(KeyError):
    """Raised when a requested model ID is not found in the registry."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Unknown model identifier '{model_id}'.")
        self.model_id = model_id
        self.status_code = 404


class DetectorService:
    """Thread-safe detector service executing SSD MobileNet V2 inference."""

    def __init__(
        self,
        model_id: str = "fine-tuned",
        model_name: str = "SSD MobileNet V2 (Open Images 20-Class)",
        model_version: str = "v1.0.0",
        num_classes: int = 21,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.model_version = model_version
        self.num_classes = num_classes
        self.device = torch.device("cpu")
        self.model = SSDMobileNetV2(num_classes=self.num_classes).to(self.device)
        self.model.eval()

        self.class_map: dict[int, ClassInfo] = {cls.id: cls for cls in CLASSES}
        self.anchors = generate_ssd_anchors()

        # Load weights if available
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)

        self._warmup()

    def _warmup(self) -> None:
        """Run a single forward pass to prime PyTorch memory buffers."""
        dummy = torch.zeros(1, 3, 300, 300, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.model(dummy)

    def validate_image_bytes(self, data: bytes) -> str:
        """
        Validate image data size and magic byte signatures.
        Returns detected MIME type.
        """
        if len(data) == 0:
            raise ImageValidationError("Empty image payload received.", status_code=400)

        if len(data) > MAX_FILE_SIZE_BYTES:
            raise ImageValidationError(
                f"Image size ({len(data) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of 10 MB.",
                status_code=413,
            )

        # Check magic bytes
        for signature, mime in ALLOWED_MAGIC_SIGNATURES:
            if data.startswith(signature):
                if mime == "image/webp" and len(data) >= 12 and data[8:12] != b"WEBP":
                    continue
                return mime

        raise ImageValidationError(
            "Unsupported file format. Please upload JPEG, PNG, or WebP images.",
            status_code=415,
        )

    def decode_and_sanitize(self, data: bytes) -> tuple[Image.Image, int, int]:
        """
        Safely open image, transpose EXIF orientation, and convert to RGB.
        Returns (sanitized_pil_image, original_width, original_height).
        """
        try:
            pil_img = Image.open(io.BytesIO(data))
            # Auto-orient based on EXIF tag (strip orientation distortion)
            pil_img = ImageOps.exif_transpose(pil_img)
            orig_w, orig_h = pil_img.size
            # Convert to standard RGB 8-bit
            rgb_img = pil_img.convert("RGB")
            return rgb_img, orig_w, orig_h
        except Exception as e:
            raise ImageValidationError(
                f"Corrupt or unreadable image: {e}", status_code=400
            ) from e

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
    ) -> dict[str, Any]:
        """
        Run object detection on raw image bytes.

        Args:
            image_bytes: Raw binary image payload.
            confidence_threshold: Minimum prediction score in [0.0, 1.0].
            nms_iou_threshold: Overlap threshold for NMS in [0.0, 1.0].

        Returns:
            Structured dict matching DetectionResponse schema.
        """
        start_total = time.perf_counter()

        # 1. Validation
        self.validate_image_bytes(image_bytes)

        # 2. Decoding & Sanitization
        img, orig_w, orig_h = self.decode_and_sanitize(image_bytes)

        # 3. Preprocessing
        # Resize to 300x300 and normalize to [-1.0, 1.0]
        resized = img.resize((300, 300), Image.Resampling.BILINEAR)
        img_np = np.array(resized, dtype=np.float32) / 127.5 - 1.0
        # Shape: (1, 3, 300, 300)
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 4. Model Inference
        start_inference = time.perf_counter()
        with torch.no_grad():
            pred_offsets, class_logits = self.model(tensor)
            # Softmax class probabilities
            probs = torch.softmax(class_logits, dim=-1).squeeze(0).cpu().numpy()
            offsets = pred_offsets.squeeze(0).cpu().numpy()
        inference_ms = (time.perf_counter() - start_inference) * 1000.0

        # 5. Postprocessing (NMS and Decoding)
        raw_detections = postprocess_ssd_predictions(
            offsets=offsets,
            class_probs=probs,
            anchors=self.anchors,
            confidence_threshold=confidence_threshold,
            iou_threshold=nms_iou_threshold,
            max_detections=50,
        )

        # 6. Format Detections
        formatted_detections = []
        for box, class_id, score in raw_detections:
            ymin, xmin, ymax, xmax = [float(np.clip(v, 0.0, 1.0)) for v in box]

            # Pixel bounding box
            px_ymin = int(round(ymin * orig_h))
            px_xmin = int(round(xmin * orig_w))
            px_ymax = int(round(ymax * orig_h))
            px_xmax = int(round(xmax * orig_w))
            px_w = max(0, px_xmax - px_xmin)
            px_h = max(0, px_ymax - px_ymin)

            c_info = self.class_map.get(class_id)
            if c_info:
                c_name = c_info.name
                c_hex = c_info.color_hex
                c_rgb = list(c_info.color_rgb)
            elif self.model_id == "coco-baseline" and 1 <= class_id < len(COCO_CLASSES):
                coco_raw = COCO_CLASSES[class_id]
                subset_id, subset_name, is_target = map_coco_id_to_subset(class_id)
                if is_target and subset_id is not None:
                    target_info = CLASS_BY_ID.get(subset_id)
                    c_name = target_info.name if target_info else coco_raw.title()
                    c_hex = target_info.color_hex if target_info else "#0072B2"
                    c_rgb = (
                        list(target_info.color_rgb) if target_info else [0, 114, 178]
                    )
                else:
                    c_name = coco_raw.title()
                    c_hex = "#56B4E9"
                    c_rgb = [86, 180, 233]
            else:
                c_name = f"Class_{class_id}"
                c_hex = "#0072B2"
                c_rgb = [0, 114, 178]

            formatted_detections.append(
                {
                    "class_id": class_id,
                    "class_name": c_name,
                    "confidence": round(score, 4),
                    "box_normalized": {
                        "ymin": round(ymin, 4),
                        "xmin": round(xmin, 4),
                        "ymax": round(ymax, 4),
                        "xmax": round(xmax, 4),
                    },
                    "box_pixel": {
                        "ymin": px_ymin,
                        "xmin": px_xmin,
                        "ymax": px_ymax,
                        "xmax": px_xmax,
                        "width": px_w,
                        "height": px_h,
                    },
                    "color_hex": c_hex,
                    "color_rgb": c_rgb,
                }
            )

        total_ms = (time.perf_counter() - start_total) * 1000.0

        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "total_detections": len(formatted_detections),
            "inference_time_ms": round(inference_ms, 2),
            "total_time_ms": round(total_ms, 2),
            "image_dimensions": {
                "width": orig_w,
                "height": orig_h,
                "channels": 3,
            },
            "thresholds_applied": {
                "confidence_threshold": confidence_threshold,
                "nms_iou_threshold": nms_iou_threshold,
            },
            "detections": formatted_detections,
        }


class ModelCache:
    """Thread-safe lazy-loading cache for registered detection models."""

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._cache: dict[str, DetectorService] = {}
        self.registry_path = registry_path

    def get(self, model_id: str = "fine-tuned") -> DetectorService:
        """
        Retrieve a model from cache or lazily load it on first use.
        Thread-safe. Raises UnknownModelError if model_id is not registered.
        """
        with self._lock:
            if model_id in self._cache:
                return self._cache[model_id]

            entry = get_model_entry(model_id, self.registry_path)
            if entry is None:
                raise UnknownModelError(model_id)

            detector = self._create_detector(entry)
            self._cache[model_id] = detector
            return detector

    def _create_detector(self, entry: ModelEntry) -> DetectorService:
        root = Path(__file__).parent.parent.parent
        checkpoint_path: Path | None = None
        if entry.path:
            cand = root / entry.path
            if cand.exists():
                checkpoint_path = cand

        return DetectorService(
            model_id=entry.id,
            model_name=entry.display_name,
            model_version=entry.version,
            num_classes=81 if entry.id == "coco-baseline" else 21,
            checkpoint_path=checkpoint_path,
        )

    def is_loaded(self, model_id: str) -> bool:
        """Check whether a model is currently loaded in memory."""
        with self._lock:
            return model_id in self._cache

    def loaded_model_ids(self) -> list[str]:
        """Return list of currently loaded model IDs."""
        with self._lock:
            return list(self._cache.keys())

    def clear(self) -> None:
        """Clear all loaded models from cache."""
        with self._lock:
            self._cache.clear()


# Global cache instance
_cache_instance: ModelCache | None = None
_cache_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    """Retrieve or initialize the global thread-safe ModelCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = ModelCache()
    return _cache_instance


def get_detector(model_id: str = "fine-tuned") -> DetectorService:
    """Retrieve or lazily load the requested detector model."""
    return get_model_cache().get(model_id)
