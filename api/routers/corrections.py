"""
api/routers/corrections.py
--------------------------
Endpoints for persisting user annotations in standard COCO format.
Follows zero-storage privacy by default: only annotations, model metadata,
original predictions, and an image SHA-256 hash are retained.
Raw image data is stored only if the user explicitly opts in with store_image=True.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.schemas import CorrectionCreateRequest, CorrectionResponse
from ml.data.classes import CLASSES

router = APIRouter(prefix="/v1", tags=["Corrections"])


def get_corrections_dir() -> Path:
    """Return directory where corrections are stored."""
    target = os.getenv("CORRECTIONS_DIR", "data/corrections")
    p = Path(target)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_coco_categories() -> list[dict[str, Any]]:
    """Generate standard COCO categories for all 20 classes."""
    return [
        {
            "id": c.id,
            "name": c.name,
            "supercategory": "object",
        }
        for c in CLASSES
    ]


@router.post(
    "/corrections",
    response_model=CorrectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store user annotation corrections in COCO format",
    description=(
        "Stores corrected bounding boxes with original prediction, model ID, "
        "and SHA-256 image hash in standard COCO JSON format. "
        "PRIVACY: Stores NO image by default. Raw image is saved only if store_image=True."
    ),
)
def create_correction(payload: CorrectionCreateRequest) -> CorrectionResponse:
    """Validate and persist user-corrected annotations in COCO format."""
    corrections_dir = get_corrections_dir()
    images_dir = corrections_dir / "images"

    # Validate image_hash (must be non-empty hex string)
    clean_hash = payload.image_hash.strip().lower()
    if not clean_hash or not re.match(r"^[a-f0-9]{32,64}$", clean_hash):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="image_hash must be a valid 32-character or 64-character hex string (e.g. SHA-256).",
        )

    # Validate image dimensions
    w = payload.image_dimensions.width
    h = payload.image_dimensions.height
    if w <= 0 or h <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Image dimensions width and height must be strictly positive.",
        )

    # Validate bounding boxes
    valid_class_ids = {c.id for c in CLASSES}
    for idx, box in enumerate(payload.corrected_boxes):
        if box.class_id not in valid_class_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Box {idx}: invalid class_id {box.class_id}. Must be between 1 and 20.",
            )
        b = box.box_normalized
        if b.ymin >= b.ymax or b.xmin >= b.xmax:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Box {idx}: degenerate coordinates (ymin={b.ymin}, ymax={b.ymax}, xmin={b.xmin}, xmax={b.xmax}).",
            )

    correction_id = f"corr_{uuid.uuid4().hex[:12]}"
    created_at = datetime.datetime.now(datetime.UTC).isoformat()

    has_image = False
    image_rel_path = ""

    # PRIVACY RULE: Store image ONLY if explicitly opted in
    if payload.store_image and payload.image_base64:
        images_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Handle possible data URI prefix
            b64_data = payload.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)

            # Optional: verify image hash against decoded bytes
            computed_hash = hashlib.sha256(img_bytes).hexdigest()
            # Save raw image
            img_filename = f"{correction_id}.jpg"
            img_dest = images_dir / img_filename
            with open(img_dest, "wb") as f:
                f.write(img_bytes)

            has_image = True
            image_rel_path = f"images/{img_filename}"
            clean_hash = clean_hash or computed_hash
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode base64 image payload: {exc}",
            ) from exc

    # Build COCO JSON structure
    coco_image = {
        "id": 1,
        "width": w,
        "height": h,
        "file_name": image_rel_path if has_image else f"hash_{clean_hash}.jpg",
        "image_hash": clean_hash,
        "has_image": has_image,
        "date_captured": created_at,
    }

    coco_annotations: list[dict[str, Any]] = []
    for ann_idx, box in enumerate(payload.corrected_boxes, start=1):
        b = box.box_normalized
        xmin_px = round(b.xmin * w, 2)
        ymin_px = round(b.ymin * h, 2)
        box_w = round((b.xmax - b.xmin) * w, 2)
        box_h = round((b.ymax - b.ymin) * h, 2)
        area = round(box_w * box_h, 2)

        coco_annotations.append(
            {
                "id": ann_idx,
                "image_id": 1,
                "category_id": box.class_id,
                "bbox": [xmin_px, ymin_px, box_w, box_h],
                "area": area,
                "segmentation": [],
                "iscrowd": 0,
                "confidence": box.confidence if box.confidence is not None else 1.0,
                "is_manual": box.is_manual,
            }
        )

    coco_document = {
        "info": {
            "description": "Object Detection System - User Corrections",
            "url": "https://github.com/example/object-detection-system",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "User",
            "date_created": created_at,
        },
        "images": [coco_image],
        "annotations": coco_annotations,
        "categories": build_coco_categories(),
        "metadata": {
            "correction_id": correction_id,
            "model_id": payload.model_id,
            "model_version": payload.model_version,
            "image_hash": clean_hash,
            "original_predictions": [
                det.model_dump() for det in payload.original_detections
            ],
            "total_annotations": len(coco_annotations),
            "stored_raw_image": has_image,
        },
    }

    coco_filename = f"{correction_id}.json"
    coco_path = corrections_dir / coco_filename
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_document, f, indent=2)

    return CorrectionResponse(
        id=correction_id,
        image_hash=clean_hash,
        created_at=created_at,
        model_id=payload.model_id,
        model_version=payload.model_version,
        total_annotations=len(coco_annotations),
        has_image=has_image,
        coco_path=str(coco_filename),
    )


@router.get(
    "/corrections/{correction_id}",
    summary="Get a stored correction COCO document by ID",
)
def get_correction(correction_id: str) -> dict[str, Any]:
    """Retrieve the stored COCO JSON document for a correction."""
    corrections_dir = get_corrections_dir()
    coco_file = corrections_dir / f"{correction_id}.json"
    if not coco_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Correction '{correction_id}' not found.",
        )
    with open(coco_file, encoding="utf-8") as f:
        return json.load(f)


@router.delete(
    "/corrections/{correction_id}",
    summary="Delete a stored correction and any associated raw image",
)
def delete_correction(correction_id: str) -> dict[str, Any]:
    """Delete the correction COCO JSON and any stored image."""
    corrections_dir = get_corrections_dir()
    coco_file = corrections_dir / f"{correction_id}.json"

    if not coco_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Correction '{correction_id}' not found.",
        )

    # Remove any associated stored image
    images_dir = corrections_dir / "images"
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        img_file = images_dir / f"{correction_id}{ext}"
        if img_file.exists():
            img_file.unlink()

    # Remove COCO JSON
    coco_file.unlink()

    return {"deleted": True, "id": correction_id}
