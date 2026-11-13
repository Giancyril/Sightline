"""
api/routers/model.py
--------------------
Endpoint returning model architecture details, input constraints, and benchmark metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from api.schemas import (
    ModelInfoResponse,
    ModelMetrics,
    ModelRegistryItem,
    ModelsResponse,
)
from api.services.detector import MAX_FILE_SIZE_BYTES, get_model_cache
from ml.models.registry import load_model_registry

router = APIRouter(prefix="/v1", tags=["Model Metadata"])


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Get model metadata and benchmark metrics",
    description="Returns architecture information, input resolution requirements, and evaluation scores.",
)
def get_model_info() -> ModelInfoResponse:
    """Return model specification and validation benchmark metrics."""
    root = Path(__file__).parent.parent.parent
    metrics_path = root / "docs" / "fine_tuned_metrics.json"

    # Default fallback metrics
    m_all = 0.2644
    m_shared = 0.3471
    m_novel = 0.1816
    m_range = 0.1603
    best_thresh = 0.25

    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            m_all = data.get("mAP_all", m_all)
            m_shared = data.get("mAP_coco_shared", m_shared)
            m_novel = data.get("mAP_novel_classes", m_novel)
            m_range = data.get("coco_mAP_50_95", m_range)
            best_thresh = data.get("threshold_tuning", {}).get(
                "best_threshold", best_thresh
            )
        except Exception:
            pass

    return ModelInfoResponse(
        name="SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
        version="v1.0.0",
        architecture="SSDMobileNetV2",
        input_resolution=[300, 300],
        num_classes=21,
        optimal_confidence_threshold=best_thresh,
        supported_formats=["JPEG", "PNG", "WebP"],
        max_file_size_bytes=MAX_FILE_SIZE_BYTES,
        metrics=ModelMetrics(
            mAP_all=m_all,
            mAP_coco_shared=m_shared,
            mAP_novel_classes=m_novel,
            coco_mAP_50_95=m_range,
        ),
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List all registered models with loading status",
    description=(
        "Returns metadata, supported class lists, benchmark metrics, "
        "and runtime in-memory status for each registered model."
    ),
)
def list_models() -> ModelsResponse:
    """Return registry of available detection models and their current in-memory status."""
    registry = load_model_registry()
    cache = get_model_cache()

    items = []
    for entry in registry:
        loaded = cache.is_loaded(entry.id)
        items.append(
            ModelRegistryItem(
                id=entry.id,
                display_name=entry.display_name,
                version=entry.version,
                path=entry.path,
                input_size=entry.input_size,
                num_classes=len(entry.class_list),
                class_list=entry.class_list,
                metrics=entry.metrics,
                loaded=loaded,
                status="loaded" if loaded else "ready",
            )
        )

    return ModelsResponse(total_models=len(items), models=items)
