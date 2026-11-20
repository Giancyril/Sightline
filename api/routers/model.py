"""
api/routers/model.py
--------------------
Endpoint returning model architecture details, input constraints, and benchmark metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import (
    ClassThresholdResponseItem,
    ModelInfoResponse,
    ModelMetrics,
    ModelRegistryItem,
    ModelsResponse,
    ModelThresholdsResponse,
)
from api.services.detector import MAX_FILE_SIZE_BYTES, get_model_cache
from ml.models.registry import get_model_entry, load_model_registry

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


@router.get(
    "/models/{model_id}/thresholds",
    response_model=ModelThresholdsResponse,
    summary="Get recommended per-class thresholds and evaluation metrics for a model",
    description="Returns the calibrated per-class thresholds table, precision, recall, and fallback status.",
    responses={
        200: {"description": "Per-class thresholds table."},
        404: {
            "description": "Model not found or model does not have a threshold file."
        },
    },
)
def get_model_thresholds(model_id: str) -> ModelThresholdsResponse:
    """Return per-class threshold table for a specific model."""
    entry = get_model_entry(model_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id: '{model_id}'",
        )
    if entry.thresholds is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' does not have recommended thresholds or calibration files.",
        )

    t = entry.thresholds
    return ModelThresholdsResponse(
        model_id=t.model_id,
        rule=t.rule,
        date=t.date,
        data_hash=t.data_hash,
        global_threshold=t.global_threshold,
        fallbacks=t.fallbacks,
        thresholds=t.thresholds,
        per_class=[
            ClassThresholdResponseItem(
                class_id=c.class_id,
                class_name=c.class_name,
                threshold=c.threshold,
                precision=c.precision,
                recall=c.recall,
                f1=c.f1,
                precision_raw=c.precision_raw,
                recall_raw=c.recall_raw,
                f1_raw=c.f1_raw,
                num_detections=c.num_detections,
                fell_back=c.fell_back,
                fallback_reason=c.fallback_reason,
            )
            for c in t.per_class
        ],
    )
