"""
ml/models/registry.py
---------------------
Model registry schema and loader for VisionSSD detection models.
Defines schemas for model entries, metadata, input specs, and benchmark metrics,
with file loading, validation, and retrieval helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelMetrics(BaseModel):
    """Benchmark metrics for registered detection models."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    mAP_all: float = Field(..., ge=0.0, le=1.0, description="Overall mAP at IoU=0.5")
    mAP_coco_shared: float = Field(
        ..., ge=0.0, le=1.0, description="mAP on COCO-shared subset classes"
    )
    mAP_novel_classes: float = Field(
        ..., ge=0.0, le=1.0, description="mAP on novel non-COCO classes"
    )
    coco_mAP_50_95: float = Field(
        ..., ge=0.0, le=1.0, description="COCO standard mAP@[0.50:0.95]"
    )


class ModelEntry(BaseModel):
    """Metadata and specification for a registered object detection model."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., min_length=1, description="Unique model identifier")
    display_name: str = Field(
        ..., min_length=1, alias="displayName", description="Human-readable model name"
    )
    version: str = Field(..., min_length=1, description="Semantic version string")
    path: str | None = Field(
        default=None, description="Relative path to weights checkpoint or None"
    )
    input_size: list[int] = Field(
        ...,
        alias="inputSize",
        description="Input dimensions as [width, height] in pixels",
    )
    class_list: list[str] = Field(
        ...,
        alias="classList",
        min_length=1,
        description="Supported class taxonomy names",
    )
    metrics: dict[str, float] = Field(
        ..., description="Benchmark accuracy metrics mapping"
    )

    @field_validator("input_size")
    @classmethod
    def validate_input_size(cls, v: list[int]) -> list[int]:
        if len(v) != 2:
            raise ValueError(
                f"input_size must contain exactly 2 integers [W, H], got {v}"
            )
        if any(dim <= 0 for dim in v):
            raise ValueError(
                f"input_size dimensions must be positive integers, got {v}"
            )
        return v

    @field_validator("class_list")
    @classmethod
    def validate_class_list(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("class_list must contain at least one class")
        return v


class ModelRegistry(BaseModel):
    """Top-level model registry container."""

    model_config = ConfigDict(populate_by_name=True)

    version: str = Field(default="1.0.0", description="Registry format version")
    models: list[ModelEntry] = Field(
        default_factory=list, description="List of registered model entries"
    )


DEFAULT_REGISTRY_PATHS: list[Path] = [
    Path("models/registry.json"),
    Path(__file__).parent.parent.parent / "models" / "registry.json",
    Path(__file__).parent / "registry.json",
]


def resolve_registry_path(custom_path: str | Path | None = None) -> Path:
    """Resolve the registry JSON file path from candidate locations."""
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Specified registry path does not exist: {custom_path}"
        )

    for candidate in DEFAULT_REGISTRY_PATHS:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Model registry file could not be located in candidate paths: {[str(p) for p in DEFAULT_REGISTRY_PATHS]}"
    )


def validate_registry_data(data: dict[str, Any] | list[Any]) -> list[ModelEntry]:
    """Validate raw dictionary/list data against the model registry schema."""
    if isinstance(data, list):
        return [ModelEntry.model_validate(item) for item in data]
    if isinstance(data, dict):
        if "models" in data:
            registry = ModelRegistry.model_validate(data)
            return registry.models
        # Single entry dict
        return [ModelEntry.model_validate(data)]
    raise TypeError(
        f"Expected dict or list for registry data, got {type(data).__name__}"
    )


def load_model_registry(registry_path: str | Path | None = None) -> list[ModelEntry]:
    """
    Load and validate the model registry JSON file.
    Returns list of validated ModelEntry instances.
    """
    path = resolve_registry_path(registry_path)
    content = json.loads(path.read_text(encoding="utf-8"))
    return validate_registry_data(content)


def get_model_entry(
    model_id: str, registry_path: str | Path | None = None
) -> ModelEntry | None:
    """Retrieve a specific registered model by its id, or None if not found."""
    models = load_model_registry(registry_path)
    for m in models:
        if m.id == model_id:
            return m
    return None
