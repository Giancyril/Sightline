"""
ml/models/registry.py
---------------------
Model registry schema and loader for VisionSSD detection models.
Defines schemas for model entries, metadata, input specs, benchmark metrics,
per-class confidence thresholds, and calibration statistics,
with file loading, validation, and retrieval helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class RegistryValidationError(ValueError):
    """Raised when registry, thresholds, or calibration validation fails."""


# ---------------------------------------------------------------------------
# Thresholds and Calibration Schemas
# ---------------------------------------------------------------------------


class ClassThresholdDetail(BaseModel):
    """Detailed validation metrics and selected threshold for a single class."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    class_id: int = Field(..., ge=1, description="Integer class ID")
    class_name: str = Field(..., min_length=1, description="Class name")
    threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Selected confidence threshold"
    )
    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Precision at selected threshold"
    )
    recall: float = Field(
        ..., ge=0.0, le=1.0, description="Recall at selected threshold"
    )
    f1: float = Field(..., ge=0.0, le=1.0, description="F1 score at selected threshold")
    precision_raw: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Precision at default global threshold",
    )
    recall_raw: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recall at default global threshold"
    )
    f1_raw: float = Field(
        default=0.0, ge=0.0, le=1.0, description="F1 score at default global threshold"
    )
    num_detections: int = Field(
        default=0, ge=0, description="Number of validation detections for this class"
    )
    fell_back: bool = Field(
        default=False, description="Whether selection fell back to global threshold"
    )
    fallback_reason: str | None = Field(
        default=None, description="Reason for fallback if fell_back is true"
    )


class ThresholdsModel(BaseModel):
    """Per-class confidence threshold metadata and lookup dictionary for a model."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    model_id: str = Field(..., min_length=1, description="Model identifier")
    rule: str = Field(
        default="best_f1",
        description="Selection rule (best_f1, target_precision, etc.)",
    )
    date: str = Field(
        ..., description="Timestamp or date when thresholds were selected"
    )
    data_hash: str = Field(
        ..., min_length=1, description="SHA-256 hash of validation manifest"
    )
    global_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Default global fallback threshold"
    )
    fallbacks: list[str] = Field(
        default_factory=list, description="List of class names that fell back"
    )
    thresholds: dict[str, float] = Field(
        ..., description="Fast lookup mapping of class_name -> threshold [0, 1]"
    )
    per_class: list[ClassThresholdDetail] = Field(
        default_factory=list, description="Per-class metric details and fallbacks"
    )

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        for cname, thresh in v.items():
            if not (0.0 <= thresh <= 1.0):
                raise ValueError(
                    f"Threshold for '{cname}' must be in [0.0, 1.0], got {thresh}"
                )
        return v


class CalibrationBin(BaseModel):
    """Statistics for a single confidence interval bin in a reliability diagram."""

    model_config = ConfigDict(populate_by_name=True)

    bin_lower: float = Field(..., ge=0.0, le=1.0)
    bin_upper: float = Field(..., ge=0.0, le=1.0)
    mean_confidence: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    count: int = Field(..., ge=0)


class CalibrationModel(BaseModel):
    """Temperature scaling parameters and Expected Calibration Error stats."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    model_id: str = Field(..., min_length=1, description="Model identifier")
    temperature: float = Field(
        ..., gt=0.0, description="Fitted temperature scalar T > 0"
    )
    ece_before: float = Field(
        ..., ge=0.0, le=1.0, description="ECE on uncalibrated raw scores"
    )
    ece_after: float = Field(
        ..., ge=0.0, le=1.0, description="ECE after temperature scaling calibration"
    )
    nll_before: float | None = Field(
        default=None, description="Negative log-likelihood before"
    )
    nll_after: float | None = Field(
        default=None, description="Negative log-likelihood after"
    )
    date: str = Field(..., description="Timestamp or date when calibration was fitted")
    data_hash: str = Field(
        ..., min_length=1, description="SHA-256 hash of validation manifest"
    )
    bins_before: list[CalibrationBin] = Field(
        default_factory=list, description="Reliability bins before calibration"
    )
    bins_after: list[CalibrationBin] = Field(
        default_factory=list, description="Reliability bins after calibration"
    )


# ---------------------------------------------------------------------------
# Core Model Registry Schemas
# ---------------------------------------------------------------------------


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
    thresholds: ThresholdsModel | None = Field(
        default=None, description="Loaded and validated per-class thresholds"
    )
    calibration: CalibrationModel | None = Field(
        default=None, description="Loaded and validated temperature calibration"
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

    def get_threshold_for_class(
        self,
        class_name: str,
        mode: str = "global",
        custom_thresholds: dict[str, float] | None = None,
        global_threshold: float = 0.25,
    ) -> float:
        """
        Get effective confidence threshold for a class given threshold_mode.

        Args:
            class_name: Name of the detected class.
            mode: "global", "recommended", or "custom".
            custom_thresholds: Optional dict mapping class_name -> threshold for custom mode.
            global_threshold: Fallback threshold when in global mode or class is missing.

        Returns:
            Confidence threshold float in [0.0, 1.0].
        """
        if mode == "recommended" and self.thresholds is not None:
            return float(
                self.thresholds.thresholds.get(
                    class_name, self.thresholds.global_threshold
                )
            )
        if mode == "custom" and custom_thresholds is not None:
            return float(custom_thresholds.get(class_name, global_threshold))
        return float(global_threshold)

    @property
    def has_recommended_thresholds(self) -> bool:
        """True if model has validated per-class thresholds."""
        return self.thresholds is not None

    @property
    def has_calibration(self) -> bool:
        """True if model has validated temperature calibration."""
        return self.calibration is not None


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


# ---------------------------------------------------------------------------
# Loader and Validator Helpers
# ---------------------------------------------------------------------------


def load_thresholds_file(file_path: Path | str) -> ThresholdsModel:
    """Load and validate a thresholds.json file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Thresholds file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ThresholdsModel.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise RegistryValidationError(f"Invalid thresholds file at '{p}': {e}") from e


def load_calibration_file(file_path: Path | str) -> CalibrationModel:
    """Load and validate a calibration.json file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return CalibrationModel.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise RegistryValidationError(f"Invalid calibration file at '{p}': {e}") from e


def find_model_file(base_dir: Path, model_id: str, filename: str) -> Path | None:
    """Search candidate relative paths for a model-associated file."""
    candidates = [
        base_dir / model_id / filename,
        base_dir / f"{model_id}_{filename}",
        base_dir / filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def validate_registry_data(
    data: dict[str, Any] | list[Any],
    base_dir: Path | None = None,
) -> list[ModelEntry]:
    """Validate raw dictionary/list data against the model registry schema."""
    if isinstance(data, list):
        models = [ModelEntry.model_validate(item) for item in data]
    elif isinstance(data, dict):
        if "models" in data:
            registry = ModelRegistry.model_validate(data)
            models = registry.models
        else:
            models = [ModelEntry.model_validate(data)]
    else:
        raise TypeError(
            f"Expected dict or list for registry data, got {type(data).__name__}"
        )

    # Attach thresholds and calibration if present
    if base_dir is not None:
        for model in models:
            t_path = find_model_file(base_dir, model.id, "thresholds.json")
            if t_path is not None:
                model.thresholds = load_thresholds_file(t_path)
            c_path = find_model_file(base_dir, model.id, "calibration.json")
            if c_path is not None:
                model.calibration = load_calibration_file(c_path)

    return models


def load_model_registry(registry_path: str | Path | None = None) -> list[ModelEntry]:
    """
    Load and validate the model registry JSON file.
    Also validates associated thresholds.json and calibration.json files when present.
    Returns list of validated ModelEntry instances.
    """
    path = resolve_registry_path(registry_path)
    base_dir = path.parent
    content = json.loads(path.read_text(encoding="utf-8"))
    return validate_registry_data(content, base_dir=base_dir)


def get_model_entry(
    model_id: str, registry_path: str | Path | None = None
) -> ModelEntry | None:
    """Retrieve a specific registered model by its id, or None if not found."""
    models = load_model_registry(registry_path)
    for m in models:
        if m.id == model_id:
            return m
    return None
