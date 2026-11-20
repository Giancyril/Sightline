"""
api/schemas.py
--------------
Pydantic v2 schemas for the Object Detection Inference API.
Covers request validation, detection outputs, class definitions,
and model metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NormalizedBBox(BaseModel):
    """Normalized bounding box coordinates in [0.0, 1.0]."""

    model_config = ConfigDict(frozen=True)

    ymin: float = Field(..., ge=0.0, le=1.0, description="Top edge (0.0 to 1.0)")
    xmin: float = Field(..., ge=0.0, le=1.0, description="Left edge (0.0 to 1.0)")
    ymax: float = Field(..., ge=0.0, le=1.0, description="Bottom edge (0.0 to 1.0)")
    xmax: float = Field(..., ge=0.0, le=1.0, description="Right edge (0.0 to 1.0)")


class PixelBBox(BaseModel):
    """Pixel coordinate bounding box with calculated width and height."""

    model_config = ConfigDict(frozen=True)

    ymin: int = Field(..., ge=0, description="Top pixel coordinate")
    xmin: int = Field(..., ge=0, description="Left pixel coordinate")
    ymax: int = Field(..., ge=0, description="Bottom pixel coordinate")
    xmax: int = Field(..., ge=0, description="Right pixel coordinate")
    width: int = Field(..., ge=0, description="Box width in pixels")
    height: int = Field(..., ge=0, description="Box height in pixels")


class DetectionItem(BaseModel):
    """Single detected object instance."""

    class_id: int = Field(..., ge=1, le=90, description="1-indexed class ID")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    box_normalized: NormalizedBBox = Field(
        ..., description="Normalized coordinates [0, 1]"
    )
    box_pixel: PixelBBox = Field(..., description="Absolute pixel coordinates")
    color_hex: str = Field(..., description="Categorical hex color for bounding box UI")
    color_rgb: list[int] = Field(..., description="RGB color tuple [R, G, B]")


class ImageDimensions(BaseModel):
    """Source image dimensions."""

    width: int = Field(..., ge=1, description="Image width in pixels")
    height: int = Field(..., ge=1, description="Image height in pixels")
    channels: int = Field(default=3, description="Color channels (3 for RGB)")


class ThresholdsApplied(BaseModel):
    """Thresholds applied to generate this response."""

    confidence_threshold: float = Field(..., description="Minimum detection score")
    nms_iou_threshold: float = Field(..., description="IoU suppression threshold")
    threshold_mode: str = Field(
        default="global",
        description="Threshold mode used: 'global', 'recommended', or 'custom'",
    )
    calibration_applied: bool = Field(
        default=False,
        description="Whether score calibration (temperature scaling) was applied",
    )


class DetectionResponse(BaseModel):
    """Response payload for POST /v1/detect."""

    model_id: str = Field(default="fine-tuned", description="Model registry ID")
    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version tag")
    threshold_mode: str = Field(
        default="global",
        description="Threshold selection mode ('global', 'recommended', or 'custom')",
    )
    calibration_applied: bool = Field(
        default=False,
        description="True if temperature scaling calibration was applied to scores",
    )
    total_detections: int = Field(..., ge=0, description="Count of detected objects")
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Pure model latency in ms"
    )
    total_time_ms: float = Field(..., ge=0.0, description="Total request latency in ms")
    image_dimensions: ImageDimensions = Field(
        ..., description="Dimensions of the input image"
    )
    thresholds_applied: ThresholdsApplied = Field(..., description="Thresholds used")
    detections: list[DetectionItem] = Field(
        default_factory=list, description="List of detections"
    )


class ClassItem(BaseModel):
    """Metadata for a supported object class."""

    id: int = Field(..., description="1-indexed class ID")
    name: str = Field(..., description="Class name")
    mid: str = Field(..., description="Open Images Machine Identifier")
    is_coco: bool = Field(..., description="Whether class overlaps with COCO dataset")
    color_hex: str = Field(..., description="Hex color")
    color_rgb: list[int] = Field(..., description="RGB color [R, G, B]")
    description: str = Field(..., description="Visual description of the concept")


class ClassesResponse(BaseModel):
    """Response payload for GET /v1/classes."""

    total_classes: int = Field(..., description="Total number of supported classes")
    classes: list[ClassItem] = Field(..., description="List of all class metadata")


class ModelMetrics(BaseModel):
    """Model accuracy benchmark metrics."""

    mAP_all: float
    mAP_coco_shared: float
    mAP_novel_classes: float
    coco_mAP_50_95: float


class ModelInfoResponse(BaseModel):
    """Response payload for GET /v1/model."""

    name: str
    version: str
    architecture: str
    input_resolution: list[int]
    num_classes: int
    optimal_confidence_threshold: float
    supported_formats: list[str]
    max_file_size_bytes: int
    metrics: ModelMetrics


class HealthResponse(BaseModel):
    """Response payload for GET /healthz."""

    status: str
    model_loaded: bool
    version: str


class ModelRegistryItem(BaseModel):
    """Registered model metadata including current runtime loading status."""

    id: str = Field(..., description="Unique model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    version: str = Field(..., description="Semantic version string")
    path: str | None = Field(default=None, description="Path to model checkpoint")
    input_size: list[int] = Field(..., description="Input dimensions [W, H]")
    num_classes: int = Field(..., description="Total class count")
    class_list: list[str] = Field(..., description="Supported class names")
    metrics: dict[str, float] = Field(..., description="Validation benchmark metrics")
    loaded: bool = Field(..., description="Whether model is currently loaded in memory")
    status: str = Field(..., description="Runtime status: 'loaded' or 'ready'")


class ModelsResponse(BaseModel):
    """Response payload for GET /v1/models."""

    total_models: int = Field(..., description="Number of registered models")
    models: list[ModelRegistryItem] = Field(
        ..., description="List of registered model entries"
    )


class CompareResponse(BaseModel):
    """Response payload for POST /v1/compare."""

    model_a_result: DetectionResponse = Field(
        ..., description="Detection result from model A"
    )
    model_b_result: DetectionResponse = Field(
        ..., description="Detection result from model B"
    )
    total_time_ms: float = Field(
        ..., ge=0.0, description="Total wall-clock latency for parallel inference in ms"
    )


class ClassThresholdResponseItem(BaseModel):
    """Per-class confidence threshold metrics and status."""

    class_id: int = Field(..., description="1-indexed class ID")
    class_name: str = Field(..., description="Class name")
    threshold: float = Field(..., description="Selected confidence threshold")
    precision: float = Field(..., description="Precision at selected threshold")
    recall: float = Field(..., description="Recall at selected threshold")
    f1: float = Field(..., description="F1 score at selected threshold")
    precision_raw: float = Field(
        default=0.0, description="Precision at global threshold (0.25)"
    )
    recall_raw: float = Field(
        default=0.0, description="Recall at global threshold (0.25)"
    )
    f1_raw: float = Field(
        default=0.0, description="F1 score at global threshold (0.25)"
    )
    num_detections: int = Field(
        default=0, description="Number of validation detections"
    )
    fell_back: bool = Field(
        default=False, description="True if selection fell back to global threshold"
    )
    fallback_reason: str | None = Field(
        default=None, description="Reason for fallback if fell_back is true"
    )


class ModelThresholdsResponse(BaseModel):
    """Response payload for GET /v1/models/{id}/thresholds."""

    model_id: str = Field(..., description="Unique model identifier")
    rule: str = Field(..., description="Threshold selection rule (e.g. best_f1)")
    date: str = Field(..., description="Timestamp of threshold calculation")
    data_hash: str = Field(..., description="SHA-256 hash of validation manifest")
    global_threshold: float = Field(
        ..., description="Default global fallback threshold"
    )
    fallbacks: list[str] = Field(
        default_factory=list, description="Class names that fell back"
    )
    thresholds: dict[str, float] = Field(
        ..., description="Fast lookup mapping of class_name -> threshold"
    )
    per_class: list[ClassThresholdResponseItem] = Field(
        default_factory=list, description="Detailed per-class thresholds and metrics"
    )


class CorrectionAnnotation(BaseModel):
    """Single corrected bounding box annotation."""

    class_id: int = Field(..., ge=1, le=20, description="1-indexed class ID (1-20)")
    class_name: str = Field(..., description="Human-readable class name")
    box_normalized: NormalizedBBox = Field(
        ..., description="Normalized coordinates [0, 1]"
    )
    confidence: float | None = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    is_manual: bool = Field(
        default=True, description="Whether annotation was manually drawn or edited"
    )


class CorrectionCreateRequest(BaseModel):
    """Request payload for storing user corrections."""

    image_hash: str = Field(
        ...,
        pattern=r"^[a-fA-F0-9]{32,64}$",
        description="SHA-256 hash of the target image",
    )
    image_dimensions: ImageDimensions = Field(
        ..., description="Dimensions of the corrected image"
    )
    corrected_boxes: list[CorrectionAnnotation] = Field(
        ..., description="List of corrected annotations"
    )
    original_detections: list[DetectionItem] = Field(
        default_factory=list, description="Original model detections prior to edit"
    )
    model_id: str = Field(default="fine-tuned", description="Model registry ID")
    model_version: str = Field(default="v1.0", description="Model version")
    store_image: bool = Field(
        default=False,
        description="Explicit user opt-in to store raw image bytes. Default is False (zero image storage).",
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image payload; stored only if store_image is True.",
    )


class CorrectionResponse(BaseModel):
    """Response payload returned when a correction is stored."""

    id: str = Field(..., description="Unique correction ID")
    image_hash: str = Field(..., description="SHA-256 hash of target image")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    model_id: str = Field(..., description="Model registry ID")
    model_version: str = Field(..., description="Model version")
    total_annotations: int = Field(
        ..., ge=0, description="Number of corrected annotations"
    )
    has_image: bool = Field(
        ..., description="Whether raw image was stored via user opt-in"
    )
    coco_path: str = Field(..., description="Relative storage path to COCO JSON")

