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

    class_id: int = Field(
        ..., ge=1, le=20, description="1-indexed Open Images class ID"
    )
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


class DetectionResponse(BaseModel):
    """Response payload for POST /v1/detect."""

    model_name: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version tag")
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
