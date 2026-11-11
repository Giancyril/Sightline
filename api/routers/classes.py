"""
api/routers/classes.py
----------------------
Endpoint listing the 20-class target taxonomy with labels, colors, and descriptions.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import ClassesResponse, ClassItem
from ml.data.classes import CLASSES

router = APIRouter(prefix="/v1", tags=["Taxonomy"])


@router.get(
    "/classes",
    response_model=ClassesResponse,
    summary="Get 20-class object detection taxonomy",
    description="Returns the full list of supported object classes, color codes, and descriptions.",
)
def get_classes() -> ClassesResponse:
    """Return all 20 classes with UI color palettes and descriptions."""
    items = [
        ClassItem(
            id=c.id,
            name=c.name,
            mid=c.mid,
            is_coco=c.is_coco,
            color_hex=c.color_hex,
            color_rgb=list(c.color_rgb),
            description=c.description,
        )
        for c in CLASSES
    ]
    return ClassesResponse(total_classes=len(items), classes=items)
