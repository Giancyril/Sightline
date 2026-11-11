"""
ml/models/coco_labels.py
------------------------
COCO 80-class taxonomy and mapping to the 20-class Open Images subset.
Enables baseline pre-trained COCO detectors to map detections into the subset taxonomy.
"""

from __future__ import annotations

from ml.data.classes import CLASS_BY_ID, ClassInfo

# 80 standard COCO classes (1-indexed, background is 0)
COCO_CLASSES: list[str] = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(COCO_CLASSES)}

# Mapping: COCO class name -> Subset ClassInfo
COCO_TO_SUBSET_MAP: dict[str, ClassInfo] = {}
for subset_cls in CLASS_BY_ID.values():
    if subset_cls.is_coco and subset_cls.coco_name:
        COCO_TO_SUBSET_MAP[subset_cls.coco_name.lower()] = subset_cls


def map_coco_id_to_subset(
    coco_id: int,
) -> tuple[int | None, str | None, bool]:
    """
    Map a COCO label ID (1-80) to our subset class ID (1-20).
    Returns (subset_class_id, subset_class_name, is_target_class).
    If the detected COCO class is not in our 20-class subset,
    returns (None, coco_name, False).
    """
    if not (1 <= coco_id < len(COCO_CLASSES)):
        return None, None, False

    coco_name = COCO_CLASSES[coco_id]
    if coco_name in COCO_TO_SUBSET_MAP:
        subset_cls = COCO_TO_SUBSET_MAP[coco_name]
        return subset_cls.id, subset_cls.name, True

    return None, coco_name, False
