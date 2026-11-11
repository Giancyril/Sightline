"""
ml/data/classes.py
------------------
20-class Open Images subset definition with metadata and categorical colors.
Includes 10 COCO-overlapping classes and 10 non-COCO classes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str
    mid: str
    is_coco: bool
    coco_name: str | None
    color_hex: str
    color_rgb: tuple[int, int, int]
    description: str


# 20 categorical, color-blind accessible colors (Okabe-Ito / IBM Carbon inspired)
CLASSES: list[ClassInfo] = [
    # --- COCO Overlapping Classes (IDs 1-10) ---
    ClassInfo(
        id=1,
        name="Person",
        mid="/m/01g317",
        is_coco=True,
        coco_name="person",
        color_hex="#0072B2",
        color_rgb=(0, 114, 178),
        description="Individual human instance (adult, child, infant).",
    ),
    ClassInfo(
        id=2,
        name="Laptop",
        mid="/m/01c648",
        is_coco=True,
        coco_name="laptop",
        color_hex="#E69F00",
        color_rgb=(230, 159, 0),
        description="Portable personal computer with integrated screen and keyboard.",
    ),
    ClassInfo(
        id=3,
        name="Chair",
        mid="/m/01mzpv",
        is_coco=True,
        coco_name="chair",
        color_hex="#56B4E9",
        color_rgb=(86, 180, 233),
        description="Single-person seating furniture.",
    ),
    ClassInfo(
        id=4,
        name="Book",
        mid="/m/0bt9lr",
        is_coco=True,
        coco_name="book",
        color_hex="#009E73",
        color_rgb=(0, 158, 115),
        description="Bound collection of printed or written pages.",
    ),
    ClassInfo(
        id=5,
        name="Backpack",
        mid="/m/0130jx",
        is_coco=True,
        coco_name="backpack",
        color_hex="#F0E442",
        color_rgb=(240, 228, 66),
        description="Bag carried on back with shoulder straps.",
    ),
    ClassInfo(
        id=6,
        name="Cell phone",
        mid="/m/050k8",
        is_coco=True,
        coco_name="cell phone",
        color_hex="#D55E00",
        color_rgb=(213, 94, 0),
        description="Mobile smartphone or handset.",
    ),
    ClassInfo(
        id=7,
        name="Bottle",
        mid="/m/04dr76x",
        is_coco=True,
        coco_name="bottle",
        color_hex="#CC79A7",
        color_rgb=(204, 121, 167),
        description="Rigid container with narrow neck for liquids.",
    ),
    ClassInfo(
        id=8,
        name="Clock",
        mid="/m/01x3z",
        is_coco=True,
        coco_name="clock",
        color_hex="#6929C4",
        color_rgb=(105, 41, 196),
        description="Timepiece (wall, desk, or mantle).",
    ),
    ClassInfo(
        id=9,
        name="Car",
        mid="/m/0k4j",
        is_coco=True,
        coco_name="car",
        color_hex="#1192E8",
        color_rgb=(17, 146, 232),
        description="Automobile or passenger motor vehicle.",
    ),
    ClassInfo(
        id=10,
        name="Dog",
        mid="/m/0bt9lr_dog",
        is_coco=True,
        coco_name="dog",
        color_hex="#005D5D",
        color_rgb=(0, 93, 93),
        description="Domestic canine instance.",
    ),
    # --- Novel / Non-COCO Classes (IDs 11-20) ---
    ClassInfo(
        id=11,
        name="Guitar",
        mid="/m/0342h",
        is_coco=False,
        coco_name=None,
        color_hex="#9F1853",
        color_rgb=(159, 24, 83),
        description="Stringed musical instrument (acoustic or electric).",
    ),
    ClassInfo(
        id=12,
        name="Microphone",
        mid="/m/0hg7b",
        is_coco=False,
        coco_name=None,
        color_hex="#FA4D56",
        color_rgb=(250, 77, 86),
        description="Acoustic-to-electric transducer for sound recording.",
    ),
    ClassInfo(
        id=13,
        name="Helmet",
        mid="/m/0zvk5",
        is_coco=False,
        coco_name=None,
        color_hex="#570408",
        color_rgb=(87, 4, 8),
        description="Protective headgear for sports, cycling, or industrial use.",
    ),
    ClassInfo(
        id=14,
        name="Headphones",
        mid="/m/01b7fy",
        is_coco=False,
        coco_name=None,
        color_hex="#198038",
        color_rgb=(25, 128, 56),
        description="Pair of small listening speakers worn on or over ears.",
    ),
    ClassInfo(
        id=15,
        name="Wheelchair",
        mid="/m/012w5l",
        is_coco=False,
        coco_name=None,
        color_hex="#002D9C",
        color_rgb=(0, 45, 156),
        description="Mobility chair with wheels for individuals with disabilities.",
    ),
    ClassInfo(
        id=16,
        name="Whiteboard",
        mid="/m/02688pb",
        is_coco=False,
        coco_name=None,
        color_hex="#EE538B",
        color_rgb=(238, 83, 139),
        description="Dry-erase white writing surface for office/classroom.",
    ),
    ClassInfo(
        id=17,
        name="Flashlight",
        mid="/m/01b_21",
        is_coco=False,
        coco_name=None,
        color_hex="#B28600",
        color_rgb=(178, 134, 0),
        description="Portable battery-operated electric light torch.",
    ),
    ClassInfo(
        id=18,
        name="Calculator",
        mid="/m/01kggc",
        is_coco=False,
        coco_name=None,
        color_hex="#8A3800",
        color_rgb=(138, 56, 0),
        description="Electronic device used for mathematical arithmetic.",
    ),
    ClassInfo(
        id=19,
        name="Stapler",
        mid="/m/033cnk",
        is_coco=False,
        coco_name=None,
        color_hex="#A56EFF",
        color_rgb=(165, 110, 255),
        description="Mechanical office device for fastening sheets of paper.",
    ),
    ClassInfo(
        id=20,
        name="Wrench",
        mid="/m/02jnhm",
        is_coco=False,
        coco_name=None,
        color_hex="#009D9A",
        color_rgb=(0, 157, 154),
        description="Hand tool used for gripping and turning nuts and bolts.",
    ),
]

# Lookup indexes
CLASS_BY_ID: dict[int, ClassInfo] = {c.id: c for c in CLASSES}
CLASS_BY_NAME: dict[str, ClassInfo] = {c.name.lower(): c for c in CLASSES}
CLASS_BY_MID: dict[str, ClassInfo] = {c.mid: c for c in CLASSES}

BACKGROUND_ID = 0
NUM_CLASSES = len(CLASSES)  # 20 foreground classes


def get_class_by_id(class_id: int) -> ClassInfo:
    """Retrieve class info by 1-based class ID."""
    if class_id not in CLASS_BY_ID:
        raise KeyError(f"Invalid class ID: {class_id}. Must be in [1, {NUM_CLASSES}].")
    return CLASS_BY_ID[class_id]


def get_class_by_name(name: str) -> ClassInfo:
    """Retrieve class info by case-insensitive name."""
    normalized = name.strip().lower()
    if normalized not in CLASS_BY_NAME:
        raise KeyError(f"Unknown class name: '{name}'.")
    return CLASS_BY_NAME[normalized]


def get_class_by_mid(mid: str) -> ClassInfo:
    """Retrieve class info by Open Images MID."""
    if mid not in CLASS_BY_MID:
        raise KeyError(f"Unknown MID: '{mid}'.")
    return CLASS_BY_MID[mid]


def get_coco_classes() -> list[ClassInfo]:
    """Return all 10 COCO-overlapping classes."""
    return [c for c in CLASSES if c.is_coco]


def get_non_coco_classes() -> list[ClassInfo]:
    """Return all 10 novel non-COCO classes."""
    return [c for c in CLASSES if not c.is_coco]


def export_classes_json(output_path: Path | str | None = None) -> Path:
    """Export the classes dictionary to JSON format."""
    if output_path is None:
        output_path = Path(__file__).parent / "classes.json"
    else:
        output_path = Path(output_path)

    data = {
        "num_classes": NUM_CLASSES,
        "classes": [asdict(c) for c in CLASSES],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = export_classes_json()
    print(f"Exported {NUM_CLASSES} classes to {out}")
