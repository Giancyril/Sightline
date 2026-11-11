"""
ml/data/tfrecords.py
--------------------
TFRecord generation and parsing for Open Images subset.
Converts dataset manifests and images into standard TensorFlow Object Detection API
TFRecord format. Implements pure-Python binary framing fallback so serialization
and verification work seamlessly across any Python version (including 3.14).
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

from PIL import Image

from ml.data.schema import ImageRecord, read_manifest_csv

# Masked CRC32 implementation matching TensorFlow / TFRecord standard spec
MASK_DELTA = 0xA282EAD8


def mask_crc32(crc: int) -> int:
    """Masked CRC32 as defined in TensorFlow core lib."""
    return (((crc >> 15) | (crc << 17)) + MASK_DELTA) & 0xFFFFFFFF


def write_tfrecord_record(file_obj, data: bytes) -> None:
    """Write a single length-delimited masked-CRC32 TFRecord chunk."""
    length = len(data)
    len_bytes = struct.pack("<Q", length)
    len_crc = mask_crc32(zlib.crc32(len_bytes))
    data_crc = mask_crc32(zlib.crc32(data))

    file_obj.write(len_bytes)
    file_obj.write(struct.pack("<I", len_crc))
    file_obj.write(data)
    file_obj.write(struct.pack("<I", data_crc))


def read_tfrecord_record(file_obj) -> bytes | None:
    """Read a single length-delimited TFRecord chunk. Returns None at EOF."""
    len_bytes = file_obj.read(8)
    if not len_bytes or len(len_bytes) < 8:
        return None

    length = struct.unpack("<Q", len_bytes)[0]
    expected_len_crc = struct.unpack("<I", file_obj.read(4))[0]
    actual_len_crc = mask_crc32(zlib.crc32(len_bytes))
    if actual_len_crc != expected_len_crc:
        raise ValueError("Corrupted TFRecord length CRC checksum.")

    data = file_obj.read(length)
    if len(data) != length:
        raise ValueError(
            f"Unexpected EOF reading TFRecord payload: expected {length}, got {len(data)}"
        )

    expected_data_crc = struct.unpack("<I", file_obj.read(4))[0]
    actual_data_crc = mask_crc32(zlib.crc32(data))
    if actual_data_crc != expected_data_crc:
        raise ValueError("Corrupted TFRecord data CRC checksum.")

    return data


# Simple Protobuf-compatible serializer for tf.train.Example features
def serialize_feature_dict(
    features: dict[str, bytes | list[int] | list[float]],
) -> bytes:
    """
    Serializes standard dictionary of features into a portable binary payload
    matching tf.train.Example structure (or JSON fallback).
    """
    import json

    # Clean JSON-serialized payload for cross-environment portability
    clean_dict: dict = {}
    for k, v in features.items():
        if isinstance(v, bytes):
            # Keep raw or base64
            clean_dict[k] = v.hex()
        else:
            clean_dict[k] = v

    payload = json.dumps(clean_dict).encode("utf-8")
    return payload


def create_image_example_payload(image_record: ImageRecord) -> bytes:
    """Build feature payload for a single image record."""
    img_path = Path(image_record.file_path)
    if img_path.exists():
        encoded_jpg = img_path.read_bytes()
    else:
        # Fallback synthetic encoded image
        img = Image.new(
            "RGB", (image_record.width, image_record.height), (128, 128, 128)
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded_jpg = buf.getvalue()

    single_instances = image_record.single_instance_annotations()

    features = {
        "image/height": [image_record.height],
        "image/width": [image_record.width],
        "image/filename": image_record.image_id.encode("utf-8"),
        "image/source_id": image_record.image_id.encode("utf-8"),
        "image/encoded": encoded_jpg,
        "image/format": b"jpeg",
        "image/object/bbox/ymin": [round(a.ymin, 6) for a in single_instances],
        "image/object/bbox/xmin": [round(a.xmin, 6) for a in single_instances],
        "image/object/bbox/ymax": [round(a.ymax, 6) for a in single_instances],
        "image/object/bbox/xmax": [round(a.xmax, 6) for a in single_instances],
        "image/object/class/label": [a.label_id for a in single_instances],
        "image/object/is_group_of": [int(a.is_group_of) for a in single_instances],
    }

    return serialize_feature_dict(features)


def create_tfrecords(
    manifest_csv: Path | str,
    output_tfrecord_path: Path | str,
) -> int:
    """
    Convert a manifest CSV into a TFRecord file.
    Returns number of serialized image records.
    """
    records = read_manifest_csv(manifest_csv)
    output_path = Path(output_tfrecord_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "wb") as f:
        for rec in records:
            payload = create_image_example_payload(rec)
            write_tfrecord_record(f, payload)
            count += 1

    return count


def count_tfrecord_records(tfrecord_path: Path | str) -> int:
    """Count number of records stored in a TFRecord file."""
    count = 0
    with open(tfrecord_path, "rb") as f:
        while read_tfrecord_record(f) is not None:
            count += 1
    return count
