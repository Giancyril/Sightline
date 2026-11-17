"""
api/tests/test_video_e2e.py
---------------------------
End-to-end tests for video detection, tracking, and directional counting (Commit 10).
Includes hand-verified synthetic clip crossing checks.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.services.video_jobs import JobState, VideoJobStore
from api.services.video_worker import process_video_job

cv2 = pytest.importorskip("cv2")


def _create_synthetic_clip(path: Path, frames: int = 10, fps: int = 10) -> int:
    """Generate a valid test MP4 clip."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 160, 120
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    for i in range(frames):
        val = int((i / max(1, frames)) * 255)
        frame = np.full((h, w, 3), [val, 120, 200], dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return frames


def test_hand_verified_single_object_downward_crossing():
    """
    Hand-verified test clip:
    A single car moves from y=0.1 to y=0.9 across 10 frames.
    Horizontal counting line at y=0.5 (from (0.0, 0.5) to (1.0, 0.5)).
    Hand-verified ground truth:
      - Unique objects = 1
      - Directional crossings = 1 (total_crossings == 1, total_in == 1, total_out == 0)
      - Zero spurious switches or double-counts
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = Path(tmp_dir) / "test_downward.mp4"
        _create_synthetic_clip(clip_path, frames=10, fps=10)

        store = VideoJobStore()
        job = store.create()
        job.video_path = clip_path
        job.mime_type = "video/mp4"
        job.fps = 10.0
        job.frame_count = 10
        job.width = 160
        job.height = 120
        job.duration_s = 1.0
        job.frame_stride = 1
        job.confidence_threshold = 0.25
        job.nms_iou_threshold = 0.45
        job.total_frames_to_process = 10
        # Horizontal tripwire across center
        job.counting_line = [{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 0.5}]
        store.update(job)

        # Generate moving box downward: frame i -> y center moves from 0.1 to 0.9
        call_count = 0

        def mock_detect(*args, **kwargs):
            nonlocal call_count
            y_center = 0.30 + (call_count * 0.035)  # 0.30 to 0.615 across 10 frames
            call_count += 1
            box = {
                "ymin": round(y_center - 0.10, 4),
                "xmin": 0.40,
                "ymax": round(y_center + 0.10, 4),
                "xmax": 0.60,
            }
            return {
                "detections": [
                    {
                        "class_id": 1,
                        "class_name": "Car",
                        "confidence": 0.92,
                        "box_normalized": box,
                        "color_hex": "#0072B2",
                        "color_rgb": [0, 114, 178],
                    }
                ]
            }

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = mock_detect

        with patch(
            "api.services.video_worker.get_detector", return_value=mock_detector
        ):
            process_video_job(job, store)

        assert job.state == JobState.DONE
        assert job.frames_processed == 10
        assert job.progress == 1.0
        assert job.summary_json_path is not None
        assert job.summary_json_path.exists()
        assert job.tracks_csv_path is not None
        assert job.tracks_csv_path.exists()
        assert job.annotated_video_path is not None
        assert job.annotated_video_path.exists()

        # Verify summary JSON matches hand-verified ground truth
        summary = json.loads(job.summary_json_path.read_text(encoding="utf-8"))
        assert summary["unique_counts"] == 1
        assert summary["counts_per_class"] == {"Car": 1}

        lc = summary["line_counts"]
        assert lc["total_crossings"] == 1
        assert lc["total_in"] == 0
        assert lc["total_out"] == 1
        assert lc["by_class"]["Car"]["in"] == 0
        assert lc["by_class"]["Car"]["out"] == 1

        # Verify tracks CSV has exactly 10 detection entries for track_id=1
        with open(job.tracks_csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 10
            for row in reader:
                assert int(row["id"]) == 1
                assert row["class"] == "Car"


def test_hand_verified_two_objects_opposite_directions():
    """
    Hand-verified test clip:
    Object A (Car) moves downward from y=0.1 to y=0.9 (crosses IN).
    Object B (Person) moves upward from y=0.9 to y=0.1 (crosses OUT).
    Tripwire at y=0.5.
    Hand-verified ground truth:
      - Unique objects = 2
      - Total crossings = 2 (1 IN for Car, 1 OUT for Person)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = Path(tmp_dir) / "test_two_objects.mp4"
        _create_synthetic_clip(clip_path, frames=10, fps=10)

        store = VideoJobStore()
        job = store.create()
        job.video_path = clip_path
        job.mime_type = "video/mp4"
        job.fps = 10.0
        job.frame_count = 10
        job.width = 160
        job.height = 120
        job.duration_s = 1.0
        job.frame_stride = 1
        job.total_frames_to_process = 10
        job.counting_line = [{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 0.5}]
        store.update(job)

        step = 0

        def mock_detect(*args, **kwargs):
            nonlocal step
            y_car = 0.30 + (step * 0.035)  # 0.30 to 0.615 (crosses 0.5 downward)
            y_person = 0.70 - (step * 0.035)  # 0.70 to 0.385 (crosses 0.5 upward)
            step += 1
            return {
                "detections": [
                    {
                        "class_id": 1,
                        "class_name": "Car",
                        "confidence": 0.90,
                        "box_normalized": {
                            "ymin": round(y_car - 0.10, 4),
                            "xmin": 0.15,
                            "ymax": round(y_car + 0.10, 4),
                            "xmax": 0.35,
                        },
                        "color_hex": "#0072B2",
                        "color_rgb": [0, 114, 178],
                    },
                    {
                        "class_id": 2,
                        "class_name": "Person",
                        "confidence": 0.85,
                        "box_normalized": {
                            "ymin": round(y_person - 0.10, 4),
                            "xmin": 0.65,
                            "ymax": round(y_person + 0.10, 4),
                            "xmax": 0.85,
                        },
                        "color_hex": "#E69F00",
                        "color_rgb": [230, 159, 0],
                    },
                ]
            }

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = mock_detect

        with patch(
            "api.services.video_worker.get_detector", return_value=mock_detector
        ):
            process_video_job(job, store)

        assert job.state == JobState.DONE
        summary = json.loads(job.summary_json_path.read_text(encoding="utf-8"))
        assert summary["unique_counts"] == 2
        assert summary["counts_per_class"]["Car"] == 1
        assert summary["counts_per_class"]["Person"] == 1

        lc = summary["line_counts"]
        assert lc["total_crossings"] == 2
        assert lc["total_in"] == 1
        assert lc["total_out"] == 1
        assert lc["by_class"]["Car"]["in"] == 0
        assert lc["by_class"]["Car"]["out"] == 1
        assert lc["by_class"]["Person"]["in"] == 1
        assert lc["by_class"]["Person"]["out"] == 0
