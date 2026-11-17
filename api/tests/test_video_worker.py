"""
api/tests/test_video_worker.py
------------------------------
Tests for the frame-extraction / per-frame detection worker (Commit 2).
Uses a small synthetic MP4 clip generated with OpenCV VideoWriter so no
real video file or model weights are required.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.services.video_jobs import JobState, VideoJob, VideoJobStore
from api.services.video_worker import FrameResult, _parse_detections, process_video_job

cv2 = pytest.importorskip("cv2")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_clip(path: Path, fps: int = 10, seconds: int = 2) -> int:
    """Write a small coloured synthetic MP4; return total frame count."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 160, 120
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    total = fps * seconds
    for i in range(total):
        colour = int((i / total) * 255)
        frame = np.full((h, w, 3), [colour, 128, 255 - colour], dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return total


def _make_job(video_path: Path, fps: int = 10, total_frames: int = 20) -> VideoJob:
    store = VideoJobStore()
    job = store.create()
    job.video_path = video_path
    job.fps = float(fps)
    job.frame_count = total_frames
    job.width = 160
    job.height = 120
    job.duration_s = total_frames / fps
    job.frame_stride = 1
    job.model_id = "fine-tuned"
    job.confidence_threshold = 0.25
    job.nms_iou_threshold = 0.45
    job.total_frames_to_process = total_frames
    store.update(job)
    return job, store


# ---------------------------------------------------------------------------
# _parse_detections unit tests
# ---------------------------------------------------------------------------


def test_parse_detections_empty():
    assert _parse_detections([]) == []


def test_parse_detections_one_object():
    raw = [
        {
            "class_id": 1,
            "class_name": "Person",
            "confidence": 0.9,
            "box_normalized": {"ymin": 0.1, "xmin": 0.2, "ymax": 0.8, "xmax": 0.7},
            "color_hex": "#FF0000",
            "color_rgb": [255, 0, 0],
        }
    ]
    result = _parse_detections(raw)
    assert len(result) == 1
    det = result[0]
    assert det.class_id == 1
    assert det.class_name == "Person"
    assert abs(det.confidence - 0.9) < 1e-6
    assert abs(det.ymin - 0.1) < 1e-6
    assert det.color_hex == "#FF0000"


# ---------------------------------------------------------------------------
# Worker integration test with mock detector
# ---------------------------------------------------------------------------


FAKE_DETECT_RESPONSE = {
    "detections": [
        {
            "class_id": 2,
            "class_name": "Car",
            "confidence": 0.75,
            "box_normalized": {"ymin": 0.0, "xmin": 0.0, "ymax": 0.5, "xmax": 0.5},
            "color_hex": "#0072B2",
            "color_rgb": [0, 114, 178],
        }
    ]
}


def test_worker_completes_with_synthetic_clip():
    """Worker runs to DONE state on a real (synthetic) video file."""
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / "test_clip.mp4"
        total_frames = _make_synthetic_clip(clip_path, fps=10, seconds=2)

        job, store = _make_job(clip_path, fps=10, total_frames=total_frames)
        frame_results: list[FrameResult] = []

        fake_detector = MagicMock()
        fake_detector.detect.return_value = FAKE_DETECT_RESPONSE

        with patch(
            "api.services.video_worker.get_detector", return_value=fake_detector
        ):
            process_video_job(job, store, frame_results_out=frame_results)

        final = store.get(job.job_id)
        assert final is not None
        assert final.state == JobState.DONE
        assert final.progress == 1.0
        assert final.frames_processed == total_frames
        assert len(frame_results) == total_frames
        # Each frame should have one detection from our mock
        assert all(len(fr.detections) == 1 for fr in frame_results)
        assert frame_results[0].detections[0].class_name == "Car"


def test_worker_respects_frame_stride():
    """With stride=5 only every 5th frame is processed."""
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / "stride_clip.mp4"
        total_frames = _make_synthetic_clip(clip_path, fps=10, seconds=2)

        job, store = _make_job(clip_path, fps=10, total_frames=total_frames)
        job.frame_stride = 5
        job.total_frames_to_process = total_frames // 5
        store.update(job)
        frame_results: list[FrameResult] = []

        fake_detector = MagicMock()
        fake_detector.detect.return_value = {"detections": []}

        with patch(
            "api.services.video_worker.get_detector", return_value=fake_detector
        ):
            process_video_job(job, store, frame_results_out=frame_results)

        final = store.get(job.job_id)
        assert final is not None
        assert final.state == JobState.DONE
        # Processed frames should equal total / stride
        assert len(frame_results) == total_frames // 5


def test_worker_fails_on_missing_file():
    """Worker marks job FAILED when video file does not exist."""
    store = VideoJobStore()
    job = store.create()
    job.video_path = Path("/nonexistent/clip.mp4")
    job.frame_count = 10
    job.fps = 10.0
    job.frame_stride = 1
    job.total_frames_to_process = 10
    store.update(job)

    with patch("api.services.video_worker.get_detector"):
        process_video_job(job, store)

    final = store.get(job.job_id)
    assert final is not None
    assert final.state == JobState.FAILED
    assert final.error is not None


def test_worker_stops_on_cancel():
    """Worker stops early when job is marked FAILED (cancelled) mid-run."""
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / "long_clip.mp4"
        # Enough frames to cancel mid-way
        total_frames = _make_synthetic_clip(clip_path, fps=10, seconds=3)

        job, store = _make_job(clip_path, fps=10, total_frames=total_frames)
        job.frame_stride = 1
        job.total_frames_to_process = total_frames
        store.update(job)

        call_count = 0
        cancel_after = 5

        def cancelling_detect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= cancel_after:
                # Simulate cancellation from another thread
                j = store.get(job.job_id)
                if j:
                    j.state = JobState.FAILED
                    j.error = "Cancelled by client."
                    store.update(j)
            return {"detections": []}

        fake_detector = MagicMock()
        fake_detector.detect.side_effect = cancelling_detect

        with patch(
            "api.services.video_worker.get_detector", return_value=fake_detector
        ):
            process_video_job(job, store)

        final = store.get(job.job_id)
        # Could be None (deleted) or FAILED — either is acceptable
        assert final is None or final.state == JobState.FAILED
        # Should have processed far fewer frames than the total
        assert call_count < total_frames


def test_worker_in_background_thread():
    """Verify worker can be started in a daemon thread and completes."""
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / "bg_clip.mp4"
        total_frames = _make_synthetic_clip(clip_path, fps=10, seconds=1)

        job, store = _make_job(clip_path, fps=10, total_frames=total_frames)

        fake_detector = MagicMock()
        fake_detector.detect.return_value = {"detections": []}

        done = threading.Event()

        def _run():
            with patch(
                "api.services.video_worker.get_detector", return_value=fake_detector
            ):
                process_video_job(job, store)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        finished = done.wait(timeout=30)

        assert finished, "Worker thread did not complete within 30 s"
        final = store.get(job.job_id)
        assert final is not None
        assert final.state == JobState.DONE
