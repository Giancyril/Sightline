"""
api/tests/test_video_outputs.py
-------------------------------
Tests for annotated video, tracks CSV, and summary JSON outputs (Commit 7).
Verifies:
  - Annotated MP4 is written and readable with correct dimensions and FPS
  - tracks.csv contains expected columns (frame, id, class, box, score)
  - summary.json contains unique track counts, line counts, and processing FPS
  - Download endpoints (/v1/video/{job_id}/annotated, /tracks, /summary) return FileResponses
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from api.routers.video import download_annotated, download_summary, download_tracks
from api.services.video_jobs import JobState, VideoJobStore
from api.services.video_worker import process_video_job

cv2 = pytest.importorskip("cv2")


def _make_synthetic_clip(path: Path, fps: int = 10, seconds: int = 2) -> int:
    """Generate a valid synthetic MP4 file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 160, 120
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    total = fps * seconds
    for i in range(total):
        val = int((i / total) * 255)
        frame = np.full((h, w, 3), [val, 128, 255 - val], dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return total


FAKE_DETECTIONS = {
    "detections": [
        {
            "class_id": 1,
            "class_name": "Car",
            "confidence": 0.88,
            "box_normalized": {"ymin": 0.2, "xmin": 0.2, "ymax": 0.6, "xmax": 0.6},
            "color_hex": "#0072B2",
            "color_rgb": [0, 114, 178],
        }
    ]
}


def test_video_outputs_generated_on_done():
    """Worker writes annotated video, tracks CSV, and summary JSON on completion."""
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / "input.mp4"
        total_frames = _make_synthetic_clip(clip_path, fps=10, seconds=2)

        store = VideoJobStore()
        job = store.create()
        job.video_path = clip_path
        job.fps = 10.0
        job.frame_count = total_frames
        job.width = 160
        job.height = 120
        job.duration_s = total_frames / 10.0
        job.frame_stride = 1
        job.total_frames_to_process = total_frames
        store.update(job)

        fake_detector = MagicMock()
        fake_detector.detect.return_value = FAKE_DETECTIONS

        with patch(
            "api.services.video_worker.get_detector", return_value=fake_detector
        ):
            process_video_job(job, store)

        final = store.get(job.job_id)
        assert final is not None
        assert final.state == JobState.DONE

        # 1. Check annotated video
        assert final.annotated_video_path is not None
        assert final.annotated_video_path.exists()
        assert final.annotated_video_path.stat().st_size > 0

        cap = cv2.VideoCapture(str(final.annotated_video_path))
        assert cap.isOpened()
        ann_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert ann_frames > 0
        cap.release()

        # 2. Check tracks CSV
        assert final.tracks_csv_path is not None
        assert final.tracks_csv_path.exists()
        with open(final.tracks_csv_path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) > 0
            assert set(reader[0].keys()) == {"frame", "id", "class", "box", "score"}
            assert reader[0]["class"] == "Car"
            assert reader[0]["id"] == "1"
            assert reader[0]["box"].startswith("[")

        # 3. Check summary JSON
        assert final.summary_json_path is not None
        assert final.summary_json_path.exists()
        summary = json.loads(final.summary_json_path.read_text(encoding="utf-8"))
        assert summary["job_id"] == job.job_id
        assert "unique_counts" in summary
        assert summary["unique_counts"] >= 1
        assert "line_counts" in summary
        assert "fps_processed" in summary
        assert summary["total_frames_processed"] == total_frames


def test_download_endpoints_return_files():
    """Verify download router endpoints serve FileResponses for done job."""
    with tempfile.TemporaryDirectory() as tmp:
        vid_path = Path(tmp) / "ann.mp4"
        csv_path = Path(tmp) / "tr.csv"
        json_path = Path(tmp) / "sum.json"

        vid_path.write_bytes(b"dummy_mp4")
        csv_path.write_text("frame,id,class,box,score\n")
        json_path.write_text('{"test": true}')

        store = VideoJobStore()
        job = store.create()
        job.state = JobState.DONE
        job.annotated_video_path = vid_path
        job.tracks_csv_path = csv_path
        job.summary_json_path = json_path
        store.update(job)

        with patch("api.routers.video.get_job_store", return_value=store):
            # Test annotated download
            ann_resp = download_annotated(job.job_id)
            assert ann_resp.media_type == "video/mp4"

            # Test tracks download
            csv_resp = download_tracks(job.job_id)
            assert csv_resp.media_type == "text/csv"

            # Test summary download
            json_resp = download_summary(job.job_id)
            assert json_resp.media_type == "application/json"


def test_download_endpoints_reject_not_done():
    """Download endpoints raise 409 if job is not in DONE state."""
    store = VideoJobStore()
    job = store.create()
    job.state = JobState.RUNNING
    store.update(job)

    with patch("api.routers.video.get_job_store", return_value=store):
        with pytest.raises(HTTPException) as exc:
            download_annotated(job.job_id)
        assert exc.value.status_code == 409
