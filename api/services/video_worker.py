"""
api/services/video_worker.py
----------------------------
Background worker that processes a VideoJob:
  - Reads frames from the uploaded video using OpenCV
  - Runs per-frame detection via the existing DetectorService
  - Updates job progress in the shared store
  - Stores raw per-frame detections for later use by the annotation pipeline
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

from api.services.detector import UnknownModelError, get_detector
from api.services.video_jobs import JobState, VideoJob, VideoJobStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame result model
# ---------------------------------------------------------------------------


@dataclass
class FrameDetection:
    """Raw detection output for a single object in a single frame."""

    class_id: int
    class_name: str
    confidence: float
    # Normalized [0,1] coordinates: ymin, xmin, ymax, xmax
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    color_hex: str = "#0072B2"
    color_rgb: list[int] = field(default_factory=lambda: [0, 114, 178])


@dataclass
class FrameResult:
    """Detections for one processed frame."""

    frame_idx: int
    timestamp_s: float
    detections: list[FrameDetection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def process_video_job(
    job: VideoJob,
    store: VideoJobStore,
    *,
    frame_results_out: list[FrameResult] | None = None,
) -> None:
    """
    Process a video job in a background thread.

    Reads every ``job.frame_stride``-th frame, runs detection on it, and
    updates job progress.  On completion sets state to DONE; on any
    unhandled error sets state to FAILED.

    ``frame_results_out`` is an optional list that will be populated with
    ``FrameResult`` objects (useful for testing without touching disk).
    """
    import cv2  # noqa: PLC0415

    job.state = JobState.RUNNING
    job.touch()
    store.update(job)

    try:
        detector = get_detector(job.model_id)
    except UnknownModelError as exc:
        _fail(job, store, f"Unknown model '{job.model_id}': {exc}")
        return

    if job.video_path is None or not job.video_path.exists():
        _fail(job, store, "Video file missing or was deleted before processing.")
        return

    cap = cv2.VideoCapture(str(job.video_path))
    if not cap.isOpened():
        _fail(job, store, "OpenCV could not open the video file.")
        return

    fps = job.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_to_process = max(1, job.frame_count // job.frame_stride)
    job.total_frames_to_process = total_to_process
    store.update(job)

    frames_done = 0
    frame_idx = 0
    results: list[FrameResult] = []

    try:
        while True:
            # Allow cancellation
            current = store.get(job.job_id)
            if current is None or current.state == JobState.FAILED:
                logger.info("Job %s cancelled — stopping worker.", job.job_id)
                cap.release()
                return

            ret, bgr_frame = cap.read()
            if not ret:
                break  # end of video

            if frame_idx % job.frame_stride != 0:
                frame_idx += 1
                continue

            timestamp_s = frame_idx / fps

            # Encode frame to JPEG bytes for the detector
            ok, jpeg_buf = cv2.imencode(
                ".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                frame_idx += 1
                continue
            image_bytes = io.BytesIO(jpeg_buf.tobytes()).read()

            try:
                result = detector.detect(
                    image_bytes,
                    confidence_threshold=job.confidence_threshold,
                    nms_iou_threshold=job.nms_iou_threshold,
                )
                frame_dets = _parse_detections(result.get("detections", []))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Detection failed on frame %d of job %s: %s",
                    frame_idx,
                    job.job_id,
                    exc,
                )
                frame_dets = []

            fr = FrameResult(
                frame_idx=frame_idx,
                timestamp_s=round(timestamp_s, 4),
                detections=frame_dets,
            )
            results.append(fr)
            if frame_results_out is not None:
                frame_results_out.append(fr)

            frames_done += 1
            job.frames_processed = frames_done
            job.progress = min(1.0, frames_done / total_to_process)
            job.touch()
            store.update(job)

            frame_idx += 1
    finally:
        cap.release()

    # Store results on job for downstream pipeline (Commit 7 annotation)
    job.frame_results = results  # type: ignore[attr-defined]  # dynamic attribute

    job.state = JobState.DONE
    job.progress = 1.0
    job.frames_processed = frames_done
    job.touch()
    store.update(job)
    logger.info(
        "Job %s finished — processed %d frames in %.1f s.",
        job.job_id,
        frames_done,
        job.duration_s,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fail(job: VideoJob, store: VideoJobStore, reason: str) -> None:
    job.state = JobState.FAILED
    job.error = reason
    job.touch()
    store.update(job)
    logger.error("Job %s failed: %s", job.job_id, reason)


def _parse_detections(raw: list[dict[str, Any]]) -> list[FrameDetection]:
    out: list[FrameDetection] = []
    for d in raw:
        box = d.get("box_normalized", {})
        out.append(
            FrameDetection(
                class_id=d.get("class_id", 0),
                class_name=d.get("class_name", "unknown"),
                confidence=d.get("confidence", 0.0),
                ymin=box.get("ymin", 0.0),
                xmin=box.get("xmin", 0.0),
                ymax=box.get("ymax", 1.0),
                xmax=box.get("xmax", 1.0),
                color_hex=d.get("color_hex", "#0072B2"),
                color_rgb=d.get("color_rgb", [0, 114, 178]),
            )
        )
    return out
