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

import csv
import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from api.services.detector import UnknownModelError, get_detector
from api.services.video_jobs import JobState, VideoJob, VideoJobStore
from ml.utils.counting_line import CountingLine
from ml.utils.tracker import IoUTracker

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
    Process a video job in a background thread:
      - Reads frames from video file
      - Runs detection per frame
      - Tracks objects with persistent IDs and smoothing
      - Tallies counting line crossings
      - Writes annotated MP4, tracks.csv, and summary.json
      - Updates job state and progress
    """
    import cv2  # noqa: PLC0415

    job.state = JobState.RUNNING
    job.touch()
    store.update(job)

    start_time = time.time()

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
    w = job.width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = job.height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_to_process = max(1, job.frame_count // job.frame_stride)
    job.total_frames_to_process = total_to_process
    store.update(job)

    # Output file paths
    parent_dir = job.video_path.parent
    out_video_path = parent_dir / f"annotated_{job.job_id}.mp4"
    out_csv_path = parent_dir / f"tracks_{job.job_id}.csv"
    out_json_path = parent_dir / f"summary_{job.job_id}.json"

    # Initialize tracker and counting line
    tracker = IoUTracker(
        iou_threshold=0.3,
        max_age=5,
        class_consistency=True,
        smooth_factor=0.8,
    )

    if job.counting_line and len(job.counting_line) >= 2:
        p1 = job.counting_line[0]
        p2 = job.counting_line[1]
        counting_line = CountingLine(p1=p1, p2=p2)
    else:
        # Default counting line across center
        counting_line = CountingLine(p1=(0.0, 0.5), p2=(1.0, 0.5))

    # Initialize VideoWriter
    out_fps = max(1.0, fps / job.frame_stride)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, out_fps, (w, h))

    frames_done = 0
    frame_idx = 0
    results: list[FrameResult] = []
    csv_rows: list[dict[str, Any]] = []

    try:
        while True:
            # Allow cancellation
            current = store.get(job.job_id)
            if current is None or current.state == JobState.FAILED:
                logger.info("Job %s cancelled — stopping worker.", job.job_id)
                cap.release()
                writer.release()
                return

            ret, bgr_frame = cap.read()
            if not ret:
                break  # end of video

            if frame_idx % job.frame_stride != 0:
                frame_idx += 1
                continue

            timestamp_s = frame_idx / fps

            # Encode frame to JPEG bytes for detector
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

            # Update tracker and counting line
            active_tracks = tracker.update(frame_dets, frame_idx=frame_idx)
            counting_line.update(
                active_tracks, frame_idx=frame_idx, timestamp_s=timestamp_s
            )

            # Record CSV rows
            for trk in active_tracks:
                ymin, xmin, ymax, xmax = trk.bbox
                box_str = f"[{ymin:.4f}, {xmin:.4f}, {ymax:.4f}, {xmax:.4f}]"
                csv_rows.append(
                    {
                        "frame": frame_idx,
                        "id": trk.track_id,
                        "class": trk.class_name,
                        "box": box_str,
                        "score": round(trk.score, 4),
                    }
                )

            # Draw annotations on bgr_frame
            _draw_annotations(bgr_frame, active_tracks, counting_line, w, h)
            writer.write(bgr_frame)

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
        writer.release()

    # Write tracks.csv
    try:
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame", "id", "class", "box", "score"]
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(csv_rows)
        job.tracks_csv_path = out_csv_path
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed writing tracks.csv for job %s: %s", job.job_id, exc)

    # Compute per-class unique track count
    unique_by_class: dict[str, int] = {}
    for trk in tracker.tracks:
        unique_by_class[trk.class_name] = unique_by_class.get(trk.class_name, 0) + 1

    elapsed_s = max(0.001, time.time() - start_time)
    fps_processed = round(frames_done / elapsed_s, 2)

    # Write summary.json
    try:
        summary_data = {
            "job_id": job.job_id,
            "unique_counts": len(tracker.tracks),
            "counts_per_class": unique_by_class,
            "line_counts": counting_line.get_summary(),
            "fps_processed": fps_processed,
            "total_frames_processed": frames_done,
            "duration_s": round(job.duration_s, 2),
        }
        out_json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        job.summary_json_path = out_json_path
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed writing summary.json for job %s: %s", job.job_id, exc)

    job.annotated_video_path = out_video_path
    job.frame_results = results  # type: ignore[attr-defined]
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


def _draw_annotations(
    frame: Any,
    tracks: list[Any],
    counting_line: CountingLine,
    width: int,
    height: int,
) -> None:
    """Draw bounding boxes with track IDs and counting line on BGR frame."""
    import cv2  # noqa: PLC0415

    # 1. Draw counting line
    lx1 = int(round(counting_line.p1.x * width))
    ly1 = int(round(counting_line.p1.y * height))
    lx2 = int(round(counting_line.p2.x * width))
    ly2 = int(round(counting_line.p2.y * height))
    cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 230, 255), 2, cv2.LINE_AA)

    # 2. Draw line count banner at top-left
    banner = f"IN: {counting_line.total_in}  |  OUT: {counting_line.total_out}"
    (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (8, 8), (16 + bw, 14 + bh), (20, 20, 20), -1)
    cv2.putText(
        frame,
        banner,
        (12, 10 + bh),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )

    # 3. Draw active tracks with persistent IDs
    for trk in tracks:
        ymin, xmin, ymax, xmax = trk.bbox
        py1 = max(0, int(round(ymin * height)))
        px1 = max(0, int(round(xmin * width)))
        py2 = min(height - 1, int(round(ymax * height)))
        px2 = min(width - 1, int(round(xmax * width)))

        # Color: RGB to BGR
        rgb = trk.color_rgb if hasattr(trk, "color_rgb") else [0, 114, 178]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        cv2.rectangle(frame, (px1, py1), (px2, py2), bgr, 2)

        # Label tag: #ID Class score
        lbl = f"#{trk.track_id} {trk.class_name} {trk.score:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, py1 - th - 6)
        cv2.rectangle(frame, (px1, tag_y1), (px1 + tw + 6, py1), bgr, -1)
        cv2.putText(
            frame,
            lbl,
            (px1 + 3, py1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
