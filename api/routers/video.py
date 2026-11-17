"""
api/routers/video.py
--------------------
Router for Day 2 video upload, processing status, and results.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from api.services.video_jobs import (
    MAX_VIDEO_DURATION_S,
    JobState,
    VideoJob,
    VideoValidationError,
    get_job_store,
    validate_video_bytes,
)

router = APIRouter(prefix="/v1/video", tags=["Video"])

# Lazy import to avoid circular deps / import-time GPU init
_TEMP_DIR = Path(tempfile.gettempdir()) / "vision_ssd_video"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", summary="Upload video and queue detection job")
def upload_video(
    file: UploadFile,
    model: str = Query(default="fine-tuned", description="Model to use for detection"),
    frame_stride: int = Query(
        default=3, ge=1, le=30, description="Process every N-th frame"
    ),
    confidence_threshold: float = Query(default=0.25, ge=0.01, le=1.0),
    nms_iou_threshold: float = Query(default=0.45, ge=0.01, le=1.0),
) -> dict[str, Any]:
    """
    Upload an MP4 or WebM clip (max 50 MB, max 30 s) and queue a detection+tracking job.
    Returns a job_id for polling.
    """
    data = file.file.read()
    filename = file.filename or "upload.mp4"

    try:
        mime_type = validate_video_bytes(data, filename)
    except VideoValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # Persist to temp file
    suffix = ".mp4" if "mp4" in mime_type else ".webm"
    tmp_path = _TEMP_DIR / f"upload_{__import__('uuid').uuid4().hex}{suffix}"
    tmp_path.write_bytes(data)

    # Quick duration check with OpenCV
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(tmp_path))
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = frame_count / fps_val if fps_val > 0 else 0.0
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail="Could not read video metadata."
        ) from None

    if duration > MAX_VIDEO_DURATION_S:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Video duration {duration:.1f}s exceeds the {MAX_VIDEO_DURATION_S}s limit.",
        )

    if frame_count == 0 or w == 0 or h == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail="Video contains no readable frames."
        )

    # Create job
    store = get_job_store()
    job = store.create()
    job.video_path = tmp_path
    job.mime_type = mime_type
    job.file_size_bytes = len(data)
    job.fps = fps_val
    job.frame_count = frame_count
    job.width = w
    job.height = h
    job.duration_s = duration
    job.model_id = model
    job.frame_stride = frame_stride
    job.confidence_threshold = confidence_threshold
    job.nms_iou_threshold = nms_iou_threshold
    job.total_frames_to_process = max(1, frame_count // frame_stride)
    store.update(job)

    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "fps": round(job.fps, 2),
        "duration_s": round(job.duration_s, 2),
        "frame_count": job.frame_count,
        "total_frames_to_process": job.total_frames_to_process,
        "width": job.width,
        "height": job.height,
        "mime_type": job.mime_type,
        "file_size_bytes": job.file_size_bytes,
    }


@router.get("/{job_id}", summary="Get job status and results")
def get_job_status(job_id: str) -> dict[str, Any]:
    """Return current state, progress, and result links for a video job."""
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    result = job.to_status_dict()

    if job.state == JobState.DONE:
        result["results"] = {
            "annotated_video": f"/v1/video/{job_id}/annotated",
            "tracks_csv": f"/v1/video/{job_id}/tracks",
            "summary_json": f"/v1/video/{job_id}/summary",
        }

    return result


@router.delete("/{job_id}", summary="Cancel or delete a video job")
def delete_job(job_id: str) -> dict[str, str]:
    """Cancel a queued/running job or delete a completed one, cleaning up temp files."""
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Mark as failed to signal worker to stop
    if job.state in (JobState.QUEUED, JobState.RUNNING):
        job.state = JobState.FAILED
        job.error = "Cancelled by client."
        store.update(job)

    # Remove files and store entry
    _cleanup_job(job)
    store.delete(job_id)
    return {"deleted": job_id}


@router.get("/{job_id}/annotated", summary="Download annotated video")
def download_annotated(job_id: str) -> FileResponse:
    job = _require_done_job(job_id)
    if not job.annotated_video_path or not job.annotated_video_path.exists():
        raise HTTPException(
            status_code=404, detail="Annotated video not yet available."
        )
    return FileResponse(
        path=str(job.annotated_video_path),
        media_type="video/mp4",
        filename=f"annotated_{job_id}.mp4",
    )


@router.get("/{job_id}/tracks", summary="Download tracks CSV")
def download_tracks(job_id: str) -> FileResponse:
    job = _require_done_job(job_id)
    if not job.tracks_csv_path or not job.tracks_csv_path.exists():
        raise HTTPException(status_code=404, detail="Tracks CSV not yet available.")
    return FileResponse(
        path=str(job.tracks_csv_path),
        media_type="text/csv",
        filename=f"tracks_{job_id}.csv",
    )


@router.get("/{job_id}/summary", summary="Download summary JSON")
def download_summary(job_id: str) -> FileResponse:
    job = _require_done_job(job_id)
    if not job.summary_json_path or not job.summary_json_path.exists():
        raise HTTPException(status_code=404, detail="Summary JSON not yet available.")
    return FileResponse(
        path=str(job.summary_json_path),
        media_type="application/json",
        filename=f"summary_{job_id}.json",
    )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _require_done_job(job_id: str) -> VideoJob:
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.state != JobState.DONE:
        raise HTTPException(
            status_code=409, detail=f"Job is not done yet (state: {job.state.value})."
        )
    return job


def _cleanup_job(job: VideoJob) -> None:
    for attr in (
        "video_path",
        "annotated_video_path",
        "tracks_csv_path",
        "summary_json_path",
    ):
        p = getattr(job, attr, None)
        if p and Path(p).exists():
            try:
                Path(p).unlink()
            except OSError:
                pass
