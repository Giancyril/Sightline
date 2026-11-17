"""
api/services/video_jobs.py
--------------------------
In-memory video job store with thread-safe state management.
Jobs are auto-expired and purged after 1 hour (TTL enforced by cleanup task).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_VIDEO_DURATION_S = 30.0  # 30 seconds
JOB_TTL_S = 3600.0  # 1 hour

ALLOWED_VIDEO_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x00\x00\x00\x08ftyp", "video/mp4"),  # MP4 ftyp at offset 4 — handled below
    (b"\x1aE\xdf\xa3", "video/webm"),  # WebM / MKV EBML
    (b"\x00\x00\x00\x14ftyp", "video/mp4"),
    (b"\x00\x00\x00\x18ftyp", "video/mp4"),
    (b"\x00\x00\x00\x1cftyp", "video/mp4"),
    (b"\x00\x00\x00 ftyp", "video/mp4"),
    (b"\x00\x00\x00$ftyp", "video/mp4"),
    (b"ftyp", "video/mp4"),  # rare — ftyp without leading size
    (b"RIFF", "video/webm"),  # AVI / other RIFF (rejected below by ext)
]


# -------------------------------------------------------------------------
# Job model
# -------------------------------------------------------------------------


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class VideoJob:
    """Represents a single video processing job."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: JobState = JobState.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Inputs (populated after validation)
    video_path: Path | None = None
    mime_type: str = ""
    file_size_bytes: int = 0
    duration_s: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0

    # Processing config
    model_id: str = "fine-tuned"
    frame_stride: int = 3
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45

    # Progress
    progress: float = 0.0  # 0.0 to 1.0
    frames_processed: int = 0
    total_frames_to_process: int = 0

    # Outputs (populated on completion)
    annotated_video_path: Path | None = None
    tracks_csv_path: Path | None = None
    summary_json_path: Path | None = None

    # Error
    error: str | None = None

    # Counting line (optional, two normalized [0,1] points)
    counting_line: list[dict[str, float]] | None = None

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_status_dict(self) -> dict[str, Any]:
        """Return a minimal status dict suitable for API responses."""
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "frames_processed": self.frames_processed,
            "total_frames_to_process": self.total_frames_to_process,
            "fps": round(self.fps, 2),
            "duration_s": round(self.duration_s, 2),
            "width": self.width,
            "height": self.height,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_results": self.state == JobState.DONE,
        }


# -------------------------------------------------------------------------
# Job store
# -------------------------------------------------------------------------


class VideoJobStore:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, VideoJob] = {}

    def create(self) -> VideoJob:
        job = VideoJob()
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: VideoJob) -> None:
        job.touch()
        with self._lock:
            self._jobs[job.job_id] = job

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def all_jobs(self) -> list[VideoJob]:
        with self._lock:
            return list(self._jobs.values())

    def purge_expired(self, ttl_s: float = JOB_TTL_S) -> int:
        """Delete jobs older than ttl_s. Returns count of purged jobs."""
        cutoff = time.time() - ttl_s
        to_delete: list[VideoJob] = []
        with self._lock:
            for jid, job in list(self._jobs.items()):
                if job.created_at < cutoff:
                    to_delete.append(job)
                    self._jobs.pop(jid, None)
        for job in to_delete:
            _remove_job_files(job)
        return len(to_delete)


class VideoCleanupTask:
    """Background periodic task that purges expired jobs and their temp files."""

    def __init__(
        self,
        store: VideoJobStore | None = None,
        interval_s: float = 600.0,
        ttl_s: float = JOB_TTL_S,
    ) -> None:
        self.store = store or get_job_store()
        self.interval_s = interval_s
        self.ttl_s = ttl_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="video-cleanup-task",
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self.store.purge_expired(self.ttl_s)
            except Exception:
                pass

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run_once(self) -> int:
        """Run a single purge pass synchronously."""
        return self.store.purge_expired(self.ttl_s)


def start_cleanup_task(
    store: VideoJobStore | None = None,
    interval_s: float = 600.0,
    ttl_s: float = JOB_TTL_S,
) -> VideoCleanupTask:
    task = VideoCleanupTask(store=store, interval_s=interval_s, ttl_s=ttl_s)
    task.start()
    return task


# -------------------------------------------------------------------------
# File cleanup helper
# -------------------------------------------------------------------------


def _remove_job_files(job: VideoJob) -> None:
    """Best-effort removal of all temp files associated with a job."""
    for attr in (
        "video_path",
        "annotated_video_path",
        "tracks_csv_path",
        "summary_json_path",
    ):
        p: Path | None = getattr(job, attr, None)
        if p and p.exists():
            try:
                p.unlink()
            except OSError:
                pass


# -------------------------------------------------------------------------
# Video magic-byte validation
# -------------------------------------------------------------------------


def _is_mp4(data: bytes) -> bool:
    """Check if data contains an MP4 ftyp box at standard positions."""
    # ftyp box: 4-byte big-endian size, then b"ftyp"
    if len(data) < 8:
        return False
    for offset in range(0, min(64, len(data) - 4)):
        if data[offset : offset + 4] == b"ftyp":
            return True
    return False


def _is_webm(data: bytes) -> bool:
    """Check EBML header for WebM/MKV."""
    return data[:4] == b"\x1a\x45\xdf\xa3"


def validate_video_bytes(data: bytes, filename: str) -> str:
    """
    Validate uploaded video bytes.
    Returns MIME type string.
    Raises VideoValidationError on failure.
    """
    if len(data) == 0:
        raise VideoValidationError("Empty video payload.", status_code=400)

    if len(data) > MAX_VIDEO_SIZE_BYTES:
        raise VideoValidationError(
            f"Video size {len(data) / (1024 * 1024):.1f} MB exceeds the 50 MB limit.",
            status_code=413,
        )

    ext = Path(filename).suffix.lower()
    if ext not in {".mp4", ".webm"}:
        raise VideoValidationError(
            f"Unsupported file extension '{ext}'. Only .mp4 and .webm are accepted.",
            status_code=415,
        )

    if _is_mp4(data):
        return "video/mp4"
    if _is_webm(data):
        return "video/webm"

    raise VideoValidationError(
        "File content does not match a valid MP4 or WebM container.",
        status_code=415,
    )


class VideoValidationError(ValueError):
    """Raised when an uploaded video fails validation."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# -------------------------------------------------------------------------
# Module-level singleton
# -------------------------------------------------------------------------
_job_store: VideoJobStore | None = None
_store_lock = threading.Lock()


def get_job_store() -> VideoJobStore:
    global _job_store
    if _job_store is None:
        with _store_lock:
            if _job_store is None:
                _job_store = VideoJobStore()
    return _job_store
