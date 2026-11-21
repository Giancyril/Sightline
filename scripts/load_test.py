"""
scripts/load_test.py
--------------------
HTTP load-test for the Object Detection API.

Hits POST /v1/detect concurrently with a synthetic or real image and reports
aggregate throughput statistics (total requests, success rate, p50/p95 latency).

Usage
-----
# Synthetic image (no external file needed):
python scripts/load_test.py --workers 4 --requests 40

# Real image file:
python scripts/load_test.py --workers 4 --requests 40 --image path/to/image.jpg

# Custom base URL:
python scripts/load_test.py --url http://localhost:8000 --workers 8 --requests 100
"""

from __future__ import annotations

import argparse
import io
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------


def _make_synthetic_jpeg(width: int = 320, height: int = 240) -> bytes:
    """Return a minimal valid JPEG encoded as raw bytes (no PIL required)."""
    # Try PIL/Pillow first for a proper colour image
    try:
        from PIL import Image  # type: ignore[import]

        img = Image.new("RGB", (width, height), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: construct a minimal 1×1 white JPEG using raw bytes
    # (a valid JPEG baseline that all servers accept)
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"C  C\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b"
        b"\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b"
        b"\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
        b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05"
        b"\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06"
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br'
        b"\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
        b"cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94"
        b"\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa"
        b"\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7"
        b"\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3"
        b"\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8"
        b"\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0f\xff\xd9"
    )


# ---------------------------------------------------------------------------
# Single-request worker
# ---------------------------------------------------------------------------


def _send_one(url: str, image_bytes: bytes, timeout: float) -> dict[str, object]:
    """POST one detect request; return timing and status."""
    boundary = b"----LoadTestBoundary7x9z"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
        + image_bytes
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary.decode()}",
        "Content-Length": str(len(body)),
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception:  # noqa: BLE001
        status = 0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {"status": status, "latency_ms": elapsed_ms}


# ---------------------------------------------------------------------------
# Main load-test runner
# ---------------------------------------------------------------------------


def run_load_test(
    *,
    base_url: str,
    workers: int,
    total_requests: int,
    image_bytes: bytes,
    timeout: float,
) -> None:
    detect_url = base_url.rstrip("/") + "/v1/detect"

    print(f"\n{'=' * 60}")
    print("  Object Detection API — Load Test")
    print(f"{'=' * 60}")
    print(f"  Target  : {detect_url}")
    print(f"  Workers : {workers}")
    print(f"  Requests: {total_requests}")
    print(f"  Payload : {len(image_bytes):,} bytes (JPEG)")
    print(f"  Timeout : {timeout:.1f}s per request")
    print(f"{'=' * 60}\n")

    results: list[dict[str, object]] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_send_one, detect_url, image_bytes, timeout)
            for _ in range(total_requests)
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            status = r["status"]
            sym = "✓" if isinstance(status, int) and 200 <= status < 300 else "✗"
            print(
                f"  [{i:>4}/{total_requests}] {sym}  {r['latency_ms']:.1f} ms  HTTP {status}"
            )

    wall_elapsed = time.perf_counter() - wall_start

    # ---- Aggregate stats ------------------------------------------------
    latencies = [r["latency_ms"] for r in results]  # type: ignore[misc]
    successes = sum(
        1 for r in results if isinstance(r["status"], int) and 200 <= r["status"] < 300
    )
    errors = total_requests - successes

    sorted_lat = sorted(latencies)
    p50 = statistics.median(sorted_lat)
    p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
    p95 = sorted_lat[p95_idx]
    mean_lat = statistics.mean(sorted_lat)
    throughput = total_requests / wall_elapsed

    print(f"\n{'=' * 60}")
    print("  Results")
    print(f"{'=' * 60}")
    print(f"  Total requests  : {total_requests}")
    print(
        f"  Successful       : {successes}  ({100 * successes / total_requests:.1f} %)"
    )
    print(f"  Errors           : {errors}")
    print(f"  Wall time        : {wall_elapsed:.2f} s")
    print(f"  Throughput       : {throughput:.2f} req/s")
    print(f"  Latency  mean    : {mean_lat:.1f} ms")
    print(f"  Latency  p50     : {p50:.1f} ms")
    print(f"  Latency  p95     : {p95:.1f} ms")
    print(f"  Latency  min     : {sorted_lat[0]:.1f} ms")
    print(f"  Latency  max     : {sorted_lat[-1]:.1f} ms")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTP load-test for the Object Detection API (POST /v1/detect).",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=40,
        help="Total number of requests to send (default: 40).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a JPEG/PNG image file to use. Omit to use a synthetic image.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.image:
        image_bytes = Path(args.image).read_bytes()
        print(f"Using image file: {args.image}  ({len(image_bytes):,} bytes)")
    else:
        image_bytes = _make_synthetic_jpeg()
        print("Using synthetic JPEG image (no --image provided).")

    run_load_test(
        base_url=args.url,
        workers=args.workers,
        total_requests=args.requests,
        image_bytes=image_bytes,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
