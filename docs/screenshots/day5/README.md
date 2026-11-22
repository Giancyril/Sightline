# Day 5 Screenshots — Observability & Production Readiness

This directory holds reference screenshots for the Day 5 deliverables.
Capture these after starting the API (`uvicorn api.main:app`) and the frontend (`pnpm run dev`).

---

## Screenshot Catalogue

### 1. `operations-kpi-cards.png`
**What to capture:** The Operations tab with all four KPI cards visible at the top.

Expected state:
- **Model** card: "SSD MobileNet V2 (Open Images 20-Class)" · version "v1.1.0"
- **Latency** card: p50 and p95 values in milliseconds (e.g. 120 ms / 145 ms)
- **Error Rate** card: percentage of failed requests (should be near 0 % on a fresh server)
- **Drift Status** card: "nominal" badge in green, PSI value

---

### 2. `operations-latency-timeline.png`
**What to capture:** The SVG latency timeline chart in the Operations tab after sending ~20 detect requests.

Expected state:
- Blue line: p50 rolling latency per request
- Orange dashed line: p95 rolling latency
- Y-axis in milliseconds, X-axis is recent request index
- Both lines visible and distinct

---

### 3. `operations-confidence-drift.png`
**What to capture:** The confidence distribution chart in the Operations tab after sending ~50 detect requests.

Expected state:
- Blue bars: baseline (validation) distribution across 10 bins [0.0, 1.0]
- Orange bars: live distribution from recent inferences
- Bar groups side by side, x-axis labeled with bin edges
- PSI value and drift status badge visible below or above chart

---

### 4. `metrics-prometheus.png`
**What to capture:** Browser or terminal output from `GET /metrics`.

Steps:
```
curl http://localhost:8000/metrics
```

Expected content:
```
# HELP api_requests_total Total HTTP requests
# TYPE api_requests_total counter
api_requests_total{method="POST",endpoint="/v1/detect",status="2xx"} 42.0
...
# HELP inference_latency_seconds Inference latency histogram
# TYPE inference_latency_seconds histogram
inference_latency_seconds_bucket{le="0.05"} 0.0
inference_latency_seconds_bucket{le="0.1"} 3.0
...
```

---

### 5. `load-test-output.png`
**What to capture:** Terminal output of running the load-test script.

Command:
```powershell
python scripts/load_test.py --workers 4 --requests 40
```

Expected output (approximate):
```
============================================================
  Object Detection API — Load Test
============================================================
  Target  : http://localhost:8000/v1/detect
  Workers : 4
  Requests: 40
  ...
============================================================
  Results
============================================================
  Total requests  : 40
  Successful       : 40  (100.0 %)
  Errors           : 0
  Wall time        : 14.32 s
  Throughput       : 2.79 req/s
  Latency  mean    : 118.4 ms
  Latency  p50     : 115.2 ms
  Latency  p95     : 142.7 ms
  Latency  min     :  98.1 ms
  Latency  max     : 189.3 ms
============================================================
```

---

## Capture Instructions

1. Start the API: `uvicorn api.main:app --reload --port 8000`
2. Start the frontend: `cd web && pnpm run dev`
3. Open `http://localhost:5173` and navigate to the **Operations** tab
4. Send ~50 requests using the detect form or the load-test script
5. Refresh the Operations tab (it auto-polls every 3 s)
6. Take screenshots and save them in this directory with the exact filenames listed above
