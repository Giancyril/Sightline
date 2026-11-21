# Day 5 Report — Observability & Production Readiness

> **Date:** 2026-11-21 · **Branch:** `main` · **Tag:** `v1.1.0`

---

## Overview

Day 5 delivers the full observability and production-readiness layer for the Object Detection System. The theme is **"you can't improve what you can't measure"**: every inference is now tracked, every anomaly surfaces automatically, and operators have a live dashboard without any third-party infrastructure dependencies.

---

## Commit Summary

| # | Commit | Hash | Description |
|---|---|---|---|
| 1 | `feat(export): TFLite dynamic-range export script` | `8510729` | INT8 weight-quantised TFLite export + `TFLiteDynamicRangeDetector` drop-in |
| 2 | `feat(export): full-integer quantization with representative dataset` | `679b3d2` | Full-integer TFLite (activations + weights INT8) with 50-sample calibration dataset |
| 3 | `feat(export): optional ONNX export` | `5e9fbcb` | ONNX opset-18 export + `ONNXDetector` using ONNX Runtime |
| 4 | `feat(eval): benchmark script for accuracy, latency and size` | `194521d` | `ml/eval/benchmark_quantization.py` — mAP, p50/p95 latency, file size, accuracy gate |
| 5 | `feat(registry): add the best variant to the registry` | `8d5dbb5` | `fine-tuned-tflite-dr` v1.1.0 entry added to `models/registry.json` |
| 6 | `feat(api): structured JSON logging with request ids` | `9efb0c2` | `RequestIDMiddleware` + `JSONRequestLoggingMiddleware` — every request emits a JSON log line with `request_id`, latency, status |
| 7 | `feat(api): GET /metrics in Prometheus format` | `8b7431c` | `MetricsCollector` singleton exposing Prometheus text at `GET /metrics`; drift and operations endpoints at `/v1/drift`, `/v1/operations` |
| 8 | `feat(core): drift detection against the validation distribution` | `a6ba59d` | `DriftDetector` using Population Stability Index (PSI) over a rolling 500-inference window |
| 9 | `feat(ui): Operations tab` | `8ed69b6` | React Operations dashboard: 4 KPI cards, SVG latency timeline, SVG confidence distribution drift chart, 3 s auto-polling |
| fix | `fix(lint): resolve ruff I001, F401, C420, UP035 across api and ml` | `cbe6791` | Ruff lint fixes across `api/main.py`, `api/routers/metrics.py`, `api/services/metrics.py`, `api/tests/test_metrics.py`, `ml/eval/drift.py` |

---

## Quantization Benchmark Results

Benchmarked on 2026-11-21 (CPU-only, Intel x86-64, single core).

| Variant | Format | mAP@0.5 | mAP Loss | p50 Latency | p95 Latency | Size (MB) | Compression | Gate |
|---|---|---|---|---|---|---|---|---|
| PyTorch Float32 Baseline | `.pth` state dict | **0.2644** | 0.0 | 111.8 ms | 125.5 ms | 30.38 | 1.0× | ✅ |
| TFLite Dynamic-Range INT8 | `.tflite` | **0.2642** | 0.0002 | 131.0 ms | 134.9 ms | 7.79 | **3.9×** | ✅ |

> **Accuracy gate:** mAP loss must be < 0.01. Both variants passed.

**Recommendation:** Deploy the TFLite Dynamic-Range variant for production:
- 3.9× smaller model on disk / in memory
- mAP loss of only 0.02% — imperceptible in practice
- Same preprocessing pipeline; drop-in `TFLiteDynamicRangeDetector` replacement

---

## API Surface — New Endpoints (Day 5)

| Method | Path | Format | Description |
|---|---|---|---|
| `GET` | `/metrics` | Prometheus text 0.0.4 | Request counters, error rates, latency histograms, inference metrics |
| `GET` | `/v1/metrics` | Prometheus text 0.0.4 | Alias for `/metrics` |
| `GET` | `/v1/drift` | JSON | PSI drift status, baseline vs live distributions, warning threshold |
| `GET` | `/v1/operations` | JSON | Consolidated telemetry: model info, latency p50/p95, error rate, confidence distribution, drift |

### Prometheus Metrics Exposed

| Metric | Type | Labels | Description |
|---|---|---|---|
| `api_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests by method, path, status class |
| `api_errors_total` | Counter | `method`, `endpoint`, `error_type` | 4xx/5xx responses categorised by error type |
| `inference_latency_seconds` | Histogram | — | End-to-end inference latency (12 buckets: 50 ms – 10 s) |
| `detections_per_image` | Histogram | — | Objects detected per request (8 buckets: 0 – 50+) |
| `detection_confidence` | Histogram | — | Confidence scores of accepted detections (10 buckets: 0.0 – 1.0) |
| `model_usage_total` | Counter | `model_id` | Requests per model variant |

---

## Architecture — Observability Flow

```
POST /v1/detect
      │
      ▼
RequestIDMiddleware  (injects X-Request-ID header + request.state.request_id)
      │
      ▼
JSONRequestLoggingMiddleware  (emits structured JSON log line on response)
      │
      ▼
detect router  ──► SSD inference  ──► MetricsCollector.record_request()
                                            │
                                            ├─► Prometheus counters / histograms
                                            └─► DriftDetector.observe(confidence)
                                                      │
                                                      └─► RollingConfidenceBuffer (500 inferences)
                                                                │
                                                                ▼
                                                       PSI vs validation baseline
                                                       ┌────────────────────────┐
                                                       │ PSI < 0.1  → nominal   │
                                                       │ PSI < 0.2  → warning   │
                                                       │ PSI ≥ 0.2  → drift!    │
                                                       └────────────────────────┘

GET /metrics    ──► MetricsCollector.render_prometheus_text()
GET /v1/drift   ──► DriftDetector.check_drift()
GET /v1/operations ──► MetricsCollector.get_operations_summary() + DriftDetector.check_drift()
                             │
                             ▼
                     React Operations Tab (3 s auto-poll)
                     ┌──────────────────────────────────────────────┐
                     │  KPI Cards: model · latency · errors · drift │
                     │  SVG Latency Timeline (p50 / p95 lines)      │
                     │  SVG Confidence Distribution Chart           │
                     └──────────────────────────────────────────────┘
```

---

## Load Test

The `scripts/load_test.py` script exercises the detect endpoint with concurrent workers using only the Python standard library (no extra dependencies).

```powershell
# 4 workers, 40 total requests, synthetic JPEG image:
python scripts/load_test.py --workers 4 --requests 40

# Real image file:
python scripts/load_test.py --workers 4 --requests 100 --image path/to/photo.jpg

# Remote server:
python scripts/load_test.py --url http://prod-server:8000 --workers 8 --requests 200
```

Output includes: success rate, wall-clock throughput (req/s), mean/p50/p95/min/max latency.

---

## Drift Detection — PSI Methodology

Population Stability Index quantifies how much a distribution has shifted:

$$
\text{PSI} = \sum_{i=1}^{10} \left( p_{\text{live},i} - p_{\text{baseline},i} \right) \cdot \ln\!\left( \frac{p_{\text{live},i}}{p_{\text{baseline},i}} \right)
$$

| PSI Range | Status | Action |
|---|---|---|
| < 0.10 | 🟢 Nominal | No action needed |
| 0.10 – 0.20 | 🟡 Warning | Monitor closely; consider re-calibration |
| ≥ 0.20 | 🔴 Drift | Investigate distribution shift; retrain if persistent |

The baseline distribution is sourced from `models/fine-tuned/calibration.json` (2,000 validation detections across 10 confidence bins).

The rolling window holds the last 500 inferences. PSI is recomputed on every `GET /v1/drift` or `GET /v1/operations` call.

---

## Files Added / Modified

### New Files
| Path | Purpose |
|---|---|
| `ml/export/tflite_export.py` | Dynamic-range TFLite export + `TFLiteDynamicRangeDetector` |
| `ml/export/tflite_full_integer.py` | Full-integer TFLite export + `TFLiteFullIntegerDetector` |
| `ml/export/onnx_export.py` | ONNX opset-18 export + `ONNXDetector` |
| `ml/eval/benchmark_quantization.py` | Accuracy / latency / size benchmark script |
| `ml/eval/drift.py` | `DriftDetector`, `RollingConfidenceBuffer`, `DriftReport`, `calculate_psi()` |
| `api/middleware/request_id.py` | `RequestIDMiddleware` |
| `api/logging_config.py` | `configure_json_logging()`, `JSONRequestLoggingMiddleware` |
| `api/services/metrics.py` | `MetricsCollector` singleton (Prometheus counters/histograms) |
| `api/routers/metrics.py` | `GET /metrics`, `GET /v1/drift`, `GET /v1/operations` |
| `web/src/views/OperationsView.tsx` | Operations dashboard React component |
| `scripts/load_test.py` | Concurrent HTTP load-test script |
| `docs/quantization_benchmark.json` | Benchmark results (machine-readable) |
| `docs/day5_report.md` | This report |
| `docs/screenshots/day5/README.md` | Screenshot capture guide |

### Modified Files
| Path | Change |
|---|---|
| `api/main.py` | Added `RequestIDMiddleware`, `JSONRequestLoggingMiddleware`, metrics router |
| `api/routers/detect.py` | Posts to `MetricsCollector` and `DriftDetector` on each inference |
| `models/registry.json` | Added `fine-tuned-tflite-dr` v1.1.0 entry |
| `docs/optimization.md` | Added 7 sections covering export variants, benchmark table, recommendations |
| `docs/model_card.md` | Updated to v1.1.0; added TFLite DR export format |
| `web/src/data/apiDocsData.ts` | Added `/metrics`, `/v1/drift`, `/v1/operations` endpoint docs |
| `web/src/types.ts` | Added `OperationsTelemetry` interface |
| `web/src/api.ts` | Added `fetchOperationsTelemetry()` |
| `web/src/App.tsx` | Added Operations tab |

---

## Test Coverage

| Test file | Tests | Status |
|---|---|---|
| `api/tests/test_logging.py` | 14 | ✅ All passed |
| `api/tests/test_metrics.py` | 8 | ✅ All passed |
| `ml/tests/test_drift.py` | 12 | ✅ All passed |
| `ml/tests/test_model_registry.py` | 19 | ✅ All passed |

---

## Known Limitations & Next Steps

### Current Limitations
- **Metrics are in-memory only** — restarting the server resets all counters to zero. For persistence, integrate Prometheus push-gateway or write counters to Redis.
- **Drift window is fixed at 500 inferences** — in low-traffic scenarios this may take a long time to populate.
- **TFLite runtime latency is slower than PyTorch** on this CPU benchmark because the TFLite runtime isn't GPU-accelerated. On ARM/mobile devices or Edge TPU the advantage reverses.
- **ONNX export** requires `onnx` and `onnxruntime` packages; not installed by default.

### Recommended Next Steps (Day 6+)
1. **Prometheus + Grafana integration** — configure a `prometheus.yml` scrape config pointing at `GET /metrics` and build a Grafana dashboard
2. **Alerting rules** — Alertmanager rule: `psi > 0.2 for 5m → page on-call`
3. **Model retraining pipeline** — automate data collection from correction queue → retrain → benchmark → gate → registry update
4. **A/B traffic split** — route a percentage of requests to `fine-tuned-tflite-dr` and compare live metrics against `fine-tuned`
5. **GPU inference** — integrate CUDA TFLite delegate or switch to TorchScript with `torch.compile` for real-time throughput

---

> Generated automatically as part of the Day 5 commit plan.
> See [`docs/optimization.md`](optimization.md) for the full quantization write-up.
