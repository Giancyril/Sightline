import { API_DOCS_DISPLAY_BASE_URL } from "../api";

export interface ParameterSpec {
  name: string;
  in: "query" | "formData" | "header" | "path";
  type: string;
  required: boolean;
  default?: string;
  description: string;
}

export interface SchemaFieldSpec {
  field: string;
  type: string;
  description: string;
}

export interface ErrorCodeSpec {
  status: number;
  name: string;
  description: string;
}

export interface EndpointSpec {
  id: string;
  method: "POST" | "GET";
  path: string;
  summary: string;
  description: string;
  parameters: ParameterSpec[];
  responseFields: SchemaFieldSpec[];
  errorCodes: ErrorCodeSpec[];
  snippets: {
    curl: string;
    js: string;
    python: string;
  };
  exampleResponse: string;
}

export const ENDPOINTS_DATA: EndpointSpec[] = [
  {
    id: "detect",
    method: "POST",
    path: "/v1/detect",
    summary: "Detect objects in image or frame",
    description:
      "Performs real-time object detection across 20 target categories using the fine-tuned SSD MobileNet V2 model. Accepts multipart form data with image binary and optional threshold tuners.",
    parameters: [
      {
        name: "file",
        in: "formData",
        type: "binary",
        required: true,
        description: "Image payload. Supported formats: JPEG, PNG, WebP. Maximum size: 10 MB.",
      },
      {
        name: "model",
        in: "query",
        type: "string",
        required: false,
        default: "fine-tuned",
        description:
          "Target model identifier to invoke ('fine-tuned' or 'coco-baseline'). Defaults to 'fine-tuned'.",
      },
      {
        name: "confidence_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.25",
        description:
          "Confidence cutoff for reported detections (0.01 to 1.0). Detections below this score are filtered.",
      },
      {
        name: "nms_iou_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.45",
        description:
          "Non-Maximum Suppression (NMS) intersection-over-union threshold (0.01 to 1.0) for deduplicating overlapping boxes.",
      },
      {
        name: "threshold_mode",
        in: "query",
        type: "string",
        required: false,
        default: "global",
        description:
          "Threshold strategy: 'global' (default — uses confidence_threshold for all classes, existing behaviour unchanged), 'recommended' (applies per-class thresholds from thresholds.json with calibrated scores), or 'custom' (caller supplies per-class values via custom_thresholds).",
      },
      {
        name: "custom_thresholds",
        in: "query",
        type: "string (JSON)",
        required: false,
        description:
          "JSON object mapping class names to thresholds, e.g. {\"Laptop\":0.35,\"Person\":0.28}. Only used when threshold_mode=custom. Classes not listed fall back to confidence_threshold.",
      },
    ],
    responseFields: [
      { field: "model_name", type: "string", description: "Name of the served detector architecture" },
      { field: "model_version", type: "string", description: "Semantic version of the fine-tuned checkpoint" },
      { field: "total_detections", type: "integer", description: "Count of predicted bounding boxes exceeding confidence threshold" },
      { field: "inference_time_ms", type: "float", description: "Raw neural network forward pass time in milliseconds" },
      { field: "total_time_ms", type: "float", description: "End-to-end pipeline latency including decode and post-processing" },
      { field: "image_dimensions", type: "object", description: "Input width, height, and channel dimensions in pixels" },
      { field: "threshold_mode", type: "string", description: "Effective threshold mode applied: 'global', 'recommended', or 'custom'" },
      { field: "calibration_applied", type: "boolean", description: "True if temperature scaling was applied to scores before filtering" },
      { field: "detections[]", type: "array<object>", description: "List of detected objects with class labels, coordinates, and color tokens" },
    ],
    errorCodes: [
      { status: 400, name: "Bad Request", description: "Corrupt image payload or unreadable image stream" },
      { status: 413, name: "Payload Too Large", description: "Image file exceeds maximum limit of 10,485,760 bytes (10 MB)" },
      { status: 415, name: "Unsupported Media Type", description: "File format is not JPEG, PNG, or WebP" },
      { status: 422, name: "Validation Error", description: "Missing required file parameter or threshold query parameters out of range" },
    ],
    snippets: {
      curl: `curl -X POST "${API_DOCS_DISPLAY_BASE_URL}/v1/detect?confidence_threshold=0.25&nms_iou_threshold=0.45" \\
  -H "Accept: application/json" \\
  -F "file=@image.jpg"`,
      js: `const formData = new FormData();
formData.append("file", imageFile, "photo.jpg");

const url = new URL("${API_DOCS_DISPLAY_BASE_URL}/v1/detect");
url.searchParams.set("confidence_threshold", "0.25");
url.searchParams.set("nms_iou_threshold", "0.45");

const response = await fetch(url, {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(\`Detected \${result.total_detections} objects in \${result.inference_time_ms}ms\`);`,
      python: `import requests

url = "${API_DOCS_DISPLAY_BASE_URL}/v1/detect"
params = {
    "confidence_threshold": 0.25,
    "nms_iou_threshold": 0.45
}

with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post(url, params=params, files=files)

data = response.json()
print(f"Total: {data['total_detections']} (latency: {data['inference_time_ms']}ms)")
for det in data["detections"]:
    print(f"- {det['class_name']}: {det['confidence']*100:.1f}%")`,
    },
    exampleResponse: `{
  "model_name": "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
  "model_version": "v1.0.0",
  "total_detections": 2,
  "inference_time_ms": 14.28,
  "total_time_ms": 18.52,
  "image_dimensions": {
    "width": 1280,
    "height": 720,
    "channels": 3
  },
  "thresholds_applied": {
    "confidence_threshold": 0.25,
    "nms_iou_threshold": 0.45
  },
  "detections": [
    {
      "class_id": 2,
      "class_name": "Laptop",
      "confidence": 0.9412,
      "box_normalized": {
        "ymin": 0.342,
        "xmin": 0.215,
        "ymax": 0.781,
        "xmax": 0.658
      },
      "box_pixel": {
        "ymin": 246,
        "xmin": 275,
        "ymax": 562,
        "xmax": 842,
        "width": 567,
        "height": 316
      },
      "color_hex": "#E69F00",
      "color_rgb": [230, 159, 0]
    }
  ]
}`,
  },
  {
    id: "classes",
    method: "GET",
    path: "/v1/classes",
    summary: "Get 20-class target taxonomy",
    description:
      "Returns the full list of supported object classes along with their COCO overlap flags, Open Images MID identifiers, accessible color palettes, and semantic definitions.",
    parameters: [],
    responseFields: [
      { field: "total_classes", type: "integer", description: "Total count of active target classes (20)" },
      { field: "classes[]", type: "array<object>", description: "Array of class taxonomy objects" },
      { field: "classes[].id", type: "integer", description: "Deterministic class ID (1 to 20)" },
      { field: "classes[].name", type: "string", description: "Human-readable concept name (e.g., Person, Laptop)" },
      { field: "classes[].mid", type: "string", description: "Open Images ontology Machine Identifier (/m/...)" },
      { field: "classes[].is_coco", type: "boolean", description: "True if concept overlaps with COCO 80-class pretraining" },
      { field: "classes[].color_hex", type: "string", description: "WCAG-compliant hex color code assigned to class" },
      { field: "classes[].description", type: "string", description: "Official taxonomy scope and semantic boundary definition" },
    ],
    errorCodes: [
      { status: 500, name: "Internal Server Error", description: "Failed to read taxonomy registry file" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/v1/classes" \\
  -H "Accept: application/json"`,
      js: `const response = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/classes");
const data = await response.json();
console.log(\`Loaded \${data.total_classes} classes\`);`,
      python: `import requests

response = requests.get("${API_DOCS_DISPLAY_BASE_URL}/v1/classes")
data = response.json()
for c in data["classes"]:
    tag = "COCO" if c["is_coco"] else "Novel"
    print(f"#{c['id']} {c['name']} [{tag}]")`,
    },
    exampleResponse: `{
  "total_classes": 20,
  "classes": [
    {
      "id": 1,
      "name": "Person",
      "mid": "/m/01g317",
      "is_coco": true,
      "color_hex": "#0072B2",
      "color_rgb": [0, 114, 178],
      "description": "Individual human instance (adult, child, infant)."
    },
    {
      "id": 11,
      "name": "Guitar",
      "mid": "/m/0342h",
      "is_coco": false,
      "color_hex": "#332288",
      "color_rgb": [51, 34, 136],
      "description": "Acoustic or electric stringed musical instrument."
    }
  ]
}`,
  },
  {
    id: "model",
    method: "GET",
    path: "/v1/model",
    summary: "Get model metadata & benchmarks",
    description:
      "Retrieves technical architecture details, input resolution constraints (300×300), anchor box counts, and validation evaluation metrics (mAP@0.5 and COCO mAP@[.5:.95]).",
    parameters: [],
    responseFields: [
      { field: "name", type: "string", description: "Full descriptive model name" },
      { field: "version", type: "string", description: "Model checkpoint version (v1.0.0)" },
      { field: "architecture", type: "string", description: "Backbone and detector type (SSDMobileNetV2)" },
      { field: "input_resolution", type: "array<int>", description: "Required network input dimensions [width, height]" },
      { field: "num_classes", type: "integer", description: "Output logit dimension including background class (21)" },
      { field: "optimal_confidence_threshold", type: "float", description: "F1-tuned threshold operating point (0.25)" },
      { field: "metrics", type: "object", description: "Quantitative validation benchmarks (mAP_all, mAP_novel_classes)" },
    ],
    errorCodes: [
      { status: 503, name: "Service Unavailable", description: "Model weights not yet loaded into memory" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/v1/model" \\
  -H "Accept: application/json"`,
      js: `const response = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/model");
const metadata = await response.json();
console.log(\`Model: \${metadata.name} (input: \${metadata.input_resolution.join("x")})\`);`,
      python: `import requests

response = requests.get("${API_DOCS_DISPLAY_BASE_URL}/v1/model")
meta = response.json()
print(f"Architecture: {meta['architecture']}")
print(f"Novel mAP@0.5: {meta['metrics']['mAP_novel_classes']:.4f}")`,
    },
    exampleResponse: `{
  "name": "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
  "version": "v1.0.0",
  "architecture": "SSDMobileNetV2",
  "input_resolution": [300, 300],
  "num_classes": 21,
  "optimal_confidence_threshold": 0.25,
  "supported_formats": ["JPEG", "PNG", "WebP"],
  "max_file_size_bytes": 10485760,
  "metrics": {
    "mAP_all": 0.2644,
    "mAP_coco_shared": 0.3471,
    "mAP_novel_classes": 0.1816,
    "coco_mAP_50_95": 0.1603
  }
}`,
  },
  {
    id: "health",
    method: "GET",
    path: "/healthz",
    summary: "Service liveness & readiness probe",
    description:
      "Health check endpoint designed for orchestration systems (Docker, Kubernetes) and monitoring probes. Returns model load status and runtime service version.",
    parameters: [],
    responseFields: [
      { field: "status", type: "string", description: "Service health state ('ok')" },
      { field: "model_loaded", type: "boolean", description: "True if detector PyTorch weights are initialized in memory" },
      { field: "version", type: "string", description: "Service API version" },
    ],
    errorCodes: [
      { status: 503, name: "Service Unavailable", description: "Detector failed to initialize or model weights missing" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/healthz" \\
  -H "Accept: application/json"`,
      js: `const res = await fetch("${API_DOCS_DISPLAY_BASE_URL}/healthz");
const health = await res.json();
console.log(\`Service status: \${health.status}, Model loaded: \${health.model_loaded}\`);`,
      python: `import requests

res = requests.get("${API_DOCS_DISPLAY_BASE_URL}/healthz")
assert res.status_code == 200
print("Service healthy:", res.json())`,
    },
    exampleResponse: `{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}`,
  },
  {
    id: "models",
    method: "GET",
    path: "/v1/models",
    summary: "List registered models and memory cache status",
    description:
      "Returns the complete registry of available object detection models with metadata, input resolutions, evaluation metrics, and active memory residency status.",
    parameters: [],
    responseFields: [
      { field: "models[]", type: "array<object>", description: "Array of registered model metadata entries" },
      { field: "models[].id", type: "string", description: "Unique model key ('fine-tuned', 'coco-baseline')" },
      { field: "models[].display_name", type: "string", description: "Human-readable descriptive model name" },
      { field: "models[].version", type: "string", description: "Semantic version string" },
      { field: "models[].input_size", type: "array<int>", description: "Required network input size [width, height]" },
      { field: "models[].is_loaded", type: "boolean", description: "True if model weights are cached in memory" },
      { field: "models[].metrics", type: "object", description: "Validation performance metrics" },
    ],
    errorCodes: [
      { status: 500, name: "Internal Server Error", description: "Failed to read model registry file" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/v1/models" \\
  -H "Accept: application/json"`,
      js: `const res = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/models");
const data = await res.json();
console.log(data.models);`,
      python: `import requests

res = requests.get("${API_DOCS_DISPLAY_BASE_URL}/v1/models")
models = res.json()["models"]
for m in models:
    print(f"{m['id']} ({m['version']}) - loaded: {m['is_loaded']}")`,
    },
    exampleResponse: `{
  "models": [
    {
      "id": "fine-tuned",
      "display_name": "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
      "version": "v1.0.0",
      "path": "ml/export/saved_model",
      "input_size": [300, 300],
      "class_list": "classes.json",
      "is_loaded": true,
      "metrics": {
        "mAP_all": 0.2644,
        "mAP_novel_classes": 0.1816,
        "mAP_coco_shared": 0.3471
      }
    },
    {
      "id": "coco-baseline",
      "display_name": "SSD MobileNet V2 (COCO Baseline)",
      "version": "v1.0.0",
      "path": "ml/export/saved_model",
      "input_size": [300, 300],
      "class_list": "classes.json",
      "is_loaded": false,
      "metrics": {
        "mAP_all": 0.163,
        "mAP_novel_classes": 0.0,
        "mAP_coco_shared": 0.326
      }
    }
  ]
}`,
  },
  {
    id: "model-thresholds",
    method: "GET",
    path: "/v1/models/{model_id}/thresholds",
    summary: "Get per-class thresholds and calibration metrics",
    description:
      "Returns the per-class confidence threshold table for a registered model, including precision, recall, and F1 at the recommended threshold versus the global 0.25 baseline. Requires a thresholds.json file next to the model entry. Returns 404 for models without threshold files.",
    parameters: [
      {
        name: "model_id",
        in: "path",
        type: "string",
        required: true,
        description: "Model identifier (e.g. 'fine-tuned'). Must match a registered model entry.",
      },
    ],
    responseFields: [
      { field: "model_id", type: "string", description: "Model identifier" },
      { field: "rule", type: "string", description: "Threshold selection rule used (e.g. 'best_f1')" },
      { field: "date", type: "string (ISO 8601)", description: "Timestamp of threshold calculation run" },
      { field: "data_hash", type: "string", description: "SHA-256 hash of the validation manifest used" },
      { field: "global_threshold", type: "float", description: "Fallback global threshold (0.25)" },
      { field: "fallbacks", type: "array<string>", description: "Class names that fell back to global threshold" },
      { field: "thresholds", type: "object", description: "Fast-lookup map: class_name → recommended threshold" },
      { field: "per_class[]", type: "array<object>", description: "Detailed per-class rows with P/R/F1 at recommended and global thresholds" },
      { field: "per_class[].threshold", type: "float", description: "Recommended threshold for this class" },
      { field: "per_class[].precision", type: "float", description: "Precision at recommended threshold" },
      { field: "per_class[].recall", type: "float", description: "Recall at recommended threshold" },
      { field: "per_class[].f1", type: "float", description: "F1 score at recommended threshold" },
      { field: "per_class[].precision_raw", type: "float", description: "Precision at global 0.25 threshold" },
      { field: "per_class[].recall_raw", type: "float", description: "Recall at global 0.25 threshold" },
      { field: "per_class[].fell_back", type: "boolean", description: "True if class fell back to global threshold" },
    ],
    errorCodes: [
      { status: 404, name: "Not Found", description: "No thresholds.json found for this model (coco-baseline, or model without calibration)" },
      { status: 422, name: "Validation Error", description: "Threshold file exists but fails schema or value-range validation" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/v1/models/fine-tuned/thresholds" \\
  -H "Accept: application/json"`,
      js: `const res = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/models/fine-tuned/thresholds");
const data = await res.json();
// Get the recommended threshold for a specific class
const laptopThresh = data.thresholds["Laptop"]; // e.g. 0.28
console.log(\`Laptop threshold: \${laptopThresh}\`);`,
      python: `import requests

res = requests.get("${API_DOCS_DISPLAY_BASE_URL}/v1/models/fine-tuned/thresholds")
data = res.json()
print(f"Rule: {data['rule']}, Global fallback: {data['global_threshold']}")
for cls in data["per_class"]:
    delta_p = (cls["precision"] - cls["precision_raw"]) * 100
    delta_r = (cls["recall"] - cls["recall_raw"]) * 100
    print(f"{cls['class_name']}: {cls['threshold']:.2f} | ΔP={delta_p:+.1f}pp ΔR={delta_r:+.1f}pp")`,
    },
    exampleResponse: `{
  "model_id": "fine-tuned",
  "rule": "best_f1",
  "date": "2026-11-19T00:00:00Z",
  "global_threshold": 0.25,
  "fallbacks": ["Helmet", "Wheelchair", "Flashlight", "Stapler", "Wrench"],
  "thresholds": {
    "Person": 0.26,
    "Laptop": 0.28,
    "Guitar": 0.35
  },
  "per_class": [
    {
      "class_id": 1,
      "class_name": "Person",
      "threshold": 0.26,
      "precision": 0.72,
      "recall": 0.68,
      "f1": 0.6994,
      "precision_raw": 0.65,
      "recall_raw": 0.71,
      "fell_back": false
    },
    {
      "class_id": 13,
      "class_name": "Helmet",
      "threshold": 0.25,
      "precision": 0.60,
      "recall": 0.60,
      "f1": 0.60,
      "precision_raw": 0.60,
      "recall_raw": 0.60,
      "fell_back": true,
      "fallback_reason": "too_few_detections"
    }
  ]
}`,
  },
  {
    id: "compare",
    method: "POST",
    path: "/v1/compare",
    summary: "Parallel dual-model detection & benchmark",
    description:
      "Executes parallel inference of two models on a single image concurrently using worker threads. Returns individual model detections, latency timings, and speedup benchmarks.",
    parameters: [
      {
        name: "file",
        in: "formData",
        type: "binary",
        required: true,
        description: "Image payload. Supported formats: JPEG, PNG, WebP. Maximum size: 10 MB.",
      },
      {
        name: "model_a",
        in: "query",
        type: "string",
        required: false,
        default: "fine-tuned",
        description: "First model identifier to evaluate.",
      },
      {
        name: "model_b",
        in: "query",
        type: "string",
        required: false,
        default: "coco-baseline",
        description: "Second model identifier to evaluate.",
      },
      {
        name: "confidence_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.25",
        description: "Confidence cutoff threshold (0.01 to 1.0).",
      },
      {
        name: "nms_iou_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.45",
        description: "Non-Maximum Suppression IoU threshold (0.01 to 1.0).",
      },
    ],
    responseFields: [
      { field: "model_a_result", type: "object", description: "Detection response object from Model A" },
      { field: "model_b_result", type: "object", description: "Detection response object from Model B" },
      { field: "total_time_ms", type: "float", description: "Parallel dual-model execution wall clock time in ms" },
    ],
    errorCodes: [
      { status: 400, name: "Bad Request", description: "Unrecognized model ID specified or corrupt image payload" },
      { status: 413, name: "Payload Too Large", description: "Image file exceeds maximum limit of 10 MB" },
      { status: 415, name: "Unsupported Media Type", description: "File format is not JPEG, PNG, or WebP" },
      { status: 422, name: "Validation Error", description: "Missing required file parameter or invalid parameter value" },
    ],
    snippets: {
      curl: `curl -X POST "${API_DOCS_DISPLAY_BASE_URL}/v1/compare?model_a=fine-tuned&model_b=coco-baseline" \\
  -H "Accept: application/json" \\
  -F "file=@image.jpg"`,
      js: `const formData = new FormData();
formData.append("file", imageFile, "photo.jpg");

const url = new URL("${API_DOCS_DISPLAY_BASE_URL}/v1/compare");
url.searchParams.set("model_a", "fine-tuned");
url.searchParams.set("model_b", "coco-baseline");

const response = await fetch(url, {
  method: "POST",
  body: formData,
});

const comparison = await response.json();
console.log(
  \`Model A (\${comparison.model_a_result.model_id}): \${comparison.model_a_result.total_detections} objs, \` +
  \`Model B (\${comparison.model_b_result.model_id}): \${comparison.model_b_result.total_detections} objs\`
);`,
      python: `import requests

url = "${API_DOCS_DISPLAY_BASE_URL}/v1/compare"
params = {
    "model_a": "fine-tuned",
    "model_b": "coco-baseline",
    "confidence_threshold": 0.25,
}

with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    res = requests.post(url, params=params, files=files)

data = res.json()
print(f"Parallel time: {data['total_time_ms']:.1f}ms")
print(f"Model A detections: {data['model_a_result']['total_detections']}")
print(f"Model B detections: {data['model_b_result']['total_detections']}")`,
    },
    exampleResponse: `{
  "model_a_result": {
    "model_name": "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
    "model_id": "fine-tuned",
    "model_version": "v1.0.0",
    "total_detections": 3,
    "inference_time_ms": 14.5,
    "detections": []
  },
  "model_b_result": {
    "model_name": "SSD MobileNet V2 (COCO Baseline)",
    "model_id": "coco-baseline",
    "model_version": "v1.0.0",
    "total_detections": 1,
    "inference_time_ms": 13.9,
    "detections": []
  },
  "total_time_ms": 16.2
}`,
  },
  {
    id: "video-upload",
    method: "POST",
    path: "/v1/video",
    summary: "Upload video and queue tracking job",
    description:
      "Uploads an MP4 or WebM video clip (up to 30s, 50MB) and dispatches a background detection, multi-object tracking, and directional counting worker. Returns a job_id for status polling.",
    parameters: [
      {
        name: "file",
        in: "formData",
        type: "binary",
        required: true,
        description: "Video payload (MP4 or WebM). Maximum size: 50 MB, maximum length: 30 seconds.",
      },
      {
        name: "model",
        in: "query",
        type: "string",
        required: false,
        default: "fine-tuned",
        description: "Detection model ID from the registry to use.",
      },
      {
        name: "frame_stride",
        in: "query",
        type: "integer",
        required: false,
        default: "3",
        description: "Process every N-th frame (1 to 30). Higher stride reduces processing latency.",
      },
      {
        name: "confidence_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.25",
        description: "Confidence cutoff filter (0.01 to 1.0).",
      },
      {
        name: "nms_iou_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.45",
        description: "Non-Maximum Suppression IoU threshold (0.01 to 1.0).",
      },
      {
        name: "line_p1_x",
        in: "query",
        type: "float",
        required: false,
        description: "Normalized X coordinate (0.0 to 1.0) of counting line endpoint 1.",
      },
      {
        name: "line_p1_y",
        in: "query",
        type: "float",
        required: false,
        description: "Normalized Y coordinate (0.0 to 1.0) of counting line endpoint 1.",
      },
      {
        name: "line_p2_x",
        in: "query",
        type: "float",
        required: false,
        description: "Normalized X coordinate (0.0 to 1.0) of counting line endpoint 2.",
      },
      {
        name: "line_p2_y",
        in: "query",
        type: "float",
        required: false,
        description: "Normalized Y coordinate (0.0 to 1.0) of counting line endpoint 2.",
      },
    ],
    responseFields: [
      { field: "job_id", type: "string", description: "Unique UUID tracking the processing job" },
      { field: "state", type: "string", description: "Initial job state (queued or running)" },
      { field: "fps", type: "float", description: "Source video frame rate" },
      { field: "duration_s", type: "float", description: "Duration of video in seconds" },
      { field: "frame_count", type: "integer", description: "Total video frames detected" },
      { field: "total_frames_to_process", type: "integer", description: "Number of frames to process given stride" },
    ],
    errorCodes: [
      { status: 413, name: "Payload Too Large", description: "Video exceeds maximum limit of 50 MB" },
      { status: 415, name: "Unsupported Media Type", description: "File is not a valid MP4 or WebM video" },
      { status: 422, name: "Unprocessable Entity", description: "Video exceeds 30s limit or cannot be decoded" },
    ],
    snippets: {
      curl: `curl -X POST "${API_DOCS_DISPLAY_BASE_URL}/v1/video?model=fine-tuned&frame_stride=3" \\
  -H "Accept: application/json" \\
  -F "file=@traffic.mp4"`,
      js: `const formData = new FormData();
formData.append("file", videoFile, "clip.mp4");

const url = new URL("${API_DOCS_DISPLAY_BASE_URL}/v1/video");
url.searchParams.set("model", "fine-tuned");
url.searchParams.set("frame_stride", "3");

const res = await fetch(url, { method: "POST", body: formData });
const { job_id } = await res.json();
console.log("Queued video job:", job_id);`,
      python: `import requests

url = "${API_DOCS_DISPLAY_BASE_URL}/v1/video"
with open("traffic.mp4", "rb") as f:
    res = requests.post(
        url,
        params={"model": "fine-tuned", "frame_stride": 3},
        files={"file": ("traffic.mp4", f, "video/mp4")},
    )
data = res.json()
print("Job ID:", data["job_id"])`,
    },
    exampleResponse: `{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "state": "running",
  "fps": 30.0,
  "duration_s": 12.4,
  "frame_count": 372,
  "total_frames_to_process": 124,
  "width": 1280,
  "height": 720,
  "mime_type": "video/mp4",
  "file_size_bytes": 14205810
}`,
  },
  {
    id: "video-status",
    method: "GET",
    path: "/v1/video/{job_id}",
    summary: "Check video job status and download links",
    description:
      "Polls progress for a queued or running video job, returning frames processed, percentage completion, and artifact download URLs once completed.",
    parameters: [
      {
        name: "job_id",
        in: "path",
        type: "string",
        required: true,
        description: "UUID of the video processing job.",
      },
    ],
    responseFields: [
      { field: "job_id", type: "string", description: "Job identifier" },
      { field: "state", type: "string", description: "Current status: queued | running | done | failed" },
      { field: "progress", type: "float", description: "Completion fraction from 0.0 to 1.0" },
      { field: "frames_processed", type: "integer", description: "Frames completed so far" },
      { field: "total_frames_to_process", type: "integer", description: "Total target frame count" },
      { field: "results", type: "object", description: "Download URLs for annotated video, tracks CSV, and summary JSON" },
    ],
    errorCodes: [
      { status: 404, name: "Not Found", description: "Job ID not found or expired after 1-hour retention" },
    ],
    snippets: {
      curl: `curl "${API_DOCS_DISPLAY_BASE_URL}/v1/video/a1b2c3d4-e5f6-7890-abcd-ef1234567890"`,
      js: `const res = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/video/" + jobId);
const status = await res.json();
console.log(\`Progress: \${Math.round(status.progress * 100)}% (\${status.state})\`);`,
      python: `import requests

res = requests.get(f"${API_DOCS_DISPLAY_BASE_URL}/v1/video/{job_id}")
status = res.json()
print(f"State: {status['state']}, Progress: {status['progress'] * 100:.1f}%")`,
    },
    exampleResponse: `{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "state": "done",
  "progress": 1.0,
  "frames_processed": 124,
  "total_frames_to_process": 124,
  "fps": 30.0,
  "duration_s": 12.4,
  "width": 1280,
  "height": 720,
  "has_results": true,
  "results": {
    "annotated_video": "/v1/video/a1b2c3d4-e5f6-7890-abcd-ef1234567890/annotated",
    "tracks_csv": "/v1/video/a1b2c3d4-e5f6-7890-abcd-ef1234567890/tracks",
    "summary_json": "/v1/video/a1b2c3d4-e5f6-7890-abcd-ef1234567890/summary"
  }
}`,
  },
  {
    id: "video-summary",
    method: "GET",
    path: "/v1/video/{job_id}/summary",
    summary: "Download analytics summary JSON",
    description:
      "Returns aggregated tracking metrics including unique object counts, per-class counts, directional counting line totals (IN / OUT), and processing throughput.",
    parameters: [
      {
        name: "job_id",
        in: "path",
        type: "string",
        required: true,
        description: "UUID of completed video processing job.",
      },
    ],
    responseFields: [
      { field: "job_id", type: "string", description: "Job identifier" },
      { field: "unique_counts", type: "integer", description: "Total unique object tracks identified" },
      { field: "counts_per_class", type: "object", description: "Map of class name to unique object count" },
      { field: "line_counts", type: "object", description: "Directional counting breakdown (total_in, total_out, by_class)" },
      { field: "fps_processed", type: "float", description: "Inference & tracking throughput in frames per second" },
    ],
    errorCodes: [
      { status: 404, name: "Not Found", description: "Job not found or summary not yet ready" },
    ],
    snippets: {
      curl: `curl "${API_DOCS_DISPLAY_BASE_URL}/v1/video/a1b2c3d4-e5f6-7890-abcd-ef1234567890/summary"`,
      js: `const res = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/video/" + jobId + "/summary");
const summary = await res.json();
console.log("Unique objects:", summary.unique_counts);
console.log("Line crossings IN:", summary.line_counts.total_in);`,
      python: `import requests

res = requests.get(f"${API_DOCS_DISPLAY_BASE_URL}/v1/video/{job_id}/summary")
summary = res.json()
print("Unique objects:", summary["unique_counts"])
print("Line IN:", summary["line_counts"]["total_in"])
print("Line OUT:", summary["line_counts"]["total_out"])`,
    },
    exampleResponse: `{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "unique_counts": 14,
  "counts_per_class": {
    "car": 10,
    "person": 3,
    "bus": 1
  },
  "line_counts": {
    "total_in": 7,
    "total_out": 4,
    "total_crossings": 11,
    "by_class": {
      "car": { "in": 6, "out": 3 },
      "person": { "in": 1, "out": 1 }
    },
    "line": {
      "p1": { "x": 0.1, "y": 0.5 },
      "p2": { "x": 0.9, "y": 0.5 }
    }
  },
  "fps_processed": 28.4,
  "total_frames_processed": 124,
  "duration_s": 12.4
}`,
  },
];

